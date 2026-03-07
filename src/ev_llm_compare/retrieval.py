from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import tempfile
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import CrossEncoder, SentenceTransformer

from .chunking import tokenize
from .schemas import Chunk, RetrievalResult
from .settings import RetrievalSettings

NORMALIZE_PATTERN = re.compile(r"[^a-z0-9]+")
QUERY_STOPWORDS = {
    "a",
    "all",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "group",
    "how",
    "in",
    "is",
    "list",
    "me",
    "of",
    "show",
    "the",
    "them",
    "what",
    "which",
    "with",
}


def normalize_text(value: str) -> str:
    return NORMALIZE_PATTERN.sub(" ", value.lower()).strip()


def build_collection_fingerprint(chunks: list[Chunk], embedding_model: str) -> str:
    digest = hashlib.sha1()
    digest.update(embedding_model.encode("utf-8"))
    digest.update(f"|{len(chunks)}|".encode("utf-8"))
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(chunk.metadata.get("row_key", "")).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:12]


@dataclass(slots=True)
class QueryPlan:
    question: str
    normalized_question: str
    intent: str
    dense_queries: list[str]
    matched_categories: list[str]
    matched_companies: list[str]
    matched_role_terms: list[str]
    group_by_role: bool
    prefer_structured: bool


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], settings: RetrievalSettings, qdrant_path: Path):
        self.chunks = chunks
        self.settings = settings
        self.qdrant_path = qdrant_path
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._reranker: CrossEncoder | None = None
        self._reranker_failed = False
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        self.idf = self._build_idf(chunks)
        self.row_records = self._build_row_records(chunks)
        self.known_categories = self._known_field_values("category")
        self.known_companies = self._known_field_values("company")
        self.role_terms = self._build_role_terms()
        self.client = self._create_client(qdrant_path)
        self.collection_name = self._collection_name(chunks, settings.embedding_model)
        self._index_chunks()

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def retrieve(self, question: str) -> list[RetrievalResult]:
        query_plan = self._plan_query(question)
        structured_results = self._structured_matches(query_plan)
        dense_rank, dense_scores = self._rank_dense(query_plan.dense_queries)
        lexical_rank, lexical_scores = self._rank_lexically(query_plan.dense_queries)

        candidate_ids = set(dense_rank) | set(lexical_rank)
        retrieved: list[RetrievalResult] = []
        for chunk_id in candidate_ids:
            chunk = self.chunk_map[str(chunk_id)]
            final_score = self._fusion_score(str(chunk_id), dense_rank, lexical_rank)
            final_score += self._metadata_boost(question, chunk.metadata)
            retrieved.append(
                RetrievalResult(
                    chunk_id=str(chunk_id),
                    text=chunk.text,
                    metadata=chunk.metadata,
                    dense_score=round(dense_scores.get(str(chunk_id), 0.0), 5),
                    lexical_score=round(lexical_scores.get(str(chunk_id), 0.0), 5),
                    final_score=round(final_score, 5),
                )
            )
        retrieved.sort(key=lambda item: item.final_score, reverse=True)
        reranked = self._rerank_candidates(question, retrieved)
        return self._select_context_results(query_plan, structured_results, reranked)

    def _index_chunks(self) -> None:
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        if self._collection_is_current():
            return

        embeddings = self.embedding_model.encode(
            [chunk.text for chunk in self.chunks],
            batch_size=self.settings.batch_size,
            show_progress_bar=False,
        )
        vector_size = len(embeddings[0])
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

        points: list[PointStruct] = []
        for chunk, vector in zip(self.chunks, embeddings, strict=True):
            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector=vector.tolist(),
                    payload={
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                    },
                )
            )

        for start in range(0, len(points), self.settings.batch_size):
            batch = points[start : start + self.settings.batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)

    def _collection_is_current(self) -> bool:
        try:
            self.client.get_collection(self.collection_name)
            count = self.client.count(collection_name=self.collection_name, exact=True).count
            return count == len(self.chunks)
        except Exception:
            return False

    def _build_idf(self, chunks: list[Chunk]) -> dict[str, float]:
        document_frequency: defaultdict[str, int] = defaultdict(int)
        for chunk in chunks:
            for token in chunk.token_set:
                document_frequency[token] += 1
        total_documents = len(chunks)
        return {
            token: math.log((1 + total_documents) / (1 + count)) + 1
            for token, count in document_frequency.items()
        }

    def _build_row_records(self, chunks: list[Chunk]) -> dict[str, dict[str, str]]:
        row_records: dict[str, dict[str, str]] = {}
        for chunk in chunks:
            row_key = str(chunk.metadata.get("row_key", "")).strip()
            if not row_key or row_key in row_records:
                continue
            row_records[row_key] = {
                "row_key": row_key,
                "company": str(chunk.metadata.get("company", "")),
                "category": str(chunk.metadata.get("category", "")),
                "ev_supply_chain_role": str(chunk.metadata.get("ev_supply_chain_role", "")),
                "product_service": str(chunk.metadata.get("product_service", "")),
                "primary_oems": str(chunk.metadata.get("primary_oems", "")),
                "location": str(chunk.metadata.get("location", "")),
                "employment": str(chunk.metadata.get("employment", "")),
                "ev_battery_relevant": str(chunk.metadata.get("ev_battery_relevant", "")),
                "source_file": str(chunk.metadata.get("source_file", "")),
                "sheet_name": str(chunk.metadata.get("sheet_name", "")),
                "row_number": str(chunk.metadata.get("row_number", "")),
                "row_summary": str(chunk.metadata.get("row_summary", "")),
            }
        return row_records

    def _known_field_values(self, field_name: str) -> list[str]:
        values = {
            str(record.get(field_name, "")).strip()
            for record in self.row_records.values()
            if str(record.get(field_name, "")).strip()
        }
        return sorted(values, key=len, reverse=True)

    def _build_role_terms(self) -> list[str]:
        terms: set[str] = set()
        for record in self.row_records.values():
            role = normalize_text(record.get("ev_supply_chain_role", ""))
            if not role:
                continue
            terms.add(role)
            for part in re.split(r"\band\b|/|,", role):
                candidate = part.strip()
                if len(candidate.split()) >= 2:
                    terms.add(candidate)
        return sorted(terms, key=len, reverse=True)

    def _plan_query(self, question: str) -> QueryPlan:
        normalized_question = normalize_text(question)
        matched_categories = [
            category
            for category in self.known_categories
            if normalize_text(category) and normalize_text(category) in normalized_question
        ]
        matched_companies = [
            company
            for company in self.known_companies
            if normalize_text(company) and normalize_text(company) in normalized_question
        ]
        matched_role_terms = [
            term
            for term in self.role_terms
            if term and term in normalized_question
        ]
        if any(term in normalized_question for term in {"define", "definition", "meaning", "methodology"}):
            intent = "definition"
        elif any(term in normalized_question for term in {"compare", "difference", "versus", "vs"}):
            intent = "comparison"
        elif any(term in normalized_question for term in {"count", "how many", "list", "show all", "group"}):
            intent = "aggregation"
        else:
            intent = "fact"

        dense_queries = [question]
        focused_terms = [
            token
            for token in normalized_question.split()
            if token not in QUERY_STOPWORDS
        ]
        if focused_terms:
            dense_queries.append(" ".join(focused_terms[:12]))
        filters = matched_companies + matched_categories + matched_role_terms
        if filters:
            dense_queries.append(" | ".join(dict.fromkeys(filters)))

        return QueryPlan(
            question=question,
            normalized_question=normalized_question,
            intent=intent,
            dense_queries=list(dict.fromkeys(query for query in dense_queries if query.strip())),
            matched_categories=matched_categories,
            matched_companies=matched_companies,
            matched_role_terms=matched_role_terms,
            group_by_role="group" in normalized_question and "ev supply chain role" in normalized_question,
            prefer_structured=bool(filters) and intent in {"aggregation", "comparison", "fact"},
        )

    def _rank_dense(self, dense_queries: list[str]) -> tuple[dict[str, int], dict[str, float]]:
        rank_scores: defaultdict[str, float] = defaultdict(float)
        raw_scores: dict[str, float] = {}
        for query in dense_queries:
            query_vector = self.embedding_model.encode(query).tolist()
            dense_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=self.settings.dense_top_k,
            ).points
            for rank, point in enumerate(dense_results, start=1):
                point_id = str(point.id)
                rank_scores[point_id] += 1.0 / (self.settings.rrf_k + rank)
                raw_scores[point_id] = max(raw_scores.get(point_id, 0.0), float(point.score))
        dense_rank = {
            chunk_id: rank
            for rank, (chunk_id, _) in enumerate(
                sorted(rank_scores.items(), key=lambda item: item[1], reverse=True)[: self.settings.dense_top_k],
                start=1,
            )
        }
        return dense_rank, raw_scores

    def _rank_lexically(self, dense_queries: list[str]) -> tuple[dict[str, int], dict[str, float]]:
        scores: list[tuple[str, float]] = []
        score_lookup: dict[str, float] = {}
        for chunk in self.chunks:
            score = max(self._lexical_score(query, chunk) for query in dense_queries)
            if score > 0:
                scores.append((chunk.chunk_id, score))
                score_lookup[chunk.chunk_id] = score
        scores.sort(key=lambda item: item[1], reverse=True)
        lexical_rank = {
            chunk_id: rank
            for rank, (chunk_id, _) in enumerate(scores[: self.settings.dense_top_k], start=1)
        }
        return lexical_rank, score_lookup

    def _lexical_score(self, question: str, chunk: Chunk) -> float:
        query_tokens = tokenize(question)
        if not query_tokens:
            return 0.0
        overlap = query_tokens & chunk.token_set
        base = sum(self.idf.get(token, 1.0) for token in overlap)
        phrase_bonus = 0.0
        question_lower = question.lower()
        company = str(chunk.metadata.get("company", "")).lower()
        if company and company in question_lower:
            phrase_bonus += 2.5
        if "sheet_name" in chunk.metadata and str(chunk.metadata["sheet_name"]).lower() in question_lower:
            phrase_bonus += 0.5
        return base + phrase_bonus

    def _fusion_score(
        self,
        chunk_id: str,
        dense_rank: dict[str, int],
        lexical_rank: dict[str, int],
    ) -> float:
        dense_component = 0.0
        lexical_component = 0.0
        dense_position = dense_rank.get(chunk_id)
        lexical_position = lexical_rank.get(chunk_id)
        if dense_position:
            dense_component = self.settings.dense_weight / (self.settings.rrf_k + dense_position)
        if lexical_position:
            lexical_component = self.settings.lexical_weight / (self.settings.rrf_k + lexical_position)
        return dense_component + lexical_component

    def _metadata_boost(self, question: str, metadata: dict[str, Any]) -> float:
        boost = 0.0
        question_lower = normalize_text(question)
        company = normalize_text(str(metadata.get("company", "")))
        category = normalize_text(str(metadata.get("category", "")))
        role = normalize_text(str(metadata.get("ev_supply_chain_role", "")))
        product_service = normalize_text(str(metadata.get("product_service", "")))
        chunk_type = str(metadata.get("chunk_type", ""))
        if company and company in question_lower:
            boost += 0.04
        if category and category in question_lower:
            boost += 0.04
        if role:
            for role_term in self.role_terms:
                if role_term in question_lower and role_term in role:
                    boost += 0.05
                    break
        if product_service:
            overlap = tokenize(question_lower) & tokenize(product_service)
            if len(overlap) >= 2:
                boost += 0.015
        if "employment" in question_lower and chunk_type == "location_theme":
            boost += 0.01
        if any(term in question_lower for term in {"oem", "hyundai", "kia", "rivian", "mercedes"}):
            if chunk_type == "supply_chain_theme":
                boost += 0.015
        if chunk_type == "note_reference" and any(term in question_lower for term in {"define", "definition", "methodology"}):
            boost += 0.03
        return boost

    def _structured_matches(self, query_plan: QueryPlan) -> list[RetrievalResult]:
        if not query_plan.prefer_structured:
            return []

        matched_rows = [
            record
            for record in self.row_records.values()
            if self._row_matches_filters(record, query_plan)
        ]
        if not matched_rows:
            return []

        results: list[RetrievalResult] = [
            RetrievalResult(
                chunk_id=f"structured-summary::{hashlib.sha1(query_plan.normalized_question.encode('utf-8')).hexdigest()[:12]}",
                text=self._build_structured_summary(query_plan, matched_rows),
                metadata={
                    "chunk_type": "structured_match_summary",
                    "company": "",
                    "source_file": matched_rows[0]["source_file"],
                    "sheet_name": matched_rows[0]["sheet_name"],
                },
                dense_score=1.0,
                lexical_score=1.0,
                final_score=1.0,
            )
        ]

        row_limit = min(self.settings.structured_summary_limit, max(2, self.settings.final_top_k - 2))
        for row in matched_rows[:row_limit]:
            results.append(
                RetrievalResult(
                    chunk_id=f"structured-row::{row['row_key']}",
                    text=row["row_summary"],
                    metadata={
                        "chunk_type": "structured_row_match",
                        "company": row["company"],
                        "source_file": row["source_file"],
                        "sheet_name": row["sheet_name"],
                        "row_number": row["row_number"],
                        "row_key": row["row_key"],
                    },
                    dense_score=0.99,
                    lexical_score=0.99,
                    final_score=0.99,
                )
            )
        return results

    def _row_matches_filters(self, row: dict[str, str], query_plan: QueryPlan) -> bool:
        row_category = normalize_text(row.get("category", ""))
        row_company = normalize_text(row.get("company", ""))
        row_role = normalize_text(row.get("ev_supply_chain_role", ""))
        row_product = normalize_text(row.get("product_service", ""))

        if query_plan.matched_categories:
            category_filters = {normalize_text(value) for value in query_plan.matched_categories}
            if row_category not in category_filters:
                return False
        if query_plan.matched_companies:
            company_filters = {normalize_text(value) for value in query_plan.matched_companies}
            if row_company not in company_filters:
                return False
        if query_plan.matched_role_terms and not any(
            term in row_role or term in row_product
            for term in query_plan.matched_role_terms
        ):
            return False
        return True

    def _build_structured_summary(self, query_plan: QueryPlan, matched_rows: list[dict[str, str]]) -> str:
        lines = ["Structured workbook matches from exact metadata filters:"]
        applied_filters: list[str] = []
        if query_plan.matched_categories:
            applied_filters.append(f"category in {query_plan.matched_categories}")
        if query_plan.matched_companies:
            applied_filters.append(f"company in {query_plan.matched_companies}")
        if query_plan.matched_role_terms:
            applied_filters.append(f"role terms in {query_plan.matched_role_terms}")
        lines.append(f"Applied filters: {', '.join(applied_filters)}")
        lines.append(f"Matched rows: {len(matched_rows)}")

        if len(matched_rows) > 12 or query_plan.group_by_role:
            grouped: defaultdict[str, list[str]] = defaultdict(list)
            for row in matched_rows:
                grouped[row.get("ev_supply_chain_role") or "Unspecified"].append(row.get("company") or "Unknown")
            lines.append("Grouped by EV Supply Chain Role:")
            for role in sorted(grouped):
                companies = sorted(dict.fromkeys(grouped[role]))
                preview = companies[: self.settings.structured_summary_limit]
                lines.append(f"- {role}: {', '.join(preview)}")
                if len(companies) > len(preview):
                    lines.append(f"  + {len(companies) - len(preview)} more companies")
            return "\n".join(lines)

        lines.append("Detailed rows:")
        for row in matched_rows[: self.settings.structured_summary_limit]:
            lines.append(
                "- "
                + " | ".join(
                    value
                    for value in [
                        f"Company: {row.get('company', '')}",
                        f"Category: {row.get('category', '')}",
                        f"EV Supply Chain Role: {row.get('ev_supply_chain_role', '')}",
                        f"Product / Service: {row.get('product_service', '')}",
                        f"Location: {row.get('location', '')}",
                        f"Employment: {row.get('employment', '')}",
                    ]
                    if value.split(': ', 1)[1]
                )
            )
        if len(matched_rows) > self.settings.structured_summary_limit:
            lines.append(
                f"Additional matched rows omitted: {len(matched_rows) - self.settings.structured_summary_limit}"
            )
        return "\n".join(lines)

    def _rerank_candidates(
        self,
        question: str,
        candidates: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        if not self.settings.reranker_enabled or len(candidates) < 2:
            return candidates

        reranker = self._load_reranker()
        if reranker is None:
            return candidates

        rerank_candidates = candidates[: self.settings.reranker_top_k]
        try:
            scores = reranker.predict(
                [(question, candidate.text) for candidate in rerank_candidates],
                show_progress_bar=False,
            )
        except Exception:
            self._reranker_failed = True
            return candidates

        rerank_order = sorted(
            range(len(rerank_candidates)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )
        rerank_bonus = {
            rerank_candidates[index].chunk_id: self.settings.reranker_weight / (self.settings.rrf_k + rank)
            for rank, index in enumerate(rerank_order, start=1)
        }
        reranked: list[RetrievalResult] = []
        for candidate in candidates:
            reranked.append(
                RetrievalResult(
                    chunk_id=candidate.chunk_id,
                    text=candidate.text,
                    metadata=candidate.metadata,
                    dense_score=candidate.dense_score,
                    lexical_score=candidate.lexical_score,
                    final_score=round(candidate.final_score + rerank_bonus.get(candidate.chunk_id, 0.0), 5),
                )
            )
        reranked.sort(key=lambda item: item.final_score, reverse=True)
        return reranked

    def _load_reranker(self) -> CrossEncoder | None:
        if self._reranker_failed:
            return None
        if self._reranker is not None:
            return self._reranker
        try:
            self._reranker = CrossEncoder(self.settings.reranker_model)
        except Exception:
            self._reranker_failed = True
            return None
        return self._reranker

    def _select_context_results(
        self,
        query_plan: QueryPlan,
        structured_results: list[RetrievalResult],
        candidates: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        ordered_candidates = sorted(
            candidates,
            key=lambda item: (self._context_priority(item, query_plan), item.final_score),
            reverse=True,
        )

        selected: list[RetrievalResult] = []
        seen_keys: set[str] = set()
        company_counts: defaultdict[str, int] = defaultdict(int)

        for result in structured_results[:1]:
            selected.append(result)
            seen_keys.add(result.chunk_id)

        for result in structured_results[1:] + ordered_candidates:
            unique_key = str(result.metadata.get("row_key", "")).strip() or result.chunk_id
            if unique_key in seen_keys:
                continue
            company = str(result.metadata.get("company", "")).strip()
            if company and company_counts[company] >= self.settings.max_chunks_per_company:
                continue
            selected.append(result)
            seen_keys.add(unique_key)
            if company:
                company_counts[company] += 1
            if len(selected) >= self.settings.final_top_k:
                break
        return selected

    def _context_priority(self, result: RetrievalResult, query_plan: QueryPlan) -> float:
        chunk_type = str(result.metadata.get("chunk_type", ""))
        if chunk_type == "structured_match_summary":
            return 3.0
        if query_plan.intent == "definition" and chunk_type == "note_reference":
            return 2.5
        if chunk_type in {"structured_row_match", "company_profile", "row_full"}:
            return 1.5
        if chunk_type == "note_reference":
            return 1.0
        return 0.0

    def _collection_name(self, chunks: list[Chunk], embedding_model: str) -> str:
        return f"ev_compare_{build_collection_fingerprint(chunks, embedding_model)}"

    def _create_client(self, qdrant_path: Path) -> QdrantClient:
        try:
            return QdrantClient(path=str(qdrant_path))
        except RuntimeError as exc:
            if "already accessed by another instance" not in str(exc):
                raise
            self._temp_dir = tempfile.TemporaryDirectory(prefix="ev_qdrant_")
            return QdrantClient(path=self._temp_dir.name)
