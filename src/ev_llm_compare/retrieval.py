from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from pathlib import Path
import re
import tempfile
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from .chunking import tokenize
from .schemas import Chunk, RetrievalResult
from .settings import RetrievalSettings

NORMALIZE_PATTERN = re.compile(r"[^a-z0-9]+")


def normalize_text(value: str) -> str:
    return NORMALIZE_PATTERN.sub(" ", value.lower()).strip()


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], settings: RetrievalSettings, qdrant_path: Path):
        self.chunks = chunks
        self.settings = settings
        self.qdrant_path = qdrant_path
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
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
        structured_results = self._structured_matches(question)
        if structured_results:
            return structured_results[: self.settings.final_top_k]

        query_vector = self.embedding_model.encode(question).tolist()
        dense_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=self.settings.dense_top_k,
        ).points

        dense_rank = {
            point.id: rank
            for rank, point in enumerate(dense_results, start=1)
        }
        lexical_rank = self._rank_lexically(question)

        candidate_ids = set(dense_rank) | set(lexical_rank)
        retrieved: list[RetrievalResult] = []
        for chunk_id in candidate_ids:
            chunk = self.chunk_map[str(chunk_id)]
            dense_score = self._dense_score(str(chunk_id), dense_results)
            lexical_score = self._lexical_score(question, chunk)
            final_score = self._fusion_score(str(chunk_id), dense_rank, lexical_rank)
            final_score += self._metadata_boost(question, chunk.metadata)
            retrieved.append(
                RetrievalResult(
                    chunk_id=str(chunk_id),
                    text=chunk.text,
                    metadata=chunk.metadata,
                    dense_score=round(dense_score, 5),
                    lexical_score=round(lexical_score, 5),
                    final_score=round(final_score, 5),
                )
            )

        retrieved.sort(key=lambda item: item.final_score, reverse=True)

        deduped = self._dedupe_rows(retrieved)
        return deduped[: self.settings.final_top_k]

    def _index_chunks(self) -> None:
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
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

    def _rank_lexically(self, question: str) -> dict[str, int]:
        scores: list[tuple[str, float]] = []
        for chunk in self.chunks:
            score = self._lexical_score(question, chunk)
            if score > 0:
                scores.append((chunk.chunk_id, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return {
            chunk_id: rank
            for rank, (chunk_id, _) in enumerate(scores[: self.settings.dense_top_k], start=1)
        }

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

    def _dense_score(self, chunk_id: str, dense_results: list[Any]) -> float:
        for point in dense_results:
            if str(point.id) == chunk_id:
                return float(point.score)
        return 0.0

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
        return boost

    def _structured_matches(self, question: str) -> list[RetrievalResult]:
        question_norm = normalize_text(question)
        matched_categories = [
            category
            for category in self.known_categories
            if normalize_text(category) and normalize_text(category) in question_norm
        ]
        matched_companies = [
            company
            for company in self.known_companies
            if normalize_text(company) and normalize_text(company) in question_norm
        ]
        matched_role_terms = [
            term
            for term in self.role_terms
            if term and term in question_norm
        ]

        if not matched_categories and not matched_companies and not matched_role_terms:
            return []

        matched_rows = [
            record
            for record in self.row_records.values()
            if self._row_matches_filters(record, matched_categories, matched_companies, matched_role_terms)
        ]
        if not matched_rows:
            return []

        results: list[RetrievalResult] = [
            RetrievalResult(
                chunk_id=f"structured-summary::{hashlib.sha1(question_norm.encode('utf-8')).hexdigest()[:12]}",
                text=self._build_structured_summary(
                    question=question,
                    matched_rows=matched_rows,
                    matched_categories=matched_categories,
                    matched_companies=matched_companies,
                    matched_role_terms=matched_role_terms,
                ),
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

        if len(matched_rows) <= max(6, self.settings.final_top_k - 2):
            for row in matched_rows:
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
                        },
                        dense_score=0.99,
                        lexical_score=0.99,
                        final_score=0.99,
                    )
                )
        return results

    def _row_matches_filters(
        self,
        row: dict[str, str],
        matched_categories: list[str],
        matched_companies: list[str],
        matched_role_terms: list[str],
    ) -> bool:
        row_category = normalize_text(row.get("category", ""))
        row_company = normalize_text(row.get("company", ""))
        row_role = normalize_text(row.get("ev_supply_chain_role", ""))
        row_product = normalize_text(row.get("product_service", ""))

        if matched_categories and row_category not in {normalize_text(value) for value in matched_categories}:
            return False
        if matched_companies and row_company not in {normalize_text(value) for value in matched_companies}:
            return False
        if matched_role_terms and not any(
            term in row_role or term in row_product
            for term in matched_role_terms
        ):
            return False
        return True

    def _build_structured_summary(
        self,
        question: str,
        matched_rows: list[dict[str, str]],
        matched_categories: list[str],
        matched_companies: list[str],
        matched_role_terms: list[str],
    ) -> str:
        lines = ["Structured workbook matches from exact row filters:"]
        applied_filters: list[str] = []
        if matched_categories:
            applied_filters.append(f"category in {matched_categories}")
        if matched_companies:
            applied_filters.append(f"company in {matched_companies}")
        if matched_role_terms:
            applied_filters.append(f"role terms in {matched_role_terms}")
        lines.append(f"Applied filters: {', '.join(applied_filters)}")
        lines.append(f"Matched rows: {len(matched_rows)}")

        group_by_role = "group" in question.lower() and "ev supply chain role" in question.lower()
        if len(matched_rows) > 12 or group_by_role:
            grouped: defaultdict[str, list[str]] = defaultdict(list)
            for row in matched_rows:
                grouped[row.get("ev_supply_chain_role") or "Unspecified"].append(row.get("company") or "Unknown")
            lines.append("Grouped by EV Supply Chain Role:")
            for role in sorted(grouped):
                lines.append(f"- {role}:")
                for company in sorted(grouped[role]):
                    lines.append(f"  - {company}")
            return "\n".join(lines)

        lines.append("Detailed rows:")
        for row in matched_rows:
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
                    if value.split(": ", 1)[1]
                )
            )
        return "\n".join(lines)

    def _dedupe_rows(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        deduped: list[RetrievalResult] = []
        seen_rows: set[str] = set()
        for result in results:
            row_key = str(result.metadata.get("row_key", "")).strip()
            if row_key:
                if row_key in seen_rows:
                    continue
                seen_rows.add(row_key)
                row_summary = str(result.metadata.get("row_summary", "")).strip()
                if row_summary:
                    result = RetrievalResult(
                        chunk_id=result.chunk_id,
                        text=row_summary,
                        metadata=result.metadata,
                        dense_score=result.dense_score,
                        lexical_score=result.lexical_score,
                        final_score=result.final_score,
                    )
            deduped.append(result)
        return deduped


    def _collection_name(self, chunks: list[Chunk], embedding_model: str) -> str:
        digest = hashlib.sha1(
            f"{embedding_model}|{len(chunks)}|{chunks[0].chunk_id if chunks else 'empty'}".encode("utf-8")
        ).hexdigest()[:12]
        return f"ev_compare_{digest}"

    def _create_client(self, qdrant_path: Path) -> QdrantClient:
        try:
            return QdrantClient(path=str(qdrant_path))
        except RuntimeError as exc:
            if "already accessed by another instance" not in str(exc):
                raise
            self._temp_dir = tempfile.TemporaryDirectory(prefix="ev_qdrant_")
            return QdrantClient(path=self._temp_dir.name)
