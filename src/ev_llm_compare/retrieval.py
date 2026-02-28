from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from pathlib import Path
import tempfile
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from .chunking import tokenize
from .schemas import Chunk, RetrievalResult
from .settings import RetrievalSettings


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], settings: RetrievalSettings, qdrant_path: Path):
        self.chunks = chunks
        self.settings = settings
        self.qdrant_path = qdrant_path
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        self.idf = self._build_idf(chunks)
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
        return retrieved[: self.settings.final_top_k]

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
        question_lower = question.lower()
        company = str(metadata.get("company", "")).lower()
        chunk_type = str(metadata.get("chunk_type", ""))
        if company and company in question_lower:
            boost += 0.04
        if "employment" in question_lower and chunk_type == "location_theme":
            boost += 0.01
        if any(term in question_lower for term in {"oem", "hyundai", "kia", "rivian", "mercedes"}):
            if chunk_type == "supply_chain_theme":
                boost += 0.015
        return boost

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
