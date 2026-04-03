"""reference_metrics.py
Deterministic and lightweight semantic reference metrics.

All functions here are side-effect-free and do NOT require LLM calls.
They are designed to run on (prediction, reference) text pairs.

Metric catalogue
----------------
normalized_exact_match  – token-level exact match after light normalisation
token_f1                – unigram token overlap F1
rouge_l                 – ROUGE-L F-measure (longest common subsequence)
semantic_similarity     – cosine similarity of sentence embeddings

The SentenceEmbedder class caches the model so it is only loaded once.
"""

from __future__ import annotations

import re
import string
from typing import Any

_ROUGE_AVAILABLE = False
try:
    from rouge_score import rouge_scorer as _rouge_module  # type: ignore[import]
    _ROUGE_AVAILABLE = True
except ImportError:
    _rouge_module = None  # type: ignore[assignment]

_ST_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer  # type: ignore[import]
    import numpy as _np
    _ST_AVAILABLE = True
except ImportError:
    _SentenceTransformer = None  # type: ignore[assignment]
    _np = None  # type: ignore[assignment]


# ──────────────────────────────────────────────────────────────────────────────
# Text normalisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Light normalisation for exact-match / token-F1 evaluation."""
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove articles
    tokens = [t for t in text.split() if t not in {"a", "an", "the"}]
    return " ".join(tokens)


def _tokens(text: str) -> list[str]:
    return _normalise(text).split()


# ──────────────────────────────────────────────────────────────────────────────
# Normalised exact match
# ──────────────────────────────────────────────────────────────────────────────

def normalized_exact_match(prediction: str, reference: str) -> float:
    """Return 1.0 if normalised texts match exactly, else 0.0."""
    if not prediction.strip() or not reference.strip():
        return 0.0
    return 1.0 if _normalise(prediction) == _normalise(reference) else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Token F1
# ──────────────────────────────────────────────────────────────────────────────

def token_f1(prediction: str, reference: str) -> float:
    """Unigram token overlap F1 (SQuAD-style)."""
    if not prediction.strip() or not reference.strip():
        return 0.0
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counter: dict[str, int] = {}
    for t in pred_tokens:
        pred_counter[t] = pred_counter.get(t, 0) + 1

    ref_counter: dict[str, int] = {}
    for t in ref_tokens:
        ref_counter[t] = ref_counter.get(t, 0) + 1

    common = sum(
        min(pred_counter.get(t, 0), ref_counter.get(t, 0))
        for t in ref_counter
    )
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


# ──────────────────────────────────────────────────────────────────────────────
# ROUGE-L
# ──────────────────────────────────────────────────────────────────────────────

def rouge_l(prediction: str, reference: str) -> float | None:
    """ROUGE-L F-measure using the rouge_score library.

    Returns None if rouge_score is not installed.
    """
    if not _ROUGE_AVAILABLE or _rouge_module is None:
        return None
    if not prediction.strip() or not reference.strip():
        return 0.0
    scorer = _rouge_module.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return round(float(scores["rougeL"].fmeasure), 4)


# ──────────────────────────────────────────────────────────────────────────────
# Semantic similarity via sentence embeddings
# ──────────────────────────────────────────────────────────────────────────────

class SentenceEmbedder:
    """Thin wrapper around SentenceTransformer with lazy loading and caching."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if not _ST_AVAILABLE or _SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
        self._model = _SentenceTransformer(self._model_name)

    def encode(self, texts: list[str]) -> Any:
        self._load()
        return self._model.encode(texts, show_progress_bar=False)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity in [0, 1] between two texts."""
        if not text_a.strip() or not text_b.strip():
            return 0.0
        vecs = self.encode([text_a, text_b])
        # Cosine similarity
        a, b = vecs[0], vecs[1]
        denom = (_np.linalg.norm(a) * _np.linalg.norm(b))  # type: ignore[union-attr]
        if denom < 1e-9:
            return 0.0
        cos = float(_np.dot(a, b) / denom)  # type: ignore[union-attr]
        return round(max(0.0, min(1.0, cos)), 4)


# Module-level default embedder (lazy-loaded on first call)
_DEFAULT_EMBEDDER: SentenceEmbedder | None = None


def semantic_similarity(
    prediction: str,
    reference: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float | None:
    """Cosine similarity of sentence embeddings.

    Returns None if sentence-transformers is not installed.
    Uses a module-level cached embedder for efficiency across many calls.
    """
    if not _ST_AVAILABLE:
        return None
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None or _DEFAULT_EMBEDDER._model_name != model_name:
        _DEFAULT_EMBEDDER = SentenceEmbedder(model_name)
    return _DEFAULT_EMBEDDER.similarity(prediction, reference)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: compute all reference metrics in one call
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_reference_metrics(
    prediction: str,
    reference: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, float | None]:
    """Compute all deterministic reference metrics and return as a dict.

    Keys:
        normalized_exact_match  float  [0, 1]
        token_f1                float  [0, 1]
        rouge_l                 float  [0, 1]  or None if library missing
        semantic_similarity     float  [0, 1]  or None if library missing
    """
    return {
        "normalized_exact_match": normalized_exact_match(prediction, reference),
        "token_f1": token_f1(prediction, reference),
        "rouge_l": rouge_l(prediction, reference),
        "semantic_similarity": semantic_similarity(
            prediction, reference, model_name=embedding_model
        ),
    }
