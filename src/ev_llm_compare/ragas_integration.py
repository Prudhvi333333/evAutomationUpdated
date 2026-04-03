"""ragas_integration.py
Wrapper around the actual ragas library (>= 0.2.0).

This module integrates the *real* ragas library — not a custom reimplementation
with ragas-inspired names.  The wrapper:
  - builds a proper ragas EvaluationDataset from ModelResponse objects
  - runs only on RAG responses (context metrics are undefined for no-RAG)
  - attaches a configurable LLM judge (Gemini or Ollama via LangChain)
  - falls back gracefully when ragas is not installed or the LLM is unavailable
  - returns a flat dict of per-response metric scores

Metrics computed (where ragas >= 0.2 supports them):
  ragas_faithfulness          – fraction of answer claims supported by context
  ragas_context_precision     – precision@k of retrieved contexts
  ragas_context_recall        – fraction of reference claims covered by context
  ragas_answer_correctness    – semantic + factual match to golden reference

Only RAG responses (rag_enabled=True) receive context metrics.
All responses with a golden answer receive ragas_answer_correctness.
"""

from __future__ import annotations

import os
from typing import Any

from .schemas import ModelResponse

# ──────────────────────────────────────────────────────────────────────────────
# Availability checks — ragas and langchain are optional dependencies
# ──────────────────────────────────────────────────────────────────────────────

_RAGAS_AVAILABLE = False
_LANGCHAIN_GEMINI_AVAILABLE = False
_LANGCHAIN_OLLAMA_AVAILABLE = False

try:
    import ragas  # noqa: F401
    from ragas import evaluate as _ragas_evaluate  # type: ignore[attr-defined]
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample  # type: ignore[import]
    from ragas.metrics import (  # type: ignore[import]
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        AnswerCorrectness,
    )
    _RAGAS_AVAILABLE = True
except ImportError:
    pass

try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import]
    _LANGCHAIN_GEMINI_AVAILABLE = True
except ImportError:
    pass

try:
    from langchain_ollama import ChatOllama  # type: ignore[import]
    _LANGCHAIN_OLLAMA_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.chat_models import ChatOllama  # type: ignore[import]
        _LANGCHAIN_OLLAMA_AVAILABLE = True
    except ImportError:
        pass

try:
    from ragas.llms import LangchainLLMWrapper  # type: ignore[import]
    from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore[import]
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore[import]
    _RAGAS_LLM_WRAPPER_AVAILABLE = True
except ImportError:
    _RAGAS_LLM_WRAPPER_AVAILABLE = False


def ragas_available() -> bool:
    return _RAGAS_AVAILABLE


# ──────────────────────────────────────────────────────────────────────────────
# LLM / embedder factory for ragas
# ──────────────────────────────────────────────────────────────────────────────

def _build_ragas_llm(judge_provider: str, judge_model: str) -> Any:
    """Return a ragas-compatible LLM wrapper."""
    if not _RAGAS_LLM_WRAPPER_AVAILABLE:
        raise RuntimeError(
            "ragas LLM wrapper not available. Install: "
            "ragas langchain-google-genai or langchain-ollama"
        )
    if judge_provider == "gemini":
        if not _LANGCHAIN_GEMINI_AVAILABLE:
            raise RuntimeError(
                "langchain-google-genai is required for Gemini judge. "
                "Install: pip install langchain-google-genai"
            )
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set for ragas Gemini judge.")
        llm = ChatGoogleGenerativeAI(
            model=judge_model,
            google_api_key=api_key,
            temperature=0.0,
        )
        return LangchainLLMWrapper(llm)

    if judge_provider == "ollama":
        if not _LANGCHAIN_OLLAMA_AVAILABLE:
            raise RuntimeError(
                "langchain-ollama is required for Ollama judge. "
                "Install: pip install langchain-ollama"
            )
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = ChatOllama(model=judge_model, base_url=base_url, temperature=0.0)
        return LangchainLLMWrapper(llm)

    raise ValueError(
        f"Unsupported ragas judge provider: '{judge_provider}'. "
        "Supported: 'gemini', 'ollama'."
    )


def _build_ragas_embedder(embedding_model: str) -> Any:
    """Return a ragas-compatible embedding wrapper."""
    if not _RAGAS_LLM_WRAPPER_AVAILABLE:
        return None
    try:
        embedder = HuggingFaceEmbeddings(model_name=embedding_model)
        return LangchainEmbeddingsWrapper(embedder)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_evaluation_dataset(
    responses: list[ModelResponse],
    golden_answers: dict[str, str],
    context_result_limit: int = 6,
    context_char_budget: int = 4200,
) -> "EvaluationDataset":
    """Build a ragas EvaluationDataset from ModelResponse objects.

    Only includes responses that:
      - succeeded (success=True)
      - have a non-empty answer
    For RAG responses, retrieved context chunks are serialised as strings.
    For no-RAG responses, retrieved_contexts is [].
    """
    from .prompts import compact_context_segments

    samples: list[Any] = []
    for response in responses:
        if not response.success or not (response.answer or "").strip():
            continue

        # Build context strings
        if response.rag_enabled and response.retrieved_chunks:
            ctx_blocks = compact_context_segments(
                response.question,
                response.retrieved_chunks,
                max_results=context_result_limit,
                max_chars=context_char_budget,
            )
        else:
            ctx_blocks = []

        reference = golden_answers.get(response.question, "")

        sample = SingleTurnSample(
            user_input=response.question,
            retrieved_contexts=ctx_blocks,
            response=response.answer,
            reference=reference if reference else None,
        )
        samples.append(sample)

    return EvaluationDataset(samples=samples)


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation entry-point
# ──────────────────────────────────────────────────────────────────────────────

RAGAS_COLUMN_MAP = {
    "faithfulness": "ragas_faithfulness",
    "llm_context_precision_with_reference": "ragas_context_precision",
    "context_precision": "ragas_context_precision",
    "llm_context_recall": "ragas_context_recall",
    "context_recall": "ragas_context_recall",
    "answer_correctness": "ragas_answer_correctness",
}


def run_ragas_evaluation(
    responses: list[ModelResponse],
    golden_answers: dict[str, str],
    judge_provider: str = "gemini",
    judge_model: str = "gemini-2.5-flash",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    context_result_limit: int = 6,
    context_char_budget: int = 4200,
) -> dict[str, dict[str, float | None]]:
    """Run actual ragas metrics for RAG responses.

    Returns:
        dict keyed by (response.question + "||" + response.run_name) mapping
        to a dict of metric-name → score.

    Metric names in the output use the canonical 'ragas_*' prefix:
        ragas_faithfulness
        ragas_context_precision
        ragas_context_recall
        ragas_answer_correctness

    Scores are floats in [0, 1] or None if the metric failed.

    If ragas is not installed or the judge LLM is unavailable, the function
    logs a warning and returns an empty dict (caller should treat all scores
    as None).
    """
    if not _RAGAS_AVAILABLE:
        print(
            "[ragas_integration] WARNING: ragas library not installed. "
            "Skipping ragas metrics. Install with: pip install ragas"
        )
        return {}

    try:
        ragas_llm = _build_ragas_llm(judge_provider, judge_model)
    except Exception as exc:
        print(
            f"[ragas_integration] WARNING: Could not build ragas LLM judge: {exc}. "
            "Skipping ragas metrics."
        )
        return {}

    ragas_embedder = _build_ragas_embedder(embedding_model)

    # Separate RAG responses (need context metrics) from all responses
    rag_responses = [r for r in responses if r.rag_enabled and r.success]
    all_valid_responses = [r for r in responses if r.success and r.answer.strip()]

    results: dict[str, dict[str, float | None]] = {}

    # ── RAG-only metrics: Faithfulness, ContextPrecision, ContextRecall ───────
    if rag_responses:
        try:
            dataset = _build_evaluation_dataset(
                rag_responses,
                golden_answers,
                context_result_limit=context_result_limit,
                context_char_budget=context_char_budget,
            )
            rag_metrics = [
                Faithfulness(llm=ragas_llm),
                LLMContextPrecisionWithReference(llm=ragas_llm),
                LLMContextRecall(llm=ragas_llm),
            ]
            rag_result = _ragas_evaluate(dataset=dataset, metrics=rag_metrics)
            rag_df = rag_result.to_pandas()

            for idx, response in enumerate(rag_responses):
                if idx >= len(rag_df):
                    break
                row = rag_df.iloc[idx]
                key = _response_key(response)
                scores = results.setdefault(key, {})
                for col in rag_df.columns:
                    canonical = RAGAS_COLUMN_MAP.get(col.lower())
                    if canonical:
                        val = row.get(col)
                        scores[canonical] = float(val) if val is not None else None
        except Exception as exc:
            print(
                f"[ragas_integration] WARNING: RAG metrics evaluation failed: {exc}"
            )

    # ── All responses: AnswerCorrectness (needs golden reference) ─────────────
    scored_responses = [
        r for r in all_valid_responses if golden_answers.get(r.question, "")
    ]
    if scored_responses:
        try:
            dataset = _build_evaluation_dataset(
                scored_responses,
                golden_answers,
                context_result_limit=context_result_limit,
                context_char_budget=context_char_budget,
            )
            ac_metric = [AnswerCorrectness(llm=ragas_llm)]
            if ragas_embedder:
                ac_metric[0].embeddings = ragas_embedder  # type: ignore[attr-defined]
            ac_result = _ragas_evaluate(dataset=dataset, metrics=ac_metric)
            ac_df = ac_result.to_pandas()

            for idx, response in enumerate(scored_responses):
                if idx >= len(ac_df):
                    break
                row = ac_df.iloc[idx]
                key = _response_key(response)
                scores = results.setdefault(key, {})
                for col in ac_df.columns:
                    canonical = RAGAS_COLUMN_MAP.get(col.lower())
                    if canonical:
                        val = row.get(col)
                        scores[canonical] = float(val) if val is not None else None
        except Exception as exc:
            print(
                f"[ragas_integration] WARNING: AnswerCorrectness evaluation failed: {exc}"
            )

    return results


def _response_key(response: ModelResponse) -> str:
    return f"{response.question}||{response.run_name}"
