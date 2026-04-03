from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args: object, **kwargs: object) -> bool:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval settings
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class RetrievalSettings:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dense_top_k: int = 30
    final_top_k: int = 20
    batch_size: int = 64
    lexical_weight: float = 0.45
    dense_weight: float = 0.55
    rrf_k: int = 60
    note_chunk_size: int = 1200
    note_chunk_overlap: int = 150
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 20
    reranker_weight: float = 0.35
    max_chunks_per_company: int = 4
    structured_summary_limit: int = 15
    structured_exhaustive_limit: int = 210
    compact_context_enabled: bool = True
    generation_context_result_limit: int = 15
    generation_context_char_budget: int = 16000
    evaluation_context_result_limit: int = 8
    evaluation_context_char_budget: int = 6000


# ──────────────────────────────────────────────────────────────────────────────
# Runtime settings
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class RuntimeSettings:
    ollama_base_url: str = "http://localhost:11434"
    qdrant_path: Path = Path("artifacts/qdrant")
    output_dir: Path = Path("artifacts/results")


# ──────────────────────────────────────────────────────────────────────────────
# Model specification
#
# provider values:
#   "ollama"             – local Ollama instance
#   "gemini"             – Google Gemini API (google-genai)
#   "openai_compatible"  – any OpenAI-compatible REST API
#                          (OpenRouter, DashScope/Qwen, vLLM, etc.)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class ModelSpec:
    run_name: str           # unique identifier, e.g. "qwen14b_rag"
    provider: str           # "ollama" | "gemini" | "openai_compatible"
    model_name: str         # model id passed to the provider
    rag_enabled: bool
    enabled: bool = True
    temperature: float = 0.1
    max_tokens: int = 2048
    seed: int | None = None
    # OpenAI-compatible extras (ignored for ollama / gemini)
    base_url: str | None = None     # e.g. "https://openrouter.ai/api/v1"
    api_key_env: str | None = None  # name of the env-var holding the key


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation / judge settings
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class EvaluationSettings:
    judge_provider: str = "ollama"
    judge_model: str = "qwen2.5:14b"
    max_retries: int = 2
    parallelism: int = 1
    # Semantic-similarity embedding model for reference metrics
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Whether to run actual ragas metrics on RAG runs
    ragas_enabled: bool = True
    # Judge base_url / api_key_env for openai_compatible judge (optional)
    judge_base_url: str | None = None
    judge_api_key_env: str | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Top-level config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class AppConfig:
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    models: list[ModelSpec] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_value(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value
    return default


# ──────────────────────────────────────────────────────────────────────────────
# Model registry — generation models + 2 modes each
#
# All three Qwen models run via OpenRouter (OpenAI-compatible API).
# Gemma 27B and Gemini 2.5 Flash are kept for optional use.
#
# ENV-VAR reference (generation models):
#   OPENROUTER_API_KEY      shared API key for all OpenRouter models
#   QWEN14B_MODEL           OpenRouter model id  (default: qwen/qwen2.5-14b-instruct)
#   QWEN35B_MODEL           OpenRouter model id  (default: qwen/qwen3-30b-a3b)
#   QWEN36PLUS_MODEL        OpenRouter model id  (default: qwen/qwen-plus-0828)
#   OPENROUTER_BASE_URL     base url             (default: https://openrouter.ai/api/v1)
#   GEMMA27B_MODEL          ollama model tag     (default: gemma3:27b)
#   GEMINI_MODEL            gemini model id      (default: gemini-2.5-flash)
#
#   ENABLE_<UPPER_RUN_NAME>  enable/disable individual run slot (default: true)
#   e.g. ENABLE_QWEN14B_RAG=false
#   Gemma and Gemini are disabled by default; set ENABLE_GEMMA27B_RAG=true to activate.
# ──────────────────────────────────────────────────────────────────────────────

def _build_models() -> list[ModelSpec]:  # noqa: C901
    global_seed_str = os.getenv("EXPERIMENT_SEED")
    global_seed: int | None = int(global_seed_str) if global_seed_str else None

    # All Qwen generation models run locally via Ollama
    qwen14b_model  = os.getenv("QWEN14B_MODEL",  "qwen2.5:14b")
    qwen32b_model  = os.getenv("QWEN32B_MODEL",  "qwen2.5:32b")
    qwen35b_model  = os.getenv("QWEN35B_MODEL",  "qwen3:30b-a3b")
    # qwen36plus: API-only cloud model — no Ollama version exists.
    # Disabled by default; set ENABLE_QWEN36PLUS_RAG=true and configure
    # QWEN36PLUS_BASE_URL / QWEN36PLUS_API_KEY_ENV if you want to use it via an API.
    qwen36plus_model       = os.getenv("QWEN36PLUS_MODEL",       "qwen-plus-latest")
    qwen36plus_base_url    = os.getenv("QWEN36PLUS_BASE_URL",    "https://dashscope.aliyuncs.com/compatible-mode/v1")
    qwen36plus_api_key_env = os.getenv("QWEN36PLUS_API_KEY_ENV", "DASHSCOPE_API_KEY")

    gemma27b_model = os.getenv("GEMMA27B_MODEL", "gemma3:27b")
    gemini_model   = os.getenv("GEMINI_MODEL",   "gemini-2.5-flash")

    specs: list[ModelSpec] = [
        # ── Qwen 2.5 14B (Ollama) ─────────────────────────────────────────────
        # Pull: ollama pull qwen2.5:14b
        ModelSpec(
            run_name="qwen14b_rag",
            provider="ollama",
            model_name=qwen14b_model,
            rag_enabled=True,
            enabled=_env_flag("ENABLE_QWEN14B_RAG", True),
            seed=global_seed,
        ),
        ModelSpec(
            run_name="qwen14b_no_rag",
            provider="ollama",
            model_name=qwen14b_model,
            rag_enabled=False,
            enabled=_env_flag("ENABLE_QWEN14B_NO_RAG", True),
            seed=global_seed,
        ),
        # ── Qwen 2.5 32B (Ollama) ────────────────────────────────────────────
        # Pull: ollama pull qwen2.5:32b
        ModelSpec(
            run_name="qwen32b_rag",
            provider="ollama",
            model_name=qwen32b_model,
            rag_enabled=True,
            enabled=_env_flag("ENABLE_QWEN32B_RAG", True),
            seed=global_seed,
        ),
        ModelSpec(
            run_name="qwen32b_no_rag",
            provider="ollama",
            model_name=qwen32b_model,
            rag_enabled=False,
            enabled=_env_flag("ENABLE_QWEN32B_NO_RAG", True),
            seed=global_seed,
        ),
        # ── Qwen3 30B-A3B MoE (Ollama) ───────────────────────────────────────
        # Pull: ollama pull qwen3:30b-a3b
        # This is the closest local equivalent to "Qwen3.5-35B-A3B".
        ModelSpec(
            run_name="qwen35b_rag",
            provider="ollama",
            model_name=qwen35b_model,
            rag_enabled=True,
            enabled=_env_flag("ENABLE_QWEN35B_RAG", True),
            seed=global_seed,
        ),
        ModelSpec(
            run_name="qwen35b_no_rag",
            provider="ollama",
            model_name=qwen35b_model,
            rag_enabled=False,
            enabled=_env_flag("ENABLE_QWEN35B_NO_RAG", True),
            seed=global_seed,
        ),
        # ── Qwen Plus (API-only — disabled by default) ────────────────────────
        # "Qwen3.6-Plus" is a cloud model with no open-weight Ollama release.
        # To enable: set ENABLE_QWEN36PLUS_RAG=true and configure the API vars.
        ModelSpec(
            run_name="qwen36plus_rag",
            provider="openai_compatible",
            model_name=qwen36plus_model,
            rag_enabled=True,
            enabled=_env_flag("ENABLE_QWEN36PLUS_RAG", False),
            base_url=qwen36plus_base_url,
            api_key_env=qwen36plus_api_key_env,
            seed=global_seed,
        ),
        ModelSpec(
            run_name="qwen36plus_no_rag",
            provider="openai_compatible",
            model_name=qwen36plus_model,
            rag_enabled=False,
            enabled=_env_flag("ENABLE_QWEN36PLUS_NO_RAG", False),
            base_url=qwen36plus_base_url,
            api_key_env=qwen36plus_api_key_env,
            seed=global_seed,
        ),
        # ── Gemma 27B (Ollama — disabled by default) ──────────────────────────
        # Pull: ollama pull gemma3:27b
        ModelSpec(
            run_name="gemma27b_rag",
            provider="ollama",
            model_name=gemma27b_model,
            rag_enabled=True,
            enabled=_env_flag("ENABLE_GEMMA27B_RAG", False),
            seed=global_seed,
        ),
        ModelSpec(
            run_name="gemma27b_no_rag",
            provider="ollama",
            model_name=gemma27b_model,
            rag_enabled=False,
            enabled=_env_flag("ENABLE_GEMMA27B_NO_RAG", False),
            seed=global_seed,
        ),
        # ── Gemini 2.5 Flash (API — disabled by default) ──────────────────────
        ModelSpec(
            run_name="gemini_flash_rag",
            provider="gemini",
            model_name=gemini_model,
            rag_enabled=True,
            enabled=_env_flag("ENABLE_GEMINI_FLASH_RAG", False),
            seed=global_seed,
        ),
        ModelSpec(
            run_name="gemini_flash_no_rag",
            provider="gemini",
            model_name=gemini_model,
            rag_enabled=False,
            enabled=_env_flag("ENABLE_GEMINI_FLASH_NO_RAG", False),
            seed=global_seed,
        ),
    ]
    return specs


# ──────────────────────────────────────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────────────────────────────────────

def load_config(*, dotenv_enabled: bool = True) -> AppConfig:
    if dotenv_enabled:
        load_dotenv(override=False)

    config = AppConfig()
    config.models = [spec for spec in _build_models() if spec.enabled]

    default_temperature = float(os.getenv("MODEL_TEMPERATURE", "0.1"))
    default_max_tokens = int(os.getenv("MODEL_MAX_TOKENS", "2048"))
    for spec in config.models:
        spec.temperature = default_temperature
        spec.max_tokens = default_max_tokens

    config.retrieval.embedding_model = os.getenv(
        "EMBEDDING_MODEL", config.retrieval.embedding_model
    )
    config.retrieval.reranker_enabled = _env_flag(
        "RERANKER_ENABLED", config.retrieval.reranker_enabled
    )
    config.retrieval.reranker_model = os.getenv(
        "RERANKER_MODEL", config.retrieval.reranker_model
    )
    config.retrieval.reranker_top_k = int(
        os.getenv("RERANKER_TOP_K", str(config.retrieval.reranker_top_k))
    )
    config.retrieval.reranker_weight = float(
        os.getenv("RERANKER_WEIGHT", str(config.retrieval.reranker_weight))
    )
    config.retrieval.max_chunks_per_company = int(
        os.getenv("MAX_CHUNKS_PER_COMPANY", str(config.retrieval.max_chunks_per_company))
    )
    config.retrieval.structured_summary_limit = int(
        os.getenv("STRUCTURED_SUMMARY_LIMIT", str(config.retrieval.structured_summary_limit))
    )
    config.retrieval.structured_exhaustive_limit = int(
        os.getenv("STRUCTURED_EXHAUSTIVE_LIMIT", str(config.retrieval.structured_exhaustive_limit))
    )
    config.retrieval.compact_context_enabled = _env_flag(
        "COMPACT_CONTEXT_ENABLED", config.retrieval.compact_context_enabled
    )
    config.retrieval.generation_context_result_limit = int(
        os.getenv(
            "GENERATION_CONTEXT_RESULT_LIMIT",
            str(config.retrieval.generation_context_result_limit),
        )
    )
    config.retrieval.generation_context_char_budget = int(
        os.getenv(
            "GENERATION_CONTEXT_CHAR_BUDGET",
            str(config.retrieval.generation_context_char_budget),
        )
    )
    config.retrieval.evaluation_context_result_limit = int(
        _env_value(
            "EVALUATION_CONTEXT_RESULT_LIMIT",
            "RAGAS_CONTEXT_RESULT_LIMIT",
            default=str(config.retrieval.evaluation_context_result_limit),
        )
    )
    config.retrieval.evaluation_context_char_budget = int(
        _env_value(
            "EVALUATION_CONTEXT_CHAR_BUDGET",
            "RAGAS_CONTEXT_CHAR_BUDGET",
            default=str(config.retrieval.evaluation_context_char_budget),
        )
    )

    config.runtime.ollama_base_url = os.getenv(
        "OLLAMA_BASE_URL", config.runtime.ollama_base_url
    )
    config.runtime.qdrant_path = Path(
        os.getenv("QDRANT_PATH", str(config.runtime.qdrant_path))
    )
    config.runtime.output_dir = Path(
        os.getenv("OUTPUT_DIR", str(config.runtime.output_dir))
    )

    config.evaluation.judge_provider = _env_value(
        "EVALUATION_JUDGE_PROVIDER",
        "RAGAS_JUDGE_PROVIDER",
        default=config.evaluation.judge_provider,
    )
    config.evaluation.judge_model = _env_value(
        "EVALUATION_JUDGE_MODEL",
        "RAGAS_JUDGE_MODEL",
        default=config.evaluation.judge_model,
    )
    config.evaluation.max_retries = int(
        _env_value(
            "EVALUATION_MAX_RETRIES",
            "RAGAS_MAX_RETRIES",
            default=str(config.evaluation.max_retries),
        )
    )
    parallelism_str = os.getenv("EVALUATION_PARALLELISM") or os.getenv("RAGAS_PARALLELISM")
    if parallelism_str is not None:
        config.evaluation.parallelism = max(1, int(parallelism_str))
    else:
        config.evaluation.parallelism = (
            1 if config.evaluation.judge_provider == "ollama" else 4
        )
    config.evaluation.ragas_enabled = _env_flag(
        "RAGAS_ENABLED", config.evaluation.ragas_enabled
    )
    # Judge base_url/api_key — only needed when judge_provider is openai_compatible
    judge_base_url = os.getenv("EVALUATION_JUDGE_BASE_URL")
    if judge_base_url:
        config.evaluation.judge_base_url = judge_base_url
    judge_api_key_env = os.getenv("EVALUATION_JUDGE_API_KEY_ENV")
    if judge_api_key_env:
        config.evaluation.judge_api_key_env = judge_api_key_env

    return config
