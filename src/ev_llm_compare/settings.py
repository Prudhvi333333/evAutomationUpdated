from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass(slots=True)
class RetrievalSettings:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dense_top_k: int = 18
    final_top_k: int = 8
    batch_size: int = 64
    lexical_weight: float = 0.45
    dense_weight: float = 0.55
    rrf_k: int = 60
    note_chunk_size: int = 1200
    note_chunk_overlap: int = 150


@dataclass(slots=True)
class RuntimeSettings:
    ollama_base_url: str = "http://localhost:11434"
    qdrant_path: Path = Path("artifacts/qdrant")
    output_dir: Path = Path("artifacts/results")
    dotenv_enabled: bool = True


@dataclass(slots=True)
class ModelSpec:
    run_name: str
    provider: str
    model_name: str
    rag_enabled: bool
    enabled: bool = True
    temperature: float = 0.1
    max_tokens: int = 700


@dataclass(slots=True)
class AppConfig:
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    models: list[ModelSpec] = field(default_factory=list)
    ragas_judge_provider: str = "gemini"
    ragas_judge_model: str = "gemini-2.5-flash"
    ragas_embedding_provider: str = "google"
    ragas_embedding_model: str = "models/embedding-001"


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            run_name="qwen_rag",
            provider="ollama",
            model_name=os.getenv("QWEN_MODEL", "qwen2.5:14b"),
            rag_enabled=True,
            enabled=_env_flag("ENABLE_QWEN_RAG", True),
        ),
        ModelSpec(
            run_name="qwen_no_rag",
            provider="ollama",
            model_name=os.getenv("QWEN_MODEL", "qwen2.5:14b"),
            rag_enabled=False,
            enabled=_env_flag("ENABLE_QWEN_NO_RAG", True),
        ),
        ModelSpec(
            run_name="tinyllama_rag",
            provider="ollama",
            model_name=os.getenv("TINYLLAMA_MODEL", "tinyllama"),
            rag_enabled=True,
            enabled=_env_flag("ENABLE_TINYLLAMA_RAG", True),
        ),
        ModelSpec(
            run_name="tinyllama_no_rag",
            provider="ollama",
            model_name=os.getenv("TINYLLAMA_MODEL", "tinyllama"),
            rag_enabled=False,
            enabled=_env_flag("ENABLE_TINYLLAMA_NO_RAG", True),
        ),
        ModelSpec(
            run_name="gemini_rag",
            provider="gemini",
            model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            rag_enabled=True,
            enabled=_env_flag("ENABLE_GEMINI_RAG", True),
        ),
        ModelSpec(
            run_name="gemini_no_rag",
            provider="gemini",
            model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            rag_enabled=False,
            enabled=_env_flag("ENABLE_GEMINI_NO_RAG", True),
        ),
    ]


def load_config() -> AppConfig:
    config = AppConfig()
    config.models = [model for model in _build_models() if model.enabled]
    config.retrieval.embedding_model = os.getenv(
        "EMBEDDING_MODEL",
        config.retrieval.embedding_model,
    )
    config.runtime.ollama_base_url = os.getenv(
        "OLLAMA_BASE_URL",
        config.runtime.ollama_base_url,
    )
    config.runtime.qdrant_path = Path(
        os.getenv("QDRANT_PATH", str(config.runtime.qdrant_path))
    )
    config.runtime.output_dir = Path(
        os.getenv("OUTPUT_DIR", str(config.runtime.output_dir))
    )
    config.ragas_judge_provider = os.getenv(
        "RAGAS_JUDGE_PROVIDER",
        config.ragas_judge_provider,
    )
    config.ragas_judge_model = os.getenv(
        "RAGAS_JUDGE_MODEL",
        config.ragas_judge_model,
    )
    config.ragas_embedding_provider = os.getenv(
        "RAGAS_EMBEDDING_PROVIDER",
        config.ragas_embedding_provider,
    )
    config.ragas_embedding_model = os.getenv(
        "RAGAS_EMBEDDING_MODEL",
        config.ragas_embedding_model,
    )
    return config
