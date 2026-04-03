from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any

from .prompts import NON_RAG_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT
from .settings import ModelSpec, RuntimeSettings


# ──────────────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class GenerationMetadata:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    finish_reason: str | None = None
    raw_response: Any | None = None


@dataclass(slots=True)
class GenerationResult:
    text: str
    metadata: GenerationMetadata


# ──────────────────────────────────────────────────────────────────────────────
# Base client
# ──────────────────────────────────────────────────────────────────────────────

class LLMClient:
    provider: str
    model_name: str

    def generate_with_metadata(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str | None = None,
        seed: int | None = None,
    ) -> str:
        return self.generate_with_metadata(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            seed=seed,
        ).text


# ──────────────────────────────────────────────────────────────────────────────
# Ollama client  (fixes dict-style response access → attribute access)
# ──────────────────────────────────────────────────────────────────────────────

class OllamaClient(LLMClient):
    def __init__(self, model_name: str, base_url: str):
        import ollama

        self.provider = "ollama"
        self.model_name = model_name
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)

    def generate_with_metadata(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        effective_system = system_prompt or RAG_SYSTEM_PROMPT
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        if seed is not None:
            options["seed"] = seed

        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            system=effective_system,
            options=options,
        )

        # ollama >= 0.6.0 returns a Pydantic GenerateResponse object.
        # Use attribute access; fall back to dict-style for older versions.
        try:
            text: str = response.response  # type: ignore[union-attr]
            prompt_tokens: int | None = response.prompt_eval_count  # type: ignore[union-attr]
            completion_tokens: int | None = response.eval_count  # type: ignore[union-attr]
            finish_reason: str | None = getattr(response, "done_reason", None)
        except AttributeError:
            # Older ollama versions returned a dict
            text = response["response"]  # type: ignore[index]
            prompt_tokens = response.get("prompt_eval_count")  # type: ignore[union-attr]
            completion_tokens = response.get("eval_count")  # type: ignore[union-attr]
            finish_reason = None

        text = (text or "").strip()
        total_tokens: int | None = None
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = int(prompt_tokens) + int(completion_tokens)

        return GenerationResult(
            text=text,
            metadata=GenerationMetadata(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                finish_reason=finish_reason,
                raw_response=response,
            ),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Gemini client  (fixes thinking_budget hard-coded to 0, adds safety settings)
# ──────────────────────────────────────────────────────────────────────────────

class GeminiClient(LLMClient):
    def __init__(self, model_name: str):
        from google import genai

        self.provider = "gemini"
        self.model_name = model_name
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set GEMINI_API_KEY or GOOGLE_API_KEY before using Gemini."
            )
        self.client = genai.Client(api_key=api_key)
        # Thinking budget: default 0 (fast inference). Set GEMINI_THINKING_BUDGET
        # env-var to a positive integer to enable extended thinking.
        self._thinking_budget: int = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))

    def generate_with_metadata(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        from google.genai import types

        effective_system = system_prompt or RAG_SYSTEM_PROMPT

        # Safety settings: pass-through for all harm categories so that
        # domain-specific supply-chain / employment questions are not blocked.
        safety_settings = [
            types.SafetySetting(
                category=category,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            )
            for category in [
                types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            ]
        ]

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=effective_system,
                temperature=temperature,
                max_output_tokens=max_tokens,
                seed=seed,
                safety_settings=safety_settings,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=self._thinking_budget
                ),
            ),
        )

        text = _extract_gemini_text(response)
        if text is None:
            # Check for blocked response
            candidates = getattr(response, "candidates", []) or []
            if candidates:
                finish = getattr(candidates[0], "finish_reason", None)
                if finish and str(finish).upper() in {"SAFETY", "RECITATION", "OTHER"}:
                    raise RuntimeError(
                        f"Gemini response blocked. finish_reason={finish}"
                    )
            raise RuntimeError("Gemini returned no text content.")

        usage = getattr(response, "usage_metadata", None)
        finish_reason: str | None = None
        candidates = getattr(response, "candidates", []) or []
        if candidates:
            finish_reason = str(getattr(candidates[0], "finish_reason", "") or "")

        return GenerationResult(
            text=text,
            metadata=GenerationMetadata(
                prompt_tokens=getattr(usage, "prompt_token_count", None),
                completion_tokens=getattr(usage, "candidates_token_count", None),
                total_tokens=getattr(usage, "total_token_count", None),
                finish_reason=finish_reason,
                raw_response=response,
            ),
        )


def _extract_gemini_text(response: Any) -> str | None:
    if getattr(response, "text", None):
        return str(response.text).strip()
    parts: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            t = getattr(part, "text", None)
            if t:
                parts.append(str(t))
    return "\n".join(parts).strip() if parts else None


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible client
# Supports: DashScope (Qwen3.6 Plus), OpenRouter, vLLM, local OpenAI-compat
# ──────────────────────────────────────────────────────────────────────────────

class OpenAICompatibleClient(LLMClient):
    """Generic client for any OpenAI-compatible REST API.

    Env-var for the API key is specified via `api_key_env` in ModelSpec.
    Example configurations:

    DashScope (Qwen3.6 Plus):
        base_url  = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key   = DASHSCOPE_API_KEY

    OpenRouter:
        base_url  = "https://openrouter.ai/api/v1"
        api_key   = OPENROUTER_API_KEY
    """

    def __init__(self, model_name: str, base_url: str, api_key_env: str):
        from openai import OpenAI

        self.provider = "openai_compatible"
        self.model_name = model_name
        self.base_url = base_url
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"API key not found. Set the env-var '{api_key_env}' before "
                f"using the openai_compatible provider for model '{model_name}'."
            )
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate_with_metadata(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        effective_system = system_prompt or RAG_SYSTEM_PROMPT
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": effective_system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        usage = response.usage
        finish_reason = str(choice.finish_reason or "")
        return GenerationResult(
            text=text,
            metadata=GenerationMetadata(
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
                finish_reason=finish_reason,
                raw_response=response,
            ),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def create_client(spec: ModelSpec, runtime: RuntimeSettings) -> LLMClient:
    if spec.provider == "ollama":
        return OllamaClient(spec.model_name, runtime.ollama_base_url)
    if spec.provider == "gemini":
        return GeminiClient(spec.model_name)
    if spec.provider == "openai_compatible":
        if not spec.base_url:
            raise ValueError(
                f"ModelSpec '{spec.run_name}' uses provider 'openai_compatible' "
                "but 'base_url' is not set."
            )
        if not spec.api_key_env:
            raise ValueError(
                f"ModelSpec '{spec.run_name}' uses provider 'openai_compatible' "
                "but 'api_key_env' is not set."
            )
        return OpenAICompatibleClient(spec.model_name, spec.base_url, spec.api_key_env)
    raise ValueError(
        f"Unsupported provider '{spec.provider}' for run '{spec.run_name}'. "
        "Valid providers: 'ollama', 'gemini', 'openai_compatible'."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Safe generation wrappers
# ──────────────────────────────────────────────────────────────────────────────

def safe_generate_with_metadata(
    client: LLMClient,
    prompt: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str | None = None,
    seed: int | None = None,
) -> tuple[str, float, bool, str | None, GenerationMetadata]:
    start = time.perf_counter()
    try:
        result = client.generate_with_metadata(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            seed=seed,
        )
        elapsed = round(time.perf_counter() - start, 3)
        return result.text, elapsed, True, None, result.metadata
    except Exception as exc:
        elapsed = round(time.perf_counter() - start, 3)
        return (
            f"ERROR: {exc}",
            elapsed,
            False,
            str(exc),
            GenerationMetadata(),
        )


def safe_generate(
    client: LLMClient,
    prompt: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str | None = None,
    seed: int | None = None,
) -> tuple[str, float, bool, str | None]:
    text, elapsed, success, error, _ = safe_generate_with_metadata(
        client,
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        seed=seed,
    )
    return text, elapsed, success, error
