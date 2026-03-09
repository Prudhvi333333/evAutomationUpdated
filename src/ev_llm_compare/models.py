from __future__ import annotations

import os
import time

from .prompts import SYSTEM_PROMPT
from .settings import ModelSpec, RuntimeSettings


class LLMClient:
    provider: str
    model_name: str

    def generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str | None = None,
    ) -> str:
        raise NotImplementedError


class OllamaClient(LLMClient):
    def __init__(self, model_name: str, base_url: str):
        import ollama

        self.provider = "ollama"
        self.model_name = model_name
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)

    def generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str | None = None,
    ) -> str:
        effective_system_prompt = system_prompt or SYSTEM_PROMPT
        response = self.client.generate(
            model=self.model_name,
            prompt=f"{effective_system_prompt}\n\n{prompt}",
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
        return response["response"].strip()


class GeminiClient(LLMClient):
    def __init__(self, model_name: str):
        from google import genai

        self.provider = "gemini"
        self.model_name = model_name
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY before using Gemini.")
        self.client = genai.Client(api_key=api_key)

    def generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str | None = None,
    ) -> str:
        from google.genai import types

        effective_system_prompt = system_prompt or SYSTEM_PROMPT
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=effective_system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        if getattr(response, "text", None):
            return response.text.strip()

        parts: list[str] = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                text = getattr(part, "text", None)
                if text:
                    parts.append(text)
        if parts:
            return "\n".join(parts).strip()

        raise RuntimeError("Gemini returned no text content.")


def create_client(spec: ModelSpec, runtime: RuntimeSettings) -> LLMClient:
    if spec.provider == "ollama":
        return OllamaClient(spec.model_name, runtime.ollama_base_url)
    if spec.provider == "gemini":
        return GeminiClient(spec.model_name)
    raise ValueError(f"Unsupported provider: {spec.provider}")


def safe_generate(
    client: LLMClient,
    prompt: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str | None = None,
) -> tuple[str, float, bool, str | None]:
    start = time.perf_counter()
    try:
        answer = client.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
        return answer, round(time.perf_counter() - start, 2), True, None
    except Exception as exc:
        return f"ERROR: {exc}", round(time.perf_counter() - start, 2), False, str(exc)
