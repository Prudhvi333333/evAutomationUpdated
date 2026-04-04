#!/usr/bin/env python3
"""
Validate strict offline/private Hugging Face setup for this project.

This script never attempts network access by design:
- requires offline env flags
- loads from a local model path only
- passes local_files_only=True to HF loaders
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _print_status(name: str, value: str | None) -> None:
    print(f"{name}={value if value is not None else ''}")


def _require_env_exact(name: str, expected: str, errors: list[str]) -> None:
    value = os.getenv(name)
    _print_status(name, value)
    if value != expected:
        errors.append(f"{name} must be '{expected}', got '{value}'")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check local/private HF setup and local model path readiness."
    )
    parser.add_argument(
        "--model-path",
        default="models/hf/gemma3-27b",
        help="Local path to HF-format model directory.",
    )
    parser.add_argument(
        "--check-tokenizer",
        action="store_true",
        help="Also try local tokenizer load (still offline-only).",
    )
    args = parser.parse_args()

    errors: list[str] = []

    print("== Offline/Privacy Environment Check ==")
    _require_env_exact("HF_HUB_OFFLINE", "1", errors)
    _require_env_exact("TRANSFORMERS_OFFLINE", "1", errors)
    _require_env_exact("HF_HUB_DISABLE_TELEMETRY", "1", errors)
    _require_env_exact("DO_NOT_TRACK", "1", errors)
    _require_env_exact("WANDB_DISABLED", "true", errors)

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        errors.append("HF token env var is set; unset it for strict privacy.")
    print("HF_TOKEN_SET=yes" if hf_token else "HF_TOKEN_SET=no")

    print("\n== Local Model Path Check ==")
    model_path = Path(args.model_path).expanduser().resolve()
    print(f"MODEL_PATH={model_path}")
    has_config = False
    has_weights = False
    if not model_path.exists():
        errors.append(f"Model path does not exist: {model_path}")
    elif not model_path.is_dir():
        errors.append(f"Model path is not a directory: {model_path}")
    else:
        # Minimal expected files for HF model directory.
        has_config = (model_path / "config.json").exists()
        has_weights = any(
            path.name.startswith("pytorch_model")
            or path.suffix == ".safetensors"
            for path in model_path.iterdir()
            if path.is_file()
        )
        print(f"HAS_CONFIG_JSON={'yes' if has_config else 'no'}")
        print(f"HAS_WEIGHT_FILES={'yes' if has_weights else 'no'}")
        if not has_config:
            errors.append("Missing config.json in model directory.")
        if not has_weights:
            errors.append("No model weight files found (.safetensors or pytorch_model*).")

    print("\n== HF Local Load Check ==")
    try:
        from transformers import AutoConfig  # type: ignore[import]
    except Exception as exc:
        errors.append(f"transformers is not installed/importable: {exc}")
        AutoConfig = None  # type: ignore[assignment]

    if AutoConfig is not None and model_path.exists() and model_path.is_dir() and has_config:
        try:
            cfg = AutoConfig.from_pretrained(
                str(model_path),
                local_files_only=True,
                trust_remote_code=False,
            )
            print(f"CONFIG_MODEL_TYPE={getattr(cfg, 'model_type', 'unknown')}")
        except Exception as exc:
            errors.append(f"Local config load failed: {exc}")

        if args.check_tokenizer:
            try:
                from transformers import AutoTokenizer  # type: ignore[import]

                _ = AutoTokenizer.from_pretrained(
                    str(model_path),
                    local_files_only=True,
                    trust_remote_code=False,
                )
                print("TOKENIZER_LOAD=ok")
            except Exception as exc:
                errors.append(f"Local tokenizer load failed: {exc}")

    if errors:
        print("\nFAILED:")
        for item in errors:
            print(f"- {item}")
        return 1

    print("\nOK: offline/private HF setup is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
