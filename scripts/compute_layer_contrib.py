#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class EvalRow:
    prompt_id: str
    question: str
    reference_answer: str


def _load_calibration_rows(path: Path, max_samples: int | None) -> list[EvalRow]:
    df = pd.read_csv(path)
    missing = [col for col in ("question", "reference_answer") if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in calibration file: {missing}")

    rows: list[EvalRow] = []
    for idx, row in df.iterrows():
        question = str(row.get("question", "")).strip()
        answer = str(row.get("reference_answer", "")).strip()
        if not question or not answer:
            continue
        prompt_id = str(row.get("pack_id", f"row_{idx+1:04d}")).strip()
        rows.append(EvalRow(prompt_id=prompt_id, question=question, reference_answer=answer))

    if max_samples is not None:
        rows = rows[:max_samples]

    if not rows:
        raise ValueError("No usable calibration rows found with non-empty question+reference_answer.")
    return rows


def _resolve_layers(model: Any) -> tuple[str, Any]:
    candidates = (
        "language_model.layers",       # Gemma3ForConditionalGeneration
        "language_model.model.layers",
        "model.layers",
        "model.model.layers",
        "model.decoder.layers",
        "transformer.h",
        "gpt_neox.layers",
    )
    for name in candidates:
        current = model
        ok = True
        for part in name.split("."):
            if not hasattr(current, part):
                ok = False
                break
            current = getattr(current, part)
        if not ok:
            continue
        try:
            _ = len(current)
        except Exception:
            continue
        return name, current

    # Fallback: scan for a likely transformer block ModuleList.
    try:
        import torch.nn as nn

        for name, module in model.named_modules():
            if not isinstance(module, nn.ModuleList):
                continue
            if len(module) == 0:
                continue
            first = module[0]
            class_name = first.__class__.__name__.lower()
            if any(token in class_name for token in ("layer", "block", "decoder")):
                return name, module
    except Exception:
        pass

    raise RuntimeError(
        f"Unable to locate transformer layer stack on this model ({model.__class__.__name__})."
    )


def _validate_local_model_dir(model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model path is not a directory: {model_path}")

    config_ok = (model_path / "config.json").exists()
    weight_ok = any(
        p.is_file() and (p.suffix == ".safetensors" or p.name.startswith("pytorch_model"))
        for p in model_path.iterdir()
    )
    tokenizer_ok = any(
        (model_path / name).exists()
        for name in (
            "tokenizer.model",
            "tokenizer.json",
            "spiece.model",
            "tokenizer_config.json",
        )
    )

    missing: list[str] = []
    if not config_ok:
        missing.append("config.json")
    if not weight_ok:
        missing.append("model weights (.safetensors or pytorch_model*)")
    if not tokenizer_ok:
        missing.append("tokenizer files (tokenizer.model/tokenizer.json/...)")

    if missing:
        missing_display = ", ".join(missing)
        raise FileNotFoundError(
            "Local HF model directory is incomplete. Missing: "
            f"{missing_display}. "
            "If this is a gated model, complete the authenticated download first."
        )


def _build_scoring_tensors(
    tokenizer: Any,
    question: str,
    answer: str,
    max_length: int,
) -> tuple[Any, Any, Any] | None:
    prompt = f"Question:\n{question}\n\nAnswer:\n"
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None:
        answer_ids = answer_ids + [int(tokenizer.eos_token_id)]

    input_ids = prompt_ids + answer_ids
    labels = ([-100] * len(prompt_ids)) + answer_ids
    attention_mask = [1] * len(input_ids)

    overflow = len(input_ids) - max_length
    if overflow > 0:
        # Prefer trimming from the prompt side to preserve answer supervision.
        trim_from_prompt = min(overflow, len(prompt_ids))
        input_ids = input_ids[trim_from_prompt:]
        labels = labels[trim_from_prompt:]
        attention_mask = attention_mask[trim_from_prompt:]

        # If still overflowing, trim from the left of the remaining sequence.
        overflow2 = len(input_ids) - max_length
        if overflow2 > 0:
            input_ids = input_ids[overflow2:]
            labels = labels[overflow2:]
            attention_mask = attention_mask[overflow2:]

    if not input_ids or all(value == -100 for value in labels):
        return None

    import torch

    input_ids_t = torch.tensor([input_ids], dtype=torch.long)
    attn_t = torch.tensor([attention_mask], dtype=torch.long)
    labels_t = torch.tensor([labels], dtype=torch.long)
    return input_ids_t, attn_t, labels_t


def _model_input_device(model: Any) -> Any:
    for param in model.parameters():
        return param.device
    raise RuntimeError("Model has no parameters.")


def _mean_loss(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[EvalRow],
    max_length: int,
) -> tuple[float, int, float]:
    import torch

    model.eval()
    device = _model_input_device(model)

    loss_sum = 0.0
    count = 0
    start = time.perf_counter()
    for row in rows:
        batch = _build_scoring_tensors(tokenizer, row.question, row.reference_answer, max_length)
        if batch is None:
            continue
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss_sum += float(output.loss.detach().cpu())
        count += 1

    elapsed = time.perf_counter() - start
    if count == 0:
        raise RuntimeError("No rows produced valid scoring batches.")
    return (loss_sum / count), count, elapsed


def _identity_layer_hook(module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
    if not inputs:
        return output
    hidden_in = inputs[0]
    if isinstance(output, tuple):
        if not output:
            return output
        out = list(output)
        out[0] = hidden_in
        return tuple(out)
    return hidden_in


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute per-layer contribution via identity ablation.")
    parser.add_argument(
        "--model-path",
        default="models/hf/gemma3-27b",
        help="Local HF model path (must exist offline).",
    )
    parser.add_argument(
        "--calibration-csv",
        default="data/private/eval_pack_v1/calibration.csv",
        help="Calibration split CSV from fixed evaluation pack.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/private/eval_pack_v1/layer_contrib_v1",
        help="Output directory for contribution artifacts.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=64,
        help="Maximum calibration rows to evaluate (for runtime control).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Max sequence length for scoring batches.",
    )
    parser.add_argument(
        "--start-layer",
        type=int,
        default=0,
        help="Layer index to start ablation sweep.",
    )
    parser.add_argument(
        "--end-layer",
        type=int,
        default=None,
        help="Exclusive end layer index for sweep. Default: all layers.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Normalized contribution threshold for prune candidates.",
    )
    parser.add_argument(
        "--metric-only",
        action="store_true",
        help="Only compute/write layer contribution metric; do not apply pruning threshold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and planned sweep without loading model.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    model_path = Path(args.model_path).resolve()
    calibration_path = Path(args.calibration_csv).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_calibration_rows(calibration_path, args.max_samples)
    print(f"Loaded calibration rows: {len(rows)}")
    print(f"Model path: {model_path}")
    print(f"Calibration file: {calibration_path}")
    print(f"Output dir: {out_dir}")

    if args.dry_run:
        print("Dry run complete.")
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _validate_local_model_dir(model_path)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=False,
    )

    load_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map="auto",
            **load_kwargs,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
        if torch.cuda.is_available():
            model = model.to("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            model = model.to("mps")
        else:
            model = model.to("cpu")

    layer_path, layers = _resolve_layers(model)
    total_layers = len(layers)
    start_layer = max(0, int(args.start_layer))
    end_layer = int(args.end_layer) if args.end_layer is not None else total_layers
    end_layer = min(end_layer, total_layers)
    if start_layer >= end_layer:
        raise ValueError(
            f"Invalid layer sweep range: start={start_layer}, end={end_layer}, total={total_layers}"
        )
    print(f"Layer stack: {layer_path} ({total_layers} layers)")
    print(f"Ablation sweep: [{start_layer}, {end_layer})")

    baseline_loss, scored_count, baseline_time = _mean_loss(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        max_length=args.max_length,
    )
    if baseline_loss <= 0:
        raise RuntimeError(f"Baseline loss is non-positive ({baseline_loss}); cannot normalize.")

    print(
        f"Baseline mean loss: {baseline_loss:.6f} "
        f"(rows scored: {scored_count}, elapsed: {baseline_time:.1f}s)"
    )

    records: list[dict[str, Any]] = []
    for layer_idx in range(start_layer, end_layer):
        layer = layers[layer_idx]
        handle = layer.register_forward_hook(_identity_layer_hook)
        try:
            ablated_loss, _, ablated_time = _mean_loss(
                model=model,
                tokenizer=tokenizer,
                rows=rows,
                max_length=args.max_length,
            )
        finally:
            handle.remove()

        signed_delta = (ablated_loss - baseline_loss) / baseline_loss
        positive_delta = max(0.0, signed_delta)
        records.append(
            {
                "layer_idx": layer_idx,
                "baseline_mean_loss": baseline_loss,
                "ablated_mean_loss": ablated_loss,
                "signed_delta_over_baseline": signed_delta,
                "raw_contribution": positive_delta,
                "elapsed_seconds": ablated_time,
            }
        )
        print(
            f"Layer {layer_idx:03d}: ablated_loss={ablated_loss:.6f}, "
            f"signed_delta={signed_delta:+.6f}"
        )

    contrib_sum = sum(record["raw_contribution"] for record in records)
    for record in records:
        record["normalized_contribution"] = (
            (record["raw_contribution"] / contrib_sum) if contrib_sum > 0 else 0.0
        )
        if not args.metric_only:
            record["prune_candidate"] = bool(record["normalized_contribution"] < args.threshold)

    df = pd.DataFrame(records).sort_values("layer_idx")
    csv_path = out_dir / "layer_contributions.csv"
    json_path = out_dir / "layer_contributions.json"
    summary_path = out_dir / "summary.json"

    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")

    prune_candidates: list[int] = []
    if not args.metric_only:
        prune_candidates = df.loc[df["prune_candidate"], "layer_idx"].astype(int).tolist()
    top_layers = (
        df.sort_values("normalized_contribution", ascending=False)
        .head(10)[["layer_idx", "normalized_contribution"]]
        .to_dict(orient="records")
    )
    summary = {
        "model_path": str(model_path),
        "calibration_csv": str(calibration_path),
        "rows_scored": int(scored_count),
        "baseline_mean_loss": float(baseline_loss),
        "layer_stack_path": layer_path,
        "total_layers": int(total_layers),
        "sweep_start": int(start_layer),
        "sweep_end": int(end_layer),
        "threshold": (None if args.metric_only else float(args.threshold)),
        "metric_only": bool(args.metric_only),
        "contribution_sum": float(contrib_sum),
        "prune_candidate_count": int(len(prune_candidates)),
        "prune_candidate_layers": [int(x) for x in prune_candidates],
        "top_layers": top_layers,
        "artifacts": {
            "csv": str(csv_path),
            "json": str(json_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {summary_path}")
    if args.metric_only:
        print("Metric generation complete. No pruning applied in this run.")
    else:
        print(f"Prune candidates (< {args.threshold:.4f} normalized): {len(prune_candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
