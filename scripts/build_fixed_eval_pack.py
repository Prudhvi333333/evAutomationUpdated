#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


QUESTION_COL_CANDIDATES = (
    "question",
    "questions",
    "query",
    "prompt",
    "sample query",
)

ANSWER_COL_CANDIDATES = (
    "golden_answer",
    "golden answer",
    "reference_answer",
    "reference answer",
    "reference",
    "answer",
    "human validated answers",
    "human validated answer",
    "answer from web",
)


@dataclass
class SourceSpec:
    path: Path
    sheet: str | int
    source_name: str


def _norm_col(text: object) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _norm_question(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _clean_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _pick_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized_to_original = {_norm_col(col): col for col in columns}
    for c in candidates:
        if c in normalized_to_original:
            return normalized_to_original[c]
    return None


def _heuristic_task_type(question: str) -> str:
    q = _norm_question(question)
    if q.startswith(("show all", "list all", "map all", "identify all")):
        return "exhaustive_listing"
    if q.startswith("how many") or " number of " in q or " count " in q:
        return "counting"
    if any(token in q for token in (" compare ", " versus ", " vs ")):
        return "comparative"
    if any(token in q for token in ("which ", "identify ", "find ", "what locations", "top ")):
        return "retrieval_filter"
    return "general_qa"


def _heuristic_difficulty(question: str) -> str:
    q = _norm_question(question)
    score = 0
    if len(q.split()) >= 16:
        score += 1
    if len(q.split()) >= 24:
        score += 1
    if any(token in q for token in (" and ", " with ", " within ", " across ", " grouped by ", " broken down by ")):
        score += 1
    if any(token in q for token in ("full set", "network", "linked to each", "connected to each", "single-point-of-failure", "fragility")):
        score += 1
    if score >= 3:
        return "high"
    if score == 2:
        return "medium"
    return "low"


def _load_source(spec: SourceSpec) -> list[dict[str, Any]]:
    if not spec.path.exists():
        return []

    df = pd.read_excel(spec.path, sheet_name=spec.sheet)
    if df.empty:
        return []

    columns = [str(c) for c in df.columns]
    q_col = _pick_column(columns, QUESTION_COL_CANDIDATES)
    if q_col is None:
        return []

    a_col = _pick_column(columns, ANSWER_COL_CANDIDATES)

    rows: list[dict[str, Any]] = []
    for row_index, (_, row) in enumerate(df.iterrows(), start=1):
        q_raw = row.get(q_col)
        if pd.isna(q_raw):
            continue
        question = _clean_text(q_raw)
        if len(question) < 8:
            continue

        reference_answer = ""
        if a_col is not None:
            a_raw = row.get(a_col)
            if a_raw is not None and not pd.isna(a_raw):
                reference_answer = str(a_raw).strip()

        rows.append(
            {
                "question": question,
                "question_norm": _norm_question(question),
                "reference_answer": reference_answer,
                "source_name": spec.source_name,
                "source_file": str(spec.path),
                "source_sheet": str(spec.sheet),
                "source_row": row_index,
            }
        )
    return rows


def _collect_rows() -> list[dict[str, Any]]:
    sources = [
        SourceSpec(Path("Sample questions.xlsx"), "Table 1", "sample_questions"),
        SourceSpec(Path("data/Human validated 50 questions.xlsx"), "Sheet1", "human_validated_50"),
        SourceSpec(Path("Grouped_use_cases_for_Sample 100 questions.xlsx"), "Grouped Questions", "grouped_questions"),
        SourceSpec(Path("artifacts/Golden_answers.xlsx"), "Sheet1", "golden_answers"),
    ]
    out: list[dict[str, Any]] = []
    for spec in sources:
        out.extend(_load_source(spec))
    return out


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row["question_norm"]
        existing = merged.get(key)
        if existing is None:
            merged[key] = {
                "question": row["question"],
                "question_norm": row["question_norm"],
                "reference_answer": row["reference_answer"] or "",
                "source_names": [row["source_name"]],
                "source_files": [row["source_file"]],
            }
            continue

        if not existing["reference_answer"] and row["reference_answer"]:
            existing["reference_answer"] = row["reference_answer"]
        if row["source_name"] not in existing["source_names"]:
            existing["source_names"].append(row["source_name"])
        if row["source_file"] not in existing["source_files"]:
            existing["source_files"].append(row["source_file"])

    deduped = list(merged.values())
    deduped.sort(key=lambda item: item["question_norm"])
    return deduped


def _assign_metadata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        digest = hashlib.sha1(row["question_norm"].encode("utf-8")).hexdigest()[:12]
        final.append(
            {
                "pack_id": f"Q{idx:04d}_{digest}",
                "question": row["question"],
                "question_norm": row["question_norm"],
                "task_type": _heuristic_task_type(row["question"]),
                "difficulty": _heuristic_difficulty(row["question"]),
                "reference_answer": row["reference_answer"],
                "has_reference_answer": bool(str(row["reference_answer"]).strip()),
                "source_count": len(row["source_names"]),
                "source_names": ",".join(sorted(row["source_names"])),
                "source_files": ",".join(sorted(row["source_files"])),
            }
        )
    return final


def _stratified_split(
    rows: list[dict[str, Any]],
    holdout_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"{row['task_type']}|{row['difficulty']}"
        buckets.setdefault(key, []).append(row)

    calibration: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for _, bucket_rows in sorted(buckets.items()):
        rng.shuffle(bucket_rows)
        n = len(bucket_rows)
        n_holdout = int(round(n * holdout_ratio))
        if n >= 6:
            n_holdout = max(1, n_holdout)
        n_holdout = min(n - 1, n_holdout) if n > 1 else 0
        holdout.extend(bucket_rows[:n_holdout])
        calibration.extend(bucket_rows[n_holdout:])

    calibration.sort(key=lambda item: item["pack_id"])
    holdout.sort(key=lambda item: item["pack_id"])
    return calibration, holdout


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a fixed private eval pack.")
    parser.add_argument(
        "--out-dir",
        default="data/private/eval_pack_v1",
        help="Output directory for frozen pack artifacts.",
    )
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.2,
        help="Holdout ratio in [0,1].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split.",
    )
    args = parser.parse_args()

    if args.holdout_ratio <= 0 or args.holdout_ratio >= 1:
        raise ValueError("--holdout-ratio must be between 0 and 1 (exclusive).")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = _collect_rows()
    deduped_rows = _dedupe_rows(raw_rows)
    rows = _assign_metadata(deduped_rows)

    calibration, holdout = _stratified_split(
        rows=rows,
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
    )

    all_df = pd.DataFrame(rows)
    calibration_df = pd.DataFrame(calibration)
    holdout_df = pd.DataFrame(holdout)

    all_csv = out_dir / "prompts_all.csv"
    calibration_csv = out_dir / "calibration.csv"
    holdout_csv = out_dir / "holdout.csv"
    calibration_jsonl = out_dir / "calibration.jsonl"
    holdout_jsonl = out_dir / "holdout.jsonl"
    manifest_json = out_dir / "manifest.json"

    all_df.to_csv(all_csv, index=False)
    calibration_df.to_csv(calibration_csv, index=False)
    holdout_df.to_csv(holdout_csv, index=False)
    _write_jsonl(calibration_jsonl, calibration)
    _write_jsonl(holdout_jsonl, holdout)

    manifest = {
        "pack_name": out_dir.name,
        "seed": args.seed,
        "holdout_ratio": args.holdout_ratio,
        "total_rows_before_dedup": len(raw_rows),
        "total_unique_prompts": len(rows),
        "calibration_count": len(calibration),
        "holdout_count": len(holdout),
        "task_type_distribution": all_df["task_type"].value_counts().to_dict(),
        "difficulty_distribution": all_df["difficulty"].value_counts().to_dict(),
        "reference_answer_coverage": float(all_df["has_reference_answer"].mean()) if len(all_df) else 0.0,
        "files": {
            "all_csv": str(all_csv),
            "calibration_csv": str(calibration_csv),
            "holdout_csv": str(holdout_csv),
            "calibration_jsonl": str(calibration_jsonl),
            "holdout_jsonl": str(holdout_jsonl),
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Fixed evaluation pack created.")
    print(f"Output directory: {out_dir}")
    print(f"Unique prompts: {len(rows)}")
    print(f"Calibration: {len(calibration)}")
    print(f"Holdout: {len(holdout)}")
    if len(rows) < 200:
        print(
            "Note: fewer than 200 unique prompts were found in local sources. "
            "Add more private prompts if you need the 200-500 target range."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
