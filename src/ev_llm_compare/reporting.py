"""reporting.py — Post-run analysis and report generation.

Usage after experiment.py completes:

    python -m ev_llm_compare.reporting \
        --results artifacts/study_20260402/results_all_*.csv \
        --output-dir artifacts/study_20260402

Generates:
  leaderboard_cross_mode.csv   – cross-mode comparison on answer quality only
  leaderboard_rag_only.csv     – RAG-mode composite ranking
  per_model_summary.csv        – mean metrics per model
  per_mode_summary.csv         – mean metrics per mode
  per_question_diagnostics.csv – per-question score variance
  STUDY_REPORT.md              – human-readable narrative report
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Column groups used in reporting
# ──────────────────────────────────────────────────────────────────────────────

CROSS_MODE_COLS = [
    "answer_accuracy",
    "rouge_l",
    "token_f1",
    "semantic_similarity",
    "normalized_exact_match",
    "answer_quality_score",      # mode-safe composite
    "ragas_answer_correctness",  # ragas library metric
]

RAG_ONLY_COLS = [
    "custom_faithfulness",
    "custom_response_groundedness",
    "custom_context_precision",
    "custom_context_recall",
    "ragas_faithfulness",
    "ragas_context_precision",
    "ragas_context_recall",
    "unsupported_claim_ratio",
    "contradicted_claim_ratio",
    "citation_coverage",
    "rag_composite_score",
]

OPERATIONAL_COLS = [
    "success",
    "abstained",
    "response_length_words",
    "latency_seconds",
    "tokens_in",
    "tokens_out",
    "citation_count",
]

MODEL_DISPLAY_ORDER = [
    "qwen14b",
    "qwen35b",
    "qwen36plus",
    "gemma27b",
    "gemini_flash",
]


# ──────────────────────────────────────────────────────────────────────────────
# Loading & validation
# ──────────────────────────────────────────────────────────────────────────────

def load_results(results_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv, low_memory=False)
    # Ensure mode column exists
    if "mode" not in df.columns and "rag_enabled" in df.columns:
        df["mode"] = df["rag_enabled"].map({True: "rag", False: "no_rag", 1: "rag", 0: "no_rag"})
    if "mode" not in df.columns and "run_name" in df.columns:
        df["mode"] = df["run_name"].apply(
            lambda x: "rag" if str(x).endswith("_rag") else "no_rag"
        )
    # Extract model name from run_name if not present
    if "model_display" not in df.columns and "run_name" in df.columns:
        df["model_display"] = df["run_name"].str.replace(r"_(rag|no_rag)$", "", regex=True)
    return df


def _available_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


# ──────────────────────────────────────────────────────────────────────────────
# Leaderboards
# ──────────────────────────────────────────────────────────────────────────────

def build_cross_mode_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Rank all model×mode combinations by answer_quality_score only.

    This is the only methodologically valid leaderboard that mixes RAG and
    no-RAG rows, because answer_quality_score uses reference metrics only.
    """
    avail = _available_cols(df, CROSS_MODE_COLS)
    if not avail:
        return pd.DataFrame()
    grouped = (
        df.groupby(["run_name", "mode", "model_display"], dropna=False)[avail]
        .mean()
        .reset_index()
    )
    sort_col = "answer_quality_score" if "answer_quality_score" in grouped.columns else avail[0]
    grouped = grouped.sort_values(sort_col, ascending=False, na_position="last")
    grouped.insert(0, "rank", range(1, len(grouped) + 1))
    return grouped.round(4)


def build_rag_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Rank RAG-mode runs by rag_composite_score."""
    rag_df = df[df["mode"] == "rag"].copy()
    if rag_df.empty:
        return pd.DataFrame()
    avail = _available_cols(rag_df, RAG_ONLY_COLS + CROSS_MODE_COLS)
    grouped = (
        rag_df.groupby(["run_name", "model_display"], dropna=False)[avail]
        .mean()
        .reset_index()
    )
    sort_col = "rag_composite_score" if "rag_composite_score" in grouped.columns else (
        "answer_quality_score" if "answer_quality_score" in grouped.columns else avail[0]
    )
    grouped = grouped.sort_values(sort_col, ascending=False, na_position="last")
    grouped.insert(0, "rank", range(1, len(grouped) + 1))
    return grouped.round(4)


# ──────────────────────────────────────────────────────────────────────────────
# Summaries
# ──────────────────────────────────────────────────────────────────────────────

def build_per_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per model (aggregating across both modes)."""
    avail = _available_cols(df, CROSS_MODE_COLS + OPERATIONAL_COLS)
    if "model_display" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("model_display", dropna=False)[avail]
        .mean()
        .reset_index()
        .sort_values(
            "answer_quality_score" if "answer_quality_score" in avail else avail[0],
            ascending=False,
            na_position="last",
        )
        .round(4)
    )


def build_per_mode_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per mode (rag vs no_rag), averaged over all models."""
    avail = _available_cols(df, CROSS_MODE_COLS + OPERATIONAL_COLS)
    return (
        df.groupby("mode", dropna=False)[avail]
        .mean()
        .reset_index()
        .round(4)
    )


def build_per_model_per_mode_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Full summary: mean metrics per (model × mode)."""
    avail = _available_cols(df, CROSS_MODE_COLS + RAG_ONLY_COLS + OPERATIONAL_COLS)
    group_cols = [c for c in ["run_name", "model_display", "mode"] if c in df.columns]
    return (
        df.groupby(group_cols, dropna=False)[avail]
        .mean()
        .reset_index()
        .sort_values(
            "answer_quality_score" if "answer_quality_score" in avail else avail[0],
            ascending=False,
            na_position="last",
        )
        .round(4)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Per-question diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def build_per_question_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-question statistics: mean, std, and worst-model across all runs."""
    avail = _available_cols(df, ["answer_accuracy", "answer_quality_score", "rouge_l"])
    if not avail or "q_id" not in df.columns:
        return pd.DataFrame()

    diag = (
        df.groupby("q_id", dropna=False)[avail]
        .agg(["mean", "std", "min"])
        .reset_index()
    )
    # Flatten multi-level columns
    diag.columns = ["q_id"] + [
        f"{col}_{stat}" for col, stat in diag.columns[1:]
    ]
    # Attach question text
    if "question" in df.columns:
        q_text = df.groupby("q_id")["question"].first().reset_index()
        diag = diag.merge(q_text, on="q_id", how="left")
    return diag.round(4)


# ──────────────────────────────────────────────────────────────────────────────
# Markdown narrative report
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No data available._\n"
    sub = df.head(max_rows)
    return sub.to_markdown(index=False) + "\n"


def build_markdown_report(
    df: pd.DataFrame,
    cross_mode_lb: pd.DataFrame,
    rag_lb: pd.DataFrame,
    per_model: pd.DataFrame,
    per_mode: pd.DataFrame,
    per_q: pd.DataFrame,
) -> str:
    n_responses = len(df)
    n_models = df["model_display"].nunique() if "model_display" in df.columns else "?"
    n_questions = df["q_id"].nunique() if "q_id" in df.columns else "?"
    n_modes = df["mode"].nunique() if "mode" in df.columns else "?"
    success_rate = (df["success"].mean() * 100) if "success" in df.columns else float("nan")
    abstain_rate = (df["abstained"].mean() * 100) if "abstained" in df.columns else float("nan")

    lines: list[str] = []
    lines.append("# EV Supply Chain LLM Comparative Study — Results Report\n")
    lines.append(f"**Total responses:** {n_responses}  ")
    lines.append(f"**Models:** {n_models}  **Modes:** {n_modes}  **Questions:** {n_questions}  ")
    lines.append(f"**Success rate:** {success_rate:.1f}%  **Abstention rate:** {abstain_rate:.1f}%\n")

    lines.append("---\n")
    lines.append("## Methodology Notes\n")
    lines.append(
        "- **Cross-mode leaderboard** uses `answer_quality_score` (reference metrics only), "
        "which is the only composite score valid for comparing RAG vs no-RAG responses.\n"
        "- **RAG-only leaderboard** uses `rag_composite_score` (includes retrieval/grounding metrics). "
        "Do not use this to compare against no-RAG.\n"
        "- `ragas_*` metrics are from the actual ragas library (where available); "
        "`custom_*` metrics are custom LLM-judge implementations.\n"
    )

    lines.append("---\n")
    lines.append("## 1. Cross-Mode Leaderboard (answer quality, both modes)\n")
    lines.append(
        "> Sorted by `answer_quality_score` — the only valid metric for direct RAG vs no-RAG comparison.\n\n"
    )
    lines.append(_fmt_table(cross_mode_lb))

    lines.append("## 2. RAG-Only Leaderboard (retrieval + grounding metrics)\n")
    lines.append(
        "> RAG mode only. Includes retrieval-dependent metrics not applicable to no-RAG.\n\n"
    )
    lines.append(_fmt_table(rag_lb))

    lines.append("## 3. Per-Model Summary (mean across both modes)\n")
    lines.append(_fmt_table(per_model))

    lines.append("## 4. Per-Mode Summary (mean across all models)\n")
    lines.append(_fmt_table(per_mode))

    lines.append("## 5. Per-Question Diagnostics (highest variance questions)\n")
    if not per_q.empty and "answer_quality_score_std" in per_q.columns:
        high_var = per_q.sort_values("answer_quality_score_std", ascending=False)
        lines.append(_fmt_table(high_var.head(15)))
    else:
        lines.append(_fmt_table(per_q))

    lines.append("## 6. Failure / Abstention Summary\n")
    fail_cols = [c for c in ["run_name", "mode", "success", "abstained"] if c in df.columns]
    if fail_cols:
        fail_df = (
            df.groupby([c for c in ["run_name", "mode"] if c in df.columns])[
                [c for c in ["success", "abstained"] if c in df.columns]
            ]
            .mean()
            .reset_index()
            .round(3)
        )
        lines.append(_fmt_table(fail_df))

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def generate_reports(results_csv: Path, output_dir: Path) -> None:
    print(f"[reporting] Loading results from {results_csv}…", flush=True)
    df = load_results(results_csv)
    print(f"[reporting] {len(df)} rows loaded.", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    cross_lb = build_cross_mode_leaderboard(df)
    rag_lb = build_rag_leaderboard(df)
    per_model = build_per_model_summary(df)
    per_mode = build_per_mode_summary(df)
    per_model_mode = build_per_model_per_mode_summary(df)
    per_q = build_per_question_diagnostics(df)

    cross_lb.to_csv(output_dir / "leaderboard_cross_mode.csv", index=False)
    rag_lb.to_csv(output_dir / "leaderboard_rag_only.csv", index=False)
    per_model.to_csv(output_dir / "summary_per_model.csv", index=False)
    per_mode.to_csv(output_dir / "summary_per_mode.csv", index=False)
    per_model_mode.to_csv(output_dir / "summary_per_model_per_mode.csv", index=False)
    per_q.to_csv(output_dir / "diagnostics_per_question.csv", index=False)

    report_md = build_markdown_report(
        df, cross_lb, rag_lb, per_model, per_mode, per_q
    )
    report_path = output_dir / "STUDY_REPORT.md"
    report_path.write_text(report_md, encoding="utf-8")

    print(f"[reporting] Reports written to {output_dir}")
    print(f"[reporting] Summary report → {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate comparison reports from experiment results CSV."
    )
    parser.add_argument("--results", required=True, help="Path to results_all_*.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for report files")
    args = parser.parse_args()
    generate_reports(Path(args.results), Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
