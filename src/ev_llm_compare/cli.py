from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare multiple LLMs on Excel-based EV supply chain questions."
    )
    parser.add_argument(
        "--data-workbook",
        default="GNEM updated excel (1).xlsx",
        help="Workbook containing the source data to index.",
    )
    parser.add_argument(
        "--question-workbook",
        default="Sample questions.xlsx",
        help="Workbook containing evaluation questions.",
    )
    parser.add_argument(
        "--question-sheet",
        default=None,
        help="Optional sheet name for the question workbook.",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Run model comparison without RAGAS evaluation.",
    )
    parser.add_argument(
        "--question-limit",
        type=int,
        default=None,
        help="Optional limit for the number of questions to run.",
    )
    parser.add_argument(
        "--run-name",
        action="append",
        dest="run_names",
        default=None,
        help="Limit execution to one or more configured run names. Repeat the flag for multiple runs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for the comparison workbook.",
    )
    parser.add_argument(
        "--response-dir",
        default="artifacts/correct_responses",
        help="Directory where per-run response files will be written.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    from .runner import ComparisonRunner
    from .settings import load_config

    config = load_config()
    runner = ComparisonRunner(config)
    report_path = runner.run(
        data_workbook=args.data_workbook,
        question_workbook=args.question_workbook,
        question_sheet=args.question_sheet,
        skip_ragas=args.skip_ragas,
        question_limit=args.question_limit,
        selected_run_names=args.run_names,
        output_dir=args.output_dir,
        response_output_dir=args.response_dir,
    )
    print(f"Report written to {Path(report_path).resolve()}")
    return 0
