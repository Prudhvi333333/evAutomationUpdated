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
    )
    print(f"Report written to {Path(report_path).resolve()}")
    return 0
