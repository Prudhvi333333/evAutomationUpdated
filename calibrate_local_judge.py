#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ev_llm_compare.calibration import export_calibration_workbook, run_judge_calibration
from ev_llm_compare.settings import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate the local evaluation judge against a small human-labeled set."
    )
    parser.add_argument("--input", required=True, help="Path to the calibration workbook or CSV.")
    parser.add_argument("--sheet", default=None, help="Optional Excel sheet name.")
    parser.add_argument("--output-dir", default="artifacts/judge_calibration", help="Directory for the calibration workbook.")
    parser.add_argument("--judge-provider", default=None, help="Optional override for the judge provider.")
    parser.add_argument("--judge-model", default=None, help="Optional override for the judge model.")
    parser.add_argument("--max-retries", type=int, default=None, help="Optional retry override.")
    return parser


def main() -> int:
    config = load_config()
    parser = build_parser()
    args = parser.parse_args()

    result = run_judge_calibration(
        args.input,
        judge_provider=args.judge_provider or config.evaluation.judge_provider,
        judge_model=args.judge_model or config.evaluation.judge_model,
        max_retries=args.max_retries if args.max_retries is not None else config.evaluation.max_retries,
        sheet_name=args.sheet,
    )
    workbook_path = export_calibration_workbook(args.output_dir, result)
    print(f"Calibration workbook written to {workbook_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
