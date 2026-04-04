"""Canonical file resolution and workbook validation utilities.

All entry points must use this module for locating the data workbook,
questions, and golden answers. The order of candidates is opinionated to
prefer the updated workbook with latitude/longitude columns.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_WORKBOOK_CANDIDATES: list[Path] = [
    PROJECT_ROOT / "data" / "GNEM - Auto Landscape Lat Long Updated.xlsx",
    PROJECT_ROOT / "data" / "GNEM landscape lat long updated.xlsx",
    PROJECT_ROOT / "data" / "GNEM_auto_landscape_lat_long_updated.xlsx",
    PROJECT_ROOT / "data" / "GNEM auto landscape.xlsx",
    PROJECT_ROOT / "data" / "GNEM_auto_landscape.xlsx",
    PROJECT_ROOT / "data" / "GNEM updated excel.xlsx",
    PROJECT_ROOT / "GNEM updated excel (1).xlsx",
]

QUESTION_CANDIDATES: list[Path] = [
    PROJECT_ROOT / "data" / "Human validated 50 questions.xlsx",
    PROJECT_ROOT / "data" / "GNEM_Golden_Questions.xlsx",
    PROJECT_ROOT / "data" / "Sample questions.xlsx",
    PROJECT_ROOT / "Sample questions.xlsx",
    PROJECT_ROOT / "data" / "questions.xlsx",
]

GOLDEN_ANSWER_CANDIDATES: list[Path] = [
    PROJECT_ROOT / "data" / "Human validated 50 questions.xlsx",
    PROJECT_ROOT / "data" / "human_validated_answers.xlsx",
    PROJECT_ROOT / "data" / "Golden_answers.xlsx",
    PROJECT_ROOT / "artifacts" / "Golden_answers_updated.xlsx",
    PROJECT_ROOT / "artifacts" / "Golden_answers.xlsx",
]


def resolve_file(hint: Optional[str], candidates: list[Path], label: str) -> Path:
    """Resolve a file path by explicit hint or ordered candidate search."""

    if hint:
        path = Path(hint).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
        print(f"[file_resolution] {label}: {path}")
        return path

    for candidate in candidates:
        if candidate.exists():
            print(f"[file_resolution] {label}: {candidate}")
            return candidate.resolve()

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find {label}. Searched:\n  {searched}\n"
        "Pass the path explicitly via CLI argument."
    )


def resolve_data_workbook(hint: Optional[str] = None) -> Path:
    return resolve_file(hint, DATA_WORKBOOK_CANDIDATES, "Data workbook")


def resolve_questions(hint: Optional[str] = None) -> Path:
    return resolve_file(hint, QUESTION_CANDIDATES, "Questions file")


def resolve_golden_answers(hint: Optional[str] = None) -> Optional[Path]:
    try:
        return resolve_file(hint, GOLDEN_ANSWER_CANDIDATES, "Golden answers")
    except FileNotFoundError:
        print("[file_resolution] Golden answers: not found (fallback to generated references)")
        return None


def validate_data_workbook(path: Path) -> None:
    """Lightweight validation of expected workbook structure."""

    import openpyxl  # local import to avoid hard dependency in non-Excel flows

    wb = openpyxl.load_workbook(path, read_only=True)
    print(f"[validate] Workbook: {path.name}")
    print(f"[validate] Sheets: {wb.sheetnames}")
    print(f"[validate] File size: {path.stat().st_size / 1024:.1f} KB")
    wb.close()

    df = pd.read_excel(path, sheet_name=0, nrows=5)
    cols_lower = [str(c).lower() for c in df.columns]
    has_lat = any("lat" in c for c in cols_lower)
    has_long = any("lon" in c for c in cols_lower)
    has_updated_loc = any("updated location" in c for c in cols_lower)

    if has_lat and has_long:
        print("[validate] ✓ Lat/Long columns detected")
    else:
        print(f"[validate] ⚠ WARNING: No Lat/Long columns found. Columns: {list(df.columns)}")

    if has_updated_loc:
        print("[validate] ✓ Updated Location column detected")
    else:
        print("[validate] ⚠ WARNING: No 'Updated Location' column found")

    row_count = len(pd.read_excel(path, sheet_name=0))
    print(f"[validate] Row count (first sheet): {row_count}")


def workbook_fingerprint(path: Path) -> str:
    """Compute a short SHA-256 fingerprint for the workbook contents."""

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()[:16]

