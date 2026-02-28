from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from .schemas import TableRow, WorkbookNote


def normalize_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return re.sub(r"\s+", " ", str(value)).strip()


def _is_tabular_sheet(df: pd.DataFrame) -> bool:
    populated_columns = [
        column for column in df.columns if normalize_cell(column) and not str(column).startswith("Unnamed")
    ]
    return len(populated_columns) >= 2 and len(df.index) > 0


def load_workbook(workbook_path: str | Path) -> tuple[list[TableRow], list[WorkbookNote]]:
    path = Path(workbook_path).expanduser().resolve()
    excel_file = pd.ExcelFile(path)
    rows: list[TableRow] = []
    notes: list[WorkbookNote] = []

    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet_name)
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            continue

        cleaned_columns = [normalize_cell(column) for column in df.columns]
        df.columns = cleaned_columns

        if _is_tabular_sheet(df):
            for row_idx, (_, series) in enumerate(df.iterrows(), start=1):
                values = {
                    column: normalize_cell(value)
                    for column, value in series.to_dict().items()
                    if normalize_cell(value)
                }
                if values:
                    rows.append(
                        TableRow(
                            workbook_path=path,
                            sheet_name=sheet_name,
                            row_number=row_idx,
                            values=values,
                        )
                    )
        else:
            parts: list[str] = []
            for column in df.columns:
                for value in df[column].tolist():
                    text = normalize_cell(value)
                    if text:
                        parts.append(text)
            if parts:
                notes.append(
                    WorkbookNote(
                        workbook_path=path,
                        sheet_name=sheet_name,
                        text="\n".join(parts),
                    )
                )

    if not rows and not notes:
        raise ValueError(f"No usable content found in workbook: {path}")

    return rows, notes


def load_questions(question_workbook: str | Path, sheet_name: str | None = None) -> list[str]:
    path = Path(question_workbook).expanduser().resolve()
    df = pd.read_excel(path, sheet_name=sheet_name or 0)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        raise ValueError(f"No questions found in workbook: {path}")

    preferred_columns = [
        column
        for column in df.columns
        if normalize_cell(column).lower() in {"question", "questions", "query", "prompt"}
    ]
    target_column = preferred_columns[0] if preferred_columns else df.columns[0]

    questions: list[str] = []
    seen: set[str] = set()
    for value in df[target_column].tolist():
        text = normalize_cell(value)
        if len(text) < 5:
            continue
        if text not in seen:
            seen.add(text)
            questions.append(text)

    if not questions:
        raise ValueError(f"No valid questions found in workbook: {path}")

    return questions
