from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.ev_llm_compare.excel_loader import (
    infer_state,
    load_questions,
    load_reference_answers,
    load_workbook,
    preferred_location,
)


class ExcelLoaderTests(unittest.TestCase):
    def test_load_workbook_handles_tabular_and_note_sheets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "input.xlsx"
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                pd.DataFrame(
                    [{"Company": "A", "Category": "Tier 1", "Location": "Atlanta"}]
                ).to_excel(writer, sheet_name="Data", index=False)
                pd.DataFrame({"Definitions": ["Tier 1 means direct supplier"]}).to_excel(
                    writer, sheet_name="Definitions", index=False
                )

            rows, notes = load_workbook(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(notes), 1)
            self.assertEqual(rows[0].values["Company"], "A")

    def test_load_questions_picks_question_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "questions.xlsx"
            pd.DataFrame({"Question": ["What is A?", "What is B?"]}).to_excel(
                path, index=False
            )
            questions = load_questions(path)
            self.assertEqual(questions, ["What is A?", "What is B?"])

    def test_load_reference_answers_reads_golden_answer_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "golden.xlsx"
            pd.DataFrame(
                {
                    "Question": ["What is A?", "What is B?"],
                    "Golden_Answer": ["Ref A", "Ref B"],
                }
            ).to_excel(path, index=False)
            references = load_reference_answers(path)
            self.assertEqual(references, {"What is A?": "Ref A", "What is B?": "Ref B"})

    def test_load_reference_answers_reads_human_validated_answers_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "golden.xlsx"
            pd.DataFrame(
                {
                    "Num": [1, 2],
                    "Question": ["What is A?", "What is B?"],
                    "Human validated answers": ["Ref A", "Ref B"],
                }
            ).to_excel(path, index=False)
            references = load_reference_answers(path)
            self.assertEqual(references, {"What is A?": "Ref A", "What is B?": "Ref B"})

    def test_load_reference_answers_preserves_multiline_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "golden.xlsx"
            pd.DataFrame(
                {
                    "Question": ["What is A?"],
                    "Human validated answers": ["Summary line\n\n- Item A\n- Item B"],
                }
            ).to_excel(path, index=False)

            references = load_reference_answers(path)

            self.assertEqual(
                references["What is A?"],
                "Summary line\n\n- Item A\n- Item B",
            )

    def test_preferred_location_uses_updated_location_before_location(self) -> None:
        values = {"Location": "Old County", "Updated Location": "New County"}
        self.assertEqual(preferred_location(values), "New County")
        self.assertEqual(preferred_location({"Location": "Old County"}), "Old County")

    def test_infer_state_uses_address_abbreviation(self) -> None:
        values = {"Address": "1 Kia Parkway, West Point, GA 31833"}
        self.assertEqual(infer_state(values), "Georgia")

    def test_infer_state_accepts_full_state_name(self) -> None:
        values = {"Updated Location": "Montgomery, Alabama"}
        self.assertEqual(infer_state(values), "Alabama")

    def test_infer_state_returns_empty_when_missing(self) -> None:
        values = {"Location": "Troup County"}
        self.assertEqual(infer_state(values), "")


if __name__ == "__main__":
    unittest.main()
