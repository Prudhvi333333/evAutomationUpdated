from pathlib import Path
import unittest

from src.ev_llm_compare.derived_analytics import build_derived_summary_chunks
from src.ev_llm_compare.schemas import TableRow


class DerivedAnalyticsTests(unittest.TestCase):
    def test_build_derived_summary_chunks_includes_global_and_state_scopes(self) -> None:
        rows = [
            TableRow(
                workbook_path=Path("input.xlsx"),
                sheet_name="Data",
                row_number=1,
                values={
                    "Company": "Kia Georgia Inc.",
                    "Category": "OEM",
                    "Updated Location": "West Point, Troup County",
                    "Address": "1 Kia Parkway, West Point, GA 31833",
                    "Employment": "2700",
                    "EV Supply Chain Role": "Vehicle Assembly",
                },
            ),
            TableRow(
                workbook_path=Path("input.xlsx"),
                sheet_name="Data",
                row_number=2,
                values={
                    "Company": "Hyundai Motor Manufacturing Alabama",
                    "Category": "OEM",
                    "Updated Location": "Montgomery, Alabama",
                    "Address": "700 Hyundai Blvd, Montgomery, AL 36105",
                    "Employment": "3000",
                    "EV Supply Chain Role": "Vehicle Assembly",
                },
            ),
        ]

        chunks = build_derived_summary_chunks(rows)

        self.assertEqual(len(chunks), 15)
        scopes = {(chunk.metadata.get("analysis_scope"), chunk.metadata.get("state")) for chunk in chunks}
        self.assertIn(("global", ""), scopes)
        self.assertIn(("state", "Georgia"), scopes)
        self.assertIn(("state", "Alabama"), scopes)

        georgia_titles = [
            chunk.metadata.get("analysis_title", "")
            for chunk in chunks
            if chunk.metadata.get("state") == "Georgia"
        ]
        self.assertTrue(any("for Georgia" in title for title in georgia_titles))
        self.assertTrue(
            any("Scope: Georgia only" in chunk.text for chunk in chunks if chunk.metadata.get("state") == "Georgia")
        )


if __name__ == "__main__":
    unittest.main()
