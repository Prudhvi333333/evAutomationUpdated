import unittest

from src.ev_llm_compare.evaluation import _segment_response_units
from src.ev_llm_compare.prompts import compact_context_segments
from src.ev_llm_compare.schemas import RetrievalResult


class SegmentationAndPromptTests(unittest.TestCase):
    def test_segment_response_units_splits_dense_bracketed_catalog_entries(self) -> None:
        reference = (
            "There are 3 Tier 1/2 companies in Georgia. "
            "F&P Georgia Manufacturing [Tier 1/2] | Role: Battery Pack | Product: Lithium-ion battery recycler "
            "Hyundai MOBIS (Georgia) [Tier 1/2] | Role: General Automotive | Product: Electronic automotive components "
            "Hitachi Astemo [Tier 1/2] | Role: Battery Cell | Product: Battery cells for electric mobility"
        )

        units = _segment_response_units(reference)

        self.assertEqual(units[0], "There are 3 Tier 1/2 companies in Georgia.")
        self.assertGreaterEqual(len(units), 4)
        self.assertTrue(any(unit.startswith("F&P Georgia Manufacturing [Tier 1/2]") for unit in units))
        self.assertTrue(any(unit.startswith("Hyundai MOBIS (Georgia) [Tier 1/2]") for unit in units))
        self.assertTrue(any(unit.startswith("Hitachi Astemo [Tier 1/2]") for unit in units))

    def test_segment_response_units_splits_semicolon_delimited_structured_entries(self) -> None:
        answer = (
            "Company: Novelis Inc. | Updated Location: Gainesville, Hall County | Primary Facility Type: Manufacturing Plant; "
            "Company: Novelis Inc. | Updated Location: Trenton, Dade County | Primary Facility Type: Manufacturing Plant; "
            "Company: Novelis Inc. | Updated Location: Lawrenceville, Gwinnett County | Primary Facility Type: Manufacturing Plant"
        )

        units = _segment_response_units(answer)

        self.assertEqual(len(units), 3)
        self.assertTrue(all("Primary Facility Type" in unit for unit in units))
        self.assertTrue(any("Gainesville" in unit for unit in units))
        self.assertTrue(any("Trenton" in unit for unit in units))
        self.assertTrue(any("Lawrenceville" in unit for unit in units))

    def test_compact_context_segments_expand_network_questions_and_keep_row_summary(self) -> None:
        question = "Show all Vehicle Assembly OEMs in Georgia and the full set of Tier 1 suppliers connected to each within the state."
        results = [
            RetrievalResult(
                chunk_id="summary",
                text="Structured workbook matches from exact metadata filters:\nMatched rows: 6\nDetailed rows:",
                metadata={"chunk_type": "structured_match_summary", "company": "", "source_file": "input.xlsx", "sheet_name": "Data"},
                dense_score=1.0,
                lexical_score=1.0,
                final_score=1.0,
            ),
            RetrievalResult(
                chunk_id="structured-row-1",
                text="Company: Kia Georgia Inc. | Category: OEM | EV Supply Chain Role: Vehicle Assembly | Updated Location: West Point, Troup County | Employment: 3000",
                metadata={
                    "chunk_type": "structured_row_match",
                    "company": "Kia Georgia Inc.",
                    "category": "OEM",
                    "ev_supply_chain_role": "Vehicle Assembly",
                    "location": "West Point, Troup County",
                    "employment": "3000",
                    "row_summary": "Company: Kia Georgia Inc. | Category: OEM | EV Supply Chain Role: Vehicle Assembly | Updated Location: West Point, Troup County | Employment: 3000",
                    "source_file": "input.xlsx",
                    "sheet_name": "Data",
                    "row_number": "12",
                    "row_key": "row-12",
                },
                dense_score=0.99,
                lexical_score=0.99,
                final_score=0.99,
            ),
            RetrievalResult(
                chunk_id="structured-row-2",
                text="Company: PPG Industries Inc. | Category: Tier 1 | Primary OEMs: Kia Georgia Inc. | Updated Location: Troup County | Employment: 120",
                metadata={
                    "chunk_type": "structured_row_match",
                    "company": "PPG Industries Inc.",
                    "category": "Tier 1",
                    "primary_oems": "Kia Georgia Inc.",
                    "location": "Troup County",
                    "employment": "120",
                    "row_summary": "Company: PPG Industries Inc. | Category: Tier 1 | Primary OEMs: Kia Georgia Inc. | Updated Location: Troup County | Employment: 120",
                    "source_file": "input.xlsx",
                    "sheet_name": "Data",
                    "row_number": "44",
                    "row_key": "row-44",
                },
                dense_score=0.98,
                lexical_score=0.98,
                final_score=0.98,
            ),
            RetrievalResult(
                chunk_id="row-1",
                text="Company: Teklas USA | Category: Tier 1 | Primary OEMs: Kia Georgia Inc. | Updated Location: Troup County | Employment: 500",
                metadata={"chunk_type": "row_full", "company": "Teklas USA", "category": "Tier 1", "primary_oems": "Kia Georgia Inc.", "location": "Troup County", "employment": "500", "row_summary": "Company: Teklas USA | Category: Tier 1 | Primary OEMs: Kia Georgia Inc. | Updated Location: Troup County | Employment: 500", "row_key": "row-1", "source_file": "input.xlsx", "sheet_name": "Data", "row_number": "45"},
                dense_score=0.97,
                lexical_score=0.97,
                final_score=0.97,
            ),
            RetrievalResult(
                chunk_id="row-2",
                text="Company: Lear Corporation | Category: Tier 1 | Primary OEMs: Kia Georgia Inc. | Updated Location: Columbus, Muscogee County | Employment: 900",
                metadata={"chunk_type": "row_full", "company": "Lear Corporation", "category": "Tier 1", "primary_oems": "Kia Georgia Inc.", "location": "Columbus, Muscogee County", "employment": "900", "row_summary": "Company: Lear Corporation | Category: Tier 1 | Primary OEMs: Kia Georgia Inc. | Updated Location: Columbus, Muscogee County | Employment: 900", "row_key": "row-2", "source_file": "input.xlsx", "sheet_name": "Data", "row_number": "46"},
                dense_score=0.96,
                lexical_score=0.96,
                final_score=0.96,
            ),
            RetrievalResult(
                chunk_id="row-3",
                text="Company: Hyundai MOBIS (Georgia) | Category: Tier 1 | Primary OEMs: Hyundai Motor Group | Updated Location: West Point, Troup County | Employment: 800",
                metadata={"chunk_type": "row_full", "company": "Hyundai MOBIS (Georgia)", "category": "Tier 1", "primary_oems": "Hyundai Motor Group", "location": "West Point, Troup County", "employment": "800", "row_summary": "Company: Hyundai MOBIS (Georgia) | Category: Tier 1 | Primary OEMs: Hyundai Motor Group | Updated Location: West Point, Troup County | Employment: 800", "row_key": "row-3", "source_file": "input.xlsx", "sheet_name": "Data", "row_number": "47"},
                dense_score=0.95,
                lexical_score=0.95,
                final_score=0.95,
            ),
        ]

        blocks = compact_context_segments(question, results, max_results=4, max_chars=6000)

        self.assertEqual(len(blocks), 6)
        joined = "\n".join(blocks)
        self.assertIn("Primary OEMs: Kia Georgia Inc.", joined)
        self.assertIn("Updated Location: West Point, Troup County", joined)

    def test_compact_context_segments_keep_multiple_rows_for_same_company_locations(self) -> None:
        question = "What locations does Novelis Inc. operate in, and what primary facility types are associated with each location?"
        results = [
            RetrievalResult(
                chunk_id="n1",
                text="Company: Novelis Inc. | Updated Location: Gainesville, Hall County | Primary Facility Type: Manufacturing Plant",
                metadata={"chunk_type": "row_full", "company": "Novelis Inc.", "location": "Gainesville, Hall County", "primary_facility_type": "Manufacturing Plant", "row_summary": "Company: Novelis Inc. | Updated Location: Gainesville, Hall County | Primary Facility Type: Manufacturing Plant", "row_key": "n1", "source_file": "input.xlsx", "sheet_name": "Data", "row_number": "10"},
                dense_score=0.99,
                lexical_score=0.99,
                final_score=0.99,
            ),
            RetrievalResult(
                chunk_id="n2",
                text="Company: Novelis Inc. | Updated Location: Trenton, Dade County | Primary Facility Type: Manufacturing Plant",
                metadata={"chunk_type": "row_full", "company": "Novelis Inc.", "location": "Trenton, Dade County", "primary_facility_type": "Manufacturing Plant", "row_summary": "Company: Novelis Inc. | Updated Location: Trenton, Dade County | Primary Facility Type: Manufacturing Plant", "row_key": "n2", "source_file": "input.xlsx", "sheet_name": "Data", "row_number": "11"},
                dense_score=0.98,
                lexical_score=0.98,
                final_score=0.98,
            ),
            RetrievalResult(
                chunk_id="n3",
                text="Company: Novelis Inc. | Updated Location: Lawrenceville, Gwinnett County | Primary Facility Type: Manufacturing Plant",
                metadata={"chunk_type": "row_full", "company": "Novelis Inc.", "location": "Lawrenceville, Gwinnett County", "primary_facility_type": "Manufacturing Plant", "row_summary": "Company: Novelis Inc. | Updated Location: Lawrenceville, Gwinnett County | Primary Facility Type: Manufacturing Plant", "row_key": "n3", "source_file": "input.xlsx", "sheet_name": "Data", "row_number": "12"},
                dense_score=0.97,
                lexical_score=0.97,
                final_score=0.97,
            ),
        ]

        blocks = compact_context_segments(question, results, max_results=3, max_chars=3000)

        joined = "\n".join(blocks)
        self.assertIn("Gainesville, Hall County", joined)
        self.assertIn("Trenton, Dade County", joined)
        self.assertIn("Lawrenceville, Gwinnett County", joined)

    def test_compact_context_segments_fall_back_to_row_summary_when_metadata_is_sparse(self) -> None:
        question = "Show all Tier 1 suppliers linked to Kia Georgia Inc. within the state."
        results = [
            RetrievalResult(
                chunk_id="sparse-1",
                text="Company: PPG Industries Inc. | Category: Tier 1 | Primary OEMs: Kia Georgia Inc. | Updated Location: Troup County",
                metadata={"chunk_type": "row_full", "company": "PPG Industries Inc.", "row_summary": "Company: PPG Industries Inc. | Category: Tier 1 | Primary OEMs: Kia Georgia Inc. | Updated Location: Troup County", "row_key": "sparse-1", "source_file": "input.xlsx", "sheet_name": "Data", "row_number": "21"},
                dense_score=0.9,
                lexical_score=0.9,
                final_score=0.9,
            )
        ]

        blocks = compact_context_segments(question, results, max_results=1, max_chars=1000)

        self.assertIn("Primary OEMs: Kia Georgia Inc.", blocks[0])
        self.assertIn("Updated Location: Troup County", blocks[0])


if __name__ == "__main__":
    unittest.main()
