import unittest

from src.ev_llm_compare.retrieval import HybridRetriever, build_collection_fingerprint
from src.ev_llm_compare.schemas import Chunk, RetrievalResult
from src.ev_llm_compare.settings import RetrievalSettings


class RetrievalTests(unittest.TestCase):
    def test_collection_fingerprint_changes_when_chunk_content_changes(self) -> None:
        chunks_a = [
            Chunk(chunk_id="1", text="Company: A", metadata={"row_key": "row-1"}),
            Chunk(chunk_id="2", text="Company: B", metadata={"row_key": "row-2"}),
        ]
        chunks_b = [
            Chunk(chunk_id="1", text="Company: A", metadata={"row_key": "row-1"}),
            Chunk(chunk_id="2", text="Company: C", metadata={"row_key": "row-2"}),
        ]

        self.assertNotEqual(
            build_collection_fingerprint(chunks_a, "embedding-model"),
            build_collection_fingerprint(chunks_b, "embedding-model"),
        )

    def test_query_plan_detects_structured_aggregation_queries(self) -> None:
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.known_categories = ["Tier 1", "Tier 1/2"]
        retriever.known_companies = ["Acme EV"]
        retriever.known_locations = []
        retriever.known_primary_oems = []
        retriever.role_terms = ["battery pack", "battery cell"]

        plan = HybridRetriever._plan_query(
            retriever,
            "Show all Tier 1 companies grouped by EV Supply Chain Role for battery pack.",
        )

        self.assertEqual(plan.intent, "aggregation")
        self.assertTrue(plan.prefer_structured)
        self.assertTrue(plan.group_by_role)
        self.assertEqual(plan.matched_categories, ["Tier 1"])
        self.assertEqual(plan.matched_role_terms, ["battery pack"])
        self.assertFalse(plan.relationship_heavy)

    def test_query_plan_marks_relationship_heavy_and_adds_relationship_queries(self) -> None:
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.known_categories = ["OEM", "Tier 1"]
        retriever.known_companies = []
        retriever.known_locations = []
        retriever.known_primary_oems = []
        retriever.role_terms = ["vehicle assembly"]

        plan = HybridRetriever._plan_query(
            retriever,
            "Show all Vehicle Assembly OEMs in Georgia and the full set of Tier 1 suppliers connected to each within the state.",
        )

        self.assertTrue(plan.relationship_heavy)
        self.assertTrue(plan.broad_context_required)
        self.assertTrue(any("relationship map" in query for query in plan.dense_queries))

    def test_structured_summary_prefers_grouped_output_for_exhaustive_role_queries(self) -> None:
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.settings = RetrievalSettings(structured_summary_limit=2)
        query_plan = HybridRetriever._plan_query(
            self._seed_retriever_for_summary(retriever),
            "Show all Tier 1 companies grouped by EV Supply Chain Role.",
        )
        matched_rows = [
            {
                "company": "A",
                "category": "Tier 1",
                "ev_supply_chain_role": "Battery Pack",
                "product_service": "Pack",
                "location": "Atlanta",
                "employment": "100",
                "source_file": "input.xlsx",
                "sheet_name": "Data",
                "row_number": "1",
                "row_key": "row-1",
                "row_summary": "A",
            },
            {
                "company": "B",
                "category": "Tier 1",
                "ev_supply_chain_role": "Battery Pack",
                "product_service": "Pack",
                "location": "Atlanta",
                "employment": "100",
                "source_file": "input.xlsx",
                "sheet_name": "Data",
                "row_number": "2",
                "row_key": "row-2",
                "row_summary": "B",
            },
            {
                "company": "C",
                "category": "Tier 1",
                "ev_supply_chain_role": "Battery Pack",
                "product_service": "Pack",
                "location": "Atlanta",
                "employment": "100",
                "source_file": "input.xlsx",
                "sheet_name": "Data",
                "row_number": "3",
                "row_key": "row-3",
                "row_summary": "C",
            },
        ]

        summary = HybridRetriever._build_structured_summary(retriever, query_plan, matched_rows)
        self.assertIn("Grouped by EV Supply Chain Role:", summary)
        self.assertIn("- Battery Pack: A; B; C", summary)

    def test_structured_matches_preserve_row_summary_fields_for_compaction(self) -> None:
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.settings = RetrievalSettings(final_top_k=8, structured_exhaustive_limit=20)
        retriever.row_records = {
            "row-1": {
                "row_key": "row-1",
                "company": "PPG Industries Inc.",
                "category": "Tier 1",
                "ev_supply_chain_role": "General Automotive",
                "product_service": "Coatings",
                "primary_oems": "Kia Georgia Inc.",
                "location": "Troup County",
                "industry_group": "Coatings",
                "primary_facility_type": "Manufacturing Plant",
                "supplier_or_affiliation_type": "Supplier",
                "classification_method": "Workbook row",
                "employment": "120",
                "ev_battery_relevant": "Indirect",
                "source_file": "input.xlsx",
                "sheet_name": "Data",
                "row_number": "44",
                "row_summary": "Company: PPG Industries Inc. | Category: Tier 1 | Primary OEMs: Kia Georgia Inc. | Updated Location: Troup County | Employment: 120",
            }
        }
        plan = HybridRetriever._plan_query(
            self._seed_retriever_for_summary(retriever),
            "Show all Vehicle Assembly OEMs in Georgia and the full set of Tier 1 suppliers connected to each within the state.",
        )

        matches = HybridRetriever._structured_matches(retriever, plan)

        self.assertEqual(matches[1].metadata["row_summary"], retriever.row_records["row-1"]["row_summary"])
        self.assertEqual(matches[1].metadata["primary_oems"], "Kia Georgia Inc.")
        self.assertEqual(matches[1].metadata["location"], "Troup County")

    def test_query_plan_does_not_treat_oem_contracts_as_oem_category(self) -> None:
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.known_categories = ["OEM", "Tier 1", "Tier 2/3"]
        retriever.known_companies = []
        retriever.known_locations = []
        retriever.known_primary_oems = []
        retriever.role_terms = ["dc fast charging"]

        plan = HybridRetriever._plan_query(
            retriever,
            "Which suppliers manufacture DC fast charging hardware and have existing OEM contracts?",
        )

        self.assertEqual(plan.matched_categories, [])

    def test_match_locations_skips_generic_georgia(self) -> None:
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.known_locations = ["Georgia", "Lawrenceville, Gwinnett County"]

        matches = HybridRetriever._match_locations(
            retriever,
            "map all thermal management suppliers in georgia",
        )

        self.assertEqual(matches, [])

    def test_is_exhaustive_question_detects_network_style_prompt(self) -> None:
        retriever = HybridRetriever.__new__(HybridRetriever)
        plan = HybridRetriever._plan_query(
            self._seed_retriever_for_summary(retriever),
            "Show all Vehicle Assembly OEMs in Georgia and the full set of Tier 1 suppliers connected to each within the state.",
        )

        self.assertTrue(HybridRetriever._is_exhaustive_question(retriever, plan))

    def test_select_context_results_allows_more_relationship_rows_per_company(self) -> None:
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.settings = RetrievalSettings(max_chunks_per_company=2, final_top_k=12)
        plan = HybridRetriever._plan_query(
            self._seed_retriever_for_summary(retriever),
            "Show all Vehicle Assembly OEMs in Georgia and the full set of Tier 1 suppliers connected to each within the state.",
        )
        candidates = [
            self._make_result(f"lear-{index}", "Lear Corporation", 0.95 - (index * 0.01), primary_oems="Kia Georgia Inc.")
            for index in range(1, 5)
        ]
        candidates.append(self._make_result("ppg-1", "PPG Industries Inc.", 0.8, primary_oems="Kia Georgia Inc."))

        selected = HybridRetriever._select_context_results(
            retriever,
            query_plan=plan,
            structured_results=[],
            candidates=candidates,
            limit=6,
        )

        learner_count = sum(1 for result in selected if result.metadata.get("company") == "Lear Corporation")
        self.assertEqual(learner_count, 4)

    def _make_result(
        self,
        row_key: str,
        company: str,
        final_score: float,
        *,
        primary_oems: str,
    ) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=row_key,
            text=f"Company: {company} | Primary OEMs: {primary_oems} | Updated Location: Troup County",
            metadata={
                "chunk_type": "supply_chain_theme",
                "company": company,
                "primary_oems": primary_oems,
                "category": "Tier 1",
                "row_key": row_key,
                "source_file": "input.xlsx",
                "sheet_name": "Data",
                "row_number": row_key,
            },
            dense_score=final_score,
            lexical_score=final_score,
            final_score=final_score,
        )

    def _seed_retriever_for_summary(self, retriever: HybridRetriever) -> HybridRetriever:
        retriever.known_categories = ["Tier 1", "OEM"]
        retriever.known_companies = []
        retriever.known_locations = []
        retriever.known_primary_oems = []
        retriever.role_terms = ["battery pack", "vehicle assembly"]
        return retriever


if __name__ == "__main__":
    unittest.main()
