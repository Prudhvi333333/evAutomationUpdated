from pathlib import Path
import unittest
from unittest.mock import patch

from src.ev_llm_compare.runner import ComparisonRunner
from src.ev_llm_compare.schemas import Chunk, TableRow
from src.ev_llm_compare.settings import AppConfig, ModelSpec


class _FakeRetriever:
    instances: list["_FakeRetriever"] = []

    def __init__(self, chunks, settings, qdrant_path):
        self.chunks = chunks
        self.settings = settings
        self.qdrant_path = qdrant_path
        self.closed = False
        self.__class__.instances.append(self)

    def retrieve(self, question: str):
        return []

    def close(self) -> None:
        self.closed = True


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeRetriever.instances = []

    def test_main_runner_includes_derived_summary_chunks(self) -> None:
        config = AppConfig(
            models=[
                ModelSpec(
                    run_name="qwen_rag",
                    provider="ollama",
                    model_name="qwen-test",
                    rag_enabled=True,
                )
            ]
        )
        runner = ComparisonRunner(config)
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
                    "Primary OEMs": "",
                    "EV Supply Chain Role": "Vehicle Assembly",
                },
            )
        ]
        derived_chunks = [
            Chunk(
                chunk_id="derived-1",
                text="Derived analytic summary table: county supplier density.",
                metadata={"chunk_type": "derived_analytic_summary"},
            )
        ]

        with (
            patch("src.ev_llm_compare.runner.load_workbook", return_value=(rows, [])),
            patch("src.ev_llm_compare.runner.load_questions", return_value=["Which county has the highest employment?"]),
            patch("src.ev_llm_compare.runner.build_derived_summary_chunks", return_value=derived_chunks) as mock_derived,
            patch("src.ev_llm_compare.runner.HybridRetriever", _FakeRetriever),
            patch("src.ev_llm_compare.runner.create_client", return_value=object()),
            patch("src.ev_llm_compare.runner.safe_generate", return_value=("Fulton County", 0.1, True, None)),
            patch("src.ev_llm_compare.runner.export_results", return_value=Path("artifacts/results/fake.xlsx")),
            patch.object(ComparisonRunner, "_resolve_reference_workbook", return_value=None),
        ):
            output_path = runner.run(
                data_workbook="data.xlsx",
                question_workbook="questions.xlsx",
                skip_evaluation=True,
                selected_run_names=["qwen_rag"],
                export_response_files=False,
                single_sheet_only=True,
            )

        self.assertEqual(output_path, Path("artifacts/results/fake.xlsx"))
        mock_derived.assert_called_once_with(rows)
        self.assertEqual(len(_FakeRetriever.instances), 1)
        retriever = _FakeRetriever.instances[0]
        self.assertTrue(retriever.closed)
        self.assertIn("derived_analytic_summary", {chunk.metadata.get("chunk_type") for chunk in retriever.chunks})


if __name__ == "__main__":
    unittest.main()
