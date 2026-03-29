from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.ev_llm_compare.calibration import export_calibration_workbook, run_judge_calibration


class CalibrationTests(unittest.TestCase):
    def test_run_judge_calibration_merges_expected_and_judged_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "calibration.csv"
            pd.DataFrame(
                [
                    {
                        "question": "What is A?",
                        "answer": "A is in Georgia.",
                        "reference_answer": "A is in Georgia.",
                        "retrieved_contexts": '["A is in Georgia."]',
                        "expected_answer_accuracy": 1.0,
                        "expected_faithfulness": 1.0,
                    }
                ]
            ).to_csv(csv_path, index=False)

            fake_metrics = pd.DataFrame(
                [
                    {
                        "run_name": "calibration_local_judge",
                        "question": "What is A?",
                        "answer_accuracy": 0.9,
                        "faithfulness": 0.8,
                        "response_groundedness": 0.75,
                        "context_precision": 1.0,
                        "context_recall": 1.0,
                        "grounded_claim_ratio": 0.8,
                        "unsupported_claim_ratio": 0.2,
                        "contradicted_claim_ratio": 0.0,
                    }
                ]
            )

            with patch(
                "src.ev_llm_compare.calibration.run_evaluation_metrics",
                return_value=(fake_metrics, pd.DataFrame()),
            ):
                result = run_judge_calibration(
                    csv_path,
                    judge_provider="ollama",
                    judge_model="judge-model",
                )

            self.assertEqual(len(result.scored_rows), 1)
            self.assertAlmostEqual(result.scored_rows.iloc[0]["answer_accuracy"], 0.9)
            self.assertAlmostEqual(result.scored_rows.iloc[0]["answer_accuracy_abs_error"], 0.1)
            self.assertEqual(result.summary.iloc[0]["metric"], "answer_accuracy")

    def test_export_calibration_workbook_writes_expected_sheets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = type(
                "CalibrationResult",
                (),
                {
                    "scored_rows": pd.DataFrame([{"question": "What is A?", "answer_accuracy": 0.9}]),
                    "summary": pd.DataFrame([{"metric": "answer_accuracy", "mae": 0.1}]),
                },
            )()
            workbook_path = export_calibration_workbook(tmp_dir, result)
            self.assertTrue(workbook_path.exists())


if __name__ == "__main__":
    unittest.main()
