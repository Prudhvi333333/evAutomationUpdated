from pathlib import Path
import re
import unittest
from unittest.mock import patch

import pandas as pd

from src.ev_llm_compare.evaluation import (
    _compute_overall_metric_score_pct,
    _score_response_metrics,
    _segment_response_units,
)
from src.ev_llm_compare.models import LLMClient
from src.ev_llm_compare.prompts import compact_context_segments
from src.ev_llm_compare.schemas import ModelResponse, RetrievalResult


ARTIFACT_DIR = Path("artifacts/correct_responses/qwen_rag_human50_localjudge")
SUPPLIER_LIST_QUESTION = "Show all Tier 1/2 suppliers in Georgia, list their EV Supply Chain Role and Product / Service."
NETWORK_QUESTION = "Show all Vehicle Assembly OEMs in Georgia and the full set of Tier 1 suppliers connected to each within the state."


class _DummyJudgeClient(LLMClient):
    provider = "dummy"
    model_name = "dummy-model"

    def generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str | None = None,
    ) -> str:
        raise AssertionError("Judge calls should be patched in this test")


class QwenHuman50LocalJudgeRegressionTests(unittest.TestCase):
    def test_artifact_rows_ignore_grounded_claim_ratio_in_final_score(self) -> None:
        frame = pd.read_csv(ARTIFACT_DIR / "qwen_rag_responses.csv")

        metric_columns = [
            "answer_accuracy",
            "faithfulness",
            "response_groundedness",
            "context_precision",
            "context_recall",
            "grounded_claim_ratio",
            "unsupported_claim_ratio",
            "contradicted_claim_ratio",
        ]
        for column in metric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        recomputed_scores: list[float] = []
        for _, row in frame.iterrows():
            record = {column: row[column] for column in metric_columns}
            base_score = _compute_overall_metric_score_pct(record)
            self.assertIsNotNone(base_score)
            self.assertGreaterEqual(base_score, 0.0)
            self.assertLessEqual(base_score, 100.0)

            toggled_record = dict(record)
            toggled_record["grounded_claim_ratio"] = (
                0.0 if float(row["grounded_claim_ratio"]) > 0 else 1.0
            )
            toggled_score = _compute_overall_metric_score_pct(toggled_record)
            self.assertAlmostEqual(base_score, toggled_score)
            recomputed_scores.append(float(base_score))

        self.assertEqual(len(recomputed_scores), 50)

    def test_artifact_response_uses_compacted_contexts_for_precision(self) -> None:
        frame = pd.read_csv(ARTIFACT_DIR / "qwen_rag_responses.csv")
        row = frame.iloc[0]

        retrieved_chunks = [
            RetrievalResult(
                chunk_id=f"chunk_{index}",
                text=f"Context block {index} for {row['question']}",
                metadata={"company": f"Company {index}", "sheet_name": "Data", "chunk_type": "row_full"},
                dense_score=1.0 - (index * 0.01),
                lexical_score=1.0 - (index * 0.01),
                final_score=1.0 - (index * 0.01),
            )
            for index in range(1, 9)
        ]
        response = ModelResponse(
            run_name="qwen_rag",
            provider="ollama",
            model_name="qwen2.5:14b",
            rag_enabled=True,
            question=str(row["question"]),
            answer=str(row["answer"]),
            latency_seconds=float(row["latency_seconds"]),
            retrieved_chunks=retrieved_chunks,
            prompt_tokens_estimate=100,
            success=bool(row["success"]),
        )

        captured_context_counts: list[int] = []

        def fake_claim_labels(*args, **kwargs):
            claims = kwargs["claims"]
            return {index: "supported" for index in range(1, len(claims) + 1)}

        def fake_context_labels(*args, **kwargs):
            prompt = kwargs["prompt"] if "prompt" in kwargs else args[1]
            ids = re.findall(r"(?m)^(\d+)\.\s", prompt)
            captured_context_counts.append(len(ids))
            return {int(context_id): True for context_id in ids}

        with patch(
            "src.ev_llm_compare.evaluation._dual_judge_score_metric",
            return_value=0.8,
        ), patch(
            "src.ev_llm_compare.evaluation._dual_judge_metric",
            return_value=0.75,
        ), patch(
            "src.ev_llm_compare.evaluation._classify_claims_against_context",
            side_effect=fake_claim_labels,
        ), patch(
            "src.ev_llm_compare.evaluation._llm_judge_context_labels",
            side_effect=fake_context_labels,
        ):
            record = _score_response_metrics(
                response=response,
                reference_answers={str(row["question"]): str(row["reference_answer"])} ,
                judge_client=_DummyJudgeClient(),
                max_retries=1,
                context_result_limit=4,
                context_char_budget=1000,
                compact_context=True,
            )

        self.assertEqual(record["context_precision"], 1.0)
        self.assertEqual(captured_context_counts, [4])

    def test_artifact_supplier_reference_segments_into_many_claims(self) -> None:
        frame = pd.read_csv(ARTIFACT_DIR / "qwen_rag_responses.csv")
        row = frame.loc[frame["question"] == SUPPLIER_LIST_QUESTION].iloc[0]

        claims = _segment_response_units(str(row["reference_answer"]))

        self.assertGreaterEqual(len(claims), 10)
        self.assertEqual(claims[0], "There are 18 Tier 1/2 companies in Georgia.")
        self.assertTrue(any(claim.startswith("F&P Georgia Manufacturing [Tier 1/2]") for claim in claims))
        self.assertTrue(any(claim.startswith("Hyundai MOBIS (Georgia) [Tier 1/2]") for claim in claims))

    def test_artifact_network_question_triggers_broad_compaction_limit(self) -> None:
        frame = pd.read_csv(ARTIFACT_DIR / "qwen_rag_responses.csv")
        question = str(frame.loc[frame["question"] == NETWORK_QUESTION].iloc[0]["question"])
        results = [
            RetrievalResult(
                chunk_id=f"net-{index}",
                text=f"Company: Supplier {index} | Category: Tier 1 | Primary OEMs: Kia Georgia Inc. | Updated Location: County {index}",
                metadata={
                    "chunk_type": "row_full",
                    "company": f"Supplier {index}",
                    "category": "Tier 1",
                    "primary_oems": "Kia Georgia Inc.",
                    "location": f"County {index}",
                    "row_summary": f"Company: Supplier {index} | Category: Tier 1 | Primary OEMs: Kia Georgia Inc. | Updated Location: County {index}",
                    "row_key": f"net-{index}",
                    "source_file": "input.xlsx",
                    "sheet_name": "Data",
                    "row_number": str(index),
                },
                dense_score=1.0 - (index * 0.01),
                lexical_score=1.0 - (index * 0.01),
                final_score=1.0 - (index * 0.01),
            )
            for index in range(1, 8)
        ]

        blocks = compact_context_segments(question, results, max_results=4, max_chars=6000)

        self.assertEqual(len(blocks), 6)


if __name__ == "__main__":
    unittest.main()
