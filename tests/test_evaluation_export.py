from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.ev_llm_compare.evaluation import (
    _derive_grounding_metrics,
    _score_answer_accuracy,
    export_results,
)
from src.ev_llm_compare.schemas import ModelResponse, RetrievalResult


class EvaluationExportTests(unittest.TestCase):
    def test_export_results_writes_pivoted_responses_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            responses = [
                ModelResponse(
                    run_name="qwen_rag",
                    provider="ollama",
                    model_name="qwen2.5:14b",
                    rag_enabled=True,
                    question="What is A?",
                    answer="Answer A with RAG",
                    latency_seconds=1.1,
                    retrieved_chunks=[],
                    prompt_tokens_estimate=111,
                    success=True,
                ),
                ModelResponse(
                    run_name="qwen_no_rag",
                    provider="ollama",
                    model_name="qwen2.5:14b",
                    rag_enabled=False,
                    question="What is A?",
                    answer="Answer A without RAG",
                    latency_seconds=0.9,
                    retrieved_chunks=[],
                    prompt_tokens_estimate=77,
                    success=True,
                ),
                ModelResponse(
                    run_name="qwen_rag",
                    provider="ollama",
                    model_name="qwen2.5:14b",
                    rag_enabled=True,
                    question="What is B?",
                    answer="Answer B with RAG",
                    latency_seconds=1.3,
                    retrieved_chunks=[],
                    prompt_tokens_estimate=120,
                    success=True,
                ),
                ModelResponse(
                    run_name="qwen_no_rag",
                    provider="ollama",
                    model_name="qwen2.5:14b",
                    rag_enabled=False,
                    question="What is B?",
                    answer="Answer B without RAG",
                    latency_seconds=1.0,
                    retrieved_chunks=[],
                    prompt_tokens_estimate=80,
                    success=True,
                ),
            ]
            retrievals = {
                "What is A?": [
                    RetrievalResult(
                        chunk_id="c1",
                        text="Chunk A",
                        metadata={"company": "A Co", "sheet_name": "Data", "chunk_type": "row"},
                        dense_score=0.9,
                        lexical_score=0.8,
                        final_score=0.85,
                    )
                ],
                "What is B?": [],
            }
            references = {"What is A?": "Ref A", "What is B?": "Ref B"}
            reference_sources = {"What is A?": "golden", "What is B?": "generated"}
            ragas_per_run = pd.DataFrame(
                [
                    {
                            "run_name": "qwen_rag",
                            "question": "What is A?",
                            "answer_accuracy": 0.95,
                            "faithfulness": 0.9,
                            "response_groundedness": 0.85,
                            "grounded_claim_ratio": 0.85,
                            "unsupported_claim_ratio": 0.1,
                            "contradicted_claim_ratio": 0.05,
                            "ragas_answer_accuracy": 0.75,
                            "ragas_faithfulness": 0.7,
                            "ragas_response_groundedness": 0.65,
                        },
                        {
                            "run_name": "qwen_no_rag",
                            "question": "What is A?",
                            "answer_accuracy": 0.8,
                            "faithfulness": None,
                            "response_groundedness": None,
                            "grounded_claim_ratio": None,
                            "unsupported_claim_ratio": None,
                            "contradicted_claim_ratio": None,
                            "ragas_answer_accuracy": 0.6,
                            "ragas_faithfulness": None,
                            "ragas_response_groundedness": None,
                        },
                    ]
                )

            workbook_path = export_results(
                output_dir=output_dir,
                responses=responses,
                retrievals=retrievals,
                references=references,
                reference_sources=reference_sources,
                ragas_per_run=ragas_per_run,
                ragas_summary=None,
            )

            df = pd.read_excel(workbook_path, sheet_name="responses")
            self.assertEqual(
                df.columns.tolist(),
                [
                    "Question",
                    "reference_answer",
                    "reference_source",
                    "qwen_rag",
                    "qwen_no_rag",
                    "qwen_rag_answer_accuracy",
                    "qwen_rag_faithfulness",
                    "qwen_rag_response_groundedness",
                    "qwen_rag_grounded_claim_ratio",
                    "qwen_rag_unsupported_claim_ratio",
                    "qwen_rag_contradicted_claim_ratio",
                    "qwen_rag_ragas_answer_accuracy",
                    "qwen_rag_ragas_faithfulness",
                    "qwen_rag_ragas_response_groundedness",
                    "qwen_no_rag_answer_accuracy",
                    "qwen_no_rag_faithfulness",
                    "qwen_no_rag_response_groundedness",
                    "qwen_no_rag_grounded_claim_ratio",
                    "qwen_no_rag_unsupported_claim_ratio",
                    "qwen_no_rag_contradicted_claim_ratio",
                    "qwen_no_rag_ragas_answer_accuracy",
                    "qwen_no_rag_ragas_faithfulness",
                    "qwen_no_rag_ragas_response_groundedness",
                    "qwen_rag_latency_seconds",
                    "qwen_no_rag_latency_seconds",
                    "qwen_rag_prompt_tokens_estimate",
                    "qwen_no_rag_prompt_tokens_estimate",
                ],
            )
            self.assertEqual(df.iloc[0]["Question"], "What is A?")
            self.assertEqual(df.iloc[0]["reference_answer"], "Ref A")
            self.assertEqual(df.iloc[0]["reference_source"], "golden")
            self.assertEqual(df.iloc[0]["qwen_rag"], "Answer A with RAG")
            self.assertEqual(df.iloc[0]["qwen_no_rag"], "Answer A without RAG")
            self.assertAlmostEqual(df.iloc[0]["qwen_rag_answer_accuracy"], 0.95)
            self.assertAlmostEqual(df.iloc[0]["qwen_rag_faithfulness"], 0.9)
            self.assertAlmostEqual(df.iloc[0]["qwen_rag_response_groundedness"], 0.85)
            self.assertAlmostEqual(df.iloc[0]["qwen_rag_grounded_claim_ratio"], 0.85)
            self.assertAlmostEqual(df.iloc[0]["qwen_rag_contradicted_claim_ratio"], 0.05)
            self.assertAlmostEqual(df.iloc[0]["qwen_rag_ragas_answer_accuracy"], 0.75)
            self.assertAlmostEqual(df.iloc[0]["qwen_no_rag_answer_accuracy"], 0.8)
            self.assertTrue(pd.isna(df.iloc[0]["qwen_no_rag_faithfulness"]))
            self.assertTrue(pd.isna(df.iloc[0]["qwen_no_rag_response_groundedness"]))
            self.assertAlmostEqual(df.iloc[0]["qwen_no_rag_ragas_answer_accuracy"], 0.6)
            self.assertAlmostEqual(df.iloc[0]["qwen_rag_latency_seconds"], 1.1)
            self.assertEqual(df.iloc[0]["qwen_no_rag_prompt_tokens_estimate"], 77)

            raw_df = pd.read_excel(workbook_path, sheet_name="responses_raw")
            self.assertIn("run_name", raw_df.columns)

            single_sheet_df = pd.read_excel(workbook_path, sheet_name="all_in_one")
            self.assertEqual(
                single_sheet_df.columns.tolist(),
                [
                    "Question",
                    "reference_answer",
                    "reference_source",
                    "qwen_rag",
                    "qwen_rag_answer_accuracy",
                    "qwen_rag_faithfulness",
                    "qwen_rag_response_groundedness",
                    "qwen_rag_grounded_claim_ratio",
                    "qwen_rag_unsupported_claim_ratio",
                    "qwen_rag_contradicted_claim_ratio",
                    "qwen_rag_ragas_answer_accuracy",
                    "qwen_rag_ragas_faithfulness",
                    "qwen_rag_ragas_response_groundedness",
                    "qwen_no_rag",
                    "qwen_no_rag_answer_accuracy",
                    "qwen_no_rag_faithfulness",
                    "qwen_no_rag_response_groundedness",
                    "qwen_no_rag_grounded_claim_ratio",
                    "qwen_no_rag_unsupported_claim_ratio",
                    "qwen_no_rag_contradicted_claim_ratio",
                    "qwen_no_rag_ragas_answer_accuracy",
                    "qwen_no_rag_ragas_faithfulness",
                    "qwen_no_rag_ragas_response_groundedness",
                ],
            )
            self.assertEqual(single_sheet_df.iloc[0]["reference_answer"], "Ref A")
            self.assertEqual(single_sheet_df.iloc[0]["reference_source"], "golden")
            self.assertEqual(single_sheet_df.iloc[0]["qwen_rag"], "Answer A with RAG")
            self.assertEqual(single_sheet_df.iloc[0]["qwen_no_rag"], "Answer A without RAG")
            self.assertAlmostEqual(single_sheet_df.iloc[0]["qwen_rag_answer_accuracy"], 0.95)
            self.assertAlmostEqual(single_sheet_df.iloc[0]["qwen_rag_faithfulness"], 0.9)
            self.assertAlmostEqual(single_sheet_df.iloc[0]["qwen_rag_response_groundedness"], 0.85)
            self.assertAlmostEqual(single_sheet_df.iloc[0]["qwen_rag_grounded_claim_ratio"], 0.85)
            self.assertAlmostEqual(single_sheet_df.iloc[0]["qwen_rag_ragas_answer_accuracy"], 0.75)
            self.assertAlmostEqual(single_sheet_df.iloc[0]["qwen_no_rag_answer_accuracy"], 0.8)
            self.assertTrue(pd.isna(single_sheet_df.iloc[0]["qwen_no_rag_faithfulness"]))
            self.assertTrue(pd.isna(single_sheet_df.iloc[0]["qwen_no_rag_response_groundedness"]))

    def test_structured_answer_accuracy_penalizes_false_negative_list_answers(self) -> None:
        question = (
            "Which EV Supply Chain Roles have at least one company with EV / Battery Relevant = Yes? "
            "Provide the roles and the matching companies."
        )
        reference_answer = (
            "Battery Cell: Hitachi Astemo Americas Inc.; Honda Development & Manufacturing "
            "Battery Pack: F&P Georgia Manufacturing; Hollingsworth & Vose Co.; Hyundai Motor Group; IMMI"
        )
        answer = (
            "Based on the provided evidence, there are no companies listed with EV / Battery Relevant = Yes."
        )

        score = _score_answer_accuracy(question, answer, reference_answer)
        self.assertEqual(score, 0.0)

    def test_grounding_metrics_flag_contradicted_negative_answer(self) -> None:
        question = (
            "Which EV Supply Chain Roles have at least one company with EV / Battery Relevant = Yes? "
            "Provide the roles and the matching companies."
        )
        answer = "There are no companies listed with EV / Battery Relevant = Yes."
        contexts = [
            "EV Supply Chain Roles with EV / Battery Relevant = Yes companies:\n"
            "- Battery Cell: Hitachi Astemo Americas Inc.; Honda Development & Manufacturing"
        ]

        metrics = _derive_grounding_metrics(question, answer, contexts)
        self.assertEqual(metrics["contradicted_claim_ratio"], 1.0)
        self.assertEqual(metrics["faithfulness"], 0.0)
        self.assertEqual(metrics["response_groundedness"], 0.0)


if __name__ == "__main__":
    unittest.main()
