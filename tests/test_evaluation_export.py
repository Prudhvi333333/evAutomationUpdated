from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.ev_llm_compare.evaluation import export_results
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
            ragas_per_run = pd.DataFrame(
                [
                    {
                        "run_name": "qwen_rag",
                        "question": "What is A?",
                        "faithfulness": 0.9,
                    },
                    {
                        "run_name": "qwen_no_rag",
                        "question": "What is A?",
                        "faithfulness": 0.7,
                    },
                ]
            )

            workbook_path = export_results(
                output_dir=output_dir,
                responses=responses,
                retrievals=retrievals,
                references=references,
                ragas_per_run=ragas_per_run,
                ragas_summary=None,
            )

            df = pd.read_excel(workbook_path, sheet_name="responses")
            self.assertEqual(
                df.columns.tolist(),
                [
                    "Question",
                    "qwen_rag",
                    "qwen_no_rag",
                    "qwen_rag_faithfulness",
                    "qwen_no_rag_faithfulness",
                    "qwen_rag_latency_seconds",
                    "qwen_no_rag_latency_seconds",
                    "qwen_rag_prompt_tokens_estimate",
                    "qwen_no_rag_prompt_tokens_estimate",
                ],
            )
            self.assertEqual(df.iloc[0]["Question"], "What is A?")
            self.assertEqual(df.iloc[0]["qwen_rag"], "Answer A with RAG")
            self.assertEqual(df.iloc[0]["qwen_no_rag"], "Answer A without RAG")
            self.assertAlmostEqual(df.iloc[0]["qwen_rag_faithfulness"], 0.9)
            self.assertAlmostEqual(df.iloc[0]["qwen_no_rag_faithfulness"], 0.7)
            self.assertAlmostEqual(df.iloc[0]["qwen_rag_latency_seconds"], 1.1)
            self.assertEqual(df.iloc[0]["qwen_no_rag_prompt_tokens_estimate"], 77)

            raw_df = pd.read_excel(workbook_path, sheet_name="responses_raw")
            self.assertIn("run_name", raw_df.columns)

            single_sheet_df = pd.read_excel(workbook_path, sheet_name="all_in_one")
            self.assertEqual(
                single_sheet_df.columns.tolist(),
                [
                    "Question",
                    "qwen_rag",
                    "qwen_rag_scores",
                    "qwen_no_rag",
                    "qwen_no_rag_scores",
                ],
            )
            self.assertEqual(single_sheet_df.iloc[0]["qwen_rag"], "Answer A with RAG")
            self.assertEqual(single_sheet_df.iloc[0]["qwen_no_rag"], "Answer A without RAG")
            self.assertEqual(
                single_sheet_df.iloc[0]["qwen_rag_scores"],
                "faithfulness=0.9000",
            )
            self.assertEqual(
                single_sheet_df.iloc[0]["qwen_no_rag_scores"],
                "faithfulness=0.7000",
            )


if __name__ == "__main__":
    unittest.main()
