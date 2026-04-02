from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import pandas as pd

from src.ev_llm_compare.evaluation import (
    LLM_JSON_SYSTEM_PROMPT,
    LLM_JUDGE_SCORE_SYSTEM_PROMPT,
    _classify_claims_against_context,
    _compute_overall_metric_score_pct,
    _evaluate_context_precision,
    _parse_llm_judge_score,
    attribute_response_sources,
    export_results,
    export_single_model_report,
    run_evaluation_metrics,
)
from src.ev_llm_compare.models import LLMClient
from src.ev_llm_compare.schemas import ModelResponse, RetrievalResult


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
        raise AssertionError("safe_generate should be patched in this test")


class EvaluationExportTests(unittest.TestCase):
    def test_export_results_writes_current_metrics_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            responses = [
                ModelResponse(
                    run_name="qwen_rag",
                    provider="ollama",
                    model_name="qwen3:8b",
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
                    model_name="qwen3:8b",
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
                    model_name="qwen3:8b",
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
                    model_name="qwen3:8b",
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
            metrics_per_run = pd.DataFrame(
                [
                    {
                        "run_name": "qwen_rag",
                        "question": "What is A?",
                        "answer_accuracy": 0.95,
                        "faithfulness": 0.9,
                        "response_groundedness": 0.85,
                        "context_precision": 0.8,
                        "context_recall": 0.75,
                        "grounded_claim_ratio": 0.85,
                        "unsupported_claim_ratio": 0.1,
                        "contradicted_claim_ratio": 0.05,
                        "overall_metric_score_pct": 87.6,
                    },
                    {
                        "run_name": "qwen_no_rag",
                        "question": "What is A?",
                        "answer_accuracy": 0.8,
                        "faithfulness": None,
                        "response_groundedness": None,
                        "context_precision": None,
                        "context_recall": None,
                        "grounded_claim_ratio": None,
                        "unsupported_claim_ratio": None,
                        "contradicted_claim_ratio": None,
                        "overall_metric_score_pct": 80.0,
                    },
                ]
            )

            workbook_path = export_results(
                output_dir=output_dir,
                responses=responses,
                retrievals=retrievals,
                references=references,
                reference_sources=reference_sources,
                metrics_per_run=metrics_per_run,
                metrics_summary=None,
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
                    "qwen_rag_context_precision",
                    "qwen_rag_context_recall",
                    "qwen_rag_grounded_claim_ratio",
                    "qwen_rag_unsupported_claim_ratio",
                    "qwen_rag_contradicted_claim_ratio",
                    "qwen_rag_overall_metric_score_pct",
                    "qwen_no_rag_answer_accuracy",
                    "qwen_no_rag_faithfulness",
                    "qwen_no_rag_response_groundedness",
                    "qwen_no_rag_context_precision",
                    "qwen_no_rag_context_recall",
                    "qwen_no_rag_grounded_claim_ratio",
                    "qwen_no_rag_unsupported_claim_ratio",
                    "qwen_no_rag_contradicted_claim_ratio",
                    "qwen_no_rag_overall_metric_score_pct",
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
            self.assertAlmostEqual(df.iloc[0]["qwen_rag_context_precision"], 0.8)
            self.assertAlmostEqual(df.iloc[0]["qwen_rag_grounded_claim_ratio"], 0.85)
            self.assertAlmostEqual(df.iloc[0]["qwen_rag_overall_metric_score_pct"], 87.6)
            self.assertAlmostEqual(df.iloc[0]["qwen_no_rag_answer_accuracy"], 0.8)
            self.assertTrue(pd.isna(df.iloc[0]["qwen_no_rag_faithfulness"]))
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
                    "qwen_rag_context_precision",
                    "qwen_rag_context_recall",
                    "qwen_rag_grounded_claim_ratio",
                    "qwen_rag_unsupported_claim_ratio",
                    "qwen_rag_contradicted_claim_ratio",
                    "qwen_rag_overall_metric_score_pct",
                    "qwen_no_rag",
                    "qwen_no_rag_answer_accuracy",
                    "qwen_no_rag_faithfulness",
                    "qwen_no_rag_response_groundedness",
                    "qwen_no_rag_context_precision",
                    "qwen_no_rag_context_recall",
                    "qwen_no_rag_grounded_claim_ratio",
                    "qwen_no_rag_unsupported_claim_ratio",
                    "qwen_no_rag_contradicted_claim_ratio",
                    "qwen_no_rag_overall_metric_score_pct",
                ],
            )

    def test_run_evaluation_metrics_uses_retry_budget_and_scores_rag_metrics(self) -> None:
        response = ModelResponse(
            run_name="qwen_rag",
            provider="ollama",
            model_name="qwen3:8b",
            rag_enabled=True,
            question="What is A?",
            answer="A is supported by the evidence.",
            latency_seconds=0.8,
            retrieved_chunks=[
                RetrievalResult(
                    chunk_id="c1",
                    text="Company: A | Category: Tier 1",
                    metadata={"company": "A", "sheet_name": "Data", "chunk_type": "row_full"},
                    dense_score=0.9,
                    lexical_score=0.8,
                    final_score=0.85,
                )
            ],
            prompt_tokens_estimate=42,
            success=True,
        )

        answers = iter(
            [
                ("not-a-score", 0.01, True, None),
                ("SCORE=0.90", 0.01, True, None),
                ("SCORE=0.70", 0.01, True, None),
                ('{"labels":[{"claim_id":1,"label":"supported"}]}', 0.01, True, None),
                ("RATING=2", 0.01, True, None),
                ("RATING=1", 0.01, True, None),
                ('{"labels":[{"context_id":1,"label":"relevant"}]}', 0.01, True, None),
                ('{"labels":[{"claim_id":1,"label":"supported"}]}', 0.01, True, None),
            ]
        )

        with patch(
            "src.ev_llm_compare.evaluation._make_judge_client",
            return_value=_DummyJudgeClient(),
        ), patch(
            "src.ev_llm_compare.evaluation.safe_generate",
            side_effect=lambda *args, **kwargs: next(answers),
        ) as mocked_generate:
            metrics_per_run, metrics_summary = run_evaluation_metrics(
                responses=[response],
                reference_answers={"What is A?": "Reference A"},
                judge_provider="ollama",
                judge_model="judge-model",
                max_retries=1,
                context_result_limit=4,
                context_char_budget=1000,
            )

        self.assertEqual(mocked_generate.call_count, 8)
        self.assertAlmostEqual(metrics_per_run.iloc[0]["answer_accuracy"], 0.8)
        self.assertAlmostEqual(metrics_per_run.iloc[0]["faithfulness"], 1.0)
        self.assertAlmostEqual(metrics_per_run.iloc[0]["response_groundedness"], 0.75)
        self.assertAlmostEqual(metrics_per_run.iloc[0]["context_precision"], 1.0)
        self.assertAlmostEqual(metrics_per_run.iloc[0]["context_recall"], 1.0)
        self.assertAlmostEqual(metrics_per_run.iloc[0]["contradicted_claim_ratio"], 0.0)
        self.assertAlmostEqual(metrics_per_run.iloc[0]["overall_metric_score_pct"], 90.0)
        self.assertAlmostEqual(metrics_summary.iloc[0]["answer_accuracy"], 0.8)

    def test_run_evaluation_metrics_parallelism_preserves_input_order(self) -> None:
        responses = [
            ModelResponse(
                run_name="qwen_rag",
                provider="ollama",
                model_name="qwen3:8b",
                rag_enabled=False,
                question="Question 1",
                answer="Answer 1",
                latency_seconds=0.1,
                retrieved_chunks=[],
                prompt_tokens_estimate=10,
                success=True,
            ),
            ModelResponse(
                run_name="qwen_rag",
                provider="ollama",
                model_name="qwen3:8b",
                rag_enabled=False,
                question="Question 2",
                answer="Answer 2",
                latency_seconds=0.1,
                retrieved_chunks=[],
                prompt_tokens_estimate=10,
                success=True,
            ),
        ]
        progress_events: list[tuple[int, int, str]] = []

        def fake_score(
            *,
            response: ModelResponse,
            reference_answers: dict[str, str],
            judge_client: LLMClient,
            max_retries: int,
            context_result_limit: int,
            context_char_budget: int,
            compact_context: bool,
        ) -> dict[str, object]:
            if response.question == "Question 1":
                time.sleep(0.05)
            return {
                "run_name": response.run_name,
                "question": response.question,
                "answer_accuracy": 0.9 if response.question == "Question 1" else 0.7,
                "faithfulness": None,
                "response_groundedness": None,
                "context_precision": None,
                "context_recall": None,
                "grounded_claim_ratio": None,
                "unsupported_claim_ratio": None,
                "contradicted_claim_ratio": None,
                "overall_metric_score_pct": 90.0 if response.question == "Question 1" else 70.0,
            }

        with patch(
            "src.ev_llm_compare.evaluation._make_judge_client",
            return_value=_DummyJudgeClient(),
        ), patch(
            "src.ev_llm_compare.evaluation._score_response_metrics",
            side_effect=fake_score,
        ):
            metrics_per_run, _ = run_evaluation_metrics(
                responses=responses,
                reference_answers={"Question 1": "Ref 1", "Question 2": "Ref 2"},
                judge_provider="gemini",
                judge_model="judge-model",
                parallelism=2,
                progress_callback=lambda completed, total, response: progress_events.append(
                    (completed, total, response.question)
                ),
            )

        self.assertEqual(metrics_per_run["question"].tolist(), ["Question 1", "Question 2"])
        self.assertEqual(metrics_per_run["answer_accuracy"].tolist(), [0.9, 0.7])
        self.assertEqual(len(progress_events), 2)
        self.assertEqual(progress_events[-1][0], 2)

    def test_partial_claim_labels_only_default_missing_claims(self) -> None:
        with patch(
            "src.ev_llm_compare.evaluation._llm_judge_claim_labels",
            return_value={1: "supported"},
        ):
            labels = _classify_claims_against_context(
                judge_client=_DummyJudgeClient(),
                question="What is A?",
                retrieved_contexts=["A is in Georgia.", "B is in Alabama."],
                claims=["A is in Georgia.", "B is in Georgia."],
                retries=1,
            )

        self.assertEqual(labels[1], "supported")
        self.assertEqual(labels[2], "unsupported")

    def test_partial_context_labels_only_default_missing_contexts(self) -> None:
        chunks = [
            RetrievalResult(
                chunk_id="c1",
                text="A is in Georgia.",
                metadata={"company": "A", "sheet_name": "Data", "chunk_type": "row_full"},
                dense_score=0.9,
                lexical_score=0.8,
                final_score=0.85,
            ),
            RetrievalResult(
                chunk_id="c2",
                text="Unrelated filler.",
                metadata={"company": "B", "sheet_name": "Data", "chunk_type": "row_full"},
                dense_score=0.7,
                lexical_score=0.5,
                final_score=0.6,
            ),
        ]

        with patch(
            "src.ev_llm_compare.evaluation._llm_judge_context_labels",
            return_value={1: True},
        ):
            precision = _evaluate_context_precision(
                judge_client=_DummyJudgeClient(),
                question="What is A?",
                reference_answer="A is in Georgia.",
                retrieved_contexts=[chunk.text for chunk in chunks],
                retries=1,
            )

        self.assertAlmostEqual(precision, 1.0)

    def test_compute_overall_metric_score_pct_matches_weighted_formula(self) -> None:
        record = {
            "answer_accuracy": 0.8,
            "faithfulness": 0.7,
            "response_groundedness": 0.75,
            "context_precision": 0.6,
            "context_recall": 0.5,
            "grounded_claim_ratio": 0.0,
            "unsupported_claim_ratio": 0.2,
            "contradicted_claim_ratio": 0.1,
        }

        score = _compute_overall_metric_score_pct(record)

        expected = 100.0 * (
            0.28 * 0.8
            + 0.20 * 0.7
            + 0.16 * 0.75
            + 0.12 * 0.6
            + 0.12 * 0.5
            + 0.04 * (1.0 - 0.2)
            + 0.04 * (1.0 - 0.1)
        ) / 0.96
        self.assertAlmostEqual(score, expected)

    def test_compute_overall_metric_score_pct_normalizes_missing_metrics(self) -> None:
        record = {
            "answer_accuracy": 0.8,
            "faithfulness": None,
            "response_groundedness": None,
            "context_precision": None,
            "context_recall": None,
            "grounded_claim_ratio": 1.0,
            "unsupported_claim_ratio": None,
            "contradicted_claim_ratio": None,
        }

        score = _compute_overall_metric_score_pct(record)

        self.assertAlmostEqual(score, 80.0)

    def test_parse_llm_judge_score_accepts_strict_score_output(self) -> None:
        self.assertEqual(_parse_llm_judge_score("SCORE=0.83"), 0.83)
        self.assertEqual(_parse_llm_judge_score("0.40"), 0.4)
        self.assertIsNone(_parse_llm_judge_score("SCORE=1.20"))

    def test_system_prompts_require_conservative_no_world_knowledge_judging(self) -> None:
        self.assertIn("Do not use world knowledge", LLM_JUDGE_SCORE_SYSTEM_PROMPT)
        self.assertIn("choose the lower score", LLM_JUDGE_SCORE_SYSTEM_PROMPT)
        self.assertIn("Do not use world knowledge", LLM_JSON_SYSTEM_PROMPT)
        self.assertIn("prefer the lower-support label", LLM_JSON_SYSTEM_PROMPT)

    def test_attribute_response_sources_treats_no_rag_response_as_pretrained(self) -> None:
        response = ModelResponse(
            run_name="qwen_no_rag",
            provider="ollama",
            model_name="qwen3:8b",
            rag_enabled=False,
            question="What is A?",
            answer="A is a battery supplier in Georgia.",
            latency_seconds=0.5,
            retrieved_chunks=[],
            prompt_tokens_estimate=25,
            success=True,
        )

        attribution = attribute_response_sources(
            response,
            judge_client=None,
        )

        self.assertEqual(attribution["knowledge_source_data"], "")
        self.assertEqual(attribution["pretrained_data"], "A is a battery supplier in Georgia.")
        self.assertEqual(attribution["overall_response"], "A is a battery supplier in Georgia.")

    def test_attribute_response_sources_keeps_partial_attribution_labels(self) -> None:
        response = ModelResponse(
            run_name="qwen_rag",
            provider="ollama",
            model_name="qwen3:8b",
            rag_enabled=True,
            question="What is A?",
            answer="A is in Georgia. It is a major battery innovator.",
            latency_seconds=0.5,
            retrieved_chunks=[
                RetrievalResult(
                    chunk_id="c1",
                    text="A is in Georgia.",
                    metadata={"company": "A", "sheet_name": "Data", "chunk_type": "row_full"},
                    dense_score=0.9,
                    lexical_score=0.8,
                    final_score=0.85,
                )
            ],
            prompt_tokens_estimate=25,
            success=True,
        )

        with patch(
            "src.ev_llm_compare.evaluation._llm_judge_attribution",
            return_value={1: "knowledge_source"},
        ):
            attribution = attribute_response_sources(
                response,
                judge_client=_DummyJudgeClient(),
                context_result_limit=4,
                context_char_budget=1000,
                max_retries=1,
            )

        self.assertEqual(attribution["knowledge_source_data"], "A is in Georgia.")
        self.assertEqual(attribution["pretrained_data"], "It is a major battery innovator.")

    def test_export_single_model_report_writes_attribution_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            response = ModelResponse(
                run_name="qwen_rag",
                provider="ollama",
                model_name="qwen3:8b",
                rag_enabled=True,
                question="What is A?",
                answer="Company A is in Atlanta. It is a major battery innovator.",
                latency_seconds=0.8,
                retrieved_chunks=[
                    RetrievalResult(
                        chunk_id="c1",
                        text="Company: A | Location: Atlanta",
                        metadata={"company": "A", "sheet_name": "Data", "chunk_type": "row_full"},
                        dense_score=0.9,
                        lexical_score=0.8,
                        final_score=0.85,
                    )
                ],
                prompt_tokens_estimate=42,
                success=True,
            )
            metrics_per_run = pd.DataFrame(
                [
                    {
                        "run_name": "qwen_rag",
                        "question": "What is A?",
                        "answer_accuracy": 0.85,
                        "faithfulness": 0.9,
                        "response_groundedness": 0.5,
                        "context_precision": 0.7,
                        "context_recall": 0.6,
                        "grounded_claim_ratio": 0.5,
                        "unsupported_claim_ratio": 0.5,
                        "contradicted_claim_ratio": 0.0,
                        "overall_metric_score_pct": 73.4,
                    }
                ]
            )

            with patch(
                "src.ev_llm_compare.evaluation._make_judge_client",
                return_value=_DummyJudgeClient(),
            ), patch(
                "src.ev_llm_compare.evaluation.safe_generate",
                return_value=(
                    '{"labels":[{"unit_id":1,"label":"knowledge_source"},{"unit_id":2,"label":"pretrained"}]}',
                    0.01,
                    True,
                    None,
                ),
            ):
                workbook_path = export_single_model_report(
                    output_dir=Path(tmp_dir),
                    responses=[response],
                    references={"What is A?": "Ref A"},
                    reference_sources={"What is A?": "golden"},
                    judge_provider="ollama",
                    judge_model="judge-model",
                    metrics_per_run=metrics_per_run,
                )

            report_df = pd.read_excel(workbook_path, sheet_name="report")
            self.assertEqual(
                report_df.columns.tolist(),
                [
                    "Question",
                    "reference_answer",
                    "reference_source",
                    "overall_response",
                    "knowledge_source_data",
                    "pretrained_data",
                    "answer_accuracy",
                    "faithfulness",
                    "response_groundedness",
                    "context_precision",
                    "context_recall",
                    "grounded_claim_ratio",
                    "unsupported_claim_ratio",
                    "contradicted_claim_ratio",
                    "overall_metric_score_pct",
                ],
            )
            self.assertEqual(report_df.iloc[0]["knowledge_source_data"], "Company A is in Atlanta.")
            self.assertEqual(report_df.iloc[0]["pretrained_data"], "It is a major battery innovator.")
            self.assertEqual(
                report_df.iloc[0]["overall_response"],
                "Company A is in Atlanta. It is a major battery innovator.",
            )

            attribution_df = pd.read_excel(workbook_path, sheet_name="attribution_units")
            self.assertEqual(attribution_df.iloc[0]["label"], "knowledge_source")
            self.assertEqual(attribution_df.iloc[1]["label"], "pretrained")


if __name__ == "__main__":
    unittest.main()
