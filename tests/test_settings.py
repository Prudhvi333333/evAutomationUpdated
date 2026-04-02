import os
import unittest
from unittest.mock import patch

from src.ev_llm_compare.settings import load_config


class SettingsTests(unittest.TestCase):
    def test_load_config_uses_qwen_gemma_and_gemini_runs(self) -> None:
        config = load_config(dotenv_enabled=False)
        self.assertEqual(
            [model.run_name for model in config.models],
            [
                "qwen_rag",
                "qwen_no_rag",
                "gemma_rag",
                "gemma_no_rag",
                "gemini_rag",
                "gemini_no_rag",
            ],
        )

    def test_load_config_reads_evaluation_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "EVALUATION_JUDGE_PROVIDER": "gemini",
                "EVALUATION_JUDGE_MODEL": "gemini-2.5-flash",
                "EVALUATION_MAX_RETRIES": "3",
                "EVALUATION_PARALLELISM": "6",
            },
            clear=False,
        ):
            config = load_config(dotenv_enabled=False)

        self.assertEqual(config.evaluation.judge_provider, "gemini")
        self.assertEqual(config.evaluation.judge_model, "gemini-2.5-flash")
        self.assertEqual(config.evaluation.max_retries, 3)
        self.assertEqual(config.evaluation.parallelism, 6)

    def test_load_config_accepts_legacy_ragas_env_aliases(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RAGAS_JUDGE_PROVIDER": "gemini",
                "RAGAS_JUDGE_MODEL": "gemini-2.5-pro",
                "RAGAS_MAX_RETRIES": "4",
                "RAGAS_PARALLELISM": "5",
            },
            clear=False,
        ):
            config = load_config(dotenv_enabled=False)

        self.assertEqual(config.evaluation.judge_provider, "gemini")
        self.assertEqual(config.evaluation.judge_model, "gemini-2.5-pro")
        self.assertEqual(config.evaluation.max_retries, 4)
        self.assertEqual(config.evaluation.parallelism, 5)

    def test_load_config_defaults_parallelism_by_provider(self) -> None:
        with patch.dict(
            os.environ,
            {
                "EVALUATION_JUDGE_PROVIDER": "gemini",
            },
            clear=True,
        ):
            gemini_config = load_config(dotenv_enabled=False)
        self.assertEqual(gemini_config.evaluation.parallelism, 4)

        with patch.dict(
            os.environ,
            {
                "EVALUATION_JUDGE_PROVIDER": "ollama",
            },
            clear=True,
        ):
            ollama_config = load_config(dotenv_enabled=False)
        self.assertEqual(ollama_config.evaluation.parallelism, 1)

    def test_load_config_reads_retrieval_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RERANKER_ENABLED": "false",
                "RERANKER_TOP_K": "6",
                "RERANKER_WEIGHT": "0.5",
                "MAX_CHUNKS_PER_COMPANY": "1",
                "STRUCTURED_SUMMARY_LIMIT": "5",
                "EVALUATION_CONTEXT_RESULT_LIMIT": "6",
            },
            clear=False,
        ):
            config = load_config(dotenv_enabled=False)

        self.assertFalse(config.retrieval.reranker_enabled)
        self.assertEqual(config.retrieval.reranker_top_k, 6)
        self.assertAlmostEqual(config.retrieval.reranker_weight, 0.5)
        self.assertEqual(config.retrieval.max_chunks_per_company, 1)
        self.assertEqual(config.retrieval.structured_summary_limit, 5)
        self.assertEqual(config.retrieval.evaluation_context_result_limit, 6)


if __name__ == "__main__":
    unittest.main()
