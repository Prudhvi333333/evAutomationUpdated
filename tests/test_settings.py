import os
import unittest
from unittest.mock import patch

from src.ev_llm_compare.settings import load_config


class SettingsTests(unittest.TestCase):
    def test_load_config_uses_qwen_gemma_and_gemini_runs(self) -> None:
        config = load_config()
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

    def test_load_config_reads_ragas_runtime_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RAGAS_TIMEOUT": "900",
                "RAGAS_MAX_RETRIES": "3",
                "RAGAS_MAX_WAIT": "45",
                "RAGAS_MAX_WORKERS": "2",
            },
            clear=False,
        ):
            config = load_config()

        self.assertEqual(config.ragas_timeout, 900)
        self.assertEqual(config.ragas_max_retries, 3)
        self.assertEqual(config.ragas_max_wait, 45)
        self.assertEqual(config.ragas_max_workers, 2)

    def test_load_config_reads_retrieval_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RERANKER_ENABLED": "false",
                "RERANKER_TOP_K": "6",
                "RERANKER_WEIGHT": "0.5",
                "MAX_CHUNKS_PER_COMPANY": "1",
                "STRUCTURED_SUMMARY_LIMIT": "5",
            },
            clear=False,
        ):
            config = load_config()

        self.assertFalse(config.retrieval.reranker_enabled)
        self.assertEqual(config.retrieval.reranker_top_k, 6)
        self.assertAlmostEqual(config.retrieval.reranker_weight, 0.5)
        self.assertEqual(config.retrieval.max_chunks_per_company, 1)
        self.assertEqual(config.retrieval.structured_summary_limit, 5)


if __name__ == "__main__":
    unittest.main()
