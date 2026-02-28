from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from .chunking import ExcelChunkBuilder
from .evaluation import build_reference_answers, export_results, run_ragas
from .excel_loader import load_questions, load_workbook
from .models import create_client, safe_generate
from .prompts import build_non_rag_prompt, build_rag_prompt, format_context
from .retrieval import HybridRetriever
from .schemas import ModelResponse
from .settings import AppConfig, ModelSpec


class ComparisonRunner:
    def __init__(self, config: AppConfig):
        self.config = config
        if config.runtime.dotenv_enabled:
            load_dotenv()

    def run(
        self,
        data_workbook: str,
        question_workbook: str,
        question_sheet: str | None = None,
        skip_ragas: bool = False,
        question_limit: int | None = None,
    ) -> Path:
        self._log(f"Loading data workbook: {data_workbook}")
        rows, notes = load_workbook(data_workbook)
        self._log(f"Loaded workbook content: {len(rows)} tabular rows, {len(notes)} note sheets")
        self._log(f"Loading question workbook: {question_workbook}")
        questions = load_questions(question_workbook, sheet_name=question_sheet)
        if question_limit is not None:
            questions = questions[:question_limit]
        self._log(f"Loaded {len(questions)} questions")

        self._log("Building structured chunks")
        chunk_builder = ExcelChunkBuilder(self.config.retrieval)
        chunks = chunk_builder.build(rows, notes)
        self._log(f"Built {len(chunks)} chunks")
        self._log("Initializing retriever and indexing chunks")
        retriever = HybridRetriever(
            chunks=chunks,
            settings=self.config.retrieval,
            qdrant_path=self.config.runtime.qdrant_path,
        )
        try:
            self._log("Running retrieval for all questions")
            retrievals = {question: retriever.retrieve(question) for question in questions}
            self._log("Retrieval complete")
            responses: list[ModelResponse] = []

            for model_index, spec in enumerate(self.config.models, start=1):
                self._log(
                    f"Running model {model_index}/{len(self.config.models)}: "
                    f"{spec.run_name} ({spec.model_name}, rag={spec.rag_enabled})"
                )
                client = create_client(spec, self.config.runtime)
                for question_index, question in enumerate(questions, start=1):
                    self._log(
                        f"  Question {question_index}/{len(questions)} for {spec.run_name}"
                    )
                    question_retrieval = retrievals[question]
                    prompt = (
                        build_rag_prompt(question, format_context(question_retrieval))
                        if spec.rag_enabled
                        else build_non_rag_prompt(question)
                    )
                    answer, latency, success, error = safe_generate(
                        client,
                        prompt,
                        temperature=spec.temperature,
                        max_tokens=spec.max_tokens,
                    )
                    responses.append(
                        ModelResponse(
                            run_name=spec.run_name,
                            provider=spec.provider,
                            model_name=spec.model_name,
                            rag_enabled=spec.rag_enabled,
                            question=question,
                            answer=answer,
                            latency_seconds=latency,
                            retrieved_chunks=question_retrieval,
                            prompt_tokens_estimate=max(1, len(prompt) // 4),
                            success=success,
                            error_message=error,
                        )
                    )

            ragas_per_run = None
            ragas_summary = None
            if skip_ragas:
                self._log("Skipping reference generation and RAGAS evaluation")
                references = {question: "" for question in questions}
            else:
                self._log("Generating reference answers for evaluation")
                judge_spec = ModelSpec(
                    run_name="ragas_judge",
                    provider=self.config.ragas_judge_provider,
                    model_name=self.config.ragas_judge_model,
                    rag_enabled=False,
                )
                judge_client = create_client(judge_spec, self.config.runtime)
                references = build_reference_answers(questions, retrievals, judge_client)
                self._log("Running RAGAS evaluation")
                ragas_per_run, ragas_summary = run_ragas(
                    responses=responses,
                    reference_answers=references,
                    judge_provider=self.config.ragas_judge_provider,
                    judge_model=self.config.ragas_judge_model,
                    embedding_provider=self.config.ragas_embedding_provider,
                    embedding_model=self.config.ragas_embedding_model,
                )

            self._log("Exporting results workbook")
            output_path = export_results(
                output_dir=self.config.runtime.output_dir,
                responses=responses,
                retrievals=retrievals,
                references=references,
                ragas_per_run=ragas_per_run,
                ragas_summary=ragas_summary,
            )
            self._log(f"Done. Report written to {output_path}")
            return output_path
        finally:
            retriever.close()

    def _log(self, message: str) -> None:
        print(f"[ev-llm-compare] {message}", flush=True)
