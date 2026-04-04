from __future__ import annotations

import json
from pathlib import Path

from .chunking import ExcelChunkBuilder
from .derived_analytics import build_derived_summary_chunks
from .evaluation import (
    build_reference_answers,
    export_metrics_workbook,
    export_response_sets,
    export_results,
    export_single_model_report,
    run_evaluation_metrics,
)
from .excel_loader import (
    join_golden_answers,
    load_eval_questions,
    load_golden_answers,
    load_questions,
    load_reference_answers,
    load_workbook,
)
from .models import create_client, safe_generate
from .prompts import (
    NON_RAG_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_non_rag_prompt,
    build_rag_prompt,
    build_two_pass_answer,
    format_context,
    _detect_answer_mode,
)
from .retrieval import HybridRetriever
from .schemas import ModelResponse
from .settings import AppConfig, ModelSpec


class ComparisonRunner:
    def __init__(self, config: AppConfig):
        self.config = config

    def run(
        self,
        data_workbook: str,
        question_workbook: str,
        question_sheet: str | None = None,
        skip_evaluation: bool = False,
        question_limit: int | None = None,
        selected_run_names: list[str] | None = None,
        output_dir: str | None = None,
        response_output_dir: str | None = None,
        single_sheet_only: bool = False,
        export_response_files: bool = True,
        golden_workbook: str | None = None,
        golden_sheet: str | None = None,
        write_checkpoint: bool = False,
        single_model_report: bool = False,
        no_ragas: bool = False,
        resume_from: str | None = None,
        **legacy_kwargs: object,
    ) -> Path:
        if "skip_ragas" in legacy_kwargs:
            skip_evaluation = skip_evaluation or bool(legacy_kwargs.pop("skip_ragas"))
        if legacy_kwargs:
            unknown = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")

        active_models = self._select_models(selected_run_names)
        if output_dir is not None:
            self.config.runtime.output_dir = Path(output_dir)
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
        derived_chunks = build_derived_summary_chunks(rows)
        if derived_chunks:
            chunks.extend(derived_chunks)
            self._log(f"Added {len(derived_chunks)} derived analytic summary chunks")
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

            if resume_from is not None:
                # ── Resume from JSONL checkpoint — skip generation ────────────
                jsonl_path = Path(resume_from).expanduser().resolve()
                if not jsonl_path.exists():
                    raise FileNotFoundError(
                        f"Checkpoint file not found: {jsonl_path}\n"
                        "Run without --resume first to generate responses."
                    )
                self._log(f"Resuming from checkpoint: {jsonl_path}")
                responses = self._load_responses_from_jsonl(jsonl_path, retrievals)
                self._log(
                    f"Loaded {len(responses)} responses from checkpoint "
                    f"({len({r.run_name for r in responses})} run(s))"
                )
            else:
                # ── Normal generation path ────────────────────────────────────
                for model_index, spec in enumerate(active_models, start=1):
                    self._log(
                        f"Running model {model_index}/{len(active_models)}: "
                        f"{spec.run_name} ({spec.model_name}, rag={spec.rag_enabled})"
                    )
                    client = create_client(spec, self.config.runtime)
                    for question_index, question in enumerate(questions, start=1):
                        self._log(
                            f"  Question {question_index}/{len(questions)} for {spec.run_name}"
                        )
                        question_retrieval = retrievals[question]
                        if spec.rag_enabled:
                            answer_mode = _detect_answer_mode(question)
                            # Pass retrieved_chunks so the deterministic renderer
                            # can compute counts/groups in code for structured questions.
                            # format_context() is still called to provide the freeform
                            # fallback context string (used if chunks are empty).
                            fallback_context = format_context(
                                question_retrieval,
                                question=question,
                                max_results=self.config.retrieval.generation_context_result_limit,
                                max_chars=self.config.retrieval.generation_context_char_budget,
                                compact=self.config.retrieval.compact_context_enabled,
                            )
                            prompt = build_rag_prompt(
                                question,
                                fallback_context,
                                retrieved_chunks=question_retrieval if question_retrieval else None,
                            )
                            # Structured list and comparison: temperature=0
                            # — the answer is deterministic from workbook data
                            effective_temperature = (
                                0.0
                                if answer_mode in {"structured_list", "comparison"}
                                else spec.temperature
                            )
                        else:
                            prompt = build_non_rag_prompt(question)
                            effective_temperature = spec.temperature
                            answer_mode = "freeform"
                        if spec.rag_enabled and answer_mode in {"structured_list", "comparison"}:
                            # 2-pass: Pass 1 extracts strict JSON, Pass 2 verbalizes it.
                            # The LLM runs on every call but does constrained extraction
                            # rather than noisy chunk selection.
                            _seed = spec.seed
                            _max_tokens = spec.max_tokens

                            def generate_fn(
                                _prompt: str,
                                system_prompt: str,
                                temperature: float,
                                max_tokens: int,
                                _s=_seed,
                                _c=client,
                            ):
                                return safe_generate(
                                    _c,
                                    _prompt,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    system_prompt=system_prompt,
                                    seed=_s,
                                )

                            answer, latency, success, error = build_two_pass_answer(
                                question=question,
                                context=fallback_context,
                                retrieved_chunks=question_retrieval or [],
                                generate_fn=generate_fn,
                                mode=answer_mode,
                            )
                        else:
                            answer, latency, success, error = safe_generate(
                                client,
                                prompt,
                                temperature=effective_temperature,
                                max_tokens=spec.max_tokens,
                                system_prompt=RAG_SYSTEM_PROMPT if spec.rag_enabled else NON_RAG_SYSTEM_PROMPT,
                                seed=spec.seed,
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
                                retrieved_chunks=question_retrieval if spec.rag_enabled else [],
                                prompt_tokens_estimate=max(1, len(prompt) // 4),
                                success=success,
                                error_message=error,
                            )
                        )

                # ── Save generation checkpoint as JSONL ───────────────────────
                self.config.runtime.output_dir.mkdir(parents=True, exist_ok=True)
                jsonl_path = self.config.runtime.output_dir / "responses_raw.jsonl"
                with jsonl_path.open("w", encoding="utf-8") as fh:
                    for resp in responses:
                        fh.write(json.dumps({
                            "run_name": resp.run_name,
                            "model_name": resp.model_name,
                            "rag_enabled": resp.rag_enabled,
                            "question": resp.question,
                            "answer": resp.answer,
                            "success": resp.success,
                            "latency_seconds": resp.latency_seconds,
                            "error_message": resp.error_message,
                        }, ensure_ascii=False) + "\n")
                self._log(f"Generation checkpoint saved to {jsonl_path}")

            metrics_per_run = None
            metrics_summary = None
            references = {question: "" for question in questions}
            reference_sources = {question: "missing" for question in questions}
            golden_path = self._resolve_reference_workbook(golden_workbook)
            if golden_path is not None:
                self._log(f"Loading golden answers workbook: {golden_path}")
                # Use ID-based loading for robust matching
                golden_by_id = load_golden_answers(golden_path, sheet_name=golden_sheet)
                # Build EvalQuestion list so we can use join_golden_answers
                eval_qs = load_eval_questions(question_workbook, sheet_name=question_sheet)
                if question_limit is not None:
                    eval_qs = eval_qs[:question_limit]
                enriched_qs, unmatched = join_golden_answers(eval_qs, golden_by_id)
                matched_count = 0
                for eq in enriched_qs:
                    if eq.golden_answer:
                        references[eq.question] = eq.golden_answer
                        reference_sources[eq.question] = "golden"
                        matched_count += 1
                self._log(
                    f"Matched {matched_count}/{len(questions)} questions to golden answers "
                    f"(unmatched IDs: {unmatched or 'none'})"
                )
                if unmatched:
                    self._log(
                        f"WARNING: {len(unmatched)} question(s) had no golden answer match: "
                        f"{unmatched}"
                    )
            if skip_evaluation:
                self._log("Skipping reference generation and evaluation metrics")
            else:
                try:
                    missing_reference_questions = [
                        question for question in questions if not references.get(question)
                    ]
                    if missing_reference_questions:
                        self._log(
                            "Generating fallback reference answers for "
                            f"{len(missing_reference_questions)} questions not covered by golden answers"
                        )
                        judge_spec = ModelSpec(
                            run_name="evaluation_judge",
                            provider=self.config.evaluation.judge_provider,
                            model_name=self.config.evaluation.judge_model,
                            rag_enabled=False,
                        )
                        judge_client = create_client(judge_spec, self.config.runtime)
                        generated_references = build_reference_answers(
                            missing_reference_questions,
                            retrievals,
                            judge_client,
                            context_result_limit=self.config.retrieval.generation_context_result_limit,
                            context_char_budget=self.config.retrieval.generation_context_char_budget,
                            compact_context=self.config.retrieval.compact_context_enabled,
                        )
                        for question, answer in generated_references.items():
                            references[question] = answer
                            reference_sources[question] = "generated"
                    else:
                        self._log("Using golden answers for answer_accuracy evaluation")
                    if write_checkpoint:
                        checkpoint_path = export_results(
                            output_dir=self.config.runtime.output_dir,
                            responses=responses,
                            retrievals=retrievals,
                            references=references,
                            reference_sources=reference_sources,
                            metrics_per_run=None,
                            metrics_summary=None,
                            filename_prefix="comparison_checkpoint",
                            single_sheet_only=single_sheet_only,
                        )
                        self._log(f"Checkpoint workbook written to {checkpoint_path}")
                    self._log("Running evaluation metrics")
                    last_metric_progress = 0

                    def log_metric_progress(completed: int, total: int, response: ModelResponse) -> None:
                        nonlocal last_metric_progress
                        if completed == total or completed - last_metric_progress >= 5:
                            self._log(
                                "Evaluation progress: "
                                f"{completed}/{total} "
                                f"({response.run_name}: {response.question[:80]})"
                            )
                            last_metric_progress = completed

                    metrics_per_run, metrics_summary = run_evaluation_metrics(
                        responses=responses,
                        reference_answers=references,
                        judge_provider=self.config.evaluation.judge_provider,
                        judge_model=self.config.evaluation.judge_model,
                        max_retries=self.config.evaluation.max_retries,
                        context_result_limit=self.config.retrieval.evaluation_context_result_limit,
                        context_char_budget=self.config.retrieval.evaluation_context_char_budget,
                        compact_context=self.config.retrieval.compact_context_enabled,
                        parallelism=self.config.evaluation.parallelism,
                        progress_callback=log_metric_progress,
                    )
                except Exception as exc:
                    self._log(
                        "Reference generation or evaluation failed: "
                        f"{exc}. Continuing without metric sheets."
                    )

                # ── Optional RAGAS phase ──────────────────────────────────────
                if (
                    self.config.evaluation.ragas_enabled
                    and not no_ragas
                    and not skip_evaluation
                    and metrics_per_run is not None
                ):
                    try:
                        from .ragas_integration import ragas_available, run_ragas_evaluation, _response_key
                        if ragas_available():
                            self._log("Running RAGAS evaluation phase")
                            ragas_scores = run_ragas_evaluation(
                                responses=responses,
                                golden_answers={q: r for q, r in references.items() if r},
                                judge_provider=self.config.evaluation.judge_provider,
                                judge_model=self.config.evaluation.judge_model,
                                embedding_model=self.config.evaluation.embedding_model,
                                context_result_limit=self.config.retrieval.evaluation_context_result_limit,
                                context_char_budget=self.config.retrieval.evaluation_context_char_budget,
                            )
                            if ragas_scores and not metrics_per_run.empty:
                                import pandas as _pd
                                ragas_cols = set()
                                for scores in ragas_scores.values():
                                    ragas_cols.update(scores.keys())
                                for col in ragas_cols:
                                    if col not in metrics_per_run.columns:
                                        metrics_per_run[col] = None
                                for resp in responses:
                                    key = _response_key(resp)
                                    if key not in ragas_scores:
                                        continue
                                    mask = (
                                        (metrics_per_run["run_name"] == resp.run_name) &
                                        (metrics_per_run["question"] == resp.question)
                                    )
                                    for col, val in ragas_scores[key].items():
                                        metrics_per_run.loc[mask, col] = val
                                self._log(f"RAGAS scores merged for {len(ragas_scores)} responses")
                        else:
                            self._log("RAGAS library not installed — skipping RAGAS phase")
                    except Exception as exc:
                        self._log(f"RAGAS evaluation failed: {exc}. Continuing without RAGAS scores.")

            if export_response_files and response_output_dir:
                response_dir_path = export_response_sets(
                    output_dir=Path(response_output_dir),
                    responses=responses,
                    references=references,
                    reference_sources=reference_sources,
                    metrics_per_run=metrics_per_run,
                    metrics_summary=metrics_summary,
                )
                self._log(f"Per-run response files written to {response_dir_path}")

            if single_model_report:
                if len(active_models) != 1:
                    raise ValueError(
                        "--single-model-report requires exactly one selected run. "
                        "Use --run-name once."
                    )
                last_report_progress = 0

                def log_report_progress(completed: int, total: int, response: ModelResponse) -> None:
                    nonlocal last_report_progress
                    if completed == total or completed - last_report_progress >= 5:
                        self._log(
                            "Single-model report progress: "
                            f"{completed}/{total} "
                            f"({response.run_name}: {response.question[:80]})"
                        )
                        last_report_progress = completed

                single_model_path = export_single_model_report(
                    output_dir=self.config.runtime.output_dir,
                    responses=responses,
                    references=references,
                    reference_sources=reference_sources,
                    judge_provider=self.config.evaluation.judge_provider,
                    judge_model=self.config.evaluation.judge_model,
                    max_retries=self.config.evaluation.max_retries,
                    context_result_limit=self.config.retrieval.evaluation_context_result_limit,
                    context_char_budget=self.config.retrieval.evaluation_context_char_budget,
                    compact_context=self.config.retrieval.compact_context_enabled,
                    metrics_per_run=metrics_per_run,
                    parallelism=self.config.evaluation.parallelism,
                    progress_callback=log_report_progress,
                )
                self._log(f"Single-model report written to {single_model_path}")

            self._log("Exporting results workbook")
            output_path = export_results(
                output_dir=self.config.runtime.output_dir,
                responses=responses,
                retrievals=retrievals,
                references=references,
                reference_sources=reference_sources,
                metrics_per_run=metrics_per_run,
                metrics_summary=metrics_summary,
                single_sheet_only=single_sheet_only,
            )
            metrics_path = None
            if not single_sheet_only:
                metrics_path = export_metrics_workbook(
                    output_dir=self.config.runtime.output_dir,
                    metrics_per_run=metrics_per_run,
                    metrics_summary=metrics_summary,
                )
                if metrics_path is not None:
                    self._log(f"Metrics workbook written to {metrics_path}")
            self._log(f"Done. Report written to {output_path}")
            return output_path
        finally:
            retriever.close()

    def _log(self, message: str) -> None:
        print(f"[ev-llm-compare] {message}", flush=True)

    def _select_models(self, selected_run_names: list[str] | None) -> list[ModelSpec]:
        if not selected_run_names:
            return self.config.models

        requested = set(selected_run_names)
        selected = [model for model in self.config.models if model.run_name in requested]
        missing = sorted(requested - {model.run_name for model in selected})
        if missing:
            available = ", ".join(model.run_name for model in self.config.models)
            missing_display = ", ".join(missing)
            raise ValueError(f"Unknown run name(s): {missing_display}. Available runs: {available}")
        return selected

    def _load_responses_from_jsonl(
        self,
        jsonl_path: Path,
        retrievals: dict[str, list],
    ) -> list[ModelResponse]:
        """Reconstruct ModelResponse objects from a responses_raw.jsonl checkpoint file.

        Retrieved chunks are re-attached from the in-memory retrievals dict so that
        evaluation metrics that need context (faithfulness, context precision, etc.)
        still work correctly.
        """
        spec_by_run_name = {spec.run_name: spec for spec in self.config.models}
        responses: list[ModelResponse] = []
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line_num, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    self._log(f"WARNING: Skipping malformed JSONL line {line_num}: {exc}")
                    continue
                run_name = data["run_name"]
                spec = spec_by_run_name.get(run_name)
                provider = spec.provider if spec else "ollama"
                rag_enabled = data.get("rag_enabled", False)
                question = data["question"]
                retrieved_chunks = retrievals.get(question, []) if rag_enabled else []
                responses.append(
                    ModelResponse(
                        run_name=run_name,
                        provider=provider,
                        model_name=data["model_name"],
                        rag_enabled=rag_enabled,
                        question=question,
                        answer=data.get("answer", ""),
                        latency_seconds=data.get("latency_seconds", 0.0),
                        retrieved_chunks=retrieved_chunks,
                        prompt_tokens_estimate=max(1, len(question) // 4),
                        success=data.get("success", True),
                        error_message=data.get("error_message"),
                    )
                )
        return responses

    def _resolve_reference_workbook(self, golden_workbook: str | None) -> Path | None:
        if golden_workbook:
            path = Path(golden_workbook).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Golden answers workbook not found: {path}")
            return path

        for candidate in (
            Path("data/Human validated 50 questions.xlsx"),
            Path("artifacts/Golden_answers_updated.xlsx"),
            Path("artifacts/Golden_answers.xlsx"),
        ):
            if candidate.exists():
                return candidate.resolve()
        return None
