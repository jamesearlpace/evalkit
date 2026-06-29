"""Eval execution orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from evalkit.app_under_test import AppUnderTest, build_app
from evalkit.dataset import DataRow, load_dataset, split_dataset
from evalkit.evaluators import build_evaluators
from evalkit.evaluators.base import EvaluationResult, Evaluator
from evalkit.failure_modes import cluster_failure_modes
from evalkit.knobs.prompt_optimizer import PromptOptimizationResult, PromptOptimizer
from evalkit.llm_client import LLMClient, build_llm_client
from evalkit.report import write_reports
from evalkit.spec import EvalSpec


@dataclass
class ItemRunResult:
    row: DataRow
    output: str
    app_metadata: dict[str, Any]
    evaluations: list[EvaluationResult]

    @property
    def item_score(self) -> float:
        if not self.evaluations:
            return 0.0
        return mean(result.score for result in self.evaluations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.row.id,
            "input": self.row.input,
            "expected_output": self.row.expected_output,
            "context": self.row.context,
            "metadata": self.row.metadata,
            "output": self.output,
            "app_metadata": self.app_metadata,
            "item_score": round(self.item_score, 6),
            "evaluations": [result.to_dict() for result in self.evaluations],
        }


@dataclass
class EvaluationRun:
    split_name: str
    items: list[ItemRunResult]
    summary: dict[str, Any]
    failure_modes: list[dict[str, Any]]


def run_spec(spec: EvalSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = load_dataset(spec.dataset.path, spec.dataset.format)
    splits = split_dataset(rows, spec.split.ratios, spec.split.seed)
    evaluators = build_evaluators(spec.evaluators)
    llm_client = build_llm_client(spec.llm)
    app = build_app(spec.app)

    prompt_result: PromptOptimizationResult | None = None
    target_split = spec.split.target
    if spec.prompt_optimization.enabled:
        base_prompt = spec.prompt_optimization.base_prompt
        if base_prompt is None:
            base_prompt = str(spec.app.params.get(spec.prompt_optimization.app_param, ""))
        optimizer = PromptOptimizer(
            app_param=spec.prompt_optimization.app_param,
            base_prompt=base_prompt,
            candidates=spec.prompt_optimization.candidates,
            num_variants=spec.prompt_optimization.num_variants,
            train_split=spec.prompt_optimization.train_split,
            eval_split=spec.prompt_optimization.eval_split,
        )
        prompt_result = optimizer.optimize(
            splits=splits,
            app=app,
            evaluators=evaluators,
            llm_client=llm_client,
            evaluate_func=evaluate_rows,
        )
        app = app.with_params({spec.prompt_optimization.app_param: prompt_result.best_prompt})
        target_split = spec.prompt_optimization.eval_split

    if target_split not in splits:
        available = ", ".join(splits)
        raise ValueError(f"Split '{target_split}' does not exist. Available splits: {available}")

    run = evaluate_rows(splits[target_split], app, evaluators, llm_client, target_split)
    payload = _build_payload(spec, run, prompt_result)
    written = write_reports(payload, spec.report.output_dir, spec.report.formats)
    return payload, {key: str(value) for key, value in written.items()}


def evaluate_rows(
    rows: list[DataRow],
    app: AppUnderTest,
    evaluators: list[Evaluator],
    llm_client: LLMClient | None,
    split_name: str,
) -> EvaluationRun:
    items: list[ItemRunResult] = []
    for row in rows:
        app_result = app.run(row)
        evaluations = [
            evaluator.evaluate(row, app_result.output, llm_client=llm_client)
            for evaluator in evaluators
        ]
        items.append(
            ItemRunResult(
                row=row,
                output=app_result.output,
                app_metadata=app_result.metadata,
                evaluations=evaluations,
            )
        )

    failure_modes = [
        item.to_dict()
        for item in cluster_failure_modes(
            (item.row.id, item.row.input, item.evaluations) for item in items
        )
    ]
    return EvaluationRun(
        split_name=split_name,
        items=items,
        summary=_summarize(items, evaluators),
        failure_modes=failure_modes,
    )


def _summarize(items: list[ItemRunResult], evaluators: list[Evaluator]) -> dict[str, Any]:
    evaluator_metrics: dict[str, dict[str, float]] = {}
    for evaluator in evaluators:
        results = [
            result
            for item in items
            for result in item.evaluations
            if result.name == evaluator.name
        ]
        if results:
            evaluator_metrics[evaluator.name] = {
                "mean_score": mean(result.score for result in results),
                "pass_rate": sum(1 for result in results if result.passed) / len(results),
            }
        else:
            evaluator_metrics[evaluator.name] = {"mean_score": 0.0, "pass_rate": 0.0}

    return {
        "total_items": len(items),
        "mean_score": mean(item.item_score for item in items) if items else 0.0,
        "evaluators": evaluator_metrics,
    }


def _build_payload(
    spec: EvalSpec,
    run: EvaluationRun,
    prompt_result: PromptOptimizationResult | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_name": spec.report.run_name,
        "spec_path": _display_path(spec.path),
        "split": run.split_name,
        "summary": run.summary,
        "failure_modes": run.failure_modes,
        "items": [item.to_dict() for item in run.items],
    }
    if prompt_result is not None:
        payload["prompt_optimization"] = prompt_result.to_dict()
    return payload


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)
