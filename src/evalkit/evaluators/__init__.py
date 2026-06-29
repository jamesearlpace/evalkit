"""Evaluator factory."""

from __future__ import annotations

from evalkit.evaluators.base import EvaluationResult, Evaluator
from evalkit.evaluators.deterministic import (
    ExactMatchEvaluator,
    JsonFieldMatchEvaluator,
    RegexMatchEvaluator,
)
from evalkit.evaluators.llm_judge import LLMJudgeEvaluator
from evalkit.spec import EvaluatorSpec


def build_evaluator(spec: EvaluatorSpec) -> Evaluator:
    evaluator_type = spec.type.replace("-", "_")
    params = dict(spec.params)
    params.setdefault("name", spec.name)

    if evaluator_type in {"exact", "exact_match"}:
        return ExactMatchEvaluator(**params)
    if evaluator_type in {"regex", "regex_match"}:
        return RegexMatchEvaluator(**params)
    if evaluator_type in {"json_field", "json_field_match"}:
        return JsonFieldMatchEvaluator(**params)
    if evaluator_type in {"llm", "llm_judge", "judge"}:
        return LLMJudgeEvaluator(**params)

    raise ValueError(f"Unsupported evaluator type '{spec.type}'.")


def build_evaluators(specs: list[EvaluatorSpec]) -> list[Evaluator]:
    return [build_evaluator(spec) for spec in specs]


__all__ = [
    "EvaluationResult",
    "Evaluator",
    "ExactMatchEvaluator",
    "JsonFieldMatchEvaluator",
    "LLMJudgeEvaluator",
    "RegexMatchEvaluator",
    "build_evaluator",
    "build_evaluators",
]
