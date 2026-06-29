"""Failure-mode clustering and prioritization."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Iterable

from evalkit.evaluators.base import EvaluationResult


@dataclass
class FailureModeSummary:
    failure_mode: str
    occurrence: int
    average_severity: float
    average_impact: float
    priority: float
    suggested_knob: str | None
    examples: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "failure_mode": self.failure_mode,
            "occurrence": self.occurrence,
            "average_severity": round(self.average_severity, 3),
            "average_impact": round(self.average_impact, 3),
            "priority": round(self.priority, 3),
            "suggested_knob": self.suggested_knob,
            "examples": self.examples,
        }


def cluster_failure_modes(
    item_results: Iterable[tuple[str, str, list[EvaluationResult]]],
    *,
    max_examples: int = 3,
) -> list[FailureModeSummary]:
    """Group failed evaluator results by failure-mode label."""

    grouped: dict[str, list[tuple[str, str, EvaluationResult]]] = defaultdict(list)
    for row_id, input_text, evaluations in item_results:
        for result in evaluations:
            if result.passed:
                continue
            label = result.failure_mode or _fallback_label(result.reasoning)
            grouped[label].append((row_id, input_text, result))

    summaries: list[FailureModeSummary] = []
    for label, failures in grouped.items():
        severities = [failure.severity or 1.0 for _, _, failure in failures]
        impacts = [failure.impact or 1.0 for _, _, failure in failures]
        knob_counts = Counter(failure.suggested_knob for _, _, failure in failures if failure.suggested_knob)
        suggested_knob = knob_counts.most_common(1)[0][0] if knob_counts else None
        occurrence = len(failures)
        avg_severity = mean(severities)
        avg_impact = mean(impacts)
        summaries.append(
            FailureModeSummary(
                failure_mode=label,
                occurrence=occurrence,
                average_severity=avg_severity,
                average_impact=avg_impact,
                priority=occurrence * avg_severity * avg_impact,
                suggested_knob=suggested_knob,
                examples=[f"{row_id}: {input_text}" for row_id, input_text, _ in failures[:max_examples]],
            )
        )

    return sorted(summaries, key=lambda item: item.priority, reverse=True)


def _fallback_label(reasoning: str) -> str:
    text = reasoning.lower()
    if "json" in text:
        return "Invalid or mismatched structured output"
    if "regex" in text or "pattern" in text:
        return "Pattern mismatch"
    if "expected" in text:
        return "Expected answer mismatch"
    return "Unclassified quality failure"
