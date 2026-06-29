"""Evaluator interfaces and shared result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from evalkit.dataset import DataRow
from evalkit.llm_client import LLMClient


@dataclass
class EvaluationResult:
    name: str
    score: float
    passed: bool
    reasoning: str = ""
    failure_mode: str | None = None
    severity: float = 0.0
    impact: float = 0.0
    suggested_knob: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "passed": self.passed,
            "reasoning": self.reasoning,
            "failure_mode": self.failure_mode,
            "severity": self.severity,
            "impact": self.impact,
            "suggested_knob": self.suggested_knob,
            "details": self.details,
        }


class Evaluator(Protocol):
    name: str
    requires_llm: bool

    def evaluate(
        self,
        row: DataRow,
        output: str,
        *,
        llm_client: LLMClient | None = None,
    ) -> EvaluationResult:
        """Evaluate one app output against one dataset row."""
