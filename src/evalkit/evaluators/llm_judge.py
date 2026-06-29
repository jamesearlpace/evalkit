"""LLM-as-judge evaluators."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from evalkit.dataset import DataRow
from evalkit.evaluators.base import EvaluationResult
from evalkit.llm_client import LLMClient


@dataclass
class LLMJudgeEvaluator:
    name: str = "llm_judge"
    criterion: str = "correctness"
    threshold: float = 0.7
    temperature: float = 0.0
    suggested_knob: str = "prompt"

    requires_llm: bool = True

    def evaluate(
        self,
        row: DataRow,
        output: str,
        *,
        llm_client: LLMClient | None = None,
    ) -> EvaluationResult:
        if llm_client is None:
            raise RuntimeError(f"Evaluator '{self.name}' requires an llm block in the spec.")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an evaluator for AI application outputs. "
                    "Return only JSON with keys score, reasoning, failure_mode, severity, impact, suggested_knob. "
                    "score must be 0.0 to 1.0. severity and impact are 0 to 5."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Criterion: {self.criterion}\n"
                    f"Input: {row.input}\n"
                    f"Context: {row.context}\n"
                    f"Expected output: {row.expected_output}\n"
                    f"Actual output: {output}\n"
                    "Judge whether the actual output satisfies the criterion."
                ),
            },
        ]
        content = llm_client.complete(
            messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        parsed = _parse_json_object(content)
        score = _normalize_score(parsed.get("score", 0.0))
        passed = score >= self.threshold
        failure_mode = parsed.get("failure_mode") if not passed else None
        return EvaluationResult(
            name=self.name,
            score=score,
            passed=passed,
            reasoning=str(parsed.get("reasoning", "")).strip(),
            failure_mode=str(failure_mode) if failure_mode else None,
            severity=0.0 if passed else _float_or_default(parsed.get("severity"), 3.0),
            impact=0.0 if passed else _float_or_default(parsed.get("impact"), 3.0),
            suggested_knob=None if passed else str(parsed.get("suggested_knob") or self.suggested_knob),
            details={"criterion": self.criterion, "raw_response": content},
        )


def _parse_json_object(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"score": 0.0, "reasoning": f"Judge did not return JSON: {content}"}
        try:
            parsed = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return {"score": 0.0, "reasoning": f"Judge returned invalid JSON: {content}"}
    return parsed if isinstance(parsed, dict) else {"score": 0.0, "reasoning": "Judge JSON was not an object."}


def _normalize_score(value: Any) -> float:
    score = _float_or_default(value, 0.0)
    if score > 1.0:
        score = score / 5.0
    return max(0.0, min(1.0, score))


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
