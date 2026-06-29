"""Deterministic evaluators that do not require an LLM."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from evalkit.dataset import DataRow
from evalkit.evaluators.base import EvaluationResult
from evalkit.llm_client import LLMClient


@dataclass
class ExactMatchEvaluator:
    name: str = "exact_match"
    case_sensitive: bool = False
    strip: bool = True
    threshold: float = 1.0
    suggested_knob: str = "prompt"

    requires_llm: bool = False

    def evaluate(
        self,
        row: DataRow,
        output: str,
        *,
        llm_client: LLMClient | None = None,
    ) -> EvaluationResult:
        expected = _normalize(row.expected_output, self.case_sensitive, self.strip)
        actual = _normalize(output, self.case_sensitive, self.strip)
        score = 1.0 if actual == expected else 0.0
        passed = score >= self.threshold
        return EvaluationResult(
            name=self.name,
            score=score,
            passed=passed,
            reasoning="Exact match." if passed else f"Expected '{row.expected_output}' but got '{output}'.",
            failure_mode=None if passed else "Expected answer mismatch",
            severity=0.0 if passed else 3.0,
            impact=0.0 if passed else 3.0,
            suggested_knob=None if passed else self.suggested_knob,
        )


@dataclass
class RegexMatchEvaluator:
    name: str = "regex_match"
    pattern: str | None = None
    flags: str = "IGNORECASE"
    threshold: float = 1.0
    suggested_knob: str = "prompt"

    requires_llm: bool = False

    def evaluate(
        self,
        row: DataRow,
        output: str,
        *,
        llm_client: LLMClient | None = None,
    ) -> EvaluationResult:
        pattern = self.pattern or row.expected_output
        compiled_flags = 0
        if "IGNORECASE" in self.flags.upper():
            compiled_flags |= re.IGNORECASE
        matched = re.search(pattern, output, compiled_flags) is not None
        score = 1.0 if matched else 0.0
        passed = score >= self.threshold
        return EvaluationResult(
            name=self.name,
            score=score,
            passed=passed,
            reasoning="Regex matched." if passed else f"Output did not match regex '{pattern}'.",
            failure_mode=None if passed else "Pattern mismatch",
            severity=0.0 if passed else 2.0,
            impact=0.0 if passed else 3.0,
            suggested_knob=None if passed else self.suggested_knob,
            details={"pattern": pattern},
        )


@dataclass
class JsonFieldMatchEvaluator:
    name: str = "json_field_match"
    field: str = ""
    case_sensitive: bool = False
    threshold: float = 1.0
    suggested_knob: str = "prompt"

    requires_llm: bool = False

    def evaluate(
        self,
        row: DataRow,
        output: str,
        *,
        llm_client: LLMClient | None = None,
    ) -> EvaluationResult:
        if not self.field:
            raise ValueError("json_field_match requires a 'field' parameter.")
        try:
            parsed_output = json.loads(output)
        except json.JSONDecodeError:
            return EvaluationResult(
                name=self.name,
                score=0.0,
                passed=False,
                reasoning="Output is not valid JSON.",
                failure_mode="Invalid JSON output",
                severity=3.0,
                impact=3.0,
                suggested_knob=self.suggested_knob,
            )

        actual = _extract_path(parsed_output, self.field)
        expected = _expected_json_value(row, self.field)
        matched = _normalize(actual, self.case_sensitive, True) == _normalize(expected, self.case_sensitive, True)
        score = 1.0 if matched else 0.0
        passed = score >= self.threshold
        return EvaluationResult(
            name=self.name,
            score=score,
            passed=passed,
            reasoning="JSON field matched." if passed else f"Field '{self.field}' expected '{expected}' but got '{actual}'.",
            failure_mode=None if passed else "JSON field mismatch",
            severity=0.0 if passed else 3.0,
            impact=0.0 if passed else 3.0,
            suggested_knob=None if passed else self.suggested_knob,
            details={"field": self.field, "expected": expected, "actual": actual},
        )


def _normalize(value: Any, case_sensitive: bool, strip: bool) -> str:
    text = "" if value is None else str(value)
    if strip:
        text = text.strip()
    if not case_sensitive:
        text = text.lower()
    return text


def _expected_json_value(row: DataRow, field: str) -> Any:
    try:
        parsed_expected = json.loads(row.expected_output)
    except json.JSONDecodeError:
        return row.expected_output
    return _extract_path(parsed_expected, field)


def _extract_path(value: Any, dotted_path: str) -> Any:
    current = value
    for part in dotted_path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            current = current[int(part)]
        else:
            return None
    return current
