from __future__ import annotations

from evalkit.dataset import DataRow
from evalkit.evaluators.deterministic import ExactMatchEvaluator, JsonFieldMatchEvaluator, RegexMatchEvaluator


def test_exact_match_ignores_case_by_default():
    row = DataRow(id="1", input="q", expected_output="Hello")

    result = ExactMatchEvaluator().evaluate(row, " hello ")

    assert result.passed
    assert result.score == 1.0


def test_regex_match_uses_expected_output_as_pattern_by_default():
    row = DataRow(id="1", input="q", expected_output=r"order-\d+")

    result = RegexMatchEvaluator().evaluate(row, "Your id is ORDER-123")

    assert result.passed


def test_json_field_match_reports_invalid_json():
    row = DataRow(id="1", input="q", expected_output='{"status":"approved"}')

    result = JsonFieldMatchEvaluator(field="status").evaluate(row, "not json")

    assert not result.passed
    assert result.failure_mode == "Invalid JSON output"
