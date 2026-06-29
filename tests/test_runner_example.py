from __future__ import annotations

from evalkit.runner import run_spec
from evalkit.spec import load_spec


def test_rag_chatbot_example_runs_end_to_end(tmp_path):
    spec = load_spec("examples/rag_chatbot/spec.yaml")
    spec.report.output_dir = tmp_path

    payload, written = run_spec(spec)

    assert payload["summary"]["total_items"] == 2
    assert payload["summary"]["mean_score"] == 1.0
    assert payload["prompt_optimization"]["lift"] >= 0
    assert "json" in written
    assert "markdown" in written
