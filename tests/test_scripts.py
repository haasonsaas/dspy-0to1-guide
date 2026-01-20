"""Unit tests for CLI scripts with minimal side effects."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import evaluate_all  # noqa: E402


@pytest.fixture()
def runner(monkeypatch) -> evaluate_all.EvaluationRunner:
    monkeypatch.setattr(
        evaluate_all.EvaluationRunner,
        "_configure_dspy",
        lambda self: setattr(self, "model_name", "mock-lm"),
    )
    monkeypatch.setattr(
        evaluate_all, "dspy", SimpleNamespace(configure=lambda **__: None)
    )

    instance = evaluate_all.EvaluationRunner()
    instance.results = {"basic": {}, "persona": {}, "advanced": {}}
    return instance


def test_extract_accuracy_from_output(runner):
    sample = "Correct: 4/6 (66.6%)"
    assert runner._extract_accuracy_from_output(sample) == pytest.approx(66.6)


def test_generate_report_aggregates_success(runner):
    runner.results = {
        "basic": {
            "hello_world": {"success": True},
            "math_qa": {"success": True, "accuracy": 42.0},
            "summarizer": {"success": False, "error": "Timeout"},
        },
        "persona": {},
        "advanced": {},
    }

    report = runner.generate_report()

    summary = report["evaluation_summary"]
    assert summary["successful"] == 2
    assert summary["failed"] == 1
    assert report[
        "recommendations"
    ], "Should recommend when accuracy is low or timeouts occur"
