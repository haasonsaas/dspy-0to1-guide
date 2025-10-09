"""Deterministic unit tests for key DSPy examples."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.advanced.pydantic_validation import (  # noqa: E402
    ContractAnalysisResult,
    StructuredContractAnalyzer,
)
from examples.basic.math_qa import MathQA, validate_solution  # noqa: E402
from examples.basic.summarizer import (  # noqa: E402
    AdaptiveSummarizer,
    DocumentSummarizer,
)


class DummyPredictor:
    """Deterministic predictor stub for replacing LLM-backed modules."""

    def __init__(self, factory: Callable[..., Any] | Any):
        self.factory = factory
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if callable(self.factory):
            return self.factory(**kwargs)
        return self.factory


def as_namespace(**fields: Any) -> SimpleNamespace:
    return SimpleNamespace(**fields)


def test_math_qa_extracts_numeric_solution():
    math_qa = MathQA()
    math_qa.solve = DummyPredictor(
        lambda **_: as_namespace(solution="Answer: 4", explanation="addition")
    )

    result = math_qa("What is 2 + 2?")

    assert result.solution == pytest.approx(4.0)
    assert validate_solution("2+2", 4.0, result)


def test_validate_solution_tolerates_small_error():
    response = as_namespace(solution=4.009, explanation="floating tolerance")
    assert validate_solution("calc", 4.0, response) is True


def test_document_summarizer_respects_max_length():
    summarizer = DocumentSummarizer("comprehensive")
    dummy = DummyPredictor(
        lambda **kwargs: as_namespace(
            summary="short summary", key_points=["point one", "point two"], **kwargs
        )
    )
    summarizer.summarize = dummy

    document = "word " * 50
    result = summarizer(document, max_length=42)

    assert dummy.calls[0]["max_length"] == "42 words maximum"
    assert result.summary == "short summary"
    assert result.key_points == ["point one", "point two"]


@pytest.mark.parametrize(
    "document, doc_type, expected_strategy, summary_attr",
    [
        ("Deep learning research", "technical briefing", "technical", "technical_summary"),
        ("Quarterly revenue update", "business report", "executive", "executive_summary"),
        ("Community garden update", "news article", "comprehensive", "summary"),
    ],
)
def test_adaptive_summarizer_strategy_selection(document, doc_type, expected_strategy, summary_attr):
    adaptive = AdaptiveSummarizer()

    adaptive.classifier = DummyPredictor(lambda **_: as_namespace(document_type=doc_type))
    adaptive.technical.summarize = DummyPredictor(
        lambda **_: as_namespace(technical_summary="technical", key_concepts=["attention"])
    )
    adaptive.executive.summarize = DummyPredictor(
        lambda **_: as_namespace(executive_summary="executive", action_items=["review goals"])
    )
    adaptive.comprehensive.summarize = DummyPredictor(
        lambda **_: as_namespace(summary="general", key_points=["growth"])
    )

    result = adaptive(document, max_length=75)

    assert result.strategy_used == expected_strategy
    assert result.detected_type == doc_type
    assert hasattr(result, summary_attr)


def test_structured_contract_analyzer_uses_llm_signal():
    analyzer = StructuredContractAnalyzer()
    analyzer.analyzer = DummyPredictor(
        lambda **_: as_namespace(
            analysis=(
                "Annual service fee of $120,000 payable monthly with termination in 45 days. "
                "Liability limited and automatic renewal applies."
            )
        )
    )

    contract_text = "Consulting services agreement between Provider Inc. and Client LLC."
    result = analyzer(contract_text)

    assert isinstance(result, ContractAnalysisResult)
    assert result.contract_type == "service"
    assert result.financial_terms and result.financial_terms[0].amount == pytest.approx(120000.0)
    assert "45" in (result.duration or "")
    assert result.termination_notice >= 30
    red_flags = {flag.lower() for flag in result.red_flags}
    assert "automatic renewal" in red_flags


def test_structured_contract_analyzer_fallback(monkeypatch):
    analyzer = StructuredContractAnalyzer()

    def failing_predictor(**_):
        raise RuntimeError("llm offline")

    analyzer.analyzer = DummyPredictor(failing_predictor)

    result = analyzer("Short contract text")

    assert isinstance(result, ContractAnalysisResult)
    assert result.contract_type == "other"
    assert result.compliance_score == 75