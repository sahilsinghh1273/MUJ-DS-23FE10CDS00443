"""Tests for evaluation metrics calculation, MAE, error handling, and formatting."""
import csv
from pathlib import Path
import pytest
from src.evaluate import evaluate, format_evaluation


def create_eval_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["feedback", "sentiment", "category", "urgency", "sentiment_score"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


class MockAnalyzer:
    def __init__(self, responses: list):
        self.responses = list(responses)
        self.call_count = 0

    def analyze(self, text: str) -> dict:
        self.call_count += 1
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


def test_evaluate_perfect_match(tmp_path):
    csv_file = tmp_path / "eval.csv"
    create_eval_csv(csv_file, [
        {"feedback": "Text 1", "sentiment": "positive", "category": "billing", "urgency": "low", "sentiment_score": "0.8"},
        {"feedback": "Text 2", "sentiment": "negative", "category": "delivery", "urgency": "high", "sentiment_score": "-0.9"},
    ])

    analyzer = MockAnalyzer([
        {"sentiment": "positive", "sentiment_score": 0.8, "category": "billing", "urgency": "low"},
        {"sentiment": "negative", "sentiment_score": -0.9, "category": "delivery", "urgency": "high"},
    ])

    res = evaluate(analyzer, csv_file)
    assert res["total"] == 2
    assert res["evaluated"] == 2
    assert res["errors"] == 0
    assert res["overall_accuracy"] == 1.0
    assert res["field_accuracy"]["sentiment"] == 1.0
    assert res["field_accuracy"]["category"] == 1.0
    assert res["field_accuracy"]["urgency"] == 1.0
    assert res["sentiment_score_mae"] == 0.0
    assert len(res["mismatches"]) == 0


def test_evaluate_partial_mismatches(tmp_path):
    csv_file = tmp_path / "eval.csv"
    create_eval_csv(csv_file, [
        # Row 1: All match
        {"feedback": "T1", "sentiment": "positive", "category": "billing", "urgency": "low", "sentiment_score": "0.8"},
        # Row 2: Sentiment mismatch (predicted negative)
        {"feedback": "T2", "sentiment": "positive", "category": "billing", "urgency": "low", "sentiment_score": "0.8"},
        # Row 3: Category mismatch (predicted delivery)
        {"feedback": "T3", "sentiment": "positive", "category": "billing", "urgency": "low", "sentiment_score": "0.8"},
        # Row 4: Urgency mismatch (predicted high)
        {"feedback": "T4", "sentiment": "positive", "category": "billing", "urgency": "low", "sentiment_score": "0.8"},
    ])

    analyzer = MockAnalyzer([
        {"sentiment": "positive", "sentiment_score": 0.8, "category": "billing", "urgency": "low"},
        {"sentiment": "negative", "sentiment_score": -0.6, "category": "billing", "urgency": "low"},
        {"sentiment": "positive", "sentiment_score": 0.8, "category": "delivery", "urgency": "low"},
        {"sentiment": "positive", "sentiment_score": 0.8, "category": "billing", "urgency": "high"},
    ])

    res = evaluate(analyzer, csv_file)
    assert res["total"] == 4
    assert res["evaluated"] == 4
    assert res["errors"] == 0
    # Exactly 1 out of 4 rows has all 3 fields correct -> 0.25 overall
    assert res["overall_accuracy"] == 0.25
    # Each field was correct 3 out of 4 times -> 0.75
    assert res["field_accuracy"]["sentiment"] == 0.75
    assert res["field_accuracy"]["category"] == 0.75
    assert res["field_accuracy"]["urgency"] == 0.75
    assert len(res["mismatches"]) == 3

    # Check mismatch structure
    m0 = res["mismatches"][0]
    assert m0["text"] == "T2"
    assert m0["field"] == "sentiment"
    assert m0["expected"] == "positive"
    assert m0["predicted"] == "negative"


def test_evaluate_sentiment_score_mae(tmp_path):
    csv_file = tmp_path / "eval.csv"
    create_eval_csv(csv_file, [
        {"feedback": "T1", "sentiment": "positive", "category": "other", "urgency": "low", "sentiment_score": "1.0"},
        {"feedback": "T2", "sentiment": "negative", "category": "other", "urgency": "low", "sentiment_score": "-1.0"},
    ])

    analyzer = MockAnalyzer([
        {"sentiment": "positive", "sentiment_score": 0.8, "category": "other", "urgency": "low"},   # diff = 0.2
        {"sentiment": "negative", "sentiment_score": -0.6, "category": "other", "urgency": "low"},  # diff = 0.4
    ])

    res = evaluate(analyzer, csv_file)
    # MAE = (0.2 + 0.4) / 2 = 0.3
    assert res["sentiment_score_mae"] == 0.3


def test_evaluate_handles_analysis_failure_gracefully(tmp_path):
    csv_file = tmp_path / "eval.csv"
    create_eval_csv(csv_file, [
        {"feedback": "Good text", "sentiment": "positive", "category": "billing", "urgency": "low", "sentiment_score": "0.8"},
        {"feedback": "Bad text", "sentiment": "negative", "category": "delivery", "urgency": "high", "sentiment_score": "-0.9"},
    ])

    analyzer = MockAnalyzer([
        {"sentiment": "positive", "sentiment_score": 0.8, "category": "billing", "urgency": "low"},
        RuntimeError("API crashed unexpectedly"),
    ])

    res = evaluate(analyzer, csv_file)
    assert res["total"] == 2
    assert res["evaluated"] == 1
    assert res["errors"] == 1
    assert res["field_accuracy"]["sentiment"] == 1.0
    assert any(m["field"] == "analysis_error" for m in res["mismatches"])


def test_format_evaluation_renders_markdown():
    mock_result = {
        "total": 10,
        "evaluated": 9,
        "errors": 1,
        "field_accuracy": {"sentiment": 0.8889, "category": 0.7778, "urgency": 1.0},
        "overall_accuracy": 0.7778,
        "sentiment_score_mae": 0.1542,
        "mismatches": [
            {"text": "Sample feedback text", "field": "sentiment", "expected": "positive", "predicted": "neutral"}
        ],
    }

    report = format_evaluation(mock_result)
    assert "# Evaluation Results" in report
    assert "Total Rows: 10" in report
    assert "Overall Exact Match (All Fields): 77.8%" in report
    assert "Sentiment: 88.9%" in report
    assert "Sample feedback text" in report
