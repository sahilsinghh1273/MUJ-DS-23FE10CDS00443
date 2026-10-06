"""Tests for CLI entry points: analyze, batch, and evaluate subcommands."""
import csv
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

import main


class FakeCLIAnalyzer:
    def __init__(self, result=None):
        self.result = result or {
            "sentiment": "positive",
            "sentiment_score": 0.8,
            "category": "customer_support",
            "urgency": "low",
            "entities": [],
            "key_issues": ["helpful support"],
            "summary": "Great support experience.",
            "suggested_reply": "Thank you!",
        }
        self.cache = MagicMock()
        self.cache.flush = MagicMock()

    def analyze(self, text: str) -> dict:
        if text == "trigger_error":
            return {"error": "Simulated analysis failure"}
        return self.result

    def analyze_batch(self, texts: list[str], on_progress=None) -> list[dict]:
        return [self.analyze(t) for t in texts]


@pytest.fixture
def mock_build_analyzer(monkeypatch, tmp_path):
    analyzer = FakeCLIAnalyzer()
    cfg = {
        "pipeline": {"text_column": "feedback", "max_workers": 2},
        "paths": {"output_dir": str(tmp_path / "output")},
    }
    monkeypatch.setattr(main, "build_analyzer", lambda config_path=None: (analyzer, cfg))
    return analyzer, cfg


def test_cli_analyze_success(mock_build_analyzer, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py", "analyze", "The service was fantastic!"])
    main.main()

    captured = capsys.readouterr()
    assert '"sentiment": "positive"' in captured.out
    assert '"category": "customer_support"' in captured.out


def test_cli_analyze_error_exits_with_code_1(mock_build_analyzer, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py", "analyze", "trigger_error"])
    with pytest.raises(SystemExit) as exc_info:
        main.main()
    assert exc_info.value.code == 1


def test_cli_batch_success(mock_build_analyzer, tmp_path, monkeypatch, capsys):
    input_csv = tmp_path / "test_feedback.csv"
    with open(input_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "feedback"])
        writer.writeheader()
        writer.writerow({"id": "1", "feedback": "Great customer service!"})
        writer.writerow({"id": "2", "feedback": "Super fast delivery"})

    monkeypatch.setattr(sys, "argv", ["main.py", "batch", "--input", str(input_csv)])
    main.main()

    captured = capsys.readouterr()
    assert "Feedback Analysis Report" in captured.out
    assert "Saved:" in captured.out

    out_dir = tmp_path / "output"
    assert (out_dir / "results.json").exists()
    assert (out_dir / "report.md").exists()

    saved_results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    assert len(saved_results) == 2


def test_cli_batch_missing_column_exits(mock_build_analyzer, tmp_path, monkeypatch):
    input_csv = tmp_path / "bad_columns.csv"
    with open(input_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "comments"])
        writer.writeheader()
        writer.writerow({"id": "1", "comments": "Missing 'feedback' column"})

    monkeypatch.setattr(sys, "argv", ["main.py", "batch", "--input", str(input_csv)])
    with pytest.raises(SystemExit) as exc_info:
        main.main()
    assert "not found" in str(exc_info.value)


def test_cli_batch_nonexistent_file_exits(mock_build_analyzer, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "batch", "--input", "nonexistent.csv"])
    with pytest.raises(SystemExit) as exc_info:
        main.main()
    assert "not found" in str(exc_info.value)


def test_cli_evaluate_success(mock_build_analyzer, tmp_path, monkeypatch, capsys):
    eval_csv = tmp_path / "eval.csv"
    with open(eval_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["feedback", "sentiment", "category", "urgency"])
        writer.writeheader()
        writer.writerow({"feedback": "Support was amazing", "sentiment": "positive", "category": "customer_support", "urgency": "low"})

    monkeypatch.setattr(sys, "argv", ["main.py", "evaluate", "--input", str(eval_csv)])
    main.main()

    captured = capsys.readouterr()
    assert "# Evaluation Results" in captured.out
    assert "Total Rows: 1" in captured.out
    assert "Sentiment: 100.0%" in captured.out


def test_cli_evaluate_missing_file_exits(mock_build_analyzer, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "evaluate", "--input", "missing_eval.csv"])
    with pytest.raises(SystemExit) as exc_info:
        main.main()
    assert "not found" in str(exc_info.value)
