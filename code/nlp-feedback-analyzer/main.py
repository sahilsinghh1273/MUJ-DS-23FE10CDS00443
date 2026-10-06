"""CLI entry point for the Customer Feedback Analyzer.

Usage:
  python main.py analyze "The app crashes every time I open settings."
  python main.py batch --input data/sample_feedback.csv
  python main.py evaluate --input data/labeled_eval.csv
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.analyzer import FeedbackAnalyzer
from src.config import load_config, load_prompts
from src.evaluate import evaluate, format_evaluation
from src.llm_client import LLMClient
from src.report import summarize, to_markdown


def build_analyzer(config_path: str | None = None) -> tuple[FeedbackAnalyzer, dict]:
    cfg = load_config(config_path) if config_path else load_config()
    prompts = load_prompts(cfg=cfg)
    return FeedbackAnalyzer(LLMClient(cfg), prompts, cfg), cfg


def cmd_analyze(args):
    analyzer, _ = build_analyzer(args.config)
    try:
        result = analyzer.analyze(args.text)
        analyzer.cache.flush()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if "error" in result:
            sys.exit(1)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False))
        sys.exit(1)


def cmd_batch(args):
    analyzer, cfg = build_analyzer(args.config)
    column = args.column or cfg["pipeline"]["text_column"]

    if not Path(args.input).exists():
        sys.exit(f"Input file not found: {args.input}")

    with open(args.input, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or column not in rows[0]:
        sys.exit(f"Column '{column}' not found in {args.input}")

    results = analyzer.analyze_batch([r[column] for r in rows])

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = [{**row, "analysis": res} for row, res in zip(rows, results)]
    (out_dir / "results.json").write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    stats = summarize(results)
    report_md = to_markdown(stats)
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")
    print(report_md)
    print(f"Saved: {out_dir / 'results.json'} and {out_dir / 'report.md'}")


def cmd_evaluate(args):
    analyzer, _ = build_analyzer(args.config)
    if not Path(args.input).exists():
        sys.exit(f"Evaluation dataset not found: {args.input}")
    result = evaluate(analyzer, args.input)
    print(format_evaluation(result))


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="LLM-powered customer feedback analyzer")
    parser.add_argument("--config", help="path to config.yaml (default: ./config.yaml)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="analyze a single piece of feedback")
    a.add_argument("text", help="feedback text to analyze")
    a.set_defaults(func=cmd_analyze)

    b = sub.add_parser("batch", help="analyze a CSV file")
    b.add_argument("--input", required=True, help="path to input CSV")
    b.add_argument("--column", help="CSV column with the feedback text")
    b.set_defaults(func=cmd_batch)

    e = sub.add_parser("evaluate", help="evaluate against a labeled CSV benchmark")
    e.add_argument("--input", default="data/labeled_eval.csv", help="path to labeled benchmark CSV")
    e.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    try:
        args.func(args)
    except RuntimeError as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
