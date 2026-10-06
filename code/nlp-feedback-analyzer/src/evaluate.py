"""Model evaluation on labeled benchmark data."""
import csv
from pathlib import Path
from typing import Any

from .schema import CATEGORIES, SENTIMENTS, URGENCIES


def evaluate(analyzer: Any, csv_path: str | Path) -> dict:
    """Evaluate an analyzer against a labeled CSV dataset.

    Compares sentiment, category, urgency (exact match) and sentiment_score (MAE).
    Returns per-field accuracy, overall accuracy, error count, and a list of mismatches.
    Rows that fail analysis are counted in errors and do not crash the run.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Evaluation file not found: {csv_file}")

    with open(csv_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    if total == 0:
        return {
            "total": 0,
            "evaluated": 0,
            "errors": 0,
            "field_accuracy": {"sentiment": 0.0, "category": 0.0, "urgency": 0.0},
            "overall_accuracy": 0.0,
            "sentiment_score_mae": None,
            "mismatches": [],
        }

    sentiment_matches = 0
    category_matches = 0
    urgency_matches = 0
    all_matched = 0
    error_count = 0
    score_diff_sum = 0.0
    valid_scores_count = 0
    mismatches = []

    # Nominal baseline score by sentiment if ground truth score column is omitted
    default_sentiment_scores = {
        "positive": 0.8,
        "neutral": 0.0,
        "negative": -0.8,
        "mixed": 0.0,
    }

    for row in rows:
        text = row.get("feedback") or row.get("text", "")
        exp_sentiment = str(row.get("sentiment", "")).strip().lower()
        exp_category = str(row.get("category", "")).strip().lower()
        exp_urgency = str(row.get("urgency", "")).strip().lower()

        # Parse expected score if column exists
        has_score_col = "sentiment_score" in row and row["sentiment_score"] != ""
        exp_score = (
            float(row["sentiment_score"])
            if has_score_col
            else default_sentiment_scores.get(exp_sentiment, 0.0)
        )

        try:
            pred = analyzer.analyze(text)
            if not isinstance(pred, dict) or "error" in pred:
                error_count += 1
                mismatches.append({
                    "text": text,
                    "field": "analysis_error",
                    "expected": f"valid result (sentiment={exp_sentiment})",
                    "predicted": pred.get("error", "unknown error") if isinstance(pred, dict) else "error",
                })
                continue
        except Exception as exc:
            error_count += 1
            mismatches.append({
                "text": text,
                "field": "analysis_error",
                "expected": f"valid result (sentiment={exp_sentiment})",
                "predicted": f"{type(exc).__name__}: {exc}",
            })
            continue

        pred_sentiment = str(pred.get("sentiment", "")).strip().lower()
        pred_category = str(pred.get("category", "")).strip().lower()
        pred_urgency = str(pred.get("urgency", "")).strip().lower()

        sent_ok = pred_sentiment == exp_sentiment
        cat_ok = pred_category == exp_category
        urg_ok = pred_urgency == exp_urgency

        if sent_ok:
            sentiment_matches += 1
        else:
            mismatches.append({
                "text": text,
                "field": "sentiment",
                "expected": exp_sentiment,
                "predicted": pred_sentiment,
            })

        if cat_ok:
            category_matches += 1
        else:
            mismatches.append({
                "text": text,
                "field": "category",
                "expected": exp_category,
                "predicted": pred_category,
            })

        if urg_ok:
            urgency_matches += 1
        else:
            mismatches.append({
                "text": text,
                "field": "urgency",
                "expected": exp_urgency,
                "predicted": pred_urgency,
            })

        if sent_ok and cat_ok and urg_ok:
            all_matched += 1

        if "sentiment_score" in pred:
            score_diff_sum += abs(float(pred["sentiment_score"]) - exp_score)
            valid_scores_count += 1

    evaluated = total - error_count
    denom = evaluated if evaluated > 0 else 1

    field_accuracy = {
        "sentiment": round(sentiment_matches / denom, 4) if evaluated > 0 else 0.0,
        "category": round(category_matches / denom, 4) if evaluated > 0 else 0.0,
        "urgency": round(urgency_matches / denom, 4) if evaluated > 0 else 0.0,
    }
    overall_accuracy = round(all_matched / denom, 4) if evaluated > 0 else 0.0
    mae = round(score_diff_sum / valid_scores_count, 4) if valid_scores_count > 0 else None

    return {
        "total": total,
        "evaluated": evaluated,
        "errors": error_count,
        "field_accuracy": field_accuracy,
        "overall_accuracy": overall_accuracy,
        "sentiment_score_mae": mae,
        "mismatches": mismatches,
    }


def format_evaluation(result: dict) -> str:
    """Format evaluation dictionary into a markdown report."""
    total = result.get("total", 0)
    evaluated = result.get("evaluated", 0)
    errors = result.get("errors", 0)
    field_acc = result.get("field_accuracy", {})
    overall_acc = result.get("overall_accuracy", 0.0)
    mae = result.get("sentiment_score_mae")
    mismatches = result.get("mismatches", [])

    lines = [
        "# Evaluation Results",
        f"- Total Rows: {total}",
        f"- Successfully Evaluated: {evaluated}",
        f"- Failed / Errors: {errors}",
        f"- Overall Exact Match (All Fields): {overall_acc:.1%}",
        f"- Sentiment Score MAE: {mae if mae is not None else 'N/A'}",
        "",
        "## Per-Field Accuracy",
        f"- Sentiment: {field_acc.get('sentiment', 0.0):.1%}",
        f"- Category: {field_acc.get('category', 0.0):.1%}",
        f"- Urgency: {field_acc.get('urgency', 0.0):.1%}",
        "",
        f"## Mismatches ({len(mismatches)})",
    ]

    if not mismatches:
        lines.append("No mismatches found. Perfect prediction!")
    else:
        lines.append("| Field | Expected | Predicted | Text Snippet |")
        lines.append("|---|---|---|---|")
        for m in mismatches:
            snippet = m['text'].replace('\n', ' ')
            if len(snippet) > 60:
                snippet = snippet[:57] + "..."
            lines.append(f"| {m['field']} | `{m['expected']}` | `{m['predicted']}` | {snippet} |")

    return "\n".join(lines) + "\n"
