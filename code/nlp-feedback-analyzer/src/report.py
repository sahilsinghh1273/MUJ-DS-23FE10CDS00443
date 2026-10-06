"""Aggregate batch results into a readable summary."""
from collections import Counter


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    n = len(ok)
    issues = Counter(i.lower() for r in ok for i in r["key_issues"])
    return {
        "total": len(rows),
        "analyzed": n,
        "failed": len(rows) - n,
        "avg_sentiment": round(sum(r["sentiment_score"] for r in ok) / n, 3) if n else None,
        "by_sentiment": dict(Counter(r["sentiment"] for r in ok)),
        "by_category": dict(Counter(r["category"] for r in ok)),
        "by_urgency": dict(Counter(r["urgency"] for r in ok)),
        "top_issues": issues.most_common(5),
        "high_urgency_count": sum(r["urgency"] == "high" for r in ok),
    }


def to_markdown(stats: dict) -> str:
    def block(title, d):
        return "\n".join([f"### {title}"] + [f"- {k}: {v}" for k, v in d.items()])

    parts = [
        "# Feedback Analysis Report",
        f"- Total: {stats['total']} | Analyzed: {stats['analyzed']} | Failed: {stats['failed']}",
        f"- Average sentiment score: {stats['avg_sentiment']}",
        f"- High-urgency items: {stats['high_urgency_count']}",
        block("Sentiment", stats["by_sentiment"]),
        block("Category", stats["by_category"]),
        block("Urgency", stats["by_urgency"]),
        block("Top issues", dict(stats["top_issues"])),
    ]
    return "\n\n".join(parts) + "\n"
