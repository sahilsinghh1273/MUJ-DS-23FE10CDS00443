"""Validation of the LLM's structured output."""

SENTIMENTS = {"positive", "neutral", "negative", "mixed"}
CATEGORIES = {
    "billing",
    "delivery",
    "product_quality",
    "customer_support",
    "technical_issue",
    "feature_request",
    "other",
}
URGENCIES = {"low", "medium", "high"}
ENTITY_TYPES = {"PRODUCT", "ORG", "PERSON", "LOCATION", "DATE", "MONEY", "ORDER_ID"}

MAX_SUMMARY_WORDS = 25
MAX_REPLY_WORDS = 70


class ValidationError(ValueError):
    """Raised when structured LLM output fails schema validation."""
    pass


def _enum(data: dict, key: str, allowed: set) -> str:
    value = str(data.get(key, "")).strip().lower()
    if value not in allowed:
        raise ValidationError(f"'{key}' must be one of {sorted(allowed)}, got {value!r}")
    return value


def validate(data: dict) -> dict:
    """Validate and normalise a parsed LLM response. Raises ValidationError."""
    if not isinstance(data, dict):
        raise ValidationError("Top-level JSON must be an object")

    try:
        score = float(data.get("sentiment_score"))
    except (TypeError, ValueError):
        raise ValidationError("'sentiment_score' must be a number")

    entities = []
    for e in data.get("entities") or []:
        if isinstance(e, dict) and e.get("text"):
            etype = str(e.get("type", "")).strip().upper()
            entities.append({
                "text": str(e["text"]).strip(),
                "type": etype if etype in ENTITY_TYPES else "OTHER",
            })

    summary = str(data.get("summary", "")).strip()
    if not summary:
        raise ValidationError("'summary' is required")
    summary_words = len(summary.split())
    if summary_words > MAX_SUMMARY_WORDS:
        raise ValidationError(
            f"'summary' exceeds max word limit of {MAX_SUMMARY_WORDS} words (got {summary_words})"
        )

    suggested_reply = str(data.get("suggested_reply", "")).strip()
    if suggested_reply:
        reply_words = len(suggested_reply.split())
        if reply_words > MAX_REPLY_WORDS:
            raise ValidationError(
                f"'suggested_reply' exceeds max word limit of {MAX_REPLY_WORDS} words (got {reply_words})"
            )

    return {
        "sentiment": _enum(data, "sentiment", SENTIMENTS),
        "sentiment_score": max(-1.0, min(1.0, score)),
        "category": _enum(data, "category", CATEGORIES),
        "urgency": _enum(data, "urgency", URGENCIES),
        "entities": entities,
        "key_issues": [str(x).strip() for x in (data.get("key_issues") or []) if str(x).strip()][:3],
        "summary": summary,
        "suggested_reply": suggested_reply,
    }
