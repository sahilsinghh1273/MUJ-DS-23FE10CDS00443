"""Tests for JSON extraction from model outputs."""
import pytest
from src.analyzer import extract_json
from src.schema import ValidationError


def test_extract_plain_json():
    raw = '{"sentiment": "positive", "sentiment_score": 0.9}'
    assert extract_json(raw) == {"sentiment": "positive", "sentiment_score": 0.9}


def test_extract_json_with_code_fences():
    raw = """```json
{
  "sentiment": "negative",
  "category": "billing"
}
```"""
    parsed = extract_json(raw)
    assert parsed["sentiment"] == "negative"
    assert parsed["category"] == "billing"


def test_extract_json_with_generic_code_fences():
    raw = """```
{
  "sentiment": "neutral",
  "urgency": "low"
}
```"""
    parsed = extract_json(raw)
    assert parsed == {"sentiment": "neutral", "urgency": "low"}


def test_extract_json_with_extra_prose():
    raw = """Here is the structured analysis of the customer's comment:

{
  "sentiment": "mixed",
  "category": "delivery"
}

I hope this helps! Let me know if you need anything else."""
    parsed = extract_json(raw)
    assert parsed["sentiment"] == "mixed"
    assert parsed["category"] == "delivery"


def test_extract_json_with_prose_and_fences():
    raw = """Certainly! Here is the JSON output:
```json
{
  "sentiment": "positive",
  "summary": "Great customer service."
}
```
Thank you!"""
    parsed = extract_json(raw)
    assert parsed["sentiment"] == "positive"
    assert parsed["summary"] == "Great customer service."


def test_extract_json_invalid_no_object():
    with pytest.raises(ValidationError, match="No JSON object found"):
        extract_json("There is no structured data anywhere in this text.")


def test_extract_json_invalid_malformed():
    with pytest.raises(ValidationError, match="Malformed JSON"):
        extract_json("Here is the JSON: {bad_key: 'unquoted_value', broken}")


def test_extract_json_empty():
    with pytest.raises(ValidationError, match="No JSON object found"):
        extract_json("")
