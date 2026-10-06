"""Tests for schema validation of LLM outputs."""
import pytest
from src.schema import ValidationError, validate


def test_validate_valid_input(sample_result):
    validated = validate(sample_result)
    assert validated["sentiment"] == "negative"
    assert validated["sentiment_score"] == -0.8
    assert validated["category"] == "billing"
    assert validated["urgency"] == "high"
    assert len(validated["entities"]) == 1
    assert validated["entities"][0] == {"text": "Rs. 499", "type": "MONEY"}
    assert validated["key_issues"] == ["double charge"]
    assert validated["summary"] == "Customer was charged twice."


def test_validate_rejects_non_dict():
    with pytest.raises(ValidationError, match="Top-level JSON must be an object"):
        validate(["not", "a", "dict"])


def test_validate_bad_sentiment(sample_result):
    with pytest.raises(ValidationError, match="'sentiment' must be one of"):
        validate({**sample_result, "sentiment": "super_happy"})


def test_validate_bad_category(sample_result):
    with pytest.raises(ValidationError, match="'category' must be one of"):
        validate({**sample_result, "category": "astrology"})


def test_validate_bad_urgency(sample_result):
    with pytest.raises(ValidationError, match="'urgency' must be one of"):
        validate({**sample_result, "urgency": "extreme"})


def test_validate_clamps_score_upper(sample_result):
    val = validate({**sample_result, "sentiment_score": 4.5})
    assert val["sentiment_score"] == 1.0


def test_validate_clamps_score_lower(sample_result):
    val = validate({**sample_result, "sentiment_score": -99.0})
    assert val["sentiment_score"] == -1.0


def test_validate_non_numeric_score(sample_result):
    with pytest.raises(ValidationError, match="'sentiment_score' must be a number"):
        validate({**sample_result, "sentiment_score": "bad_score"})


def test_validate_summary_word_limit(sample_result):
    long_summary = " ".join(["word"] * 35)
    with pytest.raises(ValidationError, match="exceeds max word limit"):
        validate({**sample_result, "summary": long_summary})


def test_validate_reply_word_limit(sample_result):
    long_reply = " ".join(["reply"] * 85)
    with pytest.raises(ValidationError, match="'suggested_reply' exceeds max word limit"):
        validate({**sample_result, "suggested_reply": long_reply})


def test_validate_empty_summary(sample_result):
    with pytest.raises(ValidationError, match="'summary' is required"):
        validate({**sample_result, "summary": "   "})


def test_validate_entity_cleaning(sample_result):
    raw_entities = [
        {"text": "iPhone 15", "type": "product"},
        {"text": "Apple Store", "type": "ORG"},
        {"text": "unknown thing", "type": "ALIEN_TYPE"},
        {"text": ""},  # ignored (empty text)
        "not a dict",   # ignored
    ]
    val = validate({**sample_result, "entities": raw_entities})
    assert len(val["entities"]) == 3
    assert val["entities"][0] == {"text": "iPhone 15", "type": "PRODUCT"}
    assert val["entities"][1] == {"text": "Apple Store", "type": "ORG"}
    assert val["entities"][2] == {"text": "unknown thing", "type": "OTHER"}


def test_validate_key_issues_clamped_to_three(sample_result):
    val = validate({**sample_result, "key_issues": ["one", "two", "three", "four", "five"]})
    assert val["key_issues"] == ["one", "two", "three"]
