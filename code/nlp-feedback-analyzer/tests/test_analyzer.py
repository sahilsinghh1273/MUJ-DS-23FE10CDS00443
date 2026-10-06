"""Tests for FeedbackAnalyzer pipeline, repairs, caching, and batch processing."""
import json
import pytest
from src.schema import ValidationError


def test_repair_retry_fails_once_then_succeeds(make_analyzer, sample_result):
    analyzer, client = make_analyzer(["invalid json {", json.dumps(sample_result)])
    result = analyzer.analyze("App crashed when opening payment screen")
    assert result["sentiment"] == "negative"
    assert result["category"] == "billing"
    assert result["cached"] is False
    assert client.calls == 2
    # Verify the second call received the repair prompt
    assert "Fix error:" in client.history[1]["user"]


def test_repair_retry_fails_twice_then_raises(make_analyzer):
    analyzer, client = make_analyzer(["broken json 1", "still broken json 2"])
    with pytest.raises(ValidationError):
        analyzer.analyze("App crashed")
    assert client.calls == 2


def test_cache_hit_and_miss(make_analyzer, sample_result):
    analyzer, client = make_analyzer([json.dumps(sample_result)])
    text = "Great support from customer service"
    
    # First call: cache miss
    first = analyzer.analyze(text)
    assert first["cached"] is False
    assert client.calls == 1

    # Second call: cache hit
    second = analyzer.analyze(text)
    assert second["cached"] is True
    assert client.calls == 1
    assert second["sentiment"] == first["sentiment"]


def test_cache_key_changes_with_model(make_analyzer, fake_cfg, fake_prompts):
    cfg1 = dict(fake_cfg)
    cfg1["llm"] = dict(fake_cfg["llm"], model="grok-beta")
    cfg2 = dict(fake_cfg)
    cfg2["llm"] = dict(fake_cfg["llm"], model="grok-2-latest")

    a1, _ = make_analyzer(cfg_override=cfg1)
    a2, _ = make_analyzer(cfg_override=cfg2)

    text = "Same customer comment"
    assert a1._key(text) != a2._key(text)


def test_cache_key_changes_with_prompt_version(make_analyzer, fake_prompts):
    p1 = dict(fake_prompts, version="1.0")
    p2 = dict(fake_prompts, version="2.0")

    a1, _ = make_analyzer(prompts_override=p1)
    a2, _ = make_analyzer(prompts_override=p2)

    text = "Same customer comment"
    assert a1._key(text) != a2._key(text)


def test_batch_keeps_input_order(make_analyzer, sample_result):
    def make_res(cat):
        return json.dumps({**sample_result, "category": cat})

    replies = [make_res("billing"), make_res("delivery"), make_res("technical_issue")]
    analyzer, _ = make_analyzer(replies)

    inputs = ["billing issue", "delivery delayed", "system bug"]
    results = analyzer.analyze_batch(inputs)

    assert len(results) == 3
    assert results[0]["category"] == "billing"
    assert results[1]["category"] == "delivery"
    assert results[2]["category"] == "technical_issue"


def test_batch_one_failed_row_does_not_stop_batch(make_analyzer, sample_result):
    good_reply = json.dumps(sample_result)
    # Row 1 succeeds, Row 2 fails both attempts, Row 3 succeeds
    replies = [good_reply, "bad json", "still bad json", good_reply]
    analyzer, _ = make_analyzer(replies)

    inputs = ["Valid text 1", "Failing text 2", "Valid text 3"]
    results = analyzer.analyze_batch(inputs)

    assert len(results) == 3
    assert "sentiment" in results[0]
    assert "error" in results[1]
    assert "sentiment" in results[2]


def test_batch_progress_callback(make_analyzer, sample_result):
    analyzer, _ = make_analyzer([json.dumps(sample_result)] * 3)

    progress_calls = []

    def on_progress(done, total):
        progress_calls.append((done, total))

    results = analyzer.analyze_batch(["one", "two", "three"], on_progress=on_progress)
    assert len(results) == 3
    assert progress_calls == [(1, 3), (2, 3), (3, 3)]
