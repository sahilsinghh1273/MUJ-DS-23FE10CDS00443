"""Shared pytest fixtures, fake clients, and mock data."""
import json
from pathlib import Path
import pytest
import httpx
import openai

from src.analyzer import FeedbackAnalyzer

VALID_RESULT = {
    "sentiment": "negative",
    "sentiment_score": -0.8,
    "category": "billing",
    "urgency": "high",
    "entities": [{"text": "Rs. 499", "type": "MONEY"}],
    "key_issues": ["double charge"],
    "summary": "Customer was charged twice.",
    "suggested_reply": "Sorry about that, we are looking into it.",
}


class FakeClient:
    """Mock LLM client that returns pre-scripted responses or raises exceptions."""

    def __init__(self, replies: list | None = None):
        self.replies = list(replies) if replies is not None else []
        self.calls = 0
        self.history: list[dict] = []

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        self.history.append({"system": system, "user": user})
        if not self.replies:
            return json.dumps(VALID_RESULT)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def sample_result() -> dict:
    return dict(VALID_RESULT)


@pytest.fixture
def fake_cfg(tmp_path: Path) -> dict:
    return {
        "llm": {
            "provider": "xai",
            "base_url": "https://api.x.ai/v1",
            "model": "grok-beta",
            "max_tokens": 700,
            "temperature": 0.0,
            "timeout_seconds": 30,
            "max_retries": 4,
            "backoff_base_seconds": 1.0,
        },
        "pipeline": {
            "max_workers": 4,
            "text_column": "feedback",
            "max_input_chars": 2000,
            "repair_attempts": 1,
        },
        "cache": {
            "enabled": True,
            "path": str(tmp_path / "cache.json"),
        },
        "paths": {
            "prompts": "prompts.yaml",
            "output_dir": str(tmp_path / "output"),
        },
    }


@pytest.fixture
def fake_prompts() -> dict:
    return {
        "version": "1.0",
        "system": "System instructions for analysis.",
        "user_template": "Analyze: {feedback}",
        "repair_template": "Fix error: {error} in {previous}",
    }


@pytest.fixture
def make_analyzer(fake_cfg, fake_prompts):
    def _factory(replies: list | None = None, cfg_override: dict | None = None, prompts_override: dict | None = None):
        cfg = cfg_override or fake_cfg
        prompts = prompts_override or fake_prompts
        client = FakeClient(replies)
        analyzer = FeedbackAnalyzer(client, prompts, cfg)
        return analyzer, client
    return _factory


# Helpers for creating official OpenAI exceptions
def make_rate_limit_error(msg="rate limit exceeded"):
    req = httpx.Request("POST", "https://api.x.ai/v1/chat/completions")
    return openai.RateLimitError(msg, response=httpx.Response(429, request=req), body=None)


def make_connection_error(msg="connection error"):
    req = httpx.Request("POST", "https://api.x.ai/v1/chat/completions")
    return openai.APIConnectionError(request=req)


def make_timeout_error(msg="request timed out"):
    req = httpx.Request("POST", "https://api.x.ai/v1/chat/completions")
    return openai.APITimeoutError(request=req)


def make_auth_error(msg="invalid api key"):
    req = httpx.Request("POST", "https://api.x.ai/v1/chat/completions")
    return openai.AuthenticationError(msg, response=httpx.Response(401, request=req), body=None)


def make_status_error(status_code: int, msg="api status error"):
    req = httpx.Request("POST", "https://api.x.ai/v1/chat/completions")
    return openai.APIStatusError(msg, response=httpx.Response(status_code, request=req), body=None)
