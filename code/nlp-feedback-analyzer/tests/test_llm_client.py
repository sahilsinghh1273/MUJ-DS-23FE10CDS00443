"""Tests for LLMClient: retries, backoff delays, and error classifications."""
from unittest.mock import MagicMock
import pytest
import openai

from src.llm_client import LLMClient
from tests.conftest import (
    make_auth_error,
    make_connection_error,
    make_rate_limit_error,
    make_status_error,
    make_timeout_error,
)


def make_mock_completion(text: str):
    """Helper to build a mock OpenAI chat completion object."""
    mock_choice = MagicMock()
    mock_choice.message.content = text
    mock_choice.message.role = "assistant"
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    return mock_resp


@pytest.fixture
def mock_client_factory(fake_cfg, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "dummy_xai_key")

    def _factory(max_retries=4, backoff=1.0):
        cfg = dict(fake_cfg)
        cfg["llm"] = dict(fake_cfg["llm"], max_retries=max_retries, backoff_base_seconds=backoff)
        slept_times = []

        def fake_sleep(seconds):
            slept_times.append(seconds)

        client = LLMClient(cfg, sleep_fn=fake_sleep)
        mock_create = MagicMock()
        client._client.chat.completions.create = mock_create
        return client, mock_create, slept_times

    return _factory


def test_missing_api_key_raises_runtime_error(fake_cfg, monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="XAI_API_KEY not set"):
        LLMClient(fake_cfg)


def test_retry_on_rate_limit_error(mock_client_factory):
    client, mock_create, slept = mock_client_factory()
    mock_create.side_effect = [make_rate_limit_error(), make_mock_completion("Success after rate limit")]

    out = client.complete("system", "user")
    assert out == "Success after rate limit"
    assert mock_create.call_count == 2
    assert slept == [1.0]


def test_retry_on_connection_error(mock_client_factory):
    client, mock_create, slept = mock_client_factory()
    mock_create.side_effect = [make_connection_error(), make_mock_completion("Success after network drop")]

    out = client.complete("system", "user")
    assert out == "Success after network drop"
    assert mock_create.call_count == 2
    assert slept == [1.0]


def test_retry_on_timeout_error(mock_client_factory):
    client, mock_create, slept = mock_client_factory()
    mock_create.side_effect = [make_timeout_error(), make_mock_completion("Success after timeout")]

    out = client.complete("system", "user")
    assert out == "Success after timeout"
    assert mock_create.call_count == 2
    assert slept == [1.0]


def test_retry_on_5xx_status_error(mock_client_factory):
    client, mock_create, slept = mock_client_factory()
    mock_create.side_effect = [make_status_error(500), make_status_error(503), make_mock_completion("Success")]

    out = client.complete("system", "user")
    assert out == "Success"
    assert mock_create.call_count == 3
    assert slept == [1.0, 2.0]


def test_no_retry_on_authentication_error(mock_client_factory):
    client, mock_create, slept = mock_client_factory()
    mock_create.side_effect = make_auth_error()

    with pytest.raises(openai.AuthenticationError):
        client.complete("system", "user")
    assert mock_create.call_count == 1
    assert slept == []


def test_no_retry_on_4xx_status_error(mock_client_factory):
    client, mock_create, slept = mock_client_factory()
    mock_create.side_effect = make_status_error(400, "Bad Request")

    with pytest.raises(openai.APIStatusError):
        client.complete("system", "user")
    assert mock_create.call_count == 1
    assert slept == []


def test_exponential_backoff_delays_applied(mock_client_factory):
    client, mock_create, slept = mock_client_factory(max_retries=3, backoff=0.5)
    mock_create.side_effect = [
        make_rate_limit_error(),
        make_connection_error(),
        make_status_error(502),
        make_mock_completion("Success"),
    ]

    out = client.complete("system", "user")
    assert out == "Success"
    assert mock_create.call_count == 4
    # Delays: 0.5 * 2^0 = 0.5, 0.5 * 2^1 = 1.0, 0.5 * 2^2 = 2.0
    assert slept == [0.5, 1.0, 2.0]


def test_gives_up_after_max_retries(mock_client_factory):
    client, mock_create, slept = mock_client_factory(max_retries=2, backoff=1.0)
    mock_create.side_effect = [
        make_rate_limit_error(),
        make_rate_limit_error(),
        make_rate_limit_error(),
    ]

    with pytest.raises(openai.RateLimitError):
        client.complete("system", "user")
    assert mock_create.call_count == 3  # initial + 2 retries
    assert slept == [1.0, 2.0]
