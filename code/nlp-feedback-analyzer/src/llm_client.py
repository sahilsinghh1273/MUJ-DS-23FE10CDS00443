"""xAI Grok / OpenAI-compatible API wrapper with retry + exponential backoff."""
import logging
import os
import time

import openai

log = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, cfg: dict, sleep_fn=time.sleep):
        llm = cfg["llm"]
        key = os.getenv("XAI_API_KEY")
        if not key:
            raise RuntimeError("XAI_API_KEY not set. Copy .env.example to .env and add your key.")
        self.model = llm["model"]
        self.base_url = llm.get("base_url", "https://api.x.ai/v1")
        self.max_tokens = llm.get("max_tokens", 700)
        self.temperature = llm.get("temperature", 0.0)
        self.max_retries = llm.get("max_retries", 4)
        self.backoff = llm.get("backoff_base_seconds", 1.0)
        self.timeout = llm.get("timeout_seconds", 30)
        self._sleep = sleep_fn
        self._client = openai.OpenAI(
            api_key=key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
        )

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        if isinstance(exc, openai.AuthenticationError):
            return False
        if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)):
            return True
        if isinstance(exc, openai.APIStatusError):
            return exc.status_code is not None and exc.status_code >= 500
        return False

    def complete(self, system: str, user: str) -> str:
        """Send one prompt with system and user messages, returning the reply text."""
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                choice = resp.choices[0] if resp.choices else None
                if not choice or not choice.message:
                    return ""
                return choice.message.content or ""
            except Exception as exc:
                if attempt == self.max_retries or not self._retryable(exc):
                    raise
                delay = self.backoff * (2 ** attempt)
                log.warning(
                    "API error (%s: %s); retry %d/%d in %.1fs",
                    type(exc).__name__,
                    exc,
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                self._sleep(delay)
