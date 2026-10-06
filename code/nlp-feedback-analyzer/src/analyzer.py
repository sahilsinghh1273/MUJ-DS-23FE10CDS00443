"""Core NLP pipeline: prompt -> LLM -> parse -> validate -> cache."""
import hashlib
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .schema import ValidationError, validate

log = logging.getLogger(__name__)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class ResultCache:
    """Tiny thread-safe JSON file cache keyed by hash(model + prompt version + text)."""

    def __init__(self, path: str, enabled: bool = True):
        self.enabled = enabled
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: dict = {}
        if enabled and self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def get(self, key: str):
        return self._data.get(key) if self.enabled else None

    def set(self, key: str, value: dict):
        if not self.enabled:
            return
        with self._lock:
            self._data[key] = value

    def flush(self):
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")


def extract_json(text: str) -> dict:
    """Parse JSON from a model reply, tolerating code fences or stray prose."""
    text = (text or "").strip()
    # Strip markdown code fences if wrapped
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        unfenced = "\n".join(lines).strip()
        try:
            return json.loads(unfenced)
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_RE.search(text)
        if not match:
            raise ValidationError("No JSON object found in model output")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise ValidationError(f"Malformed JSON: {e}")


class FeedbackAnalyzer:
    def __init__(self, client, prompts: dict, cfg: dict):
        self.client = client
        self.prompts = prompts
        self.pipe = cfg["pipeline"]
        self.cache = ResultCache(cfg["cache"]["path"], cfg["cache"]["enabled"])
        self.model = cfg["llm"]["model"]

    def _key(self, text: str) -> str:
        raw = f"{self.model}|{self.prompts['version']}|{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def analyze(self, text: str) -> dict:
        text = (text or "").strip()[: self.pipe["max_input_chars"]]
        if not text:
            return {"error": "empty input"}

        key = self._key(text)
        cached = self.cache.get(key)
        if cached:
            return {**cached, "cached": True}

        user_prompt = self.prompts["user_template"].replace("{feedback}", text)
        output = self.client.complete(self.prompts["system"], user_prompt)

        attempts = self.pipe["repair_attempts"]
        for attempt in range(attempts + 1):
            try:
                result = validate(extract_json(output))
                break
            except ValidationError as err:
                if attempt == attempts:
                    log.error("Giving up after %d repair attempt(s): %s", attempts, err)
                    raise err
                repair = (
                    self.prompts["repair_template"]
                    .replace("{error}", str(err))
                    .replace("{previous}", output)
                )
                output = self.client.complete(self.prompts["system"], user_prompt + "\n\n" + repair)

        self.cache.set(key, result)
        return {**result, "cached": False}

    def analyze_batch(self, texts: list[str], on_progress=None) -> list[dict]:
        """Analyze many texts in parallel; output order matches input order.

        on_progress(done, total) is called as results are collected (used by the Streamlit UI).
        """
        with ThreadPoolExecutor(max_workers=self.pipe["max_workers"]) as pool:
            futures = [pool.submit(self._safe_analyze, t) for t in texts]
            results = []
            for i, fut in enumerate(futures, 1):
                results.append(fut.result())
                if on_progress:
                    on_progress(i, len(futures))
        self.cache.flush()
        return results

    def _safe_analyze(self, text: str) -> dict:
        try:
            return self.analyze(text)
        except Exception as exc:  # one bad row must not kill the batch
            log.error("Analysis failed: %s", exc)
            return {"error": f"{type(exc).__name__}: {exc}"}
