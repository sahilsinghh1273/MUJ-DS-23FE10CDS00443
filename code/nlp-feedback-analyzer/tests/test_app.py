"""Smoke test: the Streamlit app renders in demo mode without errors."""
from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_app_runs_in_demo_mode(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=30).run()
    assert not at.exception
    assert any("Feedback items" == m.label for m in at.metric)
