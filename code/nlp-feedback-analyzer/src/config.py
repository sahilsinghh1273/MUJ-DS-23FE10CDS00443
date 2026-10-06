"""Load YAML config and prompt files."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    return _load_yaml(Path(path))


def load_prompts(path: str | Path | None = None, cfg: dict | None = None) -> dict:
    if path is None:
        rel = (cfg or load_config())["paths"]["prompts"]
        path = ROOT / rel
    return _load_yaml(Path(path))
