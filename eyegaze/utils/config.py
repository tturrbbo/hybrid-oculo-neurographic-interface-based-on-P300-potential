from __future__ import annotations

from pathlib import Path
import yaml


def project_root() -> Path:
    return Path.cwd()


def load_config(path: str | Path = "config/experiment.yaml") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
