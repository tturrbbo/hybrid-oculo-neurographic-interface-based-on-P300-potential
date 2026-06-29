from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "external_arduino_lcard_reader" / "control" / "main.py"

if not APP.exists():
    raise FileNotFoundError(f"Не найден {APP}")

runpy.run_path(str(APP), run_name="__main__")
