from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _project_root() -> Path:
    # file: eyegaze/app/hybrid_gaze_p300_manual_runner.py
    return Path(__file__).resolve().parents[2]


def _open_p300_window(root: Path, *, no_p300: bool) -> subprocess.Popen | None:
    if no_p300:
        return None

    script = root / "scripts" / "p300_analyzer.py"
    if not script.exists():
        raise FileNotFoundError(
            f"Не найден {script}. Положи scripts/p300_analyzer.py и папку p300_analysis рядом с eyegaze."
        )

    return subprocess.Popen([sys.executable, str(script)], cwd=str(root))


def parse_args():
    p = argparse.ArgumentParser(
        description="Ручной гибридный запуск: P300 Analyzer + наши плитки с выбором цели/раундов."
    )
    p.add_argument("--participant", required=True)
    p.add_argument("--config", default="config/experiment.yaml")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--no-p300", action="store_true", help="не запускать окно P300 Analyzer автоматически")
    p.add_argument(
        "--keep-p300",
        action="store_true",
        help="не закрывать P300 Analyzer при выходе из окна плиток",
    )
    return p.parse_args()


def main():
    args = parse_args()
    root = _project_root()

    p300_proc = _open_p300_window(root, no_p300=bool(args.no_p300))

    if p300_proc is not None:
        print("[hybrid-manual] P300 Analyzer запущен отдельным окном.")
        print("[hybrid-manual] В P300 Analyzer нажми: 🔄 -> выбрать EEG -> Подключиться к LSL -> Начать анализ.")
        print("[hybrid-manual] Потом в окне плиток выбери цель/раунды и нажми START.")
        time.sleep(1.0)

    cmd = [
        sys.executable,
        "-m",
        "eyegaze.app.gaze_tiles_test",
        "--participant",
        str(args.participant),
        "--config",
        str(args.config),
        "--trials",
        str(int(args.trials)),
    ]

    # ВАЖНО: специально НЕ добавляем --auto.
    # Тогда окно плиток открывается как раньше: цель 0-8, раунды, START/STOP.
    print("[hybrid-manual] Запуск окна плиток в ручном режиме:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(root))

    if p300_proc is not None and not args.keep_p300:
        try:
            p300_proc.terminate()
        except Exception:
            pass

    raise SystemExit(rc)


if __name__ == "__main__":
    main()
