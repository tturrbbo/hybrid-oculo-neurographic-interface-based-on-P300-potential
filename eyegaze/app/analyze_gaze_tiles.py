from __future__ import annotations

import csv
import math
from pathlib import Path


def _float(v):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _rmse(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return math.sqrt(sum(x*x for x in xs) / len(xs))


def main():
    files = sorted(Path("data/logs/gaze_tiles_existing_calibration").glob("*_gaze_tiles_existing_calibration_*.csv"))
    rows = []
    for fp in files:
        with fp.open("r", encoding="utf-8", newline="") as f:
            rows.extend(list(csv.DictReader(f)))

    valid = [r for r in rows if r.get("gaze_x_px") not in ("", None)]
    hits = [_float(r.get("tile_hit")) for r in valid]
    yaw = [_float(r.get("yaw_error_deg")) for r in valid]
    pitch = [_float(r.get("pitch_error_deg")) for r in valid]

    out_dir = Path("data/results/gaze_tiles_existing_calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "gaze_tiles_existing_calibration_summary.csv"

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "files",
            "frames_total",
            "frames_valid",
            "valid_percent",
            "tile_accuracy_percent",
            "mean_yaw_error_deg",
            "rmse_yaw_error_deg",
            "mean_pitch_error_deg",
            "rmse_pitch_error_deg",
        ])
        w.writeheader()
        w.writerow({
            "files": len(files),
            "frames_total": len(rows),
            "frames_valid": len(valid),
            "valid_percent": "" if not rows else f"{100*len(valid)/len(rows):.3f}",
            "tile_accuracy_percent": "" if _mean(hits) is None else f"{100*_mean(hits):.3f}",
            "mean_yaw_error_deg": "" if _mean(yaw) is None else f"{_mean(yaw):.4f}",
            "rmse_yaw_error_deg": "" if _rmse(yaw) is None else f"{_rmse(yaw):.4f}",
            "mean_pitch_error_deg": "" if _mean(pitch) is None else f"{_mean(pitch):.4f}",
            "rmse_pitch_error_deg": "" if _rmse(pitch) is None else f"{_rmse(pitch):.4f}",
        })

    print(f"Saved summary: {out.resolve()}")


if __name__ == "__main__":
    main()
