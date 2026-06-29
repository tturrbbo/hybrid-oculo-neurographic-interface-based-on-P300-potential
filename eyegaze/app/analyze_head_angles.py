from __future__ import annotations

import csv
from pathlib import Path
import math
from collections import defaultdict


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


def _std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _rmse(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def _fmt(v):
    if v is None:
        return ""
    return f"{v:.4f}"


def main():
    root = Path("data/logs")
    out_dir = Path("data/results/head_angles")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list((root / "head_angles").glob("*.csv")) + list(root.glob("*head_angles*.csv"))

    rows = []
    for fp in files:
        try:
            with fp.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if "target_angle_deg" not in r:
                        continue
                    r["_source_file"] = str(fp)
                    rows.append(r)
        except Exception:
            continue

    if not rows:
        print("No head angle CSV files found in data/logs/head_angles or data/logs.")
        return

    summary_rows = []

    for method, err_col, valid_col, angle_col in [
        ("laser", "laser_axis_error_deg", "laser_valid", "laser_axis_angle_deg"),
        ("face_geometry", "face_geometry_axis_error_deg", "face_geometry_valid", "face_geometry_axis_angle_deg"),
    ]:
        groups = defaultdict(list)

        for r in rows:
            target_dir = r.get("target_direction", "")
            target_angle = r.get("target_angle_deg", "")
            axis = r.get("target_axis", "")

            valid = str(r.get(valid_col, "")).strip()
            err = _float(r.get(err_col))
            angle = _float(r.get(angle_col))

            groups[("ALL", "ALL", "ALL")].append((err, angle, valid))
            groups[(target_dir, target_angle, axis)].append((err, angle, valid))

        for (target_dir, target_angle, axis), vals in sorted(groups.items()):
            errors = [v[0] for v in vals if v[0] is not None]
            angles = [v[1] for v in vals if v[1] is not None]
            valid_count = sum(1 for v in vals if str(v[2]) in ("1", "True", "true"))
            n = len(vals)

            summary_rows.append({
                "method": method,
                "target_direction": target_dir,
                "target_angle_deg": target_angle,
                "target_axis": axis,
                "frames_total": n,
                "frames_valid": valid_count,
                "valid_percent": _fmt(100.0 * valid_count / n if n else None),
                "mae_axis_deg": _fmt(_mean(errors)),
                "rmse_axis_deg": _fmt(_rmse(errors)),
                "std_axis_error_deg": _fmt(_std(errors)),
                "max_axis_error_deg": _fmt(max(errors) if errors else None),
                "mean_measured_axis_deg": _fmt(_mean(angles)),
                "std_measured_axis_deg": _fmt(_std(angles)),
            })

    summary_path = out_dir / "head_angle_comparison_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "method",
            "target_direction",
            "target_angle_deg",
            "target_axis",
            "frames_total",
            "frames_valid",
            "valid_percent",
            "mae_axis_deg",
            "rmse_axis_deg",
            "std_axis_error_deg",
            "max_axis_error_deg",
            "mean_measured_axis_deg",
            "std_measured_axis_deg",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"Loaded frames: {len(rows)}")
    print(f"Saved summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
