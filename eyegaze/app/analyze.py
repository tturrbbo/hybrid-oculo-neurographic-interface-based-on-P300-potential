from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logs-dir", default="data/logs")
    p.add_argument("--output-dir", default="data/results")
    p.add_argument("--pattern", default="*.csv")
    return p.parse_args()


def main():
    args = parse_args()
    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(logs_dir.glob(args.pattern))
    if not files:
        print(f"No CSV logs found in {logs_dir}")
        return

    dfs = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            if {"participant_id", "head_target_label", "hit"}.issubset(df.columns):
                df["source_file"] = fp.name
                dfs.append(df)
            else:
                print(f"SKIP {fp.name}: not experiment log")
        except Exception as e:
            print(f"SKIP {fp.name}: {e}")

    if not dfs:
        print("No valid logs.")
        return

    data = pd.concat(dfs, ignore_index=True)
    for col in ["hit", "head_target_angle_deg", "measured_yaw_deg", "measured_pitch_deg", "distance_cm"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    by_participant = (
        data.groupby(["participant_id", "head_target_label"], as_index=False)
        .agg(
            frames=("hit", "count"),
            hits=("hit", "sum"),
            hit_rate_percent=("hit", lambda s: round(float(s.mean() * 100), 2)),
            mean_yaw_deg=("measured_yaw_deg", "mean"),
            mean_pitch_deg=("measured_pitch_deg", "mean"),
            mean_distance_cm=("distance_cm", "mean"),
        )
    )
    by_participant.to_csv(out_dir / "summary_by_participant_and_angle.csv", index=False, encoding="utf-8-sig")

    by_angle = (
        data.groupby(["head_target_label"], as_index=False)
        .agg(
            participants=("participant_id", "nunique"),
            frames=("hit", "count"),
            hits=("hit", "sum"),
            hit_rate_percent=("hit", lambda s: round(float(s.mean() * 100), 2)),
            mean_yaw_deg=("measured_yaw_deg", "mean"),
            mean_pitch_deg=("measured_pitch_deg", "mean"),
            mean_distance_cm=("distance_cm", "mean"),
        )
    )
    by_angle.to_csv(out_dir / "summary_by_head_angle.csv", index=False, encoding="utf-8-sig")

    by_cell = (
        data.groupby(["head_target_label", "stim_cell"], as_index=False)
        .agg(
            frames=("hit", "count"),
            hits=("hit", "sum"),
            hit_rate_percent=("hit", lambda s: round(float(s.mean() * 100), 2)),
        )
    )
    by_cell.to_csv(out_dir / "summary_by_angle_and_tile.csv", index=False, encoding="utf-8-sig")

    overall = pd.DataFrame([{
        "participants": data["participant_id"].nunique(),
        "files": data["source_file"].nunique(),
        "frames": len(data),
        "hits": int(data["hit"].fillna(0).sum()),
        "hit_rate_percent": round(float(data["hit"].fillna(0).mean() * 100), 2),
        "mean_distance_cm": data["distance_cm"].mean(),
    }])
    overall.to_csv(out_dir / "summary_overall.csv", index=False, encoding="utf-8-sig")

    print("\n=== SUMMARY BY HEAD ANGLE ===")
    print(by_angle.to_string(index=False))
    print(f"\nSaved results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
