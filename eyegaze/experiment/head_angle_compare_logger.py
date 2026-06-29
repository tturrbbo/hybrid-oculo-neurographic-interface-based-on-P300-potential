from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class HeadAngleCompareLogger:
    """Small per-frame logger for comparing two head-angle sources.

    It writes one CSV per method, so the same experiment can be analyzed as:
    - laser
    - face_geometry / no_laser
    """

    FIELDS = [
        "participant_id",
        "method",
        "timestamp_unix",
        "head_target_label",
        "head_target_direction",
        "head_target_angle_deg",
        "head_marker",
        "yaw_deg",
        "pitch_deg",
        "valid",
        "distance_cm",
        "eye_distance_px",
        "laser_x_px",
        "laser_y_px",
        "laser_confidence",
        "laser_reason",
        "trial_id",
        "stim_row",
        "stim_col",
        "stim_cell",
        "gaze_row",
        "gaze_col",
        "gaze_cell",
        "hit",
        "blink",
    ]

    def __init__(self, output_path: str | Path, method: str):
        self.path = Path(output_path).resolve()
        self.method = method
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.file.flush()

    def log(self, **row: Any) -> None:
        clean = {field: row.get(field, "") for field in self.FIELDS}
        clean["method"] = self.method
        self.writer.writerow(clean)
        self.file.flush()

    def close(self) -> None:
        try:
            self.file.flush()
            self.file.close()
        except Exception:
            pass
