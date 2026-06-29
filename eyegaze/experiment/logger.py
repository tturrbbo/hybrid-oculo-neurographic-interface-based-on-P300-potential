from __future__ import annotations

import csv
from pathlib import Path


class ExperimentLogger:
    """
    CSV logger for the experiment.

    This header includes both head-angle methods at once:
      - theoretical target angle
      - laser result
      - face_geometry result
      - errors for both methods
    """

    FIELDNAMES = [
        "participant_id",
        "timestamp_unix",

        # theoretical / target head angle
        "head_target_label",
        "head_target_direction",
        "head_target_angle_deg",
        "head_target_yaw_deg",
        "head_target_pitch_deg",
        "head_target_axis",
        "head_marker",

        # old gaze/head metadata kept for compatibility
        "head_deviation_direction",
        "head_deviation_angle_deg",
        "head_deviation_text_ru",
        "dominant_eye",
        "left_eye_weight",
        "right_eye_weight",
        "left_eye_calibration_error_px",
        "right_eye_calibration_error_px",
        "measured_yaw_deg",
        "measured_pitch_deg",
        "measured_roll_deg",
        "distance_cm",
        "eye_distance_px",
        "distance_status",

        # LASER method
        "laser_valid",
        "laser_yaw_deg",
        "laser_pitch_deg",
        "laser_axis_angle_deg",
        "laser_axis_error_deg",
        "laser_yaw_error_deg",
        "laser_pitch_error_deg",
        "laser_x_px",
        "laser_y_px",
        "laser_x_wall_cm",
        "laser_y_wall_cm",
        "laser_dx_px",
        "laser_dy_px",
        "laser_dx_cm",
        "laser_dy_cm",
        "laser_area_px",
        "laser_confidence",
        "laser_reason",
        "laser_method",

        # FACE GEOMETRY method, without laser
        "face_geometry_valid",
        "face_geometry_yaw_deg",
        "face_geometry_pitch_deg",
        "face_geometry_axis_angle_deg",
        "face_geometry_axis_error_deg",
        "face_geometry_yaw_error_deg",
        "face_geometry_pitch_error_deg",
        "face_geometry_distance_cm",
        "face_geometry_ref_px",
        "face_geometry_dx_px",
        "face_geometry_dy_px",
        "face_geometry_dx_cm",
        "face_geometry_dy_cm",
        "face_geometry_reason",

        # tile/gaze columns kept for old analysis compatibility
        "trial_id",
        "stim_row",
        "stim_col",
        "stim_cell",
        "gaze_x",
        "gaze_y",
        "gaze_row",
        "gaze_col",
        "gaze_cell",
        "hit",
        "blink",
    ]

    def __init__(self, output_path: str | Path):
        self.path = Path(output_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDNAMES, extrasaction="ignore")
        self.writer.writeheader()
        self.file.flush()

    def log(self, **row):
        self.writer.writerow({k: row.get(k, "") for k in self.FIELDNAMES})
        self.file.flush()

    def close(self):
        try:
            self.file.flush()
            self.file.close()
        except Exception:
            pass
