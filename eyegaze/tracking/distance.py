from __future__ import annotations

import numpy as np

LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263


class DistanceEstimator:
    def __init__(self, baseline_distance_cm: float = 60.0, tolerance_cm: float = 7.0):
        self.baseline_distance_cm = float(baseline_distance_cm)
        self.tolerance_cm = float(tolerance_cm)
        self.baseline_eye_distance_px: float | None = None

    @staticmethod
    def eye_distance_px(landmarks: np.ndarray) -> float:
        left = landmarks[LEFT_EYE_OUTER, :2]
        right = landmarks[RIGHT_EYE_OUTER, :2]
        return float(np.linalg.norm(left - right))

    def calibrate_baseline(self, landmarks: np.ndarray) -> float:
        px = self.eye_distance_px(landmarks)
        if px <= 1e-6:
            raise ValueError("Invalid eye distance during baseline calibration")
        self.baseline_eye_distance_px = px
        return px

    def estimate(self, landmarks: np.ndarray) -> tuple[float | None, float, str]:
        current_px = self.eye_distance_px(landmarks)
        if self.baseline_eye_distance_px is None or current_px <= 1e-6:
            return None, current_px, "UNKNOWN"

        distance_cm = self.baseline_distance_cm * self.baseline_eye_distance_px / current_px
        if distance_cm < self.baseline_distance_cm - self.tolerance_cm:
            status = "CLOSER"
        elif distance_cm > self.baseline_distance_cm + self.tolerance_cm:
            status = "FARTHER"
        else:
            status = "OK"
        return float(distance_cm), float(current_px), status
