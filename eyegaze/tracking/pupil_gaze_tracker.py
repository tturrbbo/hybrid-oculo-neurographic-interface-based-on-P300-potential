from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
import cv2


@dataclass
class PupilGazeResult:
    valid: bool
    pupil_x_norm: float | None = None
    pupil_y_norm: float | None = None
    screen_x_norm: float | None = None
    screen_y_norm: float | None = None
    gaze_yaw_deg: float | None = None
    gaze_pitch_deg: float | None = None
    reason: str = ""


def _pt(lm: Any, idx: int, w: int, h: int):
    p = lm[idx]
    return float(p.x) * w, float(p.y) * h


def _mean(ps):
    return sum(p[0] for p in ps) / len(ps), sum(p[1] for p in ps) / len(ps)


class PupilGazeTracker:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.screen_width_cm = float(cfg.get("screen_width_cm", 60.0))
        self.screen_height_cm = float(cfg.get("screen_height_cm", 34.0))
        self.screen_distance_cm = float(cfg.get("screen_distance_cm", cfg.get("distance_cm", 66.0)))
        self.gaze_gain_x = float(cfg.get("gaze_gain_x", 3.0))
        self.gaze_gain_y = float(cfg.get("gaze_gain_y", 3.0))
        self.invert_x = bool(cfg.get("invert_x", False))
        self.invert_y = bool(cfg.get("invert_y", False))
        self.smoothing_alpha = float(cfg.get("smoothing_alpha", 0.45))
        self.deadzone_norm = float(cfg.get("deadzone_norm", 0.004))
        self.center_x_norm = None
        self.center_y_norm = None
        self._sx = None
        self._sy = None

        self.face_mesh = None
        try:
            import mediapipe as mp
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception:
            self.face_mesh = None

    def _eye_ratio(self, lm, w, h, side):
        if side == "left":
            outer, inner = _pt(lm, 33, w, h), _pt(lm, 133, w, h)
            upper = _mean([_pt(lm, 159, w, h), _pt(lm, 160, w, h)])
            lower = _mean([_pt(lm, 145, w, h), _pt(lm, 144, w, h)])
            iris_ids = [468, 469, 470, 471, 472]
        else:
            outer, inner = _pt(lm, 263, w, h), _pt(lm, 362, w, h)
            upper = _mean([_pt(lm, 386, w, h), _pt(lm, 387, w, h)])
            lower = _mean([_pt(lm, 374, w, h), _pt(lm, 373, w, h)])
            iris_ids = [473, 474, 475, 476, 477]
        iris = _mean([_pt(lm, i, w, h) for i in iris_ids])
        left_corner = outer if outer[0] < inner[0] else inner
        right_corner = inner if outer[0] < inner[0] else outer
        eye_w = max(1e-6, right_corner[0] - left_corner[0])
        eye_h = max(1e-6, lower[1] - upper[1])
        return (iris[0] - left_corner[0]) / eye_w, (iris[1] - upper[1]) / eye_h

    def extract_pupil_norm(self, frame_bgr):
        if frame_bgr is None or self.face_mesh is None:
            return None
        h, w = frame_bgr.shape[:2]
        res = self.face_mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        try:
            l = self._eye_ratio(lm, w, h, "left")
            r = self._eye_ratio(lm, w, h, "right")
        except Exception:
            return None
        return ((l[0] + r[0]) / 2.0, (l[1] + r[1]) / 2.0)

    def calibrate_center_from_samples(self, samples):
        vals = [(float(x), float(y)) for x, y in samples if x is not None and y is not None]
        if len(vals) < 5:
            return False
        self.center_x_norm = sum(x for x, _ in vals) / len(vals)
        self.center_y_norm = sum(y for _, y in vals) / len(vals)
        self._sx = None
        self._sy = None
        return True

    def process_frame(self, frame_bgr):
        p = self.extract_pupil_norm(frame_bgr)
        if p is None:
            return PupilGazeResult(False, reason="no_eye_or_iris")
        px, py = p
        if self.center_x_norm is None:
            self.center_x_norm = px
            self.center_y_norm = py
        dx = px - float(self.center_x_norm)
        dy = py - float(self.center_y_norm)
        if abs(dx) < self.deadzone_norm:
            dx = 0.0
        if abs(dy) < self.deadzone_norm:
            dy = 0.0
        if self.invert_x:
            dx = -dx
        if self.invert_y:
            dy = -dy
        sx = max(0.0, min(1.0, 0.5 + dx * self.gaze_gain_x))
        sy = max(0.0, min(1.0, 0.5 + dy * self.gaze_gain_y))
        a = max(0.0, min(1.0, self.smoothing_alpha))
        if self._sx is None:
            self._sx, self._sy = sx, sy
        else:
            self._sx = a * sx + (1-a) * self._sx
            self._sy = a * sy + (1-a) * self._sy
        yaw = math.degrees(math.atan2((self._sx - 0.5) * self.screen_width_cm, max(1e-6, self.screen_distance_cm)))
        pitch = math.degrees(math.atan2((0.5 - self._sy) * self.screen_height_cm, max(1e-6, self.screen_distance_cm)))
        return PupilGazeResult(True, px, py, self._sx, self._sy, yaw, pitch, "ok")
