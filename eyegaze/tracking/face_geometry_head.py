from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cv2


@dataclass
class FaceGeometryHeadResult:
    valid: bool
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = 0.0
    distance_cm: float | None = None
    ref_distance_px: float | None = None
    eye_distance_px: float | None = None
    brow_distance_px: float | None = None
    dx_px: float | None = None
    dy_px: float | None = None
    dx_cm: float | None = None
    dy_cm: float | None = None
    neutral_eye_distance_px: float | None = None
    current_visible_eye_distance_cm: float | None = None
    real_eye_distance_cm: float | None = None
    eye_distance_ratio: float | None = None
    yaw_unsigned_deg: float | None = None
    yaw_sign_source: str = ""
    current_nose_eye_cm: float | None = None
    neutral_nose_eye_cm: float | None = None
    pitch_signal_cm: float | None = None
    pitch_model: str = ""
    reason: str = ""


def _get_xy(lm: Any, idx: int, w: int | None = None, h: int | None = None):
    p = lm[idx]
    x = float(getattr(p, "x", 0.0))
    y = float(getattr(p, "y", 0.0))
    if w is not None and h is not None and 0.0 <= x <= 1.5 and 0.0 <= y <= 1.5:
        return x * w, y * h
    return x, y


class FaceGeometryHeadEstimator:
    """
    Face geometry head-angle estimator.

    Improvement: yaw and pitch are partially decoupled.
    If the user mostly tilts UP/DOWN, fake LEFT/RIGHT yaw is suppressed.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.camera_to_face_cm = float(cfg.get("camera_to_face_cm", 57.0))
        self.eyebrow_distance_cm = float(cfg.get("eyebrow_distance_cm", 6.3))
        self.invert_yaw = bool(cfg.get("invert_yaw", False))
        self.invert_pitch = bool(cfg.get("invert_pitch", False))
        self.yaw_gain = float(cfg.get("yaw_gain", 1.0))
        self.pitch_gain = float(cfg.get("pitch_gain", 1.0))
        self.smoothing_alpha = float(cfg.get("smoothing_alpha", 0.35))
        self.deadzone_deg = float(cfg.get("deadzone_deg", 0.4))
        self.min_eye_ratio = float(cfg.get("min_eye_ratio", 0.35))
        self.max_eye_ratio = float(cfg.get("max_eye_ratio", 1.0))
        self.nose_sign_deadzone_px = float(cfg.get("nose_sign_deadzone_px", 1.0))
        self.yaw_nose_deadzone_ratio = float(cfg.get("yaw_nose_deadzone_ratio", 0.025))
        self.suppress_yaw_when_pitch_dominates = bool(cfg.get("suppress_yaw_when_pitch_dominates", True))
        self.pitch_dominance_ratio = float(cfg.get("pitch_dominance_ratio", 1.8))
        self.yaw_when_pitch_factor = float(cfg.get("yaw_when_pitch_factor", 0.15))
        self.pitch_reference_factor = float(cfg.get("pitch_reference_factor", 1.0))
        self.neutral_dx_px: float | None = None
        self.neutral_dy_px: float | None = None
        self.neutral_ref_px: float | None = None
        self._smooth_yaw: float | None = None
        self._smooth_pitch: float | None = None
        self.tracker = None
        try:
            import mediapipe as mp
            self.tracker = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception:
            self.tracker = None

    def is_calibrated(self) -> bool:
        return self.neutral_dx_px is not None and self.neutral_dy_px is not None and self.neutral_ref_px is not None

    def calibrate_neutral(self, landmarks) -> bool:
        if landmarks is None:
            return False
        try:
            dx, dy, ref = self._geometry_from_landmarks(landmarks)
        except Exception:
            return False
        if ref is None or ref <= 1:
            return False
        self.neutral_dx_px = float(dx)
        self.neutral_dy_px = float(dy)
        self.neutral_ref_px = float(ref)
        self._smooth_yaw = None
        self._smooth_pitch = None
        return True

    def _geometry_from_landmarks(self, landmarks, image_shape=None):
        h = w = None
        if image_shape is not None:
            h, w = image_shape[:2]
        left_eye = _get_xy(landmarks, 33, w, h)
        right_eye = _get_xy(landmarks, 263, w, h)
        nose = _get_xy(landmarks, 1, w, h)
        eye_cx = (left_eye[0] + right_eye[0]) / 2.0
        eye_cy = (left_eye[1] + right_eye[1]) / 2.0
        eye_distance_px = math.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
        dx_px = nose[0] - eye_cx
        dy_px = nose[1] - eye_cy
        return dx_px, dy_px, eye_distance_px

    def set_neutral_from_samples(self, samples) -> bool:
        vals = []
        for s in samples or []:
            if s is None:
                continue
            try:
                dx, dy, ref = s
                if ref and ref > 1:
                    vals.append((float(dx), float(dy), float(ref)))
            except Exception:
                continue
        if len(vals) < 3:
            return False
        self.neutral_dx_px = sum(v[0] for v in vals) / len(vals)
        self.neutral_dy_px = sum(v[1] for v in vals) / len(vals)
        self.neutral_ref_px = sum(v[2] for v in vals) / len(vals)
        self._smooth_yaw = None
        self._smooth_pitch = None
        return True

    def _estimate_from_geometry(self, dx_px: float, dy_px: float, eye_distance_px: float) -> FaceGeometryHeadResult:
        if not eye_distance_px or eye_distance_px <= 1:
            return FaceGeometryHeadResult(valid=False, reason="bad_eye_distance")
        if not self.is_calibrated():
            self.neutral_dx_px = dx_px
            self.neutral_dy_px = dy_px
            self.neutral_ref_px = eye_distance_px
        neutral_eye_px = float(self.neutral_ref_px)
        if neutral_eye_px <= 1:
            return FaceGeometryHeadResult(valid=False, reason="bad_neutral_eye_distance")

        cm_per_px_neutral = self.eyebrow_distance_cm / neutral_eye_px
        current_visible_eye_cm = eye_distance_px * cm_per_px_neutral
        ratio = current_visible_eye_cm / self.eyebrow_distance_cm
        ratio = max(self.min_eye_ratio, min(self.max_eye_ratio, ratio))
        yaw_unsigned = math.degrees(math.acos(ratio))

        rel_dx_px = dx_px - float(self.neutral_dx_px)
        rel_dy_px = dy_px - float(self.neutral_dy_px)
        adaptive_yaw_deadzone_px = max(self.nose_sign_deadzone_px, self.yaw_nose_deadzone_ratio * eye_distance_px)
        if abs(rel_dx_px) < adaptive_yaw_deadzone_px:
            yaw_sign = 0.0
            sign_source = "nose_x_deadzone"
        elif rel_dx_px > 0:
            yaw_sign = 1.0
            sign_source = "nose_right"
        else:
            yaw_sign = -1.0
            sign_source = "nose_left"
        yaw = yaw_sign * yaw_unsigned * self.yaw_gain

        if self.suppress_yaw_when_pitch_dominates:
            if abs(rel_dy_px) > self.pitch_dominance_ratio * max(abs(rel_dx_px), adaptive_yaw_deadzone_px):
                yaw *= self.yaw_when_pitch_factor
                sign_source += "_pitch_suppressed"

        current_cm_per_px = self.eyebrow_distance_cm / eye_distance_px
        neutral_cm_per_px = self.eyebrow_distance_cm / neutral_eye_px
        current_nose_eye_cm = dy_px * current_cm_per_px
        neutral_nose_eye_cm = float(self.neutral_dy_px) * neutral_cm_per_px
        pitch_signal_cm = current_nose_eye_cm - neutral_nose_eye_cm
        pitch_reference_cm = max(1e-6, self.eyebrow_distance_cm * self.pitch_reference_factor)
        pitch = math.degrees(math.atan2(pitch_signal_cm, pitch_reference_cm)) * self.pitch_gain
        rel_dx_cm = rel_dx_px * current_cm_per_px
        rel_dy_cm = pitch_signal_cm

        if self.invert_yaw:
            yaw = -yaw
        if self.invert_pitch:
            pitch = -pitch
        if abs(yaw) < self.deadzone_deg:
            yaw = 0.0
        if abs(pitch) < self.deadzone_deg:
            pitch = 0.0
        if self._smooth_yaw is None:
            self._smooth_yaw = yaw
            self._smooth_pitch = pitch
        else:
            a = self.smoothing_alpha
            self._smooth_yaw = a * yaw + (1.0 - a) * self._smooth_yaw
            self._smooth_pitch = a * pitch + (1.0 - a) * self._smooth_pitch
        distance_cm = self.camera_to_face_cm * (neutral_eye_px / eye_distance_px)

        return FaceGeometryHeadResult(
            valid=True,
            yaw_deg=float(self._smooth_yaw),
            pitch_deg=float(self._smooth_pitch),
            roll_deg=0.0,
            distance_cm=distance_cm,
            ref_distance_px=eye_distance_px,
            eye_distance_px=eye_distance_px,
            brow_distance_px=eye_distance_px,
            dx_px=rel_dx_px,
            dy_px=rel_dy_px,
            dx_cm=rel_dx_cm,
            dy_cm=rel_dy_cm,
            neutral_eye_distance_px=neutral_eye_px,
            current_visible_eye_distance_cm=current_visible_eye_cm,
            real_eye_distance_cm=self.eyebrow_distance_cm,
            eye_distance_ratio=ratio,
            yaw_unsigned_deg=yaw_unsigned,
            yaw_sign_source=sign_source,
            current_nose_eye_cm=current_nose_eye_cm,
            neutral_nose_eye_cm=neutral_nose_eye_cm,
            pitch_signal_cm=pitch_signal_cm,
            pitch_model="nose_pitch_with_yaw_pitch_decoupling",
            reason="ok_decoupled_face_geometry",
        )

    def process_frame(self, frame) -> FaceGeometryHeadResult:
        if frame is None:
            return FaceGeometryHeadResult(valid=False, reason="no_frame")
        if self.tracker is None:
            return FaceGeometryHeadResult(valid=False, reason="mediapipe_not_available")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.tracker.process(rgb)
        if not res.multi_face_landmarks:
            return FaceGeometryHeadResult(valid=False, reason="no_face")
        landmarks = res.multi_face_landmarks[0].landmark
        dx, dy, ref = self._geometry_from_landmarks(landmarks, frame.shape)
        return self._estimate_from_geometry(dx, dy, ref)

    def estimate(self, landmarks) -> FaceGeometryHeadResult:
        if landmarks is None:
            return FaceGeometryHeadResult(valid=False, reason="no_landmarks")
        try:
            dx, dy, ref = self._geometry_from_landmarks(landmarks)
            return self._estimate_from_geometry(dx, dy, ref)
        except Exception as e:
            return FaceGeometryHeadResult(valid=False, reason=f"estimate_error:{e}")


def face_geometry_direction_text(result: FaceGeometryHeadResult | None) -> str:
    if result is None:
        return "Без лазера: нет данных"
    if not result.valid:
        return f"Без лазера: нет данных ({result.reason})"
    yaw = result.yaw_deg or 0.0
    pitch = result.pitch_deg or 0.0
    parts = []
    if abs(yaw) < 0.5:
        parts.append("ровно")
    elif yaw > 0:
        parts.append(f"вправо {abs(yaw):.1f}°")
    else:
        parts.append(f"влево {abs(yaw):.1f}°")
    if abs(pitch) >= 0.5:
        if pitch > 0:
            parts.append(f"вверх {abs(pitch):.1f}°")
        else:
            parts.append(f"вниз {abs(pitch):.1f}°")
    return "Без лазера: " + "; ".join(parts)
