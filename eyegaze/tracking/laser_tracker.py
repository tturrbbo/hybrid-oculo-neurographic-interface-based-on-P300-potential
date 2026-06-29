from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class LaserDetection:
    valid: bool
    x_px: float | None = None
    y_px: float | None = None
    x_wall_cm: float | None = None
    y_wall_cm: float | None = None
    dx_px: float | None = None
    dy_px: float | None = None
    dx_cm: float | None = None
    dy_cm: float | None = None
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    area_px: float = 0.0
    confidence: float = 0.0
    reason: str = ""
    method: str = ""


class LaserTracker:
    """
    Трекинг лазерной точки второй камерой.

    Углы считаются НЕ по ручной калибровке L/R/U/D, а по пикселям:
        dx_px = x_laser - x_center
        dy_px = y_laser - y_center
        dx_cm = dx_px * cm_per_px_x
        dy_cm = dy_px * cm_per_px_y
        angle = atan(dx_cm / distance_cm)

    Что улучшено относительно простой версии:
    - отдельный масштаб по X/Y;
    - auto-scale от размера кадра, если задан wall_width_cm / wall_height_cm;
    - если есть ArUco/homography, используется реальная плоскость стены;
    - subpixel centroid по яркости внутри пятна;
    - EMA-сглаживание углов;
    - deadzone около нуля, чтобы не прыгало +-0.5°.
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        cfg = cfg or {}
        hsv = cfg.get("hsv", {})
        self.lower1 = np.array(hsv.get("lower1", [0, 70, 100]), dtype=np.uint8)
        self.upper1 = np.array(hsv.get("upper1", [15, 255, 255]), dtype=np.uint8)
        self.lower2 = np.array(hsv.get("lower2", [165, 70, 100]), dtype=np.uint8)
        self.upper2 = np.array(hsv.get("upper2", [180, 255, 255]), dtype=np.uint8)

        self.min_area_px = float(cfg.get("min_area_px", 2.0))
        self.max_area_px = float(cfg.get("max_area_px", 5000.0))
        self.blur_kernel = int(cfg.get("blur_kernel", 3))
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1

        self.detection_mode = str(cfg.get("detection_mode", "combined"))
        self.min_red_channel = int(cfg.get("min_red_channel", 90))
        self.red_delta = int(cfg.get("red_delta", 15))
        self.min_saturation = int(cfg.get("min_saturation", 35))
        self.min_value = int(cfg.get("min_value", 70))

        # Bright fallback. Lower defaults help when the laser is static before tracking starts.
        self.bright_percentile = float(cfg.get("bright_percentile", 99.0))
        self.bright_min_value = int(cfg.get("bright_min_value", 160))
        self.allow_white_hotspot = bool(cfg.get("allow_white_hotspot", True))

        # Blue/magenta background support:
        # On blue digits a red laser often becomes magenta, so R may NOT be much larger than B.
        # We only require R to be larger than G.
        self.magenta_red_min = int(cfg.get("magenta_red_min", 80))
        self.magenta_red_over_green = int(cfg.get("magenta_red_over_green", 8))
        self.magenta_min_value = int(cfg.get("magenta_min_value", 80))

        # Local contrast detector for a static bright laser dot.
        # It finds small pixels that are brighter than their local neighborhood.
        self.local_contrast_enabled = bool(cfg.get("local_contrast_enabled", True))
        self.local_contrast_threshold = int(cfg.get("local_contrast_threshold", 22))
        self.local_contrast_blur = int(cfg.get("local_contrast_blur", 31))
        if self.local_contrast_blur % 2 == 0:
            self.local_contrast_blur += 1
        self.morph_kernel = int(cfg.get("morph_kernel", 3))
        self.debug_masks = bool(cfg.get("debug_masks", False))
        self.last_mask: np.ndarray | None = None
        self.last_method: str = ""

        # Основная геометрия.
        # distance_cm — расстояние от центра вращения головы/лазера до стены.
        self.wall_distance_cm = float(cfg.get("wall_distance_cm", cfg.get("distance_cm", 99.0)))
        self.yaw_distance_cm = float(cfg.get("yaw_distance_cm", self.wall_distance_cm))
        self.pitch_distance_cm = float(cfg.get("pitch_distance_cm", self.wall_distance_cm))

        # Масштаб пиксель -> сантиметр на плоскости стены.
        # Если ArUco нет, это самый важный параметр точности.
        self.px_to_cm_x = float(cfg.get("px_to_cm_x", cfg.get("cm_per_px_x", 0.1)))
        self.px_to_cm_y = float(cfg.get("px_to_cm_y", cfg.get("cm_per_px_y", self.px_to_cm_x)))
        self.auto_scale_from_frame = bool(cfg.get("auto_scale_from_frame", True))
        self.wall_width_cm = cfg.get("wall_width_cm")
        self.wall_height_cm = cfg.get("wall_height_cm")

        self.center_px = tuple(cfg.get("center_px", [0.0, 0.0]))
        self.center_wall_cm = tuple(cfg.get("center_wall_cm", [0.0, 0.0]))
        self.wall_size_cm = tuple(cfg.get("wall_size_cm", [100.0, 70.0]))
        self.homography: np.ndarray | None = None

        self.invert_yaw = bool(cfg.get("invert_yaw", False))
        self.invert_pitch = bool(cfg.get("invert_pitch", True))
        self.yaw_offset_deg = float(cfg.get("yaw_offset_deg", 0.0))
        self.pitch_offset_deg = float(cfg.get("pitch_offset_deg", 0.0))
        self.deadzone_deg = float(cfg.get("deadzone_deg", 0.3))
        self.smoothing_alpha = float(cfg.get("smoothing_alpha", 0.35))
        self._smooth_yaw: float | None = None
        self._smooth_pitch: float | None = None

        # Persistence against one-frame losses during movement.
        self.persistence_frames = int(cfg.get("persistence_frames", 5))
        self.frames_since_seen = 10**9
        self.last_valid_detection: LaserDetection | None = None

        calibration_json = cfg.get("calibration_json")
        if calibration_json:
            self.load_calibration(calibration_json, missing_ok=True)

    def load_calibration(self, path: str | Path, *, missing_ok: bool = False) -> None:
        path = Path(path)
        if not path.exists():
            if missing_ok:
                return
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("homography") is not None:
            self.homography = np.asarray(data["homography"], dtype=np.float64)
        self.wall_distance_cm = float(data.get("wall_distance_cm", self.wall_distance_cm))
        self.yaw_distance_cm = float(data.get("yaw_distance_cm", self.yaw_distance_cm))
        self.pitch_distance_cm = float(data.get("pitch_distance_cm", self.pitch_distance_cm))
        self.center_px = tuple(data.get("center_px", self.center_px))
        self.center_wall_cm = tuple(data.get("center_wall_cm", self.center_wall_cm))
        self.wall_size_cm = tuple(data.get("wall_size_cm", self.wall_size_cm))
        self.px_to_cm_x = float(data.get("px_to_cm_x", self.px_to_cm_x))
        self.px_to_cm_y = float(data.get("px_to_cm_y", self.px_to_cm_y))
        self.yaw_offset_deg = float(data.get("yaw_offset_deg", self.yaw_offset_deg))
        self.pitch_offset_deg = float(data.get("pitch_offset_deg", self.pitch_offset_deg))

    def save_calibration(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "homography": None if self.homography is None else self.homography.tolist(),
            "wall_distance_cm": self.wall_distance_cm,
            "yaw_distance_cm": self.yaw_distance_cm,
            "pitch_distance_cm": self.pitch_distance_cm,
            "center_px": list(self.center_px),
            "center_wall_cm": list(self.center_wall_cm),
            "wall_size_cm": list(self.wall_size_cm),
            "px_to_cm_x": self.px_to_cm_x,
            "px_to_cm_y": self.px_to_cm_y,
            "yaw_offset_deg": self.yaw_offset_deg,
            "pitch_offset_deg": self.pitch_offset_deg,
            "angle_model": "pixel_tangent",
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def set_wall_homography(self, image_points_px: np.ndarray, wall_size_cm: tuple[float, float]) -> None:
        """image_points_px: TL, TR, BR, BL в пикселях камеры."""
        wall_w, wall_h = float(wall_size_cm[0]), float(wall_size_cm[1])
        dst = np.array([[0, 0], [wall_w, 0], [wall_w, wall_h], [0, wall_h]], dtype=np.float32)
        src = np.asarray(image_points_px, dtype=np.float32)
        if src.shape != (4, 2):
            raise ValueError("image_points_px must have shape (4, 2): TL, TR, BR, BL")
        H, _ = cv2.findHomography(src, dst)
        if H is None:
            raise RuntimeError("cannot compute wall homography")
        self.homography = H
        self.wall_size_cm = (wall_w, wall_h)
        self.center_wall_cm = (wall_w / 2.0, wall_h / 2.0)

    def set_center_from_detection(self, detection: LaserDetection) -> None:
        if not detection.valid or detection.x_px is None or detection.y_px is None:
            raise ValueError("cannot set center from invalid laser detection")
        self.center_px = (float(detection.x_px), float(detection.y_px))
        if detection.x_wall_cm is not None and detection.y_wall_cm is not None:
            self.center_wall_cm = (float(detection.x_wall_cm), float(detection.y_wall_cm))
        self._smooth_yaw = None
        self._smooth_pitch = None

    def _update_scale_from_frame(self, frame_bgr: np.ndarray) -> None:
        if not self.auto_scale_from_frame or self.homography is not None:
            return
        h, w = frame_bgr.shape[:2]
        if self.wall_width_cm:
            self.px_to_cm_x = float(self.wall_width_cm) / max(1.0, float(w))
        if self.wall_height_cm:
            self.px_to_cm_y = float(self.wall_height_cm) / max(1.0, float(h))

    def point_to_wall_cm(self, x_px: float, y_px: float) -> tuple[float, float]:
        if self.homography is not None:
            src = np.array([[[float(x_px), float(y_px)]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(src, self.homography)[0, 0]
            return float(dst[0]), float(dst[1])
        cx, cy = self.center_px
        cwx, cwy = self.center_wall_cm
        return cwx + (float(x_px) - float(cx)) * self.px_to_cm_x, cwy + (float(y_px) - float(cy)) * self.px_to_cm_y

    def _apply_smoothing(self, yaw: float, pitch: float) -> tuple[float, float]:
        a = max(0.0, min(1.0, self.smoothing_alpha))
        if self._smooth_yaw is None:
            self._smooth_yaw = yaw
            self._smooth_pitch = pitch
        else:
            self._smooth_yaw = a * yaw + (1.0 - a) * self._smooth_yaw
            self._smooth_pitch = a * pitch + (1.0 - a) * self._smooth_pitch
        yaw_s = float(self._smooth_yaw)
        pitch_s = float(self._smooth_pitch)
        if abs(yaw_s) < self.deadzone_deg:
            yaw_s = 0.0
        if abs(pitch_s) < self.deadzone_deg:
            pitch_s = 0.0
        return yaw_s, pitch_s

    def wall_cm_to_angles(self, x_cm: float, y_cm: float, x_px: float | None = None, y_px: float | None = None) -> tuple[float, float]:
        cx_cm, cy_cm = self.center_wall_cm
        dx_cm = float(x_cm) - float(cx_cm)
        dy_cm = float(y_cm) - float(cy_cm)

        yaw = math.degrees(math.atan2(dx_cm, max(1e-6, self.yaw_distance_cm))) + self.yaw_offset_deg
        pitch = math.degrees(math.atan2(dy_cm, max(1e-6, self.pitch_distance_cm))) + self.pitch_offset_deg
        if self.invert_yaw:
            yaw = -yaw
        if self.invert_pitch:
            pitch = -pitch
        return self._apply_smoothing(yaw, pitch)

    def _build_masks(self, frame_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
        if self.blur_kernel >= 3:
            work = cv2.GaussianBlur(frame_bgr, (self.blur_kernel, self.blur_kernel), 0)
        else:
            work = frame_bgr

        b, g, r = cv2.split(work)
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)

        # 1) Classic red HSV.
        hsv_red = cv2.bitwise_or(
            cv2.inRange(hsv, self.lower1, self.upper1),
            cv2.inRange(hsv, self.lower2, self.upper2),
        )

        r16 = r.astype(np.int16)
        g16 = g.astype(np.int16)
        b16 = b.astype(np.int16)

        # 2) Red dominance for normal backgrounds.
        red_dom = (
            (r16 >= self.min_red_channel)
            & ((r16 - g16) >= self.red_delta)
            & ((r16 - b16) >= max(6, self.red_delta // 2))
            & (v >= self.min_value)
        ).astype(np.uint8) * 255

        # 3) Magenta/blue-background laser.
        # On blue digits, the red dot can look magenta, so B can be high.
        # Do not require R >> B; require R > G.
        magenta_red = (
            (r16 >= self.magenta_red_min)
            & ((r16 - g16) >= self.magenta_red_over_green)
            & (v >= self.magenta_min_value)
        ).astype(np.uint8) * 255

        # 4) Bright hotspot. Good for overexposed white laser core.
        q = np.percentile(v, self.bright_percentile)
        bright_thr = int(max(self.bright_min_value, q))
        bright = (v >= bright_thr).astype(np.uint8) * 255
        if not self.allow_white_hotspot:
            bright = cv2.bitwise_and(bright, cv2.inRange(s, self.min_saturation, 255))

        # 5) Local contrast hotspot.
        # This helps when the laser is already static before the camera starts.
        if self.local_contrast_enabled:
            local_blur = cv2.GaussianBlur(v, (self.local_contrast_blur, self.local_contrast_blur), 0)
            contrast = cv2.subtract(v, local_blur)
            local_hotspot = (contrast >= self.local_contrast_threshold).astype(np.uint8) * 255
            # Keep only relatively bright pixels, otherwise texture/noise may pass.
            local_hotspot = cv2.bitwise_and(local_hotspot, (v >= self.bright_min_value).astype(np.uint8) * 255)
        else:
            local_hotspot = np.zeros_like(v, dtype=np.uint8)

        if self.detection_mode == "hsv":
            masks = [("hsv", hsv_red)]
        elif self.detection_mode == "red_dominance":
            masks = [("red_dominance", red_dom)]
        elif self.detection_mode == "bright":
            masks = [("bright", bright), ("local_hotspot", local_hotspot)]
        elif self.detection_mode == "magenta":
            masks = [("magenta_red", magenta_red)]
        else:
            combined = cv2.bitwise_or(hsv_red, red_dom)
            combined = cv2.bitwise_or(combined, magenta_red)
            combined = cv2.bitwise_or(combined, bright)
            combined = cv2.bitwise_or(combined, local_hotspot)
            masks = [
                ("combined", combined),
                ("hsv", hsv_red),
                ("red_dominance", red_dom),
                ("magenta_red", magenta_red),
                ("bright", bright),
                ("local_hotspot", local_hotspot),
            ]

        k = max(1, int(self.morph_kernel))
        kernel = np.ones((k, k), np.uint8)
        cleaned: list[tuple[str, np.ndarray]] = []
        for name, mask in masks:
            # For tiny laser dots, OPEN can delete the dot.
            # Use CLOSE + DILATE instead.
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
            cleaned.append((name, mask))
        return cleaned

    def _score_contour(self, frame_bgr: np.ndarray, contour: np.ndarray) -> tuple[float, float, tuple[float, float]]:
        area = float(cv2.contourArea(contour))
        if area <= 0:
            return -1.0, area, (0.0, 0.0)

        mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Subpixel centroid: берем не просто геометрический центр контура, а центр яркости.
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return -1.0, area, (0.0, 0.0)
        weights = gray[ys, xs] + 1.0
        x = float(np.sum(xs * weights) / np.sum(weights))
        y = float(np.sum(ys * weights) / np.sum(weights))

        b_mean, g_mean, r_mean, _ = cv2.mean(frame_bgr, mask=mask)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        _, s_mean, v_mean, _ = cv2.mean(hsv, mask=mask)

        perimeter = float(cv2.arcLength(contour, True))
        circularity = 0.0 if perimeter <= 0 else min(1.0, 4.0 * math.pi * area / (perimeter * perimeter))
        red_score = max(0.0, r_mean - max(g_mean, b_mean))
        bright_score = v_mean / 255.0
        sat_score = s_mean / 255.0
        area_score = 1.0 / (1.0 + area / max(1.0, self.max_area_px * 0.18))
        score = (2.0 * bright_score) + (1.6 * min(1.0, red_score / 80.0)) + (0.6 * sat_score) + (0.4 * circularity) + (0.5 * area_score)
        return score, area, (x, y)

    def _copy_last_as_predicted(self, reason: str) -> LaserDetection:
        if self.last_valid_detection is None:
            return LaserDetection(False, reason=reason)
        d = self.last_valid_detection
        return LaserDetection(
            True,
            x_px=d.x_px,
            y_px=d.y_px,
            x_wall_cm=d.x_wall_cm,
            y_wall_cm=d.y_wall_cm,
            dx_px=d.dx_px,
            dy_px=d.dy_px,
            dx_cm=d.dx_cm,
            dy_cm=d.dy_cm,
            yaw_deg=d.yaw_deg,
            pitch_deg=d.pitch_deg,
            area_px=d.area_px,
            confidence=max(0.05, d.confidence * 0.55),
            reason="predicted_from_last",
            method="persistence",
        )

    def _remember_valid(self, detection: LaserDetection) -> LaserDetection:
        if detection.valid:
            self.last_valid_detection = detection
            self.frames_since_seen = 0
        return detection

    def _lost_or_predict(self, reason: str) -> LaserDetection:
        self.frames_since_seen += 1
        if self.last_valid_detection is not None and self.frames_since_seen <= self.persistence_frames:
            return self._copy_last_as_predicted(reason)
        return LaserDetection(False, reason=reason)

    def detect(self, frame_bgr: np.ndarray | None) -> LaserDetection:
        if frame_bgr is None:
            return self._lost_or_predict("no_frame")

        self._update_scale_from_frame(frame_bgr)

        best_score = -1.0
        best_area = 0.0
        best_xy = (0.0, 0.0)
        best_method = ""
        best_mask: np.ndarray | None = None

        for method, mask in self._build_masks(frame_bgr):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < self.min_area_px or area > self.max_area_px:
                    continue
                score, area, xy = self._score_contour(frame_bgr, contour)

                # If we already tracked a laser before, prefer candidates near the last point.
                # This reduces jumps to random highlights on a noisy/blue background.
                try:
                    last = getattr(self, "last_valid_detection", None)
                    if last is not None and last.x_px is not None and last.y_px is not None:
                        dist = math.hypot(float(xy[0]) - float(last.x_px), float(xy[1]) - float(last.y_px))
                        near_bonus = max(0.0, 1.0 - dist / 180.0)
                        score += 0.75 * near_bonus
                except Exception:
                    pass

                if score > best_score:
                    best_score = score
                    best_area = area
                    best_xy = xy
                    best_method = method
                    best_mask = mask

        self.last_mask = best_mask
        self.last_method = best_method

        if best_score < 0:
            return self._lost_or_predict("no_laser_candidate")

        x_px, y_px = best_xy
        x_cm, y_cm = self.point_to_wall_cm(x_px, y_px)
        yaw, pitch = self.wall_cm_to_angles(x_cm, y_cm, x_px=x_px, y_px=y_px)
        confidence = float(max(0.0, min(1.0, best_score / 5.0)))

        dx_px = float(x_px) - float(self.center_px[0])
        dy_px = float(y_px) - float(self.center_px[1])
        dx_cm = float(x_cm) - float(self.center_wall_cm[0])
        dy_cm = float(y_cm) - float(self.center_wall_cm[1])

        detection = LaserDetection(
            True,
            x_px=x_px,
            y_px=y_px,
            x_wall_cm=x_cm,
            y_wall_cm=y_cm,
            dx_px=dx_px,
            dy_px=dy_px,
            dx_cm=dx_cm,
            dy_cm=dy_cm,
            yaw_deg=yaw,
            pitch_deg=pitch,
            area_px=best_area,
            confidence=confidence,
            reason="ok",
            method=best_method,
        )
        return self._remember_valid(detection)


def _axis_direction_text(value: float | None, negative_word: str, positive_word: str, deadzone_deg: float = 1.0) -> str:
    if value is None:
        return "нет данных"
    v = float(value)
    if abs(v) < deadzone_deg:
        return "ровно"
    return f"{positive_word} {abs(v):.1f}°" if v > 0 else f"{negative_word} {abs(v):.1f}°"


def laser_direction_text(detection: LaserDetection | None, deadzone_deg: float = 1.0) -> str:
    if detection is None:
        return "Лазер: выключен"
    if not detection.valid:
        return f"Лазер: не найден ({detection.reason})"
    yaw_text = _axis_direction_text(detection.yaw_deg, "влево", "вправо", deadzone_deg)
    pitch_text = _axis_direction_text(detection.pitch_deg, "вниз", "вверх", deadzone_deg)
    return f"Голова: {yaw_text}; {pitch_text}"


def laser_direction_lines(detection: LaserDetection | None, deadzone_deg: float = 1.0) -> list[str]:
    if detection is None:
        return ["Лазер: выключен"]
    if not detection.valid:
        return ["Лазер: не найден", f"Причина: {detection.reason}"]
    yaw_text = _axis_direction_text(detection.yaw_deg, "влево", "вправо", deadzone_deg)
    pitch_text = _axis_direction_text(detection.pitch_deg, "вниз", "вверх", deadzone_deg)
    raw = f"yaw={detection.yaw_deg:.1f}°, pitch={detection.pitch_deg:.1f}°"
    px = f"dx={detection.dx_px:.1f}px, dy={detection.dy_px:.1f}px"
    cm = f"dx={detection.dx_cm:.1f}cm, dy={detection.dy_cm:.1f}cm"
    return [f"Голова: {yaw_text}", f"Наклон: {pitch_text}", raw, px, cm]


def draw_laser_debug(frame_bgr: np.ndarray, detection: LaserDetection) -> np.ndarray:
    out = frame_bgr.copy()
    if detection.valid and detection.x_px is not None and detection.y_px is not None:
        p = (int(detection.x_px), int(detection.y_px))
        cv2.circle(out, p, 12, (0, 255, 255), 2)
        cv2.drawMarker(out, p, (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(out, laser_direction_text(detection), (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(out, f"dx={detection.dx_px:.1f}px dy={detection.dy_px:.1f}px", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(out, f"area={detection.area_px:.0f} method={detection.method} reason={detection.reason}", (20, 101), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    else:
        cv2.putText(out, f"laser: {detection.reason}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out
