from __future__ import annotations

from contextlib import nullcontext

import cv2
import numpy as np

from eyegaze.utils.config import load_config
from eyegaze.utils.video import camera, fullscreen, iter_frames
from eyegaze.utils.screen import get_screen_size
from eyegaze.tracking.gaze_estimator import GazeEstimator
from eyegaze.tracking.smoother import EMAGazeSmoother
from eyegaze.tracking.laser_tracker import LaserTracker, draw_laser_debug, laser_direction_text
from eyegaze.calibration.screen_calibration import run_screen_calibration
from eyegaze.calibration.laser_runtime_calibration import run_laser_runtime_calibration
from eyegaze.ui.draw import draw_cursor, draw_text, make_thumbnail


def gaze_to_grid(x, y, w, h, rows=3, cols=3):
    if x is None or y is None:
        return None
    row = max(0, min(rows - 1, int(y / max(1, h) * rows)))
    col = max(0, min(cols - 1, int(x / max(1, w) * cols)))
    return row, col


def _fmt(value, digits=1):
    if value is None:
        return "?"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "?"


def main():
    cfg = load_config()
    cam_cfg = cfg["camera"]
    screen_w, screen_h = get_screen_size()

    head_angle_cfg = cfg.get("head_angle", {})
    gaze = GazeEstimator(
        baseline_distance_cm=cfg["screen"]["baseline_distance_cm"],
        tolerance_cm=cfg["screen"]["distance_tolerance_cm"],
        head_angle_cfg=head_angle_cfg,
    )
    smoother = EMAGazeSmoother(alpha_min=0.24, alpha_max=0.62, fast_distance_px=420.0)

    head_angle_cfg = cfg.get("head_angle", {})
    head_angle_mode = str(head_angle_cfg.get("mode", "laser" if cfg.get("laser", {}).get("enabled", False) else "face_geometry")).lower()

    laser_cfg = cfg.get("laser", {})
    laser_enabled = bool(laser_cfg.get("enabled", False)) and head_angle_mode == "laser"
    laser_tracker = LaserTracker(laser_cfg) if laser_enabled else None
    laser_cm = None
    if laser_enabled:
        lcam = laser_cfg.get("camera", {})
        laser_cm = camera(
            index=lcam.get("index", 1),
            name=lcam.get("name"),
            fallback_index=lcam.get("fallback_index", 0),
            width=lcam.get("width"),
            height=lcam.get("height"),
            fps=lcam.get("fps"),
        )

    with camera(
        index=cam_cfg.get("index", 0),
        name=cam_cfg.get("name"),
        fallback_index=cam_cfg.get("fallback_index", 0),
        width=cam_cfg.get("width"),
        height=cam_cfg.get("height"),
        fps=cam_cfg.get("fps"),
    ) as cap, (laser_cm if laser_cm is not None else nullcontext()) as laser_cap, fullscreen("Monitor"):
        # Старую калибровку углов головы можно отключить через head_precalibration.enabled=false.
        # Калибровка взгляда/дистанции для фронтальной камеры всё ещё нужна.
        run_screen_calibration(gaze, cap, screen_w, screen_h, cfg, "Monitor")

        if laser_enabled and laser_tracker is not None and laser_cap is not None:
            run_laser_runtime_calibration(laser_cap, laser_tracker, cfg, "Monitor", screen_w, screen_h)
            laser_frames = iter_frames(laser_cap)
        else:
            laser_frames = None

        for frame in iter_frames(cap):
            result = gaze.process_frame(frame)
            raw_xy = result.gaze_xy
            smooth_xy = smoother.update(raw_xy)
            x, y = smooth_xy if smooth_xy is not None else (None, None)
            cell = gaze_to_grid(x, y, screen_w, screen_h)

            laser_frame = next(laser_frames) if laser_frames is not None else None
            laser_result = laser_tracker.detect(laser_frame) if laser_tracker is not None and laser_frame is not None else None

            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            canvas[:] = (35, 35, 35)
            if x is not None:
                draw_cursor(canvas, x, y)

            meta = result.meta
            dist = meta.get("distance_cm")
            canvas = draw_text(canvas, f"GAZE CELL: {cell}", (40, 50), 34)
            canvas = draw_text(canvas, f"Distance: {_fmt(dist)} cm | {meta.get('distance_status')}", (40, 95), 28)
            canvas = draw_text(canvas, f"Ведущий глаз: {meta.get('dominant_eye', 'BOTH')}", (40, 135), 26)

            if laser_result is not None:
                canvas = draw_text(
                    canvas,
                    laser_direction_text(laser_result),
                    (40, 180),
                    32,
                    color=(0, 255, 255) if laser_result.valid else (0, 120, 255),
                )
                canvas = draw_text(
                    canvas,
                    f"Технически: yaw={_fmt(laser_result.yaw_deg)}° pitch={_fmt(laser_result.pitch_deg)}° | {laser_result.reason}",
                    (40, 220),
                    22,
                )
                canvas = draw_text(canvas, f"Laser px: {_fmt(laser_result.x_px)} / {_fmt(laser_result.y_px)}", (40, 250), 22)
            else:
                canvas = draw_text(canvas, f"Режим угла: {head_angle_mode}", (40, 180), 32)
                canvas = draw_text(
                    canvas,
                    f"Голова: yaw={_fmt(meta.get('yaw_deg'))}° pitch={_fmt(meta.get('pitch_deg'))}° | {meta.get('face_geometry_reason', '')}",
                    (40, 220),
                    24,
                    color=(0, 255, 255),
                )
                canvas = draw_text(canvas, f"Eye px: {_fmt(meta.get('eye_distance_px'))} | Brow px: {_fmt(meta.get('brow_distance_px'))}", (40, 250), 22)

            if raw_xy is not None:
                canvas = draw_text(canvas, f"Raw gaze: {raw_xy}", (40, 295), 22)
            if smooth_xy is not None:
                canvas = draw_text(canvas, f"Smooth gaze: {smooth_xy}", (40, 325), 22)
            canvas = draw_text(canvas, "ESC = выход", (40, screen_h - 50), 26)

            thumb = make_thumbnail(frame, (320, 240))
            th, tw = thumb.shape[:2]
            canvas[screen_h - th - 20:screen_h - 20, screen_w - tw - 20:screen_w - 20] = thumb

            if laser_frame is not None and laser_result is not None:
                lthumb = make_thumbnail(draw_laser_debug(laser_frame, laser_result), (260, 195))
                lth, ltw = lthumb.shape[:2]
                canvas[screen_h - th - lth - 35:screen_h - th - 35, screen_w - ltw - 20:screen_w - 20] = lthumb

            cv2.imshow("Monitor", canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                break


if __name__ == "__main__":
    main()
