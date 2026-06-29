from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import time
from contextlib import nullcontext as _nullcontext

import cv2
import numpy as np

from eyegaze.utils.config import load_config
from eyegaze.utils.screen import get_screen_size
from eyegaze.utils.video import camera, fullscreen, iter_frames
from eyegaze.tracking.laser_tracker import LaserTracker, draw_laser_debug, laser_direction_text
from eyegaze.tracking.face_geometry_head import FaceGeometryHeadEstimator, face_geometry_direction_text
from eyegaze.calibration.laser_runtime_calibration import run_laser_runtime_calibration
from eyegaze.calibration.face_geometry_runtime_calibration import run_face_geometry_runtime_calibration
from eyegaze.experiment.head_protocol import HeadProtocol, targets_from_config
from eyegaze.ui.draw import draw_center_box, draw_text, make_thumbnail


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--participant", required=True, help="ID участника, например P01")
    p.add_argument("--config", default="config/experiment.yaml")
    return p.parse_args()


def _target_signed_angles(direction: str, angle_deg: float) -> tuple[float, float, str]:
    d = str(direction).upper()
    a = float(angle_deg or 0.0)

    if d == "RIGHT":
        return +a, 0.0, "yaw"
    if d == "LEFT":
        return -a, 0.0, "yaw"
    if d == "UP":
        return 0.0, +a, "pitch"
    if d == "DOWN":
        return 0.0, -a, "pitch"

    return 0.0, 0.0, "center"


def _axis_value(axis: str, yaw, pitch):
    if axis == "yaw":
        return yaw
    if axis == "pitch":
        return pitch

    if yaw is None or pitch is None:
        return None

    try:
        return math.sqrt(float(yaw) ** 2 + float(pitch) ** 2)
    except Exception:
        return None


def _err(target, measured):
    if target is None or measured is None:
        return ""
    try:
        return abs(float(target) - float(measured))
    except Exception:
        return ""


def _fmt(v, digits=3):
    if v is None or v == "":
        return ""
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


class HeadAnglesLogger:
    FIELDNAMES = [
        "participant_id",
        "timestamp_unix",
        "block_idx",

        # theoretical target
        "target_label",
        "target_marker",
        "target_direction",
        "target_angle_deg",
        "target_axis",
        "target_yaw_deg",
        "target_pitch_deg",
        "target_axis_angle_deg",

        # laser method
        "laser_valid",
        "laser_yaw_deg",
        "laser_pitch_deg",
        "laser_axis_angle_deg",
        "laser_axis_error_deg",
        "laser_yaw_error_deg",
        "laser_pitch_error_deg",
        "laser_x_px",
        "laser_y_px",
        "laser_dx_px",
        "laser_dy_px",
        "laser_dx_cm",
        "laser_dy_cm",
        "laser_area_px",
        "laser_confidence",
        "laser_reason",
        "laser_method",
        "laser_calibration_json",

        # face geometry method
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
    ]

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=self.FIELDNAMES, extrasaction="ignore")
        self.w.writeheader()
        self.f.flush()

    def log(self, **row):
        self.w.writerow({k: row.get(k, "") for k in self.FIELDNAMES})
        self.f.flush()

    def close(self):
        try:
            self.f.flush()
            self.f.close()
        except Exception:
            pass


def main():
    args = parse_args()
    cfg = load_config(args.config)

    screen_w, screen_h = get_screen_size()

    cam_cfg = cfg["camera"]
    exp_cfg = cfg.get("experiment", {})
    laser_cfg = cfg.get("laser", {})
    laser_enabled = bool(laser_cfg.get("enabled", False))

    # Head-only experiment uses only head_targets.
    # It does NOT run gaze calibration and does NOT use 3x3 tiles.
    targets = targets_from_config(cfg)
    head_protocol = HeadProtocol(
        targets,
        prep_seconds=float(exp_cfg.get("prep_seconds", 5.0)),
        record_seconds=float(exp_cfg.get("record_seconds", 15.0)),
    )

    face_geom_cfg = cfg.get("head_angle", {}).get("face_geometry", {})
    if not face_geom_cfg:
        face_geom_cfg = cfg.get("face_geometry", {})
    face_head = FaceGeometryHeadEstimator(face_geom_cfg)

    laser_tracker = LaserTracker(laser_cfg) if laser_enabled else None

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path("data/logs/head_angles") / f"{args.participant}_head_angles_{timestamp}.csv"
    logger = HeadAnglesLogger(log_path)

    # Каждый запуск эксперимента получает свой laser_wall.json.
    # Это важно, если менялось разрешение камеры, положение камеры, расстояние или фон.
    laser_calibration_path = Path("data/calibration/laser_wall") / f"{args.participant}_{timestamp}_laser_wall.json"
    if laser_enabled:
        laser_calibration_path.parent.mkdir(parents=True, exist_ok=True)
        laser_cfg["calibration_json"] = str(laser_calibration_path)

    laser_cap_cm = None
    if laser_enabled:
        laser_cam_cfg = laser_cfg.get("camera", {})
        laser_cap_cm = camera(
            index=laser_cam_cfg.get("index", 0),
            name=laser_cam_cfg.get("name"),
            fallback_index=laser_cam_cfg.get("fallback_index"),
            width=laser_cam_cfg.get("width"),
            height=laser_cam_cfg.get("height"),
            fps=laser_cam_cfg.get("fps"),
        )

    with camera(
        index=cam_cfg.get("index", 0),
        name=cam_cfg.get("name"),
        fallback_index=cam_cfg.get("fallback_index"),
        width=cam_cfg.get("width"),
        height=cam_cfg.get("height"),
        fps=cam_cfg.get("fps"),
    ) as face_cap, (laser_cap_cm if laser_cap_cm is not None else _nullcontext()) as laser_cap, fullscreen("HeadAnglesExperiment"):
        try:
            # 1) Calibration for face geometry only.
            run_face_geometry_runtime_calibration(
                face_cap,
                face_head,
                "HeadAnglesExperiment",
                screen_w,
                screen_h,
            )

            # 2) Calibration for laser only.
            if laser_enabled and laser_tracker is not None and laser_cap is not None:
                run_laser_runtime_calibration(
                    laser_cap,
                    laser_tracker,
                    cfg,
                    "HeadAnglesExperiment",
                    screen_w,
                    screen_h,
                )
                # Сохраняем именно калибровку этого эксперимента в отдельный файл.
                laser_tracker.save_calibration(laser_calibration_path)
                print(f"Saved per-experiment laser calibration: {laser_calibration_path.resolve()}")

            face_iter = iter_frames(face_cap)
            laser_iter = iter_frames(laser_cap) if laser_enabled and laser_cap is not None else None

            for block_idx, target in enumerate(head_protocol.targets, start=1):
                target_yaw, target_pitch, target_axis = _target_signed_angles(target.direction, target.angle_deg)
                target_axis_value = _axis_value(target_axis, target_yaw, target_pitch)

                prep_start = time.perf_counter()
                while time.perf_counter() - prep_start < head_protocol.prep_seconds:
                    left = head_protocol.prep_seconds - (time.perf_counter() - prep_start)

                    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                    canvas[:] = (35, 35, 35)

                    lines = target.prompt_lines_ru + [
                        "",
                        "Эксперимент только по углам головы",
                        "Взгляд и плитки 3×3 сейчас НЕ проверяются",
                        f"До записи: {left:.1f} сек",
                    ]
                    canvas = draw_center_box(canvas, lines, font_size=44)
                    canvas = draw_text(canvas, f"Блок {block_idx}/{len(head_protocol.targets)}", (40, 40), 30)
                    cv2.imshow("HeadAnglesExperiment", canvas)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("s"), ord("S")):
                        return

                record_start = time.perf_counter()
                while time.perf_counter() - record_start < head_protocol.record_seconds:
                    face_frame = next(face_iter)
                    laser_frame = next(laser_iter) if laser_iter is not None else None

                    face_result = face_head.process_frame(face_frame)
                    laser_result = laser_tracker.detect(laser_frame) if laser_tracker is not None and laser_frame is not None else None

                    laser_yaw = None if laser_result is None else laser_result.yaw_deg
                    laser_pitch = None if laser_result is None else laser_result.pitch_deg
                    laser_axis = _axis_value(target_axis, laser_yaw, laser_pitch)

                    face_yaw = None if face_result is None else face_result.yaw_deg
                    face_pitch = None if face_result is None else face_result.pitch_deg
                    face_axis = _axis_value(target_axis, face_yaw, face_pitch)

                    logger.log(
                        participant_id=args.participant,
                        timestamp_unix=f"{time.time():.6f}",
                        block_idx=block_idx,

                        target_label=target.label,
                        target_marker=target.marker,
                        target_direction=target.direction,
                        target_angle_deg=_fmt(target.angle_deg),
                        target_axis=target_axis,
                        target_yaw_deg=_fmt(target_yaw),
                        target_pitch_deg=_fmt(target_pitch),
                        target_axis_angle_deg=_fmt(target_axis_value),

                        laser_valid="" if laser_result is None else int(laser_result.valid),
                        laser_yaw_deg=_fmt(laser_yaw),
                        laser_pitch_deg=_fmt(laser_pitch),
                        laser_axis_angle_deg=_fmt(laser_axis),
                        laser_axis_error_deg=_fmt(_err(target_axis_value, laser_axis)),
                        laser_yaw_error_deg=_fmt(_err(target_yaw, laser_yaw)),
                        laser_pitch_error_deg=_fmt(_err(target_pitch, laser_pitch)),
                        laser_x_px="" if laser_result is None else _fmt(laser_result.x_px),
                        laser_y_px="" if laser_result is None else _fmt(laser_result.y_px),
                        laser_dx_px="" if laser_result is None else _fmt(laser_result.dx_px),
                        laser_dy_px="" if laser_result is None else _fmt(laser_result.dy_px),
                        laser_dx_cm="" if laser_result is None else _fmt(laser_result.dx_cm),
                        laser_dy_cm="" if laser_result is None else _fmt(laser_result.dy_cm),
                        laser_area_px="" if laser_result is None else _fmt(laser_result.area_px),
                        laser_confidence="" if laser_result is None else _fmt(laser_result.confidence),
                        laser_reason="" if laser_result is None else laser_result.reason,
                        laser_method="" if laser_result is None else laser_result.method,
                        laser_calibration_json=str(laser_calibration_path) if laser_enabled else "",

                        face_geometry_valid="" if face_result is None else int(face_result.valid),
                        face_geometry_yaw_deg=_fmt(face_yaw),
                        face_geometry_pitch_deg=_fmt(face_pitch),
                        face_geometry_axis_angle_deg=_fmt(face_axis),
                        face_geometry_axis_error_deg=_fmt(_err(target_axis_value, face_axis)),
                        face_geometry_yaw_error_deg=_fmt(_err(target_yaw, face_yaw)),
                        face_geometry_pitch_error_deg=_fmt(_err(target_pitch, face_pitch)),
                        face_geometry_distance_cm="" if face_result is None else _fmt(face_result.distance_cm),
                        face_geometry_ref_px="" if face_result is None else _fmt(face_result.ref_distance_px),
                        face_geometry_dx_px="" if face_result is None else _fmt(face_result.dx_px),
                        face_geometry_dy_px="" if face_result is None else _fmt(face_result.dy_px),
                        face_geometry_dx_cm="" if face_result is None else _fmt(face_result.dx_cm),
                        face_geometry_dy_cm="" if face_result is None else _fmt(face_result.dy_cm),
                        face_geometry_reason="" if face_result is None else face_result.reason,
                    )

                    left = head_protocol.record_seconds - (time.perf_counter() - record_start)

                    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                    canvas[:] = (45, 45, 45)

                    canvas = draw_text(canvas, f"Блок {block_idx}/{len(head_protocol.targets)}: {target.label}", (40, 40), 30)
                    canvas = draw_text(canvas, f"Теория: yaw={target_yaw:.1f}° pitch={target_pitch:.1f}°", (40, 85), 30, color=(255, 255, 0))
                    canvas = draw_text(canvas, f"Запись: осталось {left:.1f} сек", (40, 130), 28, color=(0, 255, 0))

                    canvas = draw_text(canvas, "Сравнение методов:", (40, 190), 30)
                    canvas = draw_text(
                        canvas,
                        f"Лазер: axis={_fmt(laser_axis, 1)}° | error={_fmt(_err(target_axis_value, laser_axis), 1)}°",
                        (40, 235),
                        28,
                        color=(0, 255, 255) if laser_result is not None and laser_result.valid else (0, 120, 255),
                    )
                    canvas = draw_text(
                        canvas,
                        f"Без лазера: axis={_fmt(face_axis, 1)}° | error={_fmt(_err(target_axis_value, face_axis), 1)}°",
                        (40, 280),
                        28,
                        color=(180, 255, 180) if face_result is not None and face_result.valid else (0, 120, 255),
                    )

                    if laser_result is not None:
                        canvas = draw_text(canvas, laser_direction_text(laser_result), (40, 330), 24)
                    canvas = draw_text(canvas, face_geometry_direction_text(face_result), (40, 365), 24)

                    # preview face camera
                    face_thumb = make_thumbnail(face_frame, (320, 240))
                    th, tw = face_thumb.shape[:2]
                    canvas[screen_h - th - 20:screen_h - 20, screen_w - tw - 20:screen_w - 20] = face_thumb

                    # preview laser camera
                    if laser_frame is not None and laser_result is not None:
                        laser_thumb = make_thumbnail(draw_laser_debug(laser_frame, laser_result), (320, 240))
                        lth, ltw = laser_thumb.shape[:2]
                        canvas[screen_h - th - lth - 40:screen_h - th - 40, screen_w - ltw - 20:screen_w - 20] = laser_thumb

                    cv2.imshow("HeadAnglesExperiment", canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("s"), ord("S")):
                        return

            done = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            done = draw_center_box(done, ["Эксперимент углов завершён", f"Файл: {log_path}"], font_size=42)
            cv2.imshow("HeadAnglesExperiment", done)
            cv2.waitKey(2000)

        finally:
            logger.close()
            print(f"Saved head angles log: {logger.path}")


if __name__ == "__main__":
    main()
