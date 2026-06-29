from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from eyegaze.tracking.laser_tracker import LaserTracker, draw_laser_debug
from eyegaze.ui.draw import draw_center_box, draw_text, make_thumbnail

ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
}


def _detect_wall_corners_by_aruco(frame_bgr: np.ndarray, laser_cfg: dict[str, Any]):
    """Returns wall corner centers in order TL, TR, BR, BL, or None."""
    aruco_cfg = laser_cfg.get("aruco", {})
    ids_expected = [int(x) for x in aruco_cfg.get("corner_ids", [0, 1, 3, 2])]
    dict_name = aruco_cfg.get("dictionary", "DICT_4X4_50")
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS.get(dict_name, cv2.aruco.DICT_4X4_50))

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    try:
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
    except AttributeError:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)

    if ids is None:
        return None, []

    ids_list = ids.flatten().astype(int).tolist()
    centers = {}
    for marker_corners, marker_id in zip(corners, ids_list):
        pts = marker_corners.reshape(-1, 2)
        centers[int(marker_id)] = pts.mean(axis=0)

    if not all(marker_id in centers for marker_id in ids_expected):
        return None, ids_list

    ordered = np.array([centers[marker_id] for marker_id in ids_expected], dtype=np.float32)
    return ordered, ids_list


def _read(cap):
    ok, frame = cap.read()
    return frame if ok else None


def run_laser_runtime_calibration(
    cap,
    tracker: LaserTracker,
    cfg: dict[str, Any],
    window_name: str,
    screen_w: int,
    screen_h: int,
    *,
    save: bool = True,
) -> LaserTracker:
    """
    Runtime-калибровка лазера без ручной калибровки углов.

    Что делает:
    - при желании A считывает ArUco и настраивает перспективу стены;
    - SPACE/Enter/C сохраняет текущую точку лазера как 0° головы;
    - дальше углы считаются только по пиксельному смещению и atan().
    """
    laser_cfg = cfg.get("laser", {})
    wall_size_cm = tuple(laser_cfg.get("wall_size_cm", [100.0, 70.0]))
    out_path = Path(laser_cfg.get("calibration_json", "data/calibration/laser_wall.json"))
    require_aruco = bool(laser_cfg.get("require_aruco", False))
    auto_continue_after_center = bool(laser_cfg.get("auto_continue_after_center", False))

    message = "Смотрите прямо. SPACE/Enter = сохранить ноль. A = считать ArUco. ESC = пропустить."
    last_seen_ids: list[int] = []
    homography_saved = tracker.homography is not None

    while True:
        frame = _read(cap)
        if frame is None:
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            canvas = draw_center_box(canvas, ["Задняя камера не отдаёт кадры", "Проверьте laser.camera.index", "ESC — пропустить"], font_size=36)
            cv2.imshow(window_name, canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                return tracker
            continue

        detection = tracker.detect(frame)
        debug = draw_laser_debug(frame, detection)
        if last_seen_ids:
            cv2.putText(debug, f"ArUco: {last_seen_ids}", (20, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        canvas[:] = (25, 25, 25)
        lines = [
            "Калибровка лазера",
            "1) Наденьте лазер и смотрите прямо.",
            "2) Убедитесь, что задняя камера видит красную точку.",
            "3) SPACE/Enter — сохранить текущую точку как 0° головы.",
            "4) A — считать ArUco, если маркеры есть.",
            "",
            "Углы считаются по пикселям: atan((px * cm_per_px) / distance).",
            f"distance={tracker.wall_distance_cm:.1f} cm, scale X={tracker.px_to_cm_x:.4f} cm/px, Y={tracker.px_to_cm_y:.4f} cm/px",
            message,
        ]
        canvas = draw_center_box(canvas, lines, font_size=30)
        thumb = make_thumbnail(debug, (520, 390))
        th, tw = thumb.shape[:2]
        x1 = screen_w - tw - 35
        y1 = screen_h - th - 35
        if x1 >= 0 and y1 >= 0:
            canvas[y1:y1 + th, x1:x1 + tw] = thumb

        status = "laser OK" if detection.valid else f"laser: {detection.reason}"
        canvas = draw_text(canvas, status, (35, screen_h - 42), 26, color=(0, 255, 255) if detection.valid else (0, 120, 255))
        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return tracker

        if key in (ord("a"), ord("A")):
            corners, ids = _detect_wall_corners_by_aruco(frame, laser_cfg)
            last_seen_ids = ids
            if corners is None:
                message = f"Не вижу все ArUco. Найдены IDs: {ids}. Можно продолжить без них."
            else:
                tracker.set_wall_homography(corners, wall_size_cm)
                homography_saved = True
                message = "ArUco считаны: перспектива стены настроена. Теперь SPACE для нуля."
            continue

        if key in (32, 13, ord("c"), ord("C")):
            if require_aruco and not homography_saved:
                message = "Включён require_aruco=true. Сначала нажмите A и считайте 4 маркера."
                continue
            if detection.valid:
                tracker.set_center_from_detection(detection)
                if save:
                    tracker.save_calibration(out_path)
                    print(f"[laser] saved center calibration: {out_path}")
                message = "Ноль сохранён. Углы считаются по пикселям. Продолжаю..."
                time.sleep(0.35)
                if auto_continue_after_center or True:
                    return tracker
            else:
                message = "Красная точка не найдена. Наведите лазер на стену и повторите."
