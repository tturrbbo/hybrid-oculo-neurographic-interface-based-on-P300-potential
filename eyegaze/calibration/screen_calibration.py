from __future__ import annotations

import time
import cv2
import numpy as np

from eyegaze.ui.draw import draw_center_box, draw_text


def _camera_background(frame, screen_w: int, screen_h: int):
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)

    fh, fw = frame.shape[:2]
    if fw <= 0 or fh <= 0:
        return canvas

    scale = min(screen_w / fw, screen_h / fh)
    new_w = max(1, int(fw * scale))
    new_h = max(1, int(fh * scale))

    resized = cv2.resize(frame, (new_w, new_h))
    x1 = (screen_w - new_w) // 2
    y1 = (screen_h - new_h) // 2
    canvas[y1:y1 + new_h, x1:x1 + new_w] = resized
    return canvas


def _darken(canvas, alpha: float = 0.05):
    overlay = np.zeros_like(canvas)
    return cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0)


def _direction_ru(direction: str) -> str:
    d = str(direction).upper()
    if d == "CENTER":
        return "центр"
    if d == "LEFT":
        return "влево"
    if d == "RIGHT":
        return "вправо"
    if d == "UP":
        return "вверх"
    if d == "DOWN":
        return "вниз"
    return d


def _target_prompt(target: dict) -> list[str]:
    marker = target.get("marker")
    direction = str(target.get("direction", "")).upper()
    angle = float(target.get("angle_deg", 0.0))

    if direction == "CENTER":
        return [
            f"Метка {marker}: центр",
            "Сядьте ровно и смотрите на квадрат 1",
        ]

    return [
        f"Метка {marker}: поверните голову {_direction_ru(direction)}",
        f"Угол: {angle:.1f}°",
        f"Смотрите на квадрат {marker}",
    ]


def _collect_head_pose_samples(gaze, cap, seconds: float):
    yaw_samples = []
    pitch_samples = []
    start = time.time()

    while time.time() - start < seconds:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        yaw, pitch = gaze.get_head_pose_from_frame(frame)
        if yaw is not None and pitch is not None:
            yaw_samples.append(float(yaw))
            pitch_samples.append(float(pitch))

        cv2.waitKey(1)

    if len(yaw_samples) < 5 or len(pitch_samples) < 5:
        return None, None

    return float(np.median(yaw_samples)), float(np.median(pitch_samples))


def _run_head_marker_precalibration(
    gaze,
    cap,
    screen_w: int,
    screen_h: int,
    cfg: dict,
    window_name: str,
):
    pcfg = cfg.get("head_precalibration", {})
    if not pcfg or not pcfg.get("enabled", False):
        return

    targets = pcfg.get("targets", [])
    if not targets:
        return

    prepare_seconds = float(pcfg.get("prepare_seconds", 2.0))
    collect_seconds = float(pcfg.get("collect_seconds", 1.2))

    collected = []

    for idx, target in enumerate(targets, start=1):
        # 1) Экран подготовки.
        prep_start = time.time()
        while time.time() - prep_start < prepare_seconds:
            left = max(0.0, prepare_seconds - (time.time() - prep_start))
            ok, frame = cap.read()

            if ok and frame is not None:
                canvas = _camera_background(frame, screen_w, screen_h)
                canvas = _darken(canvas, 0.20)
            else:
                canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                canvas[:] = (25, 25, 25)

            lines = [
                "Предварительная калибровка углов головы",
                f"Позиция {idx}/{len(targets)}",
                "",
            ] + _target_prompt(target) + [
                "",
                f"Сбор начнётся через: {left:.1f} сек",
            ]

            canvas = draw_center_box(canvas, lines, font_size=38)
            cv2.imshow(window_name, canvas)

            if cv2.waitKey(1) & 0xFF == 27:
                raise RuntimeError("Калибровка остановлена пользователем")

        # 2) Сбор стабильных yaw/pitch.
        collect_start = time.time()
        yaw_samples = []
        pitch_samples = []

        while time.time() - collect_start < collect_seconds:
            left = max(0.0, collect_seconds - (time.time() - collect_start))
            ok, frame = cap.read()

            if ok and frame is not None:
                yaw, pitch = gaze.get_head_pose_from_frame(frame)
                if yaw is not None and pitch is not None:
                    yaw_samples.append(float(yaw))
                    pitch_samples.append(float(pitch))

                canvas = _camera_background(frame, screen_w, screen_h)
                canvas = _darken(canvas, 0.20)
            else:
                canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                canvas[:] = (25, 25, 25)

            lines = [
                "Не двигайтесь",
                "Идёт сбор угла головы",
                "",
            ] + _target_prompt(target) + [
                "",
                f"Осталось: {left:.1f} сек",
                f"Кадров собрано: {len(yaw_samples)}",
            ]

            canvas = draw_center_box(canvas, lines, font_size=38)
            cv2.imshow(window_name, canvas)

            if cv2.waitKey(1) & 0xFF == 27:
                raise RuntimeError("Калибровка остановлена пользователем")

        if len(yaw_samples) >= 5 and len(pitch_samples) >= 5:
            yaw_med = float(np.median(yaw_samples))
            pitch_med = float(np.median(pitch_samples))

            sample = {
                "marker": target.get("marker"),
                "direction": target.get("direction"),
                "angle_deg": float(target.get("angle_deg", 0.0)),
                "yaw_deg": yaw_med,
                "pitch_deg": pitch_med,
            }
            collected.append(sample)

            if str(target.get("direction", "")).upper() == "CENTER":
                gaze.set_head_pose_baseline([yaw_med], [pitch_med])

    # Если центр почему-то не был задан, берём marker 1.
    if gaze.baseline_yaw_deg is None or gaze.baseline_pitch_deg is None:
        center_samples = [s for s in collected if str(s.get("direction", "")).upper() == "CENTER"]
        if center_samples:
            c = center_samples[0]
            gaze.set_head_pose_baseline([c["yaw_deg"]], [c["pitch_deg"]])

    if collected:
        gaze.set_head_angle_calibration_from_marker_samples(collected)

    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    canvas[:] = (25, 25, 25)
    canvas = draw_center_box(
        canvas,
        [
            "Калибровка углов головы завершена",
            "Теперь начнётся калибровка взгляда",
        ],
        font_size=42,
    )
    cv2.imshow(window_name, canvas)
    cv2.waitKey(900)


def _wait_for_face_and_distance_baseline(
    gaze,
    cap,
    screen_w: int,
    screen_h: int,
    window_name: str,
    wait_seconds: float = 2.0,
):
    stable_start = None
    baseline_px = None
    last_meta = {}

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            canvas = draw_center_box(
                canvas,
                [
                    "Камера не отдаёт кадры",
                    "Проверьте iVCam и индекс камеры",
                    "ESC — выход",
                ],
                font_size=40,
            )
            cv2.imshow(window_name, canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                raise RuntimeError("Калибровка остановлена пользователем")
            continue

        canvas = _camera_background(frame, screen_w, screen_h)
        canvas = _darken(canvas, 0.05)

        features, blink, meta = gaze.extract_features_meta(frame)
        last_meta = meta
        face_seen = bool(meta.get("face_found", False))

        if face_seen:
            if stable_start is None:
                stable_start = time.time()

            try:
                px = gaze.calibrate_distance_baseline_from_frame(frame)
                if px is not None:
                    baseline_px = px
            except Exception:
                pass

            elapsed = time.time() - stable_start
            left = max(0.0, wait_seconds - elapsed)

            canvas = draw_center_box(
                canvas,
                [
                    "Лицо найдено",
                    "Сядьте ровно на расстоянии baseline",
                    "Смотрите в центр экрана",
                    f"До продолжения: {left:.1f} сек",
                ],
                font_size=38,
            )

            if elapsed >= wait_seconds and baseline_px is not None:
                return baseline_px
        else:
            stable_start = None
            baseline_px = None
            canvas = draw_center_box(
                canvas,
                [
                    "Лицо не найдено",
                    "Камера сейчас показана фоном",
                    "Если видео видно — поднесите лицо в кадр",
                ],
                font_size=38,
            )

        status = "FACE: OK" if face_seen else "FACE: NOT FOUND"
        color = (0, 255, 0) if face_seen else (0, 0, 255)
        canvas = draw_text(canvas, status, (40, 40), 34, color=color)

        canvas = draw_text(
            canvas,
            f"baseline_eye_px: {baseline_px:.1f}" if baseline_px is not None else "baseline_eye_px: ?",
            (40, 90),
            28,
        )

        dist = last_meta.get("distance_cm")
        if dist is not None:
            canvas = draw_text(canvas, f"distance_cm: {dist:.1f}", (40, 130), 28)

        canvas = draw_text(canvas, "ESC — выход", (40, screen_h - 55), 26)

        cv2.imshow(window_name, canvas)
        if cv2.waitKey(1) & 0xFF == 27:
            raise RuntimeError("Калибровка остановлена пользователем")


def run_screen_calibration(
    gaze,
    cap,
    screen_w: int,
    screen_h: int,
    cfg: dict,
    window_name: str = "Calibration",
):
    points_rel = cfg["calibration"]["points"]
    samples_per_point = int(cfg["calibration"].get("samples_per_point", 35))
    seconds_per_point = float(cfg["calibration"].get("seconds_per_point", 2.0))

    X = []
    y = []

    # Робастная калибровка:
    # для каждой точки сначала собираем признаки во временный список,
    # потом отбрасываем выбросы и добавляем небольшое "облако" вокруг медианы.
    # Это даёт +- устойчивость к микрошуму landmarks/радужки.
    robust_cfg = cfg.get("gaze_calibration_robust", {})
    robust_enabled = bool(robust_cfg.get("enabled", True))
    augmentation_count = int(robust_cfg.get("augmentation_count", 4))
    augmentation_noise = float(robust_cfg.get("augmentation_noise", 0.006))
    outlier_mad_k = float(robust_cfg.get("outlier_mad_k", 3.0))
    rng = np.random.default_rng(int(robust_cfg.get("random_seed", 42)))

    def _append_calibration_point(point_features, tx, ty):
        if not point_features:
            return 0

        arr = np.asarray(point_features, dtype=np.float32)

        if robust_enabled and len(arr) >= 5:
            med = np.median(arr, axis=0)
            dist = np.linalg.norm(arr - med, axis=1)
            mad = np.median(np.abs(dist - np.median(dist))) + 1e-6
            keep = dist <= (np.median(dist) + outlier_mad_k * mad)
            clean = arr[keep]
            if len(clean) < 3:
                clean = arr
        else:
            clean = arr

        # Добавляем очищенные реальные samples.
        for row in clean:
            X.append(row.astype(np.float32))
            y.append([tx, ty])

        # Добавляем медиану + лёгкий шум, чтобы модель не реагировала
        # слишком резко на минимальные изменения признаков.
        if robust_enabled and augmentation_count > 0:
            med = np.median(clean, axis=0).astype(np.float32)
            scale = np.maximum(np.std(clean, axis=0), augmentation_noise).astype(np.float32)

            X.append(med)
            y.append([tx, ty])

            for _ in range(augmentation_count):
                noise = rng.normal(0.0, augmentation_noise, size=med.shape).astype(np.float32)
                # Для разных признаков шум ограничиваем scale, чтобы не создавать нереальные точки.
                noise = np.clip(noise, -scale, scale)
                X.append((med + noise).astype(np.float32))
                y.append([tx, ty])

        return len(clean)

    _wait_for_face_and_distance_baseline(
        gaze=gaze,
        cap=cap,
        screen_w=screen_w,
        screen_h=screen_h,
        window_name=window_name,
        wait_seconds=2.0,
    )

    # Новое: калибровка головы по настенным меткам перед калибровкой взгляда.
    _run_head_marker_precalibration(
        gaze=gaze,
        cap=cap,
        screen_w=screen_w,
        screen_h=screen_h,
        cfg=cfg,
        window_name=window_name,
    )

    for idx, (rx, ry) in enumerate(points_rel, start=1):
        tx, ty = int(rx * screen_w), int(ry * screen_h)

        start = time.time()
        collected = 0
        point_features = []

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            features, blink, meta = gaze.extract_features_meta(frame)

            if features is not None and not blink:
                point_features.append(np.asarray(features, dtype=np.float32))
                collected += 1

            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            canvas[:] = (25, 25, 25)

            cv2.circle(canvas, (tx, ty), 30, (255, 255, 255), -1)
            cv2.circle(canvas, (tx, ty), 13, (0, 0, 255), -1)

            canvas = draw_text(canvas, f"Точка калибровки {idx}/{len(points_rel)}", (40, 40), 32)
            canvas = draw_text(canvas, f"Собрано кадров: {collected}/{samples_per_point}", (40, 85), 28)

            face_seen = bool(meta.get("face_found", False))
            blink_now = bool(blink)

            canvas = draw_text(
                canvas,
                "FACE: OK" if face_seen else "FACE: NOT FOUND",
                (40, 125),
                28,
                color=(0, 255, 0) if face_seen else (0, 0, 255),
            )
            canvas = draw_text(
                canvas,
                "BLINK: YES" if blink_now else "BLINK: NO",
                (40, 165),
                28,
                color=(0, 0, 255) if blink_now else (0, 255, 0),
            )

            dist = meta.get("distance_cm")
            if dist is not None:
                canvas = draw_text(canvas, f"distance_cm: {dist:.1f}", (40, 205), 28)

            calibrated = "YES" if meta.get("head_angle_calibrated") else "NO"
            canvas = draw_text(canvas, f"HEAD ANGLE CALIBRATED: {calibrated}", (40, 245), 24)

            canvas = draw_text(canvas, "Смотрите на красную точку. ESC — выход.", (40, screen_h - 55), 26)

            cv2.imshow(window_name, canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                raise RuntimeError("Калибровка остановлена пользователем")

            elapsed = time.time() - start
            if collected >= samples_per_point and elapsed >= seconds_per_point:
                _append_calibration_point(point_features, tx, ty)
                break

    if len(X) < 10:
        raise RuntimeError(
            "Недостаточно данных для калибровки. "
            "Проверьте камеру, освещение и видимость лица."
        )

    gaze.train(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32))

    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    canvas[:] = (25, 25, 25)
    canvas = draw_center_box(canvas, ["Калибровка завершена"], font_size=50)
    cv2.imshow(window_name, canvas)
    cv2.waitKey(900)

    return gaze
