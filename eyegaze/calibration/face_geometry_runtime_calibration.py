from __future__ import annotations

import time
import cv2
import numpy as np

from eyegaze.ui.draw import draw_center_box, draw_text, make_thumbnail


def _extract_landmarks(process_result):
    """
    Supports MediaPipe FaceMesh output.

    MediaPipe returns:
        result.multi_face_landmarks[0].landmark

    Older project wrappers sometimes returned:
        result.landmarks
    """
    if process_result is None:
        return None

    if hasattr(process_result, "landmarks"):
        return process_result.landmarks

    if hasattr(process_result, "multi_face_landmarks"):
        faces = process_result.multi_face_landmarks
        if faces:
            return faces[0].landmark

    return None


def run_face_geometry_runtime_calibration(
    cap,
    face_head,
    window_name: str,
    screen_w: int,
    screen_h: int,
    collect_seconds: float = 1.2,
):
    """
    Runtime neutral calibration for face_geometry.

    User looks straight ahead and presses SPACE.
    Then we collect a short batch of frames and save neutral:
      - nose offset from eye center
      - reference eye/brow distance

    This experiment does NOT calibrate gaze.
    """

    while True:
        ok, frame = cap.read()
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_head.tracker.process(rgb) if face_head.tracker is not None else None
            lm = _extract_landmarks(res)

            if lm is not None:
                dx, dy, ref = face_head._geometry_from_landmarks(lm, frame.shape)
                canvas = draw_center_box(
                    canvas,
                    [
                        "Калибровка углов БЕЗ лазера",
                        "Смотрите прямо в камеру",
                        "Нажмите SPACE",
                        f"face found | dx={dx:.1f}px dy={dy:.1f}px ref={ref:.1f}px",
                    ],
                    font_size=38,
                )
            else:
                canvas = draw_center_box(
                    canvas,
                    [
                        "Калибровка углов БЕЗ лазера",
                        "Лицо не найдено",
                        "Смотрите прямо в камеру",
                    ],
                    font_size=38,
                )

            thumb = make_thumbnail(frame, (320, 240))
            th, tw = thumb.shape[:2]
            canvas[screen_h - th - 20:screen_h - 20, screen_w - tw - 20:screen_w - 20] = thumb
        else:
            canvas = draw_center_box(canvas, ["Камера лица не дала кадр"], font_size=42)

        canvas = draw_text(canvas, "SPACE — сохранить neutral face geometry | ESC — выход", (40, screen_h - 40), 24)
        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            raise RuntimeError("Face geometry calibration cancelled")
        if key == 32:
            break

    samples = []
    start = time.perf_counter()

    while time.perf_counter() - start < collect_seconds:
        ok, frame = cap.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_head.tracker.process(rgb) if face_head.tracker is not None else None
            lm = _extract_landmarks(res)

            if lm is not None:
                try:
                    samples.append(face_head._geometry_from_landmarks(lm, frame.shape))
                except Exception:
                    pass

        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        left = collect_seconds - (time.perf_counter() - start)
        canvas = draw_center_box(
            canvas,
            [
                "Сохраняю neutral face geometry...",
                f"осталось {left:.1f} сек",
                f"samples: {len(samples)}",
            ],
            font_size=42,
        )
        cv2.imshow(window_name, canvas)
        cv2.waitKey(1)

    if not face_head.set_neutral_from_samples(samples):
        raise RuntimeError("Face geometry calibration failed: not enough valid face samples")

    return True
