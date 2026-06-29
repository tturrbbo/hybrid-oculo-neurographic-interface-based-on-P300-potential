from __future__ import annotations

from collections import deque
import bisect
import statistics


class HeadAngleStabilizer:
    def __init__(self):
        self.yaw_hist = deque(maxlen=3)
        self.pitch_hist = deque(maxlen=3)
        self.yaw_smooth = None
        self.pitch_smooth = None

    def _axis(self, raw, hist, prev):
        raw = float(raw)
        hist.append(raw)
        med = float(statistics.median(hist))

        if prev is None:
            smooth = med
        else:
            diff = med - prev

            # Быстро реагируем, но убираем одиночные скачки.
            max_step = 22.0
            if diff > max_step:
                med = prev + max_step
            elif diff < -max_step:
                med = prev - max_step

            diff = abs(med - prev)
            alpha = 0.82 if diff >= 4.0 else 0.55
            smooth = prev * (1.0 - alpha) + med * alpha

        if abs(smooth) < 0.6:
            smooth = 0.0

        return float(smooth)

    def update(self, yaw, pitch):
        self.yaw_smooth = self._axis(yaw, self.yaw_hist, self.yaw_smooth)
        self.pitch_smooth = self._axis(pitch, self.pitch_hist, self.pitch_smooth)
        return self.yaw_smooth, self.pitch_smooth


_STABILIZER = HeadAngleStabilizer()


def reset_head_deviation_filter():
    _STABILIZER.yaw_hist.clear()
    _STABILIZER.pitch_hist.clear()
    _STABILIZER.yaw_smooth = None
    _STABILIZER.pitch_smooth = None


def _clean(points):
    pts = sorted((float(x), float(y)) for x, y in points)
    out = []
    for x, y in pts:
        if out and abs(out[-1][0] - x) < 1e-5:
            px, py = out[-1]
            out[-1] = ((px + x) / 2.0, (py + y) / 2.0)
        else:
            out.append((x, y))
    return out


def _interp(raw, points):
    pts = _clean(points)
    raw = float(raw)

    if not pts:
        return raw
    if len(pts) == 1:
        x0, y0 = pts[0]
        return raw if abs(x0) < 1e-6 else raw * (y0 / x0)

    xs = [p[0] for p in pts]

    if raw <= xs[0]:
        x0, y0 = pts[0]
        x1, y1 = pts[1]
    elif raw >= xs[-1]:
        x0, y0 = pts[-2]
        x1, y1 = pts[-1]
    else:
        i = bisect.bisect_left(xs, raw)
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]

    if abs(x1 - x0) < 1e-6:
        return y0

    t = (raw - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def get_head_deviation(
    yaw_deg,
    pitch_deg,
    *,
    baseline_yaw_deg=None,
    baseline_pitch_deg=None,
    threshold_deg=1.8,
    invert_yaw=False,
    invert_pitch=False,
    head_angle_calibration=None,
):
    if yaw_deg is None or pitch_deg is None:
        return {
            "head_deviation_direction": "UNKNOWN",
            "head_deviation_angle_deg": None,
            "head_deviation_yaw_deg": None,
            "head_deviation_pitch_deg": None,
            "head_deviation_text_ru": "Отклонение головы: неизвестно",
            "head_angle_calibrated": bool(head_angle_calibration),
        }

    if baseline_yaw_deg is None:
        baseline_yaw_deg = yaw_deg
    if baseline_pitch_deg is None:
        baseline_pitch_deg = pitch_deg

    yaw_raw = float(yaw_deg) - float(baseline_yaw_deg)
    pitch_raw = float(pitch_deg) - float(baseline_pitch_deg)

    if invert_yaw:
        yaw_raw = -yaw_raw
    if invert_pitch:
        pitch_raw = -pitch_raw

    if head_angle_calibration:
        h = head_angle_calibration.get("horizontal_points", [])
        v = head_angle_calibration.get("vertical_points", [])
        if h:
            yaw_raw = _interp(yaw_raw, h)
        if v:
            pitch_raw = _interp(pitch_raw, v)

    yaw, pitch = _STABILIZER.update(yaw_raw, pitch_raw)

    if abs(yaw) < threshold_deg and abs(pitch) < threshold_deg:
        return {
            "head_deviation_direction": "CENTER",
            "head_deviation_angle_deg": 0.0,
            "head_deviation_yaw_deg": float(yaw),
            "head_deviation_pitch_deg": float(pitch),
            "head_deviation_text_ru": "Отклонение головы: центр 0°",
            "head_angle_calibrated": bool(head_angle_calibration),
        }

    if abs(yaw) >= abs(pitch):
        direction = "RIGHT" if yaw > 0 else "LEFT"
        ru = "вправо" if direction == "RIGHT" else "влево"
        angle = abs(yaw)
    else:
        direction = "UP" if pitch > 0 else "DOWN"
        ru = "вверх" if direction == "UP" else "вниз"
        angle = abs(pitch)

    return {
        "head_deviation_direction": direction,
        "head_deviation_angle_deg": float(angle),
        "head_deviation_yaw_deg": float(yaw),
        "head_deviation_pitch_deg": float(pitch),
        "head_deviation_text_ru": f"Отклонение головы: {ru} {angle:.1f}°",
        "head_angle_calibrated": bool(head_angle_calibration),
    }
