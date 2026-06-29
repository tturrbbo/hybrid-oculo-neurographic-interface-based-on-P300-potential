from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class SmoothPoint:
    x: float
    y: float


class EMAGazeSmoother:
    """
    Адаптивное сглаживание.

    Идея:
    - маленькие дрожания сглаживаются сильнее;
    - большие реальные движения проходят быстрее;
    - точка меньше отстаёт от взгляда.
    """

    def __init__(
        self,
        alpha_min: float = 0.24,
        alpha_max: float = 0.62,
        fast_distance_px: float = 420.0,
        deadzone_px: float = 3.0,
    ):
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.fast_distance_px = float(fast_distance_px)
        self.deadzone_px = float(deadzone_px)
        self.point: SmoothPoint | None = None

    def reset(self):
        self.point = None

    def update(self, raw_xy: tuple[int, int] | None) -> tuple[int, int] | None:
        if raw_xy is None:
            return None

        raw_x, raw_y = float(raw_xy[0]), float(raw_xy[1])

        if self.point is None:
            self.point = SmoothPoint(raw_x, raw_y)
            return int(raw_x), int(raw_y)

        dx = raw_x - self.point.x
        dy = raw_y - self.point.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < self.deadzone_px:
            return int(self.point.x), int(self.point.y)

        t = min(1.0, dist / max(1.0, self.fast_distance_px))
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * t

        self.point.x = self.point.x * (1.0 - alpha) + raw_x * alpha
        self.point.y = self.point.y * (1.0 - alpha) + raw_y * alpha

        return int(self.point.x), int(self.point.y)
