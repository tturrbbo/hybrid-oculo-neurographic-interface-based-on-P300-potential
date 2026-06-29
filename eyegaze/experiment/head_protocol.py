from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class HeadTarget:
    marker: int
    direction: str
    angle_deg: int

    @property
    def label(self) -> str:
        if self.angle_deg == 0:
            return "CENTER_0"
        return f"{self.direction}_{self.angle_deg}"

    @property
    def prompt_lines_ru(self) -> list[str]:
        if self.direction == "CENTER":
            return ["Поверните голову в центр", f"Смотрите на квадрат с номером {self.marker}"]

        direction_ru = {"RIGHT": "направо", "LEFT": "налево", "UP": "вверх"}.get(self.direction, self.direction)
        return [
            f"Поверните голову {direction_ru}",
            f"на {self.angle_deg} градусов",
            f"на квадрат с номером {self.marker}",
        ]


def targets_from_config(config: dict) -> list[HeadTarget]:
    targets = []
    for item in config.get("head_targets", []):
        targets.append(HeadTarget(int(item["marker"]), str(item["direction"]).upper(), int(item["angle_deg"])))
    return targets


class HeadProtocol:
    def __init__(self, targets: list[HeadTarget], prep_seconds=5.0, record_seconds=15.0, seed=None):
        self.targets = list(targets)
        random.Random(seed).shuffle(self.targets)
        self.prep_seconds = float(prep_seconds)
        self.record_seconds = float(record_seconds)
