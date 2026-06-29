from __future__ import annotations

import random
import time
from dataclasses import dataclass


@dataclass
class TileTrial:
    trial_id: int
    row: int
    col: int
    started_perf: float
    duration_sec: float


class TileProtocol:
    def __init__(self, rows=3, cols=3, trial_seconds=2.0, seed=None):
        self.rows = int(rows)
        self.cols = int(cols)
        self.trial_seconds = float(trial_seconds)
        self.rng = random.Random(seed)
        self.cells = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        self.trial_id = 0
        r, c = self.rng.choice(self.cells)
        self.current = TileTrial(self.trial_id, r, c, time.perf_counter(), self.trial_seconds)

    def reset_timer(self):
        self.current.started_perf = time.perf_counter()

    def current_trial(self) -> TileTrial:
        return self.current

    def maybe_advance(self) -> tuple[TileTrial, bool]:
        now = time.perf_counter()
        if now - self.current.started_perf < self.current.duration_sec:
            return self.current, False
        finished = self.current
        candidates = [x for x in self.cells if x != (finished.row, finished.col)]
        r, c = self.rng.choice(candidates)
        self.trial_id += 1
        self.current = TileTrial(self.trial_id, r, c, now, self.trial_seconds)
        return finished, True
