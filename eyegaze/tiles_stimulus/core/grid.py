from __future__ import annotations

from dataclasses import dataclass
from eyegaze.tiles_stimulus import config


@dataclass
class Tile:
    id: int
    row: int = 0
    col: int = 0
    active: bool = False
    x: float = 0.0
    y: float = 0.0
    x_norm: float = 0.5
    y_norm: float = 0.5


class Grid:
    def __init__(self, size: int = 3):
        total = int(getattr(config, "TILE_TOTAL", int(size) * int(size)))
        self.size = int(size)
        self.tiles = [Tile(id=i, row=i // max(1, int(size)), col=i % max(1, int(size))) for i in range(total)]

    def clear(self):
        for tile in self.tiles:
            tile.active = False
