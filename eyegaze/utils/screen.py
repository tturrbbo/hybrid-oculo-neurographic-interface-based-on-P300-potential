from __future__ import annotations

from screeninfo import get_monitors


def get_screen_size() -> tuple[int, int]:
    m = get_monitors()[0]
    return int(m.width), int(m.height)
