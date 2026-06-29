from __future__ import annotations

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_font(size: int = 42):
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_text(img_bgr, text: str, xy: tuple[int, int], font_size: int = 32, color=(255, 255, 255)):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    font = load_font(font_size)
    draw.text(xy, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_center_box(img_bgr, lines: list[str], font_size: int = 46):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb).convert("RGBA")
    draw = ImageDraw.Draw(pil)
    font = load_font(font_size)
    W, H = pil.size
    spacing = 18
    sizes = []
    max_w = 0
    total_h = 0

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        sizes.append((tw, th))
        max_w = max(max_w, tw)
        total_h += th
    total_h += spacing * (len(lines) - 1)

    x0 = int(W / 2 - max_w / 2)
    y0 = int(H / 2 - total_h / 2)

    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    pad_x, pad_y = 45, 32
    od.rounded_rectangle(
        (x0 - pad_x, y0 - pad_y, x0 + max_w + pad_x, y0 + total_h + pad_y),
        radius=28,
        fill=(20, 20, 20, 220),
        outline=(255, 255, 255, 255),
        width=3,
    )
    pil = Image.alpha_composite(pil, overlay)
    draw = ImageDraw.Draw(pil)

    y = y0
    for line, (tw, th) in zip(lines, sizes):
        x = int(W / 2 - tw / 2)
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        y += th + spacing

    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def draw_grid_3x3(canvas, active_cell=None, rows=3, cols=3):
    h, w = canvas.shape[:2]
    cell_w = w // cols
    cell_h = h // rows
    overlay = canvas.copy()

    for r in range(rows):
        for c in range(cols):
            x1, y1 = c * cell_w, r * cell_h
            x2 = (c + 1) * cell_w if c < cols - 1 else w
            y2 = (r + 1) * cell_h if r < rows - 1 else h
            color = (35, 35, 35)
            if active_cell == (r, c):
                color = (255, 255, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)

    for c in range(1, cols):
        cv2.line(canvas, (c * cell_w, 0), (c * cell_w, h), (180, 180, 180), 2)
    for r in range(1, rows):
        cv2.line(canvas, (0, r * cell_h), (w, r * cell_h), (180, 180, 180), 2)

    n = 1
    for r in range(rows):
        for c in range(cols):
            canvas = draw_text(canvas, str(n), (c * cell_w + 24, r * cell_h + 18), 34)
            n += 1
    return canvas


def draw_cursor(canvas, x: int | None, y: int | None, alpha: float = 1.0):
    if x is None or y is None:
        return canvas
    overlay = canvas.copy()
    cv2.circle(overlay, (int(x), int(y)), 28, (0, 0, 255), -1)
    cv2.circle(overlay, (int(x), int(y)), 20, (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.55 * alpha, canvas, 1 - 0.55 * alpha, 0, canvas)
    return canvas


def make_thumbnail(frame, size=(320, 240), border=2):
    img = cv2.resize(frame, size)
    return cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, value=(255, 255, 255))
