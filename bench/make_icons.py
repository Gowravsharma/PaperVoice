"""
make_icons.py - draw the extension icons (extension/icons/icon{16,32,48,128}.png): a page with sound waves.

Usage: uv run python bench/make_icons.py
Drawn at 8x and downsampled, so the small sizes stay crisp.
"""
from pathlib import Path

from PIL import Image, ImageDraw

ACCENT = (26, 95, 180, 255)  # --accent in viewer/viewer.css
WHITE = (255, 255, 255, 255)
OUT = Path(__file__).resolve().parents[1] / "extension" / "icons"


def draw(size):
    s = size * 8
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=s * 0.22, fill=ACCENT)
    # the page, with a folded corner
    l, t, r, b, fold = s * 0.18, s * 0.16, s * 0.56, s * 0.84, s * 0.12
    d.polygon([(l, t), (r - fold, t), (r, t + fold), (r, b), (l, b)], fill=WHITE)
    d.polygon([(r - fold, t), (r - fold, t + fold), (r, t + fold)], fill=(200, 215, 235, 255))
    line_w = max(s * 0.035, 8)
    for i, y in enumerate((0.34, 0.46, 0.58, 0.70)):
        d.line([(l + s * 0.06, s * y), (r - s * (0.06 if i else 0.16), s * y)], fill=ACCENT, width=int(line_w))
    # sound waves to the right of the page
    cx, cy = s * 0.60, s * 0.50
    for k, rad in enumerate((0.12, 0.22, 0.32)):
        rr = s * rad
        d.arc([cx - rr, cy - rr, cx + rr, cy + rr], start=-50, end=50, fill=WHITE, width=int(s * 0.055))
    return img.resize((size, size), Image.LANCZOS)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for size in (16, 32, 48, 128):
        draw(size).save(OUT / f"icon{size}.png")
    print(f"icons in {OUT}")


if __name__ == "__main__":
    main()
