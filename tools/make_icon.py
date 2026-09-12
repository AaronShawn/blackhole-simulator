#!/usr/bin/env python3
"""Generate assets/blackhole.ico (multi-size) procedurally with Pillow."""
import math
import os

from PIL import Image, ImageDraw, ImageFilter


def build(size: int = 512) -> Image.Image:
    img = Image.new("RGBA", (size, size), (5, 7, 12, 255))
    d = ImageDraw.Draw(img)

    cx = cy = size / 2
    # background stars
    for i in range(90):
        a = (i * 2.399963) % (2 * math.pi)
        rr = (0.18 + 0.8 * ((i * 0.6180339887) % 1.0)) * size * 0.5
        x = cx + math.cos(a) * rr
        y = cy + math.sin(a) * rr
        s = 1.0 if i % 5 else 1.8
        v = 150 + (i * 37) % 105
        d.ellipse([x - s, y - s, x + s, y + s], fill=(v, v, min(255, v + 20), 255))

    R = size * 0.42
    # accretion disk (tilted annulus), drawn as ellipses with warm colours
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for i, (w, col) in enumerate([
        (0.30, (120, 45, 8, 190)),
        (0.20, (230, 120, 30, 220)),
        (0.12, (255, 200, 120, 255)),
    ]):
        rx = R * (0.62 + 0.30 * i)
        ry = rx * 0.26
        wpx = max(1, int(size * w * 0.10))
        gd.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=col, width=wpx)
    glow = glow.filter(ImageFilter.GaussianBlur(size * 0.018))
    img.alpha_composite(glow)

    # photon ring
    ring = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ring)
    rp = size * 0.175
    rd.ellipse([cx - rp, cy - rp, cx + rp, cy + rp], outline=(255, 244, 214, 255),
               width=max(2, int(size * 0.016)))
    ring = ring.filter(ImageFilter.GaussianBlur(size * 0.006))
    img.alpha_composite(ring)

    # event horizon
    r = size * 0.166
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 0, 0, 255))
    d.ellipse([cx - r * 0.75, cy - r, cx + r * 0.75, cy + r], fill=(0, 0, 0, 255))
    return img


def main() -> None:
    os.makedirs("assets", exist_ok=True)
    base = build(512)
    base.save(
        os.path.join("assets", "blackhole.ico"),
        sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)],
    )
    base.resize((256, 256), Image.LANCZOS).save(os.path.join("assets", "blackhole.png"))
    print("wrote assets/blackhole.ico and assets/blackhole.png")


if __name__ == "__main__":
    main()
