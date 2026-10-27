"""Rysuje ikonę AI Bobra: assets/aibobr.png (okno) i assets/aibobr.ico (plik .exe).

    python tools/make_icon.py          # wymaga Pillow (pip install pillow) – tylko dla autora
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
BROWN, DARK, CREAM = (122, 74, 34), (84, 48, 20), (250, 240, 222)
FONTS = ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", "arialbd.ttf")


def font(size: int):
    for name in FONTS:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw(size: int = 512) -> Image.Image:
    s = size / 512
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((16 * s, 16 * s, 496 * s, 496 * s), radius=110 * s, fill=BROWN,
                        outline=DARK, width=max(1, int(14 * s)))

    # Litera na osobnej warstwie – jej prawdziwe granice mierzymy na pikselach,
    # bo metryki czcionki zawierają marginesy.
    letter = Image.new("L", (size, size), 0)
    ImageDraw.Draw(letter).text((0, 0), "B", font=font(int(290 * s)), fill=255)
    x0, y0, x1, y1 = letter.getbbox()
    glyph = letter.crop((x0, y0, x1, y1))
    gx, gy = int((size - glyph.width) / 2), int(58 * s)
    img.paste(Image.new("RGBA", glyph.size, CREAM), (gx, gy), glyph)

    # Dwa zęby bobra wyrastają spod litery (lekko na nią zachodzą).
    bottom, cx, tooth = gy + glyph.height, size / 2, 56 * s
    for left in (cx - tooth - 4 * s, cx + 4 * s):
        d.rounded_rectangle((left, bottom - 10 * s, left + tooth, bottom + 118 * s),
                            radius=14 * s, fill=CREAM, outline=DARK, width=max(1, int(7 * s)))
    return img


def main() -> None:
    big = draw(512)
    (ROOT / "assets").mkdir(exist_ok=True)
    big.resize((256, 256), Image.LANCZOS).save(ROOT / "assets" / "aibobr.png")
    big.save(ROOT / "assets" / "aibobr.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48),
                                                    (64, 64), (128, 128), (256, 256)])
    print("assets/aibobr.png, assets/aibobr.ico")


if __name__ == "__main__":
    main()
