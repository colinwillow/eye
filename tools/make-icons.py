#!/usr/bin/env python3
"""Generate every app icon from one drawing. No dependencies — Pillow is not
installed anywhere this runs, and a build step for eleven small PNGs is not
worth it, so this writes the PNG chunks itself.

    python3 tools/make-icons.py

Three things about icons that are easy to get wrong, all of them learned on
the other games in this account:

* Apple icons are FULL-BLEED with no rounded corners baked in. iOS applies its
  own squircle mask, and pre-rounded art shows its own corners as dark notches
  inside that mask. iOS also ignores the manifest and reads the
  <link rel="apple-touch-icon"> tags, which is why those sizes are listed
  separately from the PWA ones.
* Maskable icons need the subject in the MIDDLE, not necessarily a pad.
  Android crops to whatever the launcher likes and only guarantees the middle
  80%. This art is an eye dead centre with the background bleeding to every
  edge, so the maskable variants are the same full-bleed art.
* Favicons get a tighter crop. At 16px the whole composition turns to mush;
  the small sizes zoom in on the iris so there is still a readable shape.
"""
import zlib, struct, math, os

OUT = os.path.join(os.path.dirname(__file__), '..', 'icons')

BG     = (0x07, 0x09, 0x0d)
BG2    = (0x10, 0x1a, 0x28)
SCLERA = (0xe8, 0xf6, 0xff)
IRIS   = (0x2f, 0x9f, 0xd8)
IRIS2  = (0x7c, 0xe0, 0xff)
PUPIL  = (0x05, 0x0b, 0x12)
LID    = (0x7c, 0xe0, 0xff)

SS = 4  # supersampling; there is no antialiasing otherwise and the lid reads as stairs


def write_png(path, w, h, pixels):
    raw = b''.join(b'\x00' + bytes(pixels[y * w * 3:(y + 1) * w * 3]) for y in range(h))
    def chunk(tag, data):
        c = tag + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    png = (b'\x89PNG\r\n\x1a\n'
           + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(raw, 9))
           + chunk(b'IEND', b''))
    open(path, 'wb').write(png)


def mix(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def shade(x, y, zoom):
    """Colour at (x, y) in a -1..1 square. zoom > 1 crops in for the favicons."""
    x, y = x / zoom, y / zoom

    # background: a soft radial so it is not a flat rectangle
    col = mix(BG2, BG, min(1.0, math.hypot(x, y) / 1.25))
    if abs(x) > 1 or abs(y) > 1:
        return col

    # The eye opening: an almond, the region between two arcs. The 0.62
    # exponent is what makes it an eye rather than an ellipse — it keeps the
    # corners sharp while the middle stays round.
    a, b = 0.86, 0.46
    if abs(x) < a:
        lid = b * (1 - (x / a) ** 2) ** 0.62
        if abs(y) < lid:
            col = SCLERA
            r = math.hypot(x, y * 1.06)
            if r < 0.40:                                    # iris
                col = mix(IRIS2, IRIS, r / 0.40)
                if r < 0.175:                               # pupil
                    col = PUPIL
            if math.hypot(x + 0.13, y + 0.14) < 0.075:      # highlight
                col = (255, 255, 255)
            # darken where the upper lid would shade the eye
            col = mix(col, (0.55 * col[0], 0.6 * col[1], 0.72 * col[2]),
                      max(0.0, (y + lid) / (2 * lid) * -1 + 0.34) * 1.5)
        elif abs(y) < lid + 0.055:                          # the lid line itself
            col = mix(col, LID, 0.85)
    return col


def render(size, zoom=1.0):
    n = size * SS
    px = bytearray(size * size * 3)
    # accumulate at SSxSS then box-filter down
    rows = []
    for j in range(n):
        y = (j + 0.5) / n * 2 - 1
        row = [shade((i + 0.5) / n * 2 - 1, y, zoom) for i in range(n)]
        rows.append(row)
    for oy in range(size):
        for ox in range(size):
            r = g = bl = 0.0
            for j in range(SS):
                for i in range(SS):
                    c = rows[oy * SS + j][ox * SS + i]
                    r += c[0]; g += c[1]; bl += c[2]
            k = SS * SS
            o = (oy * size + ox) * 3
            px[o] = int(max(0, min(255, r / k)))
            px[o + 1] = int(max(0, min(255, g / k)))
            px[o + 2] = int(max(0, min(255, bl / k)))
    return px


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [
        # iOS reads these tags, not the manifest. Full bleed, no rounding.
        ('apple-touch-icon-180.png', 180, 1.0),
        ('apple-touch-icon-167.png', 167, 1.0),
        ('apple-touch-icon-152.png', 152, 1.0),
        ('apple-touch-icon-120.png', 120, 1.0),
        # PWA / Android. The maskable variants are the same art: the subject is
        # already centred and the background already bleeds to the edge.
        ('icon-192.png', 192, 1.0),
        ('icon-512.png', 512, 1.0),
        # Favicons, cropped in so 16px is still an eye and not a smudge.
        ('favicon-48.png', 48, 1.45),
        ('favicon-32.png', 32, 1.55),
        ('favicon-16.png', 16, 1.7),
    ]
    for name, size, zoom in jobs:
        write_png(os.path.join(OUT, name), size, size, render(size, zoom))
        print(f'  {name}  {size}x{size}')


if __name__ == '__main__':
    main()
