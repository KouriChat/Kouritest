import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image

LON0, LON1 = 129.0, 137.0
LAT0, LAT1 = 33.0, 35.0
W, H = 253, 64

def poly(points):
    return [(float(x), float(y)) for x, y in points]

def ellipse(cx, cy, rx, ry, n=16, phase=0.0):
    return [(cx + rx * math.cos(phase + 2 * math.pi * k / n),
             cy + ry * math.sin(phase + 2 * math.pi * k / n))
            for k in range(n)]

def inside(x, y, p):
    hit = False
    j = len(p) - 1
    for i, (xi, yi) in enumerate(p):
        xj, yj = p[j]
        if ((yi > y) != (yj > y)):
            q = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < q:
                hit = not hit
        j = i
    return hit

# The large land masses are deliberately hand-drawn geographic approximations.
# Their lower boundaries are the coastlines facing the Seto Inland Sea.
HONSHU = poly([
    (129.35, 34.12), (129.75, 34.14), (130.05, 34.22),
    (130.35, 34.28), (130.62, 34.32), (130.88, 34.39),
    (131.12, 34.39), (131.35, 34.34), (131.62, 34.30),
    (131.88, 34.34), (132.12, 34.40), (132.36, 34.42),
    (132.60, 34.47), (132.87, 34.47), (133.10, 34.43),
    (133.32, 34.43), (133.55, 34.44), (133.77, 34.53),
    (134.00, 34.61), (134.25, 34.62), (134.47, 34.66),
    (134.68, 34.72), (134.88, 34.67), (135.02, 34.62),
    (134.88, 34.47), (134.75, 34.28), (134.78, 34.10),
    (134.91, 33.91), (135.10, 33.68), (135.28, 33.49),
    (135.53, 33.40), (135.77, 33.43), (135.94, 33.59),
    (136.08, 33.82), (136.22, 34.01), (136.32, 34.22),
    (136.45, 34.41), (136.63, 34.55), (136.86, 34.58),
    (137.00, 34.72), (137.02, 36.0), (129.35, 36.0)
])

KYUSHU = poly([
    (129.46, 34.08), (129.60, 34.35), (129.84, 34.54),
    (130.08, 34.69), (130.35, 34.77), (130.63, 34.75),
    (130.86, 34.60), (131.05, 34.37), (131.20, 34.13),
    (131.40, 33.93), (131.57, 33.69), (131.68, 33.43),
    (131.59, 33.20), (131.43, 33.06), (131.14, 32.94),
    (130.82, 32.94), (130.49, 33.05), (130.18, 33.23),
    (129.92, 33.45), (129.70, 33.68), (129.52, 33.87)
])

SHIKOKU = poly([
    (131.88, 33.86), (132.13, 34.00), (132.40, 34.07),
    (132.68, 34.18), (132.94, 34.31), (133.24, 34.38),
    (133.55, 34.37), (133.85, 34.31), (134.12, 34.20),
    (134.38, 34.09), (134.61, 33.95), (134.73, 33.76),
    (134.70, 33.55), (134.53, 33.37), (134.28, 33.25),
    (133.98, 33.15), (133.68, 33.05), (133.35, 32.99),
    (133.00, 33.00), (132.67, 33.06), (132.39, 33.16),
    (132.16, 33.31), (132.00, 33.53), (131.90, 33.70)
])

TSUSHIMA = poly([
    (129.08, 34.10), (129.13, 34.38), (129.27, 34.64),
    (129.44, 34.76), (129.56, 34.62), (129.56, 34.38),
    (129.44, 34.14), (129.28, 34.02)
])

IKI = poly([
    (129.62, 33.70), (129.69, 33.86), (129.83, 33.95),
    (129.98, 33.88), (130.00, 33.72), (129.91, 33.58),
    (129.76, 33.57)
])

HIRADO = poly([
    (129.43, 33.20), (129.52, 33.38), (129.67, 33.46),
    (129.76, 33.36), (129.70, 33.17), (129.56, 33.09)
])

LAND = [HONSHU, KYUSHU, SHIKOKU, TSUSHIMA, IKI, HIRADO]
# Major islands and the more conspicuous island groups in the inland sea.
AWAJI = poly([
    (134.63, 34.68), (134.82, 34.67), (135.02, 34.55),
    (135.10, 34.35), (135.02, 34.15), (134.87, 34.08),
    (134.71, 34.18), (134.62, 34.39), (134.57, 34.56)
])

SHODO = poly([
    (133.72, 34.52), (133.90, 34.64), (134.15, 34.63),
    (134.35, 34.54), (134.42, 34.39), (134.30, 34.27),
    (134.08, 34.25), (133.87, 34.34), (133.73, 34.42)
])

NAOSHIMA = ellipse(133.99, 34.46, .075, .055, 14)
TESHIMA = ellipse(134.08, 34.48, .060, .045, 13, .2)
INUJIMA = ellipse(134.09, 34.55, .045, .070, 12)
HIME = ellipse(134.52, 34.72, .055, .045, 12)

# Western inland-sea islands (Hiroshima, Yamaguchi and Ehime sides).
WEST_ISLANDS = [
    ellipse(131.72, 34.10, .10, .08),
    ellipse(131.89, 34.18, .12, .09),
    ellipse(132.08, 34.22, .11, .08),
    ellipse(132.20, 34.28, .09, .13),
    ellipse(132.31, 34.22, .15, .10),
    ellipse(132.42, 34.17, .13, .08),
    ellipse(132.53, 34.25, .11, .07),
    ellipse(132.66, 34.20, .10, .07),
    ellipse(132.76, 34.30, .10, .07),
    ellipse(132.85, 34.20, .11, .07),
    ellipse(132.96, 34.28, .12, .09),
    ellipse(133.07, 34.22, .13, .10),
    ellipse(133.18, 34.30, .13, .10),
    ellipse(133.29, 34.22, .12, .09),
    ellipse(133.39, 34.29, .10, .07),
    ellipse(133.50, 34.23, .10, .06),
    ellipse(133.61, 34.30, .08, .06),
    ellipse(133.75, 34.24, .07, .05),
    ellipse(133.86, 34.18, .08, .06),
    ellipse(132.12, 33.99, .10, .06),
    ellipse(132.30, 33.98, .09, .06),
    ellipse(132.53, 33.99, .08, .05),
]

EAST_ISLANDS = [
    ellipse(134.56, 34.02, .09, .055),
    ellipse(134.72, 34.05, .07, .05),
    ellipse(134.87, 34.00, .06, .045),
    ellipse(135.17, 34.19, .07, .05),
    ellipse(135.35, 34.28, .07, .05),
    ellipse(135.53, 34.35, .06, .045),
    ellipse(135.72, 34.40, .07, .05),
]

SMALL_ISLANDS = [
    ellipse(130.93, 33.94, .035, .025, 10),
    ellipse(131.10, 34.00, .050, .030, 10),
    ellipse(131.40, 34.03, .045, .030, 10),
    ellipse(131.58, 34.16, .040, .028, 10),
    ellipse(131.78, 34.28, .045, .025, 10),
    ellipse(132.00, 34.33, .035, .025, 10),
    ellipse(132.58, 34.38, .050, .028, 10),
    ellipse(132.98, 34.39, .045, .030, 10),
    ellipse(133.25, 34.43, .050, .025, 10),
    ellipse(133.54, 34.45, .045, .025, 10),
    ellipse(133.72, 34.51, .040, .025, 10),
    ellipse(134.46, 34.54, .045, .028, 10),
    ellipse(134.57, 34.45, .035, .025, 10),
    ellipse(134.34, 33.99, .050, .030, 10),
    ellipse(134.48, 33.93, .040, .025, 10),
]

ISLANDS = [
    AWAJI, SHODO, NAOSHIMA, TESHIMA,
    INUJIMA, HIME
] + WEST_ISLANDS + EAST_ISLANDS + SMALL_ISLANDS
# A small clipped corner of the Korean southeast coast is within the frame.
KOREA_EDGE = poly([
    (128.55, 35.04), (129.18, 35.04), (129.22, 34.96),
    (129.15, 34.88), (128.86, 34.83), (128.55, 34.84)
])
LAND.append(KOREA_EDGE)

def is_land(lon, lat):
    for shape in LAND:
        if inside(lon, lat, shape):
            return 1
    for shape in ISLANDS:
        if inside(lon, lat, shape):
            return 1
    return 0

def build_matrix():
    a = np.zeros((H, W), dtype=np.uint8)
    for r in range(H):
        lat = LAT1 - (LAT1 - LAT0) * r / (H - 1)
        for c in range(W):
            lon = LON0 + (LON1 - LON0) * c / (W - 1)
            a[r, c] = is_land(lon, lat)
    return a

def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    a = build_matrix()
    assert a.shape == (64, 253)
    assert int(a.sum() + (a == 0).sum()) == 16192
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(a.tolist())

    # Explicit grayscale values make the two-color convention unambiguous.
    raw = Image.fromarray((a * 255).astype(np.uint8), mode="L")
    raw.save(out / "candidate.png")
    enlarged = raw.resize((W * 8, H * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(out / "candidate_8x.png")

    # Internal checks corresponding to the requested deliverables.
    assert raw.size == (253, 64)
    assert enlarged.size == (2024, 512)
    assert set(np.asarray(raw).ravel()).issubset({0, 255})

if __name__ == "__main__":
    main()
