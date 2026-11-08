"""Hand-drawn, knowledge-only reconstruction of the requested map area.

Coordinates are longitude, latitude.  The geometry is deliberately made from
small hand-written polygons and ellipses; no geographic data is read.
"""
import csv
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

WEST, EAST = 129.0, 137.0
NORTH, SOUTH = 35.0, 33.0
NX, NY = 253, 64
DX = (EAST - WEST) / (NX - 1)
DY = (NORTH - SOUTH) / (NY - 1)

def polygon(*points):
    return tuple((float(x), float(y)) for x, y in points)

def ellipse(cx, cy, rx, ry, n=32, angle=0.0):
    ca, sa = math.cos(angle), math.sin(angle)
    out = []
    for k in range(n):
        t = 2.0 * math.pi * k / n
        x, y = rx * math.cos(t), ry * math.sin(t)
        out.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    return tuple(out)

def inside(x, y, p):
    """Ray crossing test, including a harmless boundary tolerance."""
    hit = False
    j = len(p) - 1
    for i, (xi, yi) in enumerate(p):
        xj, yj = p[j]
        cross = (x - xi) * (yj - yi) - (y - yi) * (xj - xi)
        if abs(cross) < 1e-11 and min(xi, xj)-1e-10 <= x <= max(xi, xj)+1e-10 and min(yi, yj)-1e-10 <= y <= max(yi, yj)+1e-10:
            return True
        if (yi > y) != (yj > y):
            atx = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < atx:
                hit = not hit
        j = i
    return hit

# Mainland/coast outlines.  Each outline is intentionally a coarse, manually
# estimated coastline, with its landward side closed well outside the frame.
BASE = []
ISLANDS = []
HOLES = []
# Northern Honshu / Chugoku side of the Seto Inland Sea.
BASE += [polygon(
    (130.70, 34.16), (130.95, 34.22), (131.18, 34.28),
    (131.38, 34.31), (131.56, 34.38), (131.78, 34.43),
    (131.98, 34.39), (132.18, 34.45), (132.38, 34.47),
    (132.60, 34.53), (132.80, 34.48), (133.02, 34.51),
    (133.22, 34.57), (133.42, 34.62), (133.62, 34.70),
    (133.83, 34.75), (134.04, 34.77), (134.25, 34.75),
    (134.46, 34.71), (134.67, 34.76), (134.88, 34.80),
    (135.08, 34.80), (135.28, 34.75), (135.48, 34.71),
    (135.70, 34.65), (135.91, 34.62), (136.12, 34.67),
    (136.33, 34.74), (136.54, 34.72), (136.76, 34.77),
    (137.00, 34.82), (137.20, 35.20), (130.55, 35.20)
)]

# Northern and eastern Kyushu, including the irregular Nagasaki side.
BASE += [polygon(
    (128.85, 32.78), (129.05, 32.98), (129.18, 33.18),
    (129.34, 33.35), (129.43, 33.60), (129.55, 33.82),
    (129.74, 34.03), (129.97, 34.18), (130.20, 34.27),
    (130.43, 34.30), (130.65, 34.25), (130.87, 34.15),
    (131.02, 33.91), (131.08, 33.62), (131.18, 33.30),
    (131.27, 32.82), (128.85, 32.78)
)]

# Shikoku, with the north coast facing the Inland Sea.
BASE += [polygon(
    (131.86, 33.19), (132.08, 33.28), (132.31, 33.20),
    (132.54, 33.17), (132.78, 33.22), (133.02, 33.28),
    (133.27, 33.34), (133.52, 33.41), (133.78, 33.48),
    (134.02, 33.52), (134.27, 33.57), (134.49, 33.68),
    (134.68, 33.82), (134.82, 34.00), (134.81, 34.18),
    (134.69, 34.30), (134.49, 34.36), (134.25, 34.39),
    (134.00, 34.37), (133.75, 34.34), (133.51, 34.31),
    (133.27, 34.29), (133.03, 34.27), (132.79, 34.25),
    (132.55, 34.23), (132.30, 34.25), (132.10, 34.30),
    (131.92, 34.18), (131.82, 33.91), (131.82, 33.58)
)]

# Kii peninsula and the land around Ise/Mikawa bays.
BASE += [polygon(
    (134.92, 34.63), (135.17, 34.55), (135.34, 34.37),
    (135.30, 34.12), (135.22, 33.88), (135.27, 33.65),
    (135.49, 33.47), (135.77, 33.35), (136.05, 33.34),
    (136.30, 33.46), (136.49, 33.68), (136.60, 33.94),
    (136.72, 34.18), (136.84, 34.40), (136.98, 34.62),
    (137.12, 34.86), (137.18, 35.20), (134.82, 35.20)
)]

# Bays cut out of the broad mainland polygons.  Islands are added separately
# below, so these exclusions do not erase an island lying in a bay.
HOLES += [polygon(
    (134.62, 34.73), (134.82, 34.69), (135.03, 34.59),
    (135.16, 34.42), (135.05, 34.27), (134.82, 34.24),
    (134.62, 34.34), (134.48, 34.51)
)]
HOLES += [polygon(
    (136.25, 34.79), (136.50, 34.78), (136.75, 34.70),
    (136.88, 34.53), (136.84, 34.32), (136.68, 34.20),
    (136.47, 34.24), (136.32, 34.40), (136.19, 34.60)
)]
# Offshore islands and island groups.  Ellipses are rotated where useful to
# retain the characteristic long, narrow shape of several large islands.
ISLANDS += [
    ellipse(129.34, 34.48, .14, .32, 40, -.25),       # Tsushima (south half)
    ellipse(129.74, 33.78, .14, .18, 28, .20),        # Iki
    ellipse(129.27, 33.46, .10, .16, 24, -.20),       # Hirado
    ellipse(129.05, 33.08, .13, .18, 26, .35),        # Goto edge
    ellipse(130.02, 33.49, .09, .14, 24, -.15),       # western Kyushu islets

    ellipse(131.17, 34.18, .09, .13, 24, -.10),       # Suo islands
    ellipse(131.48, 34.08, .10, .16, 25, .25),
    ellipse(131.67, 34.18, .11, .18, 28, -.40),
    ellipse(132.05, 34.10, .09, .15, 24, .15),
    ellipse(132.28, 34.30, .10, .16, 26, -.40),       # Miyajima
    ellipse(132.39, 34.25, .13, .18, 28, .20),        # Etajima
    ellipse(132.49, 34.08, .13, .18, 28, -.25),       # Kurahashi
    ellipse(132.64, 34.18, .06, .11, 22, .10),
    ellipse(132.79, 34.28, .08, .13, 24, -.40),
    ellipse(132.90, 34.15, .09, .15, 24, .30),

    ellipse(133.00, 34.27, .13, .16, 28, -.30),       # Omishima
    ellipse(133.13, 34.34, .13, .10, 26, .15),        # Innoshima
    ellipse(133.17, 34.40, .09, .07, 22, -.20),       # Mukaishima
    ellipse(133.24, 34.18, .08, .15, 24, -.35),       # Hakata
    ellipse(133.34, 34.10, .13, .20, 28, -.25),       # Oshima
    ellipse(133.45, 34.26, .07, .11, 22, .40),
    ellipse(133.57, 34.39, .07, .09, 22, -.30),

    ellipse(133.77, 34.46, .07, .09, 22, .20),        # Teshima group
    ellipse(133.91, 34.47, .11, .10, 26, -.25),       # Naoshima
    ellipse(134.10, 34.48, .07, .12, 24, .20),
    ellipse(134.26, 34.51, .24, .16, 36, -.12),       # Shodoshima
    ellipse(134.47, 34.43, .07, .10, 22, .25),
    ellipse(134.72, 34.35, .08, .12, 24, -.30),
    ellipse(134.87, 34.42, .25, .34, 44, .15),        # Awaji
    ellipse(134.33, 34.15, .07, .11, 22, -.30),
    ellipse(134.53, 34.06, .06, .09, 20, .15),

    ellipse(135.63, 34.73, .07, .10, 22, -.20),
    ellipse(136.05, 34.68, .08, .10, 22, .30),
    ellipse(136.35, 34.66, .06, .10, 20, -.30),
]

# A few small islands are better represented as polygonal sandbars/reefs.
ISLANDS += [
    polygon((131.92,34.33),(132.02,34.34),(132.07,34.29),(132.00,34.24),(131.91,34.26)),
    polygon((132.72,34.34),(132.81,34.36),(132.86,34.32),(132.81,34.27),(132.73,34.29)),
    polygon((133.67,34.52),(133.74,34.54),(133.78,34.50),(133.73,34.46),(133.66,34.48)),
    polygon((134.00,34.61),(134.08,34.62),(134.11,34.57),(134.05,34.53),(133.99,34.56)),
    polygon((134.61,34.57),(134.68,34.59),(134.71,34.54),(134.65,34.51),(134.59,34.53)),
]
def land_at(lon, lat):
    # Mainland first, with the two manually outlined bay openings removed.
    mainland = any(inside(lon, lat, p) for p in BASE)
    if mainland and any(inside(lon, lat, p) for p in HOLES):
        mainland = False
    return mainland or any(inside(lon, lat, p) for p in ISLANDS)

def build_matrix():
    result = np.zeros((NY, NX), dtype=np.uint8)
    for row in range(NY):
        lat = NORTH - row * DY
        for col in range(NX):
            lon = WEST + col * DX
            result[row, col] = 1 if land_at(lon, lat) else 0
    return result

def write_outputs(a):
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerows(a.tolist())

    # Pillow's L image has exactly two possible pixel values here: 0 and 255.
    small = Image.fromarray((a * 255).astype(np.uint8), mode="L")
    small.save(out / "candidate.png")
    large = small.resize((NX * 8, NY * 8), resample=Image.Resampling.NEAREST)
    large.save(out / "candidate_8x.png")

def validate(a):
    assert a.shape == (64, 253)
    assert a.size == 16192
    assert set(np.unique(a)).issubset({0, 1})
    with Path("outputs/candidate.csv").open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 64 and all(len(r) == 253 for r in rows)
    assert all(v in {"0", "1"} for r in rows for v in r)
    assert Image.open("outputs/candidate.png").size == (253, 64)
    assert Image.open("outputs/candidate_8x.png").size == (2024, 512)

if __name__ == "__main__":
    matrix = build_matrix()
    write_outputs(matrix)
    validate(matrix)
    print("generated 64x253 matrix; PNGs: 253x64 and 2024x512")
