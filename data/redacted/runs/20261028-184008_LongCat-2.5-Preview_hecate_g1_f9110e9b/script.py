#!/usr/bin/env python3
"""Hecate Strait (northern British Columbia) land/sea map, reconstructed
purely from hand-written knowledge-based geometric rules.

Grid: lat 55N -> 53N (127 rows, N->S), lon 131W -> 129W (127 cols, W->E).
No network access, no external geographic data, no standard answers.
Outputs: outputs/candidate.csv, candidate.png, candidate_8x.png
"""
import csv
import math
import os

import numpy as np
from PIL import Image

N = 127
STEP = 2.0 / 126.0
LAT_TOP = 55.0
LON_LEFT = -131.0
ROWS = [LAT_TOP - i * STEP for i in range(N)]
COLS = [LON_LEFT + j * STEP for j in range(N)]

KX = 66.0    # km per degree longitude at ~54N
KY = 111.2   # km per degree latitude


def point_in_poly(lon, lat, poly):
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        yi, xi = poly[i][1], poly[i][0]
        yj, xj = poly[j][1], poly[j][0]
        if (yi > lat) != (yj > lat):
            xint = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < xint:
                inside = not inside
        j = i
    return inside


def point_in_ellipse(lon, lat, cx, cy, rx, ry):
    dx = (lon - cx) * KX
    dy = (lat - cy) * KY
    return (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry) <= 1.0


def dist_km(lon, lat, cx, cy):
    return math.hypot((lon - cx) * KX, (lat - cy) * KY)


# ---- hand-written land polygons (lon, lat) ------------------------------
# Graham Island (Haida Gwaii), NE-SW elongated, ~160 km.
GRAHAM_POLY = [
    (-132.60, 54.20), (-132.20, 54.10), (-131.95, 53.95), (-131.70, 53.90),
    (-131.55, 53.80), (-131.40, 53.75), (-131.30, 53.65), (-131.35, 53.55),
    (-131.45, 53.45), (-131.60, 53.40), (-131.80, 53.35), (-132.05, 53.40),
    (-132.30, 53.55), (-132.45, 53.70), (-132.55, 53.85), (-132.60, 54.00),
]
# Moresby Island, south of Graham, Skidegate Channel between them.
MORESBY_POLY = [
    (-132.30, 53.30), (-132.05, 53.25), (-131.90, 53.15), (-131.85, 53.05),
    (-131.90, 52.95), (-132.00, 52.90), (-132.15, 52.88), (-132.30, 52.90),
    (-132.42, 52.97), (-132.48, 53.08), (-132.45, 53.18), (-132.38, 53.25),
]
# Louise Island (SW of Graham, near Sandspit).
LOUISE_POLY = [
    (-131.90, 53.22), (-131.80, 53.20), (-131.72, 53.14), (-131.72, 53.08),
    (-131.80, 53.04), (-131.90, 53.06), (-131.98, 53.12), (-131.98, 53.19),
]
# Lyell Island (SE of Graham).
LYELL_POLY = [
    (-131.60, 52.85), (-131.50, 52.80), (-131.42, 52.72), (-131.44, 52.62),
    (-131.52, 52.55), (-131.62, 52.53), (-131.72, 52.58), (-131.78, 52.68),
    (-131.74, 52.78), (-131.66, 52.84),
]
# Kunghit Island (southernmost Haida Gwaii).
KUNGHIT_POLY = [
    (-132.55, 52.15), (-132.45, 52.12), (-132.38, 52.05), (-132.40, 51.98),
    (-132.50, 51.95), (-132.62, 51.97), (-132.70, 52.05), (-132.68, 52.12),
]
# Small Haida Gwaii islets: (lon, lat, radius deg).
ISLET_CIRCLES = [
    (-132.85, 54.05, 0.12), (-132.75, 53.85, 0.09), (-131.25, 53.35, 0.10),
    (-131.15, 52.55, 0.08), (-132.10, 52.55, 0.08),
]


# Porcher Island (~130.5W, 54.1N).
PORCHER_POLY = [
    (-130.70, 54.20), (-130.60, 54.22), (-130.48, 54.20), (-130.40, 54.12),
    (-130.38, 54.02), (-130.44, 53.92), (-130.55, 53.88), (-130.66, 53.90),
    (-130.72, 54.00), (-130.74, 54.10),
]
# Banks Island (north of Porcher, near Prince Rupert).
BANKS_POLY = [
    (-130.95, 53.75), (-130.82, 53.78), (-130.68, 53.74), (-130.60, 53.66),
    (-130.62, 53.56), (-130.72, 53.50), (-130.85, 53.52), (-130.96, 53.60),
    (-131.00, 53.68),
]
# Campania Island (south of Banks, ~53.1N).
CAMPANIA_POLY = [
    (-130.55, 53.25), (-130.45, 53.24), (-130.36, 53.18), (-130.34, 53.08),
    (-130.42, 53.00), (-130.54, 53.00), (-130.63, 53.08), (-130.64, 53.18),
]
# Aristazabal Island (east of Campania).
ARISTAZABAL_POLY = [
    (-130.30, 53.15), (-130.20, 53.14), (-130.12, 53.08), (-130.12, 52.98),
    (-130.22, 52.92), (-130.34, 52.94), (-130.42, 53.02), (-130.42, 53.12),
]
# Pitt Island (east of Campania/Aristazabal, ~53.2N).
PITT_POLY = [
    (-130.30, 53.45), (-130.20, 53.48), (-130.08, 53.45), (-129.98, 53.38),
    (-129.96, 53.28), (-130.02, 53.18), (-130.12, 53.12), (-130.22, 53.12),
    (-130.30, 53.20), (-130.34, 53.32),
]


# BC mainland coast (east side of the box).  Coastline runs N->S.
MAINLAND_POLY = [
    (-130.45, 55.00), (-130.40, 54.80), (-130.38, 54.60), (-130.35, 54.40),
    (-130.32, 54.20), (-130.28, 54.00), (-130.22, 53.80), (-130.16, 53.60),
    (-130.10, 53.40), (-130.04, 53.20), (-129.98, 53.00),
    (-128.90, 53.00), (-128.90, 55.00),
]

# Dundas / Stephens islets in Dixon Entrance (lon, lat, radius deg).
NORTH_ISLETS = [
    (-130.95, 54.60, 0.10), (-130.85, 54.45, 0.07), (-130.80, 54.10, 0.08),
    (-130.60, 54.35, 0.06), (-130.55, 53.85, 0.06), (-130.45, 53.30, 0.07),
    (-130.20, 53.05, 0.06), (-130.35, 53.05, 0.06), (-130.05, 53.55, 0.07),
    (-129.95, 54.00, 0.07), (-130.10, 53.90, 0.06), (-130.25, 53.70, 0.06),
]

ISLAND_POLYS = [
    PORCHER_POLY, BANKS_POLY, CAMPANIA_POLY, ARISTAZABAL_POLY, PITT_POLY,
    GRAHAM_POLY, MORESBY_POLY, LOUISE_POLY, LYELL_POLY, KUNGHIT_POLY,
]


def is_land(lon, lat):
    """Knowledge-based land test for one WGS84 point."""
    if point_in_poly(lon, lat, MAINLAND_POLY):
        return True
    for poly in ISLAND_POLYS:
        if point_in_poly(lon, lat, poly):
            return True
    for (cx, cy, r) in ISLET_CIRCLES + NORTH_ISLETS:
        if dist_km(lon, lat, cx, cy) <= r * KX:
            return True
    return False



def build_matrix():
    m = np.zeros((N, N), dtype=np.uint8)
    for i, lat in enumerate(ROWS):
        for j, lon in enumerate(COLS):
            m[i, j] = 1 if is_land(lon, lat) else 0
    return m


def main():
    os.makedirs("outputs", exist_ok=True)
    m = build_matrix()
    assert m.shape == (N, N), m.shape

    with open("outputs/candidate.csv", "w", newline="") as f:
        w = csv.writer(f)
        for row in m:
            w.writerow([int(v) for v in row])

    img = Image.fromarray(m * 255, mode="L")          # 1=land -> white
    img.save("outputs/candidate.png")

    big = img.resize((N * 8, N * 8), Image.NEAREST)   # 1016 x 1016
    big.save("outputs/candidate_8x.png")

    land = int(m.sum())
    print("matrix %dx%d  land=%d  water=%d" % (N, N, land, N * N - land))
    print("csv rows/cols:", sum(1 for _ in open("outputs/candidate.csv")),
          len(open("outputs/candidate.csv").readline().split(",")))
    print("png:", img.size, " 8x:", big.size)


if __name__ == "__main__":
    main()

