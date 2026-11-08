import csv
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path("outputs")
LAT_N = 71
LON_N = 231
LAT0 = 45.0
LAT1 = 31.0
LON0 = -9.0
LON1 = 37.0
STEP = 0.2
lats = np.round(LAT0 - np.arange(LAT_N) * STEP, 10)
lons = np.round(LON0 + np.arange(LON_N) * STEP, 10)
LON, LAT = np.meshgrid(lons, lats)


def npoly(arr):
    return np.asarray(arr, dtype=float)


def in_poly(lon, lat, poly):
    x = lon
    y = lat
    inside = np.zeros_like(x, dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
        inside ^= cond
    return inside


def in_box(lon, lat, x0, x1, y0, y1):
    return (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)


def in_ellipse(lon, lat, cx, cy, rx, ry, rot=0.0):
    th = math.radians(rot)
    dx = lon - cx
    dy = lat - cy
    xr = dx * math.cos(th) + dy * math.sin(th)
    yr = -dx * math.sin(th) + dy * math.cos(th)
    return (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0


def add_poly(M, poly):
    M[in_poly(LON, LAT, npoly(poly))] = 1


def add_box(M, x0, x1, y0, y1):
    M[in_box(LON, LAT, x0, x1, y0, y1)] = 1


def add_ellipse(M, cx, cy, rx, ry, rot=0.0):
    M[in_ellipse(LON, LAT, cx, cy, rx, ry, rot)] = 1
M = np.zeros((LAT_N, LON_N), dtype=np.uint8)

# Iberia / Portugal / Spain
add_poly(M, [
    (-9.2, 43.7), (-4.0, 43.7), (-2.0, 43.4), (0.3, 42.4),
    (3.1, 42.3), (2.0, 41.0), (0.2, 40.0), (-0.3, 39.4),
    (-1.0, 38.5), (-2.2, 36.7), (-4.4, 36.7), (-5.4, 36.0),
    (-6.3, 36.5), (-7.4, 37.2), (-8.9, 37.0), (-9.5, 38.7),
    (-9.3, 42.0),
])

# Continental Europe: France/Alps/Balkans north of the Mediterranean-Adriatic line
add_poly(M, [
    (-5.0, 43.7), (3.0, 43.1), (7.5, 43.8), (12.0, 45.8),
    (19.0, 45.5), (23.5, 45.0), (27.5, 45.6), (27.5, 46.0),
    (-5.0, 46.0),
])

# Italian peninsula/boot
add_poly(M, [
    (6.6, 45.9), (13.8, 46.3), (13.5, 45.6), (12.3, 43.8),
    (13.6, 42.8), (14.8, 41.9), (16.5, 41.2), (18.4, 40.4),
    (17.0, 40.0), (16.6, 38.9), (15.8, 37.9), (15.1, 37.5),
    (15.6, 38.9), (16.0, 40.0), (15.0, 40.5), (14.2, 40.9),
    (13.0, 41.5), (12.5, 42.0), (11.0, 42.5), (10.0, 43.5),
    (9.0, 44.2), (7.5, 43.8),
])
# Balkans / Greece mainland
add_poly(M, [
    (13.5, 45.7), (27.5, 45.0), (27.5, 44.0), (26.0, 43.0),
    (24.0, 42.5), (23.5, 41.5), (24.0, 41.0), (23.5, 40.0),
    (23.0, 39.0), (22.8, 37.5), (23.5, 36.5), (24.5, 36.0),
    (25.5, 36.0), (26.5, 37.0), (27.0, 38.0), (26.5, 39.0),
    (25.5, 40.0), (24.5, 40.5), (23.0, 40.8), (21.5, 41.5),
    (20.0, 42.0), (18.5, 42.5), (17.0, 43.0), (15.5, 43.5),
    (14.5, 44.5),
])

# Anatolia / Asian Turkey
add_poly(M, [
    (26.5, 38.0), (27.0, 37.5), (28.0, 37.0), (29.5, 37.0),
    (31.0, 36.3), (33.0, 36.3), (35.0, 36.3), (36.5, 36.2),
    (37.0, 36.5), (37.0, 42.0), (34.0, 42.0), (31.0, 41.5),
    (29.0, 41.5), (28.0, 41.0), (27.5, 40.5), (27.0, 40.0),
    (26.5, 39.0),
])
# Balearics
add_ellipse(M, 3.4, 39.65, 0.75, 0.38)
add_ellipse(M, 4.25, 39.95, 0.25, 0.20)
add_ellipse(M, 1.45, 38.95, 0.22, 0.22)
add_ellipse(M, 1.48, 38.70, 0.12, 0.12)

# Corsica, Sardinia, Sicily, Malta
add_ellipse(M, 9.1, 42.2, 0.45, 0.95)
add_ellipse(M, 9.0, 40.1, 0.85, 1.20)
add_ellipse(M, 14.0, 37.5, 1.35, 0.85, rot=-18)
add_ellipse(M, 14.4, 35.95, 0.22, 0.18)

# Crete, Cyprus, Peloponnese
add_ellipse(M, 25.0, 35.3, 1.35, 0.28)
add_ellipse(M, 33.4, 35.2, 1.05, 0.60)
add_ellipse(M, 22.2, 37.5, 0.85, 0.60)

# Crimea
add_poly(M, [
    (32.6, 45.2), (34.0, 45.9), (36.5, 45.4), (36.6, 44.7),
    (35.0, 44.4), (33.5, 44.6), (33.0, 45.0),
])

# Larger Aegean / Ionian islands
islands = [
    (23.8, 38.4, 0.9, 0.45), (26.2, 39.2, 0.4, 0.25),
    (28.0, 36.2, 0.35, 0.25), (26.1, 38.3, 0.3, 0.22),
    (26.9, 37.7, 0.35, 0.22), (25.4, 37.1, 0.25, 0.18),
    (25.5, 36.4, 0.2, 0.15), (20.6, 38.2, 0.25, 0.25),
    (19.9, 39.8, 0.25, 0.22), (20.9, 37.9, 0.2, 0.22),
    (10.3, 42.8, 0.18, 0.15), (11.1, 37.5, 0.18, 0.18),
]
for cx, cy, rx, ry in islands:
    add_ellipse(M, cx, cy, rx, ry)
# North Africa: land south of the Mediterranean coast
add_poly(M, [
    (-9.5, 35.6), (-6.3, 35.8), (-2.2, 35.1), (0.0, 35.8),
    (3.0, 36.9), (5.0, 36.7), (8.0, 37.0), (10.0, 37.2),
    (11.0, 33.8), (13.0, 32.8), (15.0, 32.5), (17.0, 32.2),
    (19.0, 31.2), (21.0, 30.6), (23.0, 30.6), (25.0, 31.4),
    (27.0, 31.6), (29.0, 31.5), (31.0, 31.3), (32.0, 31.3),
    (37.0, 31.0), (-9.0, 31.0),
])

# Sinai / Levant south of Turkey and east of the Mediterranean
add_poly(M, [
    (32.2, 31.3), (34.5, 31.4), (35.6, 31.3), (36.8, 32.2),
    (37.0, 33.5), (37.0, 36.5), (36.8, 36.3), (35.5, 35.5),
    (34.5, 34.8), (33.0, 34.5), (32.4, 34.0), (32.3, 33.0),
])
def save_outputs(M):
    if M.shape != (LAT_N, LON_N):
        raise ValueError(f"bad matrix shape: {M.shape}")
    if not np.isin(M, [0, 1]).all():
        raise ValueError("matrix must contain only 0/1")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "candidate.csv", "w", newline="") as f:
        w = csv.writer(f)
        for row in M:
            w.writerow([int(v) for v in row])
    arr = np.where(M == 1, 255, 0).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    img.save(OUT / "candidate.png")
    big = img.resize((LON_N * 8, LAT_N * 8), Image.Resampling.NEAREST)
    big.save(OUT / "candidate_8x.png")


save_outputs(M)
