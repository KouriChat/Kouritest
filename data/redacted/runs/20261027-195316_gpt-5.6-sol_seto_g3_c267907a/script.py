"""Knowledge-only hand-built approximation of land in the requested window.
No geographic data are read or downloaded.
"""
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image

NROW, NCOL = 64, 253
LAT0, LON0 = 35.0, 129.0
STEP = 2.0 / 63.0

def inside_polygon(x, y, p):
    hit = False
    j = len(p) - 1
    for i in range(len(p)):
        xi, yi = p[i]
        xj, yj = p[j]
        cross = ((yi > y) != (yj > y))
        if cross and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            hit = not hit
        j = i
    return hit

def ellipse(x, y, cx, cy, rx, ry, angle=0.0):
    t = math.radians(angle)
    dx, dy = x - cx, y - cy
    u = dx * math.cos(t) + dy * math.sin(t)
    v = -dx * math.sin(t) + dy * math.cos(t)
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1.0

# Broad mainland outlines.  Vertices follow remembered coast trends rather
# than a traced map; all coordinates are longitude, latitude.
POLYGONS = [
    # Chugoku south-facing part of Honshu, with the northern edge clipped.
    [(130.65,35.0),(137.0,35.0),(137.0,34.70),(136.55,34.72),
     (136.10,34.62),(135.65,34.58),(135.25,34.48),(134.85,34.42),
     (134.42,34.34),(134.02,34.29),(133.55,34.25),(133.05,34.24),
     (132.62,34.28),(132.20,34.23),(131.80,34.16),(131.45,34.05),
     (131.10,33.96),(130.82,33.98)],
    # The inland-facing and southern sides of the Kii peninsula.
    [(135.45,34.45),(136.00,34.48),(136.42,34.34),(136.72,34.02),
     (136.58,33.68),(136.28,33.42),(135.92,33.22),(135.52,33.16),
     (135.20,33.32),(135.20,33.70),(135.32,34.05)],
    # Northern/eastern shore around Ise and Mikawa bays.
    [(136.30,35.0),(137.0,35.0),(137.0,34.70),(136.55,34.72),
     (136.45,34.48),(136.42,34.34),(136.10,34.48),(135.85,34.66),
     (135.62,34.82),(135.60,35.0)],
    # Shikoku.
    [(131.95,34.35),(132.45,34.43),(133.05,34.43),(133.60,34.39),
     (134.10,34.32),(134.55,34.17),(134.80,33.92),(134.70,33.58),
     (134.35,33.25),(133.85,33.08),(133.25,33.02),(132.78,33.15),
     (132.36,33.36),(132.08,33.68),(131.90,34.02)],
    # Kyushu (the west side is intentionally somewhat generalized).
    [(129.35,34.05),(129.62,34.35),(130.05,34.55),(130.60,34.68),
     (131.15,34.62),(131.48,34.42),(131.62,34.05),(131.58,33.64),
     (131.46,33.22),(131.22,33.0),(130.38,33.0),(129.98,33.16),
     (129.60,33.38),(129.40,33.68)],
    # Small western end of Honshu/Yamaguchi.
    [(130.35,35.0),(131.10,35.0),(131.18,34.66),(130.72,34.57),
     (130.40,34.37),(130.10,34.18),(129.92,34.05),(130.20,33.98),
     (130.65,34.02),(130.88,34.18),(131.12,34.38),(131.42,34.56),
     (131.55,34.82),(131.70,35.0)],
]

def hand_land(lon, lat):
    """Return the remembered-rule land classification for one point."""
    for poly in POLYGONS:
        if inside_polygon(lon, lat, poly):
            return True
    return False
# Korean south-eastern tip and the larger detached islands.
POLYGONS += [
    [(128.96,35.0),(129.34,35.0),(129.30,34.88),(129.05,34.84),
     (128.96,34.91)],
    [(129.05,34.78),(129.23,34.88),(129.37,34.75),(129.29,34.55),
     (129.10,34.43),(129.02,34.55)],
    # Hirado and the northern Goto group (only their eastern portions).
    [(129.42,33.48),(129.62,33.54),(129.72,33.38),(129.63,33.20),
     (129.42,33.17),(129.30,33.32)],
]

# (longitude, latitude, half-width, half-height, rotation)
# These deliberately include named islands and remembered island clusters,
# including islands too small to deserve separate hand-drawn polygons.
ISLANDS = [
    (129.73,33.82,.20,.10,18), (129.48,33.40,.12,.25,25),
    (129.25,33.24,.12,.25,35), (129.72,34.06,.09,.08,0),
    (130.00,34.00,.10,.07,0), (130.20,33.90,.07,.06,0),
    (130.18,33.53,.09,.12,20), (130.35,33.62,.08,.10,0),
    (130.52,33.72,.06,.10,25), (130.72,33.82,.07,.06,0),
    # Suo-nada / western Seto islands
    (131.02,33.99,.10,.07,15), (131.28,34.00,.07,.06,0),
    (131.53,34.05,.08,.06,0), (131.78,34.08,.08,.07,15),
    (132.05,34.17,.10,.08,0), (132.28,34.20,.08,.08,20),
    (132.32,34.29,.10,.16,25), (132.48,34.25,.15,.16,10),
    (132.62,34.16,.08,.07,0), (132.75,34.20,.07,.10,20),
    (132.92,34.20,.08,.14,20), (133.05,34.24,.10,.13,15),
    (133.18,34.20,.08,.10,0), (133.35,34.23,.09,.12,25),
    (133.49,34.20,.08,.10,20), (133.63,34.26,.09,.07,0),
    # Geiyo chain
    (133.00,34.08,.13,.14,25), (133.16,34.10,.11,.12,20),
    (133.30,34.08,.10,.12,30), (133.48,34.12,.11,.11,20),
    (133.67,34.14,.08,.10,25), (133.82,34.20,.10,.10,15),
    (133.98,34.22,.09,.07,0), (134.16,34.20,.08,.09,0),
    # Kagawa-side islands
    (133.60,34.30,.09,.08,0), (133.76,34.37,.07,.06,0),
    (133.93,34.42,.06,.07,20), (134.00,34.47,.07,.09,20),
    (134.08,34.50,.06,.08,15), (134.18,34.48,.07,.08,20),
    (134.30,34.52,.14,.20,25), (134.43,34.48,.06,.07,20),
    (134.57,34.45,.07,.06,0), (134.70,34.38,.07,.08,30),
    # Awaji and Osaka/Kii waters
    (134.82,34.40,.14,.34,12), (134.90,34.65,.08,.08,0),
    (135.15,34.48,.08,.06,0), (135.35,34.46,.06,.05,0),
    (135.52,34.38,.07,.08,20), (135.70,34.32,.06,.05,0),
    (136.02,34.30,.07,.06,20), (136.26,34.38,.06,.07,0),
    # Ise bay islands
    (136.83,34.68,.08,.05,0), (136.92,34.67,.06,.05,0),
    (136.75,34.78,.07,.05,0), (136.56,34.84,.08,.05,0),
    (136.90,34.35,.06,.04,0),
]

def hand_land(lon, lat):
    if any(inside_polygon(lon, lat, p) for p in POLYGONS):
        return True
    return any(ellipse(lon, lat, *e) for e in ISLANDS)
def make_matrix():
    out = np.zeros((NROW, NCOL), dtype=np.uint8)
    for r in range(NROW):
        lat = LAT0 - r * STEP
        for c in range(NCOL):
            lon = LON0 + c * STEP
            out[r, c] = 1 if hand_land(lon, lat) else 0
    return out

def write_outputs(a):
    root = Path("outputs")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        w = csv.writer(f, lineterminator="\n")
        for row in a:
            w.writerow([int(v) for v in row])

    # 1 is white land, 0 is black water.  No interpolation is used.
    raw = (a * 255).astype(np.uint8)
    Image.fromarray(raw, mode="L").save(root / "candidate.png")
    Image.fromarray(raw, mode="L").resize((NCOL * 8, NROW * 8),
                                           resample=Image.Resampling.NEAREST
                                           ).save(root / "candidate_8x.png")

def validate(a):
    assert a.shape == (64, 253)
    assert int(a.sum() + (a == 0).sum()) == 16192
    assert set(np.unique(a)).issubset({0, 1})
    with open("outputs/candidate.csv", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 64 and all(len(row) == 253 for row in rows)
    assert all(v in {"0", "1"} for row in rows for v in row)
    assert Image.open("outputs/candidate.png").size == (253, 64)
    assert Image.open("outputs/candidate_8x.png").size == (2024, 512)
    assert set(Image.open("outputs/candidate.png").getdata()).issubset({0, 255})
    assert set(Image.open("outputs/candidate_8x.png").getdata()).issubset({0, 255})

if __name__ == "__main__":
    matrix = make_matrix()
    write_outputs(matrix)
    validate(matrix)
    print("validated", matrix.shape, int(matrix.sum()), "land pixels")
