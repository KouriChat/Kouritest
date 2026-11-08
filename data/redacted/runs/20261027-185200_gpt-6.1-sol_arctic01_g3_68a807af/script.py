#!/usr/bin/env python3
"""Knowledge-based, hand-drawn reconstruction of the requested Arctic window.

The polygons below are deliberately a compact approximation, not imported
cartography.  Coordinates are (longitude, latitude), in WGS84 degrees.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

LATS = np.round(np.arange(79.0, 72.999, -0.1), 10)
LONS = np.round(np.arange(-105.0, -80.999, 0.1), 10)

def inside_polygon(x, y, poly):
    """Even-odd point test, used on the complete sampling mesh."""
    p = np.asarray(poly, dtype=float)
    px, py = p[:, 0], p[:, 1]
    hit = np.zeros(x.shape, dtype=bool)
    j = len(p) - 1
    for i in range(len(p)):
        yi, yj = py[i], py[j]
        xi, xj = px[i], px[j]
        crosses = ((yi > y) != (yj > y))
        xin = (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi
        hit ^= crosses & (x < xin)
        j = i
    return hit

def land(poly, mesh_x, mesh_y):
    return inside_polygon(mesh_x, mesh_y, poly)

# Hand-estimated outlines of islands and the small portions intersecting
# the requested rectangle.  Longitudes are negative (degrees west).
MELVILLE = [
    (-105.2, 78.25), (-102.0, 78.45), (-98.0, 78.48),
    (-94.5, 78.30), (-91.0, 77.75), (-89.0, 76.85),
    (-89.8, 75.45), (-93.0, 74.65), (-97.0, 74.25),
    (-101.0, 74.55), (-104.5, 75.25), (-105.2, 76.4),
]

BATHURST = [
    (-99.2, 76.75), (-97.2, 77.45), (-94.0, 77.35),
    (-91.8, 76.75), (-92.2, 75.65), (-94.3, 75.05),
    (-97.0, 75.10), (-99.0, 75.75),
]

PRINCE_OF_WALES = [
    (-99.9, 75.75), (-98.0, 76.35), (-94.6, 76.25),
    (-91.7, 75.65), (-90.0, 74.65), (-90.2, 73.15),
    (-92.2, 72.60), (-95.6, 72.65), (-98.6, 73.35),
    (-100.3, 74.45),
]

SOMERSET = [
    (-95.2, 74.70), (-92.4, 74.85), (-89.6, 74.55),
    (-88.0, 73.55), (-88.6, 72.55), (-90.2, 71.85),
    (-92.8, 72.15), (-94.8, 72.95),
]
DEVON = [
    (-97.8, 74.00), (-97.4, 75.15), (-96.6, 76.55),
    (-94.6, 77.65), (-91.0, 78.35), (-87.0, 78.65),
    (-83.5, 78.20), (-81.8, 77.15), (-82.0, 75.85),
    (-84.0, 74.75), (-87.0, 74.20), (-91.5, 73.95),
    (-95.0, 73.90),
]

CORNWALLIS = [
    (-95.7, 75.65), (-94.4, 76.40), (-91.4, 76.35),
    (-89.2, 75.75), (-89.7, 74.75), (-92.0, 74.45),
    (-94.7, 74.75),
]

AXEL_HEIBERG = [
    (-100.8, 79.0), (-98.8, 79.0), (-95.9, 78.85),
    (-93.6, 79.0), (-90.2, 79.0), (-89.2, 78.35),
    (-91.2, 77.75), (-94.5, 77.50), (-97.1, 77.65),
    (-99.8, 78.20),
]

ELLESMERE = [
    (-88.8, 79.0), (-84.0, 79.0), (-81.0, 79.0),
    (-81.0, 76.65), (-82.2, 76.20), (-84.6, 76.45),
    (-87.0, 77.10), (-89.0, 77.75),
]

# The west edge of Baffin Island just clips the southeast corner of the
# window; the rest of that island lies east of the requested longitudes.
BAFFIN_EDGE = [
    (-81.0, 73.00), (-80.1, 73.20), (-79.0, 74.15),
    (-79.2, 74.75), (-80.2, 74.85), (-81.0, 74.35),
]

# Northeastern Victoria Island enters at the lower-left edge.  Most of
# Victoria Island is west of -105 and is intentionally outside the frame.
VICTORIA_EDGE = [
    (-105.0, 72.95), (-104.6, 73.65), (-103.4, 74.05),
    (-101.6, 73.95), (-100.5, 73.35), (-100.6, 72.95),
]
ISLANDS = [
    MELVILLE, BATHURST, PRINCE_OF_WALES, SOMERSET,
    DEVON, CORNWALLIS, AXEL_HEIBERG, ELLESMERE,
    BAFFIN_EDGE, VICTORIA_EDGE,
]

def make_matrix():
    lon, lat = np.meshgrid(LONS, LATS)
    result = np.zeros((len(LATS), len(LONS)), dtype=np.uint8)
    for outline in ISLANDS:
        result |= land(outline, lon, lat).astype(np.uint8)
    return result

def write_outputs(matrix):
    if matrix.shape != (61, 241):
        raise AssertionError(matrix.shape)
    if not np.isin(matrix, (0, 1)).all():
        raise AssertionError("non-binary matrix")
    with (OUT / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # L mode is used so the source contains exactly two byte values.
    source = Image.fromarray((matrix * 255).astype(np.uint8), mode="L")
    if source.size != (241, 61):
        raise AssertionError(source.size)
    source.save(OUT / "candidate.png")
    enlarged = source.resize((241 * 8, 61 * 8), Image.Resampling.NEAREST)
    enlarged.save(OUT / "candidate_8x.png")

def verify():
    rows = list(csv.reader((OUT / "candidate.csv").open(
        newline="", encoding="ascii")))
    assert len(rows) == 61 and all(len(r) == 241 for r in rows)
    assert all(v in {"0", "1"} for r in rows for v in r)
    a = Image.open(OUT / "candidate.png")
    b = Image.open(OUT / "candidate_8x.png")
    assert a.size == (241, 61) and b.size == (1928, 488)
    assert set(a.getdata()) <= {0, 255}
    assert set(b.getdata()) <= {0, 255}
    assert np.array_equal(
        np.asarray(b)[::8, ::8], np.asarray(a))
    print("validated: 61x241 =", 61 * 241,
          "land =", int(np.asarray(matrix_global).sum()),
          "water =", int((matrix_global == 0).sum()))

matrix_global = make_matrix()
write_outputs(matrix_global)
verify()
