#!/usr/bin/env python3
"""Approximate knowledge-based land/water raster for the Canadian Arctic."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

# Hand-drawn geographic outlines, in (longitude, latitude) degrees.
# They are deliberately approximate; no external geographic data is used.
POLYGONS = [
    # Easternmost Melville Island, clipped by the western frame.
    [(-105.0, 75.35), (-104.5, 75.45), (-104.25, 75.8),
     (-104.5, 76.25), (-105.0, 76.45)],
    # Lougheed Island
    [(-105.0, 77.75), (-104.55, 77.72), (-103.95, 77.9),
     (-103.85, 78.15), (-104.25, 78.42), (-104.75, 78.38),
     (-105.0, 78.2)],
    # King Christian Island
    [(-102.05, 77.62), (-101.6, 77.65), (-101.35, 77.9),
     (-101.45, 78.1), (-101.85, 78.15), (-102.12, 77.92)],
    # Ellef Ringnes Island
    [(-102.1, 78.05), (-101.65, 78.12), (-101.25, 78.45),
     (-100.75, 78.55), (-100.25, 78.85), (-99.75, 79.0),
     (-99.8, 79.0), (-101.8, 79.0), (-102.15, 78.7),
     (-102.3, 78.35)],
    # Amund Ringnes Island
    [(-98.9, 78.45), (-98.45, 78.5), (-97.95, 78.72),
     (-97.55, 78.93), (-97.15, 79.0), (-99.0, 79.0),
     (-99.15, 78.77)],
    # Axel Heiberg Island
    [(-95.8, 78.0), (-95.3, 78.1), (-94.9, 78.35),
     (-94.6, 78.7), (-94.2, 78.92), (-93.7, 79.0),
     (-88.0, 79.0), (-87.8, 78.75), (-88.35, 78.45),
     (-88.75, 78.15), (-89.0, 77.85), (-89.25, 77.55),
     (-89.35, 77.2), (-89.7, 76.9), (-90.2, 76.72),
     (-90.8, 76.78), (-91.1, 77.0), (-91.4, 77.35),
     (-92.0, 77.55), (-92.8, 77.7), (-93.7, 77.85),
     (-94.6, 77.95), (-95.2, 77.88)],
    # Ellesmere Island: western and northern section within the frame.
    [(-91.2, 76.15), (-90.8, 76.35), (-90.3, 76.55),
     (-89.8, 76.75), (-89.5, 77.0), (-89.1, 77.3),
     (-88.9, 77.65), (-88.5, 77.95), (-88.2, 78.25),
     (-87.8, 78.55), (-87.5, 78.85), (-87.2, 79.0),
     (-81.0, 79.0), (-81.0, 76.0), (-82.0, 75.9),
     (-83.0, 75.8), (-84.0, 75.75), (-85.0, 75.85),
     (-86.0, 75.8), (-87.0, 75.7), (-88.0, 75.55),
     (-89.0, 75.45), (-89.8, 75.5), (-90.4, 75.65),
     (-90.8, 75.85)],
    # Bathurst Island
    [(-99.7, 75.45), (-99.45, 75.8), (-99.2, 76.12),
     (-98.75, 76.25), (-98.35, 76.15), (-97.95, 76.0),
     (-97.6, 75.8), (-97.35, 75.55), (-97.6, 75.32),
     (-98.1, 75.25), (-98.65, 75.28), (-99.15, 75.32)],
    # Cornwallis Island
    [(-96.05, 74.95), (-95.85, 75.25), (-95.55, 75.42),
     (-95.15, 75.43), (-94.9, 75.2), (-95.0, 74.95),
     (-95.3, 74.8), (-95.75, 74.82)],
    # Little Cornwallis Island
    [(-96.25, 75.05), (-96.12, 75.2), (-95.98, 75.15),
     (-96.0, 75.0)],
    # Prince of Wales Island
    [(-99.3, 73.0), (-99.1, 73.35), (-99.25, 73.7),
     (-99.05, 74.05), (-98.75, 74.3), (-98.3, 74.55),
     (-97.8, 74.65), (-97.35, 74.6), (-96.95, 74.45),
     (-96.55, 74.22), (-96.2, 73.95), (-95.95, 73.65),
     (-95.8, 73.3), (-95.55, 73.0)],
    # Somerset Island
    [(-95.25, 73.0), (-95.0, 73.35), (-94.95, 73.75),
     (-94.85, 74.05), (-94.65, 74.35), (-94.35, 74.55),
     (-93.95, 74.62), (-93.55, 74.5), (-93.3, 74.2),
     (-93.18, 73.85), (-93.0, 73.55), (-92.85, 73.15),
     (-92.65, 73.0)],
    # Northern Victoria Island, eastern portion
    [(-105.0, 73.0), (-104.6, 73.0), (-104.0, 73.12),
     (-103.4, 73.25), (-102.8, 73.45), (-102.3, 73.7),
     (-101.9, 73.9), (-101.55, 74.15), (-101.25, 74.4),
     (-100.95, 74.65), (-100.65, 74.9), (-100.35, 75.0),
     (-100.05, 74.85), (-100.0, 74.55), (-100.1, 74.25),
     (-100.0, 73.95), (-100.15, 73.65), (-100.45, 73.35),
     (-100.8, 73.0)],
    # Southern/eastern Victoria Island, connected to western portion
    [(-105.0, 73.0), (-105.0, 74.6), (-104.5, 74.45),
     (-104.0, 74.35), (-103.5, 74.42), (-103.0, 74.55),
     (-102.5, 74.62), (-102.0, 74.55), (-101.6, 74.4),
     (-101.25, 74.2), (-101.0, 73.95), (-100.8, 73.65),
     (-100.8, 73.0)],
    # Small western rim of Banks Island is outside; no land added here.
]
def inside(lon, lat, polygon):
    """Ray-crossing point-in-polygon test; outlines are hand-authored above."""
    hit = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        # Count a crossing of a horizontal ray, avoiding division by zero.
        if (yi > lat) != (yj > lat):
            crossing_x = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing_x:
                hit = not hit
        j = i
    return hit


def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in range(61):
        lat = 79.0 - 0.1 * r
        row = []
        for c in range(241):
            lon = -105.0 + 0.1 * c
            row.append(int(any(inside(lon, lat, p) for p in POLYGONS)))
        rows.append(row)
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(rows)

    # Direct binary raster: land white (255), water black (0).
    raster = np.asarray(rows, dtype=np.uint8) * 255
    image = Image.fromarray(raster, mode="L")
    image.save(out / "candidate.png")
    enlarged = image.resize((241 * 8, 61 * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(out / "candidate_8x.png")

    assert len(rows) == 61 and all(len(row) == 241 for row in rows)
    assert sum(map(len, rows)) == 14701
    assert set(np.unique(raster).tolist()) <= {0, 255}
    assert enlarged.size == (1928, 488)
    assert np.array_equal(np.asarray(enlarged)[::8, ::8], raster)
    print("validated: 61x241 binary CSV; 241x61 PNG; nearest-neighbor 8x PNG")


if __name__ == "__main__":
    main()
