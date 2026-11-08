#!/usr/bin/env python3
"""Knowledge-based approximate land/water raster for Hecate Strait."""

from pathlib import Path
import csv

import numpy as np
from PIL import Image


LAT_NORTH, LAT_SOUTH = 55.0, 53.0
LON_WEST, LON_EAST = -131.0, -129.0
N = 127
STEP = 2.0 / 126.0


def inside(lon, lat, polygon):
    """Ray-casting point-in-polygon; vertices are (lon, lat)."""
    hit = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > lat) != (yj > lat):
            cross = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < cross:
                hit = not hit
        j = i
    return hit


# Hand-drawn broad coastline outlines. Coordinates are WGS84 lon/lat.
# Mainland polygons include the coastal fjords only approximately.
LAND_POLYGONS = [
    # Mainland mass, with an irregular west coast facing Hecate Strait.
    [
        (-129.0, 53.0), (-129.0, 55.0), (-130.04, 55.0),
        (-130.10, 54.88), (-130.18, 54.80), (-130.32, 54.76),
        (-130.42, 54.66), (-130.38, 54.56), (-130.48, 54.48),
        (-130.40, 54.36), (-130.50, 54.28), (-130.43, 54.17),
        (-130.33, 54.08), (-130.39, 53.98), (-130.25, 53.87),
        (-130.22, 53.75), (-130.12, 53.64), (-130.02, 53.53),
        (-129.91, 53.42), (-129.82, 53.30), (-129.67, 53.22),
        (-129.60, 53.12), (-129.48, 53.08), (-129.39, 53.00),
    ],
    # Large coastal islands and peninsulas along the northern mainland.
    [
        (-130.52, 54.82), (-130.40, 54.82), (-130.31, 54.77),
        (-130.34, 54.70), (-130.45, 54.69), (-130.53, 54.74),
    ],
    [
        (-130.78, 54.59), (-130.65, 54.62), (-130.57, 54.57),
        (-130.62, 54.49), (-130.76, 54.50), (-130.83, 54.55),
    ],
]

# Island groups on the exposed shelf and inside the mainland channels.
# Each ring is deliberately simplified, retaining only islands likely to
# intersect this two-degree window.
LAND_POLYGONS.extend([
    # Banks Island, eastern side of the northern Dixon Entrance.
    [
        (-130.30, 54.45), (-130.20, 54.46), (-130.12, 54.40),
        (-130.10, 54.31), (-130.15, 54.25), (-130.25, 54.29),
        (-130.32, 54.36),
    ],
    # Dundas Island and nearby north-coast islands.
    [
        (-130.95, 54.65), (-130.84, 54.67), (-130.76, 54.62),
        (-130.78, 54.56), (-130.90, 54.55), (-130.98, 54.59),
    ],
    [
        (-130.69, 54.70), (-130.61, 54.72), (-130.55, 54.68),
        (-130.59, 54.63), (-130.67, 54.64),
    ],
    # Porcher Island and adjacent islands.
    [
        (-130.48, 53.98), (-130.36, 54.00), (-130.26, 53.94),
        (-130.25, 53.85), (-130.32, 53.80), (-130.44, 53.83),
        (-130.52, 53.90),
    ],
    [
        (-130.65, 53.94), (-130.58, 53.96), (-130.53, 53.91),
        (-130.58, 53.87), (-130.64, 53.89),
    ],
    # Central shelf islands.
    [
        (-130.69, 53.74), (-130.58, 53.77), (-130.49, 53.71),
        (-130.48, 53.63), (-130.54, 53.57), (-130.66, 53.60),
        (-130.73, 53.67),
    ],
    [
        (-130.81, 53.62), (-130.73, 53.64), (-130.69, 53.59),
        (-130.73, 53.54), (-130.80, 53.55),
    ],
    # Aristazabal Island.
    [
        (-129.98, 53.52), (-129.88, 53.56), (-129.79, 53.51),
        (-129.77, 53.40), (-129.82, 53.31), (-129.91, 53.29),
        (-130.00, 53.35), (-130.05, 53.44),
    ],
    # Estevan Group and nearby coastal islands.
    [
        (-130.17, 53.34), (-130.07, 53.38), (-129.99, 53.33),
        (-130.01, 53.25), (-130.10, 53.23), (-130.18, 53.28),
    ],
    [
        (-129.76, 53.16), (-129.66, 53.20), (-129.57, 53.14),
        (-129.59, 53.07), (-129.70, 53.05), (-129.78, 53.10),
    ],
])
def build_matrix():
    matrix = np.zeros((N, N), dtype=np.uint8)
    for row in range(N):
        lat = LAT_NORTH - row * STEP
        for col in range(N):
            lon = LON_WEST + col * STEP
            if any(inside(lon, lat, polygon) for polygon in LAND_POLYGONS):
                matrix[row, col] = 1
    return matrix


def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()

    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # 1 is white land; 0 is black water.
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(out / "candidate.png")
    enlarged = image.resize((N * 8, N * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(out / "candidate_8x.png")

    rows = list(csv.reader((out / "candidate.csv").open(newline="")))
    assert len(rows) == N and all(len(r) == N for r in rows)
    assert all(v in ("0", "1") for r in rows for v in r)
    assert sum(map(len, rows)) == 16129
    assert image.size == (127, 127)
    assert set(image.getdata()) == {0, 255}
    assert enlarged.size == (1016, 1016)
    assert set(enlarged.getdata()) == {0, 255}
    for y in range(N):
        for x in range(N):
            assert enlarged.getpixel((x * 8, y * 8)) == image.getpixel((x, y))
    print("Validated: 127x127 CSV/PNG and nearest-neighbor 1016x1016 PNG.")


if __name__ == "__main__":
    main()
