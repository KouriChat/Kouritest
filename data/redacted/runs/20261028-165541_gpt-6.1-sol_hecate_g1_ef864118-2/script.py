#!/usr/bin/env python3
"""Knowledge-based schematic land/water raster for Hecate Strait."""

from pathlib import Path
import csv

import numpy as np
from PIL import Image


N = 127
OUT = Path("outputs")

# Coordinates are (degrees west, degrees north). Polygons are deliberately
# hand-drawn approximations based on broad regional geography, not map data.
# The mainland occupies the eastern side of the frame.
LAND = [
    [
        (129.00, 53.00), (129.95, 53.00), (129.92, 53.22),
        (130.02, 53.42), (129.96, 53.63), (130.12, 53.83),
        (130.05, 54.03), (130.20, 54.26), (130.13, 54.48),
        (130.31, 54.70), (130.22, 54.86), (130.33, 55.00),
        (129.00, 55.00),
    ],
    # Banks Island and nearby larger islands in the northern strait.
    [
        (130.32, 53.83), (130.55, 53.82), (130.73, 53.95),
        (130.77, 54.15), (130.65, 54.33), (130.47, 54.34),
        (130.33, 54.22), (130.26, 54.02),
    ],
    [
        (130.76, 53.78), (130.89, 53.83), (130.94, 53.96),
        (130.88, 54.07), (130.77, 54.02),
    ],
    # Eastern ends of Graham Island and Moresby Island, clipped at 131 W.
    [
        (130.02, 54.02), (130.20, 54.04), (130.30, 54.12),
        (130.40, 54.22), (130.57, 54.30), (130.72, 54.38),
        (130.88, 54.48), (131.00, 54.55), (131.00, 54.05),
        (130.80, 53.98), (130.58, 53.96), (130.42, 53.91),
        (130.24, 53.89),
    ],
    [
        (130.13, 53.35), (130.30, 53.36), (130.46, 53.44),
        (130.62, 53.53), (130.79, 53.63), (130.97, 53.74),
        (131.00, 53.91), (130.82, 53.87), (130.65, 53.80),
        (130.48, 53.72), (130.31, 53.62), (130.18, 53.50),
    ],
]

# Small islands, islets, and reef groups are represented as compact polygons.
LAND += [
    [(130.93, 54.16), (130.99, 54.18), (131.00, 54.24), (130.95, 54.25)],
    [(130.91, 53.55), (130.97, 53.57), (130.99, 53.63), (130.94, 53.64)],
    [(130.69, 54.50), (130.76, 54.51), (130.78, 54.57), (130.71, 54.58)],
    [(130.83, 54.62), (130.90, 54.63), (130.91, 54.69), (130.85, 54.70)],
    [(130.47, 53.73), (130.52, 53.74), (130.53, 53.79), (130.48, 53.80)],
    [(130.57, 53.66), (130.61, 53.67), (130.62, 53.71), (130.58, 53.72)],
    [(130.97, 53.30), (131.00, 53.31), (131.00, 53.36), (130.98, 53.36)],
    [(130.90, 54.78), (130.96, 54.79), (130.98, 54.84), (130.93, 54.85)],
]
# Approximate sea inlets cut into the mainland coast.
WATER = [
    [(129.72, 53.12), (129.97, 53.18), (130.02, 53.23), (129.94, 53.27)],
    [(129.73, 53.48), (129.98, 53.50), (130.03, 53.54), (129.94, 53.58)],
    [(129.76, 53.78), (130.08, 53.80), (130.13, 53.84), (130.00, 53.88)],
    [(129.77, 54.14), (130.06, 54.16), (130.18, 54.21), (130.02, 54.24)],
    [(129.81, 54.53), (130.13, 54.56), (130.27, 54.62), (130.08, 54.66)],
    [(129.78, 54.86), (130.22, 54.88), (130.31, 54.92), (130.12, 54.95)],
]


def inside(lon_west, lat, polygon):
    """Return point-in-polygon using ray casting, including boundary as land."""
    hit = False
    count = len(polygon)
    for i in range(count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]
        cross = (lon_west - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
        if abs(cross) < 1e-10 and min(x1, x2) - 1e-10 <= lon_west <= max(x1, x2) + 1e-10:
            if min(y1, y2) - 1e-10 <= lat <= max(y1, y2) + 1e-10:
                return True
        if (y1 > lat) != (y2 > lat):
            crossing_x = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon_west < crossing_x:
                hit = not hit
    return hit


def classify(lon_west, lat):
    land = any(inside(lon_west, lat, polygon) for polygon in LAND)
    if land and any(inside(lon_west, lat, inlet) for inlet in WATER):
        land = False
    return int(land)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    matrix = np.zeros((N, N), dtype=np.uint8)
    for row in range(N):
        lat = 55.0 - row * (2.0 / 126.0)
        for col in range(N):
            lon_west = 131.0 - col * (2.0 / 126.0)
            matrix[row, col] = classify(lon_west, lat)

    with (OUT / "candidate.csv").open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(matrix.tolist())

    pixels = np.where(matrix == 1, 255, 0).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    image.save(OUT / "candidate.png")
    image.resize((1016, 1016), Image.Resampling.NEAREST).save(
        OUT / "candidate_8x.png"
    )

    assert matrix.shape == (127, 127) and matrix.size == 16129
    assert set(np.unique(matrix)).issubset({0, 1})
    assert Image.open(OUT / "candidate.png").size == (127, 127)
    assert Image.open(OUT / "candidate_8x.png").size == (1016, 1016)
    print(f"Generated {matrix.size} classified points; land={int(matrix.sum())}.")


if __name__ == "__main__":
    main()
