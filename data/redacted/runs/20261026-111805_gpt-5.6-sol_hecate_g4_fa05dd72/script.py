#!/usr/bin/env python3
"""Hand-drawn point-classification approximation for Hecate Strait."""

from pathlib import Path
import csv

import numpy as np
from PIL import Image


N = 127
STEP = 2.0 / 126.0

# Coordinates are longitude degrees west, latitude degrees north.
# The mainland polygon traces an approximate outer Pacific coast, with
# the eastern edge deliberately beyond the requested map window.
MAINLAND = [
    (130.28, 55.00), (130.32, 54.88), (130.25, 54.78), (130.34, 54.66),
    (130.30, 54.56), (130.20, 54.49), (130.25, 54.40), (130.13, 54.31),
    (130.18, 54.20), (130.07, 54.10), (130.17, 54.00), (130.08, 53.91),
    (130.12, 53.82), (129.98, 53.72), (130.02, 53.63), (129.86, 53.55),
    (129.91, 53.45), (129.77, 53.37), (129.81, 53.27), (129.68, 53.19),
    (129.70, 53.00), (128.00, 53.00), (128.00, 55.00),
]

# Hand-drawn islands and outer-coast headlands, generalized at this scale.
ISLANDS = [
    # Banks Island
    [(129.95, 54.03), (130.12, 54.03), (130.20, 53.94),
     (130.20, 53.80), (130.12, 53.70), (130.00, 53.72),
     (129.91, 53.83), (129.92, 53.96)],
    # Pitt Island
    [(129.78, 53.92), (129.91, 53.94), (129.96, 53.85),
     (129.92, 53.70), (129.84, 53.62), (129.76, 53.70),
     (129.74, 53.82)],
    # Porcher Island
    [(130.20, 53.93), (130.31, 53.96), (130.34, 53.86),
     (130.29, 53.77), (130.19, 53.81)],
    # Several small islands around the lower mainland coast.
    [(129.63, 53.20), (129.73, 53.24), (129.77, 53.16),
     (129.72, 53.08), (129.62, 53.10)],
    [(129.85, 53.48), (129.94, 53.51), (129.98, 53.44),
     (129.91, 53.39), (129.84, 53.42)],
    # Eastern fringe of Graham Island (Haida Gwaii); most of it lies west
    # of the requested frame, so only the eastern coast is represented.
    [(131.00, 55.00), (131.00, 54.78), (131.02, 54.66),
     (131.05, 54.55), (131.04, 54.43), (131.02, 54.30),
     (131.00, 54.18), (130.94, 54.17), (130.93, 54.34),
     (130.94, 54.50), (130.93, 54.66), (130.95, 54.82),
     (130.94, 55.00)],
]


def contains(poly, x, y):
    """Ray-crossing point-in-polygon test."""
    inside = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < cross:
                inside = not inside
        j = i
    return inside


def classify(lon_w, lat):
    return int(contains(MAINLAND, lon_w, lat) or
               any(contains(poly, lon_w, lat) for poly in ISLANDS))


def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    matrix = np.zeros((N, N), dtype=np.uint8)
    for row in range(N):
        lat = 55.0 - row * STEP
        for col in range(N):
            lon_w = 131.0 - col * STEP
            matrix[row, col] = classify(lon_w, lat)

    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    image = Image.fromarray(np.where(matrix == 1, 255, 0).astype(np.uint8), "L")
    image.save(out / "candidate.png")
    image.resize((N * 8, N * 8), Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )


if __name__ == "__main__":
    main()
