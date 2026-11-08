#!/usr/bin/env python3
"""Approximate point-classification map of Hecate Strait; no external data."""

from pathlib import Path
import csv

import numpy as np
from PIL import Image


ROWS = COLS = 127
STEP = 2.0 / 126.0
OUT = Path("outputs")


def inside_polygon(lon_w, lat, polygon):
    """Ray-casting test; coordinates use positive degrees west."""
    hit = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > lat) != (yj > lat):
            cross = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon_w < cross:
                hit = not hit
        j = i
    return hit


# Approximate mainland coast, tracing inlets and the irregular outer shore.
# East of this boundary is the mainland; x values are degrees west.
MAINLAND = [
    (130.10, 55.00), (129.72, 54.92), (129.86, 54.82),
    (130.04, 54.73), (129.86, 54.65), (130.18, 54.57),
    (130.02, 54.49), (130.35, 54.40), (130.48, 54.31),
    (130.29, 54.21), (130.06, 54.13), (130.24, 54.04),
    (129.99, 53.96), (129.83, 53.88), (129.76, 53.78),
    (129.63, 53.69), (129.52, 53.59), (129.49, 53.48),
    (129.38, 53.37), (129.31, 53.26), (129.25, 53.15),
    (129.22, 53.00),
]


def mainland_boundary(lat):
    """Interpolate the hand-estimated western mainland edge."""
    points = sorted(MAINLAND, key=lambda p: p[1])
    if lat <= points[0][1]:
        return points[0][0]
    if lat >= points[-1][1]:
        return points[-1][0]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if y0 <= lat <= y1:
            t = (lat - y0) / (y1 - y0)
            return x0 + t * (x1 - x0)
    raise AssertionError("latitude interpolation failed")
# Island fragments reaching the western edge of this crop. Haida Gwaii lies
# mostly west of 131 W; these small polygons represent only the estimated
# easternmost Graham and Moresby shores within the requested frame.
ISLANDS = [
    [(131.00, 54.72), (130.985, 54.84), (130.99, 54.96),
     (131.00, 55.00), (131.00, 54.72)],
    [(131.00, 53.92), (130.985, 54.02), (130.99, 54.12),
     (131.00, 54.16), (131.00, 53.92)],
    # Small nearshore islands and reefs, represented as land point targets.
    [(130.66, 54.17), (130.62, 54.21), (130.56, 54.18),
     (130.60, 54.14)],
    [(130.42, 53.89), (130.38, 53.94), (130.32, 53.91),
     (130.36, 53.87)],
    [(130.18, 53.66), (130.14, 53.71), (130.08, 53.68),
     (130.12, 53.63)],
]


def is_land(lon_w, lat):
    if lon_w <= mainland_boundary(lat):
        return True
    return any(inside_polygon(lon_w, lat, polygon) for polygon in ISLANDS)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    matrix = np.zeros((ROWS, COLS), dtype=np.uint8)
    for row in range(ROWS):
        lat = 55.0 - row * STEP
        for col in range(COLS):
            lon_w = 131.0 - col * STEP
            matrix[row, col] = int(is_land(lon_w, lat))

    with (OUT / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Value 1 (land) maps to white; value 0 (water) maps to black.
    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(OUT / "candidate.png")
    enlarged = image.resize((COLS * 8, ROWS * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(OUT / "candidate_8x.png")
    validate(matrix)


def validate(matrix):
    assert matrix.shape == (127, 127)
    assert int(matrix.size) == 16129
    with (OUT / "candidate.csv").open(newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 127 and all(len(row) == 127 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    original = Image.open(OUT / "candidate.png")
    zoomed = Image.open(OUT / "candidate_8x.png")
    assert original.size == (127, 127) and zoomed.size == (1016, 1016)
    assert set(original.getdata()) <= {(0, 0, 0), (255, 255, 255)}
    assert list(zoomed.getdata()) == list(
        original.resize((1016, 1016), Image.Resampling.NEAREST).getdata()
    )


if __name__ == "__main__":
    main()
