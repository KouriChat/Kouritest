#!/usr/bin/env python3
"""Approximate Hecate Strait land/water map from hand-written geographic rules.

This is a rule-based reconstruction from general geographic knowledge, not a
map lookup and not independent reasoning for each sample point. Polygon
vertices are approximate longitude/latitude coordinates (west is negative).
"""
from pathlib import Path
import csv

import numpy as np
from PIL import Image, ImageDraw

SIZE = 127
STEP = 2.0 / 126.0
WEST = -131.0
NORTH = 55.0

# Approximate land outlines. The mainland occupies the eastern side of the
# frame. Haida Gwaii is represented by Graham Island in the north and Moresby
# Island to the south, with their channels left as water.
LAND_POLYGONS = [
    # Mainland coast, from the Alaska/BC border southward.
    [
        (-129.00, 55.00), (-129.00, 53.00), (-129.45, 53.00),
        (-129.54, 53.12), (-129.68, 53.22), (-129.72, 53.34),
        (-129.82, 53.42), (-129.83, 53.54), (-129.94, 53.65),
        (-129.93, 53.77), (-130.02, 53.86), (-129.98, 53.98),
        (-130.10, 54.08), (-130.10, 54.19), (-130.20, 54.29),
        (-130.17, 54.41), (-130.27, 54.51), (-130.22, 54.62),
        (-130.31, 54.71), (-130.28, 54.84), (-130.39, 54.94),
        (-130.39, 55.00),
    ],
    # Graham Island (northern, larger island of Haida Gwaii).
    [
        (-131.00, 54.99), (-130.92, 55.00), (-130.79, 54.96),
        (-130.70, 54.91), (-130.57, 54.89), (-130.49, 54.83),
        (-130.38, 54.82), (-130.32, 54.75), (-130.23, 54.71),
        (-130.19, 54.62), (-130.20, 54.51), (-130.26, 54.43),
        (-130.35, 54.42), (-130.42, 54.46), (-130.51, 54.45),
        (-130.59, 54.50), (-130.68, 54.50), (-130.77, 54.55),
        (-130.88, 54.54), (-130.96, 54.58), (-131.00, 54.58),
    ],
    # Moresby Island. Its eastern shore faces Hecate Strait.
    [
        (-131.00, 53.99), (-130.91, 54.00), (-130.83, 54.05),
        (-130.73, 54.06), (-130.63, 54.12), (-130.53, 54.13),
        (-130.45, 54.19), (-130.36, 54.20), (-130.30, 54.25),
        (-130.28, 54.34), (-130.32, 54.42), (-130.40, 54.43),
        (-130.48, 54.39), (-130.56, 54.35), (-130.65, 54.33),
        (-130.73, 54.27), (-130.84, 54.25), (-130.92, 54.20),
        (-131.00, 54.18),
    ],
    # Vancouver Island's northern tip barely enters the southwest corner.
    [
        (-131.00, 53.00), (-130.81, 53.00), (-130.82, 53.05),
        (-130.89, 53.10), (-130.91, 53.17), (-130.98, 53.20),
        (-131.00, 53.20),
    ],
]

# Smaller islands and coastal fragments, as approximate closed polygons.
ISLAND_POLYGONS = [
    # Banks Island and nearby islands in the northeast.
    [(-130.39, 54.96), (-130.31, 54.96), (-130.27, 55.00), (-130.39, 55.00)],
    [(-130.45, 54.79), (-130.39, 54.81), (-130.36, 54.77), (-130.41, 54.74)],
    # Haida Gwaii's west-coast islands and southern island group.
    [(-131.00, 54.39), (-130.96, 54.40), (-130.95, 54.36), (-131.00, 54.35)],
    [(-131.00, 54.29), (-130.96, 54.31), (-130.95, 54.27), (-131.00, 54.25)],
    [(-130.99, 54.08), (-130.94, 54.10), (-130.92, 54.06), (-130.98, 54.04)],
    [(-130.91, 53.97), (-130.85, 53.99), (-130.83, 53.95), (-130.89, 53.94)],
    # Aristazabal, Gil, and other offshore mainland islands.
    [(-130.77, 53.42), (-130.70, 53.43), (-130.68, 53.38), (-130.74, 53.36)],
    [(-130.67, 53.56), (-130.61, 53.57), (-130.59, 53.52), (-130.65, 53.51)],
    [(-130.56, 53.75), (-130.51, 53.78), (-130.48, 53.73), (-130.54, 53.70)],
    [(-130.47, 53.94), (-130.41, 53.95), (-130.39, 53.91), (-130.45, 53.89)],
    [(-130.52, 54.31), (-130.47, 54.33), (-130.44, 54.29), (-130.49, 54.27)],
    [(-130.64, 54.55), (-130.60, 54.57), (-130.58, 54.53), (-130.63, 54.52)],
    [(-130.52, 54.67), (-130.48, 54.69), (-130.46, 54.65), (-130.51, 54.64)],
]


def point_on_segment(x, y, ax, ay, bx, by, eps=1e-10):
    cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
    if abs(cross) > eps:
        return False
    return (min(ax, bx) - eps <= x <= max(ax, bx) + eps and
            min(ay, by) - eps <= y <= max(ay, by) + eps)


def point_in_polygon(x, y, polygon):
    """Ray casting; polygon edges are treated as land."""
    inside = False
    n = len(polygon)
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]
        if point_on_segment(x, y, ax, ay, bx, by):
            return True
        if (ay > y) != (by > y):
            crossing_x = ax + (y - ay) * (bx - ax) / (by - ay)
            if x < crossing_x:
                inside = not inside
    return inside


def main():
    longitudes = WEST + np.arange(SIZE) * STEP
    latitudes = NORTH - np.arange(SIZE) * STEP
    polygons = LAND_POLYGONS + ISLAND_POLYGONS

    matrix = np.zeros((SIZE, SIZE), dtype=np.uint8)
    for row, latitude in enumerate(latitudes):
        for col, longitude in enumerate(longitudes):
            if any(point_in_polygon(longitude, latitude, polygon)
                   for polygon in polygons):
                matrix[row, col] = 1

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "candidate.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Land is white (255); water is black (0).
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(output_dir / "candidate.png")
    image.resize((SIZE * 8, SIZE * 8), Image.Resampling.NEAREST).save(
        output_dir / "candidate_8x.png"
    )

    assert matrix.shape == (127, 127)
    assert int(matrix.size) == 16129
    assert set(np.unique(matrix)).issubset({0, 1})
    with (output_dir / "candidate.csv").open(newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 127 and all(
        len(row) == 127 and all(value in {"0", "1"} for value in row)
        for row in rows
    )
    assert image.size == (127, 127)
    assert set(image.getdata()).issubset({0, 255})
    enlarged = Image.open(output_dir / "candidate_8x.png")
    assert enlarged.size == (1016, 1016)
    assert enlarged.getpixel((8, 8)) == image.getpixel((1, 1))

    print(f"Generated {matrix.shape[0]}x{matrix.shape[1]} matrix "
          f"({matrix.size} values) in {output_dir}/")


if __name__ == "__main__":
    main()
