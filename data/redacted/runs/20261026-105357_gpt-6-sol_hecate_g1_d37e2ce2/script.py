#!/usr/bin/env python3
"""Approximate Hecate Strait land/water raster from hand-authored polygons."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image


# Coordinates are (longitude, latitude), in WGS84 decimal degrees.
# The mainland polygon follows an approximate Pacific-facing shoreline.
LAND_POLYGONS = [
    [
        (-129.43, 55.00), (-129.51, 54.88), (-129.57, 54.77),
        (-129.67, 54.66), (-129.61, 54.55), (-129.73, 54.45),
        (-129.64, 54.34), (-129.76, 54.22), (-129.67, 54.12),
        (-129.55, 54.03), (-129.61, 53.94), (-129.52, 53.84),
        (-129.59, 53.74), (-129.43, 53.66), (-129.40, 53.55),
        (-129.29, 53.46), (-129.33, 53.36), (-129.22, 53.27),
        (-129.32, 53.17), (-129.37, 53.08), (-129.45, 53.00),
        (-127.0, 53.0), (-127.0, 55.0),
    ],
    # Banks Island
    [
        (-130.12, 54.52), (-130.03, 54.49), (-129.99, 54.43),
        (-129.91, 54.41), (-129.87, 54.35), (-129.82, 54.31),
        (-129.86, 54.24), (-129.83, 54.17), (-129.89, 54.11),
        (-129.96, 54.10), (-130.01, 54.15), (-130.10, 54.16),
        (-130.15, 54.22), (-130.20, 54.27), (-130.18, 54.36),
        (-130.23, 54.41), (-130.19, 54.47),
    ],
    # North and south ends of Banks Island's nearby island group
    [
        (-130.28, 54.56), (-130.24, 54.54), (-130.22, 54.50),
        (-130.26, 54.47), (-130.31, 54.49), (-130.33, 54.53),
    ],
    [
        (-130.37, 54.08), (-130.32, 54.06), (-130.29, 54.02),
        (-130.33, 53.99), (-130.39, 54.00), (-130.42, 54.04),
    ],
    # Small islands and reefs around the eastern Hecate Strait
    [
        (-129.91, 53.94), (-129.88, 53.91), (-129.84, 53.92),
        (-129.82, 53.96), (-129.86, 53.99), (-129.90, 53.98),
    ],
    [
        (-130.12, 53.83), (-130.08, 53.80), (-130.03, 53.81),
        (-130.02, 53.85), (-130.06, 53.88), (-130.11, 53.87),
    ],
    [
        (-129.75, 53.57), (-129.72, 53.54), (-129.67, 53.55),
        (-129.65, 53.59), (-129.69, 53.62), (-129.73, 53.61),
    ],
]


def on_segment(px, py, ax, ay, bx, by):
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > 1e-12:
        return False
    return (min(ax, bx) - 1e-12 <= px <= max(ax, bx) + 1e-12
            and min(ay, by) - 1e-12 <= py <= max(ay, by) + 1e-12)


def contains(lon, lat, polygon):
    inside = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        if on_segment(lon, lat, x1, y1, x2, y2):
            return True
        if (y1 > lat) != (y2 > lat):
            crossing_lon = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < crossing_lon:
                inside = not inside
    return inside


def main():
    count = 127
    latitudes = [55.0 - i * (2.0 / 126.0) for i in range(count)]
    longitudes = [-131.0 + j * (2.0 / 126.0) for j in range(count)]
    matrix = np.zeros((count, count), dtype=np.uint8)

    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            matrix[row, col] = any(
                contains(lon, lat, polygon) for polygon in LAND_POLYGONS
            )

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "candidate.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # PIL's 0 is black (water), 255 is white (land).
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(output_dir / "candidate.png")
    image.resize((1016, 1016), resample=Image.Resampling.NEAREST).save(
        output_dir / "candidate_8x.png"
    )


if __name__ == "__main__":
    main()
