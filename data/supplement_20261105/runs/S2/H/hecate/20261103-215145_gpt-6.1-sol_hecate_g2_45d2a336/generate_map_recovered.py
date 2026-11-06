#!/usr/bin/env python3
"""Approximate Hecate Strait land/water mask from hand-authored polygons."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 127
STEP = 2.0 / 126.0

# Coordinates are (longitude, latitude) in WGS84 decimal degrees.
# These approximate polygons are hand-authored, not loaded from a map source.
#
# The mainland polygon follows an approximate shoreline from north to south;
# the area east of that shoreline is classified as land. The remaining
# polygons represent nearby islands, including approximate parts of Haida
# Gwaii and the larger islands east of Hecate Strait.
LAND_POLYGONS = [
    # Mainland coast and adjacent land, including the frame's eastern edge.
    [
        (-130.05, 55.00),
        (-130.08, 54.88),
        (-130.18, 54.76),
        (-130.12, 54.65),
        (-130.23, 54.54),
        (-130.12, 54.43),
        (-130.18, 54.32),
        (-130.08, 54.20),
        (-130.17, 54.08),
        (-130.10, 53.97),
        (-130.20, 53.86),
        (-130.08, 53.75),
        (-130.12, 53.64),
        (-129.98, 53.53),
        (-129.98, 53.42),
        (-129.84, 53.31),
        (-129.78, 53.20),
        (-129.72, 53.09),
        (-129.70, 53.00),
        (-129.00, 53.00),
        (-129.00, 55.00),
    ],

    # Banks Island, with a simplified indented outline.
    [
        (-130.28, 54.48),
        (-130.18, 54.43),
        (-130.12, 54.34),
        (-130.11, 54.20),
        (-130.16, 54.08),
        (-130.23, 54.02),
        (-130.31, 54.05),
        (-130.39, 54.13),
        (-130.42, 54.25),
        (-130.39, 54.36),
        (-130.34, 54.44),
    ],

    # Pitt Island.
    [
        (-130.37, 53.66),
        (-130.26, 53.64),
        (-130.16, 53.58),
        (-130.08, 53.48),
        (-130.07, 53.36),
        (-130.13, 53.25),
        (-130.23, 53.19),
        (-130.34, 53.22),
        (-130.42, 53.31),
        (-130.45, 53.44),
        (-130.43, 53.56),
    ],

    # Smaller coastal islands near the mainland.
    [
        (-130.43, 54.08),
        (-130.37, 54.04),
        (-130.31, 54.07),
        (-130.29, 54.15),
        (-130.33, 54.21),
        (-130.40, 54.19),
        (-130.44, 54.14),
    ],
    [
        (-130.50, 53.86),
        (-130.44, 53.82),
        (-130.38, 53.85),
        (-130.37, 53.92),
        (-130.43, 53.96),
        (-130.49, 53.92),
    ],

    # Approximate eastern edge of Graham Island (Haida Gwaii).
    # Most of the island lies west of the requested longitude frame.
    [
        (-131.00, 54.34),
        (-131.04, 54.27),
        (-131.05, 54.17),
        (-131.09, 54.08),
        (-131.10, 53.99),
        (-131.16, 53.93),
        (-131.28, 53.91),
        (-131.42, 53.94),
        (-131.42, 54.42),
        (-131.12, 54.42),
    ],

    # Approximate northern/eastern tip of Moresby Island.
    [
        (-131.00, 53.83),
        (-131.05, 53.77),
        (-131.08, 53.69),
        (-131.12, 53.62),
        (-131.20, 53.57),
        (-131.38, 53.56),
        (-131.38, 53.86),
        (-131.17, 53.87),
    ],
]


def points_in_polygon(longitudes, latitudes, polygon):
    """Return a boolean mask for points inside a polygon (ray casting)."""
    inside = np.zeros(longitudes.shape, dtype=bool)
    x1, y1 = polygon[-1]

    for x2, y2 in polygon:
        crosses_y = (y1 > latitudes) != (y2 > latitudes)
        denominator = y2 - y1
        if denominator == 0:
            denominator = 1e-300
        crossing_x = (x2 - x1) * (latitudes - y1) / denominator + x1
        inside ^= crosses_y & (longitudes < crossing_x)
        x1, y1 = x2, y2

    return inside


def build_matrix():
    # Rows run north to south; columns run west to east.
    latitudes = 55.0 - np.arange(SIZE, dtype=np.float64) * STEP
    longitudes = -131.0 + np.arange(SIZE, dtype=np.float64) * STEP
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    land = np.zeros((SIZE, SIZE), dtype=bool)
    for polygon in LAND_POLYGONS:
        land |= points_in_polygon(lon_grid, lat_grid, polygon)

    return land.astype(np.uint8)


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = build_matrix()
    if matrix.shape != (127, 127) or matrix.size != 16129:
        raise RuntimeError("Unexpected matrix dimensions")
    if not np.isin(matrix, (0, 1)).all():
        raise RuntimeError("Matrix contains values other than 0 and 1")

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # 1=land -> white; 0=water -> black.
    pixels = matrix * 255
    image = Image.fromarray(pixels, mode="L")
    image.save(output_dir / "candidate.png")

    enlarged = image.resize((1016, 1016), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")

    # Validate the written deliverables.
    with csv_path.open("r", newline="", encoding="ascii") as csv_file:
        rows = list(csv.reader(csv_file))
    assert len(rows) == 127
    assert all(len(row) == 127 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    check_image = Image.open(output_dir / "candidate.png")
    check_enlarged = Image.open(output_dir / "candidate_8x.png")
    assert check_image.size == (127, 127)
    assert set(check_image.getdata()) <= {0, 255}
    assert check_enlarged.size == (1016, 1016)
    assert set(check_enlarged.getdata()) <= {0, 255}


if __name__ == "__main__":
    main()
