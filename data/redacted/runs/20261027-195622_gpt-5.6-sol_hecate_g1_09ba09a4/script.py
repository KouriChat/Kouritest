#!/usr/bin/env python3
"""Approximate Hecate Strait land/water raster from hand-authored polygons."""

from pathlib import Path
import csv

import numpy as np
from PIL import Image

N = 127
LAT_NORTH, LAT_SOUTH = 55.0, 53.0
LON_WEST, LON_EAST = -131.0, -129.0
STEP = 2.0 / 126.0

# Polygons are lon/lat vertices, hand-estimated from geographic knowledge.
# Mainland coastal edge is explicitly traced; eastern land is then filled.
MAINLAND = [
    (-130.58, 55.00), (-130.48, 54.92), (-130.52, 54.84),
    (-130.43, 54.77), (-130.39, 54.68), (-130.48, 54.60),
    (-130.42, 54.53), (-130.34, 54.46), (-130.28, 54.37),
    (-130.35, 54.29), (-130.30, 54.22), (-130.16, 54.15),
    (-130.20, 54.08), (-130.13, 54.00), (-130.06, 53.93),
    (-130.12, 53.86), (-130.04, 53.79), (-129.96, 53.72),
    (-130.02, 53.65), (-129.93, 53.58), (-129.88, 53.51),
    (-129.91, 53.43), (-129.82, 53.36), (-129.78, 53.28),
    (-129.83, 53.20), (-129.72, 53.12), (-129.66, 53.00),
    (-129.00, 53.00), (-129.00, 55.00),
]

# Major islands and coastal island groups inside the requested rectangle.
ISLANDS = [
    # Graham Island: eastern and northern portions of Haida Gwaii.
    [(-131.00, 55.00), (-130.84, 55.00), (-130.82, 54.94),
     (-130.72, 54.90), (-130.75, 54.84), (-130.64, 54.79),
     (-130.67, 54.73), (-130.54, 54.68), (-130.48, 54.62),
     (-130.55, 54.57), (-130.50, 54.51), (-130.48, 54.44),
     (-130.54, 54.38), (-130.60, 54.31), (-130.67, 54.24),
     (-130.74, 54.18), (-130.83, 54.14), (-130.91, 54.16),
     (-131.00, 54.20)],
    # Moresby Island northern edge and east coast, clipped by map boundary.
    [(-131.00, 53.90), (-130.93, 53.93), (-130.87, 53.88),
     (-130.82, 53.80), (-130.79, 53.72), (-130.73, 53.65),
     (-130.72, 53.56), (-130.67, 53.48), (-130.66, 53.39),
     (-130.60, 53.31), (-130.57, 53.22), (-130.51, 53.13),
     (-130.47, 53.00), (-131.00, 53.00)],
    # Banks Island and nearby larger islands.
    [(-129.98, 54.13), (-129.89, 54.15), (-129.82, 54.10),
     (-129.79, 54.00), (-129.81, 53.89), (-129.87, 53.82),
     (-129.95, 53.84), (-130.00, 53.93)],
    [(-130.21, 54.25), (-130.15, 54.29), (-130.10, 54.25),
     (-130.08, 54.17), (-130.14, 54.13), (-130.20, 54.17)],
]
# Additional coastal islands: deliberately generalized, not surveyed outlines.
ISLANDS += [
    # Porcher Island and adjacent islands.
    [(-130.53, 53.98), (-130.43, 54.03), (-130.34, 54.00),
     (-130.29, 53.93), (-130.31, 53.84), (-130.39, 53.79),
     (-130.48, 53.82), (-130.55, 53.89)],
    [(-130.46, 53.75), (-130.39, 53.77), (-130.33, 53.72),
     (-130.35, 53.65), (-130.42, 53.63), (-130.48, 53.67)],
    # Stephens, Gil, and scattered islands in the southern strait.
    [(-130.28, 53.53), (-130.19, 53.56), (-130.12, 53.50),
     (-130.13, 53.42), (-130.22, 53.39), (-130.29, 53.44)],
    [(-130.48, 53.36), (-130.41, 53.39), (-130.35, 53.34),
     (-130.36, 53.27), (-130.43, 53.23), (-130.49, 53.28)],
    # Small island patches in Dixon Entrance and near the mainland coast.
    [(-130.18, 54.78), (-130.11, 54.81), (-130.05, 54.77),
     (-130.07, 54.71), (-130.14, 54.70)],
    [(-129.99, 54.57), (-129.93, 54.60), (-129.87, 54.55),
     (-129.89, 54.49), (-129.96, 54.48)],
    [(-130.44, 54.49), (-130.39, 54.52), (-130.34, 54.48),
     (-130.36, 54.42), (-130.42, 54.41)],
]


def inside_polygon(lon, lat, polygon):
    """Even-odd point-in-polygon test; boundary points count as land."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        cross = (lon - xi) * (yj - yi) - (lat - yi) * (xj - xi)
        if abs(cross) < 1e-12 and min(xi, xj) <= lon <= max(xi, xj) \
                and min(yi, yj) <= lat <= max(yi, yj):
            return True
        if (yi > lat) != (yj > lat):
            crossing_lon = xi + (lat - yi) * (xj - xi) / (yj - yi)
            if lon < crossing_lon:
                inside = not inside
        j = i
    return inside


def is_land(lon, lat):
    return inside_polygon(lon, lat, MAINLAND) or any(
        inside_polygon(lon, lat, polygon) for polygon in ISLANDS
    )
def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    lats = [LAT_NORTH - row * STEP for row in range(N)]
    lons = [LON_WEST + col * STEP for col in range(N)]
    matrix = np.array(
        [[int(is_land(lon, lat)) for lon in lons] for lat in lats],
        dtype=np.uint8,
    )

    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Land is white (255); water is black (0).
    pixels = np.where(matrix == 1, 255, 0).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    image.save(out / "candidate.png")
    image.resize((N * 8, N * 8), resample=Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )

    assert matrix.shape == (127, 127)
    assert matrix.size == 16129 and np.isin(matrix, [0, 1]).all()
    with (out / "candidate.csv").open(newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 127 and all(
        len(row) == 127 and all(value in {"0", "1"} for value in row)
        for row in rows
    )
    assert image.size == (127, 127)
    assert set(image.getdata()) == {0, 255}
    enlarged = Image.open(out / "candidate_8x.png")
    assert enlarged.size == (1016, 1016)
    assert list(enlarged.resize((127, 127), Image.Resampling.NEAREST).getdata()) \
        == list(image.getdata())


if __name__ == "__main__":
    main()
