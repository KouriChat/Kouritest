#!/usr/bin/env python3
"""Hand-drawn, approximate land mask for the Hecate Strait region."""

from pathlib import Path

import numpy as np
from PIL import Image

N = 127
LAT_NORTH, LAT_SOUTH = 55.0, 53.0
LON_WEST, LON_EAST = -131.0, -129.0

# Approximate mainland outline. Coordinates are (longitude, latitude).
POLYGONS = [
    [
        (-130.33, 55.00), (-130.18, 54.87), (-130.12, 54.75),
        (-130.23, 54.65), (-130.10, 54.55), (-130.20, 54.43),
        (-130.05, 54.33), (-130.16, 54.22), (-130.02, 54.10),
        (-130.14, 53.98), (-129.98, 53.87), (-130.06, 53.76),
        (-129.91, 53.65), (-129.99, 53.53), (-129.85, 53.41),
        (-129.91, 53.28), (-129.76, 53.16), (-129.79, 53.00),
        (-129.00, 53.00), (-129.00, 55.00),
    ],
    # Eastern fringe of Graham Island, clipped at the west edge.
    [
        (-131.00, 55.00), (-130.94, 55.00), (-130.93, 54.94),
        (-130.95, 54.88), (-130.92, 54.81), (-130.94, 54.75),
        (-130.91, 54.69), (-130.94, 54.62), (-130.91, 54.56),
        (-130.94, 54.50), (-130.91, 54.44), (-130.95, 54.38),
        (-130.93, 54.31), (-130.96, 54.24), (-130.94, 54.17),
        (-130.98, 54.10), (-131.00, 54.06),
    ],
    # Eastern coastal fringe of Moresby Island.
    [
        (-131.00, 53.89), (-130.96, 53.85), (-130.94, 53.79),
        (-130.96, 53.72), (-130.93, 53.67), (-130.96, 53.61),
        (-130.93, 53.54), (-130.97, 53.49), (-131.00, 53.46),
    ],
    # Banks Island.
    [
        (-130.18, 53.98), (-130.08, 53.96), (-129.98, 53.89),
        (-129.91, 53.78), (-129.88, 53.64), (-129.90, 53.48),
        (-129.96, 53.35), (-130.05, 53.25), (-130.15, 53.22),
        (-130.21, 53.30), (-130.23, 53.43), (-130.20, 53.57),
        (-130.23, 53.72), (-130.22, 53.87),
    ],
    # Pitt Island, clipped by the southern edge.
    [
        (-130.55, 53.49), (-130.45, 53.46), (-130.37, 53.38),
        (-130.31, 53.27), (-130.28, 53.12), (-130.30, 53.00),
        (-130.63, 53.00), (-130.65, 53.14), (-130.62, 53.30),
        (-130.59, 53.42),
    ],
    # Dundas Island and adjacent small islands.
    [
        (-130.83, 54.74), (-130.77, 54.72), (-130.72, 54.67),
        (-130.73, 54.61), (-130.79, 54.58), (-130.85, 54.61),
        (-130.87, 54.68),
    ],
]


def contains(lon, lat, polygon):
    """Point-in-polygon test by ray crossing."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > lat) != (yj > lat):
            crossing = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing:
                inside = not inside
        j = i
    return inside


def build_matrix():
    latitudes = np.linspace(LAT_NORTH, LAT_SOUTH, N)
    longitudes = np.linspace(LON_WEST, LON_EAST, N)
    result = np.zeros((N, N), dtype=np.uint8)
    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            result[row, col] = any(
                contains(float(lon), float(lat), polygon)
                for polygon in POLYGONS
            )
    return result


def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()
    with (out / "candidate.csv").open("w", encoding="ascii", newline="") as f:
        for row in matrix:
            f.write(",".join(str(int(value)) for value in row) + "\n")
    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(out / "candidate.png")
    image.resize((1016, 1016), Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )


if __name__ == "__main__":
    main()
