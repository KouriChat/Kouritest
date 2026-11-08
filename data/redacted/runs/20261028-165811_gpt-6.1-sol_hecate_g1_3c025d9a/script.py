import csv
from pathlib import Path

import numpy as np
from PIL import Image

# Each polygon is an approximate land outline, specified as (longitude, latitude).
# Coordinates and coastline details are hand-authored estimates, not map data.
MAINLAND = [
    (-131.00, 55.00), (-130.86, 54.94), (-130.78, 54.84),
    (-130.70, 54.77), (-130.64, 54.68), (-130.54, 54.61),
    (-130.47, 54.50), (-130.41, 54.39), (-130.34, 54.28),
    (-130.24, 54.20), (-130.16, 54.08), (-130.08, 53.98),
    (-130.00, 53.86), (-129.93, 53.74), (-129.85, 53.62),
    (-129.77, 53.51), (-129.69, 53.38), (-129.60, 53.25),
    (-129.52, 53.12), (-129.44, 53.00), (-129.00, 53.00),
    (-129.00, 55.00),
]

GRAHAM_ISLAND = [
    (-131.00, 54.93), (-130.88, 54.88), (-130.81, 54.82),
    (-130.76, 54.74), (-130.70, 54.68), (-130.65, 54.60),
    (-130.60, 54.51), (-130.55, 54.42), (-130.49, 54.35),
    (-130.45, 54.27), (-130.40, 54.21), (-130.35, 54.16),
    (-130.30, 54.13), (-130.24, 54.13), (-130.20, 54.17),
    (-130.17, 54.22), (-130.17, 54.29), (-130.19, 54.36),
    (-130.22, 54.43), (-130.26, 54.50), (-130.31, 54.57),
    (-130.35, 54.65), (-130.40, 54.72), (-130.46, 54.79),
    (-130.53, 54.84), (-130.63, 54.89), (-130.76, 54.94),
]

MOR回SBY_ISLAND = [
    (-130.22, 53.88), (-130.16, 53.86), (-130.09, 53.87),
    (-130.03, 53.91), (-129.99, 53.97), (-129.97, 54.04),
    (-129.99, 54.11), (-130.04, 54.17), (-130.11, 54.20),
    (-130.18, 54.18), (-130.23, 54.13), (-130.26, 54.06),
    (-130.26, 53.98), (-130.24, 53.92),
]

# Small hand-estimated island groups and reefs in the strait.
SMALL_ISLANDS = [
    [(-130.63, 54.29), (-130.60, 54.27), (-130.56, 54.29),
     (-130.57, 54.33), (-130.61, 54.34)],
    [(-130.50, 54.17), (-130.47, 54.15), (-130.43, 54.17),
     (-130.44, 54.21), (-130.49, 54.21)],
    [(-130.35, 54.02), (-130.31, 54.00), (-130.27, 54.02),
     (-130.28, 54.06), (-130.33, 54.06)],
    [(-130.83, 54.60), (-130.80, 54.58), (-130.77, 54.60),
     (-130.78, 54.63), (-130.82, 54.63)],
    [(-130.14, 53.69), (-130.10, 53.67), (-130.06, 53.69),
     (-130.07, 53.73), (-130.12, 53.73)],
]

def inside_polygon(lon, lat, polygon):
    """Ray-crossing point-in-polygon test."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > lat) != (yj > lat):
            crossing_lon = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing_lon:
                inside = not inside
        j = i
    return inside

def is_land(lon, lat):
    return any(
        inside_polygon(lon, lat, polygon)
        for polygon in (MAINLAND, GRAHAM_ISLAND, MOR回SBY_ISLAND, *SMALL_ISLANDS)
    )

def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)

    # Integer indexing avoids cumulative floating-point drift.
    lats = [55.0 - i * (2.0 / 126.0) for i in range(127)]
    lons = [-131.0 + j * (2.0 / 126.0) for j in range(127)]
    matrix = np.array(
        [[1 if is_land(lon, lat) else 0 for lon in lons] for lat in lats],
        dtype=np.uint8,
    )

    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # 0 maps to black water; 1 maps to white land.
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(out / "candidate.png")
    image.resize((1016, 1016), resample=Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )

if __name__ == "__main__":
    main()
