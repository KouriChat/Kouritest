from pathlib import Path
import csv

import numpy as np
from PIL import Image

ROWS = 127
COLS = 127
OUTPUT_DIR = Path("outputs")

# Handwritten approximate outlines in WGS84 decimal degrees (lon, lat).
MAINLAND = [
    (-129.00, 55.00), (-129.63, 55.00), (-129.72, 54.91),
    (-129.91, 54.84), (-129.78, 54.75), (-129.95, 54.64),
    (-129.81, 54.55), (-130.05, 54.46), (-129.91, 54.35),
    (-129.70, 54.28), (-129.84, 54.19), (-129.63, 54.10),
    (-129.78, 54.00), (-129.59, 53.90), (-129.75, 53.80),
    (-129.58, 53.70), (-129.70, 53.59), (-129.54, 53.48),
    (-129.70, 53.37), (-129.48, 53.27), (-129.58, 53.16),
    (-129.41, 53.05), (-129.00, 53.00),
]

ISLANDS = [
    # Banks Island
    [
        (-129.98, 53.63), (-129.78, 53.60), (-129.70, 53.48),
        (-129.73, 53.29), (-129.82, 53.16), (-129.99, 53.14),
        (-130.02, 53.35), (-130.08, 53.50),
    ],
    # Pitt Island
    [
        (-129.77, 53.36), (-129.61, 53.34), (-129.55, 53.20),
        (-129.62, 53.10), (-129.78, 53.12), (-129.84, 53.24),
    ],
    [
        (-130.12, 54.88), (-130.01, 54.84), (-129.99, 54.72),
        (-130.08, 54.69), (-130.16, 54.78),
    ],
    [
        (-130.23, 54.60), (-130.08, 54.57), (-130.05, 54.43),
        (-130.17, 54.39), (-130.25, 54.49),
    ],
    [
        (-130.18, 54.28), (-130.04, 54.25), (-130.00, 54.13),
        (-130.10, 54.09), (-130.20, 54.17),
    ],
    [
        (-130.30, 53.95), (-130.15, 53.92), (-130.11, 53.80),
        (-130.23, 53.75), (-130.33, 53.84),
    ],
    [
        (-130.30, 53.61), (-130.16, 53.57), (-130.13, 53.46),
        (-130.25, 53.41), (-130.34, 53.50),
    ],
]
ISLANDS.extend([
    # Approximate eastern edge of Haida Gwaii within the sampled box.
    [
        (-131.00, 54.22), (-130.985, 54.19), (-130.985, 53.82),
        (-131.00, 53.79),
    ],
    [
        (-131.00, 53.54), (-130.985, 53.50), (-130.985, 53.06),
        (-131.00, 53.02),
    ],
])


def point_in_polygon(lon, lat, polygon):
    """Ray-casting point test for a polygon of (lon, lat) vertices."""
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
    return int(
        point_in_polygon(lon, lat, MAINLAND)
        or any(point_in_polygon(lon, lat, polygon) for polygon in ISLANDS)
    )
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    latitudes = np.linspace(55.0, 53.0, ROWS)
    longitudes = np.linspace(-131.0, -129.0, COLS)
    matrix = np.array(
        [[is_land(lon, lat) for lon in longitudes] for lat in latitudes],
        dtype=np.uint8,
    )
    assert matrix.shape == (127, 127)
    assert int(matrix.sum() + (matrix == 0).sum()) == 16129

    csv_path = OUTPUT_DIR / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

    image = Image.fromarray(matrix * 255, mode="L")
    image.save(OUTPUT_DIR / "candidate.png")
    enlarged = image.resize((1016, 1016), resample=Image.Resampling.NEAREST)
    enlarged.save(OUTPUT_DIR / "candidate_8x.png")

    with csv_path.open("r", newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 127
    assert all(len(row) == 127 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert image.size == (127, 127)
    assert set(image.getdata()) <= {0, 255}
    assert enlarged.size == (1016, 1016)
    assert set(enlarged.getdata()) <= {0, 255}
    print("Generated and validated the matrix and PNG files.")


if __name__ == "__main__":
    main()
