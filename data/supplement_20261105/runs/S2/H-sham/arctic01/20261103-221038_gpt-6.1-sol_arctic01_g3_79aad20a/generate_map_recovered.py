#!/usr/bin/env python3
"""Knowledge-based approximate land mask for the Canadian Arctic Archipelago."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image


# Hand-drawn approximate island outlines as (longitude, latitude) vertices.
# Polygons are deliberately approximate and are not sourced from map data.
LAND_POLYGONS = [
    # Eastern tip of Victoria Island, clipped by the western frame edge.
    [
        (-105.0, 73.10), (-104.0, 73.12), (-103.3, 73.55),
        (-103.0, 74.10), (-103.8, 74.60), (-104.5, 75.05),
        (-105.0, 75.15),
    ],
    # Prince of Wales Island.
    [
        (-99.85, 72.95), (-99.65, 73.35), (-99.30, 73.72),
        (-98.90, 74.25), (-98.35, 74.65), (-97.85, 74.52),
        (-97.35, 74.15), (-96.95, 73.72), (-96.75, 73.25),
        (-96.90, 72.95), (-98.00, 72.90),
    ],
    # Somerset Island.
    [
        (-94.80, 72.95), (-94.50, 73.25), (-94.25, 73.75),
        (-93.90, 74.15), (-93.45, 74.42), (-92.95, 74.36),
        (-92.55, 74.05), (-92.35, 73.58), (-92.10, 73.15),
        (-92.15, 72.95),
    ],
    # Melville Island: eastern edge within the frame.
    [
        (-106.00, 75.45), (-105.00, 75.35), (-104.80, 75.70),
        (-105.00, 76.15), (-105.00, 76.85), (-105.50, 77.20),
        (-106.00, 77.10),
    ],
    # Bathurst Island.
    [
        (-107.75, 75.35), (-107.35, 75.75), (-107.55, 76.15),
        (-107.15, 76.45), (-106.75, 76.95), (-106.10, 77.35),
        (-105.45, 77.28), (-105.05, 76.95), (-104.95, 76.42),
        (-105.10, 75.95), (-105.40, 75.55), (-106.10, 75.30),
        (-106.85, 75.20),
    ],
    # Cornwallis Island.
    [
        (-96.15, 74.55), (-95.75, 74.78), (-95.30, 74.85),
        (-94.85, 75.08), (-94.35, 75.15), (-93.95, 74.98),
        (-93.70, 74.72), (-93.90, 74.48), (-94.45, 74.40),
        (-95.20, 74.38), (-95.80, 74.42),
    ],
    # Devon Island.
    [
        (-90.25, 74.05), (-89.75, 74.30), (-89.10, 74.38),
        (-88.50, 74.50), (-87.85, 74.62), (-87.20, 74.70),
        (-86.60, 74.82), (-85.90, 74.93), (-85.15, 75.00),
        (-84.45, 75.10), (-83.70, 75.20), (-82.90, 75.30),
        (-82.20, 75.45), (-81.55, 75.65), (-81.05, 75.95),
        (-81.00, 76.35), (-81.70, 76.42), (-82.35, 76.25),
        (-83.05, 76.18), (-83.75, 76.20), (-84.30, 76.35),
        (-84.95, 76.42), (-85.55, 76.30), (-86.20, 76.05),
        (-86.85, 75.85), (-87.55, 75.70), (-88.15, 75.55),
        (-88.80, 75.35), (-89.40, 75.12), (-89.90, 74.78),
    ],
    # Southern Ellesmere Island, continuing out through the northern/eastern frame.
    [
        (-83.35, 76.28), (-82.95, 76.75), (-82.45, 77.20),
        (-82.15, 77.75), (-82.45, 78.15), (-83.00, 78.42),
        (-83.15, 78.95), (-82.75, 79.35), (-82.10, 79.58),
        (-81.00, 79.62), (-81.00, 73.00), (-81.60, 73.00),
        (-81.80, 74.10), (-82.15, 75.00), (-82.80, 75.55),
        (-83.40, 75.85),
    ],
    # Axel Heiberg Island.
    [
        (-92.15, 78.05), (-91.75, 78.45), (-91.55, 78.95),
        (-91.05, 79.35), (-90.45, 79.55), (-89.85, 79.35),
        (-89.35, 78.95), (-89.15, 78.50), (-89.45, 78.15),
        (-90.05, 77.92), (-90.85, 77.88), (-91.55, 77.95),
    ],
    # Ellef Ringnes Island.
    [
        (-103.45, 77.30), (-103.20, 77.75), (-103.35, 78.18),
        (-103.05, 78.68), (-102.55, 79.03), (-101.90, 79.15),
        (-101.25, 79.02), (-100.80, 78.68), (-100.45, 78.25),
        (-100.55, 77.85), (-101.10, 77.55), (-101.75, 77.35),
        (-102.55, 77.25),
    ],
    # Amund Ringnes Island.
    [
        (-99.75, 77.35), (-99.55, 77.75), (-99.70, 78.12),
        (-99.35, 78.52), (-98.85, 78.72), (-98.20, 78.60),
        (-97.70, 78.32), (-97.55, 77.95), (-97.80, 77.62),
        (-98.40, 77.42), (-99.05, 77.30),
    ],
    # Lougheed Island.
    [
        (-105.00, 78.45), (-104.75, 78.76), (-104.35, 78.92),
        (-103.95, 78.75), (-103.82, 78.43), (-104.15, 78.20),
        (-104.62, 78.22),
    ],
    # Small Sverdrup-area islands.
    [
        (-96.65, 78.50), (-96.40, 78.75), (-96.05, 78.82),
        (-95.80, 78.60), (-96.00, 78.38), (-96.35, 78.35),
    ],
    [
        (-94.05, 77.65), (-93.80, 77.90), (-93.45, 77.93),
        (-93.25, 77.70), (-93.50, 77.48), (-93.85, 77.48),
    ],
    # Small islands north of Bathurst/Cornwallis.
    [
        (-102.20, 76.95), (-101.95, 77.18), (-101.65, 77.12),
        (-101.55, 76.88), (-101.82, 76.75),
    ],
    [
        (-98.20, 76.10), (-97.95, 76.35), (-97.55, 76.30),
        (-97.42, 76.02), (-97.75, 75.86),
    ],
]


def point_in_polygon(lon, lat, polygon):
    """Return whether a point is inside a polygon using the ray-crossing rule."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        crosses = (yi > lat) != (yj > lat)
        if crosses:
            x_at_lat = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_at_lat:
                inside = not inside
        j = i
    return inside


def is_land(lon, lat):
    return any(point_in_polygon(lon, lat, polygon) for polygon in LAND_POLYGONS)


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Integer tenths avoid floating-point drift in the requested grid.
    longitudes = [-1050 + col for col in range(241)]
    latitudes = [790 - row for row in range(61)]

    matrix = np.zeros((61, 241), dtype=np.uint8)
    for row, lat10 in enumerate(latitudes):
        lat = lat10 / 10.0
        for col, lon10 in enumerate(longitudes):
            matrix[row, col] = int(is_land(lon10 / 10.0, lat))

    assert matrix.shape == (61, 241)
    assert matrix.size == 14701

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Land is white (255); water is black (0).
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(output_dir / "candidate.png")

    enlarged = image.resize((241 * 8, 61 * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")

    # Verify the exported artifacts and their required dimensions/values.
    with csv_path.open("r", newline="", encoding="ascii") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 61
    assert all(len(row) == 241 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    pixels = set(image.getdata())
    assert pixels <= {0, 255}
    assert image.size == (241, 61)
    assert enlarged.size == (1928, 488)
    assert set(enlarged.getdata()) <= {0, 255}

    print(f"Wrote {csv_path}, candidate.png, and candidate_8x.png")
    print(f"Matrix: {matrix.shape[0]} x {matrix.shape[1]} = {matrix.size} values")


if __name__ == "__main__":
    main()
