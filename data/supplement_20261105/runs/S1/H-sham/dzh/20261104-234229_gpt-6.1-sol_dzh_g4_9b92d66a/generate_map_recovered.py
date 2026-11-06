#!/usr/bin/env python3
"""Knowledge-based approximate land/water raster for the Mediterranean region."""

from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image


LON_MIN = -9.0
LON_MAX = 37.0
LAT_MAX = 45.0
LAT_MIN = 31.0
STEP = 0.2
ROWS = 71
COLS = 231

# Hand-drawn coastline approximations. Coordinates are (longitude, latitude).
LAND_POLYGONS = [
    # Iberian Peninsula, with the northern coast clipped by the sampling frame.
    [
        (-9.0, 45.0), (3.5, 45.0), (3.5, 42.4), (2.3, 42.4),
        (1.4, 42.5), (0.5, 42.7), (-0.3, 43.0), (-1.5, 43.3),
        (-3.0, 43.4), (-4.5, 43.4), (-5.7, 43.5), (-7.0, 43.7),
        (-8.2, 43.6), (-9.0, 43.8),
    ],
    [
        (-9.0, 43.8), (-8.2, 43.6), (-7.0, 43.7), (-5.7, 43.5),
        (-4.5, 43.4), (-3.0, 43.4), (-1.5, 43.3), (-0.3, 43.0),
        (0.5, 42.7), (1.4, 42.5), (2.3, 42.4), (3.2, 42.4),
        (3.3, 41.8), (2.5, 41.3), (1.5, 41.1), (0.7, 40.7),
        (0.1, 40.6), (0.0, 39.5), (-0.3, 38.8), (-0.7, 38.0),
        (-1.0, 37.4), (-2.0, 36.8), (-4.3, 36.7), (-5.6, 36.0),
        (-6.3, 36.5), (-7.4, 37.1), (-7.4, 38.2), (-9.0, 38.7),
    ],
    # Mainland Europe north of the Mediterranean coast.
    [
        (3.0, 45.0), (37.0, 45.0), (37.0, 42.5), (35.0, 42.0),
        (33.0, 41.7), (31.0, 41.1), (29.0, 41.2), (27.0, 41.5),
        (26.0, 40.5), (25.0, 40.0), (24.0, 39.5), (23.5, 38.5),
        (23.0, 37.8), (22.0, 37.0), (21.0, 36.5), (20.0, 36.4),
        (19.5, 37.0), (19.0, 39.0), (18.5, 40.0), (17.5, 40.5),
        (16.5, 41.0), (16.0, 41.8), (17.0, 42.5), (18.0, 43.0),
        (17.0, 43.5), (16.0, 44.0), (15.0, 44.5), (14.0, 45.0),
        (12.5, 45.0), (11.5, 44.8), (10.5, 44.5), (9.5, 44.0),
        (8.0, 43.7), (7.5, 43.8), (7.0, 44.2), (6.5, 44.5),
        (5.0, 44.8), (3.0, 45.0),
    ],
    # Italian peninsula.
    [
        (7.4, 43.8), (8.0, 44.0), (9.0, 44.2), (10.0, 44.3),
        (11.0, 44.5), (12.0, 45.0), (12.8, 45.0), (13.0, 44.0),
        (13.2, 43.0), (13.6, 42.0), (14.0, 41.5), (15.0, 41.0),
        (16.0, 40.7), (17.2, 40.5), (18.5, 40.0), (18.2, 39.5),
        (17.2, 39.0), (16.8, 38.5), (15.8, 38.0), (15.5, 37.2),
        (15.0, 37.0), (14.7, 38.0), (14.0, 39.0), (13.5, 40.0),
        (13.0, 41.0), (12.5, 41.5), (12.0, 42.0), (11.5, 42.5),
        (10.5, 43.0), (9.5, 43.4), (8.5, 43.6), (7.4, 43.8),
    ],
    # North African coast and land south of it.
    [
        (-9.0, 31.0), (-9.0, 35.0), (-7.0, 35.8), (-5.6, 35.9),
        (-4.5, 35.2), (-2.2, 35.1), (-1.8, 35.0), (0.0, 35.6),
        (1.5, 36.0), (3.0, 36.8), (5.0, 37.1), (7.0, 37.1),
        (9.0, 37.1), (10.0, 36.8), (10.8, 36.5), (11.5, 37.2),
        (12.0, 37.0), (13.0, 37.1), (15.0, 37.5), (16.5, 37.4),
        (18.0, 37.0), (20.0, 32.9), (23.0, 32.9), (25.0, 31.3),
        (27.0, 31.0), (37.0, 31.0), (37.0, 45.0), (37.0, 31.0),
    ],
    # Anatolia and the southern shore of Turkey.
    [
        (26.0, 40.5), (27.0, 40.0), (28.0, 40.2), (29.0, 40.0),
        (30.0, 39.8), (31.0, 40.0), (32.0, 39.8), (33.0, 39.7),
        (34.0,  canary := 36.0), (36.0, 36.0), (36.0, 35.0),
        (34.0, 36.0), (32.0, 36.0), (30.0, 36.2), (28.0, 36.0),
        (27.0, 36.5), (26.0, 37.0), (26.0, 40.5),
    ],
    # Corsica is handled separately; this polygon approximates the Balkan mainland.
    [
        (13.0, 45.0), (16.0, 45.0), (16.5, 44.5), (17.5, 44.2),
        (18.5, 43.5), (18.5, 42.5), (19.0, 41.5), (20.0, 40.5),
        (20.5, 39.5), (21.0, 38.5), (22.0, 37.5), (23.0, 37.8),
        (24.0, 39.5), (25.0, 40.0), (26.0, 40.5), (27.0, 41.5),
        (29.0, 41.2), (31.0, 41.1), (33.0, 41.7), (35.0, 42.0),
        (37.0, 42.5), (37.0, 45.0),
    ],
    # Greece and its southern mainland, with the Aegean represented by the coast.
    [
        (19.0, 40.0), (20.0, 39.8), (21.0, 39.5), (22.0, 39.0),
        (23.0, 38.5), (24.0, 38.0), (24.5, 37.5), (24.0, 37.0),
        (23.0, 36.5), (22.0, 36.4), (21.0, 36.5), (20.0, 36.4),
        (19.5, 37.0), (19.0, 38.0), (19.0, 40.0),
    ],
]

# (center longitude, center latitude, east-west radius, north-south radius)
ISLAND_ELLIPSES = [
    (-1.45, 39.65, 0.42, 0.30),  # Ibiza
    (2.65, 39.65, 0.75, 0.48),   # Mallorca
    (3.00, 40.00, 0.28, 0.18),   # Menorca
    (8.75, 42.15, 0.60, 1.25),   # Corsica
    (9.00, 40.00, 0.72, 1.35),   # Sardinia
    (14.20, 37.50, 1.55, 0.75),  # Sicily
    (14.35, 35.90, 0.16, 0.10),  # Malta
    (25.15, 35.20, 1.25, 0.40),  # Crete
    (33.00, 35.05, 1.00, 0.55),  # Cyprus
    (26.30, 39.20, 0.22, 0.16),  # Lesbos
    (25.10, 36.40, 0.24, 0.12),  # Santorini
    (24.70, 37.05, 0.22, 0.15),  # Serifos / nearby Cyclades
    (26.95, 37.75, 0.30, 0.18),  # Samos
    (27.15, 36.80, 0.24, 0.14),  # Kos
    (28.20, 36.15, 0.34, 0.14),  # Rhodes
    (23.70, 35.50, 0.22, 0.10),  # Gavdos
    (19.90, 39.60, 0.45, 0.24),  # Corfu
    (20.70, 38.60, 0.35, 0.18),  # Kefalonia
    (21.70, 37.80, 0.38, 0.18),  # Euboea
    (20.70, 36.80, 0.32, 0.16),  # Peloponnese islands / coast
    (12.30, 41.25, 0.24, 0.18),  # Ischia / Capri vicinity
    (10.20, 42.80, 0.22, 0.18),  # Elba
]


def point_in_polygon(lon, lat, polygon):
    """Ray-casting point test for a hand-drawn polygon."""
    inside = False
    x1, y1 = polygon[-1]
    for x2, y2 in polygon:
        if (y1 > lat) != (y2 > lat):
            crossing_x = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
            if lon < crossing_x:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def is_land(lon, lat):
    if any(point_in_polygon(lon, lat, polygon) for polygon in LAND_POLYGONS):
        return True

    # Elliptical approximations retain the major Mediterranean islands.
    for center_lon, center_lat, radius_lon, radius_lat in ISLAND_ELLIPSES:
        dx = (lon - center_lon) / radius_lon
        dy = (lat - center_lat) / radius_lat
        if dx * dx + dy * dy <= 1.0:
            return True

    return False


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    longitudes = [LON_MIN + i * STEP for i in range(COLS)]
    latitudes = [LAT_MAX - i * STEP for i in range(ROWS)]

    matrix = np.zeros((ROWS, COLS), dtype=np.uint8)
    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            matrix[row, col] = 1 if is_land(lon, lat) else 0

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # 1 is white land, 0 is black water.
    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(output_dir / "candidate.png")

    enlarged = image.resize(
        (COLS * 8, ROWS * 8),
        resample=Image.Resampling.NEAREST,
    )
    enlarged.save(output_dir / "candidate_8x.png")

    # Verify the exported matrix and raster dimensions/colors.
    with csv_path.open("r", newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == ROWS
    assert all(len(row) == COLS for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert sum(len(row) for row in rows) == 16401

    original = Image.open(output_dir / "candidate.png")
    zoomed = Image.open(output_dir / "candidate_8x.png")
    assert original.size == (231, 71)
    assert zoomed.size == (1848, 568)
    assert set(original.getdata()) <= {(0, 0, 0), (255, 255, 255)}

    # Confirm the enlarged raster is exactly an 8x nearest-neighbor expansion.
    expected = original.resize((1848, 568), resample=Image.Resampling.NEAREST)
    assert list(expected.getdata()) == list(zoomed.getdata())

    print("Generated and verified:")
    print(f"  {csv_path}: {ROWS} rows x {COLS} columns")
    print(f"  outputs/candidate.png: {original.size[0]} x {original.size[1]}")
    print(f"  outputs/candidate_8x.png: {zoomed.size[0]} x {zoomed.size[1]}")


if __name__ == "__main__":
    main()
