#!/usr/bin/env python3
"""Approximate Mediterranean land/water grid from hand-written polygons."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image


# Coordinates are (longitude, latitude), in WGS84 decimal degrees.
# These hand-written outlines approximate the principal continental coasts.
LAND_POLYGONS = [
    # Iberian Peninsula
    [
        (-9.6, 36.9), (-9.4, 38.0), (-9.2, 39.0), (-9.5, 40.0),
        (-9.1, 41.0), (-8.9, 41.8), (-8.2, 42.1), (-7.0, 43.0),
        (-6.0, 43.6), (-4.5, 43.5), (-3.3, 43.4), (-2.2, 43.4),
        (-1.8, 43.0), (-1.8, 42.5), (-1.4, 43.0), (0.0, 42.7),
        (1.5, 42.5), (3.2, 42.4), (3.3, 41.8), (2.8, 41.3),
        (2.0, 41.0), (1.5, 40.7), (0.5, 40.5), (0.1, 40.0),
        (0.0, 39.5), (-0.3, 39.0), (-0.7, 38.7), (-0.5, 38.0),
        (-0.2, 37.6), (-1.0, 37.2), (-2.0, 36.8), (-3.0, 36.7),
        (-4.4, 36.7), (-5.3, 36.0), (-6.0, 36.0), (-7.0, 37.0),
        (-7.4, 37.2), (-7.5, 37.8), (-7.4, 38.5), (-7.0, 39.0),
        (-7.0, 40.0), (-8.0, 41.0), (-8.8, 41.8), (-9.5, 42.0),
        (-9.5, 43.0), (-9.2, 43.8), (-9.0, 44.5), (-9.6, 45.0),
        (-10.0, 45.0), (-10.0, 35.5), (-5.6, 35.5),
        (-5.6, 36.0), (-5.3, 36.1), (-5.6, 36.2),
    ],
    # Continental Europe: Mediterranean shore from the Pyrenees eastward,
    # followed by an intentionally simplified northern closure.
    [
        (1.5, 42.5), (2.5, 42.4), (3.2, 42.4), (3.3, 43.0),
        (4.0, 43.3), (4.8, 43.4), (5.0, 43.2), (5.5, 43.1),
        (6.0, 43.0), (6.5, 43.1), (7.0, 43.5), (7.6, 43.8),
        (8.0, 44.0), (8.5, 44.1), (9.0, 44.1), (9.5, 44.0),
        (10.0, 43.8), (10.5, 43.7), (11.0, 43.5), (11.5, 43.0),
        (12.0, 42.5), (12.5, 42.0), (13.0, 41.5), (13.5, 41.0),
        (14.0, 40.5), (14.5, 40.0), (15.0, 39.8), (15.5, 39.8),
        (16.0, 39.9), (16.5, 40.0), (17.0, 40.2), (17.5, 40.4),
        (18.0, 40.5), (18.5, 40.6), (19.0, 40.6), (19.5, 40.5),
        (20.0, 40.3), (20.5, 40.1), (21.0, 40.0), (21.5, 40.1),
        (22.0, 40.3), (22.5, 40.5), (23.0, 40.7), (23.5, 40.8),
        (24.0, 40.8), (24.5, 40.7), (25.0, 40.5), (25.5, 40.4),
        (26.0, 40.3), (26.3, 40.1), (26.5, 40.0), (26.4, 40.5),
        (27.0, 41.0), (28.0, 41.1), (29.0, 41.2), (30.0, 41.2),
        (31.0, 41.3), (32.0, 41.4), (33.0, 41.5), (34.0, 41.5),
        (35.0, 41.5), (36.0, 41.4), (37.0, 41.4),
        (40.0, 46.0), (40.0, 50.0), (1.5, 50.0),
    ],
    # Anatolia and the southern Turkish coast
    [
        (26.4, 40.0), (26.8, 39.5), (27.0, 39.0), (27.5, 38.5),
        (28.0, 38.0), (28.5, 37.8), (29.0, 37.5), (29.5, 36.8),
        (30.0, 36.5), (31.0, 36.2), (32.0, 36.1), (33.0, 36.0),
        (34.0, 36.0), (35.0, 36.0), (36.0, 36.2), (37.0, 36.5),
        (37.0, 45.0), (26.0, 45.0),
    ],
    # North Africa, with the Mediterranean coastline as the northern edge
    [
        (-10.0, 31.0), (34.5, 31.0), (34.3, 31.3), (33.5, 31.4),
        (32.5, 31.5), (31.5, 31.6), (30.5, 31.6), (29.5, 31.5),
        (28.5, 31.4), (27.5, 31.4), (26.5, 31.5), (25.5, 31.6),
        (24.5, 31.7), (23.5, 31.8), (22.5, 32.0), (21.5, 32.1),
        (20.5, 32.2), (19.5, 32.5), (18.5, 32.8), (17.5, 32.9),
        (16.5, 32.8), (15.5, 32.7), (14.5, 32.6), (13.5, 32.5),
        (12.5, 32.4), (11.5, 32.3), (10.5, 32.2), (9.5, 32.1),
        (8.5, 33.0), (7.5, 33.8), (6.5, 34.0), (5.5, 34.2),
        (4.5, 34.5), (3.5, 35.0), (2.5, 35.1), (1.5, 35.2),
        (0.5, 35.3), (-0.5, 35.4), (-1.5, 35.5), (-2.5, 35.6),
        (-3.5, 35.7), (-4.5, 35.8), (-5.3, 35.9), (-5.6, 36.0),
        (-6.0, 35.5), (-10.0, 35.5),
    ],
    # Major islands large enough to affect this grid
    [(-1.0, 38.6), (-0.8, 39.0), (0.0, 39.9), (1.0, 39.9),
     (2.0, 39.7), (3.2, 39.8), (3.5, 39.4), (3.0, 38.9),
     (2.0, 38.7), (1.0, 38.6), (0.0, 38.6)],  # Balearics
    [(8.4, 41.4), (8.6, 42.0), (9.3, 42.8), (9.5, 43.0),
     (9.5, 42.0), (9.4, 41.3), (9.0, 41.0), (8.5, 41.0)],  # Corsica
    [(8.1, 39.2), (8.3, 40.0), (8.3, 40.8), (8.7, 41.2),
     (9.0, 41.0), (9.7, 40.5), (9.8, 39.5), (9.5, 38.9),
     (9.0, 38.8), (8.5, 38.9)],  # Sardinia
    [(12.4, 37.9), (12.5, 38.2), (13.0, 38.3), (13.5, 38.2),
     (15.2, 38.3), (15.6, 38.0), (15.2, 37.5), (15.0, 37.0),
     (15.2, 36.7), (14.8, 36.7), (13.5, 37.0), (12.7, 37.4)],  # Sicily
    [(23.5, 35.0), (24.0, 35.2), (25.0, 35.4), (26.0, 35.4),
     (26.6, 35.0), (26.2, 34.8), (25.0, 34.7), (24.0, 34.8)],  # Crete
    [(32.2, 34.6), (32.3, 35.0), (33.0, 35.4), (34.0, 35.5),
     (34.8, 35.2), (34.6, 34.6), (33.8, 34.5), (32.8, 34.5)],  # Cyprus
    # Corsica/Sardinia, Sicily, and other islands are represented separately;
    # this coarse outline captures the larger Ionian and Aegean land.
    [(19.3, 39.7), (19.5, 40.0), (20.0, 39.8), (20.2, 39.4),
     (20.0, 39.0), (19.5, 39.2)],  # Corfu
    [(25.0, 36.0), (25.2, 36.5), (25.8, 36.8), (26.2, 36.6),
     (26.0, 36.1), (25.5, 35.9)],  # Rhodes
]


def point_in_polygon(lon, lat, polygon):
    """Ray-casting point test; boundary points are treated as land."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # Treat points on a segment as inside, with a small numeric tolerance.
        cross = (lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
        if abs(cross) < 1e-10:
            if (min(x1, x2) - 1e-10 <= lon <= max(x1, x2) + 1e-10
                    and min(y1, y2) - 1e-10 <= lat <= max(y1, y2) + 1e-10):
                return True

        if (y1 > lat) != (y2 > lat):
            crossing_lon = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < crossing_lon:
                inside = not inside
    return inside


def is_land(lon, lat):
    return any(point_in_polygon(lon, lat, polygon) for polygon in LAND_POLYGONS)


def main():
    latitudes = 45.0 - 0.2 * np.arange(71)
    longitudes = -9.0 + 0.2 * np.arange(231)

    matrix = np.zeros((71, 231), dtype=np.uint8)
    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            matrix[row, col] = int(is_land(float(lon), float(lat)))

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Matrix value 1 is white; value 0 is black.
    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(output_dir / "candidate.png")

    enlarged = image.resize((231 * 8, 71 * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")

    # Verify matrix, CSV, palette, dimensions, and exact nearest-neighbor scaling.
    assert matrix.shape == (71, 231)
    assert matrix.size == 16401
    assert set(np.unique(matrix)).issubset({0, 1})

    with (output_dir / "candidate.csv").open("r", newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 71
    assert all(len(row) == 231 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    original = Image.open(output_dir / "candidate.png").convert("RGB")
    zoomed = Image.open(output_dir / "candidate_8x.png").convert("RGB")
    assert original.size == (231, 71)
    assert set(original.getdata()).issubset({(0, 0, 0), (255, 255, 255)})
    assert zoomed.size == (1848, 568)
    assert zoomed == original.resize((1848, 568), resample=Image.Resampling.NEAREST)

    print("Generated and verified 71x231 matrix and three output files.")


if __name__ == "__main__":
    main()
