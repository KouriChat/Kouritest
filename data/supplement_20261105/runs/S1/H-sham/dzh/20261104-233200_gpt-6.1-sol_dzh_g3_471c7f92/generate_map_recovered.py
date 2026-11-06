#!/usr/bin/env python3
"""Generate a hand-drawn approximate land/water raster for the Mediterranean."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image


# Vertices are (longitude, latitude), in WGS84 decimal degrees.
# These are deliberately approximate, manually encoded geographic outlines.
EUROPE = [
    (-5.6, 36.0), (-6.2, 36.8), (-7.4, 37.0), (-8.9, 38.5),
    (-9.5, 41.0), (-9.5, 45.5), (37.0, 45.5), (37.0, 41.7),
    (36.0, 41.5), (35.0, 41.4), (33.5, 41.5), (31.5, 41.7),
    (29.5, 41.3), (28.9, 41.0), (28.0, 41.2), (27.0, 41.4),
    (26.0, 41.7), (25.3, 41.8), (24.3, 41.7), (23.5, 41.6),
    (22.5, 41.5), (21.5, 41.4), (20.5, 41.3), (19.5, 41.3),
    (18.5, 41.5), (17.5, 42.0), (16.5, 42.5), (16.0, 43.0),
    (17.0, 43.5), (18.5, 43.6), (19.5, 43.2), (19.2, 42.5),
    (18.5, 42.0), (18.0, 41.5), (17.5, 40.8), (18.0, 40.0),
    (18.5, 39.8), (19.0, 40.0), (19.5, 40.4), (20.0, 40.6),
    (20.5, 40.8), (21.0, 40.8), (21.5, 40.6), (22.0, 40.5),
    (22.5, 40.7), (23.0, 40.8), (23.5, 40.7), (24.0, 40.5),
    (24.5, 40.2), (25.0, 39.8), (25.5, 39.5), (26.0, 39.3),
    (26.5, 39.0), (26.8, 38.5), (26.5, 38.0), (26.0, 37.7),
    (25.5, 37.5), (25.0, 37.3), (24.5, 37.0), (24.0, 36.8),
    (23.5, 36.7), (23.0, 36.7), (22.5, 36.8), (22.0, 37.0),
    (21.5, 37.2), (21.0, 37.4), (20.5, 37.5), (20.0, 37.6),
    (19.5, 37.7), (19.0, 38.0), (18.5, 38.2), (18.0, 38.3),
    (17.5, 38.0), (17.0, 37.8), (16.5, 38.0), (16.0, 38.2),
    (15.5, 38.3), (15.0, 38.2), (14.5, 38.0), (14.0, 38.0),
    (13.5, 38.2), (13.0, 38.5), (12.5, 38.8), (12.0, 39.0),
    (11.5, 39.2), (11.0, 39.5), (10.5, 39.8), (10.0, 40.0),
    (9.5, 40.5), (9.0, 41.0), (8.5, 41.5), (8.0, 42.0),
    (7.5, 42.5), (7.0, 43.0), (6.5, 43.3), (6.0, 43.5),
    (5.5, 43.5), (5.0, 43.4), (4.5, 43.3), (4.0, 43.2),
    (3.5, 43.2), (3.0, 43.3), (2.5, 43.4), (2.0, 43.5),
    (1.5, 43.4), (1.0, 43.3), (0.5, 43.2), (0.0, 43.2),
    (-0.5, 43.3), (-1.0, 43.4), (-1.5, 43.5), (-2.0, 43.4),
    (-2.5, 43.3), (-3.0, 43.2), (-3.5, 43.0), (-4.0, 42.8),
    (-4.5, 42.5), (-5.0, 42.0), (-5.5, 41.5), (-6.0, 41.0),
    (-7.0, 41.0), (-7.5, 41.5), (-8.0, 42.0), (-8.5, 42.5),
    (-9.0, 43.0), (-9.5, 45.5),
]

# North Africa, including the Mediterranean coastal strip and the Sahara
# portion inside the requested latitude range.
NORTH_AFRICA = [
    (35.0, 31.0), (35.0, 31.5), (34.5, 31.8), (33.5, 31.7),
    (32.5, 31.5), (31.5, 31.4), (30.5, 31.4), (29.5, 31.3),
    (28.5, 31.3), (27.5, 31.3), (26.5, 31.4), (25.5, 31.5),
    (24.5, 31.7), (23.5, 32.0), (22.5, 32.5), (21.5, 32.8),
    (20.5, 32.9), (19.5, 32.9), (18.5, 32.8), (17.5, 32.7),
    (16.5, 32.6), (15.5, 32.5), (14.5, 32.4), (13.5, 32.3),
    (12.5, 32.2), (11.5, 32.0), (10.5, 31.8), (9.5, 31.5),
    (8.5, 31.4), (7.5, 31.3), (6.5, 31.2), (5.5, 31.1),
    (4.5, 31.0), (3.5, 31.0), (2.5, 31.0), (1.5, 31.0),
    (0.5, 31.0), (-0.5, 31.0), (-1.5, 31.0), (-2.5, 31.0),
    (-3.5, 31.0), (-4.5, 31.0), (-5.5, 31.0), (-6.5, 31.0),
    (-7.5, 31.0), (-8.5, 31.0), (-9.0, 31.0),
]

# Anatolia is represented separately to retain a broad approximate Aegean
# and eastern Mediterranean outline.
ANATOLIA = [
    (26.0, 40.5), (26.5, 40.8), (27.0, 41.0), (28.0, 41.2),
    (29.0, 41.2), (30.0, 41.3), (31.0, 41.4), (32.0, 41.5),
    (33.0, 41.4), (34.0, 41.3), (35.0, 41.2), (36.0, 41.1),
    (37.0, 41.0), (37.0, 36.0), (36.0, 36.0), (35.0, 36.0),
    (34.0, 36.0), (33.0, 36.0), (32.0, 36.0), (31.0, 36.0),
    (30.0, 36.0), (29.0, 36.0), (28.0, 36.2), (27.0, 36.5),
    (26.0, 36.7), (27.0, 37.0), (27.5, 37.5), (27.0, 38.0),
    (26.5, 38.5), (26.0, 39.0), (26.0, 40.5),
]

# Manually sketched islands large enough to matter at 0.2-degree sampling.
ISLANDS = [
    # Balearic Islands
    [(-1.0, 38.6), (0.2, 38.5), (1.5, 39.0), (2.0, 39.8),
     (1.0, 40.1), (0.0, 39.8), (-1.0, 39.2)],
    # Corsica
    [(8.5, 41.3), (9.1, 41.4), (9.5, 42.0), (9.4, 42.8),
     (9.0, 43.0), (8.6, 42.5)],
    # Sardinia
    [(8.1, 38.8), (8.2, 39.5), (8.4, 40.4), (9.0, 41.0),
     (9.6, 40.8), (9.7, 39.8), (9.5, 39.0), (9.0, 38.7)],
    # Sicily
    [(12.3, 36.6), (13.0, 36.6), (14.0, 37.0), (15.3, 37.2),
     (15.2, 38.2), (14.0, 38.2), (13.0, 38.1), (12.4, 37.5)],
    # Crete
    [(23.4, 34.8), (24.0, 34.9), (25.0, 35.0), (26.3, 35.2),
     (26.3, 35.7), (25.0, 35.6), (24.0, 35.5), (23.4, 35.3)],
    # Cyprus
    [(32.2, 34.5), (32.7, 34.6), (33.5, 34.7), (34.6, 35.0),
     (34.6, 35.5), (33.7, 35.7), (32.8, 35.5), (32.2, 35.0)],
    # Larger Aegean islands
    [(25.0, 39.0), (25.5, 39.1), (26.0, 39.5), (25.8, 40.0),
     (25.2, 39.8)],
    [(26.0, 38.2), (26.5, 38.2), (27.0, 38.6), (26.8, 39.0),
     (26.2, 38.8)],
    [(26.0, 36.0), (27.0, 36.0), (27.9, 36.2), (28.2, 36.7),
     (27.6, 36.9), (26.8, 36.7)],
    # Malta
    [(14.1, 35.7), (14.6, 35.7), (14.6, 36.1), (14.1, 36.1)],
    # Djerba
    [(10.5, 33.6), (11.0, 33.6), (11.2, 33.9), (10.8, 34.0)],
]

LAND_POLYGONS = [EUROPE, NORTH_AFRICA, ANATOLIA, *ISLANDS]


def point_in_polygon(lon, lat, polygon):
    """Ray-casting point-in-polygon test."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        crosses = (yi > lat) != (yj > lat)
        if crosses:
            crossing_lon = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing_lon:
                inside = not inside
        j = i
    return inside


def main():
    latitudes = 45.0 - 0.2 * np.arange(71)
    longitudes = -9.0 + 0.2 * np.arange(231)
    matrix = np.zeros((71, 231), dtype=np.uint8)

    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            matrix[row, col] = int(
                any(point_in_polygon(float(lon), float(lat), poly)
                    for poly in LAND_POLYGONS)
            )

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # 1 is white land; 0 is black water.
    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(output_dir / "candidate.png", format="PNG")

    enlarged = image.resize((231 * 8, 71 * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png", format="PNG")

    # Verify matrix, CSV, palette, dimensions, and exact nearest-neighbor scaling.
    assert matrix.shape == (71, 231)
    assert matrix.size == 16401
    assert set(np.unique(matrix).tolist()) <= {0, 1}

    with csv_path.open("r", newline="", encoding="ascii") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 71
    assert all(len(row) == 231 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    base = Image.open(output_dir / "candidate.png").convert("RGB")
    large = Image.open(output_dir / "candidate_8x.png").convert("RGB")
    assert base.size == (231, 71)
    assert large.size == (1848, 568)
    assert set(base.getdata()) <= {(0, 0, 0), (255, 255, 255)}
    assert large.resize((231, 71), resample=Image.Resampling.NEAREST).tobytes() == base.tobytes()

    print("Generated and verified 71x231 matrix (16401 values).")
    print("outputs/candidate.csv")
    print("outputs/candidate.png")
    print("outputs/candidate_8x.png")


if __name__ == "__main__":
    main()
