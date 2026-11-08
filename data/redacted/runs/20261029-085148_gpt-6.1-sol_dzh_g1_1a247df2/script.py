#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os

import numpy as np
from PIL import Image


# Sampling grid:
# latitude: 45.0, 44.8, ..., 31.0
# longitude: -9.0, -8.8, ..., 37.0
LAT_MAX = 45.0
LAT_MIN = 31.0
LON_MIN = -9.0
LON_MAX = 37.0
STEP = 0.2

N_ROWS = 71
N_COLS = 231


# These are hand-written approximate polygons in (longitude, latitude).
# They are knowledge-based approximations, not downloaded geographic data.
LAND_POLYGONS = []


def add_polygon(points):
    LAND_POLYGONS.append(tuple((float(x), float(y)) for x, y in points))


def add_box(lon1, lat1, lon2, lat2):
    add_polygon([
        (lon1, lat1),
        (lon2, lat1),
        (lon2, lat2),
        (lon1, lat2),
    ])


def point_on_segment(px, py, ax, ay, bx, by, eps=1e-10):
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > eps:
        return False

    return (
        min(ax, bx) - eps <= px <= max(ax, bx) + eps
        and min(ay, by) - eps <= py <= max(ay, by) + eps
    )


def point_in_polygon(lon, lat, polygon):
    inside = False
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        if point_on_segment(lon, lat, x1, y1, x2, y2):
            return True

        crosses = ((y1 > lat) != (y2 > lat))
        if crosses:
            intersection_lon = (
                (x2 - x1) * (lat - y1) / (y2 - y1) + x1
            )
            if lon < intersection_lon:
                inside = not inside

    return inside


# ----------------------------------------------------------------------
# Mainland Europe west of the Adriatic
# ----------------------------------------------------------------------

# Iberian Peninsula
add_polygon([
    (-9.0, 36.9),
    (-7.6, 37.0),
    (-6.3, 36.7),
    (-5.7, 36.0),
    (-4.7, 36.1),
    (-3.4, 36.5),
    (-2.0, 36.8),
    (-1.0, 37.3),
    (0.0, 38.0),
    (1.0, 39.0),
    (2.0, 40.0),
    (3.0, 41.0),
    (3.3, 42.4),
    (3.3, 45.0),
    (-9.0, 45.0),
])

# Southern France
add_polygon([
    (3.3, 42.4),
    (4.0, 42.5),
    (4.5, 43.0),
    (5.0, 43.3),
    (5.8, 43.3),
    (6.4, 43.4),
    (7.0, 43.6),
    (7.6, 43.8),
    (7.8, 44.2),
    (8.0, 44.5),
    (8.0, 45.0),
    (3.3, 45.0),
])

# Italian mainland
add_polygon([
    (7.5, 44.2),
    (8.2, 44.0),
    (9.0, 43.8),
    (9.7, 43.7),
    (10.4, 43.5),
    (10.9, 42.8),
    (11.1, 42.2),
    (11.5, 41.5),
    (12.3, 41.0),
    (12.6, 40.2),
    (13.0, 39.0),
    (14.0, 38.2),
    (15.4, 37.8),
    (16.2, 38.0),
    (16.0, 38.8),
    (16.5, 39.8),
    (17.3, 40.3),
    (18.0, 40.5),
    (18.4, 40.1),
    (18.3, 39.7),
    (18.2, 40.7),
    (18.0, 41.2),
    (17.5, 41.5),
    (16.8, 41.8),
    (16.0, 42.1),
    (15.2, 42.5),
    (14.5, 43.2),
    (13.8, 44.2),
    (13.5, 45.0),
    (7.5, 45.0),
])


# ----------------------------------------------------------------------
# Balkans and European mainland around the Black Sea
# ----------------------------------------------------------------------

add_polygon([
    (13.5, 45.0),
    (14.0, 44.8),
    (15.0, 44.7),
    (16.0, 44.2),
    (17.0, 43.2),
    (18.0, 42.4),
    (19.0, 41.7),
    (20.0, 41.0),
    (20.5, 40.5),
    (21.0, 39.8),
    (21.8, 39.2),
    (22.5, 38.8),
    (23.5, 38.5),
    (24.5, 39.0),
    (25.0, 40.0),
    (26.0, 40.5),
    (27.0, 41.0),
    (28.0, 41.5),
    (28.0, 42.5),
    (28.5, 43.5),
    (29.0, 44.0),
    (30.0, 44.5),
    (30.0, 45.0),
    (13.5, 45.0),
])


# ----------------------------------------------------------------------
# Anatolia and Turkey
# ----------------------------------------------------------------------

add_polygon([
    (26.0, 40.0),
    (26.4, 39.7),
    (27.0, 39.2),
    (27.5, 38.5),
    (27.3, 37.5),
    (27.8, 36.8),
    (28.5, 36.6),
    (29.5, 36.5),
    (30.5, 36.2),
    (31.5, 36.0),
    (32.5, 36.1),
    (33.5, 36.1),
    (34.5, 36.2),
    (35.5, 36.0),
    (36.5, 36.2),
    (37.0, 36.5),
    (37.0, 41.2),
    (36.3, 41.5),
    (35.5, 41.7),
    (34.5, 41.8),
    (33.5, 41.9),
    (32.5, 41.9),
    (31.5, 41.8),
    (30.5, 41.8),
    (29.5, 41.5),
    (28.5, 41.3),
    (28.0, 41.5),
    (27.5, 41.2),
    (26.8, 40.8),
])


# Levant
add_polygon([
    (35.0, 36.2),
    (35.6, 35.5),
    (35.8, 34.7),
    (35.5, 34.0),
    (35.4, 33.2),
    (35.0, 32.5),
    (34.6, 31.5),
    (34.5, 31.0),
    (37.0, 31.0),
    (37.0, 37.0),
])


# ----------------------------------------------------------------------
# North Africa
# ----------------------------------------------------------------------

add_polygon([
    (-9.0, 35.3),
    (-8.6, 35.5),
    (-7.5, 35.8),
    (-6.3, 35.8),
    (-5.5, 35.2),
    (-4.5, 35.1),
    (-3.5, 35.0),
    (-2.0, 35.0),
    (-1.0, 35.2),
    (0.0, 35.7),
    (1.0, 36.1),
    (2.0, 36.7),
    (3.0, 36.8),
    (4.0, 36.8),
    (5.0, 37.0),
    (6.0, 37.1),
    (7.0, 37.0),
    (8.0, 36.9),
    (9.0, 37.0),
    (10.0, 37.0),
    (10.8, 36.7),
    (11.2, 37.0),
    (11.3, 36.5),
    (10.8, 35.7),
    (10.5, 35.0),
    (10.5, 34.3),
    (11.0, 33.5),
    (11.5, 33.2),
    (12.5, 32.8),
    (13.5, 32.8),
    (15.0, 32.9),
    (16.0, 32.9),
    (17.0, 32.9),
    (18.0, 32.8),
    (19.0, 32.5),
    (20.0, 32.5),
    (21.0, 32.6),
    (22.0, 32.9),
    (23.0, 32.9),
    (24.0, 31.8),
    (25.0, 31.2),
    (26.0, 31.3),
    (27.0, 31.5),
    (28.0, 31.4),
    (29.0, 31.3),
    (30.0, 31.2),
    (31.0, 31.2),
    (32.0, 31.3),
    (33.0, 31.5),
    (34.0, 31.3),
    (35.0, 31.4),
    (36.0, 31.5),
    (37.0, 31.5),
    (37.0, 31.0),
    (-9.0, 31.0),
])


# ----------------------------------------------------------------------
# Islands
# ----------------------------------------------------------------------

# Corsica
add_polygon([
    (8.5, 41.3),
    (9.0, 41.2),
    (9.5, 41.5),
    (9.7, 42.2),
    (9.5, 42.8),
    (9.2, 43.0),
    (8.8, 42.8),
    (8.5, 42.2),
])

# Sardinia
add_polygon([
    (8.0, 38.8),
    (8.5, 38.7),
    (9.2, 39.0),
    (9.7, 39.8),
    (9.7, 40.7),
    (9.3, 41.2),
    (8.7, 41.3),
    (8.2, 40.8),
    (8.0, 40.0),
])

# Sicily
add_polygon([
    (12.4, 37.1),
    (13.0, 36.7),
    (14.0, 36.6),
    (15.0, 36.7),
    (15.7, 37.2),
    (15.3, 38.0),
    (14.5, 38.2),
    (13.6, 38.1),
    (12.8, 37.7),
])

# Malta
add_box(14.1, 35.7, 14.6, 36.1)

# Balearic Islands
add_polygon([
    (2.2, 39.3),
    (2.7, 39.1),
    (3.3, 39.4),
    (3.5, 40.0),
    (3.1, 40.2),
    (2.5, 40.0),
])

add_box(3.8, 39.8, 4.4, 40.1)       # Menorca
add_box(1.1, 38.8, 1.7, 39.2)       # Ibiza
add_box(1.3, 38.6, 1.6, 38.8)       # Formentera

# Crete
add_polygon([
    (23.4, 35.5),
    (24.0, 35.2),
    (25.0, 35.1),
    (26.0, 35.2),
    (26.3, 35.5),
    (26.0, 35.8),
    (25.2, 35.9),
    (24.5, 35.8),
    (23.8, 35.7),
])

# Rhodes and nearby islands
add_box(27.6, 35.7, 28.3, 36.1)
add_box(27.0, 35.4, 27.3, 35.8)     # Karpathos
add_box(26.9, 36.7, 27.3, 37.1)      # Kos

# Euboea
add_polygon([
    (23.4, 38.3),
    (23.8, 38.2),
    (24.1, 38.7),
    (24.0, 39.2),
    (23.6, 39.4),
    (23.4, 39.0),
])

# Lesbos, Chios and Samos
add_box(26.0, 38.8, 26.7, 39.3)
add_box(25.8, 38.1, 26.3, 38.6)
add_box(26.5, 37.6, 27.1, 38.0)

# Lemnos
add_box(24.8, 39.7, 25.5, 40.1)

# Cyclades approximations
add_box(24.7, 36.9, 25.3, 37.5)
add_box(25.2, 37.2, 25.7, 37.7)
add_box(25.1, 36.1, 25.6, 36.6)
add_box(24.2, 37.2, 24.7, 37.6)

# Ionian islands
add_box(19.6, 39.5, 20.1, 40.0)     # Corfu
add_box(20.3, 38.0, 20.8, 38.5)      # Kefalonia
add_box(20.7, 37.6, 21.0, 38.0)      # Zakynthos

# Cyprus
add_polygon([
    (32.2, 34.6),
    (33.0, 34.5),
    (34.0, 34.6),
    (34.8, 35.0),
    (34.6, 35.5),
    (33.8, 35.7),
    (33.0, 35.5),
    (32.3, 35.2),
    (32.0, 34.9),
])


def is_land(lon, lat):
    for polygon in LAND_POLYGONS:
        if point_in_polygon(lon, lat, polygon):
            return 1
    return 0


def build_matrix():
    matrix = np.zeros((N_ROWS, N_COLS), dtype=np.uint8)

    for row in range(N_ROWS):
        lat = LAT_MAX - row * STEP
        for col in range(N_COLS):
            lon = LON_MIN + col * STEP
            matrix[row, col] = is_land(lon, lat)

    return matrix


def write_csv(matrix, path):
    with open(path, "w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for row in matrix:
            writer.writerow([int(value) for value in row])


def write_images(matrix, original_path, enlarged_path):
    # 0 -> black water, 1 -> white land
    image_array = (matrix * 255).astype(np.uint8)
    image = Image.fromarray(image_array, mode="L")
    image.save(original_path)

    resampling = getattr(Image, "Resampling", Image)
    enlarged = image.resize(
        (N_COLS * 8, N_ROWS * 8),
        resample=resampling.NEAREST,
    )
    enlarged.save(enlarged_path)


def verify_outputs(matrix, csv_path, original_png, enlarged_png):
    assert matrix.shape == (71, 231)
    assert matrix.size == 16401
    assert set(np.unique(matrix).tolist()).issubset({0, 1})

    with open(csv_path, "r", encoding="ascii", newline="") as handle:
        rows = list(csv.reader(handle))

    assert len(rows) == 71
    assert all(len(row) == 231 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert all(value != "" for row in rows for value in row)

    image = Image.open(original_png).convert("L")
    enlarged = Image.open(enlarged_png).convert("L")

    assert image.size == (231, 71)
    assert enlarged.size == (1848, 568)

    original_pixels = np.array(image)
    enlarged_pixels = np.array(enlarged)

    assert set(np.unique(original_pixels).tolist()).issubset({0, 255})
    assert set(np.unique(enlarged_pixels).tolist()).issubset({0, 255})

    # Verify exact nearest-neighbor 8x enlargement.
    expected = np.repeat(np.repeat(original_pixels, 8, axis=0), 8, axis=1)
    assert np.array_equal(expected, enlarged_pixels)


def main():
    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    matrix = build_matrix()

    csv_path = os.path.join(output_dir, "candidate.csv")
    original_png = os.path.join(output_dir, "candidate.png")
    enlarged_png = os.path.join(output_dir, "candidate_8x.png")

    write_csv(matrix, csv_path)
    write_images(matrix, original_png, enlarged_png)
    verify_outputs(matrix, csv_path, original_png, enlarged_png)

    print("Generated:")
    print(csv_path)
    print(original_png)
    print(enlarged_png)
    print("Matrix shape:", matrix.shape)
    print("Valid values:", int(matrix.size))
    print("Verification: passed")


if __name__ == "__main__":
    main()
