#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import numpy as np
from PIL import Image


ROWS = 64
COLS = 253
LAT_NORTH = 35.0
LAT_SOUTH = 33.0
LON_WEST = 129.0
LON_EAST = 137.0


# Hand-written approximate land polygons.
# Coordinates are (longitude, latitude).
LAND_POLYGONS = []


def add_polygon(points):
    LAND_POLYGONS.append([(float(x), float(y)) for x, y in points])


def add_ellipse(cx, cy, rx, ry, count=20):
    points = []
    for i in range(count):
        a = 2.0 * np.pi * i / count
        points.append((cx + rx * np.cos(a), cy + ry * np.sin(a)))
    add_polygon(points)


# Kyushu: western and eastern parts, with the Ariake/inner-bay gap
# approximated by separating the two main polygons.
add_polygon([
    (129.48, 33.00), (129.43, 33.22), (129.52, 33.50),
    (129.58, 33.76), (129.82, 33.95), (130.05, 34.03),
    (130.40, 34.04), (130.68, 33.90), (130.56, 33.66),
    (130.36, 33.44), (130.23, 33.20), (130.06, 33.00)
])

add_polygon([
    (130.30, 33.00), (130.42, 33.25), (130.56, 33.54),
    (130.67, 33.75), (130.82, 33.92), (131.04, 34.00),
    (131.28, 33.91), (131.42, 33.66), (131.53, 33.36),
    (131.63, 33.00)
])

# Tsushima, Iki, Hirado and nearby western islands.
add_polygon([
    (129.14, 34.05), (129.10, 34.27), (129.17, 34.48),
    (129.18, 34.72), (129.36, 34.78), (129.52, 34.58),
    (129.46, 34.37), (129.49, 34.13), (129.34, 34.02)
])
add_ellipse(129.75, 33.82, 0.12, 0.17, 18)
add_ellipse(129.43, 33.35, 0.10, 0.18, 18)
add_ellipse(129.20, 33.20, 0.09, 0.18, 16)
add_ellipse(129.13, 33.00, 0.07, 0.15, 16)

# Southern Honshu / Chugoku coastline.
# The polygon covers the land north of the hand-written Seto coast.
add_polygon([
    (130.65, 35.00), (137.00, 35.00), (137.00, 34.82),
    (136.76, 34.82), (136.56, 34.90), (136.31, 34.90),
    (136.10, 34.86), (135.90, 34.82), (135.69, 34.78),
    (135.46, 34.75), (135.20, 34.72), (134.95, 34.74),
    (134.70, 34.72), (134.46, 34.63), (134.20, 34.59),
    (133.95, 34.55), (133.70, 34.48), (133.45, 34.42),
    (133.25, 34.38), (133.05, 34.33), (132.85, 34.27),
    (132.68, 34.25), (132.55, 34.32), (132.43, 34.40),
    (132.30, 34.35), (132.20, 34.20), (132.00, 34.08),
    (131.75, 34.00), (131.50, 33.95), (131.25, 33.90),
    (131.00, 33.90), (130.80, 33.95), (130.65, 34.00)
])

# Eastern Honshu around Ise Bay and Aichi.
add_polygon([
    (136.42, 35.00), (137.00, 35.00), (137.00, 34.72),
    (136.88, 34.68), (136.66, 34.68), (136.48, 34.78)
])

# Kii Peninsula.
add_polygon([
    (135.08, 34.72), (135.35, 34.69), (135.55, 34.47),
    (135.66, 34.25), (135.70, 34.02), (135.58, 33.76),
    (135.42, 33.54), (135.15, 33.39), (134.83, 33.34),
    (134.57, 33.50), (134.46, 33.78), (134.54, 34.00),
    (134.64, 34.25), (134.80, 34.46), (134.98, 34.61)
])

# Shikoku.
add_polygon([
    (131.92, 33.34), (132.08, 33.62), (132.34, 33.80),
    (132.62, 33.87), (132.88, 34.00), (133.18, 34.10),
    (133.48, 34.11), (133.78, 34.28), (134.10, 34.34),
    (134.38, 34.30), (134.62, 34.18), (134.70, 34.04),
    (134.55, 33.78), (134.32, 33.55), (134.02, 33.33),
    (133.72, 33.10), (133.40, 33.00), (132.90, 33.00),
    (132.45, 33.05), (132.12, 33.17)
])

# Awaji Island.
add_polygon([
    (134.66, 34.13), (134.72, 34.42), (134.83, 34.63),
    (135.02, 34.62), (135.15, 34.39), (135.09, 34.14),
    (134.94, 34.00), (134.76, 34.02)
])

# Larger islands in the Seto Inland Sea and Geiyo Islands.
add_ellipse(132.32, 34.30, 0.09, 0.16, 20)  # Miyajima
add_ellipse(132.45, 34.27, 0.16, 0.17, 22)  # Etajima
add_ellipse(132.38, 34.18, 0.12, 0.13, 20)  # Nomi
add_ellipse(132.51, 34.13, 0.13, 0.13, 20)  # Kurahashi
add_ellipse(132.98, 34.14, 0.14, 0.18, 20)  # Oshima
add_ellipse(132.96, 34.22, 0.09, 0.10, 18)  # Hakatajima
add_ellipse(133.05, 34.27, 0.14, 0.15, 20)  # Omishima
add_ellipse(133.09, 34.31, 0.13, 0.16, 20)  # Ikuchijima
add_ellipse(133.18, 34.32, 0.13, 0.15, 20)  # Innoshima
add_ellipse(133.20, 34.39, 0.10, 0.13, 18)  # Mukaishima
add_ellipse(133.18, 34.24, 0.08, 0.10, 18)  # Yuge
add_ellipse(133.50, 34.35, 0.09, 0.10, 18)  # Kasaoka area
add_ellipse(133.72, 34.32, 0.08, 0.09, 18)  # Shiwaku area
add_ellipse(133.76, 34.36, 0.07, 0.09, 18)
add_ellipse(133.94, 34.47, 0.10, 0.11, 18)  # Naoshima
add_ellipse(134.03, 34.49, 0.07, 0.08, 18)  # Teshima
add_ellipse(134.10, 34.56, 0.06, 0.08, 18)  # Inujima
add_ellipse(134.05, 34.42, 0.06, 0.08, 18)  # Megijima
add_ellipse(134.00, 34.43, 0.06, 0.07, 18)  # Ogijima

# Additional islands around the western and central Seto Inland Sea.
add_ellipse(131.98, 33.76, 0.08, 0.10, 18)
add_ellipse(132.19, 33.92, 0.08, 0.10, 18)
add_ellipse(132.72, 34.02, 0.08, 0.10, 18)
add_ellipse(132.77, 34.15, 0.07, 0.09, 18)
add_ellipse(133.35, 34.25, 0.07, 0.08, 18)
add_ellipse(133.58, 34.19, 0.07, 0.08, 18)
add_ellipse(134.27, 34.24, 0.06, 0.08, 18)


def point_in_polygon(x, y, polygon):
    """Ray-crossing point-in-polygon test."""
    inside = False
    j = len(polygon) - 1

    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        crosses = ((yi > y) != (yj > y))
        if crosses:
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside

        j = i

    return inside


def is_land(lon, lat):
    # The point criterion is applied directly. There is no cell averaging.
    for polygon in LAND_POLYGONS:
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]

        if lon < min(xs) or lon > max(xs):
            continue
        if lat < min(ys) or lat > max(ys):
            continue

        if point_in_polygon(lon, lat, polygon):
            return 1

    return 0


def generate_matrix():
    # Use the exact requested endpoint-inclusive sampling interval.
    latitudes = np.linspace(LAT_NORTH, LAT_SOUTH, ROWS)
    longitudes = np.linspace(LON_WEST, LON_EAST, COLS)

    matrix = np.zeros((ROWS, COLS), dtype=np.uint8)

    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            matrix[row, col] = is_land(float(lon), float(lat))

    return matrix


def write_outputs(matrix):
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "candidate.csv")
    png_path = os.path.join(output_dir, "candidate.png")
    enlarged_path = os.path.join(output_dir, "candidate_8x.png")

    # CSV without a header.
    with open(csv_path, "w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        for row in matrix:
            writer.writerow([int(value) for value in row])

    # White is land, black is water.
    image_array = (matrix * 255).astype(np.uint8)
    image = Image.fromarray(image_array, mode="L")
    image.save(png_path)

    # Explicit nearest-neighbor enlargement.
    try:
        nearest = Image.Resampling.NEAREST
    except AttributeError:
        nearest = Image.NEAREST

    enlarged = image.resize((COLS * 8, ROWS * 8), resample=nearest)
    enlarged.save(enlarged_path)

    # Required validation.
    assert matrix.shape == (64, 253)
    assert matrix.size == 16192
    assert np.all((matrix == 0) | (matrix == 1))

    with open(csv_path, "r", encoding="ascii") as f:
        csv_rows = list(csv.reader(f))

    assert len(csv_rows) == 64
    assert all(len(row) == 253 for row in csv_rows)
    assert all(value in ("0", "1") for row in csv_rows for value in row)

    raw_pixels = set(np.asarray(image).reshape(-1).tolist())
    enlarged_pixels = set(np.asarray(enlarged).reshape(-1).tolist())

    assert raw_pixels.issubset({0, 255})
    assert enlarged_pixels.issubset({0, 255})
    assert image.size == (253, 64)
    assert enlarged.size == (2024, 512)

    # Verify that every enlarged 8x8 block equals its source pixel.
    raw = np.asarray(image)
    big = np.asarray(enlarged)
    for row in range(ROWS):
        for col in range(COLS):
            block = big[row * 8:(row + 1) * 8, col * 8:(col + 1) * 8]
            assert np.all(block == raw[row, col])

    print("Generated:")
    print(" ", csv_path, "64 rows x 253 columns")
    print(" ", png_path, "253 x 64")
    print(" ", enlarged_path, "2024 x 512")
    print("Validation passed: 16192 binary point classifications.")


if __name__ == "__main__":
    matrix = generate_matrix()
    write_outputs(matrix)
