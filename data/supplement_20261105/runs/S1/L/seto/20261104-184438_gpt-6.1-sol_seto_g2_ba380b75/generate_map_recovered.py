from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image


# ------------------------------------------------------------
# Coordinate grid
# ------------------------------------------------------------

LAT_COUNT = 64
LON_COUNT = 253

LAT_NORTH = 35.0
LAT_SOUTH = 33.0
LON_WEST = 129.0
LON_EAST = 137.0

LAT_STEP = 2.0 / 63.0
LON_STEP = 8.0 / 252.0


# ------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------

def point_in_polygon(x, y, polygon):
    """Ray-casting point-in-polygon test."""
    inside = False
    n = len(polygon)

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi) + xi
        )

        if intersects:
            inside = not inside

        j = i

    return inside


def point_in_ellipse(lon, lat, center_lon, center_lat, radius_lon, radius_lat):
    dx = (lon - center_lon) / radius_lon
    dy = (lat - center_lat) / radius_lat
    return dx * dx + dy * dy <= 1.0


def in_any_polygon(lon, lat, polygons):
    return any(point_in_polygon(lon, lat, polygon) for polygon in polygons)


def in_any_ellipse(lon, lat, ellipses):
    return any(
        point_in_ellipse(lon, lat, cx, cy, rx, ry)
        for cx, cy, rx, ry in ellipses
    )


# ------------------------------------------------------------
# Hand-written approximate land polygons
#
# Coordinates are approximate WGS84 longitude/latitude pairs.
# These shapes are deliberately compact and are intended to
# reconstruct the principal land/water structure of the region.
# ------------------------------------------------------------

# Western Honshu and Chugoku
HONSHU_WEST = [
    (130.75, 34.18),
    (130.82, 34.45),
    (131.05, 34.72),
    (131.15, 35.00),
    (137.00, 35.00),
    (137.00, 34.02),
    (136.72, 34.02),
    (136.45, 34.10),
    (136.18, 34.22),
    (135.90, 34.31),
    (135.62, 34.38),
    (135.35, 34.40),
    (135.10, 34.39),
    (134.85, 34.35),
    (134.58, 34.30),
    (134.35, 34.28),
    (134.08, 34.30),
    (133.82, 34.34),
    (133.55, 34.37),
    (133.28, 34.36),
    (133.02, 34.33),
    (132.77, 34.29),
    (132.52, 34.25),
    (132.28, 34.22),
    (132.02, 34.18),
    (131.78, 34.12),
    (131.55, 34.08),
    (131.30, 34.08),
    (131.05, 34.10),
    (130.85, 34.13),
]

# Kii Peninsula and the mainland around Ise Bay
KII_PENINSULA = [
    (134.78, 34.45),
    (135.03, 34.42),
    (135.28, 34.35),
    (135.52, 34.20),
    (135.68, 34.00),
    (135.72, 33.78),
    (135.63, 33.55),
    (135.48, 33.38),
    (135.25, 33.28),
    (135.02, 33.30),
    (134.85, 33.43),
    (134.76, 33.62),
    (134.68, 33.82),
    (134.58, 34.00),
    (134.48, 34.18),
    (134.52, 34.32),
]

# Northern and central Shikoku
SHIKOKU = [
    (132.00, 34.42),
    (132.30, 34.43),
    (132.62, 34.40),
    (132.92, 34.35),
    (133.20, 34.32),
    (133.50, 34.28),
    (133.78, 34.20),
    (134.00, 34.08),
    (134.18, 33.92),
    (134.25, 33.72),
    (134.12, 33.52),
    (133.88, 33.43),
    (133.58, 33.36),
    (133.25, 33.35),
    (132.98, 33.39),
    (132.70, 33.47),
    (132.45, 33.60),
    (132.22, 33.77),
    (132.05, 33.97),
    (131.90, 34.18),
]

# Kyushu
KYUSHU = [
    (129.45, 33.85),
    (129.58, 34.05),
    (129.86, 34.20),
    (130.20, 34.28),
    (130.52, 34.25),
    (130.78, 34.10),
    (130.92, 33.88),
    (131.02, 33.62),
    (131.00, 33.38),
    (130.88, 33.18),
    (130.62, 33.05),
    (130.28, 33.00),
    (129.98, 33.05),
    (129.72, 33.20),
    (129.55, 33.43),
    (129.43, 33.65),
]

# Northern Kyushu extension toward the Korea Strait
NORTH_KYUSHU = [
    (129.00, 34.02),
    (129.25, 34.20),
    (129.55, 34.36),
    (129.90, 34.42),
    (130.25, 34.40),
    (130.48, 34.28),
    (130.25, 34.10),
    (129.90, 34.00),
    (129.55, 33.92),
    (129.25, 33.88),
]

# Ise Bay eastern mainland / Chubu coast
ISE_EAST = [
    (136.15, 35.00),
    (137.00, 35.00),
    (137.00, 34.04),
    (136.78, 34.00),
    (136.58, 34.05),
    (136.42, 34.18),
    (136.28, 34.36),
    (136.16, 34.58),
    (136.08, 34.78),
]

LAND_POLYGONS = [
    HONSHU_WEST,
    KII_PENINSULA,
    SHIKOKU,
    KYUSHU,
    NORTH_KYUSHU,
    ISE_EAST,
]


# ------------------------------------------------------------
# Approximate islands, island groups, shoals and reefs.
# Each item is (longitude, latitude, longitude_radius, latitude_radius).
# ------------------------------------------------------------

ISLANDS = [
    # Suo-nada / western Seto Inland Sea
    (130.55, 33.98, 0.06, 0.035),  # Ainoshima area
    (130.72, 34.05, 0.07, 0.045),
    (130.90, 34.08, 0.065, 0.040),
    (131.18, 34.13, 0.075, 0.050),
    (131.42, 34.16, 0.070, 0.045),
    (131.62, 34.18, 0.055, 0.040),

    # Geiyo Islands
    (132.20, 34.12, 0.085, 0.055),
    (132.38, 34.10, 0.070, 0.050),
    (132.54, 34.05, 0.075, 0.055),
    (132.72, 34.02, 0.070, 0.050),
    (132.88, 34.00, 0.080, 0.060),
    (133.06, 34.02, 0.065, 0.045),
    (133.22, 34.08, 0.075, 0.050),
    (133.40, 34.12, 0.070, 0.050),
    (133.58, 34.15, 0.060, 0.045),
    (133.76, 34.18, 0.065, 0.045),
    (133.95, 34.19, 0.070, 0.050),

    # Hiroshima / Kure / Matsuyama vicinity
    (132.22, 34.27, 0.075, 0.050),
    (132.45, 34.29, 0.065, 0.045),
    (132.68, 34.29, 0.075, 0.050),
    (132.90, 34.26, 0.060, 0.045),
    (133.12, 34.25, 0.070, 0.050),
    (133.35, 34.24, 0.065, 0.045),
    (133.62, 34.24, 0.055, 0.040),

    # Iyo-nada and Bungo Channel
    (132.05, 33.82, 0.075, 0.050),
    (132.25, 33.76, 0.070, 0.050),
    (132.48, 33.74, 0.060, 0.045),
    (132.70, 33.72, 0.070, 0.050),
    (132.90, 33.70, 0.055, 0.040),
    (133.15, 33.68, 0.065, 0.045),

    # Kagawa / northern Shikoku and Harima
    (134.05, 34.28, 0.065, 0.045),
    (134.25, 34.30, 0.055, 0.040),
    (134.45, 34.32, 0.065, 0.045),
    (134.62, 34.34, 0.055, 0.040),
    (134.78, 34.37, 0.050, 0.035),

    # Awaji Island
    (134.82, 34.42, 0.20, 0.10),
    (134.92, 34.25, 0.10, 0.18),

    # Osaka Bay islands and small coastal islands
    (135.02, 34.45, 0.045, 0.030),
    (135.18, 34.48, 0.040, 0.025),
    (135.34, 34.52, 0.035, 0.025),
    (135.48, 34.56, 0.035, 0.025),

    # Ise Bay and Atsumi peninsula vicinity
    (136.68, 34.55, 0.060, 0.040),
    (136.82, 34.62, 0.050, 0.035),
    (136.98, 34.68, 0.045, 0.030),
    (137.00, 34.43, 0.040, 0.030),

    # Islands near Kyushu and Tsushima Strait
    (129.20, 34.38, 0.090, 0.055),
    (129.38, 34.48, 0.075, 0.050),
    (129.62, 34.48, 0.065, 0.045),
    (129.84, 34.46, 0.050, 0.035),
    (130.05, 34.46, 0.055, 0.035),

    # Northern Kyushu coastal islands
    (130.35, 34.12, 0.065, 0.045),
    (130.55, 34.18, 0.060, 0.040),
    (130.78, 34.25, 0.050, 0.035),
]


def is_land(lon, lat):
    # Main hand-written coast polygons.
    if in_any_polygon(lon, lat, LAND_POLYGONS):
        return True

    # Separate islands and island groups.
    if in_any_ellipse(lon, lat, ISLANDS):
        return True

    return False


# ------------------------------------------------------------
# Generate matrix
# ------------------------------------------------------------

matrix = np.zeros((LAT_COUNT, LON_COUNT), dtype=np.uint8)

for row in range(LAT_COUNT):
    lat = LAT_NORTH - row * LAT_STEP

    for col in range(LON_COUNT):
        lon = LON_WEST + col * LON_STEP
        matrix[row, col] = 1 if is_land(lon, lat) else 0


# ------------------------------------------------------------
# Validate matrix
# ------------------------------------------------------------

assert matrix.shape == (64, 253)
assert matrix.size == 16192
assert set(np.unique(matrix)).issubset({0, 1})


# ------------------------------------------------------------
# Write outputs
# ------------------------------------------------------------

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

csv_path = output_dir / "candidate.csv"
png_path = output_dir / "candidate.png"
png_8x_path = output_dir / "candidate_8x.png"

with csv_path.open("w", newline="", encoding="ascii") as f:
    writer = csv.writer(f, lineterminator="\n")
    for row in matrix:
        writer.writerow([int(value) for value in row])

# Black = water, white = land.
image_array = matrix * 255
image = Image.fromarray(image_array, mode="L")
assert image.size == (253, 64)

image.save(png_path)

# Explicit nearest-neighbor enlargement.
image_8x = image.resize((253 * 8, 64 * 8), resample=Image.Resampling.NEAREST)
assert image_8x.size == (2024, 512)
image_8x.save(png_8x_path)


# ------------------------------------------------------------
# Final output validation
# ------------------------------------------------------------

with csv_path.open("r", newline="", encoding="ascii") as f:
    rows = list(csv.reader(f))

assert len(rows) == 64
assert all(len(row) == 253 for row in rows)
assert all(value in {"0", "1"} for row in rows for value in row)

png_check = Image.open(png_path).convert("L")
png_8x_check = Image.open(png_8x_path).convert("L")

assert png_check.size == (253, 64)
assert png_8x_check.size == (2024, 512)
assert set(png_check.getdata()).issubset({0, 255})
assert set(png_8x_check.getdata()).issubset({0, 255})

print("Generation completed.")
print(f"Matrix: {matrix.shape[0]} rows x {matrix.shape[1]} columns")
print(f"Valid values: {matrix.size}")
print(f"Land points: {int(matrix.sum())}")
print(f"Water points: {int(matrix.size - matrix.sum())}")
print(f"CSV: {csv_path}")
print(f"PNG: {png_path} ({image.size[0]}x{image.size[1]})")
print(f"8x PNG: {png_8x_path} ({image_8x.size[0]}x{image_8x.size[1]})")
