from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ROWS = 64
COLS = 253
LAT_NORTH = 35.0
LAT_SOUTH = 33.0
LON_WEST = 129.0
LON_EAST = 137.0


def ellipse(cx, cy, rx, ry, rotation=0.0, steps=32):
    points = []
    cr = math.cos(rotation)
    sr = math.sin(rotation)
    for i in range(steps):
        angle = 2.0 * math.pi * i / steps
        x = rx * math.cos(angle)
        y = ry * math.sin(angle)
        points.append((
            cx + x * cr - y * sr,
            cy + x * sr + y * cr,
        ))
    return points


def point_on_segment(x, y, ax, ay, bx, by, tolerance=1e-10):
    cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
    if abs(cross) > tolerance:
        return False
    return (
        min(ax, bx) - tolerance <= x <= max(ax, bx) + tolerance
        and min(ay, by) - tolerance <= y <= max(ay, by) + tolerance
    )


def point_in_polygon(x, y, polygon):
    inside = False
    count = len(polygon)

    for i in range(count):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % count]

        if point_on_segment(x, y, ax, ay, bx, by):
            return True

        crosses = (ay > y) != (by > y)
        if crosses:
            intersection_x = (bx - ax) * (y - ay) / (by - ay) + ax
            if x < intersection_x:
                inside = not inside

    return inside


# Main Kyushu landmass, approximated from remembered geography.
kyushu = [
    (129.42, 32.50),
    (129.42, 33.00),
    (129.50, 33.35),
    (129.56, 33.70),
    (129.72, 33.95),
    (130.00, 34.05),
    (130.38, 34.07),
    (130.72, 34.01),
    (131.05, 33.94),
    (131.34, 33.80),
    (131.60, 33.55),
    (131.78, 33.25),
    (131.97, 33.00),
    (132.08, 32.55),
    (132.10, 32.40),
    (129.42, 32.40),
]

# Mainland Honshu south coast and Chugoku region.
honshu = [
    (130.65, 35.20),
    (137.20, 35.20),
    (137.20, 34.78),
    (136.80, 34.72),
    (136.42, 34.68),
    (136.02, 34.70),
    (135.62, 34.75),
    (135.32, 34.73),
    (135.00, 34.72),
    (134.70, 34.72),
    (134.32, 34.68),
    (134.00, 34.63),
    (133.70, 34.62),
    (133.40, 34.55),
    (133.10, 34.48),
    (132.78, 34.42),
    (132.42, 34.37),
    (132.02, 34.32),
    (131.62, 34.25),
    (131.30, 34.18),
    (131.00, 34.05),
    (130.75, 33.95),
    (130.65, 35.20),
]

# Shikoku.
shikoku = [
    (131.98, 33.65),
    (132.23, 33.90),
    (132.58, 34.05),
    (132.98, 34.10),
    (133.38, 34.18),
    (133.78, 34.25),
    (134.15, 34.29),
    (134.54, 34.25),
    (134.75, 34.05),
    (134.68, 33.75),
    (134.50, 33.43),
    (134.25, 33.10),
    (133.90, 32.90),
    (133.45, 32.75),
    (133.00, 32.75),
    (132.60, 32.85),
    (132.22, 33.05),
    (132.00, 33.35),
]

# Kii Peninsula.
kii_peninsula = [
    (134.92, 34.34),
    (135.08, 34.42),
    (135.38, 34.39),
    (135.66, 34.31),
    (135.92, 34.16),
    (136.15, 33.92),
    (136.34, 33.65),
    (136.46, 33.36),
    (136.31, 33.10),
    (136.02, 32.96),
    (135.66, 32.90),
    (135.30, 32.91),
    (135.04, 33.06),
    (134.94, 33.36),
]

# Northern and eastern Mie land around Ise Bay.
mie_land = [
    (135.45, 34.64),
    (135.74, 34.71),
    (136.08, 34.75),
    (136.39, 34.72),
    (136.62, 34.62),
    (136.73, 34.50),
    (136.68, 34.30),
    (136.46, 34.20),
    (136.15, 34.19),
    (135.94, 34.34),
    (135.70, 34.40),
    (135.50, 34.50),
]

# Chita Peninsula and nearby Aichi land.
chita_peninsula = [
    (136.76, 35.20),
    (137.20, 35.20),
    (137.20, 34.54),
    (137.05, 34.48),
    (136.90, 34.56),
    (136.79, 34.72),
    (136.76, 35.20),
]

land_polygons = [
    kyushu,
    honshu,
    shikoku,
    kii_peninsula,
    mie_land,
    chita_peninsula,
]

# Named islands and island groups, represented by hand-written ellipses.
island_specs = [
    (129.70, 33.75, 0.18, 0.14, 0.10),
    (129.47, 33.36, 0.13, 0.20, -0.20),
    (129.82, 33.04, 0.08, 0.11, 0.20),
    (130.08, 33.05, 0.10, 0.12, -0.20),
    (130.25, 33.12, 0.08, 0.10, 0.10),
    (131.72, 33.73, 0.10, 0.16, 0.15),
    (132.06, 33.94, 0.14, 0.10, -0.20),
    (132.28, 33.91, 0.30, 0.13, 0.15),
    (132.55, 34.02, 0.10, 0.09, -0.20),
    (132.30, 34.29, 0.09, 0.18, 0.10),
    (132.46, 34.25, 0.22, 0.18, -0.25),
    (132.54, 34.08, 0.19, 0.13, 0.10),
    (132.78, 34.24, 0.13, 0.09, -0.20),
    (132.98, 34.24, 0.19, 0.12, 0.10),
    (133.09, 34.30, 0.13, 0.16, -0.30),
    (133.19, 34.34, 0.13, 0.12, 0.20),
    (133.22, 34.22, 0.10, 0.13, 0.20),
    (133.40, 34.35, 0.13, 0.08, 0.10),
    (133.55, 34.32, 0.13, 0.08, -0.20),
    (133.65, 34.13, 0.10, 0.08, 0.10),
    (133.82, 34.18, 0.09, 0.07, -0.20),
    (133.95, 34.46, 0.13, 0.09, 0.10),
    (134.03, 34.52, 0.12, 0.08, -0.20),
    (134.04, 34.42, 0.08, 0.10, 0.10),
    (134.11, 34.40, 0.06, 0.08, -0.20),
    (134.28, 34.53, 0.34, 0.20, 0.15),
    (134.46, 34.46, 0.10, 0.08, -0.20),
    (134.73, 34.40, 0.10, 0.08, 0.10),
    (134.91, 34.42, 0.08, 0.07, -0.20),
    (136.88, 34.54, 0.06, 0.06, 0.10),
]

for spec in island_specs:
    land_polygons.append(ellipse(*spec, steps=32))

# Awaji Island.
land_polygons.append([
    (134.64, 34.68),
    (134.87, 34.70),
    (135.06, 34.59),
    (135.15, 34.43),
    (135.09, 34.18),
    (134.96, 34.02),
    (134.77, 34.08),
    (134.64, 34.31),
    (134.60, 34.52),
])

# Hand-written water cuts for the principal indented bays.
water_polygons = [
    # Ariake Bay.
    [
        (129.74, 33.46),
        (130.02, 33.54),
        (130.38, 33.50),
        (130.58, 33.27),
        (130.60, 33.00),
        (130.43, 32.84),
        (130.04, 32.88),
        (129.78, 33.10),
        (129.68, 33.30),
    ],
    # Omura Bay.
    [
        (129.66, 33.36),
        (129.89, 33.43),
        (130.04, 33.30),
        (130.06, 33.06),
        (129.91, 32.94),
        (129.72, 33.02),
        (129.61, 33.20),
    ],
    # Beppu Bay.
    [
        (131.68, 33.52),
        (131.92, 33.55),
        (132.05, 33.40),
        (131.98, 33.17),
        (131.82, 33.08),
        (131.68, 33.22),
    ],
]


def is_land(lon, lat):
    land = any(point_in_polygon(lon, lat, polygon) for polygon in land_polygons)
    if not land:
        return False

    if any(point_in_polygon(lon, lat, polygon) for polygon in water_polygons):
        return False

    return True


matrix = np.zeros((ROWS, COLS), dtype=np.uint8)

for row in range(ROWS):
    lat = LAT_NORTH - row * (LAT_NORTH - LAT_SOUTH) / (ROWS - 1)
    for col in range(COLS):
        lon = LON_WEST + col * (LON_EAST - LON_WEST) / (COLS - 1)
        matrix[row, col] = 1 if is_land(lon, lat) else 0


assert matrix.shape == (64, 253)
assert matrix.size == 16192
assert set(np.unique(matrix)).issubset({0, 1})


csv_path = OUTPUT_DIR / "candidate.csv"
with csv_path.open("w", newline="", encoding="ascii") as handle:
    writer = csv.writer(handle, lineterminator="\n")
    for row in matrix:
        writer.writerow([str(int(value)) for value in row])


image_array = np.where(matrix == 1, 255, 0).astype(np.uint8)
image = Image.fromarray(image_array, mode="L")
image.save(OUTPUT_DIR / "candidate.png")

enlarged = image.resize((COLS * 8, ROWS * 8), resample=Image.NEAREST)
enlarged.save(OUTPUT_DIR / "candidate_8x.png")

assert image.size == (253, 64)
assert enlarged.size == (2024, 512)
assert set(image.getdata()).issubset({0, 255})
