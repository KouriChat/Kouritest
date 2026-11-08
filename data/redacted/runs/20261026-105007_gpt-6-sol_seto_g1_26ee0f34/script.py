from pathlib import Path
import math

import numpy as np
from PIL import Image


ROWS = 64
COLS = 253
LON_MIN = 129.0
LON_MAX = 137.0
LAT_MAX = 35.0
LAT_MIN = 33.0

out_dir = Path("outputs")
out_dir.mkdir(parents=True, exist_ok=True)


def polygon_contains(x, y, polygon):
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            xinters = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < xinters:
                inside = not inside
        j = i
    return inside


def ellipse_polygon(cx, cy, rx, ry, count=12, phase=0.0):
    return [
        (
            cx + rx * math.cos(phase + 2.0 * math.pi * i / count),
            cy + ry * math.sin(phase + 2.0 * math.pi * i / count),
        )
        for i in range(count)
    ]


land_polygons = []
water_polygons = []


def land(points):
    land_polygons.append(points)


def water(points):
    water_polygons.append(points)

# Southern edge of western Honshu.
land([
    (130.78, 33.95), (131.15, 34.04), (131.55, 34.11),
    (131.95, 34.18), (132.35, 34.24), (132.70, 34.33),
    (133.05, 34.42), (133.40, 34.52), (133.78, 34.62),
    (134.15, 34.68), (134.50, 34.74), (134.80, 34.76),
    (135.05, 34.72), (135.23, 34.62), (135.24, 34.48),
    (135.15, 34.32), (135.18, 34.12), (135.30, 33.87),
    (135.48, 33.62), (135.70, 33.43), (135.95, 33.35),
    (136.15, 33.57), (136.34, 33.90), (136.48, 34.20),
    (136.62, 34.42), (136.78, 34.57), (137.00, 34.72),
    (137.00, 35.00), (130.78, 35.00),
])

# Northern Kyushu.
land([
    (129.65, 33.15), (129.72, 33.55), (129.78, 33.90),
    (130.00, 34.10), (130.30, 34.18), (130.55, 34.12),
    (130.80, 33.95), (131.10, 33.78), (131.35, 33.45),
    (131.50, 33.10), (131.55, 33.00), (129.90, 33.00),
])

# Shikoku.
land([
    (132.00, 33.35), (132.18, 33.55), (132.40, 33.70),
    (132.65, 33.82), (132.82, 34.00), (133.10, 34.13),
    (133.45, 34.18), (133.82, 34.22), (134.15, 34.34),
    (134.40, 34.36), (134.60, 34.20), (134.72, 34.02),
    (134.62, 33.75), (134.45, 33.55), (134.35, 33.20),
    (134.12, 33.08), (133.82, 33.02), (133.50, 33.12),
    (133.20, 33.15), (132.90, 33.05), (132.60, 33.02),
    (132.35, 33.10),
])
# Major islands around Kyushu.
land([
    (129.18, 34.10), (129.10, 34.35), (129.18, 34.65),
    (129.35, 34.75), (129.48, 34.60), (129.48, 34.35),
    (129.40, 34.10), (129.25, 34.00),
])
land([
    (129.65, 33.72), (129.68, 33.95), (129.90, 34.02),
    (130.02, 33.85), (129.95, 33.65), (129.75, 33.63),
])
land([
    (129.40, 33.20), (129.35, 33.45), (129.50, 33.65),
    (129.65, 33.55), (129.60, 33.30), (129.50, 33.15),
])
land(ellipse_polygon(129.08, 33.06, 0.09, 0.12, 10, 0.2))
land(ellipse_polygon(129.82, 33.16, 0.10, 0.12, 10, 0.1))
land(ellipse_polygon(130.02, 33.32, 0.08, 0.16, 11, 0.2))
land(ellipse_polygon(130.20, 33.08, 0.10, 0.12, 10, 0.0))
land(ellipse_polygon(130.32, 33.48, 0.08, 0.13, 10, 0.3))
land(ellipse_polygon(130.58, 33.25, 0.08, 0.14, 10, 0.1))

# Awaji Island.
land([
    (134.65, 34.12), (134.55, 34.35), (134.65, 34.62),
    (134.85, 34.68), (135.02, 34.50), (135.10, 34.25),
    (134.95, 34.08), (134.75, 34.08),
])

# Shodoshima and nearby Kagawa islands.
land([
    (133.65, 34.30), (133.75, 34.52), (134.05, 34.62),
    (134.30, 34.50), (134.42, 34.30), (134.22, 34.18),
    (133.90, 34.20),
])
# Small islands in the Seto Inland Sea.
land(ellipse_polygon(134.00, 34.45, 0.08, 0.10, 10, 0.1))
land(ellipse_polygon(134.08, 34.42, 0.07, 0.08, 10, 0.0))
land(ellipse_polygon(134.28, 34.25, 0.07, 0.08, 10, 0.2))
land(ellipse_polygon(134.40, 34.18, 0.08, 0.06, 10, 0.0))
land(ellipse_polygon(133.68, 34.12, 0.08, 0.06, 10, 0.0))
land(ellipse_polygon(132.28, 34.29, 0.07, 0.16, 12, 0.15))
land(ellipse_polygon(132.45, 34.27, 0.11, 0.15, 12, 0.0))
land(ellipse_polygon(132.58, 34.14, 0.10, 0.09, 10, 0.1))
land(ellipse_polygon(132.72, 34.20, 0.07, 0.09, 10, 0.0))
land(ellipse_polygon(132.25, 33.92, 0.15, 0.10, 12, 0.2))
land(ellipse_polygon(132.48, 33.90, 0.14, 0.11, 12, 0.0))
land(ellipse_polygon(132.92, 34.20, 0.10, 0.12, 12, 0.1))
land(ellipse_polygon(133.02, 34.26, 0.10, 0.15, 12, 0.0))
land(ellipse_polygon(133.14, 34.30, 0.10, 0.15, 12, 0.1))
land(ellipse_polygon(133.25, 34.23, 0.08, 0.10, 10, 0.0))
land(ellipse_polygon(133.15, 34.08, 0.08, 0.08, 10, 0.2))
land(ellipse_polygon(133.40, 34.30, 0.07, 0.08, 10, 0.0))
land(ellipse_polygon(133.72, 34.42, 0.09, 0.08, 10, 0.1))
land(ellipse_polygon(133.92, 34.45, 0.09, 0.08, 10, 0.0))
land(ellipse_polygon(134.18, 34.42, 0.07, 0.06, 10, 0.2))
land(ellipse_polygon(134.55, 34.18, 0.06, 0.06, 10, 0.0))
land(ellipse_polygon(134.68, 34.10, 0.06, 0.05, 10, 0.1))
land(ellipse_polygon(134.82, 34.18, 0.07, 0.06, 10, 0.0))
land(ellipse_polygon(134.95, 34.15, 0.06, 0.05, 10, 0.2))
land(ellipse_polygon(135.10, 34.16, 0.05, 0.05, 10, 0.0))
# Ise Bay opening and eastern shore corrections.
water([
    (136.66, 34.44), (136.75, 34.55), (136.84, 34.70),
    (136.88, 34.86), (136.84, 35.00), (137.00, 35.00),
    (137.00, 34.72), (136.95, 34.58), (136.82, 34.46),
])
land([
    (136.78, 34.75), (136.90, 34.70), (137.00, 34.85),
    (137.00, 35.00), (136.82, 35.00), (136.70, 34.92),
])
land(ellipse_polygon(136.78, 34.62, 0.05, 0.04, 10, 0.0))
land(ellipse_polygon(136.92, 34.64, 0.06, 0.04, 10, 0.2))
land(ellipse_polygon(136.98, 34.73, 0.05, 0.05, 10, 0.0))


longitudes = np.linspace(LON_MIN, LON_MAX, COLS)
latitudes = np.linspace(LAT_MAX, LAT_MIN, ROWS)
matrix = np.zeros((ROWS, COLS), dtype=np.uint8)

for row, lat in enumerate(latitudes):
    for col, lon in enumerate(longitudes):
        is_land = any(
            polygon_contains(lon, lat, polygon)
            for polygon in land_polygons
        )
        if is_land and any(
            polygon_contains(lon, lat, polygon)
            for polygon in water_polygons
        ):
            is_land = False
        matrix[row, col] = int(is_land)

assert matrix.shape == (64, 253)
assert matrix.size == 16192
assert np.all((matrix == 0) | (matrix == 1))

np.savetxt(
    out_dir / "candidate.csv",
    matrix,
    fmt="%d",
    delimiter=",",
)

image = Image.fromarray(matrix * 255, mode="L")
assert image.size == (253, 64)
image.save(out_dir / "candidate.png")

scaled = image.resize(
    (2024, 512),
    resample=Image.Resampling.NEAREST,
)
assert scaled.size == (2024, 512)
scaled.save(out_dir / "candidate_8x.png")

original_check = np.asarray(Image.open(out_dir / "candidate.png"))
scaled_check = np.asarray(Image.open(out_dir / "candidate_8x.png"))
assert set(np.unique(original_check)).issubset({0, 255})
assert set(np.unique(scaled_check)).issubset({0, 255})
assert np.array_equal(
    scaled_check,
    np.repeat(np.repeat(original_check, 8, axis=0), 8, axis=1),
)
