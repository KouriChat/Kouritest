import csv
import math
import os
from PIL import Image

OUT_DIR = "outputs"
W, H = 253, 64
LON0, LAT0 = 129.0, 35.0
STEP = 2.0 / 63.0

os.makedirs(OUT_DIR, exist_ok=True)
land_polygons = []


def add_polygon(points):
    land_polygons.append([(float(x), float(y)) for x, y in points])


def add_ellipse(cx, cy, rx, ry, n=20):
    points = []
    for k in range(n):
        a = 2.0 * math.pi * k / n
        points.append((cx + rx * math.cos(a),
                      cy + ry * math.sin(a)))
    add_polygon(points)


def point_in_polygon(x, y, polygon):
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        cross = (x - xi) * (yj - yi) - (y - yi) * (xj - xi)
        if abs(cross) < 1e-12:
            if (min(xi, xj) - 1e-12 <= x <= max(xi, xj) + 1e-12
                    and min(yi, yj) - 1e-12 <= y <= max(yi, yj) + 1e-12):
                return True
        if (yi > y) != (yj > y):
            xinters = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < xinters:
                inside = not inside
        j = i
    return inside


# Mainland approximation: western Honshu
add_polygon([
    (130.05, 35.00), (134.75, 35.00), (134.75, 34.86),
    (134.55, 34.73), (134.20, 34.66), (133.80, 34.59),
    (133.40, 34.48), (133.00, 34.36), (132.60, 34.32),
    (132.20, 34.28), (131.80, 34.18), (131.30, 34.00),
    (130.92, 33.91), (130.65, 34.15), (130.45, 34.45),
    (130.25, 34.75), (130.05, 35.00)
])
# Mainland approximation: Kansai, Kii, Shikoku, and Kyushu
add_polygon([
    (134.70, 35.00), (137.00, 35.00), (137.00, 34.79),
    (136.80, 34.73), (136.55, 34.70), (136.30, 34.68),
    (136.00, 34.69), (135.70, 34.69), (135.40, 34.73),
    (135.10, 34.73), (134.85, 34.81), (134.70, 35.00)
])

add_polygon([
    (134.95, 34.40), (135.25, 34.45), (135.50, 34.35),
    (135.75, 34.20), (136.00, 34.05), (136.25, 33.90),
    (136.50, 33.65), (136.65, 33.35), (136.55, 33.00),
    (134.50, 33.00), (134.65, 33.30), (134.80, 33.65),
    (134.85, 34.00), (134.95, 34.40)
])

add_polygon([
    (131.95, 34.05), (132.30, 34.10), (132.70, 34.10),
    (133.15, 34.08), (133.60, 34.15), (134.00, 34.20),
    (134.45, 34.18), (134.85, 34.05), (134.95, 33.80),
    (134.80, 33.55), (134.55, 33.35), (134.20, 33.25),
    (133.80, 33.25), (133.40, 33.20), (132.95, 33.25),
    (132.55, 33.40), (132.20, 33.55), (131.95, 33.80),
    (131.95, 34.05)
])

add_polygon([
    (129.35, 34.25), (129.65, 34.35), (130.05, 34.32),
    (130.40, 34.20), (130.80, 34.10), (131.20, 33.90),
    (131.45, 33.55), (131.62, 33.00), (129.70, 33.00),
    (129.72, 33.25), (129.55, 33.50), (129.35, 33.80),
    (129.35, 34.25)
])

# Ise Bay peninsulas
add_polygon([
    (136.72, 35.00), (137.00, 35.00), (137.00, 34.72),
    (136.88, 34.58), (136.72, 34.48), (136.62, 34.68),
    (136.72, 35.00)
])

add_polygon([
    (136.96, 34.72), (137.00, 34.72), (137.00, 34.35),
    (136.82, 34.38), (136.67, 34.52), (136.80, 34.65),
    (136.96, 34.72)
])
# Kyushu and western Seto islands
add_polygon([
    (129.17, 34.63), (129.28, 34.72), (129.47, 34.68),
    (129.58, 34.48), (129.52, 34.22), (129.40, 34.08),
    (129.23, 34.16), (129.16, 34.40), (129.17, 34.63)
])
add_ellipse(129.72, 33.75, 0.14, 0.15, 18)
add_ellipse(129.43, 33.37, 0.10, 0.17, 16)
add_ellipse(129.56, 33.43, 0.06, 0.10, 14)
add_ellipse(131.68, 33.73, 0.06, 0.15, 14)

add_ellipse(131.82, 33.98, 0.11, 0.16, 16)
add_ellipse(132.05, 34.03, 0.10, 0.16, 16)
add_ellipse(132.30, 34.28, 0.10, 0.17, 18)
add_ellipse(132.43, 34.22, 0.18, 0.13, 18)
add_ellipse(132.46, 34.08, 0.14, 0.11, 16)
add_ellipse(132.54, 33.99, 0.13, 0.09, 16)
add_ellipse(132.66, 34.23, 0.12, 0.10, 16)
add_ellipse(132.78, 34.22, 0.11, 0.10, 16)
add_ellipse(132.86, 34.16, 0.17, 0.12, 18)
add_ellipse(132.91, 34.27, 0.14, 0.12, 18)
add_ellipse(133.02, 34.31, 0.17, 0.13, 18)
add_ellipse(133.06, 34.21, 0.12, 0.11, 16)
add_ellipse(133.15, 34.35, 0.18, 0.16, 18)
add_ellipse(133.19, 34.25, 0.10, 0.11, 16)
add_ellipse(133.27, 34.30, 0.08, 0.09, 14)
add_ellipse(133.30, 34.39, 0.12, 0.13, 16)
add_ellipse(133.39, 34.34, 0.08, 0.08, 14)

# Innoshima, Mukaishima, and nearby islands
add_ellipse(133.19, 34.37, 0.14, 0.16, 18)
add_ellipse(133.27, 34.40, 0.10, 0.12, 16)
add_ellipse(133.33, 34.41, 0.08, 0.10, 14)
add_ellipse(133.43, 34.43, 0.09, 0.09, 14)
# Central and eastern Seto islands
add_ellipse(133.78, 34.42, 0.09, 0.08, 14)
add_ellipse(133.92, 34.48, 0.16, 0.11, 18)
add_ellipse(134.02, 34.45, 0.10, 0.10, 16)
add_ellipse(134.10, 34.49, 0.08, 0.08, 14)
add_ellipse(134.20, 34.52, 0.08, 0.08, 14)
add_ellipse(134.31, 34.56, 0.09, 0.08, 14)
add_ellipse(134.43, 34.55, 0.08, 0.08, 14)

# Shodoshima
add_polygon([
    (134.02, 34.58), (134.18, 34.68), (134.40, 34.68),
    (134.55, 34.57), (134.55, 34.42), (134.43, 34.33),
    (134.20, 34.35), (134.05, 34.45), (134.02, 34.58)
])

# Awaji
add_polygon([
    (134.73, 34.62), (134.92, 34.70), (135.05, 34.58),
    (135.10, 34.38), (135.03, 34.20), (134.90, 34.05),
    (134.76, 34.10), (134.70, 34.32), (134.73, 34.62)
])

add_ellipse(134.67, 34.53, 0.06, 0.07, 14)
add_ellipse(134.60, 34.42, 0.05, 0.06, 14)
add_ellipse(134.58, 34.30, 0.04, 0.05, 14)
add_ellipse(133.55, 34.39, 0.08, 0.07, 14)
add_ellipse(133.72, 34.34, 0.08, 0.07, 14)
add_ellipse(134.00, 34.34, 0.09, 0.06, 14)
add_ellipse(134.14, 34.30, 0.06, 0.06, 14)
add_ellipse(134.39, 34.31, 0.07, 0.06, 14)
add_ellipse(134.62, 34.25, 0.06, 0.05, 14)

# Matrix generation
matrix = []
for row in range(H):
    lat = LAT0 - row * STEP
    current = []
    for col in range(W):
        lon = LON0 + col * STEP
        current.append(1 if any(
            point_in_polygon(lon, lat, p) for p in land_polygons
        ) else 0)
    matrix.append(current)
# CSV output
csv_path = os.path.join(OUT_DIR, "candidate.csv")
with open(csv_path, "w", newline="", encoding="ascii") as f:
    csv.writer(f, lineterminator="\n").writerows(matrix)

# Native black/white image
pixels = [255 if value else 0 for row in matrix for value in row]
image = Image.new("L", (W, H))
image.putdata(pixels)
png_path = os.path.join(OUT_DIR, "candidate.png")
image.save(png_path, format="PNG")

# Nearest-neighbor enlargement
try:
    nearest = Image.Resampling.NEAREST
except AttributeError:
    nearest = Image.NEAREST
large = image.resize((W * 8, H * 8), resample=nearest)
large_path = os.path.join(OUT_DIR, "candidate_8x.png")
large.save(large_path, format="PNG")

# Validation
assert len(matrix) == 64
assert all(len(row) == 253 for row in matrix)
assert all(v in (0, 1) for row in matrix for v in row)

with open(csv_path, encoding="ascii") as f:
    rows = list(csv.reader(f))
assert len(rows) == 64
assert all(len(row) == 253 for row in rows)
assert all(v in ("0", "1") for row in rows for v in row)

check = Image.open(png_path)
check_large = Image.open(large_path)
assert check.size == (253, 64)
assert check_large.size == (2024, 512)
assert set(check.getdata()).issubset({0, 255})
assert set(check_large.getdata()).issubset({0, 255})

print("generated", csv_path)
print("generated", png_path)
print("generated", large_path)
