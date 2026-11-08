from pathlib import Path
import math

import numpy as np
from PIL import Image

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

ROWS = 64
COLS = 253

lats = np.linspace(35.0, 33.0, ROWS)
lons = np.linspace(129.0, 137.0, COLS)
xx, yy = np.meshgrid(lons, lats)


def polygon_mask(poly):
    x = xx
    y = yy
    result = np.zeros(x.shape, dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        crosses = ((yi > y) != (yj > y))
        denom = yj - yi
        if abs(denom) < 1e-15:
            denom = 1e-15
        x_cross = (xj - xi) * (y - yi) / denom + xi
        result ^= crosses & (x < x_cross)
        j = i
    return result


def ellipse_mask(cx, cy, rx, ry, angle=0.0):
    theta = math.radians(angle)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = xx - cx
    dy = yy - cy
    xr = dx * cos_t + dy * sin_t
    yr = -dx * sin_t + dy * cos_t
    return (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0


land = np.zeros((ROWS, COLS), dtype=bool)


def add_polygon(points):
    global land
    land |= polygon_mask(points)


def add_ellipse(cx, cy, rx, ry, angle=0.0):
    global land
    land |= ellipse_mask(cx, cy, rx, ry, angle)


def remove_polygon(points):
    global land
    land &= ~polygon_mask(points)
add_polygon([
    (130.74, 35.00), (137.00, 35.00), (137.00, 34.72),
    (136.88, 34.58), (136.75, 34.42), (136.62, 34.24),
    (136.48, 34.04), (136.29, 33.84), (136.09, 33.67),
    (135.88, 33.55), (135.68, 33.51), (135.53, 33.63),
    (135.43, 33.83), (135.30, 34.08), (135.18, 34.31),
    (135.02, 34.51), (134.86, 34.64), (134.69, 34.72),
    (134.49, 34.76), (134.26, 34.75), (134.03, 34.68),
    (133.79, 34.61), (133.55, 34.57), (133.30, 34.54),
    (133.05, 34.48), (132.82, 34.43), (132.60, 34.38),
    (132.39, 34.35), (132.17, 34.30), (131.95, 34.24),
    (131.74, 34.17), (131.54, 34.10), (131.34, 34.04),
    (131.15, 33.98), (130.98, 34.02), (130.87, 34.18),
    (130.78, 34.45), (130.73, 34.72),
])

remove_polygon([
    (136.47, 34.28), (136.59, 34.51), (136.69, 34.75),
    (136.76, 34.96), (136.84, 35.00), (137.00, 35.00),
    (137.00, 34.72), (136.88, 34.57), (136.75, 34.40),
    (136.62, 34.24),
])

add_polygon([
    (136.78, 35.00), (137.00, 35.00), (137.00, 34.72),
    (136.91, 34.66), (136.83, 34.75), (136.78, 34.88),
])

add_polygon([
    (131.86, 33.25), (132.00, 33.54), (132.18, 33.78),
    (132.45, 33.95), (132.78, 34.03), (133.14, 34.04),
    (133.50, 34.00), (133.84, 34.02), (134.18, 34.05),
    (134.49, 34.00), (134.71, 33.85), (134.78, 33.56),
    (134.68, 33.26), (134.48, 33.06), (134.15, 32.90),
    (133.76, 32.78), (133.35, 32.73), (132.94, 32.78),
    (132.58, 32.90), (132.25, 33.05),
])

add_polygon([
    (129.44, 33.00), (129.56, 33.35), (129.69, 33.70),
    (129.88, 33.96), (130.17, 34.08), (130.52, 34.12),
    (130.83, 34.03), (131.04, 33.85), (131.15, 33.56),
    (131.18, 33.22), (131.09, 33.00), (130.80, 33.00),
    (130.49, 33.05), (130.19, 33.15), (129.90, 33.22),
    (129.65, 33.18),
])
add_polygon([
    (129.13, 34.05), (129.20, 34.34), (129.21, 34.59),
    (129.31, 34.76), (129.49, 34.84), (129.62, 34.72),
    (129.58, 34.43), (129.47, 34.13), (129.29, 33.99),
])

add_ellipse(129.72, 33.75, 0.14, 0.16, -10)
add_ellipse(129.56, 33.39, 0.10, 0.16, 15)
add_ellipse(129.45, 33.43, 0.06, 0.10, 20)
add_ellipse(129.88, 33.11, 0.08, 0.11, -25)
add_ellipse(130.00, 32.96, 0.10, 0.18, 25)
add_ellipse(130.10, 32.91, 0.07, 0.14, 20)

add_ellipse(132.12, 33.96, 0.10, 0.18, -15)
add_ellipse(132.26, 33.88, 0.15, 0.25, 10)
add_ellipse(132.43, 34.18, 0.15, 0.19, -20)
add_ellipse(132.58, 34.17, 0.09, 0.12, 25)
add_ellipse(132.30, 34.28, 0.07, 0.16, -15)
add_ellipse(132.72, 34.14, 0.08, 0.12, 20)
add_ellipse(132.82, 34.18, 0.10, 0.13, -10)
add_ellipse(132.92, 34.22, 0.08, 0.12, 15)
add_ellipse(132.55, 33.91, 0.07, 0.10, 0)
add_ellipse(132.77, 33.90, 0.08, 0.13, -15)

add_ellipse(132.98, 34.25, 0.13, 0.19, 15)
add_ellipse(133.08, 34.23, 0.09, 0.13, -20)
add_ellipse(133.18, 34.31, 0.11, 0.18, 20)
add_ellipse(133.10, 34.39, 0.08, 0.12, 15)
add_ellipse(133.23, 34.39, 0.06, 0.10, -20)
add_ellipse(133.32, 34.25, 0.10, 0.16, 10)
add_ellipse(133.40, 34.27, 0.07, 0.11, -15)
add_ellipse(133.53, 34.28, 0.07, 0.12, 20)
add_ellipse(133.63, 34.25, 0.06, 0.10, 0)
add_ellipse(133.72, 34.24, 0.06, 0.10, 15)
add_ellipse(133.87, 34.45, 0.07, 0.11, 0)
add_ellipse(133.97, 34.47, 0.10, 0.08, -25)
add_ellipse(134.04, 34.43, 0.07, 0.10, 15)
add_ellipse(134.12, 34.50, 0.06, 0.08, 20)
add_ellipse(134.24, 34.51, 0.08, 0.10, -15)
add_ellipse(134.31, 34.55, 0.23, 0.17, 25)
add_ellipse(134.44, 34.49, 0.08, 0.10, 0)
add_ellipse(134.56, 34.54, 0.07, 0.10, 20)
add_ellipse(134.61, 34.65, 0.07, 0.10, -20)
add_ellipse(134.74, 34.58, 0.07, 0.10, 15)

add_ellipse(134.91, 34.34, 0.15, 0.36, 20)
add_ellipse(134.78, 34.50, 0.06, 0.10, -15)
add_ellipse(134.57, 34.67, 0.10, 0.13, 15)
add_ellipse(134.48, 34.69, 0.06, 0.08, -10)

add_ellipse(131.72, 34.07, 0.06, 0.09, 0)
add_ellipse(131.86, 34.02, 0.05, 0.08, 20)
add_ellipse(132.02, 34.12, 0.06, 0.09, -15)
add_ellipse(132.15, 34.22, 0.05, 0.08, 0)
add_ellipse(132.38, 34.05, 0.06, 0.09, 10)
add_ellipse(132.51, 34.02, 0.05, 0.08, -20)
add_ellipse(132.66, 33.99, 0.05, 0.08, 10)
add_ellipse(133.00, 33.99, 0.05, 0.08, 0)
add_ellipse(133.22, 33.98, 0.05, 0.08, 15)
add_ellipse(133.56, 33.98, 0.05, 0.08, -15)
add_ellipse(133.82, 33.96, 0.05, 0.08, 0)
add_ellipse(134.12, 33.96, 0.05, 0.08, 20)
add_ellipse(134.43, 33.91, 0.05, 0.08, -15)

add_ellipse(132.10, 34.01, 0.06, 0.09, 0)
add_ellipse(132.84, 34.04, 0.06, 0.08, 0)
add_ellipse(133.47, 34.08, 0.05, 0.08, 15)
add_ellipse(134.03, 34.08, 0.06, 0.08, -15)
add_ellipse(134.62, 34.02, 0.05, 0.08, 10)
add_ellipse(131.36, 33.42, 0.06, 0.11, 15)
add_ellipse(131.48, 33.27, 0.05, 0.09, -10)
add_ellipse(131.61, 33.12, 0.05, 0.08, 20)
add_ellipse(131.78, 33.02, 0.06, 0.10, -20)
add_ellipse(132.00, 33.06, 0.05, 0.09, 10)
add_ellipse(132.13, 33.17, 0.06, 0.10, -15)
add_ellipse(132.25, 33.29, 0.05, 0.08, 10)

matrix = land.astype(np.uint8)

assert matrix.shape == (64, 253)
assert matrix.size == 16192
assert np.all((matrix == 0) | (matrix == 1))

np.savetxt(
    OUT / "candidate.csv",
    matrix,
    fmt="%d",
    delimiter=",",
)

raw_pixels = np.where(matrix == 1, 255, 0).astype(np.uint8)
raw_image = Image.fromarray(raw_pixels, mode="L")
assert raw_image.size == (253, 64)
raw_image.save(OUT / "candidate.png")

large_image = raw_image.resize(
    (2024, 512),
    resample=Image.Resampling.NEAREST,
)
assert large_image.size == (2024, 512)
large_image.save(OUT / "candidate_8x.png")

check = np.asarray(raw_image)
assert set(np.unique(check).tolist()) <= {0, 255}
assert np.array_equal(np.asarray(large_image)[::8, ::8], check)
