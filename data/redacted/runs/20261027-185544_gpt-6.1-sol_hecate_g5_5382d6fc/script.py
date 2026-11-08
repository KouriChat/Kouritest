from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image


N = 127
LAT_NORTH = 55.0
LAT_SOUTH = 53.0
LON_WEST = -131.0
LON_EAST = -129.0
OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)


def point_in_polygon(x, y, polygon):
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)):
            xinters = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < xinters:
                inside = not inside
        j = i
    return inside


def ellipse_polygon(cx, cy, rx, ry, steps=32):
    return [
        (
            cx + rx * math.cos(2 * math.pi * k / steps),
            cy + ry * math.sin(2 * math.pi * k / steps),
        )
        for k in range(steps)
    ]


mainland = [
    (-130.05, 55.00), (-130.14, 54.92), (-130.25, 54.83),
    (-130.31, 54.75), (-130.19, 54.66), (-130.34, 54.54),
    (-130.42, 54.40), (-130.29, 54.28), (-130.12, 54.21),
    (-129.99, 54.14), (-129.90, 54.05), (-129.76, 53.96),
    (-129.69, 53.82), (-129.55, 53.70), (-129.45, 53.53),
    (-129.31, 53.36), (-129.19, 53.17), (-129.12, 53.00),
    (-129.00, 53.00), (-129.00, 55.00),
]


water_cuts = [
    [
        (-130.06, 55.00), (-129.91, 55.00), (-129.84, 54.89),
        (-129.90, 54.77), (-129.81, 54.64), (-129.87, 54.51),
        (-129.99, 54.43), (-130.14, 54.45), (-130.20, 54.56),
        (-130.17, 54.68), (-130.07, 54.77), (-130.12, 54.89),
    ],
    [
        (-130.32, 54.40), (-130.16, 54.34), (-130.04, 54.25),
        (-129.91, 54.19), (-129.79, 54.11), (-129.61, 54.08),
        (-129.50, 53.99), (-129.54, 53.89), (-129.70, 53.87),
        (-129.83, 53.93), (-129.94, 54.02), (-130.08, 54.10),
        (-130.20, 54.22), (-130.31, 54.30),
    ],
]
islands = [
    [
        (-131.00, 54.99), (-130.92, 54.94), (-130.88, 54.84),
        (-130.91, 54.75), (-130.86, 54.65), (-130.90, 54.55),
        (-130.96, 54.45), (-131.00, 54.38),
    ],
    [
        (-131.00, 54.82), (-130.88, 54.89), (-130.72, 54.87),
        (-130.58, 54.78), (-130.53, 54.65), (-130.62, 54.55),
        (-130.80, 54.52), (-130.94, 54.59), (-131.00, 54.70),
    ],
    [
        (-130.68, 54.54), (-130.56, 54.59), (-130.44, 54.53),
        (-130.42, 54.43), (-130.51, 54.35), (-130.63, 54.37),
        (-130.72, 54.45),
    ],
    [
        (-130.54, 54.27), (-130.37, 54.32), (-130.15, 54.29),
        (-129.93, 54.20), (-129.76, 54.09), (-129.74, 53.96),
        (-129.86, 53.82), (-130.07, 53.78), (-130.28, 53.84),
        (-130.43, 53.96), (-130.55, 54.11),
    ],
    [
        (-130.42, 54.00), (-130.28, 53.90), (-130.17, 53.78),
        (-130.00, 53.69), (-129.79, 53.66), (-129.61, 53.74),
        (-129.56, 53.87), (-129.64, 53.99), (-129.83, 54.08),
        (-130.09, 54.10), (-130.28, 54.08),
    ],
    [
        (-130.10, 53.61), (-129.91, 53.59), (-129.75, 53.52),
        (-129.60, 53.41), (-129.50, 53.27), (-129.39, 53.12),
        (-129.25, 53.03), (-129.14, 53.00), (-129.40, 53.00),
        (-129.60, 53.08), (-129.78, 53.21), (-129.93, 53.36),
        (-130.08, 53.48),
    ],
    [
        (-130.17, 53.46), (-130.04, 53.40), (-129.99, 53.27),
        (-130.06, 53.17), (-130.19, 53.21), (-130.25, 53.34),
    ],
]


small_islands = [
    (-130.81, 54.39, 0.055, 0.035),
    (-130.46, 54.24, 0.070, 0.040),
    (-130.30, 54.17, 0.045, 0.030),
    (-130.08, 54.35, 0.060, 0.035),
    (-129.89, 54.30, 0.050, 0.030),
    (-129.66, 54.25, 0.060, 0.035),
    (-130.68, 54.11, 0.055, 0.035),
    (-130.49, 54.03, 0.045, 0.030),
    (-130.30, 53.98, 0.050, 0.032),
    (-130.08, 53.98, 0.040, 0.028),
    (-129.92, 53.90, 0.060, 0.035),
    (-129.69, 53.91, 0.050, 0.030),
]
small_islands += [
    (-130.51, 53.76, 0.040, 0.028),
    (-130.34, 53.69, 0.055, 0.030),
    (-130.02, 53.72, 0.050, 0.030),
    (-129.45, 53.72, 0.040, 0.025),
    (-129.28, 53.58, 0.045, 0.028),
    (-129.08, 53.45, 0.045, 0.025),
    (-129.82, 53.67, 0.040, 0.025),
    (-129.49, 53.52, 0.038, 0.025),
]

small_island_polygons = [
    ellipse_polygon(cx, cy, rx, ry)
    for cx, cy, rx, ry in small_islands
]


def is_land(lon, lat):
    land = point_in_polygon(lon, lat, mainland)
    if land:
        for cut in water_cuts:
            if point_in_polygon(lon, lat, cut):
                land = False
                break

    if not land:
        for island in islands:
            if point_in_polygon(lon, lat, island):
                land = True
                break

    if not land:
        for island in small_island_polygons:
            if point_in_polygon(lon, lat, island):
                land = True
                break

    return 1 if land else 0


def build_matrix():
    matrix = np.zeros((N, N), dtype=np.uint8)
    for row in range(N):
        lat = LAT_NORTH - row * (LAT_NORTH - LAT_SOUTH) / (N - 1)
        for col in range(N):
            lon = LON_WEST + col * (LON_EAST - LON_WEST) / (N - 1)
            matrix[row, col] = is_land(lon, lat)
    return matrix


def write_csv(matrix):
    with (OUT / "candidate.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        for row in matrix:
            writer.writerow([int(value) for value in row])


def write_images(matrix):
    pixels = np.where(matrix == 1, 255, 0).astype(np.uint8)
    original = Image.fromarray(pixels, mode="L")
    original.save(OUT / "candidate.png")
    enlarged = original.resize((N * 8, N * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(OUT / "candidate_8x.png")
def validate(matrix):
    assert matrix.shape == (127, 127)
    assert matrix.size == 16129
    assert set(np.unique(matrix)).issubset({0, 1})

    with (OUT / "candidate.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 127
    assert all(len(row) == 127 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    with Image.open(OUT / "candidate.png") as im:
        assert im.size == (127, 127)
        assert set(im.convert("L").getdata()).issubset({0, 255})

    with Image.open(OUT / "candidate_8x.png") as im:
        assert im.size == (1016, 1016)
        assert set(im.convert("L").getdata()).issubset({0, 255})


def main():
    matrix = build_matrix()
    write_csv(matrix)
    write_images(matrix)
    validate(matrix)
    print("matrix:", matrix.shape)
    print("valid values:", int(matrix.size))
    print("land:", int(matrix.sum()))
    print("water:", int(matrix.size - matrix.sum()))


if __name__ == "__main__":
    main()
