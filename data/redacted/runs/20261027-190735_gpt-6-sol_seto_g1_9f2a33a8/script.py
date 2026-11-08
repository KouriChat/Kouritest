from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image


ROWS = 64
COLS = 253
LON_MIN = 129.0
LAT_MAX = 35.0
STEP = 2.0 / 63.0


def ellipse_polygon(lon, lat, radius_lon, radius_lat,
                    count=24, phase=0.0):
    points = []
    for k in range(count):
        angle = phase + 2.0 * math.pi * k / count
        points.append((
            lon + radius_lon * math.cos(angle),
            lat + radius_lat * math.sin(angle),
        ))
    return points


LAND_POLYGONS = [
    [
        (129.000, 35.000), (129.220, 35.000),
        (129.260, 34.930), (129.150, 34.850),
        (129.000, 34.880),
    ],
    [
        (129.000, 33.000), (129.000, 34.000),
        (129.180, 34.330), (129.450, 34.540),
        (129.850, 34.670), (130.250, 34.750),
        (130.650, 34.780), (130.900, 34.650),
        (131.050, 34.450), (131.180, 34.200),
        (131.350, 33.950), (131.450, 33.680),
        (131.560, 33.350), (131.470, 33.000),
    ],
    [
        (130.650, 35.000), (135.220, 35.000),
        (135.300, 34.900), (135.220, 34.820),
        (135.050, 34.750), (134.800, 34.720),
        (134.550, 34.710), (134.300, 34.670),
        (134.000, 34.630), (133.700, 34.630),
        (133.420, 34.570), (133.100, 34.520),
        (132.850, 34.500), (132.620, 34.460),
        (132.420, 34.420), (132.250, 34.470),
        (132.080, 34.390), (131.880, 34.350),
        (131.680, 34.300), (131.480, 34.250),
        (131.280, 34.200), (131.050, 34.240),
        (130.820, 34.330), (130.650, 34.350),
    ],
]


ISLANDS = [
    ellipse_polygon(129.330, 34.420, 0.075, 0.300, 28, 0.2),
    ellipse_polygon(129.390, 34.050, 0.055, 0.160, 22, 0.4),
    ellipse_polygon(129.760, 33.780, 0.120, 0.190, 24, 0.1),
    ellipse_polygon(129.580, 33.370, 0.080, 0.130, 22, 0.3),
    ellipse_polygon(129.420, 33.120, 0.060, 0.120, 20, 0.5),
    ellipse_polygon(131.350, 33.950, 0.055, 0.110, 20, 0.2),
    ellipse_polygon(131.730, 33.920, 0.050, 0.080, 18, 0.1),
    ellipse_polygon(132.210, 33.900, 0.110, 0.270, 28, 0.2),
    ellipse_polygon(132.300, 34.300, 0.070, 0.150, 24, 0.1),
    ellipse_polygon(132.440, 34.250, 0.140, 0.180, 26, 0.4),
    ellipse_polygon(132.480, 34.080, 0.100, 0.130, 22, 0.2),
    ellipse_polygon(132.570, 34.020, 0.070, 0.110, 20, 0.5),
    ellipse_polygon(132.700, 34.220, 0.085, 0.120, 22, 0.1),
    ellipse_polygon(132.830, 34.270, 0.090, 0.130, 22, 0.3),
    ellipse_polygon(132.960, 34.290, 0.120, 0.160, 24, 0.2),
    ellipse_polygon(133.100, 34.350, 0.105, 0.145, 24, 0.1),
    ellipse_polygon(133.260, 34.270, 0.095, 0.145, 24, 0.4),
    ellipse_polygon(133.120, 34.120, 0.100, 0.140, 24, 0.2),
    ellipse_polygon(133.300, 34.120, 0.090, 0.135, 22, 0.5),
    ellipse_polygon(133.030, 34.010, 0.080, 0.120, 20, 0.3),
    ellipse_polygon(133.430, 34.200, 0.050, 0.100, 20, 0.2),
    ellipse_polygon(133.520, 34.420, 0.080, 0.100, 20, 0.1),
    ellipse_polygon(133.700, 34.470, 0.090, 0.110, 22, 0.2),
    ellipse_polygon(133.880, 34.500, 0.095, 0.120, 22, 0.3),
    ellipse_polygon(134.000, 34.470, 0.180, 0.150, 28, 0.1),
    ellipse_polygon(134.090, 34.420, 0.060, 0.090, 20, 0.4),
    ellipse_polygon(134.160, 34.550, 0.070, 0.080, 20, 0.2),
    ellipse_polygon(134.300, 34.370, 0.060, 0.090, 20, 0.5),
]
LAND_POLYGONS += [
    [
        (135.100, 35.000), (137.000, 35.000),
        (137.000, 34.620), (136.920, 34.540),
        (136.850, 34.400), (136.820, 34.300),
        (136.620, 34.330), (136.400, 34.400),
        (136.180, 34.500), (135.950, 34.590),
        (135.720, 34.660), (135.480, 34.700),
        (135.220, 34.720), (135.100, 34.750),
    ],
    [
        (132.000, 33.250), (132.120, 33.540),
        (132.180, 33.820), (132.140, 34.070),
        (132.320, 34.240), (132.620, 34.350),
        (132.980, 34.400), (133.360, 34.380),
        (133.760, 34.350), (134.150, 34.320),
        (134.500, 34.210), (134.760, 34.020),
        (134.760, 33.720), (134.640, 33.420),
        (134.450, 33.200), (134.100, 33.050),
        (133.700, 33.000), (133.200, 33.000),
        (132.700, 33.050), (132.300, 33.150),
    ],
    [
        (134.700, 34.350), (135.020, 34.350),
        (135.300, 34.260), (135.550, 34.150),
        (135.760, 34.040), (136.020, 34.000),
        (136.300, 34.080), (136.550, 34.230),
        (136.750, 34.350), (136.800, 34.150),
        (136.620, 33.950), (136.380, 33.800),
        (136.150, 33.550), (136.000, 33.300),
        (135.720, 33.050), (135.200, 33.000),
        (134.850, 33.100), (134.620, 33.450),
    ],
]

ISLANDS += [
    [
        (134.620, 34.120), (134.660, 34.350),
        (134.710, 34.570), (134.800, 34.700),
        (134.960, 34.680), (135.040, 34.500),
        (134.980, 34.270), (134.900, 34.070),
        (134.760, 33.980), (134.650, 34.000),
    ],
    ellipse_polygon(134.430, 34.300, 0.045, 0.065, 18, 0.2),
    ellipse_polygon(134.570, 34.420, 0.050, 0.070, 18, 0.4),
    ellipse_polygon(134.930, 34.080, 0.040, 0.070, 18, 0.1),
    ellipse_polygon(136.680, 34.270, 0.035, 0.055, 18, 0.2),
    ellipse_polygon(136.770, 34.330, 0.040, 0.060, 18, 0.4),
    ellipse_polygon(136.900, 34.420, 0.035, 0.050, 18, 0.1),
    ellipse_polygon(136.250, 34.080, 0.050, 0.080, 20, 0.2),
]
def point_on_segment(x, y, x1, y1, x2, y2, eps=1e-12):
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > eps:
        return False
    return (
        min(x1, x2) - eps <= x <= max(x1, x2) + eps
        and min(y1, y2) - eps <= y <= max(y1, y2) + eps
    )


def point_in_polygon(x, y, polygon):
    inside = False
    count = len(polygon)
    for i in range(count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]
        if point_on_segment(x, y, x1, y1, x2, y2):
            return True
        if (y1 > y) != (y2 > y):
            intersection = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < intersection:
                inside = not inside
    return inside


def is_land(lon, lat):
    for polygon in LAND_POLYGONS:
        if point_in_polygon(lon, lat, polygon):
            return True
    for polygon in ISLANDS:
        if point_in_polygon(lon, lat, polygon):
            return True
    return False


def build_matrix():
    matrix = np.zeros((ROWS, COLS), dtype=np.uint8)
    for row in range(ROWS):
        lat = LAT_MAX - row * STEP
        for col in range(COLS):
            lon = LON_MIN + col * STEP
            matrix[row, col] = int(is_land(lon, lat))
    return matrix


def write_csv(matrix, path):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        for row in matrix:
            writer.writerow([int(value) for value in row])


def write_images(matrix, original_path, enlarged_path):
    image = Image.fromarray((matrix * 255).astype(np.uint8), mode="L")
    image.save(original_path)
    enlarged = image.resize(
        (COLS * 8, ROWS * 8),
        resample=Image.Resampling.NEAREST,
    )
    enlarged.save(enlarged_path)
def validate(matrix, csv_path, original_path, enlarged_path):
    if matrix.shape != (64, 253):
        raise RuntimeError("matrix shape")
    if matrix.size != 16192:
        raise RuntimeError("matrix size")
    if not set(np.unique(matrix).tolist()).issubset({0, 1}):
        raise RuntimeError("matrix values")

    with csv_path.open("r", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if len(rows) != 64 or any(len(row) != 253 for row in rows):
        raise RuntimeError("csv dimensions")
    if any(value not in {"0", "1"} for row in rows for value in row):
        raise RuntimeError("csv values")

    original = Image.open(original_path).convert("L")
    enlarged = Image.open(enlarged_path).convert("L")
    if original.size != (253, 64):
        raise RuntimeError("original image dimensions")
    if enlarged.size != (2024, 512):
        raise RuntimeError("enlarged image dimensions")
    if not set(original.getdata()).issubset({0, 255}):
        raise RuntimeError("original image colors")
    if not set(enlarged.getdata()).issubset({0, 255}):
        raise RuntimeError("enlarged image colors")

    expected = np.repeat(
        np.repeat(np.asarray(original), 8, axis=0), 8, axis=1
    )
    if not np.array_equal(np.asarray(enlarged), expected):
        raise RuntimeError("not nearest-neighbor")


def main():
    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "candidate.csv"
    original_path = output_dir / "candidate.png"
    enlarged_path = output_dir / "candidate_8x.png"

    matrix = build_matrix()
    write_csv(matrix, csv_path)
    write_images(matrix, original_path, enlarged_path)
    validate(matrix, csv_path, original_path, enlarged_path)
    print("generated", matrix.shape, matrix.size)


if __name__ == "__main__":
    main()
