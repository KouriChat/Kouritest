from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image


# ============================================================
# 手工近似地理规则
# 坐标均为 (经度, 纬度)，西经使用负数。
# 这些轮廓仅来自已有地理知识，不读取外部地图或标准答案。
# ============================================================

LAND_POLYGONS = [
    # Ellef Ringnes Island
    [
        (-105.30, 78.15), (-104.95, 78.75), (-104.10, 79.10),
        (-102.75, 79.18), (-101.55, 78.96), (-100.35, 78.63),
        (-99.35, 78.30), (-99.10, 77.86), (-99.55, 77.52),
        (-100.65, 77.27), (-102.05, 77.24), (-103.35, 77.43),
        (-104.45, 77.72)
    ],

    # Amund Ringnes Island
    [
        (-98.85, 78.15), (-98.45, 78.58), (-97.65, 78.86),
        (-96.45, 78.98), (-95.25, 78.79), (-94.25, 78.46),
        (-93.78, 78.08), (-94.18, 77.78), (-95.20, 77.60),
        (-96.45, 77.62), (-97.55, 77.79), (-98.35, 77.93)
    ],

    # Axel Heiberg Island, southern portion
    [
        (-96.60, 79.30), (-94.90, 79.35), (-93.35, 79.17),
        (-92.05, 78.91), (-91.10, 78.58), (-90.60, 78.17),
        (-90.92, 77.86), (-91.65, 77.65), (-92.45, 77.78),
        (-93.25, 78.03), (-94.35, 78.25), (-95.30, 78.60),
        (-96.25, 78.93)
    ],

    # Ellesmere Island, portion inside the requested frame
    [
        (-92.30, 79.35), (-89.50, 79.40), (-86.50, 79.32),
        (-83.50, 79.18), (-80.20, 79.10), (-80.20, 76.48),
        (-81.20, 76.46), (-82.35, 76.50), (-83.55, 76.55),
        (-84.70, 76.60), (-85.75, 76.67), (-86.70, 76.62),
        (-87.55, 76.72), (-88.45, 76.79), (-89.20, 76.94),
        (-89.65, 77.18), (-90.55, 77.34), (-91.30, 77.63),
        (-91.72, 77.97), (-92.35, 78.30), (-92.05, 78.65)
    ],

    # Devon Island
    [
        (-94.85, 75.12), (-94.72, 75.66), (-94.22, 76.10),
        (-93.40, 76.43), (-92.30, 76.68), (-90.95, 76.72),
        (-89.50, 76.62), (-88.15, 76.50), (-86.75, 76.44),
        (-85.35, 76.34), (-83.90, 76.30), (-82.50, 76.20),
        (-80.30, 76.22), (-80.30, 74.10), (-81.20, 74.16),
        (-82.40, 74.22), (-83.75, 74.28), (-85.10, 74.33),
        (-86.35, 74.36), (-87.65, 74.39), (-88.85, 74.36),
        (-89.85, 74.40), (-90.75, 74.48), (-91.60, 74.43),
        (-92.45, 74.47), (-93.25, 74.58), (-94.08, 74.75)
    ],

    # Cornwallis Island
    [
        (-96.92, 75.08), (-96.65, 75.48), (-96.05, 75.72),
        (-95.15, 75.82), (-94.25, 75.70), (-93.52, 75.48),
        (-93.15, 75.08), (-93.30, 74.77), (-94.00, 74.55),
        (-94.90, 74.46), (-95.82, 74.50), (-96.52, 74.72)
    ],

    # Bathurst Island western lobe
    [
        (-102.80, 75.85), (-102.58, 76.35), (-102.20, 76.73),
        (-101.52, 76.98), (-100.75, 76.91), (-100.36, 76.58),
        (-100.48, 76.17), (-100.90, 75.78), (-101.72, 75.60),
        (-102.45, 75.66)
    ],

    # Bathurst Island central/eastern lobe
    [
        (-101.05, 75.46), (-100.45, 75.72), (-100.22, 76.12),
        (-100.06, 76.62), (-99.55, 76.89), (-98.82, 76.80),
        (-98.25, 76.54), (-97.68, 76.20), (-97.38, 75.78),
        (-97.62, 75.48), (-98.38, 75.30), (-99.28, 75.28),
        (-100.18, 75.31)
    ],

    # Bathurst northern connecting land
    [
        (-100.62, 76.35), (-100.18, 76.78), (-99.55, 77.02),
        (-98.88, 76.98), (-98.38, 76.70), (-98.55, 76.39),
        (-99.24, 76.23), (-100.00, 76.20)
    ],

    # Alexander Island
    [
        (-103.85, 75.72), (-103.48, 76.02), (-102.90, 76.20),
        (-102.32, 76.12), (-102.06, 75.83), (-102.34, 75.58),
        (-102.95, 75.45), (-103.55, 75.50)
    ],

    # Massey Island
    [
        (-104.18, 76.45), (-103.80, 76.75), (-103.13, 76.91),
        (-102.52, 76.82), (-102.26, 76.53), (-102.55, 76.31),
        (-103.20, 76.25), (-103.82, 76.29)
    ],

    # Vanier Island
    [
        (-104.48, 75.86), (-104.06, 76.18), (-103.45, 76.31),
        (-102.94, 76.15), (-102.86, 75.86), (-103.25, 75.66),
        (-103.92, 75.62)
    ],

    # Cameron Island
    [
        (-105.35, 76.22), (-104.88, 76.62), (-104.25, 76.82),
        (-103.78, 76.70), (-103.66, 76.35), (-104.08, 76.12),
        (-104.72, 76.04)
    ],

    # Eastern edge of Melville Island
    [
        (-106.00, 74.88), (-105.30, 75.05), (-104.86, 75.42),
        (-104.72, 75.86), (-104.86, 76.24), (-105.22, 76.62),
        (-106.00, 76.88)
    ],

    # Byam Martin Island
    [
        (-105.40, 74.88), (-104.92, 75.28), (-104.28, 75.50),
        (-103.78, 75.42), (-103.55, 75.14), (-103.82, 74.89),
        (-104.42, 74.72), (-105.02, 74.73)
    ],

    # Lougheed Island, eastern portion
    [
        (-106.00, 77.10), (-105.36, 77.27), (-104.78, 77.60),
        (-104.70, 77.91), (-105.15, 78.12), (-106.00, 78.18)
    ],

    # Prince of Wales Island, northern portion
    [
        (-104.65, 72.60), (-104.52, 73.18), (-104.05, 73.60),
        (-103.50, 73.95), (-102.72, 74.18), (-101.68, 74.30),
        (-100.48, 74.29), (-99.30, 74.19), (-98.25, 74.04),
        (-97.35, 73.82), (-96.72, 73.50), (-96.35, 73.08),
        (-96.18, 72.60)
    ],

    # Somerset Island, northern portion
    [
        (-95.66, 72.60), (-95.55, 73.12), (-95.35, 73.58),
        (-94.92, 73.92), (-94.25, 74.09), (-93.35, 74.14),
        (-92.48, 74.07), (-91.70, 73.91), (-91.10, 73.64),
        (-90.72, 73.25), (-90.55, 72.60)
    ],

    # Northwestern Bylot Island, mostly east of the frame
    [
        (-81.35, 72.70), (-80.92, 73.10), (-80.72, 73.55),
        (-80.48, 73.88), (-79.55, 74.05), (-78.60, 73.90),
        (-78.40, 72.70)
    ],
]


# (中心经度, 中心纬度, 经度半径, 纬度半径, 逆时针旋转角)
LAND_ELLIPSES = [
    # Norwegian Bay and Sverdrup-group islands
    (-102.20, 77.72, 0.72, 0.25, -12.0),  # King Christian
    (-94.85, 77.63, 1.13, 0.37, -8.0),    # Cornwall Island
    (-90.55, 77.39, 0.75, 0.24, 8.0),     # Graham Island
    (-91.38, 77.06, 0.47, 0.19, -10.0),   # Buckingham Island
    (-90.12, 76.73, 0.48, 0.18, 5.0),     # North Kent
    (-96.85, 77.08, 0.30, 0.15, 0.0),
    (-97.72, 77.30, 0.25, 0.12, -10.0),

    # Around Bathurst and Cornwallis
    (-101.08, 75.14, 0.39, 0.14, -8.0),
    (-99.96, 75.02, 0.27, 0.12, 12.0),
    (-97.13, 75.57, 0.33, 0.17, 0.0),     # Little Cornwallis vicinity
    (-96.84, 75.34, 0.25, 0.12, 10.0),
    (-103.20, 77.14, 0.34, 0.14, -15.0),

    # Barrow Strait islands
    (-100.72, 74.36, 0.36, 0.14, -8.0),
    (-99.30, 74.47, 0.31, 0.13, 4.0),
    (-97.75, 74.55, 0.67, 0.22, -5.0),    # Lowther Island
    (-96.48, 74.62, 0.28, 0.12, 8.0),
    (-95.52, 74.55, 0.47, 0.18, -8.0),    # Griffith Island
    (-91.86, 74.72, 0.29, 0.10, 4.0),     # Beechey Island
    (-90.06, 74.02, 0.19, 0.10, -5.0),    # Prince Leopold Island

    # Peel Sound / northern Somerset vicinity
    (-96.18, 73.36, 0.31, 0.14, 8.0),
    (-96.02, 73.05, 0.25, 0.12, -5.0),
    (-95.73, 73.72, 0.18, 0.10, 0.0),
    (-90.42, 73.74, 0.18, 0.09, 0.0),

    # Additional small islands/reefs represented at this scale
    (-93.05, 76.93, 0.18, 0.09, 0.0),
    (-92.25, 76.82, 0.16, 0.08, 0.0),
    (-98.04, 74.93, 0.15, 0.08, 0.0),
    (-94.02, 74.31, 0.14, 0.07, 0.0),
]


# 海湾和海峡切口。只有这些与外海相连的水体被扣除；
# 内陆湖泊不扣除，因为任务定义将其按陆地处理。
WATER_POLYGONS = [
    # Bathurst Island: northern and southern inlets
    [
        (-101.25, 77.12), (-100.72, 77.12), (-100.62, 76.75),
        (-100.70, 76.45), (-100.94, 76.30), (-101.15, 76.48)
    ],
    [
        (-100.95, 75.15), (-100.43, 75.15), (-100.35, 75.55),
        (-100.40, 75.91), (-100.62, 76.13), (-100.85, 75.91)
    ],
    [
        (-98.02, 75.58), (-97.30, 75.50), (-97.25, 75.82),
        (-97.72, 75.94)
    ],

    # Cornwallis southern bay
    [
        (-95.18, 74.36), (-94.68, 74.36), (-94.62, 74.67),
        (-94.82, 74.82), (-95.06, 74.67)
    ],

    # Devon southwest/south coastal indentations
    [
        (-94.92, 75.48), (-94.35, 75.50), (-94.08, 75.70),
        (-94.15, 75.93), (-94.55, 75.82)
    ],
    [
        (-92.20, 74.31), (-91.62, 74.30), (-91.58, 74.57),
        (-91.79, 74.70), (-92.06, 74.56)
    ],
    [
        (-88.82, 74.25), (-88.18, 74.24), (-88.12, 74.50),
        (-88.38, 74.63), (-88.68, 74.49)
    ],
    [
        (-84.50, 74.14), (-83.77, 74.14), (-83.72, 74.38),
        (-84.02, 74.51), (-84.34, 74.39)
    ],

    # Ellesmere southern/western coastal inlets
    [
        (-89.75, 76.76), (-89.10, 76.70), (-88.90, 76.91),
        (-89.15, 77.08), (-89.50, 77.04)
    ],
    [
        (-92.50, 78.13), (-91.80, 78.05), (-91.48, 78.28),
        (-91.68, 78.49), (-92.12, 78.43)
    ],

    # Prince of Wales northern bay
    [
        (-100.58, 74.36), (-99.86, 74.34), (-99.78, 74.07),
        (-100.02, 73.84), (-100.33, 73.91)
    ],

    # Somerset northern bay
    [
        (-93.85, 74.22), (-93.25, 74.22), (-93.19, 73.96),
        (-93.43, 73.78), (-93.70, 73.91)
    ],
]


def point_on_segment(x, y, x1, y1, x2, y2, eps=1e-10):
    """Return True when a point lies on a polygon boundary segment."""
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > eps:
        return False

    return (
        min(x1, x2) - eps <= x <= max(x1, x2) + eps
        and min(y1, y2) - eps <= y <= max(y1, y2) + eps
    )


def point_in_polygon(x, y, polygon):
    """
    Ray-casting point-in-polygon test.
    Boundary points are assigned to the polygon's interior.
    """
    inside = False
    count = len(polygon)

    for i in range(count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]

        if point_on_segment(x, y, x1, y1, x2, y2):
            return True

        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing_x:
                inside = not inside

    return inside


def point_in_rotated_ellipse(x, y, ellipse):
    cx, cy, rx, ry, angle_degrees = ellipse
    angle = math.radians(angle_degrees)

    dx = x - cx
    dy = y - cy

    # Rotate the point into the ellipse's local coordinate system.
    local_x = dx * math.cos(angle) + dy * math.sin(angle)
    local_y = -dx * math.sin(angle) + dy * math.cos(angle)

    value = (local_x / rx) ** 2 + (local_y / ry) ** 2
    return value <= 1.0 + 1e-12


def is_land(lon, lat):
    """Apply the complete hand-authored land/water rule."""
    land = any(
        point_in_polygon(lon, lat, polygon)
        for polygon in LAND_POLYGONS
    )

    if not land:
        land = any(
            point_in_rotated_ellipse(lon, lat, ellipse)
            for ellipse in LAND_ELLIPSES
        )

    if not land:
        return 0

    # Coastal inlets override land polygons.
    if any(
        point_in_polygon(lon, lat, polygon)
        for polygon in WATER_POLYGONS
    ):
        return 0

    return 1


def write_csv(path, matrix):
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(matrix.tolist())


def validate_csv(path):
    with path.open("r", encoding="ascii", newline="") as handle:
        rows = list(csv.reader(handle))

    assert len(rows) == 61
    assert all(len(row) == 241 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert sum(len(row) for row in rows) == 14701


def main():
    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Integer-derived coordinates avoid cumulative floating-point stepping.
    latitudes = [79.0 - row / 10.0 for row in range(61)]
    longitudes = [-105.0 + column / 10.0 for column in range(241)]

    matrix = np.zeros((61, 241), dtype=np.uint8)

    for row, latitude in enumerate(latitudes):
        for column, longitude in enumerate(longitudes):
            matrix[row, column] = is_land(longitude, latitude)

    assert matrix.shape == (61, 241)
    assert matrix.size == 14701
    assert set(np.unique(matrix).tolist()).issubset({0, 1})
    assert np.any(matrix == 0)
    assert np.any(matrix == 1)

    csv_path = output_dir / "candidate.csv"
    original_path = output_dir / "candidate.png"
    enlarged_path = output_dir / "candidate_8x.png"

    write_csv(csv_path, matrix)

    # Water=0 becomes black; land=1 becomes white.
    pixels = matrix * np.uint8(255)
    original = Image.fromarray(pixels, mode="L")
    assert original.size == (241, 61)
    original.save(original_path, format="PNG")

    try:
        nearest = Image.Resampling.NEAREST
    except AttributeError:
        nearest = Image.NEAREST

    enlarged = original.resize((1928, 488), resample=nearest)
    enlarged.save(enlarged_path, format="PNG")

    # Re-open files so validation also covers actual serialized artifacts.
    validate_csv(csv_path)

    with Image.open(original_path) as checked_original:
        checked_original.load()
        original_array = np.asarray(checked_original)
        assert checked_original.size == (241, 61)
        assert original_array.shape == (61, 241)
        assert set(np.unique(original_array).tolist()) == {0, 255}

    with Image.open(enlarged_path) as checked_enlarged:
        checked_enlarged.load()
        enlarged_array = np.asarray(checked_enlarged)
        assert checked_enlarged.size == (1928, 488)
        assert enlarged_array.shape == (488, 1928)
        assert set(np.unique(enlarged_array).tolist()) == {0, 255}

    expected_enlargement = np.repeat(
        np.repeat(original_array, 8, axis=0),
        8,
        axis=1,
    )
    assert np.array_equal(enlarged_array, expected_enlargement)

    print("Generated and validated:")
    print(csv_path)
    print(original_path)
    print(enlarged_path)
    print(f"Matrix: {matrix.shape[0]} x {matrix.shape[1]}")
    print(f"Values: {matrix.size}")
    print(f"Land points: {int(matrix.sum())}")
    print(f"Water points: {int(matrix.size - matrix.sum())}")


if __name__ == "__main__":
    main()
