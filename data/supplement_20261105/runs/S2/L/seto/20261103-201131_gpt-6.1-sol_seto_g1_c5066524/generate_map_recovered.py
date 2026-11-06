#!/usr/bin/env python3
from pathlib import Path
import csv
import math

from PIL import Image


ROWS = 64
COLS = 253

LAT_NORTH = 35.0
LAT_SOUTH = 33.0
LON_WEST = 129.0
LON_EAST = 137.0

OUT_DIR = Path("outputs")
CSV_PATH = OUT_DIR / "candidate.csv"
PNG_PATH = OUT_DIR / "candidate.png"
PNG_8X_PATH = OUT_DIR / "candidate_8x.png"


# 近似多边形包含测试。
# 坐标顺序为 (经度, 纬度)。
def point_in_polygon(lon, lat, polygon):
    inside = False
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        crosses = ((y1 > lat) != (y2 > lat))
        if not crosses:
            continue

        x_at_lat = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
        if lon < x_at_lat:
            inside = not inside

    return inside


def point_in_ellipse(lon, lat, center_lon, center_lat, radius_lon, radius_lat):
    dx = (lon - center_lon) / radius_lon
    dy = (lat - center_lat) / radius_lat
    return dx * dx + dy * dy <= 1.0


# 主要陆块的近似轮廓。
# 这些轮廓故意保留为低分辨率手写近似，用于重建本题范围内的主要海陆关系。
MAINLAND_POLYGONS = [
    # 九州：北部、东部及南部主体
    [
        (129.00, 34.35), (129.35, 34.28), (129.75, 34.15),
        (130.25, 34.08), (130.65, 33.95), (130.85, 33.70),
        (130.85, 33.35), (130.65, 33.05), (131.10, 33.00),
        (131.55, 33.12), (131.85, 33.35), (132.00, 33.65),
        (131.90, 33.95), (131.65, 34.20), (131.35, 34.40),
        (130.85, 34.55), (130.30, 34.58), (129.75, 34.58),
        (129.20, 34.50),
    ],

    # 九州东部延伸及 Bungo 水道一侧
    [
        (131.15, 33.85), (131.55, 34.05), (132.00, 34.02),
        (132.25, 33.82), (132.15, 33.55), (131.85, 33.35),
        (131.45, 33.30), (131.15, 33.42),
    ],

    # 中国地方南缘、濑户内海北岸及本州西部
    [
        (130.55, 34.65), (131.10, 34.70), (131.70, 34.72),
        (132.20, 34.70), (132.75, 34.65), (133.25, 34.62),
        (133.85, 34.62), (134.35, 34.58), (134.90, 34.52),
        (135.35, 34.48), (135.80, 34.42), (136.20, 34.38),
        (136.55, 34.25), (136.85, 34.10), (137.00, 34.05),
        (137.00, 35.00), (130.55, 35.00),
    ],

    # 本州南岸、纪伊半岛和伊势湾东侧
    [
        (134.05, 34.20), (134.35, 34.05), (134.60, 33.90),
        (134.85, 33.75), (135.05, 33.55), (135.15, 33.35),
        (135.35, 33.20), (135.65, 33.15), (135.95, 33.28),
        (136.20, 33.55), (136.35, 33.85), (136.55, 34.10),
        (137.00, 34.10), (137.00, 35.00), (134.05, 35.00),
    ],

    # 四国主体
    [
        (132.05, 34.35), (132.55, 34.32), (133.05, 34.30),
        (133.55, 34.25), (134.05, 34.12), (134.35, 33.95),
        (134.55, 33.70), (134.50, 33.45), (134.25, 33.20),
        (133.90, 33.08), (133.40, 33.05), (132.95, 33.08),
        (132.55, 33.18), (132.20, 33.35), (131.95, 33.58),
        (131.85, 33.90),
    ],
]


# 主要岛屿及较大的岛群。
# 每项为：中心经度、中心纬度、东西半径、南北半径。
ISLAND_ELLIPSES = [
    # 芸予诸岛及其周边
    (132.20, 34.18, 0.070, 0.060),  # 大三島附近
    (132.38, 34.18, 0.075, 0.055),  # 伯方島附近
    (132.52, 34.16, 0.085, 0.060),  # 大島附近
    (132.72, 34.16, 0.060, 0.050),  # 来島周辺
    (132.90, 34.13, 0.065, 0.050),
    (133.10, 34.16, 0.070, 0.055),
    (133.32, 34.18, 0.065, 0.045),
    (133.55, 34.20, 0.060, 0.045),
    (133.75, 34.22, 0.050, 0.040),

    # 广岛湾、山口近岸岛屿
    (132.05, 34.18, 0.060, 0.045),
    (132.25, 34.05, 0.050, 0.040),
    (132.45, 34.02, 0.065, 0.045),
    (132.72, 34.02, 0.055, 0.045),
    (132.95, 34.02, 0.050, 0.040),
    (133.18, 34.02, 0.050, 0.038),
    (133.38, 34.04, 0.045, 0.035),

    # 香川、冈山外海
    (133.72, 34.38, 0.055, 0.040),
    (133.95, 34.38, 0.050, 0.038),
    (134.20, 34.38, 0.060, 0.040),
    (134.48, 34.38, 0.050, 0.038),
    (134.72, 34.38, 0.045, 0.035),
    (134.95, 34.35, 0.050, 0.040),
    (135.20, 34.30, 0.040, 0.035),

    # 淡路岛
    (134.85, 34.35, 0.135, 0.300),

    # 小豆岛
    (134.25, 34.52, 0.105, 0.070),

    # 家岛诸岛
    (134.55, 34.72, 0.060, 0.040),
    (134.72, 34.70, 0.050, 0.035),

    # 伊势湾、三河湾及外海岛屿
    (136.72, 34.48, 0.065, 0.045),
    (136.82, 34.40, 0.055, 0.040),
    (136.95, 34.35, 0.045, 0.035),
    (137.00, 34.62, 0.040, 0.030),
    (136.55, 34.70, 0.045, 0.035),

    # 九州近海岛屿
    (129.55, 34.35, 0.080, 0.055),
    (129.82, 34.25, 0.060, 0.045),
    (130.10, 34.12, 0.070, 0.050),
    (130.40, 34.00, 0.055, 0.040),
    (130.70, 33.85, 0.050, 0.040),
    (130.95, 33.65, 0.045, 0.035),
    (131.28, 33.72, 0.050, 0.038),
    (131.58, 33.80, 0.055, 0.040),

    # 对马南部和壹岐南端在范围内的近似部分
    (129.35, 34.65, 0.090, 0.120),
    (129.90, 34.72, 0.070, 0.090),

    # 四国南岸小岛
    (132.50, 33.10, 0.040, 0.030),
    (132.82, 33.08, 0.050, 0.032),
    (133.18, 33.06, 0.045, 0.030),
    (133.58, 33.10, 0.045, 0.032),
    (133.95, 33.18, 0.040, 0.030),
]


# 细小岛屿、沙洲和礁盘的近似点状扩展。
SMALL_ISLANDS = [
    (131.82, 34.18, 0.025, 0.022),
    (131.95, 34.12, 0.020, 0.018),
    (132.12, 34.28, 0.022, 0.018),
    (132.62, 34.28, 0.025, 0.018),
    (132.82, 34.25, 0.022, 0.018),
    (133.00, 34.28, 0.020, 0.016),
    (133.25, 34.28, 0.023, 0.017),
    (133.48, 34.30, 0.020, 0.016),
    (133.82, 34.30, 0.022, 0.016),
    (134.05, 34.30, 0.020, 0.015),
    (134.42, 34.28, 0.022, 0.016),
    (134.68, 34.25, 0.020, 0.015),
    (134.95, 34.22, 0.022, 0.016),
    (135.35, 34.20, 0.025, 0.018),
    (135.72, 34.18, 0.020, 0.015),
    (136.05, 34.15, 0.020, 0.015),
    (136.38, 34.28, 0.022, 0.016),
    (136.65, 34.25, 0.020, 0.015),
    (136.88, 34.25, 0.018, 0.014),
    (130.25, 33.85, 0.025, 0.018),
    (130.45, 33.68, 0.020, 0.016),
    (130.80, 33.48, 0.025, 0.018),
    (131.05, 33.45, 0.022, 0.016),
    (131.38, 33.42, 0.020, 0.015),
    (131.75, 33.55, 0.022, 0.016),
]


def is_land(lon, lat):
    for polygon in MAINLAND_POLYGONS:
        if point_in_polygon(lon, lat, polygon):
            return 1

    for item in ISLAND_ELLIPSES:
        if point_in_ellipse(lon, lat, *item):
            return 1

    for item in SMALL_ISLANDS:
        if point_in_ellipse(lon, lat, *item):
            return 1

    return 0


def build_matrix():
    matrix = []

    for row in range(ROWS):
        lat = LAT_NORTH - (LAT_NORTH - LAT_SOUTH) * row / (ROWS - 1)
        values = []

        for col in range(COLS):
            lon = LON_WEST + (LON_EAST - LON_WEST) * col / (COLS - 1)
            values.append(is_land(lon, lat))

        matrix.append(values)

    return matrix


def write_csv(matrix):
    with CSV_PATH.open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix)


def write_images(matrix):
    image = Image.new("1", (COLS, ROWS), 0)
    pixels = image.load()

    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            pixels[x, y] = 1 if value == 1 else 0

    image.save(PNG_PATH)

    enlarged = image.resize(
        (COLS * 8, ROWS * 8),
        resample=Image.Resampling.NEAREST,
    )
    enlarged.save(PNG_8X_PATH)


def validate(matrix):
    assert len(matrix) == 64
    assert all(len(row) == 253 for row in matrix)
    assert all(value in (0, 1) for row in matrix for value in row)

    with CSV_PATH.open("r", newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))

    assert len(rows) == 64
    assert all(len(row) == 253 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    with Image.open(PNG_PATH) as image:
        assert image.size == (253, 64)
        colors = set(image.convert("1").getdata())
        assert colors <= {0, 255}

    with Image.open(PNG_8X_PATH) as image:
        assert image.size == (2024, 512)
        colors = set(image.convert("1").getdata())
        assert colors <= {0, 255}

    assert ROWS * COLS == 16192


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    matrix = build_matrix()
    write_csv(matrix)
    write_images(matrix)
    validate(matrix)

    land_count = sum(value for row in matrix for value in row)
    water_count = ROWS * COLS - land_count

    print(f"matrix: {ROWS} x {COLS} = {ROWS * COLS}")
    print(f"land: {land_count}")
    print(f"water: {water_count}")
    print(f"csv: {CSV_PATH}")
    print(f"png: {PNG_PATH} ({COLS}x{ROWS})")
    print(f"png_8x: {PNG_8X_PATH} ({COLS * 8}x{ROWS * 8})")
    print("validation: OK")


if __name__ == "__main__":
    main()
