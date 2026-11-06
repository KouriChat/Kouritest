from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image


# 输出网格
ROWS = 64
COLS = 253

LAT_NORTH = 35.0
LAT_SOUTH = 33.0
LON_WEST = 129.0
LON_EAST = 137.0

latitudes = np.linspace(LAT_NORTH, LAT_SOUTH, ROWS)
longitudes = np.linspace(LON_WEST, LON_EAST, COLS)


def point_in_polygon(x, y, polygon):
    """射线法判断点是否位于多边形内。"""
    inside = False
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        if ((y1 > y) != (y2 > y)):
            cross_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < cross_x:
                inside = not inside

    return inside


def ellipse_contains(x, y, cx, cy, rx, ry, angle=0.0):
    """判断点是否位于旋转椭圆内。角度单位为度。"""
    a = math.radians(angle)
    ca = math.cos(a)
    sa = math.sin(a)

    dx = x - cx
    dy = y - cy

    local_x = ca * dx + sa * dy
    local_y = -sa * dx + ca * dy

    return (local_x / rx) ** 2 + (local_y / ry) ** 2 <= 1.0


# 近似陆地多边形。
# 坐标顺序为 (经度, 纬度)。
#
# 这些多边形覆盖本采样框内可见的主要陆块：
# 九州北部、下关一带、本州中国地方南岸、四国、纪伊半岛、
# 伊势湾周边以及本州北侧被框截取的陆地。
LAND_POLYGONS = [
    # 九州北部与西部边缘
    [
        (129.00, 35.00), (129.00, 33.00),
        (130.00, 33.00), (130.25, 33.20),
        (130.50, 33.45), (130.72, 33.78),
        (130.70, 34.00), (130.55, 34.18),
        (130.25, 34.32), (129.90, 34.45),
        (129.55, 34.65), (129.25, 34.85),
        (129.00, 34.92),
    ],

    # 九州北端至关门海峡一带
    [
        (129.00, 35.00), (131.10, 35.00),
        (131.00, 34.82), (130.82, 34.72),
        (130.65, 34.62), (130.48, 34.53),
        (130.30, 34.47), (130.05, 34.45),
        (129.78, 34.49), (129.55, 34.61),
        (129.30, 34.78), (129.00, 34.90),
    ],

    # 本州中国地方南岸及其北侧
    [
        (130.85, 35.00), (137.00, 35.00),
        (137.00, 34.45), (136.76, 34.35),
        (136.48, 34.25), (136.20, 34.14),
        (135.90, 34.10), (135.62, 34.16),
        (135.35, 34.25), (135.12, 34.30),
        (134.88, 34.38), (134.62, 34.47),
        (134.30, 34.54), (133.98, 34.58),
        (133.68, 34.57), (133.38, 34.49),
        (133.08, 34.42), (132.82, 34.37),
        (132.55, 34.34), (132.30, 34.31),
        (132.08, 34.28), (131.82, 34.29),
        (131.58, 34.34), (131.38, 34.42),
        (131.16, 34.55), (130.98, 34.70),
        (130.86, 34.84),
    ],

    # 纪伊半岛
    [
        (135.05, 34.65), (135.30, 34.72),
        (135.62, 34.70), (135.95, 34.58),
        (136.28, 34.45), (136.60, 34.28),
        (136.90, 34.14), (137.00, 34.08),
        (137.00, 33.00), (136.45, 33.00),
        (136.28, 33.15), (136.12, 33.34),
        (135.95, 33.55), (135.78, 33.77),
        (135.60, 34.00), (135.38, 34.22),
        (135.17, 34.42),
    ],

    # 四国主体
    [
        (131.70, 34.18), (132.02, 34.10),
        (132.34, 34.06), (132.68, 34.03),
        (133.00, 33.99), (133.35, 33.94),
        (133.72, 33.88), (134.08, 33.78),
        (134.38, 33.64), (134.68, 33.46),
        (134.92, 33.28), (135.00, 33.00),
        (134.45, 33.00), (134.26, 33.12),
        (134.08, 33.20), (133.82, 33.24),
        (133.54, 33.22), (133.23, 33.18),
        (132.90, 33.15), (132.56, 33.17),
        (132.24, 33.25), (131.98, 33.36),
        (131.75, 33.52), (131.58, 33.72),
        (131.52, 33.92),
    ],

    # 四国西端与丰后水道一带
    [
        (130.95, 34.08), (131.25, 34.16),
        (131.55, 34.18), (131.70, 34.08),
        (131.58, 33.84), (131.42, 33.62),
        (131.28, 33.40), (131.08, 33.20),
        (130.82, 33.00), (130.30, 33.00),
        (130.48, 33.26), (130.66, 33.52),
        (130.78, 33.76), (130.82, 33.94),
    ],
]


# 近似岛屿、岛群、沙洲和较大的礁盘。
# (中心经度, 中心纬度, 东西半径, 南北半径, 旋转角度)
ISLANDS = [
    # 芸予诸岛
    (132.00, 34.18, 0.075, 0.045, 15),
    (132.13, 34.20, 0.060, 0.040, -10),
    (132.25, 34.18, 0.085, 0.045, 20),
    (132.38, 34.20, 0.060, 0.035, -15),
    (132.51, 34.20, 0.095, 0.045, 15),
    (132.64, 34.19, 0.065, 0.035, -20),
    (132.78, 34.22, 0.095, 0.045, 10),
    (132.95, 34.24, 0.060, 0.035, 0),
    (133.10, 34.23, 0.085, 0.040, -10),

    # 广岛湾及周边
    (132.25, 34.05, 0.070, 0.035, 20),
    (132.45, 34.06, 0.090, 0.042, -10),
    (132.70, 34.08, 0.060, 0.035, 15),
    (132.92, 34.05, 0.070, 0.035, -10),
    (133.20, 34.08, 0.055, 0.030, 0),
    (133.43, 34.12, 0.080, 0.038, 10),

    # 香川、冈山沿岸岛屿
    (133.55, 34.34, 0.090, 0.040, 15),
    (133.72, 34.35, 0.060, 0.032, -10),
    (133.90, 34.34, 0.095, 0.042, 10),
    (134.10, 34.36, 0.075, 0.035, -15),
    (134.30, 34.36, 0.060, 0.030, 5),
    (134.48, 34.40, 0.070, 0.034, 0),

    # 播磨滩和淡路岛
    (134.78, 34.43, 0.130, 0.035, 15),
    (134.92, 34.55, 0.115, 0.035, 75),
    (135.00, 34.68, 0.070, 0.030, 70),
    (134.65, 34.58, 0.060, 0.030, 0),

    # 伊势湾、三河湾岛屿
    (136.72, 34.72, 0.070, 0.035, 10),
    (136.82, 34.67, 0.050, 0.028, -10),
    (136.92, 34.75, 0.060, 0.030, 0),
    (137.00, 34.63, 0.045, 0.025, 20),

    # 九州北部、壹岐和周边可见岛屿
    (129.68, 34.78, 0.095, 0.060, 10),
    (129.88, 34.70, 0.070, 0.040, -10),
    (130.18, 34.52, 0.060, 0.035, 15),
    (130.42, 34.43, 0.050, 0.030, -15),
]


def is_land(lon, lat):
    for polygon in LAND_POLYGONS:
        if point_in_polygon(lon, lat, polygon):
            return True

    for island in ISLANDS:
        if ellipse_contains(lon, lat, *island):
            return True

    return False


def build_matrix():
    matrix = np.zeros((ROWS, COLS), dtype=np.uint8)

    for r, lat in enumerate(latitudes):
        for c, lon in enumerate(longitudes):
            matrix[r, c] = 1 if is_land(float(lon), float(lat)) else 0

    return matrix


def validate_matrix(matrix):
    assert matrix.shape == (64, 253)
    assert matrix.size == 16192
    assert np.all((matrix == 0) | (matrix == 1))


def write_csv(matrix, path):
    with path.open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        for row in matrix:
            writer.writerow([int(value) for value in row])


def write_images(matrix, original_path, enlarged_path):
    # 0 为黑色水域，1 为白色陆地。
    pixels = np.where(matrix == 1, 255, 0).astype(np.uint8)

    image = Image.fromarray(pixels, mode="L")
    assert image.size == (253, 64)
    assert set(image.getdata()).issubset({0, 255})
    image.save(original_path, format="PNG")

    enlarged = image.resize((2024, 512), resample=Image.Resampling.NEAREST)
    assert enlarged.size == (2024, 512)
    assert set(enlarged.getdata()).issubset({0, 255})
    enlarged.save(enlarged_path, format="PNG")


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = build_matrix()
    validate_matrix(matrix)

    csv_path = output_dir / "candidate.csv"
    png_path = output_dir / "candidate.png"
    png_8x_path = output_dir / "candidate_8x.png"

    write_csv(matrix, csv_path)
    write_images(matrix, png_path, png_8x_path)

    # 重新检查 CSV 结构
    with csv_path.open("r", newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))

    assert len(rows) == 64
    assert all(len(row) == 253 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    print(f"matrix: {matrix.shape[0]} rows x {matrix.shape[1]} columns")
    print(f"values: {matrix.size}")
    print(f"land points: {int(matrix.sum())}")
    print(f"water points: {int(matrix.size - matrix.sum())}")
    print(f"csv: {csv_path}")
    print(f"png: {png_path} ({Image.open(png_path).size[0]}x{Image.open(png_path).size[1]})")
    print(
        f"png_8x: {png_8x_path} "
        f"({Image.open(png_8x_path).size[0]}x{Image.open(png_8x_path).size[1]})"
    )


if __name__ == "__main__":
    main()
