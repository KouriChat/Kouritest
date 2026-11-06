from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image


# ============================================================
# 北极航道：加拿大北极群岛核心区的知识近似陆地规则
#
# 坐标：
#   longitude: 105W ... 81W，递增，即 -105.0 ... -81.0
#   latitude:  79N ... 73N，递减，即 79.0 ... 73.0
#
# 本文件不联网，不读取地图或地理数据库。
# 陆地由手工近似多边形和椭圆岛屿组成。
# 点落入这些区域记为 1，否则记为 0。
# ============================================================


# 多边形顶点采用 (longitude, latitude)
# 顶点顺序可为顺时针或逆时针。
LAND_POLYGONS = [
    # Ellesmere Island southern and central mass.
    [
        (-105.0, 77.20),
        (-103.5, 77.65),
        (-101.0, 78.10),
        (-98.0, 78.40),
        (-95.0, 78.20),
        (-92.0, 77.80),
        (-89.0, 77.35),
        (-86.0, 76.95),
        (-83.3, 76.45),
        (-81.0, 76.05),
        (-81.0, 79.0),
        (-105.0, 79.0),
    ],

    # Western Ellesmere / Axel Heiberg-connected high land.
    [
        (-105.0, 78.00),
        (-103.0, 78.65),
        (-100.0, 79.0),
        (-100.0, 78.25),
        (-101.0, 77.65),
        (-103.0, 77.25),
        (-105.0, 77.10),
    ],

    # Axel Heiberg Island.
    [
        (-97.8, 78.65),
        (-95.8, 78.85),
        (-93.5, 78.65),
        (-92.0, 78.10),
        (-92.4, 77.55),
        (-94.2, 77.30),
        (-96.1, 77.55),
        (-97.7, 78.00),
    ],

    # Devon Island, western and central body.
    [
        (-100.1, 75.25),
        (-98.0, 76.00),
        (-95.4, 76.35),
        (-92.0, 76.20),
        (-89.4, 75.80),
        (-86.8, 75.15),
        (-84.0, 74.55),
        (-83.0, 73.85),
        (-85.0, 73.40),
        (-88.0, 73.20),
        (-92.0, 73.28),
        (-96.0, 73.55),
        (-98.7, 74.10),
        (-100.2, 74.70),
    ],

    # Devon Island northern extension.
    [
        (-98.5, 76.05),
        (-96.0, 76.65),
        (-92.5, 76.55),
        (-89.3, 76.05),
        (-87.8, 75.55),
        (-90.0, 75.35),
        (-93.2, 75.50),
        (-96.3, 75.40),
    ],

    # Prince of Wales Island.
    [
        (-101.0, 74.20),
        (-99.4, 74.80),
        (-96.8, 75.00),
        (-94.0, 74.75),
        (-91.5, 74.15),
        (-90.2, 73.35),
        (-91.0, 72.95),
        (-94.0, 72.75),
        (-97.0, 72.95),
        (-99.5, 73.35),
        (-101.0, 73.75),
    ],

    # Somerset Island.
    [
        (-95.8, 73.95),
        (-93.7, 74.25),
        (-91.2, 74.10),
        (-89.0, 73.60),
        (-87.0, 73.05),
        (-86.5, 72.85),
        (-89.2, 72.65),
        (-92.0, 72.68),
        (-94.4, 72.95),
        (-95.9, 73.35),
    ],

    # Northern Baffin Island, entering from the eastern frame.
    [
        (-83.0, 75.25),
        (-81.0, 75.55),
        (-81.0, 73.0),
        (-84.2, 73.0),
        (-85.1, 73.55),
        (-84.8, 74.25),
    ],

    # Baffin Island northwestern projecting coast.
    [
        (-86.0, 74.90),
        (-84.5, 75.10),
        (-82.8, 74.80),
        (-82.0, 74.20),
        (-83.4, 73.75),
        (-85.0, 73.85),
        (-86.0, 74.30),
    ],

    # Bathurst Island.
    [
        (-98.6, 76.70),
        (-96.9, 77.10),
        (-94.8, 77.05),
        (-93.3, 76.65),
        (-94.0, 76.15),
        (-96.0, 75.95),
        (-98.0, 76.15),
    ],

    # Cornwallis Island.
    [
        (-95.7, 75.55),
        (-93.9, 75.75),
        (-92.1, 75.45),
        (-91.4, 74.95),
        (-92.6, 74.65),
        (-94.5, 74.75),
        (-95.8, 75.05),
    ],

    # Little Cornwallis Island.
    [
        (-95.7, 75.05),
        (-94.6, 75.25),
        (-93.5, 75.12),
        (-93.0, 74.80),
        (-94.0, 74.58),
        (-95.2, 74.65),
    ],

    # King William Island, only its northern portion is inside the frame.
    [
        (-99.0, 73.45),
        (-97.2, 73.90),
        (-95.2, 73.65),
        (-94.2, 73.05),
        (-94.5, 72.90),
        (-97.5, 72.90),
        (-99.0, 73.10),
    ],

    # North Somerset / Boothia Peninsula northern edge.
    [
        (-92.0, 73.30),
        (-90.4, 73.55),
        (-88.7, 73.35),
        (-87.5, 73.00),
        (-87.2, 72.85),
        (-89.8, 72.82),
        (-91.4, 72.95),
    ],
]


# 手工近似的小岛、礁盘和岛屿碎片。
# 每项为 (中心经度, 中心纬度, 半经度, 半纬度)。
LAND_ELLIPSES = [
    # Ellesmere/Axel Heiberg vicinity
    (-100.4, 78.00, 0.45, 0.24),
    (-98.8, 77.55, 0.35, 0.20),
    (-91.0, 77.05, 0.40, 0.22),
    (-88.4, 76.60, 0.38, 0.20),
    (-84.4, 76.25, 0.28, 0.18),

    # Parry Islands and central Arctic island fragments
    (-99.7, 76.55, 0.26, 0.17),
    (-98.4, 76.10, 0.30, 0.18),
    (-91.0, 76.55, 0.28, 0.17),
    (-89.8, 75.95, 0.32, 0.18),
    (-87.0, 75.60, 0.28, 0.16),
    (-84.8, 75.30, 0.25, 0.15),

    # Islands around Prince of Wales / Somerset
    (-101.9, 74.00, 0.35, 0.18),
    (-100.5, 73.45, 0.25, 0.15),
    (-97.4, 74.20, 0.22, 0.14),
    (-95.0, 74.15, 0.24, 0.14),
    (-89.2, 74.25, 0.30, 0.17),
    (-87.6, 73.65, 0.25, 0.14),

    # Islands near the Baffin side of the passage
    (-85.6, 74.80, 0.28, 0.16),
    (-83.5, 74.25, 0.22, 0.14),
    (-82.2, 73.75, 0.26, 0.15),

    # Small islands in the southern part of the sampled frame
    (-98.7, 73.15, 0.23, 0.13),
    (-96.2, 73.30, 0.20, 0.12),
    (-92.8, 73.05, 0.20, 0.12),
]


def point_in_polygon(x, y, polygon):
    """Ray-casting point-in-polygon test."""
    inside = False
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # Boundary points are classified as land.
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) < 1e-12:
            if (
                min(x1, x2) - 1e-12 <= x <= max(x1, x2) + 1e-12
                and min(y1, y2) - 1e-12 <= y <= max(y1, y2) + 1e-12
            ):
                return True

        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside

    return inside


def point_in_ellipse(x, y, cx, cy, rx, ry):
    """Approximate small island or reef as an ellipse."""
    if rx <= 0.0 or ry <= 0.0:
        return False
    value = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
    return value <= 1.0


def is_land(longitude, latitude):
    """
    Point判据：只根据当前经纬度点是否落在手写陆地区域中判断。
    不使用邻近采样点，也不进行网格面积统计。
    """
    for polygon in LAND_POLYGONS:
        if point_in_polygon(longitude, latitude, polygon):
            return 1

    for cx, cy, rx, ry in LAND_ELLIPSES:
        if point_in_ellipse(longitude, latitude, cx, cy, rx, ry):
            return 1

    return 0


def build_matrix():
    rows = 61
    cols = 241

    # 避免浮点数逐次累加造成经纬度漂移，使用整数索引计算坐标。
    matrix = np.zeros((rows, cols), dtype=np.uint8)

    for row in range(rows):
        latitude = 79.0 - row * 0.1

        for col in range(cols):
            longitude = -105.0 + col * 0.1
            matrix[row, col] = is_land(longitude, latitude)

    return matrix


def validate_matrix(matrix):
    assert matrix.shape == (61, 241)
    assert matrix.size == 14701
    assert matrix.dtype == np.uint8
    assert np.all((matrix == 0) | (matrix == 1))


def write_csv(matrix, path):
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for row in matrix:
            writer.writerow([str(int(value)) for value in row])


def write_images(matrix, original_path, enlarged_path):
    # 0 -> 黑色水域，1 -> 白色陆地。
    image_array = matrix * 255
    image = Image.fromarray(image_array.astype(np.uint8), mode="L")

    assert image.size == (241, 61)
    assert set(image.getdata()).issubset({0, 255})
    image.save(original_path, format="PNG")

    enlarged = image.resize((241 * 8, 61 * 8), resample=Image.Resampling.NEAREST)

    assert enlarged.size == (1928, 488)
    assert set(enlarged.getdata()).issubset({0, 255})
    enlarged.save(enlarged_path, format="PNG")


def validate_csv(path):
    with path.open("r", newline="", encoding="ascii") as handle:
        rows = list(csv.reader(handle))

    assert len(rows) == 61
    assert all(len(row) == 241 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert all(value != "" for row in rows for value in row)


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = build_matrix()
    validate_matrix(matrix)

    csv_path = output_dir / "candidate.csv"
    original_png = output_dir / "candidate.png"
    enlarged_png = output_dir / "candidate_8x.png"

    write_csv(matrix, csv_path)
    write_images(matrix, original_png, enlarged_png)
    validate_csv(csv_path)

    print("Generated:")
    print(f"  {csv_path}       : 61 rows x 241 columns")
    print(f"  {original_png}   : 241 x 61 PNG")
    print(f"  {enlarged_png}   : 1928 x 488 PNG, nearest-neighbor")
    print(f"  valid values     : {matrix.size}")
    print(f"  land points      : {int(matrix.sum())}")
    print(f"  water points     : {int(matrix.size - matrix.sum())}")


if __name__ == "__main__":
    main()
