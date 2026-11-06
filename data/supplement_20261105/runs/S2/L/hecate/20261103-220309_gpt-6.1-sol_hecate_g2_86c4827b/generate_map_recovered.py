#!/usr/bin/env python3
from pathlib import Path
import csv

import numpy as np
from PIL import Image


N = 127
WEST = -131.0
EAST = -129.0
NORTH = 55.0
SOUTH = 53.0

OUTPUT_DIR = Path("outputs")


def point_on_segment(x, y, x1, y1, x2, y2, eps=1e-10):
    """判断点是否落在线段上。边界按陆地处理。"""
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > eps:
        return False

    return (
        min(x1, x2) - eps <= x <= max(x1, x2) + eps
        and min(y1, y2) - eps <= y <= max(y1, y2) + eps
    )


def point_in_polygon(x, y, polygon):
    """
    射线法判断点是否在多边形内部。
    多边形顶点为 (经度, 纬度)。
    边界点按陆地处理。
    """
    inside = False
    count = len(polygon)

    for i in range(count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]

        if point_on_segment(x, y, x1, y1, x2, y2):
            return True

        crosses = (y1 > y) != (y2 > y)
        if crosses:
            x_intersection = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_intersection:
                inside = not inside

    return inside


# ----------------------------------------------------------------------
# 手写近似地理规则
#
# 所有多边形均表示陆地。未被任意多边形覆盖的区域判为水域。
# 坐标为 (经度, 纬度)，经度使用西经负值。
# ----------------------------------------------------------------------

LAND_POLYGONS = [
    # 加拿大大陆北部海岸，框内保留其西侧不规则海岸线。
    [
        (-129.00, 53.00),
        (-129.00, 55.00),
        (-129.72, 55.00),
        (-129.82, 54.88),
        (-129.70, 54.70),
        (-129.83, 54.53),
        (-129.73, 54.34),
        (-129.55, 54.18),
        (-129.49, 54.00),
        (-129.36, 53.82),
        (-129.43, 53.61),
        (-129.30, 53.42),
        (-129.27, 53.22),
        (-129.20, 53.00),
    ],

    # 格雷厄姆岛：海达瓜依北岛。
    [
        (-131.00, 54.78),
        (-130.90, 54.91),
        (-130.73, 55.00),
        (-130.21, 55.00),
        (-130.10, 54.92),
        (-130.08, 54.78),
        (-130.16, 54.65),
        (-130.29, 54.57),
        (-130.33, 54.43),
        (-130.48, 54.37),
        (-130.65, 54.38),
        (-130.80, 54.48),
        (-130.94, 54.60),
    ],

    # 格雷厄姆岛东南部较窄的延伸。
    [
        (-130.58, 54.38),
        (-130.42, 54.33),
        (-130.28, 54.22),
        (-130.17, 54.10),
        (-130.20, 53.99),
        (-130.37, 53.99),
        (-130.50, 54.08),
        (-130.61, 54.21),
    ],

    # 莫尔斯比岛主体。
    [
        (-131.00, 54.07),
        (-130.86, 54.02),
        (-130.76, 53.91),
        (-130.67, 53.77),
        (-130.60, 53.59),
        (-130.49, 53.40),
        (-130.34, 53.24),
        (-130.18, 53.12),
        (-130.10, 53.00),
        (-130.75, 53.00),
        (-130.84, 53.11),
        (-130.93, 53.27),
        (-131.00, 53.46),
    ],

    # 莫尔斯比岛西南部。
    [
        (-131.00, 53.00),
        (-130.75, 53.00),
        (-130.84, 53.11),
        (-130.95, 53.18),
        (-131.00, 53.22),
    ],

    # 莫尔斯比岛东南部及其狭长海岸。
    [
        (-130.45, 53.42),
        (-130.29, 53.36),
        (-130.16, 53.25),
        (-130.08, 53.12),
        (-130.10, 53.00),
        (-130.34, 53.00),
        (-130.38, 53.14),
        (-130.48, 53.27),
    ],

    # 安东尼岛等南部小岛的近似轮廓。
    [
        (-131.00, 53.00),
        (-130.88, 53.00),
        (-130.90, 53.07),
        (-130.98, 53.12),
        (-131.00, 53.10),
    ],

    # 桑德斯岛及格雷厄姆岛西侧小岛。
    [
        (-130.98, 54.28),
        (-130.89, 54.34),
        (-130.80, 54.31),
        (-130.78, 54.22),
        (-130.84, 54.16),
        (-130.94, 54.18),
    ],

    # 迪克森入口附近的 Dundas Island 近似。
    [
        (-130.19, 54.93),
        (-130.10, 54.98),
        (-130.00, 54.96),
        (-129.96, 54.89),
        (-130.02, 54.84),
        (-130.13, 54.86),
    ],

    # 迪克森入口附近小岛。
    [
        (-130.44, 54.95),
        (-130.37, 55.00),
        (-130.29, 55.00),
        (-130.27, 54.94),
        (-130.34, 54.89),
    ],

    # 大陆近岸的 Pitt Island 南部近似。
    [
        (-129.96, 53.16),
        (-129.83, 53.29),
        (-129.69, 53.31),
        (-129.60, 53.20),
        (-129.62, 53.05),
        (-129.76, 53.00),
        (-129.90, 53.04),
    ],

    # 大陆近岸的 Pitt Island 北部近似。
    [
        (-130.02, 53.56),
        (-129.91, 53.69),
        (-129.77, 53.70),
        (-129.68, 53.58),
        (-129.72, 53.45),
        (-129.86, 53.42),
        (-129.98, 53.47),
    ],

    # Banks Island 北端在框内的近似轮廓。
    [
        (-130.72, 54.98),
        (-130.59, 55.00),
        (-130.45, 55.00),
        (-130.43, 54.94),
        (-130.53, 54.88),
        (-130.66, 54.90),
    ],

    # 大陆海岸附近的小型岛屿和礁盘。
    [
        (-129.75, 54.72),
        (-129.67, 54.79),
        (-129.57, 54.77),
        (-129.53, 54.69),
        (-129.61, 54.63),
        (-129.72, 54.64),
    ],
    [
        (-129.55, 54.23),
        (-129.48, 54.29),
        (-129.40, 54.25),
        (-129.38, 54.18),
        (-129.46, 54.14),
        (-129.53, 54.16),
    ],
    [
        (-129.89, 53.96),
        (-129.82, 54.03),
        (-129.73, 54.01),
        (-129.70, 53.93),
        (-129.77, 53.88),
        (-129.86, 53.90),
    ],
]


def is_land(lon, lat):
    """点判据：落入任一手写陆地多边形即为陆地。"""
    return any(point_in_polygon(lon, lat, polygon) for polygon in LAND_POLYGONS)


def build_matrix():
    """
    生成北到南、 西到东排列的 127 x 127 矩阵。
    1 = 陆地，0 = 水域。
    """
    matrix = np.zeros((N, N), dtype=np.uint8)

    for row in range(N):
        lat = NORTH - row * (NORTH - SOUTH) / (N - 1)

        for col in range(N):
            lon = WEST + col * (EAST - WEST) / (N - 1)
            matrix[row, col] = 1 if is_land(lon, lat) else 0

    return matrix


def write_csv(matrix, path):
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for row in matrix:
            writer.writerow(row.tolist())


def write_images(matrix):
    # 0 -> 黑色水域，1 -> 白色陆地。
    image_data = np.where(matrix == 1, 255, 0).astype(np.uint8)

    original = Image.fromarray(image_data, mode="L")
    original.save(OUTPUT_DIR / "candidate.png")

    enlarged = original.resize(
        (N * 8, N * 8),
        resample=Image.Resampling.NEAREST,
    )
    enlarged.save(OUTPUT_DIR / "candidate_8x.png")


def validate(matrix):
    assert matrix.shape == (127, 127)
    assert matrix.size == 16129
    assert np.all((matrix == 0) | (matrix == 1))

    csv_path = OUTPUT_DIR / "candidate.csv"
    with csv_path.open("r", newline="", encoding="ascii") as handle:
        rows = list(csv.reader(handle))

    assert len(rows) == 127
    assert all(len(row) == 127 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    original = Image.open(OUTPUT_DIR / "candidate.png")
    enlarged = Image.open(OUTPUT_DIR / "candidate_8x.png")

    assert original.size == (127, 127)
    assert enlarged.size == (1016, 1016)
    assert original.mode == "L"
    assert enlarged.mode == "L"

    original_values = set(original.getdata())
    enlarged_values = set(enlarged.getdata())
    assert original_values <= {0, 255}
    assert enlarged_values <= {0, 255}

    # 最近邻放大验证：每个 8x8 块必须完全相同。
    enlarged_array = np.asarray(enlarged)
    expected = np.repeat(np.repeat(np.asarray(original), 8, axis=0), 8, axis=1)
    assert np.array_equal(enlarged_array, expected)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matrix = build_matrix()
    write_csv(matrix, OUTPUT_DIR / "candidate.csv")
    write_images(matrix)
    validate(matrix)

    print("Generated:")
    print("  outputs/candidate.csv       127 x 127")
    print("  outputs/candidate.png      127 x 127")
    print("  outputs/candidate_8x.png   1016 x 1016")
    print("  valid values: 16129")
    print("  land points:", int(matrix.sum()))
    print("  water points:", int(matrix.size - matrix.sum()))


if __name__ == "__main__":
    main()
