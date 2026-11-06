#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image


# ----------------------------------------------------------------------
# 手写近似地理轮廓
# 坐标顺序为 (longitude, latitude)
# ----------------------------------------------------------------------

land_polygons = []
water_holes = []


def add_polygon(points):
    land_polygons.append([(float(x), float(y)) for x, y in points])


def add_ellipse(cx, cy, rx, ry, n=20):
    points = []
    for i in range(n):
        angle = 2.0 * math.pi * i / n
        points.append((
            cx + rx * math.cos(angle),
            cy + ry * math.sin(angle),
        ))
    add_polygon(points)


# 中国东南沿海及朝鲜半岛东南角的极小可见部分。
# 这里只覆盖采样框内可能出现的区域。
add_polygon([
    (128.90, 35.00),
    (129.10, 35.04),
    (129.22, 35.12),
    (129.34, 35.25),
    (129.55, 35.45),
    (130.00, 35.55),
    (130.00, 35.00),
])


# 九州北部
add_polygon([
    (129.52, 33.00),
    (129.58, 33.16),
    (129.72, 33.30),
    (129.90, 33.45),
    (130.08, 33.60),
    (130.28, 33.76),
    (130.48, 33.88),
    (130.68, 33.96),
    (130.88, 33.98),
    (131.02, 33.90),
    (131.12, 33.75),
    (131.25, 33.58),
    (131.45, 33.48),
    (131.68, 33.37),
    (131.87, 33.25),
    (132.02, 33.08),
    (132.12, 32.55),
    (129.35, 32.55),
])


# 中国地方南岸及山口、广岛、冈山、兵库南部
add_polygon([
    (130.72, 33.91),
    (130.92, 33.93),
    (131.12, 33.97),
    (131.32, 34.03),
    (131.52, 34.09),
    (131.70, 34.14),
    (131.90, 34.16),
    (132.08, 34.15),
    (132.24, 34.21),
    (132.42, 34.27),
    (132.62, 34.31),
    (132.84, 34.35),
    (133.04, 34.39),
    (133.25, 34.43),
    (133.45, 34.47),
    (133.66, 34.51),
    (133.87, 34.55),
    (134.08, 34.58),
    (134.29, 34.62),
    (134.50, 34.66),
    (134.72, 34.69),
    (134.94, 34.72),
    (135.08, 34.76),
    (135.12, 35.45),
    (130.68, 35.45),
])


# 近畿、纪伊半岛、伊势湾西岸及本州中部
# 轮廓沿大阪湾、纪伊半岛、伊势湾外缘和远州灘近似描绘。
add_polygon([
    (134.94, 34.72),
    (135.12, 34.71),
    (135.32, 34.69),
    (135.48, 34.64),
    (135.53, 34.56),
    (135.43, 34.47),
    (135.27, 34.34),
    (135.10, 34.15),
    (135.03, 33.98),
    (135.08, 33.82),
    (135.25, 33.66),
    (135.45, 33.54),
    (135.66, 33.46),
    (135.84, 33.47),
    (136.00, 33.57),
    (136.14, 33.75),
    (136.28, 33.94),
    (136.43, 34.12),
    (136.57, 34.28),
    (136.69, 34.42),
    (136.74, 34.59),
    (136.78, 34.77),
    (136.85, 34.93),
    (136.96, 35.05),

    # 伊势湾东岸、知多半岛及远州灘方向
    (137.05, 35.08),
    (137.15, 34.99),
    (137.19, 34.87),
    (137.29, 34.74),
    (137.47, 34.69),
    (137.65, 34.76),
    (137.84, 34.90),
    (138.02, 35.18),
    (138.05, 35.50),
    (134.90, 35.50),
])


# 四国
add_polygon([
    (131.94, 33.96),
    (132.08, 34.04),
    (132.25, 34.13),
    (132.47, 34.21),
    (132.70, 34.28),
    (132.94, 34.32),
    (133.18, 34.35),
    (133.43, 34.35),
    (133.67, 34.32),
    (133.88, 34.23),
    (134.08, 34.16),
    (134.29, 34.15),
    (134.48, 34.22),
    (134.65, 34.18),
    (134.70, 34.04),
    (134.61, 33.90),
    (134.48, 33.76),
    (134.30, 33.62),
    (134.08, 33.50),
    (133.85, 33.39),
    (133.60, 33.30),
    (133.32, 33.25),
    (133.05, 33.25),
    (132.80, 33.31),
    (132.55, 33.42),
    (132.33, 33.55),
    (132.15, 33.68),
    (132.02, 33.82),
    (131.92, 33.96),
])


# 淡路岛
add_polygon([
    (134.62, 34.20),
    (134.72, 34.27),
    (134.83, 34.39),
    (134.91, 34.52),
    (134.96, 34.65),
    (134.91, 34.70),
    (134.80, 34.64),
    (134.72, 34.53),
    (134.65, 34.39),
    (134.60, 34.27),
])


# 琵琶湖。它位于较大的本州陆地多边形内部，因此作为水域孔洞处理。
water_holes.append([
    (135.82, 35.02),
    (135.84, 35.16),
    (135.91, 35.31),
    (136.02, 35.43),
    (136.13, 35.46),
    (136.18, 35.35),
    (136.15, 35.19),
    (136.08, 35.05),
    (135.98, 34.99),
    (135.88, 34.99),
])


# ----------------------------------------------------------------------
# 主要离岛
# ----------------------------------------------------------------------

# 对马岛
add_polygon([
    (129.15, 34.12),
    (129.20, 34.25),
    (129.18, 34.39),
    (129.23, 34.55),
    (129.31, 34.68),
    (129.43, 34.76),
    (129.53, 34.69),
    (129.55, 34.54),
    (129.49, 34.39),
    (129.43, 34.25),
    (129.34, 34.13),
    (129.23, 34.08),
])


# 壹岐岛
add_polygon([
    (129.62, 33.72),
    (129.72, 33.68),
    (129.84, 33.72),
    (129.91, 33.82),
    (129.89, 33.94),
    (129.80, 34.02),
    (129.68, 34.00),
    (129.59, 33.91),
    (129.57, 33.80),
])


# 平户岛及附近岛屿
add_ellipse(129.47, 33.37, 0.09, 0.15, 18)
add_ellipse(129.56, 33.39, 0.045, 0.07, 14)
add_ellipse(129.67, 33.45, 0.055, 0.045, 14)
add_ellipse(129.76, 33.50, 0.035, 0.04, 12)


# 福冈湾附近
add_ellipse(130.31, 33.66, 0.08, 0.055, 16)   # 能古岛
add_ellipse(130.29, 33.70, 0.075, 0.055, 16)   # 志贺岛
add_ellipse(130.55, 33.98, 0.055, 0.045, 14)   # 蓝岛
add_ellipse(130.75, 34.01, 0.035, 0.04, 12)


# 山口、广岛湾和周防灘
add_ellipse(131.69, 33.73, 0.09, 0.10, 18)    # 姬岛
add_ellipse(131.88, 33.82, 0.045, 0.045, 14)
add_ellipse(132.04, 33.96, 0.06, 0.05, 14)
add_ellipse(132.20, 33.94, 0.16, 0.08, 18)    # 周防大岛
add_ellipse(132.38, 34.05, 0.075, 0.055, 14)
add_ellipse(132.29, 34.28, 0.085, 0.035, 16)   # 宫岛
add_ellipse(132.43, 34.26, 0.15, 0.10, 18)    # 江田岛
add_ellipse(132.37, 34.34, 0.11, 0.065, 16)   # 能美岛
add_ellipse(132.52, 34.14, 0.12, 0.07, 16)    # 仓桥岛
add_ellipse(132.66, 34.17, 0.06, 0.045, 14)


# 芸予诸岛
add_ellipse(132.82, 34.24, 0.07, 0.07, 16)
add_ellipse(132.91, 34.29, 0.09, 0.065, 16)   # 大崎上岛
add_ellipse(132.93, 34.19, 0.08, 0.055, 16)   # 大崎下岛
add_ellipse(133.02, 34.19, 0.085, 0.11, 18)   # 大三岛
add_ellipse(133.07, 34.08, 0.10, 0.12, 18)    # 伯方岛
add_ellipse(133.01, 34.00, 0.07, 0.09, 16)    # 大岛
add_ellipse(132.95, 34.06, 0.045, 0.035, 12)  # 来岛
add_ellipse(133.17, 34.29, 0.10, 0.095, 16)   # 因岛
add_ellipse(133.20, 34.38, 0.10, 0.06, 16)    # 向岛
add_ellipse(133.34, 34.27, 0.07, 0.06, 14)
add_ellipse(133.46, 34.24, 0.065, 0.05, 14)
add_ellipse(133.58, 34.29, 0.06, 0.05, 14)


# 香川、冈山外海
add_ellipse(133.62, 34.15, 0.07, 0.055, 14)
add_ellipse(133.70, 34.26, 0.055, 0.045, 14)
add_ellipse(133.78, 34.34, 0.055, 0.045, 14)
add_ellipse(133.88, 34.38, 0.06, 0.05, 14)
add_ellipse(133.96, 34.45, 0.07, 0.055, 16)   # 直岛
add_ellipse(134.02, 34.42, 0.05, 0.045, 14)   # 男木岛
add_ellipse(134.07, 34.48, 0.06, 0.05, 14)    # 女木岛
add_ellipse(134.10, 34.56, 0.07, 0.055, 14)   # 犬岛
add_ellipse(134.18, 34.42, 0.05, 0.045, 14)
add_ellipse(134.32, 34.43, 0.055, 0.045, 14)
add_ellipse(134.42, 34.45, 0.055, 0.045, 14)


# 小豆岛
add_polygon([
    (134.04, 34.47),
    (134.14, 34.42),
    (134.28, 34.40),
    (134.41, 34.46),
    (134.48, 34.56),
    (134.41, 34.65),
    (134.26, 34.68),
    (134.11, 34.62),
    (134.03, 34.53),
])


# 纪伊水道、淡路岛南侧和德岛外海
add_ellipse(134.47, 34.06, 0.05, 0.04, 14)
add_ellipse(134.58, 34.00, 0.045, 0.04, 14)
add_ellipse(134.72, 34.02, 0.045, 0.035, 12)
add_ellipse(134.83, 34.10, 0.05, 0.04, 12)


# 伊势湾和三河湾中的岛屿
add_ellipse(136.84, 34.70, 0.045, 0.04, 14)
add_ellipse(136.91, 34.72, 0.05, 0.045, 14)
add_ellipse(136.96, 34.76, 0.045, 0.04, 14)
add_ellipse(136.86, 34.50, 0.045, 0.04, 14)
add_ellipse(136.93, 34.53, 0.04, 0.035, 12)
add_ellipse(137.04, 34.60, 0.05, 0.04, 14)


# ----------------------------------------------------------------------
# 点在多边形内判断
# ----------------------------------------------------------------------

def prepare_polygons(polygons):
    prepared = []
    for polygon in polygons:
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        prepared.append((
            polygon,
            min(xs),
            max(xs),
            min(ys),
            max(ys),
        ))
    return prepared


prepared_land = prepare_polygons(land_polygons)
prepared_holes = prepare_polygons(water_holes)


def point_in_polygon(x, y, polygon):
    inside = False
    n = len(polygon)

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        crosses = ((yi > y) != (yj > y))
        if crosses:
            denominator = yj - yi
            if denominator != 0.0:
                x_cross = (xj - xi) * (y - yi) / denominator + xi
                if x < x_cross:
                    inside = not inside

        j = i

    return inside


def point_is_land(x, y):
    in_land = False

    for polygon, xmin, xmax, ymin, ymax in prepared_land:
        if x < xmin or x > xmax or y < ymin or y > ymax:
            continue
        if point_in_polygon(x, y, polygon):
            in_land = True
            break

    if not in_land:
        return False

    for polygon, xmin, xmax, ymin, ymax in prepared_holes:
        if x < xmin or x > xmax or y < ymin or y > ymax:
            continue
        if point_in_polygon(x, y, polygon):
            return False

    return True


# ----------------------------------------------------------------------
# 生成 64 x 253 点判据矩阵
# ----------------------------------------------------------------------

ROWS = 64
COLS = 253
STEP = 2.0 / 63.0

latitudes = [35.0 - row * STEP for row in range(ROWS)]
longitudes = [129.0 + col * STEP for col in range(COLS)]

matrix = np.zeros((ROWS, COLS), dtype=np.uint8)

for row, latitude in enumerate(latitudes):
    for col, longitude in enumerate(longitudes):
        matrix[row, col] = 1 if point_is_land(longitude, latitude) else 0


# ----------------------------------------------------------------------
# 输出
# ----------------------------------------------------------------------

output_dir = Path.cwd() / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

csv_path = output_dir / "candidate.csv"
raw_png_path = output_dir / "candidate.png"
large_png_path = output_dir / "candidate_8x.png"


with csv_path.open("w", encoding="ascii", newline="") as f:
    writer = csv.writer(f, lineterminator="\n")
    for row in matrix:
        writer.writerow([str(int(value)) for value in row])


# 黑色为水域，白色为陆地
raw_pixels = np.where(matrix == 1, 255, 0).astype(np.uint8)
raw_image = Image.fromarray(raw_pixels, mode="L")
raw_image.save(raw_png_path)


large_image = raw_image.resize(
    (COLS * 8, ROWS * 8),
    resample=Image.Resampling.NEAREST,
)
large_image.save(large_png_path)


# ----------------------------------------------------------------------
# 验证
# ----------------------------------------------------------------------

assert matrix.shape == (64, 253)
assert matrix.size == 64 * 253
assert set(np.unique(matrix).tolist()).issubset({0, 1})

with csv_path.open("r", encoding="ascii", newline="") as f:
    csv_rows = list(csv.reader(f))

assert len(csv_rows) == 64
assert all(len(row) == 253 for row in csv_rows)
assert all(value in {"0", "1"} for row in csv_rows for value in row)
assert all(value != "" for row in csv_rows for value in row)

with Image.open(raw_png_path) as image:
    assert image.size == (253, 64)
    raw_check = np.asarray(image.convert("L"))
    assert set(np.unique(raw_check).tolist()).issubset({0, 255})

with Image.open(large_png_path) as image:
    assert image.size == (2024, 512)
    large_check = np.asarray(image.convert("L"))
    assert set(np.unique(large_check).tolist()).issubset({0, 255})

expected_large = np.repeat(np.repeat(raw_pixels, 8, axis=0), 8, axis=1)
assert np.array_equal(large_check, expected_large)

print("生成完成：")
print(f"  CSV:       {csv_path}")
print(f"  原始 PNG:  {raw_png_path}  ({COLS}x{ROWS})")
print(f"  放大 PNG:  {large_png_path}  ({COLS * 8}x{ROWS * 8})")
print(f"  矩阵尺寸:  {matrix.shape}")
print(f"  有效值数:  {matrix.size}")
print("  验证通过：CSV 仅含 0/1，PNG 仅含黑白，放大采用最近邻。")
