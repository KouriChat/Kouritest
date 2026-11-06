#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os

import numpy as np
from PIL import Image


# 近似多边形点为 (经度, 纬度)。
# 多边形仅用于凭已有地理知识重建主要大陆和岛屿的大致范围。
POLYGONS = [
    # 北美洲主体：阿拉斯加、加拿大、美国、墨西哥及中美洲的粗略外轮廓
    [
        (-168, 71), (-155, 72), (-140, 70), (-130, 69), (-120, 70),
        (-105, 73), (-90, 72), (-75, 66), (-60, 62), (-53, 55),
        (-58, 48), (-67, 45), (-75, 44), (-80, 40), (-82, 32),
        (-88, 29), (-95, 27), (-100, 24), (-105, 20), (-108, 16),
        (-102, 14), (-96, 16), (-91, 18), (-88, 20), (-86, 17),
        (-83, 12), (-80, 9), (-77, 8), (-78, 15), (-82, 21),
        (-88, 22), (-94, 20), (-103, 25), (-112, 31), (-117, 33),
        (-121, 38), (-124, 45), (-130, 51), (-140, 57), (-151, 60),
        (-162, 63),
    ],

    # 格陵兰岛
    [
        (-73, 60), (-64, 59), (-50, 60), (-42, 66), (-35, 72),
        (-26, 78), (-30, 84), (-44, 86), (-58, 84), (-68, 78),
        (-73, 70),
    ],

    # 南美洲
    [
        (-81, 12), (-75, 11), (-70, 9), (-65, 10), (-60, 8),
        (-52, 5), (-48, 1), (-44, -3), (-39, -8), (-35, -13),
        (-35, -20), (-38, -26), (-43, -31), (-48, -35), (-53, -40),
        (-60, -46), (-66, -52), (-70, -55), (-75, -50), (-74, -43),
        (-71, -35), (-70, -25), (-72, -15), (-75, -5), (-80, 2),
    ],

    # 非洲
    [
        (-17, 15), (-13, 22), (-7, 28), (2, 35), (12, 37),
        (24, 34), (33, 31), (37, 23), (43, 12), (50, 3),
        (46, -5), (42, -12), (38, -20), (34, -28), (28, -34),
        (18, -35), (8, -35), (0, -30), (-7, -20), (-13, -8),
        (-17, 3),
    ],

    # 欧亚大陆北部和主体
    [
        (-10, 36), (0, 43), (10, 44), (20, 42), (30, 43),
        (40, 47), (52, 45), (62, 46), (72, 48), (85, 50),
        (100, 52), (112, 54), (125, 52), (137, 49), (150, 52),
        (163, 59), (174, 63), (180, 68), (170, 72), (155, 72),
        (140, 70), (125, 72), (110, 70), (95, 70), (80, 70),
        (65, 69), (50, 68), (35, 69), (20, 68), (10, 65),
        (0, 60), (-8, 55), (-12, 48), (-10, 42),
    ],

    # 欧洲西部和南部的补充轮廓
    [
        (-10, 43), (-5, 49), (2, 51), (8, 54), (15, 55),
        (23, 53), (28, 48), (25, 43), (20, 40), (17, 36),
        (10, 36), (4, 38), (-2, 37),
    ],

    # 中东、阿拉伯半岛和南亚
    [
        (30, 35), (38, 37), (48, 35), (57, 31), (60, 25),
        (68, 25), (75, 30), (80, 28), (88, 27), (93, 22),
        (90, 16), (84, 9), (77, 8), (73, 16), (67, 22),
        (58, 18), (50, 14), (43, 13), (39, 20), (32, 24),
    ],

    # 东南亚大陆
    [
        (93, 22), (101, 25), (106, 22), (109, 17), (106, 12),
        (103, 8), (106, 3), (101, 1), (98, 5), (96, 12),
        (91, 17),
    ],

    # 中国南部、朝鲜半岛补充
    [
        (95, 28), (104, 30), (112, 34), (122, 39), (130, 43),
        (135, 39), (130, 33), (123, 27), (117, 22), (108, 20),
        (100, 22),
    ],

    # 澳大利亚
    [
        (113, -14), (121, -12), (132, -12), (142, -11), (151, -16),
        (154, -24), (153, -31), (148, -38), (138, -39), (128, -36),
        (119, -34), (114, -28), (112, -22),
    ],

    # 马达加斯加
    [
        (43, -12), (50, -13), (50, -22), (47, -27),
        (44, -25), (43, -18),
    ],

    # 新西兰北岛和南岛
    [
        (172, -35), (178, -37), (177, -42), (171, -41),
        (166, -45), (168, -48), (174, -47), (179, -43),
        (180, -38),
    ],

    # 日本列岛粗略范围
    [
        (130, 31), (134, 33), (139, 35), (143, 39),
        (146, 43), (145, 46), (140, 45), (135, 40),
    ],

    # 斯里兰卡
    [
        (80, 10), (82, 8), (81, 5), (79, 7),
    ],

    # 英国和爱尔兰
    [
        (-11, 51), (-6, 50), (1, 52), (2, 56),
        (-3, 59), (-6, 58), (-8, 54),
    ],

    # 冰岛
    [
        (-25, 63), (-14, 63), (-13, 67), (-20, 67),
    ],

    # 菲律宾主要岛群
    [
        (118, 19), (123, 19), (126, 15), (125, 9),
        (121, 6), (118, 10), (119, 15),
    ],

    # 婆罗洲
    [
        (109, 7), (119, 7), (119, 1), (114, -4),
        (109, -3), (108, 2),
    ],

    # 苏门答腊
    [
        (95, 5), (105, 4), (106, -2), (101, -6),
        (97, -4),
    ],

    # 爪哇
    [
        (105, -6), (114, -6), (114, -9), (106, -8),
    ],

    # 新几内亚
    [
        (130, -2), (141, 0), (153, -2), (154, -8),
        (145, -10), (137, -7), (130, -8),
    ],

    # 海南岛
    [
        (108, 20), (111, 20), (111, 18), (109, 18),
    ],

    # 台湾岛
    [
        (120, 25), (122, 25), (122, 22), (120, 21),
    ],
]


# 一些面积较小但在 2 度采样中可能出现的岛屿或岛群。
# 使用中心点和半径的椭圆近似。
ELLIPSES = [
    # 夏威夷
    (-155, 20, 3.0, 1.2),
    (-157, 21, 1.4, 0.8),

    # 加勒比海主要岛屿
    (-75, 20, 3.8, 1.5),   # 古巴
    (-70, 19, 1.8, 1.2),   # 伊斯帕尼奥拉岛
    (-66, 18, 1.0, 1.0),    # 波多黎各
    (-61, 15, 1.0, 1.8),    # 小安的列斯群岛近似

    # 菲律宾外岛和印尼岛群
    (124, 5, 2.5, 4.0),
    (116, -1, 3.0, 4.0),
    (120, -5, 4.0, 2.0),
    (123, -3, 2.0, 4.0),
    (134, -4, 3.0, 2.0),

    # 日本北部、千岛群岛
    (145, 44, 2.0, 5.0),
    (151, 48, 1.5, 5.0),

    # 地中海大岛
    (9, 40, 2.0, 1.5),      # 撒丁岛
    (14, 38, 2.0, 1.0),     # 西西里岛
    (25, 35, 2.5, 1.5),     # 克里特岛
    (34, 36, 2.0, 1.0),     # 塞浦路斯

    # 非洲附近岛屿
    (48, -21, 2.0, 1.5),    # 马斯克林群岛近似
    (55, -21, 1.2, 1.0),

    # 南大西洋、南印度洋的大岛
    (-5, 35, 1.5, 1.0),
    (-17, 28, 1.5, 1.0),     # 加那利群岛近似
]


def point_in_polygon(lon, lat, polygon):
    """射线法判断点是否位于多边形内。"""
    inside = False
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        crosses = ((y1 > lat) != (y2 > lat))
        if crosses:
            x_intersection = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
            if lon < x_intersection:
                inside = not inside

    return inside


def in_ellipse(lon, lat, cx, cy, rx, ry):
    return ((lon - cx) / rx) ** 2 + ((lat - cy) / ry) ** 2 <= 1.0


def is_land(lon, lat):
    # 南极大陆及其陆地冰盖。南极半岛向北延伸至约 60S。
    if lat <= -72:
        return True

    antarctica = [
        (-180, -70), (-160, -68), (-140, -67), (-120, -68),
        (-100, -69), (-80, -70), (-65, -69), (-60, -64),
        (-58, -60), (-70, -63), (-90, -65), (-110, -66),
        (-130, -65), (-150, -66), (-180, -68),
    ]
    if point_in_polygon(lon, lat, antarctica):
        return True

    for polygon in POLYGONS:
        if point_in_polygon(lon, lat, polygon):
            return True

    for ellipse in ELLIPSES:
        if in_ellipse(lon, lat, *ellipse):
            return True

    return False


def build_matrix():
    latitudes = list(range(89, -90, -2))
    longitudes = list(range(-179, 180, 2))

    matrix = []
    for lat in latitudes:
        row = [1 if is_land(lon, lat) else 0 for lon in longitudes]
        matrix.append(row)

    matrix = np.asarray(matrix, dtype=np.uint8)

    assert matrix.shape == (90, 180)
    assert matrix.size == 16200
    assert np.all((matrix == 0) | (matrix == 1))

    return matrix


def save_csv(matrix, path):
    with open(path, "w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        for row in matrix:
            writer.writerow(row.tolist())


def save_images(matrix, raw_path, enlarged_path):
    # 0 -> 黑色水域，1 -> 白色陆地。
    pixels = np.where(matrix == 1, 255, 0).astype(np.uint8)

    raw = Image.fromarray(pixels, mode="L")
    assert raw.size == (180, 90)
    raw.save(raw_path, format="PNG")

    enlarged = raw.resize((1440, 720), resample=Image.Resampling.NEAREST)
    assert enlarged.size == (1440, 720)
    enlarged.save(enlarged_path, format="PNG")

    # 验证图片只有黑白两种像素值。
    values = set(np.asarray(raw).ravel().tolist())
    assert values.issubset({0, 255})

    enlarged_values = set(np.asarray(enlarged).ravel().tolist())
    assert enlarged_values.issubset({0, 255})


def main():
    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    matrix = build_matrix()

    csv_path = os.path.join(output_dir, "candidate.csv")
    raw_png_path = os.path.join(output_dir, "candidate.png")
    enlarged_png_path = os.path.join(output_dir, "candidate_8x.png")

    save_csv(matrix, csv_path)
    save_images(matrix, raw_png_path, enlarged_png_path)

    with open(csv_path, "r", encoding="ascii") as f:
        rows = list(csv.reader(f))

    assert len(rows) == 90
    assert all(len(row) == 180 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    print("Generated:")
    print(f"  {csv_path}       90 x 180 CSV")
    print(f"  {raw_png_path}   180 x 90 PNG")
    print(f"  {enlarged_png_path}   1440 x 720 PNG")
    print("Valid values: 16200")
    print("CSV values: only 0 and 1")
    print("Image scaling: nearest-neighbor, 8x")


if __name__ == "__main__":
    main()
