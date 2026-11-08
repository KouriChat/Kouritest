from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image


# ------------------------------------------------------------
# 南海及周边的手工近似陆地规则
#
# 坐标顺序统一为 (longitude, latitude)。
# 所有边界均为凭已有地理知识手工估计，不读取地图或数据库。
# ------------------------------------------------------------

MIN_LON = 99.0
MAX_LON = 127.0
MIN_LAT = 1.0
MAX_LAT = 23.0
STEP = 0.2

N_ROWS = 111
N_COLS = 141


def point_in_polygon(lon, lat, polygon):
    """射线法判断点是否在多边形内。"""
    inside = False
    j = len(polygon) - 1

    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        crosses = ((yi > lat) != (yj > lat))
        if crosses:
            x_at_lat = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_at_lat:
                inside = not inside

        j = i

    return inside


def point_in_circle(lon, lat, center_lon, center_lat, radius_lon, radius_lat=None):
    """经纬度网格上的椭圆近似岛屿或礁盘。"""
    if radius_lat is None:
        radius_lat = radius_lon

    dx = (lon - center_lon) / radius_lon
    dy = (lat - center_lat) / radius_lat
    return dx * dx + dy * dy <= 1.0


# 大陆和较大的陆地块。
# 多边形只在任务框内提供足够的近似覆盖。
LAND_POLYGONS = [
    # 中国大陆东南部：广西、广东、福建南部及邻近大陆
    [
        (104.5, 23.0), (127.0, 23.0), (127.0, 24.0),
        (124.0, 24.0), (122.0, 24.2), (120.0, 24.2),
        (118.4, 24.0), (117.2, 23.8), (116.3, 23.5),
        (115.0, 23.4), (114.0, 23.2), (113.4, 22.8),
        (112.7, 22.4), (111.7, 22.0), (110.8, 21.7),
        (109.8, 21.6), (108.8, 21.8), (107.6, 22.2),
        (106.5, 22.5), (105.5, 22.8), (104.5, 23.0),
    ],

    # 越南主体
    [
        (102.1, 23.4), (104.8, 23.4), (106.0, 22.8),
        (107.2, 21.5), (108.3, 20.2), (108.9, 19.0),
        (108.8, 17.5), (108.2, 16.0), (107.7, 14.7),
        (107.1, 13.4), (106.5, 12.2), (105.8, 11.0),
        (104.8, 10.0), (103.8, 9.2), (102.8, 8.7),
        (102.0, 9.4), (102.1, 11.0), (102.5, 12.3),
        (102.6, 14.0), (102.4, 16.0), (102.0, 18.0),
        (101.8, 20.0), (102.1, 23.4),
    ],

    # 柬埔寨、泰国东部及马来半岛北部
    [
        (102.2, 14.8), (104.7, 14.8), (105.5, 13.8),
        (105.8, 12.5), (105.5, 11.5), (104.7, 10.5),
        (103.7, 9.5), (102.7, 8.6), (101.9, 7.7),
        (101.4, 6.5), (101.3, 5.4), (101.7, 4.7),
        (103.0, 4.7), (103.8, 5.4), (103.8, 6.8),
        (103.5, 8.0), (104.0, 9.0), (104.8, 10.0),
        (105.8, 10.8), (106.5, 12.2), (106.2, 13.5),
        (105.3, 14.8), (102.2, 14.8),
    ],

    # 马来西亚婆罗洲与文莱
    [
        (108.0, 7.4), (109.8, 7.1), (111.4, 6.7),
        (113.0, 6.1), (114.7, 5.5), (116.0, 4.9),
        (117.0, 4.0), (117.8, 3.0), (117.5, 2.0),
        (116.0, 1.2), (114.0, 1.0), (112.0, 1.2),
        (110.3, 2.0), (109.0, 3.2), (108.2, 4.7),
        (108.0, 7.4),
    ],

    # 苏门答腊东北部，在本框内的近似部分
    [
        (99.0, 5.5), (100.2, 5.3), (101.0, 4.7),
        (102.0, 3.8), (103.0, 2.8), (104.2, 2.0),
        (105.0, 1.0), (99.0, 1.0),
    ],

    # 菲律宾巴拉望岛
    [
        (117.0, 11.9), (118.0, 11.5), (119.0, 10.3),
        (119.5, 9.1), (119.4, 8.0), (118.8, 7.0),
        (117.8, 7.2), (117.2, 8.3), (116.8, 9.5),
        (116.7, 10.8), (117.0, 11.9),
    ],

    # 菲律宾吕宋岛
    [
        (119.5, 18.6), (120.7, 18.5), (121.7, 18.0),
        (122.3, 17.0), (122.0, 15.8), (121.5, 14.7),
        (121.0, 13.4), (120.2, 13.0), (119.4, 13.5),
        (118.7, 14.7), (118.3, 16.0), (118.8, 17.3),
        (119.5, 18.6),
    ],

    # 菲律宾萨马岛、莱特岛、宿务岛、保和岛、棉兰老岛
    [
        (123.0, 13.8), (124.0, 13.6), (125.0, 13.0),
        (126.0, 11.8), (126.3, 10.3), (125.7, 8.8),
        (124.4, 7.8), (123.0, 7.2), (122.0, 7.5),
        (121.7, 8.7), (122.2, 10.0), (122.7, 11.2),
        (122.5, 12.5), (123.0, 13.8),
    ],

    # 台湾岛
    [
        (120.0, 22.6), (120.1, 23.4), (120.7, 24.4),
        (121.5, 25.2), (122.0, 24.7), (121.9, 23.7),
        (121.4, 22.7), (120.8, 21.9), (120.2, 22.1),
        (120.0, 22.6),
    ],
]


# 较小岛屿、岛链、主要礁盘的椭圆近似。
# 半径足够小时，只会影响附近少量采样点。
ISLAND_ELLIPSES = [
    # 海南岛
    (109.8, 19.2, 0.75, 0.55),

    # 台湾外围及澎湖
    (119.6, 23.6, 0.20, 0.12),
    (119.2, 23.6, 0.12, 0.08),

    # 西沙群岛
    (111.2, 16.8, 0.11, 0.07),
    (111.6, 16.6, 0.09, 0.06),
    (112.0, 16.5, 0.08, 0.05),
    (112.4, 16.3, 0.07, 0.05),
    (112.7, 16.1, 0.06, 0.04),
    (111.0, 16.4, 0.06, 0.04),

    # 中沙群岛的岛礁近似
    (114.0, 15.8, 0.08, 0.04),
    (114.2, 15.7, 0.06, 0.04),

    # 东沙岛
    (116.8, 20.7, 0.10, 0.05),

    # 南沙主要岛礁和岛屿
    (114.4, 11.0, 0.07, 0.04),
    (114.1, 10.4, 0.06, 0.04),
    (113.9, 10.0, 0.07, 0.04),
    (114.3, 9.9, 0.05, 0.03),
    (115.8, 10.4, 0.06, 0.04),
    (116.0, 10.2, 0.05, 0.03),
    (117.1, 10.0, 0.06, 0.04),
    (117.8, 9.9, 0.05, 0.03),
    (118.3, 9.7, 0.05, 0.03),
    (118.9, 9.0, 0.07, 0.04),
    (119.3, 8.7, 0.05, 0.03),

    # 菲律宾西部、南部较小岛屿
    (120.8, 11.3, 0.18, 0.08),
    (121.0, 10.7, 0.12, 0.07),
    (122.0, 10.0, 0.10, 0.06),
    (123.7, 10.8, 0.13, 0.08),
    (124.2, 11.8, 0.12, 0.07),
    (125.4, 12.5, 0.12, 0.07),
    (126.4, 9.2, 0.12, 0.08),
]


def is_land(lon, lat):
    for polygon in LAND_POLYGONS:
        if point_in_polygon(lon, lat, polygon):
            return 1

    for center_lon, center_lat, radius_lon, radius_lat in ISLAND_ELLIPSES:
        if point_in_circle(
            lon,
            lat,
            center_lon,
            center_lat,
            radius_lon,
            radius_lat,
        ):
            return 1

    return 0


def build_matrix():
    matrix = np.zeros((N_ROWS, N_COLS), dtype=np.uint8)

    for row in range(N_ROWS):
        lat = MAX_LAT - row * STEP

        for col in range(N_COLS):
            lon = MIN_LON + col * STEP
            matrix[row, col] = is_land(lon, lat)

    return matrix


def write_csv(matrix, path):
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for row in matrix:
            writer.writerow([int(value) for value in row])


def write_images(matrix, output_dir):
    # 0 -> black water, 1 -> white land
    image_array = np.where(matrix == 1, 255, 0).astype(np.uint8)

    original = Image.fromarray(image_array, mode="L")
    original.save(output_dir / "candidate.png")

    enlarged = original.resize(
        (N_COLS * 8, N_ROWS * 8),
        resample=Image.Resampling.NEAREST,
    )
    enlarged.save(output_dir / "candidate_8x.png")


def validate(matrix, output_dir):
    assert matrix.shape == (111, 141)
    assert matrix.size == 15651
    assert set(np.unique(matrix)).issubset({0, 1})

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("r", encoding="ascii") as handle:
        rows = list(csv.reader(handle))

    assert len(rows) == 111
    assert all(len(row) == 141 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    original = Image.open(output_dir / "candidate.png")
    enlarged = Image.open(output_dir / "candidate_8x.png")

    assert original.size == (141, 111)
    assert enlarged.size == (1128, 888)
    assert original.mode == "L"
    assert enlarged.mode == "L"

    original_values = set(original.getdata())
    enlarged_values = set(enlarged.getdata())
    assert original_values.issubset({0, 255})
    assert enlarged_values.issubset({0, 255})

    # 验证放大图是严格的 8 倍最近邻复制。
    enlarged_array = np.asarray(enlarged)
    expected = np.repeat(np.repeat(np.asarray(original), 8, axis=0), 8, axis=1)
    assert np.array_equal(enlarged_array, expected)


def main():
    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = build_matrix()

    write_csv(matrix, output_dir / "candidate.csv")
    write_images(matrix, output_dir)
    validate(matrix, output_dir)

    print("Generated:")
    print(f"  {output_dir / 'candidate.csv'}")
    print(f"  {output_dir / 'candidate.png'}")
    print(f"  {output_dir / 'candidate_8x.png'}")
    print("Matrix: 111 rows x 141 columns = 15651 values")
    print("Validation: passed")


if __name__ == "__main__":
    main()
