from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image


# -----------------------------
# 手写地理规则
# 坐标顺序统一为 (经度, 纬度)
# -----------------------------

def ellipse_polygon(cx, cy, rx, ry, angle=0.0, n=32):
    """用手写中心、长短轴和旋转角近似岛屿轮廓。"""
    theta = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    ca = math.cos(angle)
    sa = math.sin(angle)

    points = []
    for t in theta:
        x = rx * math.cos(t)
        y = ry * math.sin(t)
        points.append((
            cx + x * ca - y * sa,
            cy + x * sa + y * ca,
        ))
    return points


# Kyushu 北部及东部
KYUSHU = [
    (129.62, 33.00),
    (131.72, 33.00),
    (131.76, 33.22),
    (131.82, 33.48),
    (131.96, 33.73),
    (132.02, 33.94),
    (131.78, 34.05),
    (131.48, 34.12),
    (131.12, 34.07),
    (130.78, 34.03),
    (130.50, 33.91),
    (130.27, 33.72),
    (130.08, 33.45),
    (129.82, 33.22),
]

# 中国地方南岸及近畿北缘
HONSHU_WEST = [
    (130.78, 34.00),
    (131.00, 34.01),
    (131.20, 34.10),
    (131.40, 34.18),
    (131.70, 34.20),
    (131.95, 34.16),
    (132.20, 34.22),
    (132.45, 34.30),
    (132.70, 34.34),
    (132.95, 34.39),
    (133.20, 34.42),
    (133.45, 34.46),
    (133.70, 34.52),
    (133.95, 34.57),
    (134.20, 34.62),
    (134.45, 34.70),
    (134.65, 34.77),
    (134.85, 34.80),
    (135.05, 34.77),
    (135.25, 34.68),
    (135.45, 34.72),
    (135.58, 34.90),
    (135.65, 35.00),
    (130.75, 35.00),
]

# Shikoku
SHIKOKU = [
    (132.00, 33.30),
    (132.15, 33.45),
    (132.35, 33.55),
    (132.50, 33.72),
    (132.70, 33.95),
    (132.90, 34.10),
    (133.15, 34.22),
    (133.40, 34.32),
    (133.70, 34.35),
    (134.00, 34.32),
    (134.25, 34.28),
    (134.50, 34.20),
    (134.72, 34.10),
    (134.90, 33.95),
    (135.05, 33.75),
    (135.12, 33.55),
    (135.05, 33.35),
    (134.85, 33.25),
    (134.60, 33.15),
    (134.30, 33.08),
    (134.00, 33.02),
    (133.70, 33.00),
    (133.35, 33.00),
    (133.00, 33.00),
    (132.65, 33.00),
    (132.30, 33.02),
    (132.00, 33.08),
]

# Awaji Island
AWAJI = [
    (134.65, 34.35),
    (134.75, 34.52),
    (134.90, 34.65),
    (135.03, 34.62),
    (135.12, 34.45),
    (135.10, 34.25),
    (134.97, 34.12),
    (134.82, 34.15),
    (134.70, 34.25),
]

# Kii Peninsula，含伊势湾西侧
KII_PENINSULA = [
    (135.25, 34.68),
    (135.35, 34.55),
    (135.45, 34.35),
    (135.50, 34.10),
    (135.45, 33.90),
    (135.60, 33.70),
    (135.90, 33.55),
    (136.20, 33.48),
    (136.45, 33.55),
    (136.65, 33.75),
    (136.75, 34.00),
    (136.73, 34.25),
    (136.65, 34.50),
    (136.55, 34.68),
    (136.45, 34.82),
    (136.20, 35.00),
    (135.25, 35.00),
]

# 伊势湾东北侧的本州陆地
HONSHU_CENTRAL = [
    (136.40, 35.00),
    (137.00, 35.00),
    (137.00, 34.92),
    (136.85, 34.88),
    (136.72, 34.80),
    (136.60, 34.70),
    (136.50, 34.72),
]

# 知多半岛
CHITA_PENINSULA = [
    (136.72, 34.80),
    (136.90, 34.78),
    (137.00, 34.68),
    (137.00, 34.50),
    (136.92, 34.52),
    (136.84, 34.62),
    (136.78, 34.70),
]


LAND_POLYGONS = [
    KYUSHU,
    HONSHU_WEST,
    SHIKOKU,
    AWAJI,
    KII_PENINSULA,
    HONSHU_CENTRAL,
    CHITA_PENINSULA,
]


# 主要岛屿与较大的岛群。
# 参数为：中心经度、中心纬度、东西半轴、南北半轴、旋转角、顶点数
ISLAND_SPECS = [
    # 对马、壹岐、平户及玄界灘岛屿
    (129.35, 34.58, 0.18, 0.13, -0.12, 36),
    (129.30, 34.32, 0.15, 0.12,  0.20, 36),
    (129.34, 34.16, 0.09, 0.09,  0.10, 28),
    (129.72, 33.78, 0.14, 0.11, -0.05, 32),
    (129.55, 33.37, 0.10, 0.13,  0.15, 32),
    (131.15, 34.76, 0.06, 0.07,  0.00, 24),

    # 周防灘、伊予灘
    (131.65, 33.72, 0.08, 0.06,  0.00, 24),
    (132.24, 34.00, 0.20, 0.13,  0.12, 36),
    (132.28, 34.16, 0.09, 0.08, -0.10, 28),
    (132.32, 34.28, 0.07, 0.16,  0.08, 32),
    (132.45, 34.25, 0.10, 0.13, -0.20, 32),
    (132.52, 34.11, 0.13, 0.12,  0.05, 32),
    (132.68, 34.00, 0.08, 0.06,  0.10, 24),

    # 芸予諸島
    (132.91, 34.25, 0.10, 0.09,  0.15, 28),
    (132.99, 34.24, 0.13, 0.12, -0.12, 32),
    (133.08, 34.20, 0.09, 0.09,  0.10, 28),
    (133.17, 34.30, 0.13, 0.11,  0.18, 32),
    (133.08, 34.30, 0.11, 0.10, -0.10, 32),
    (133.13, 34.26, 0.08, 0.09,  0.10, 28),
    (133.19, 34.27, 0.07, 0.06, -0.15, 24),
    (133.18, 34.38, 0.10, 0.07,  0.00, 28),
    (133.37, 34.33, 0.07, 0.06,  0.10, 24),
    (133.54, 34.39, 0.07, 0.06, -0.10, 24),
    (133.60, 34.40, 0.06, 0.05,  0.00, 24),
    (133.66, 34.34, 0.05, 0.05,  0.00, 24),

    # 濑户内海中部
    (133.82, 34.34, 0.06, 0.05,  0.00, 24),
    (134.05, 34.48, 0.09, 0.08,  0.00, 28),
    (133.99, 34.46, 0.09, 0.06,  0.10, 28),
    (134.05, 34.43, 0.05, 0.06, -0.10, 24),
    (134.00, 34.43, 0.05, 0.05,  0.00, 24),
    (134.28, 34.52, 0.18, 0.14,  0.18, 40),
    (133.53, 34.13, 0.07, 0.06,  0.00, 24),
    (133.65, 34.15, 0.06, 0.05,  0.00, 24),
    (134.42, 34.26, 0.06, 0.05,  0.00, 24),
    (134.68, 34.27, 0.05, 0.05,  0.00, 24),

    # 大阪湾口及附近
    (135.00, 34.27, 0.08, 0.05,  0.15, 28),
    (135.08, 34.28, 0.06, 0.05, -0.10, 24),
    (135.18, 34.22, 0.05, 0.04,  0.00, 24),

    # 伊势湾及其附近岛屿
    (136.84, 34.47, 0.05, 0.05,  0.00, 24),
    (136.90, 34.48, 0.06, 0.05,  0.10, 24),
    (136.98, 34.54, 0.04, 0.06,  0.00, 24),
    (136.76, 34.53, 0.04, 0.04,  0.00, 24),
]

for cx, cy, rx, ry, angle, n in ISLAND_SPECS:
    LAND_POLYGONS.append(
        ellipse_polygon(cx, cy, rx, ry, angle=angle, n=n)
    )


def points_inside_polygon(xs, ys, polygon):
    """
    射线法点判据。
    每个采样点只根据自身经纬度判断，不进行格子面积平均。
    """
    poly = np.asarray(polygon, dtype=float)
    px = poly[:, 0]
    py = poly[:, 1]

    inside = np.zeros(xs.shape, dtype=bool)
    x0 = px[-1]
    y0 = py[-1]

    for x1, y1 in zip(px, py):
        crosses = (y0 > ys) != (y1 > ys)

        # crosses 为 False 时，除数为零的边不会影响最终结果。
        with np.errstate(divide="ignore", invalid="ignore"):
            x_intersection = (
                (x1 - x0) * (ys - y0) / (y1 - y0) + x0
            )

        inside ^= crosses & (xs < x_intersection)
        x0 = x1
        y0 = y1

    return inside


def build_matrix():
    # 端点包含，得到严格的 64 行和 253 列。
    latitudes = np.linspace(35.0, 33.0, 64)
    longitudes = np.linspace(129.0, 137.0, 253)

    lon_grid, lat_grid = np.meshgrid(
        longitudes,
        latitudes,
        indexing="xy",
    )

    land = np.zeros(lon_grid.shape, dtype=bool)

    for polygon in LAND_POLYGONS:
        land |= points_inside_polygon(
            lon_grid,
            lat_grid,
            polygon,
        )

    # 陆地为 1，水域为 0。
    return land.astype(np.uint8)


def write_csv(matrix, path):
    with path.open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())


def verify_csv(path):
    with path.open("r", newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))

    assert len(rows) == 64
    assert all(len(row) == 253 for row in rows)

    values = []
    for row in rows:
        for value in row:
            assert value in ("0", "1")
            values.append(value)

    assert len(values) == 64 * 253
    return rows


def save_images(matrix, original_path, enlarged_path):
    pixels = (matrix * 255).astype(np.uint8)

    original = Image.fromarray(pixels, mode="L")
    original.save(original_path)

    try:
        nearest = Image.Resampling.NEAREST
    except AttributeError:
        nearest = Image.NEAREST

    enlarged = original.resize(
        (253 * 8, 64 * 8),
        resample=nearest,
    )
    enlarged.save(enlarged_path)


def verify_images(original_path, enlarged_path):
    original = Image.open(original_path).convert("L")
    enlarged = Image.open(enlarged_path).convert("L")

    assert original.size == (253, 64)
    assert enlarged.size == (2024, 512)

    original_values = np.asarray(original)
    enlarged_values = np.asarray(enlarged)

    assert set(np.unique(original_values).tolist()) <= {0, 255}
    assert set(np.unique(enlarged_values).tolist()) <= {0, 255}

    # 验证放大图确实等于原图每个像素复制 8×8。
    expected = np.repeat(np.repeat(original_values, 8, axis=0), 8, axis=1)
    assert np.array_equal(enlarged_values, expected)


def main():
    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "candidate.csv"
    original_png_path = output_dir / "candidate.png"
    enlarged_png_path = output_dir / "candidate_8x.png"

    matrix = build_matrix()

    assert matrix.shape == (64, 253)
    assert matrix.size == 64 * 253
    assert np.all((matrix == 0) | (matrix == 1))

    write_csv(matrix, csv_path)
    save_images(matrix, original_png_path, enlarged_png_path)

    verify_csv(csv_path)
    verify_images(original_png_path, enlarged_png_path)

    print("matrix shape:", matrix.shape)
    print("valid values:", matrix.size)
    print("land pixels:", int(matrix.sum()))
    print("water pixels:", int(matrix.size - matrix.sum()))
    print("CSV:", csv_path)
    print("original PNG:", original_png_path, "(253x64)")
    print("enlarged PNG:", enlarged_png_path, "(2024x512)")
    print("all checks passed")


if __name__ == "__main__":
    main()
