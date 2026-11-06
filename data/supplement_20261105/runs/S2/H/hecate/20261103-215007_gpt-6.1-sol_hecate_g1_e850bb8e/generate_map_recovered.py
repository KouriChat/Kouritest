from pathlib import Path

import numpy as np
from PIL import Image


ROWS = 127
COLS = 127
STEP = 2.0 / 126.0

# 坐标均为 (纬度, 经度)。这些是凭已有地理知识手写的近似规则，
# 不是从地图或地理数据库读取的轮廓。
#
# 主陆地西侧海岸线，按纬度递增排列；海岸线以东判为陆地。
COAST = np.array([
    [53.00, -129.62],
    [53.20, -129.70],
    [53.40, -129.82],
    [53.60, -129.93],
    [53.80, -130.05],
    [54.00, -130.17],
    [54.20, -130.32],
    [54.35, -130.38],
    [54.50, -130.34],
    [54.70, -130.28],
    [54.85, -130.34],
    [55.00, -130.42],
], dtype=float)

# 主要内湾、海峡和河口近似轮廓，按水域处理。
WATER_POLYGONS = [
    # Portland Canal / Portland Inlet 的南段近似。
    [
        (54.68, -130.34), (54.75, -130.33), (54.83, -130.30),
        (54.91, -130.28), (55.00, -130.31), (55.00, -130.12),
        (54.92, -130.13), (54.84, -130.15), (54.76, -130.17),
        (54.68, -130.19),
    ],
    # Skeena 河口及其近岸水道的概略切口。
    [
        (54.05, -130.37), (54.13, -130.40), (54.21, -130.39),
        (54.28, -130.36), (54.31, -130.30), (54.27, -130.25),
        (54.20, -130.27), (54.13, -130.29), (54.07, -130.30),
    ],
    # 南部若干狭长海湾的概略切口。
    [
        (53.45, -129.96), (53.51, -130.00), (53.58, -130.01),
        (53.66, -129.98), (53.73, -129.94), (53.72, -129.90),
        (53.65, -129.91), (53.57, -129.93), (53.50, -129.94),
    ],
]

# 海峡中的较大岛屿近似轮廓。
ISLAND_POLYGONS = [
    # Banks Island
    [
        (53.86, -130.13), (53.91, -130.20), (54.00, -130.23),
        (54.10, -130.20), (54.19, -130.15), (54.25, -130.06),
        (54.24, -129.96), (54.20, -129.88), (54.12, -129.83),
        (54.02, -129.82), (53.94, -129.87), (53.88, -129.96),
    ],
    # Pitt Island
    [
        (53.27, -129.80), (53.32, -129.88), (53.41, -129.91),
        (53.52, -129.89), (53.63, -129.83), (53.75, -129.75),
        (53.83, -129.67), (53.82, -129.58), (53.75, -129.52),
        (53.64, -129.54), (53.52, -129.60), (53.41, -129.67),
        (53.32, -129.72),
    ],
    # Porcher Island, only its eastern portion falls inside the sampling frame.
    [
        (53.83, -130.99), (53.90, -131.00), (53.99, -130.98),
        (54.06, -130.94), (54.08, -130.88), (54.04, -130.83),
        (53.96, -130.82), (53.88, -130.86),
    ],
    # 小型近岸岛屿的概略补充。
    [
        (53.56, -130.31), (53.60, -130.35), (53.66, -130.34),
        (53.69, -130.29), (53.66, -130.24), (53.60, -130.25),
    ],
]


def polygon_mask(lat: float, lons: np.ndarray, polygon: list[tuple[float, float]]) -> np.ndarray:
    """返回一行采样点落入多边形的掩码。"""
    vertices = np.asarray(polygon, dtype=float)
    y = vertices[:, 0]
    x = vertices[:, 1]
    inside = np.zeros(lons.shape, dtype=bool)

    j = len(vertices) - 1
    for i in range(len(vertices)):
        yi, yj = y[i], y[j]
        xi, xj = x[i], x[j]

        crosses = (yi > lat) != (yj > lat)
        denominator = yj - yi
        if denominator != 0:
            crossing_lon = (xj - xi) * (lat - yi) / denominator + xi
            inside ^= crosses & (lons < crossing_lon)
        j = i

    return inside


def build_matrix() -> np.ndarray:
    latitudes = 55.0 - np.arange(ROWS, dtype=float) * STEP
    longitudes = -131.0 + np.arange(COLS, dtype=float) * STEP
    matrix = np.zeros((ROWS, COLS), dtype=np.uint8)

    coast_lats = COAST[:, 0]
    coast_lons = COAST[:, 1]

    for row, lat in enumerate(latitudes):
        coast_lon = np.interp(lat, coast_lats, coast_lons)

        # 主陆地：近似海岸线以东。
        matrix[row, longitudes >= coast_lon] = 1

        # 先从陆地中扣除内湾和河口，再加入岛屿。
        for polygon in WATER_POLYGONS:
            matrix[row, polygon_mask(lat, longitudes, polygon)] = 0
        for polygon in ISLAND_POLYGONS:
            matrix[row, polygon_mask(lat, longitudes, polygon)] = 1

    return matrix


def main() -> None:
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = build_matrix()
    assert matrix.shape == (127, 127)
    assert matrix.size == 16129
    assert np.isin(matrix, (0, 1)).all()

    csv_path = out_dir / "candidate.csv"
    np.savetxt(csv_path, matrix, fmt="%d", delimiter=",")

    # 1 为白色陆地，0 为黑色水域；行列方向与采样顺序一致。
    pixels = np.where(matrix == 1, 255, 0).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    image.save(out_dir / "candidate.png")

    enlarged = image.resize((1016, 1016), resample=Image.Resampling.NEAREST)
    enlarged.save(out_dir / "candidate_8x.png")

    # 回读并检查交付文件的形状、取值及像素颜色。
    loaded = np.loadtxt(csv_path, delimiter=",", dtype=np.uint8)
    assert loaded.shape == (127, 127)
    assert np.isin(loaded, (0, 1)).all()

    with Image.open(out_dir / "candidate.png") as check:
        assert check.size == (127, 127)
        assert set(check.getdata()) <= {0, 255}

    with Image.open(out_dir / "candidate_8x.png") as check:
        assert check.size == (1016, 1016)
        assert set(check.getdata()) <= {0, 255}

    print(f"Generated {matrix.shape[0]}x{matrix.shape[1]} matrix ({matrix.size} values).")
    print(f"CSV: {csv_path}")
    print(f"PNG: {out_dir / 'candidate.png'}")
    print(f"8x PNG: {out_dir / 'candidate_8x.png'}")


if __name__ == "__main__":
    main()
