from pathlib import Path
import numpy as np
from PIL import Image


def polygon_mask(x, y, polygon):
    """Vectorized ray-casting point-in-polygon test."""
    px = np.asarray([p[0] for p in polygon], dtype=float)
    py = np.asarray([p[1] for p in polygon], dtype=float)
    inside = np.zeros(x.shape, dtype=bool)
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        crossing = ((yi > y) != (yj > y))
        denominator = yj - yi
        if abs(denominator) < 1e-15:
            denominator = 1e-15
        x_intersection = (xj - xi) * (y - yi) / denominator + xi
        inside ^= crossing & (x < x_intersection)
        j = i
    return inside


def ellipse_mask(x, y, lon, lat, rx, ry):
    return (
        ((x - lon) / rx) ** 2
        + ((y - lat) / ry) ** 2
        <= 1.0
    )


def add_polygon(land, x, y, points):
    return land | polygon_mask(x, y, points)


def add_ellipse(land, x, y, lon, lat, rx, ry):
    return land | ellipse_mask(x, y, lon, lat, rx, ry)


latitudes = 79.0 - np.arange(61, dtype=float) / 10.0
longitudes = -105.0 + np.arange(241, dtype=float) / 10.0
x, y = np.meshgrid(longitudes, latitudes)
land = np.zeros(x.shape, dtype=bool)


melville_island = [
    (-105.0, 78.25), (-103.0, 78.65), (-100.0, 78.85),
    (-97.0, 78.65), (-94.8, 77.85), (-95.4, 76.75),
    (-98.0, 76.05), (-102.0, 75.85), (-104.6, 76.30),
    (-105.0, 77.15),
]

bathurst_island = [
    (-104.7, 77.15), (-102.2, 77.50), (-99.2, 77.40),
    (-96.7, 76.75), (-97.0, 75.85), (-99.2, 75.45),
    (-102.2, 75.70), (-104.5, 76.25),
]

devon_island = [
    (-99.0, 75.75), (-97.0, 77.00), (-94.0, 78.00),
    (-89.5, 77.85), (-86.4, 77.15), (-83.9, 76.15),
    (-84.3, 75.10), (-88.0, 74.45), (-92.5, 74.35),
    (-96.5, 74.75),
]

for shape in (melville_island, bathurst_island, devon_island):
    land = add_polygon(land, x, y, shape)
cornwallis_island = [
    (-97.0, 75.75), (-94.4, 76.45), (-91.1, 76.42),
    (-89.7, 75.70), (-91.0, 74.95), (-94.7, 74.80),
    (-96.8, 75.15),
]

axel_heiberg_island = [
    (-98.0, 78.05), (-96.2, 79.0), (-92.5, 79.20),
    (-89.2, 78.65), (-87.8, 77.75), (-89.0, 76.85),
    (-92.0, 76.55), (-95.5, 76.95),
]

ellesmere_island = [
    (-91.8, 79.0), (-90.3, 78.25), (-88.6, 77.45),
    (-86.5, 76.95), (-84.4, 76.55), (-82.5, 76.85),
    (-81.0, 77.55), (-81.0, 79.0),
]

prince_of_wales = [
    (-104.6, 74.65), (-101.2, 75.15), (-98.0, 75.25),
    (-95.8, 74.95), (-94.0, 74.35), (-92.8, 73.35),
    (-91.5, 72.55), (-94.2, 72.10), (-98.2, 72.15),
    (-101.5, 72.65), (-104.0, 73.45),
]

somerset_island = [
    (-96.8, 74.30), (-94.2, 74.50), (-91.0, 74.40),
    (-87.2, 73.95), (-85.4, 73.25), (-86.2, 72.55),
    (-89.0, 72.15), (-92.8, 72.10), (-95.4, 72.75),
    (-96.8, 73.55),
]

boothia_peninsula = [
    (-97.8, 73.00), (-97.0, 73.65), (-95.7, 73.95),
    (-94.5, 73.55), (-94.2, 72.95), (-95.0, 72.45),
    (-97.1, 72.55),
]

for shape in (
    cornwallis_island,
    axel_heiberg_island,
    ellesmere_island,
    prince_of_wales,
    somerset_island,
    boothia_peninsula,
):
    land = add_polygon(land, x, y, shape)
small_ellipses = [
    (-91.85, 74.72, 0.28, 0.22),
    (-92.20, 74.90, 0.12, 0.10),
    (-94.15, 74.65, 0.25, 0.18),
    (-93.70, 74.80, 0.12, 0.10),
    (-90.05, 75.25, 0.22, 0.15),
    (-89.45, 75.45, 0.15, 0.12),
    (-96.85, 75.10, 0.16, 0.12),
    (-88.15, 74.55, 0.18, 0.13),
    (-86.95, 74.10, 0.12, 0.10),
    (-94.60, 73.75, 0.20, 0.15),
    (-92.95, 73.85, 0.15, 0.12),
    (-101.80, 74.35, 0.17, 0.13),
    (-96.25, 76.95, 0.18, 0.13),
    (-98.20, 76.10, 0.14, 0.10),
    (-103.20, 76.00, 0.16, 0.11),
]

for lon, lat, rx, ry in small_ellipses:
    land = add_ellipse(land, x, y, lon, lat, rx, ry)

large_lakes = [
    (-99.15, 73.35, 0.22, 0.13),
    (-97.55, 72.85, 0.18, 0.12),
    (-91.95, 72.85, 0.20, 0.12),
    (-101.10, 77.10, 0.18, 0.11),
]

water_holes = np.zeros(land.shape, dtype=bool)
for lon, lat, rx, ry in large_lakes:
    water_holes |= ellipse_mask(x, y, lon, lat, rx, ry)
land &= ~water_holes

matrix = land.astype(np.uint8)
assert matrix.shape == (61, 241)
assert matrix.size == 14701
assert np.all((matrix == 0) | (matrix == 1))

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)
csv_path = output_dir / "candidate.csv"
png_path = output_dir / "candidate.png"
png8_path = output_dir / "candidate_8x.png"

np.savetxt(csv_path, matrix, fmt="%d", delimiter=",", newline="\n")
image_array = (matrix * 255).astype(np.uint8)
image = Image.fromarray(image_array, mode="L")
assert image.size == (241, 61)
image.save(png_path)
image_8x = image.resize((1928, 488), Image.Resampling.NEAREST)
assert image_8x.size == (1928, 488)
assert set(np.unique(np.asarray(image_8x)).tolist()).issubset({0, 255})
image_8x.save(png8_path)

with csv_path.open("r", encoding="utf-8") as f:
    csv_lines = [line.rstrip("\n\r") for line in f]

assert len(csv_lines) == 61
for line in csv_lines:
    values = line.split(",")
    assert len(values) == 241
    assert all(value in {"0", "1"} for value in values)

assert Image.open(png_path).size == (241, 61)
assert Image.open(png8_path).size == (1928, 488)

print("完成")
print(f"矩阵形状: {matrix.shape}")
print(f"有效值数量: {matrix.size}")
print(f"陆地点数量: {int(matrix.sum())}")
print(f"水域点数量: {int(matrix.size - matrix.sum())}")
print(f"CSV: {csv_path}")
print(f"原始PNG: {png_path}")
print(f"8倍PNG: {png8_path}")
