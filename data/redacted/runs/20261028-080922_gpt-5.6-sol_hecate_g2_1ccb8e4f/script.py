from pathlib import Path
import numpy as np
from PIL import Image

N = 127
STEP = 2.0 / 126.0
LATS = [55.0 - i * STEP for i in range(N)]
LONS = [-131.0 + j * STEP for j in range(N)]

# Hand-drawn mainland coastline, north to south; land is east of this line.
COAST_NORTH_TO_SOUTH = [
    (-130.02, 55.00), (-130.06, 54.94), (-130.12, 54.89),
    (-130.08, 54.83), (-130.19, 54.78), (-130.16, 54.72),
    (-130.27, 54.67), (-130.22, 54.61), (-130.31, 54.57),
    (-130.28, 54.51), (-130.39, 54.47), (-130.35, 54.42),
    (-130.48, 54.38), (-130.52, 54.33), (-130.45, 54.28),
    (-130.38, 54.24), (-130.44, 54.19), (-130.36, 54.14),
    (-130.31, 54.09), (-130.26, 54.04), (-130.21, 53.99),
    (-130.15, 53.94), (-130.09, 53.89), (-130.04, 53.84),
    (-130.00, 53.79), (-129.96, 53.74), (-129.92, 53.69),
    (-129.89, 53.64), (-129.85, 53.59), (-129.81, 53.54),
    (-129.77, 53.49), (-129.73, 53.44), (-129.69, 53.39),
    (-129.65, 53.34), (-129.61, 53.29), (-129.57, 53.24),
    (-129.53, 53.19), (-129.50, 53.14), (-129.47, 53.09),
    (-129.44, 53.04), (-129.40, 53.00),
]
MAINLAND = COAST_NORTH_TO_SOUTH + [(-128.0, 53.0), (-128.0, 55.0)]

ISLANDS = [
    # Approximate Banks Island
    [
        (-129.31, 54.30), (-129.20, 54.28), (-129.12, 54.22),
        (-129.09, 54.12), (-129.12, 54.02), (-129.09, 53.93),
        (-129.13, 53.84), (-129.10, 53.75), (-129.14, 53.66),
        (-129.12, 53.56), (-129.17, 53.47), (-129.25, 53.43),
        (-129.32, 53.49), (-129.34, 53.60), (-129.31, 53.70),
        (-129.34, 53.80), (-129.30, 53.91), (-129.34, 54.02),
        (-129.31, 54.13), (-129.34, 54.22),
    ],
    # Approximate Pitt Island
    [
        (-129.77, 53.60), (-129.67, 53.60), (-129.60, 53.55),
        (-129.57, 53.47), (-129.58, 53.39), (-129.53, 53.31),
        (-129.55, 53.23), (-129.61, 53.17), (-129.70, 53.19),
        (-129.76, 53.26), (-129.81, 53.35), (-129.82, 53.45),
        (-129.80, 53.53),
    ],
]
# Additional hand-drawn nearshore rocks and reefs.
ISLANDS += [
    [(-130.46, 54.48), (-130.42, 54.49), (-130.40, 54.46),
     (-130.43, 54.44), (-130.47, 54.45)],
    [(-130.30, 54.36), (-130.27, 54.37), (-130.25, 54.34),
     (-130.28, 54.32), (-130.31, 54.33)],
    [(-130.18, 54.10), (-130.14, 54.11), (-130.12, 54.08),
     (-130.15, 54.06), (-130.19, 54.07)],
    [(-129.94, 53.82), (-129.90, 53.83), (-129.88, 53.80),
     (-129.91, 53.78), (-129.95, 53.79)],
    [(-129.83, 53.70), (-129.80, 53.71), (-129.78, 53.68),
     (-129.81, 53.66), (-129.84, 53.67)],
]


def point_in_polygon(lon, lat, polygon):
    """Ray casting; points exactly on a polygon edge count as land."""
    inside = False
    count = len(polygon)
    for i in range(count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]
        cross = (lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
        if abs(cross) < 1e-12:
            if (min(x1, x2) - 1e-12 <= lon <= max(x1, x2) + 1e-12
                    and min(y1, y2) - 1e-12 <= lat <= max(y1, y2) + 1e-12):
                return True
        if (y1 > lat) != (y2 > lat):
            x_intersection = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < x_intersection:
                inside = not inside
    return inside


def is_land(lon, lat):
    if point_in_polygon(lon, lat, MAINLAND):
        return 1
    return int(any(point_in_polygon(lon, lat, island) for island in ISLANDS))
def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = np.zeros((N, N), dtype=np.uint8)
    for row, lat in enumerate(LATS):
        for col, lon in enumerate(LONS):
            matrix[row, col] = is_land(lon, lat)

    np.savetxt(output_dir / "candidate.csv", matrix, fmt="%d", delimiter=",")

    # Grayscale 0/255: black water, white land.
    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(output_dir / "candidate.png")
    enlarged = image.resize(
        (N * 8, N * 8), resample=Image.Resampling.NEAREST
    )
    enlarged.save(output_dir / "candidate_8x.png")

    assert matrix.shape == (127, 127) and matrix.size == 16129
    assert set(np.unique(matrix)).issubset({0, 1})
    assert image.size == (127, 127) and enlarged.size == (1016, 1016)
    assert set(image.getdata()).issubset({(0, 0, 0), (255, 255, 255)})
    print("Validated 16,129 values and both PNG dimensions.")


if __name__ == "__main__":
    main()
