import csv
from pathlib import Path

import numpy as np
from PIL import Image

STEP = 2.0 / 126.0
N = 127

# Hand-drawn approximate mainland shoreline, represented as (longitude, latitude).
MAINLAND = [
    (-130.33, 54.94),
    (-130.18, 54.83),
    (-130.08, 54.72),
    (-130.17, 54.61),
    (-130.03, 54.50),
    (-130.10, 54.39),
    (-129.94, 54.27),
    (-129.84, 54.12),
    (-129.96, 53.98),
    (-129.82, 53.86),
    (-129.91, 53.73),
    (-129.73, 53.62),
    (-129.76, 53.49),
    (-129.59, 53.37),
    (-129.52, 53.23),
    (-129.38, 53.00),
    (-129.00, 53.00),
    (-129.00, 55.00),
    (-130.33, 55.00),
]

# Major islands and island groups visible at the requested sampling scale.
ISLANDS = [
    [
        (-130.00, 54.39), (-129.88, 54.35), (-129.76, 54.26),
        (-129.68, 54.14), (-129.67, 53.99), (-129.72, 53.86),
        (-129.83, 53.80), (-129.94, 53.85), (-130.00, 53.99),
        (-130.05, 54.15), (-130.06, 54.29),
    ],
    [
        (-129.99, 53.67), (-129.85, 53.65), (-129.73, 53.57),
        (-129.69, 53.45), (-129.73, 53.34), (-129.84, 53.27),
        (-129.97, 53.30), (-130.02, 53.41), (-130.03, 53.55),
    ],
    [
        (-130.07, 53.91), (-129.99, 53.88), (-129.95, 53.78),
        (-129.97, 53.66), (-130.04, 53.61), (-130.09, 53.69),
        (-130.10, 53.82),
    ],
]
ISLANDS += [
    [
        (-130.25, 54.28), (-130.18, 54.25), (-130.16, 54.18),
        (-130.20, 54.11), (-130.27, 54.14), (-130.29, 54.22),
    ],
    [
        (-130.31, 53.99), (-130.25, 53.96), (-130.23, 53.89),
        (-130.27, 53.84), (-130.33, 53.87), (-130.35, 53.94),
    ],
    [
        (-130.42, 53.76), (-130.36, 53.74), (-130.34, 53.68),
        (-130.38, 53.62), (-130.44, 53.65), (-130.46, 53.71),
    ],
    [
        (-129.58, 54.49), (-129.52, 54.47), (-129.49, 54.41),
        (-129.52, 54.36), (-129.58, 54.38), (-129.61, 54.44),
    ],
    [
        (-129.43, 54.17), (-129.37, 54.14), (-129.35, 54.08),
        (-129.39, 54.03), (-129.45, 54.07),
    ],
    [
        (-129.47, 53.82), (-129.41, 53.79), (-129.39, 53.73),
        (-129.43, 53.68), (-129.49, 53.72),
    ],
    [
        (-129.35, 53.56), (-129.29, 53.53), (-129.27, 53.47),
        (-129.31, 53.42), (-129.37, 53.46),
    ],
    [
        (-129.28, 53.30), (-129.22, 53.27), (-129.20, 53.21),
        (-129.24, 53.16), (-129.30, 53.20),
    ],
]


def point_in_polygon(lon, lat, polygon):
    """Ray-crossing point test for a hand-drawn polygon."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > lat) != (yj > lat):
            crossing_lon = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing_lon:
                inside = not inside
        j = i
    return inside


def is_land(lon, lat):
    if point_in_polygon(lon, lat, MAINLAND):
        return 1
    return int(any(point_in_polygon(lon, lat, island) for island in ISLANDS))
def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    latitudes = [55.0 - row * STEP for row in range(N)]
    longitudes = [-131.0 + col * STEP for col in range(N)]

    matrix = np.zeros((N, N), dtype=np.uint8)
    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            matrix[row, col] = is_land(lon, lat)

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as handle:
        csv.writer(handle, lineterminator="\n").writerows(matrix.tolist())

    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(output_dir / "candidate.png")
    enlarged = image.resize((1016, 1016), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")

    with csv_path.open("r", newline="", encoding="ascii") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 127 and all(len(row) == 127 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert sum(map(len, rows)) == 16129
    source = Image.open(output_dir / "candidate.png").convert("RGB")
    result = Image.open(output_dir / "candidate_8x.png").convert("RGB")
    assert source.size == (127, 127)
    assert set(source.getdata()) <= {(0, 0, 0), (255, 255, 255)}
    assert result.size == (1016, 1016)
    assert result == source.resize((1016, 1016), Image.Resampling.NEAREST)


if __name__ == "__main__":
    main()
