from pathlib import Path
import csv

import numpy as np
from PIL import Image


N = 127
STEP = 2.0 / 126.0
OUT = Path("outputs")


def point_in_polygon(x, y, polygon):
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        cross = (x - x1) * dy - (y - y1) * dx
        if abs(cross) < 1e-12 and (
            min(x1, x2) - 1e-12 <= x <= max(x1, x2) + 1e-12
            and min(y1, y2) - 1e-12 <= y <= max(y1, y2) + 1e-12
        ):
            return True
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing_x:
                inside = not inside
    return inside


# Hand-authored approximate mainland boundary; longitude is positive west.
MAINLAND = [
    (130.62, 55.00), (130.58, 54.92), (130.42, 54.82),
    (130.33, 54.71), (130.47, 54.58), (130.34, 54.45),
    (130.43, 54.30), (130.26, 54.16), (130.18, 54.02),
    (130.02, 53.88), (129.91, 53.72), (129.81, 53.55),
    (129.68, 53.39), (129.57, 53.22), (129.45, 53.00),
    (129.00, 53.00), (129.00, 55.00),
]
# Approximate islands and island groups within the sampling box.
ISLANDS = [
    # Dundas Island group.
    [
        (130.99, 54.86), (130.82, 54.88), (130.67, 54.79),
        (130.63, 54.65), (130.75, 54.54), (130.93, 54.56),
        (131.00, 54.70),
    ],
    # Stephens and nearby islands.
    [
        (130.99, 54.58), (130.88, 54.58), (130.78, 54.45),
        (130.82, 54.33), (130.96, 54.35), (131.00, 54.47),
    ],
    # Banks Island.
    [
        (130.88, 54.43), (130.68, 54.51), (130.47, 54.43),
        (130.33, 54.25), (130.38, 54.03), (130.57, 53.96),
        (130.79, 54.07), (130.92, 54.25),
    ],
    # Porcher Island.
    [
        (130.76, 54.06), (130.57, 54.15), (130.35, 54.08),
        (130.17, 53.94), (130.14, 53.77), (130.32, 53.65),
        (130.57, 53.68), (130.78, 53.83),
    ],
    # Pitt Island.
    [
        (130.49, 53.66), (130.29, 53.75), (130.04, 53.68),
        (129.86, 53.52), (129.90, 53.34), (130.11, 53.23),
        (130.34, 53.29), (130.53, 53.46),
    ],
    # Gil Island and adjacent islands.
    [
        (130.88, 53.82), (130.69, 53.87), (130.55, 53.74),
        (130.57, 53.55), (130.72, 53.45), (130.91, 53.51),
        (131.00, 53.68),
    ],
    # Small southern island group.
    [
        (130.97, 53.40), (130.82, 53.42), (130.70, 53.29),
        (130.73, 53.16), (130.90, 53.12), (131.00, 53.23),
    ],
]
# Hand-authored coastal island fragments and large-island edges.
ISLANDS.extend([
    # Eastern fringe of Graham Island.
    [
        (131.00, 54.88), (130.96, 54.75), (130.89, 54.61),
        (130.91, 54.43), (130.98, 54.28), (131.00, 54.22),
    ],
    # Eastern fringe of Moresby Island.
    [
        (131.00, 53.62), (130.96, 53.47), (130.91, 53.31),
        (130.90, 53.12), (130.97, 53.00), (131.00, 53.00),
    ],
    # Small coastal island fragments.
    [
        (130.96, 54.16), (130.82, 54.15), (130.76, 54.03),
        (130.83, 53.94), (130.96, 53.98),
    ],
])


# Approximate fjords, channels, and lakes to subtract from land polygons.
WATER_CUTS = [
    # Portland Inlet and adjoining northern channels.
    [
        (130.62, 54.98), (130.43, 54.92), (130.22, 54.83),
        (130.18, 54.70), (130.29, 54.62), (130.43, 54.69),
        (130.52, 54.82), (130.62, 54.90),
    ],
    # Chatham Sound / Prince Rupert approach.
    [
        (130.47, 54.52), (130.29, 54.45), (130.22, 54.33),
        (130.29, 54.24), (130.42, 54.28), (130.54, 54.39),
        (130.57, 54.48),
    ],
    # Channel west of the mainland near Porcher Island.
    [
        (130.42, 54.17), (130.24, 54.10), (130.07, 53.98),
        (129.98, 53.84), (130.08, 53.78), (130.24, 53.86),
        (130.39, 54.00),
    ],
    # Southern inlet.
    [
        (130.18, 53.82), (130.04, 53.73), (129.91, 53.59),
        (129.89, 53.47), (130.00, 53.43), (130.12, 53.55),
        (130.28, 53.70),
    ],
]
# Small interior lakes are water even where the mainland polygon covers them.
WATER_CUTS.extend([
    [
        (129.74, 54.72), (129.63, 54.76), (129.55, 54.70),
        (129.58, 54.61), (129.70, 54.59), (129.78, 54.65),
    ],
    [
        (129.46, 53.86), (129.36, 53.91), (129.27, 53.84),
        (129.30, 53.74), (129.42, 53.72), (129.50, 53.79),
    ],
])


def is_land(lon_west, lat):
    land = point_in_polygon(lon_west, lat, MAINLAND)
    if not land:
        land = any(point_in_polygon(lon_west, lat, island) for island in ISLANDS)
    if land and any(
        point_in_polygon(lon_west, lat, water) for water in WATER_CUTS
    ):
        return 0
    return int(land)


def generate_matrix():
    matrix = np.zeros((N, N), dtype=np.uint8)
    for row in range(N):
        lat = 55.0 - row * STEP
        for col in range(N):
            lon_west = 131.0 - col * STEP
            matrix[row, col] = is_land(lon_west, lat)
    return matrix
def write_outputs(matrix):
    OUT.mkdir(parents=True, exist_ok=True)

    with (OUT / "candidate.csv").open(
        "w", newline="", encoding="ascii"
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for row in matrix:
            writer.writerow([int(value) for value in row])

    # Black water = 0; white land = 255.
    pixels = np.where(matrix == 1, 255, 0).astype(np.uint8)
    original = Image.fromarray(pixels, mode="L")
    original.save(OUT / "candidate.png")

    resampling = getattr(Image, "Resampling", Image).NEAREST
    enlarged = original.resize((N * 8, N * 8), resample=resampling)
    enlarged.save(OUT / "candidate_8x.png")


def validate_outputs(matrix):
    assert matrix.shape == (127, 127)
    assert matrix.size == 16129
    assert set(np.unique(matrix).tolist()) <= {0, 1}

    with (OUT / "candidate.csv").open(
        "r", newline="", encoding="ascii"
    ) as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 127
    assert all(len(row) == 127 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    original = Image.open(OUT / "candidate.png").convert("L")
    enlarged = Image.open(OUT / "candidate_8x.png").convert("L")
    assert original.size == (127, 127)
    assert enlarged.size == (1016, 1016)
    assert set(np.asarray(original).reshape(-1).tolist()) <= {0, 255}
    assert set(np.asarray(enlarged).reshape(-1).tolist()) <= {0, 255}

    expected = np.repeat(
        np.repeat(np.asarray(original), 8, axis=0), 8, axis=1
    )
    assert np.array_equal(np.asarray(enlarged), expected)
def main():
    matrix = generate_matrix()
    write_outputs(matrix)
    validate_outputs(matrix)
    print("matrix:", matrix.shape)
    print("valid values:", matrix.size)
    print("land:", int(matrix.sum()))
    print("water:", int(matrix.size - matrix.sum()))
    print("outputs:", OUT.resolve())


if __name__ == "__main__":
    main()
