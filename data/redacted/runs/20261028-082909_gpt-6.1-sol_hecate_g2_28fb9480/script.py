from pathlib import Path

import numpy as np
from PIL import Image


ROWS = 127
COLS = 127
WEST = -131.0
EAST = -129.0
NORTH = 55.0
SOUTH = 53.0


def point_in_polygon(x, y, polygon):
    """Ray-casting test; points on a boundary are treated as land."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) < 1e-12:
            if (
                min(x1, x2) - 1e-12 <= x <= max(x1, x2) + 1e-12
                and min(y1, y2) - 1e-12 <= y <= max(y1, y2) + 1e-12
            ):
                return True
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing_x:
                inside = not inside
    return inside


def in_ellipse(x, y, cx, cy, rx, ry):
    dx = (x - cx) / rx
    dy = (y - cy) / ry
    return dx * dx + dy * dy <= 1.0


def build_rules():
    mainland = [
        (-130.00, 55.00), (-129.00, 55.00), (-129.00, 53.00),
        (-129.18, 53.00), (-129.22, 53.15), (-129.28, 53.30),
        (-129.32, 53.48), (-129.42, 53.62), (-129.48, 53.78),
        (-129.55, 53.93), (-129.68, 54.05), (-129.72, 54.20),
        (-129.88, 54.32), (-130.00, 54.48), (-130.18, 54.57),
        (-130.28, 54.70), (-130.22, 54.82), (-130.12, 54.92),
    ]
    mainland_water = [
        [
            (-130.28, 54.87), (-130.06, 54.81), (-129.84, 54.72),
            (-129.62, 54.64), (-129.46, 54.51), (-129.51, 54.39),
            (-129.67, 54.35), (-129.85, 54.42), (-130.03, 54.52),
            (-130.21, 54.59), (-130.34, 54.72),
        ],
        [
            (-130.25, 54.47), (-130.00, 54.45), (-129.77, 54.35),
            (-129.57, 54.22), (-129.62, 54.08), (-129.77, 54.02),
            (-129.91, 54.13), (-130.06, 54.25), (-130.27, 54.31),
        ],
        [
            (-129.78, 53.99), (-129.61, 53.87), (-129.48, 53.74),
            (-129.51, 53.64), (-129.66, 53.66), (-129.80, 53.75),
            (-129.87, 53.87),
        ],
    ]


    extra_land = [
        [
            (-131.05, 54.38), (-131.18, 54.22), (-131.36, 54.10),
            (-131.50, 53.96), (-131.45, 53.81), (-131.58, 53.68),
            (-131.46, 53.55), (-131.30, 53.47), (-131.18, 53.34),
            (-131.00, 53.20), (-130.96, 53.00), (-132.60, 53.00),
            (-132.60, 54.50), (-131.30, 54.50),
        ],
        [
            (-131.30, 54.31), (-131.10, 54.34), (-130.97, 54.29),
            (-130.96, 54.23), (-131.16, 54.18), (-131.34, 54.22),
        ],
        [
            (-131.00, 54.69), (-130.78, 54.74), (-130.55, 54.66),
            (-130.45, 54.53), (-130.55, 54.43), (-130.78, 54.43),
            (-130.94, 54.51),
        ],
        [
            (-130.96, 54.75), (-130.79, 54.79), (-130.65, 54.73),
            (-130.68, 54.62), (-130.84, 54.60), (-130.98, 54.66),
        ],
        [
            (-130.62, 54.05), (-130.39, 54.13), (-130.13, 54.05),
            (-130.00, 53.89), (-130.16, 53.74), (-130.43, 53.77),
            (-130.61, 53.90),
        ],
        [
            (-130.08, 53.61), (-129.82, 53.72), (-129.58, 53.69),
            (-129.38, 53.55), (-129.35, 53.37), (-129.47, 53.20),
            (-129.72, 53.09), (-129.98, 53.19), (-130.09, 53.40),
        ],
        [
            (-130.00, 53.22), (-129.79, 53.28), (-129.56, 53.20),
            (-129.42, 53.04), (-129.48, 53.00), (-130.02, 53.00),
        ],
        [
            (-129.66, 53.89), (-129.47, 53.99), (-129.29, 53.90),
            (-129.23, 53.72), (-129.37, 53.59), (-129.57, 53.64),
            (-129.70, 53.77),
        ],
        [
            (-129.82, 54.12), (-129.57, 54.18), (-129.32, 54.06),
            (-129.17, 53.83), (-129.23, 53.56), (-129.44, 53.31),
            (-129.69, 53.13), (-129.94, 53.19), (-129.98, 53.52),
            (-129.91, 53.84),
        ],
    ]
    small_ellipses = [
        (-130.72, 54.34, 0.055, 0.035),
        (-130.54, 54.30, 0.045, 0.030),
        (-130.83, 54.13, 0.035, 0.025),
        (-130.72, 53.99, 0.040, 0.025),
        (-130.34, 53.69, 0.045, 0.030),
        (-130.10, 53.57, 0.035, 0.025),
        (-129.98, 53.39, 0.045, 0.030),
        (-129.21, 53.35, 0.045, 0.030),
        (-129.07, 53.70, 0.035, 0.025),
        (-130.43, 54.57, 0.030, 0.020),
    ]
    return mainland, mainland_water, extra_land, small_ellipses


def classify_point(lon, lat, mainland, mainland_water, extra_land, ellipses):
    for polygon in extra_land:
        if point_in_polygon(lon, lat, polygon):
            return 1
    for cx, cy, rx, ry in ellipses:
        if in_ellipse(lon, lat, cx, cy, rx, ry):
            return 1
    if not point_in_polygon(lon, lat, mainland):
        return 0
    for polygon in mainland_water:
        if point_in_polygon(lon, lat, polygon):
            return 0
    return 1


def generate_matrix():
    rules = build_rules()
    matrix = np.zeros((ROWS, COLS), dtype=np.uint8)
    for row in range(ROWS):
        lat = NORTH - row * (NORTH - SOUTH) / (ROWS - 1)
        for col in range(COLS):
            lon = WEST + col * (EAST - WEST) / (COLS - 1)
            matrix[row, col] = classify_point(lon, lat, *rules)
    return matrix


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = generate_matrix()

    assert matrix.shape == (127, 127)
    assert matrix.size == 16129
    assert set(np.unique(matrix)).issubset({0, 1})

    np.savetxt(output_dir / "candidate.csv", matrix, fmt="%d", delimiter=",")

    pixels = np.where(matrix == 1, 255, 0).astype(np.uint8)
    raw = Image.fromarray(pixels, mode="L")
    raw.save(output_dir / "candidate.png")

    enlarged = raw.resize((1016, 1016), Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")

    expected = np.repeat(np.repeat(pixels, 8, axis=0), 8, axis=1)
    assert raw.size == (127, 127)
    assert enlarged.size == (1016, 1016)
    assert set(np.unique(pixels)).issubset({0, 255})
    assert np.array_equal(np.asarray(enlarged), expected)

    print("matrix:", matrix.shape, "valid:", matrix.size)
    print("outputs written under:", output_dir.resolve())


if __name__ == "__main__":
    main()
