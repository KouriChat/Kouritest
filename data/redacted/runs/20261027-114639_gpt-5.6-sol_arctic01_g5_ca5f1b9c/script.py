from pathlib import Path
import csv
import numpy as np
from PIL import Image

# Coordinates use (degrees west, degrees north).
# These are hand-written geographic approximations, not external map data.

LAND_POLYGONS = [
    # Eastern edge of Melville Island
    [
        (105.0, 74.95), (104.55, 75.05), (104.15, 75.35),
        (104.05, 75.85), (104.25, 76.25), (104.65, 76.55),
        (105.0, 76.65),
    ],

    # Bathurst Island
    [
        (103.65, 75.55), (103.10, 75.30), (102.35, 75.12),
        (101.75, 75.05), (101.10, 75.23), (100.45, 75.08),
        (99.90, 75.02), (99.25, 75.16), (98.75, 75.39),
        (98.25, 75.43), (97.75, 75.58), (97.45, 75.93),
        (97.80, 76.20), (97.55, 76.52), (97.85, 76.77),
        (98.40, 76.91), (99.10, 76.96), (99.65, 76.82),
        (100.15, 76.80), (100.65, 76.93), (101.15, 76.72),
        (101.75, 76.68), (102.20, 76.53), (102.85, 76.48),
        (103.35, 76.31), (103.72, 76.03),
    ],

    # Cornwallis Island
    [
        (97.65, 74.75), (97.20, 74.55), (96.45, 74.47),
        (95.75, 74.52), (95.10, 74.63), (94.45, 74.82),
        (93.75, 75.03), (93.48, 75.34), (93.72, 75.57),
        (94.25, 75.74), (95.00, 75.73), (95.55, 75.60),
        (96.18, 75.66), (96.75, 75.43), (97.25, 75.20),
    ],

    # Devon Island
    [
        (96.35, 75.05), (95.75, 74.78), (95.08, 74.65),
        (94.28, 74.55), (93.58, 74.36), (92.92, 74.20),
        (92.10, 74.07), (91.25, 74.01), (90.35, 74.00),
        (89.55, 74.11), (88.75, 74.16), (87.85, 74.14),
        (87.10, 74.27), (86.42, 74.36), (85.60, 74.34),
        (84.92, 74.39), (84.18, 74.52), (83.55, 74.69),
        (82.82, 74.88), (82.22, 74.98), (81.55, 75.05),
        (81.00, 75.20), (81.00, 76.60), (81.72, 76.63),
        (82.35, 76.70), (83.02, 76.76), (83.72, 76.70),
        (84.30, 76.58), (84.82, 76.38), (85.42, 76.34),
        (86.05, 76.43), (86.72, 76.53), (87.38, 76.55),
        (88.02, 76.48), (88.62, 76.32), (89.20, 76.20),
        (89.75, 76.07), (90.30, 76.02), (90.95, 76.17),
        (91.55, 76.35), (92.15, 76.51), (92.85, 76.61),
        (93.55, 76.70), (94.25, 76.82), (94.90, 76.76),
        (95.35, 76.62), (95.38, 76.32), (95.75, 76.15),
        (96.10, 75.92),
    ],

    # Prince of Wales Island, northern part
    [
        (102.65, 73.00), (102.40, 73.22), (102.15, 73.52),
        (101.75, 73.82), (101.20, 74.02), (100.55, 74.18),
        (99.80, 74.28), (99.05, 74.28), (98.45, 74.17),
        (97.90, 74.03), (97.50, 73.78), (97.20, 73.45),
        (97.08, 73.00),
    ],

    # Somerset Island, northern part
    [
        (96.10, 73.00), (95.92, 73.32), (95.62, 73.60),
        (95.17, 73.87), (94.58, 74.08), (93.92, 74.20),
        (93.35, 74.15), (92.82, 74.02), (92.25, 73.93),
        (91.70, 73.72), (91.18, 73.48), (90.75, 73.18),
        (90.62, 73.00),
    ],

    # Brodeur Peninsula, Baffin Island
    [
        (90.00, 73.00), (89.82, 73.30), (89.45, 73.56),
        (88.95, 73.76), (88.35, 73.88), (87.73, 73.85),
        (87.15, 73.70), (86.60, 73.43), (86.22, 73.00),
    ],

    # Borden Peninsula / northeastern Baffin edge
    [
        (84.65, 73.00), (84.45, 73.25), (84.08, 73.50),
        (83.55, 73.68), (82.90, 73.78), (82.25, 73.78),
        (81.60, 73.70), (81.00, 73.58), (81.00, 73.00),
    ],

    # Ellef Ringnes Island
    [
        (105.00, 78.14), (104.45, 77.92), (103.70, 77.70),
        (102.85, 77.55), (101.95, 77.46), (101.10, 77.48),
        (100.25, 77.55), (99.45, 77.50), (98.65, 77.62),
        (97.90, 77.79), (97.15, 78.02), (96.60, 78.27),
        (96.45, 78.57), (96.78, 78.83), (97.30, 79.00),
        (105.00, 79.00),
    ],

    # Amund Ringnes Island
    [
        (96.35, 78.20), (95.83, 78.04), (95.15, 78.02),
        (94.50, 78.12), (93.88, 78.25), (93.30, 78.44),
        (92.65, 78.56), (91.92, 78.72), (91.45, 78.91),
        (91.38, 79.00), (96.38, 79.00), (96.58, 78.72),
        (96.60, 78.43),
    ],

    # Axel Heiberg Island, southern end
    [
        (94.05, 79.00), (93.78, 78.82), (93.30, 78.60),
        (92.68, 78.47), (92.03, 78.45), (91.45, 78.55),
        (90.78, 78.51), (90.13, 78.61), (89.55, 78.76),
        (89.05, 79.00),
    ],

    # Southwestern Ellesmere Island
    [
        (86.15, 79.00), (86.18, 78.70), (85.92, 78.42),
        (85.30, 78.13), (84.98, 77.84), (85.18, 77.55),
        (85.00, 77.24), (84.60, 76.94), (84.17, 76.67),
        (83.62, 76.48), (82.95, 76.38), (82.30, 76.40),
        (81.60, 76.48), (81.00, 76.52), (81.00, 79.00),
    ],
]

# Ellipses: (center west, center north, west-radius, north-radius)
LAND_ELLIPSES = [
    # Byam Martin Island
    (104.18, 75.38, 0.62, 0.48),

    # Cameron, Vanier, Massey and Alexander islands
    (103.35, 76.62, 0.78, 0.38),
    (102.95, 77.05, 0.82, 0.30),
    (101.68, 77.08, 0.72, 0.28),
    (100.55, 76.93, 0.62, 0.27),

    # Lougheed Island edge
    (104.62, 77.48, 0.88, 0.38),

    # Helena and nearby islands
    (99.55, 76.92, 0.62, 0.22),
    (98.25, 76.98, 0.43, 0.18),

    # King Christian Island
    (102.20, 77.78, 0.72, 0.28),

    # Cornwall Island
    (94.65, 77.62, 1.38, 0.34),

    # Graham, Buckingham and North Kent islands
    (90.28, 77.42, 1.22, 0.35),
    (91.12, 77.15, 0.48, 0.20),
    (90.02, 76.72, 0.72, 0.30),

    # Baillie-Hamilton Island
    (94.35, 75.90, 0.73, 0.30),

    # Little Cornwallis Island
    (96.15, 75.52, 0.50, 0.29),

    # Truro and smaller Barrow Strait islands
    (97.18, 75.02, 0.48, 0.22),
    (96.90, 74.61, 0.43, 0.19),
    (95.75, 74.55, 0.48, 0.20),
    (95.10, 74.18, 0.54, 0.19),

    # Beechey and Prince Leopold islands
    (91.86, 74.72, 0.25, 0.12),
    (90.03, 74.04, 0.20, 0.12),

    # Small Norwegian Bay islands
    (87.85, 77.30, 0.48, 0.23),
    (86.70, 77.63, 0.34, 0.18),
]

# Hand-estimated inlets and channels cut from the coarse island outlines.
WATER_POLYGONS = [
    # Inlets in Bathurst Island
    [
        (101.05, 75.00), (100.82, 75.20), (100.72, 75.47),
        (100.78, 75.72), (101.02, 75.86), (101.18, 75.63),
        (101.20, 75.30),
    ],
    [
        (99.32, 76.98), (99.20, 76.72), (99.25, 76.52),
        (99.50, 76.35), (99.68, 76.51), (99.62, 76.76),
    ],

    # Western Cornwallis indentation
    [
        (97.70, 75.02), (97.30, 74.97), (96.98, 75.08),
        (97.12, 75.24), (97.43, 75.29),
    ],

    # Croker Bay-like indentation in Devon Island
    [
        (84.55, 76.64), (84.45, 76.39), (84.50, 76.13),
        (84.78, 75.96), (84.96, 76.16), (84.91, 76.43),
    ],

    # Southern Devon indentation
    [
        (91.62, 74.02), (91.38, 74.23), (91.34, 74.48),
        (91.58, 74.61), (91.82, 74.43), (91.83, 74.17),
    ],

    # Admiralty Inlet opening between Baffin peninsulas
    [
        (86.35, 73.00), (86.10, 73.30), (85.78, 73.52),
        (85.34, 73.62), (84.95, 73.45), (84.72, 73.00),
    ],
]

# Approximate inland lakes: (center west, center north, west-radius, north-radius)
WATER_ELLIPSES = [
    (101.65, 76.02, 0.20, 0.10),
    (99.45, 75.92, 0.22, 0.11),
    (95.25, 75.20, 0.20, 0.10),
    (92.75, 75.35, 0.25, 0.12),
    (88.15, 75.38, 0.28, 0.12),
    (84.00, 75.43, 0.25, 0.11),
    (99.45, 73.70, 0.20, 0.10),
    (93.45, 73.50, 0.20, 0.10),
    (101.00, 78.18, 0.28, 0.11),
    (94.45, 78.55, 0.20, 0.10),
]


def point_on_segment(x, y, x1, y1, x2, y2, eps=1e-10):
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > eps:
        return False
    return (
        min(x1, x2) - eps <= x <= max(x1, x2) + eps
        and min(y1, y2) - eps <= y <= max(y1, y2) + eps
    )


def point_in_polygon(x, y, polygon):
    """Boundary points are treated as inside."""
    inside = False
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        if point_on_segment(x, y, x1, y1, x2, y2):
            return True

        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if crossing_x > x:
                inside = not inside

    return inside


def point_in_ellipse(x, y, ellipse):
    cx, cy, rx, ry = ellipse
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


def classify_point(west, north):
    land = any(
        point_in_polygon(west, north, polygon)
        for polygon in LAND_POLYGONS
    )

    if not land:
        land = any(
            point_in_ellipse(west, north, ellipse)
            for ellipse in LAND_ELLIPSES
        )

    if land and any(
        point_in_polygon(west, north, polygon)
        for polygon in WATER_POLYGONS
    ):
        land = False

    if land and any(
        point_in_ellipse(west, north, ellipse)
        for ellipse in WATER_ELLIPSES
    ):
        land = False

    return 1 if land else 0


def generate_matrix():
    # Row 0 = 79 N; row 60 = 73 N.
    latitudes = [79.0 - row / 10.0 for row in range(61)]

    # Column 0 = 105 W; column 240 = 81 W.
    west_longitudes = [105.0 - col / 10.0 for col in range(241)]

    matrix = np.zeros((61, 241), dtype=np.uint8)

    for row, north in enumerate(latitudes):
        for col, west in enumerate(west_longitudes):
            matrix[row, col] = classify_point(west, north)

    return matrix


def validate_csv(path):
    with path.open("r", newline="", encoding="ascii") as file:
        rows = list(csv.reader(file))

    assert len(rows) == 61
    assert all(len(row) == 241 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert sum(len(row) for row in rows) == 14701


def validate_images(original_path, enlarged_path, expected):
    with Image.open(original_path) as image:
        original = np.asarray(image)
        assert image.size == (241, 61)
        assert set(np.unique(original).tolist()) <= {0, 255}
        assert np.array_equal(original, expected)

    with Image.open(enlarged_path) as image:
        enlarged = np.asarray(image)
        assert image.size == (1928, 488)
        assert set(np.unique(enlarged).tolist()) <= {0, 255}

    expected_8x = np.repeat(np.repeat(expected, 8, axis=0), 8, axis=1)
    assert np.array_equal(enlarged, expected_8x)


def main():
    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "candidate.csv"
    png_path = output_dir / "candidate.png"
    enlarged_path = output_dir / "candidate_8x.png"

    matrix = generate_matrix()

    assert matrix.shape == (61, 241)
    assert matrix.size == 14701
    assert set(np.unique(matrix).tolist()) <= {0, 1}

    with csv_path.open("w", newline="", encoding="ascii") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerows(matrix.tolist())

    pixels = matrix * np.uint8(255)
    original = Image.fromarray(pixels, mode="L")
    original.save(png_path, format="PNG")

    try:
        nearest = Image.Resampling.NEAREST
    except AttributeError:
        nearest = Image.NEAREST

    enlarged = original.resize((1928, 488), resample=nearest)
    enlarged.save(enlarged_path, format="PNG")

    validate_csv(csv_path)
    validate_images(png_path, enlarged_path, pixels)

    print("Generated and validated:")
    print(csv_path)
    print(png_path)
    print(enlarged_path)
    print(f"Matrix: {matrix.shape[0]} x {matrix.shape[1]}")
    print(f"Land points: {int(matrix.sum())}")
    print(f"Water points: {int(matrix.size - matrix.sum())}")


if __name__ == "__main__":
    main()
