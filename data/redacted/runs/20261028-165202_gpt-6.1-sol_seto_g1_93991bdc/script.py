from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image


WIDTH = 253
HEIGHT = 64
LON_MIN = 129.0
LON_MAX = 137.0
LAT_MAX = 35.0
DLON = 2.0 / 63.0
DLAT = 2.0 / 63.0

# Polygon vertices are (longitude, latitude).
# These are deliberately hand-written geographic approximations.
LAND_POLYGONS = [
    # Southern Korean Peninsula and Busan area.
    [
        (129.0, 35.0), (129.0, 35.25), (129.75, 35.35),
        (130.25, 35.15), (130.35, 34.95), (129.85, 34.88),
        (129.35, 34.92),
    ],

    # Kyushu, northern and eastern portions.
    [
        (129.35, 34.15), (129.45, 34.55), (129.85, 34.75),
        (130.25, 34.65), (130.65, 34.35), (131.0, 34.05),
        (131.25, 33.65), (131.35, 33.1), (130.95, 33.0),
        (129.0, 33.0), (129.0, 33.55),
    ],

    # Western Honshu and the Chugoku region.
    [
        (130.65, 35.0), (137.0, 35.0), (137.0, 34.25),
        (136.65, 34.08), (136.2, 34.0), (135.75, 33.95),
        (135.25, 34.05), (134.85, 34.18), (134.35, 34.3),
        (133.95, 34.32), (133.55, 34.25), (133.1, 34.12),
        (132.7, 34.0), (132.25, 33.92), (131.85, 33.9),
        (131.45, 34.0), (131.05, 34.15), (130.75, 34.38),
        (130.55, 34.65),
    ],

    # Kii Peninsula and the central Honshu coast.
    [
        (134.7, 34.35), (135.15, 34.45), (135.65, 34.55),
        (136.05, 34.65), (136.55, 34.85), (137.0, 35.0),
        (137.0, 33.0), (136.15, 33.0), (135.85, 33.25),
        (135.65, 33.5), (135.35, 33.65), (135.05, 33.75),
        (134.8, 33.9), (134.55, 34.05),
    ],

    # Shikoku.
    [
        (131.85, 33.85), (132.25, 34.05), (132.9, 34.12),
        (133.55, 34.05), (134.1, 33.95), (134.65, 33.82),
        (134.95, 33.55), (134.85, 33.2), (134.45, 33.0),
        (133.55, 33.0), (132.9, 33.12), (132.3, 33.25),
        (131.95, 33.5),
    ],

    # Awaji Island.
    [
        (134.55, 34.65), (134.85, 34.7), (135.0, 34.45),
        (134.95, 34.1), (134.7, 34.05), (134.55, 34.28),
    ],

    # Large islands near western Seto Inland Sea.
    [
        (131.0, 34.35), (131.25, 34.48), (131.55, 34.4),
        (131.65, 34.15), (131.45, 34.02), (131.15, 34.1),
    ],
    [
        (132.3, 34.25), (132.65, 34.38), (132.95, 34.3),
        (133.0, 34.05), (132.75, 33.92), (132.45, 34.0),
    ],
    [
        (133.1, 34.4), (133.45, 34.5), (133.75, 34.38),
        (133.8, 34.15), (133.5, 34.08), (133.2, 34.18),
    ],
]

# Island centers and approximate radii:
# (longitude, latitude, horizontal radius, vertical radius)
ISLAND_ELLIPSES = [
    # Tsushima and nearby islands.
    (129.32, 34.45, 0.13, 0.28),
    (129.42, 34.02, 0.12, 0.22),
    (129.60, 33.75, 0.08, 0.13),

    # Goto and northwest Kyushu islands.
    (129.20, 33.85, 0.10, 0.18),
    (129.38, 33.55, 0.10, 0.17),
    (129.72, 33.42, 0.08, 0.14),
    (129.85, 33.18, 0.07, 0.11),

    # Hiroshima and Yamaguchi island groups.
    (130.95, 33.98, 0.06, 0.10),
    (131.35, 34.03, 0.08, 0.12),
    (131.72, 34.12, 0.07, 0.11),
    (131.92, 34.28, 0.08, 0.12),
    (132.18, 34.25, 0.07, 0.11),
    (132.55, 34.16, 0.09, 0.12),
    (132.82, 34.15, 0.06, 0.10),

    # Geiyo Islands.
    (132.95, 34.08, 0.09, 0.14),
    (133.18, 34.08, 0.07, 0.11),
    (133.38, 34.12, 0.08, 0.12),
    (133.62, 34.2, 0.07, 0.1),
    (133.78, 34.05, 0.06, 0.1),
    (134.02, 34.12, 0.07, 0.1),
    (134.25, 34.17, 0.06, 0.09),

    # Shodoshima and islands northeast of Shikoku.
    (134.28, 34.52, 0.16, 0.13),
    (134.55, 34.48, 0.08, 0.09),
    (134.72, 34.45, 0.06, 0.08),
    (134.88, 34.35, 0.05, 0.07),

    # Islands in the Kii and Ise approaches.
    (135.02, 34.28, 0.06, 0.09),
    (135.25, 34.32, 0.05, 0.07),
    (135.55, 34.35, 0.05, 0.08),
    (136.02, 34.38, 0.05, 0.07),
    (136.48, 34.47, 0.05, 0.08),

    # Small islands around Shikoku.
    (132.18, 33.62, 0.07, 0.1),
    (132.55, 33.55, 0.06, 0.08),
    (132.95, 33.48, 0.06, 0.08),
    (133.32, 33.38, 0.07, 0.08),
    (133.75, 33.3, 0.06, 0.08),
    (134.18, 33.28, 0.05, 0.07),
    (134.62, 33.2, 0.05, 0.07),

    # Ise Bay islands and offshore rocks.
    (136.72, 34.55, 0.04, 0.07),
    (136.84, 34.35, 0.03, 0.05),
]


def point_in_polygon(x, y, polygon):
    inside = False
    n = len(polygon)
    j = n - 1

    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # Boundary points are classified as land.
        cross = (x - xi) * (yj - yi) - (y - yi) * (xj - xi)
        if abs(cross) < 1e-12:
            if min(xi, xj) - 1e-12 <= x <= max(xi, xj) + 1e-12:
                if min(yi, yj) - 1e-12 <= y <= max(yi, yj) + 1e-12:
                    return True

        intersects = ((yi > y) != (yj > y))
        if intersects:
            x_cross = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < x_cross:
                inside = not inside

        j = i

    return inside


def point_in_ellipse(x, y, cx, cy, rx, ry):
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


def is_land(lon, lat):
    for polygon in LAND_POLYGONS:
        if point_in_polygon(lon, lat, polygon):
            return True

    for cx, cy, rx, ry in ISLAND_ELLIPSES:
        if point_in_ellipse(lon, lat, cx, cy, rx, ry):
            return True

    return False


def build_matrix():
    matrix = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    for row in range(HEIGHT):
        lat = LAT_MAX - row * DLAT
        for col in range(WIDTH):
            lon = LON_MIN + col * DLON
            matrix[row, col] = 1 if is_land(lon, lat) else 0

    return matrix


def write_outputs(matrix):
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        for row in matrix:
            writer.writerow([str(int(value)) for value in row])

    # Black is water and white is land.
    image_array = np.where(matrix == 1, 255, 0).astype(np.uint8)
    image = Image.fromarray(image_array, mode="L")
    image.save(output_dir / "candidate.png")

    enlarged = image.resize((WIDTH * 8, HEIGHT * 8), Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")


def validate_outputs(matrix):
    assert matrix.shape == (64, 253)
    assert matrix.size == 16192
    assert np.all((matrix == 0) | (matrix == 1))

    output_dir = Path("outputs")
    with (output_dir / "candidate.csv").open("r", encoding="ascii") as f:
        rows = list(csv.reader(f))

    assert len(rows) == 64
    assert all(len(row) == 253 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    original = Image.open(output_dir / "candidate.png")
    enlarged = Image.open(output_dir / "candidate_8x.png")

    assert original.size == (253, 64)
    assert enlarged.size == (2024, 512)
    assert set(original.convert("L").getdata()) <= {0, 255}
    assert set(enlarged.convert("L").getdata()) <= {0, 255}

    original_pixels = np.asarray(original.convert("L"))
    enlarged_pixels = np.asarray(enlarged.convert("L"))
    expected = np.repeat(np.repeat(original_pixels, 8, axis=0), 8, axis=1)
    assert np.array_equal(enlarged_pixels, expected)


if __name__ == "__main__":
    result = build_matrix()
    write_outputs(result)
    validate_outputs(result)
    print("Generated and validated:")
    print("  outputs/candidate.csv")
    print("  outputs/candidate.png")
    print("  outputs/candidate_8x.png")
    print("  matrix shape:", result.shape)
    print("  valid values:", result.size)
