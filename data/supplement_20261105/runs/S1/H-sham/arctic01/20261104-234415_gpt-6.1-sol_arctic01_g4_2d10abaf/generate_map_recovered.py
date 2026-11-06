from pathlib import Path
import csv

import numpy as np
from PIL import Image


# Coordinates use degrees west as positive values:
# 105 means 105W, 81 means 81W.
# Each polygon is an approximate land outline inferred from geographic knowledge.
LAND_POLYGONS = [
    # Melville Island
    [
        (105.5, 74.35),
        (104.0, 74.25),
        (102.3, 74.35),
        (100.7, 74.50),
        (99.1, 74.75),
        (97.7, 75.15),
        (96.65, 75.75),
        (96.05, 76.40),
        (96.65, 76.95),
        (98.30, 77.35),
        (100.00, 77.35),
        (101.70, 77.55),
        (103.60, 77.65),
        (105.5, 77.35),
    ],

    # Victoria Island northeastern portion
    [
        (105.5, 72.75),
        (105.5, 74.75),
        (104.25, 75.05),
        (103.00, 74.95),
        (101.90, 74.55),
        (101.05, 73.95),
        (100.25, 73.35),
        (99.70, 72.75),
        (101.70, 72.55),
        (103.70, 72.60),
    ],

    # Prince of Wales Island
    [
        (103.40, 75.45),
        (101.70, 75.75),
        (99.25, 75.65),
        (97.25, 75.25),
        (96.35, 74.55),
        (96.45, 73.80),
        (97.45, 73.10),
        (99.05, 72.65),
        (100.75, 72.45),
        (102.25, 72.60),
        (102.95, 73.45),
    ],

    # Somerset Island
    [
        (101.15, 74.40),
        (99.30, 74.45),
        (97.15, 74.25),
        (95.45, 73.80),
        (94.55, 73.20),
        (94.70, 72.75),
        (96.00, 72.45),
        (98.10, 72.40),
        (100.15, 72.65),
        (101.30, 73.30),
    ],

    # Devon Island
    [
        (95.10, 76.55),
        (94.00, 77.20),
        (91.25, 77.45),
        (88.30, 77.10),
        (85.20, 76.55),
        (82.55, 75.95),
        (81.25, 75.25),
        (82.25, 74.55),
        (85.10, 74.05),
        (88.75, 74.05),
        (91.70, 74.65),
        (94.45, 75.45),
    ],

    # Ellesmere Island, southwestern portion
    [
        (91.20, 79.00),
        (90.15, 78.45),
        (89.00, 77.55),
        (87.00, 76.90),
        (84.50, 76.70),
        (82.05, 77.15),
        (80.45, 77.90),
        (80.35, 79.00),
    ],

    # Axel Heiberg Island
    [
        (96.45, 79.00),
        (95.55, 79.55),
        (92.65, 79.75),
        (90.55, 79.25),
        (89.65, 78.50),
        (90.25, 77.75),
        (92.05, 77.30),
        (94.75, 77.50),
        (96.30, 78.20),
    ],

    # Amund Ringnes Island
    [
        (99.20, 78.65),
        (97.35, 78.95),
        (95.75, 78.50),
        (95.15, 77.75),
        (95.95, 77.05),
        (97.75, 76.90),
        (99.35, 77.45),
    ],

    # Ellef Ringnes Island
    [
        (103.75, 78.60),
        (102.05, 78.95),
        (100.05, 78.80),
        (98.85, 78.20),
        (99.10, 77.35),
        (100.95, 76.95),
        (102.95, 77.35),
        (104.05, 78.00),
    ],

    # Bathurst Island
    [
        (98.00, 77.20),
        (96.10, 77.50),
        (93.55, 77.50),
        (91.05, 77.20),
        (89.45, 76.55),
        (88.95, 75.80),
        (90.45, 75.20),
        (93.80, 75.10),
        (96.25, 75.45),
        (97.75, 76.15),
    ],

    # Cornwallis Island
    [
        (96.35, 76.20),
        (95.00, 76.65),
        (93.05, 76.50),
        (92.25, 75.80),
        (92.45, 75.05),
        (93.90, 74.60),
        (95.90, 74.80),
        (96.80, 75.50),
    ],

    # Northern Boothia Peninsula
    [
        (98.55, 73.10),
        (97.25, 73.40),
        (95.65, 73.20),
        (94.05, 72.80),
        (94.25, 72.35),
        (96.05, 72.10),
        (97.80, 72.30),
        (98.75, 72.70),
    ],

    # Northwestern Baffin Island edge
    [
        (82.10, 73.00),
        (82.45, 73.65),
        (82.20, 74.20),
        (81.40, 74.45),
        (80.55, 74.05),
        (79.95, 73.00),
    ],
]


# Approximate water channels that separate nearby islands.
# These are applied only after a point has landed inside a land polygon.
WATER_CUTOUTS = [
    # Eureka Sound between Axel Heiberg and Ellesmere
    [
        (90.00, 79.00),
        (91.55, 79.00),
        (91.25, 78.35),
        (90.50, 77.75),
        (89.70, 78.10),
    ],

    # Channel between Melville and Bathurst
    [
        (96.35, 77.10),
        (97.95, 77.35),
        (98.05, 76.45),
        (97.35, 75.55),
        (96.45, 75.70),
        (96.05, 76.40),
    ],

    # Channel between Ringnes islands and Melville
    [
        (98.45, 77.75),
        (100.00, 77.85),
        (100.05, 77.15),
        (99.10, 76.85),
        (98.45, 77.05),
    ],

    # Channel near Bathurst and Cornwallis
    [
        (95.75, 76.65),
        (97.10, 76.55),
        (97.20, 75.75),
        (96.55, 75.05),
        (95.85, 75.20),
        (95.55, 75.85),
    ],

    # Victoria Strait area between Victoria and Prince of Wales
    [
        (102.05, 74.75),
        (104.45, 74.95),
        (104.55, 73.35),
        (103.00, 73.05),
        (101.85, 73.55),
    ],

    # Franklin Strait area between Prince of Wales and Somerset
    [
        (99.00, 74.50),
        (101.05, 74.45),
        (101.25, 73.20),
        (100.10, 72.75),
        (98.95, 73.05),
    ],
]


def point_on_segment(x, y, ax, ay, bx, by, eps=1e-10):
    cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
    if abs(cross) > eps:
        return False

    return (
        min(ax, bx) - eps <= x <= max(ax, bx) + eps
        and min(ay, by) - eps <= y <= max(ay, by) + eps
    )


def point_in_polygon(x, y, polygon):
    """Ray casting with boundary points classified as inside."""
    inside = False
    n = len(polygon)

    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]

        if point_on_segment(x, y, ax, ay, bx, by):
            return True

        crosses = (ay > y) != (by > y)
        if crosses:
            crossing_x = ax + (y - ay) * (bx - ax) / (by - ay)
            if x < crossing_x:
                inside = not inside

    return inside


def is_land(west_longitude, latitude):
    land = any(
        point_in_polygon(west_longitude, latitude, polygon)
        for polygon in LAND_POLYGONS
    )

    if not land:
        return False

    in_channel = any(
        point_in_polygon(west_longitude, latitude, polygon)
        for polygon in WATER_CUTOUTS
    )

    return not in_channel


def build_matrix():
    rows = 61
    cols = 241
    matrix = np.zeros((rows, cols), dtype=np.uint8)

    for row in range(rows):
        latitude = 79.0 - row * 0.1

        for col in range(cols):
            west_longitude = 105.0 - col * 0.1
            matrix[row, col] = 1 if is_land(west_longitude, latitude) else 0

    return matrix


def write_csv(matrix, path):
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for row in matrix:
            writer.writerow([int(value) for value in row])


def write_images(matrix, raw_path, scaled_path):
    # 0 is water and 255 is land.
    pixels = (matrix * 255).astype(np.uint8)
    raw_image = Image.fromarray(pixels, mode="L")
    raw_image.save(raw_path, format="PNG")

    scaled_image = raw_image.resize(
        (241 * 8, 61 * 8),
        resample=Image.Resampling.NEAREST,
    )
    scaled_image.save(scaled_path, format="PNG")


def validate_outputs(matrix, csv_path, raw_path, scaled_path):
    assert matrix.shape == (61, 241)
    assert matrix.size == 14701
    assert set(np.unique(matrix)).issubset({0, 1})

    with csv_path.open("r", newline="", encoding="ascii") as handle:
        rows = list(csv.reader(handle))

    assert len(rows) == 61
    assert all(len(row) == 241 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert all(value != "" for row in rows for value in row)

    raw_image = Image.open(raw_path)
    assert raw_image.size == (241, 61)
    assert raw_image.mode == "L"
    assert set(raw_image.getdata()).issubset({0, 255})

    scaled_image = Image.open(scaled_path)
    assert scaled_image.size == (1928, 488)
    assert scaled_image.mode == "L"
    assert set(scaled_image.getdata()).issubset({0, 255})

    scaled_pixels = np.asarray(scaled_image)
    raw_pixels = np.asarray(raw_image)

    # Every 8x8 block must be identical to the corresponding source pixel.
    for row in range(61):
        for col in range(241):
            block = scaled_pixels[
                row * 8:(row + 1) * 8,
                col * 8:(col + 1) * 8,
            ]
            assert np.all(block == raw_pixels[row, col])


def main():
    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = build_matrix()

    csv_path = output_dir / "candidate.csv"
    raw_png_path = output_dir / "candidate.png"
    scaled_png_path = output_dir / "candidate_8x.png"

    write_csv(matrix, csv_path)
    write_images(matrix, raw_png_path, scaled_png_path)
    validate_outputs(matrix, csv_path, raw_png_path, scaled_png_path)

    print("matrix:", matrix.shape)
    print("values:", matrix.size)
    print("csv:", csv_path)
    print("raw png:", raw_png_path, "241x61")
    print("scaled png:", scaled_png_path, "1928x488")
    print("validation: OK")


if __name__ == "__main__":
    main()
