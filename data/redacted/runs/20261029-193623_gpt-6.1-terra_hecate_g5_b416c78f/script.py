import csv
from pathlib import Path

from PIL import Image


SIZE = 127
SCALE = 8
OUTPUT = Path("outputs")

# Hand-written approximate land polygons, (longitude, latitude).
# The mainland polygon is deliberately clipped only by the task frame.
LAND = [
    [
        (-129.00, 55.20), (-130.35, 55.20), (-130.34, 54.98),
        (-130.26, 54.92), (-130.20, 54.84), (-130.10, 54.78),
        (-130.13, 54.72), (-130.03, 54.67), (-129.98, 54.58),
        (-130.05, 54.51), (-129.96, 54.44), (-129.90, 54.35),
        (-129.92, 54.27), (-129.82, 54.18), (-129.76, 54.08),
        (-129.66, 54.04), (-129.63, 53.96), (-129.52, 53.91),
        (-129.46, 53.82), (-129.36, 53.77), (-129.29, 53.68),
        (-129.17, 53.60), (-129.10, 53.49), (-128.90, 53.45),
        (-128.80, 55.20),
    ],
    # Dundas, Melville, Stephens, Digby, and Kaien islands.
    [
        (-130.97, 54.73), (-130.92, 54.78), (-130.82, 54.80),
        (-130.71, 54.77), (-130.65, 54.72), (-130.63, 54.66),
        (-130.68, 54.61), (-130.73, 54.57), (-130.83, 54.56),
        (-130.91, 54.60), (-130.96, 54.66),
    ],
    [
        (-130.65, 54.78), (-130.58, 54.83), (-130.49, 54.82),
        (-130.43, 54.77), (-130.45, 54.71), (-130.53, 54.68),
        (-130.60, 54.71),
    ],
    [
        (-130.69, 54.49), (-130.64, 54.54), (-130.56, 54.53),
        (-130.50, 54.48), (-130.51, 54.42), (-130.58, 54.39),
        (-130.65, 54.41),
    ],
    [
        (-130.52, 54.34), (-130.48, 54.39), (-130.40, 54.39),
        (-130.35, 54.35), (-130.36, 54.30), (-130.42, 54.27),
        (-130.49, 54.28),
    ],
    [
        (-130.36, 54.34), (-130.32, 54.37), (-130.25, 54.36),
        (-130.21, 54.32), (-130.24, 54.27), (-130.31, 54.26),
        (-130.35, 54.29),
    ],
]
LAND += [
    # Porcher Island and small islands near the Skeena mouth.
    [
        (-130.67, 54.27), (-130.63, 54.33), (-130.55, 54.34),
        (-130.49, 54.30), (-130.43, 54.26), (-130.36, 54.23),
        (-130.33, 54.17), (-130.36, 54.11), (-130.42, 54.07),
        (-130.46, 54.02), (-130.54, 54.01), (-130.60, 54.06),
        (-130.64, 54.12), (-130.68, 54.18),
    ],
    [
        (-130.37, 54.20), (-130.32, 54.24), (-130.26, 54.23),
        (-130.23, 54.18), (-130.26, 54.13), (-130.33, 54.13),
    ],
    [
        (-130.27, 54.12), (-130.22, 54.16), (-130.16, 54.14),
        (-130.13, 54.09), (-130.17, 54.05), (-130.23, 54.06),
    ],
    # Banks Island.
    [
        (-130.49, 53.83), (-130.43, 53.88), (-130.35, 53.88),
        (-130.30, 53.84), (-130.23, 53.82), (-130.17, 53.76),
        (-130.15, 53.69), (-130.09, 53.63), (-130.08, 53.55),
        (-130.04, 53.49), (-130.07, 53.42), (-130.03, 53.35),
        (-130.05, 53.27), (-130.10, 53.20), (-130.16, 53.14),
        (-130.23, 53.12), (-130.29, 53.17), (-130.34, 53.24),
        (-130.38, 53.31), (-130.42, 53.39), (-130.43, 53.47),
        (-130.47, 53.54), (-130.45, 53.61), (-130.49, 53.68),
        (-130.46, 53.74), (-130.50, 53.79),
    ],
    # Pitt Island.
    [
        (-130.04, 53.85), (-129.97, 53.89), (-129.89, 53.87),
        (-129.82, 53.82), (-129.74, 53.80), (-129.68, 53.74),
        (-129.61, 53.72), (-129.57, 53.65), (-129.55, 53.57),
        (-129.50, 53.52), (-129.53, 53.45), (-129.59, 53.39),
        (-129.67, 53.35), (-129.75, 53.36), (-129.81, 53.41),
        (-129.85, 53.48), (-129.91, 53.52), (-129.91, 53.59),
        (-129.97, 53.65), (-129.99, 53.73), (-130.05, 53.78),
    ],
]
LAND += [
    # Campania Island and Estevan Island group.
    [
        (-129.93, 53.31), (-129.85, 53.35), (-129.76, 53.33),
        (-129.70, 53.27), (-129.67, 53.20), (-129.70, 53.12),
        (-129.76, 53.07), (-129.84, 53.08), (-129.90, 53.14),
        (-129.94, 53.22),
    ],
    [
        (-129.62, 53.18), (-129.55, 53.23), (-129.46, 53.21),
        (-129.40, 53.15), (-129.41, 53.08), (-129.48, 53.02),
        (-129.56, 53.01), (-129.62, 53.07),
    ],
    [
        (-129.39, 53.16), (-129.31, 53.19), (-129.23, 53.15),
        (-129.19, 53.08), (-129.21, 52.98), (-129.38, 52.98),
        (-129.43, 53.06),
    ],
    # Islands along the eastern edge of the frame.
    [
        (-129.48, 53.78), (-129.41, 53.82), (-129.34, 53.79),
        (-129.29, 53.72), (-129.27, 53.65), (-129.31, 53.59),
        (-129.37, 53.58), (-129.43, 53.63), (-129.47, 53.69),
    ],
    [
        (-129.27, 53.55), (-129.20, 53.60), (-129.12, 53.58),
        (-129.06, 53.53), (-129.02, 53.45), (-129.05, 53.37),
        (-129.12, 53.33), (-129.20, 53.36), (-129.25, 53.42),
    ],
    [
        (-129.17, 53.30), (-129.10, 53.34), (-129.02, 53.32),
        (-128.94, 53.27), (-128.92, 53.13), (-129.00, 53.11),
        (-129.09, 53.15), (-129.15, 53.21),
    ],
]


def contains(polygon, x, y):
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if (abs(cross) < 1e-12 and min(x1, x2) <= x <= max(x1, x2)
                and min(y1, y2) <= y <= max(y1, y2)):
            return True
        if (y1 > y) != (y2 > y):
            hit = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < hit:
                inside = not inside
        previous = current
    return inside


def build_matrix():
    matrix = []
    for row in range(SIZE):
        latitude = 55.0 - 2.0 * row / (SIZE - 1)
        line = []
        for col in range(SIZE):
            longitude = -131.0 + 2.0 * col / (SIZE - 1)
            land = any(contains(poly, longitude, latitude) for poly in LAND)
            line.append(int(land))
        matrix.append(line)
    return matrix


def write_csv(matrix):
    path = OUTPUT / "candidate.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(matrix)
    return path


def write_images(matrix):
    pixels = [255 * value for line in matrix for value in line]
    original = Image.new("L", (SIZE, SIZE))
    original.putdata(pixels)
    original.save(OUTPUT / "candidate.png")
    resampling = getattr(Image, "Resampling", Image)
    enlarged = original.resize(
        (SIZE * SCALE, SIZE * SCALE), resampling.NEAREST
    )
    enlarged.save(OUTPUT / "candidate_8x.png")


def validate(matrix, csv_path):
    assert len(matrix) == SIZE
    assert all(len(line) == SIZE for line in matrix)
    assert sum(len(line) for line in matrix) == 16129
    assert all(value in (0, 1) for line in matrix for value in line)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == SIZE
    assert all(len(row) == SIZE for row in rows)
    assert all(value in ("0", "1") for row in rows for value in row)
    assert [[int(value) for value in row] for row in rows] == matrix


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()
    csv_path = write_csv(matrix)
    write_images(matrix)
    validate(matrix, csv_path)
    print("validated", SIZE * SIZE, "binary cells")


if __name__ == "__main__":
    main()
