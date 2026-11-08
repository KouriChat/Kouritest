from pathlib import Path
import csv

from PIL import Image


# Handwritten approximate land polygons, as (longitude, latitude).
LAND = [
    # Eastern Melville Island.
    [
        (-108.0, 77.2), (-106.8, 77.1), (-105.8, 76.95),
        (-105.0, 76.85), (-104.4, 76.65), (-104.0, 76.42),
        (-104.35, 76.23), (-104.05, 76.05), (-104.4, 75.88),
        (-104.2, 75.68), (-104.65, 75.47), (-105.25, 75.36),
        (-105.85, 75.18), (-106.7, 75.15), (-108.0, 75.25),
    ],
    # Bathurst Island.
    [
        (-101.75, 76.95), (-101.05, 77.12), (-100.3, 77.20),
        (-99.7, 77.12), (-99.2, 76.95), (-98.55, 76.95),
        (-98.15, 76.78), (-97.55, 76.65), (-97.15, 76.43),
        (-96.55, 76.30), (-96.25, 76.05), (-96.55, 75.85),
        (-97.05, 75.76), (-97.4, 75.52), (-98.0, 75.40),
        (-98.65, 75.43), (-99.2, 75.57), (-99.65, 75.68),
        (-100.15, 75.62), (-100.75, 75.82), (-101.35, 75.89),
        (-101.8, 76.12), (-102.15, 76.34), (-102.0, 76.55),
        (-101.55, 76.70),
    ],
    # Western Bathurst satellite island.
    [
        (-103.35, 76.72), (-102.9, 76.85), (-102.35, 76.82),
        (-102.08, 76.67), (-102.3, 76.49), (-102.9, 76.45),
        (-103.35, 76.54),
    ],
    # Cornwallis Island.
    [
        (-97.05, 75.75), (-96.55, 75.85), (-95.85, 75.82),
        (-95.15, 75.68), (-94.65, 75.45), (-94.45, 75.20),
        (-94.62, 75.00), (-94.35, 74.82), (-94.75, 74.62),
        (-95.35, 74.60), (-95.85, 74.72), (-96.35, 74.83),
        (-96.75, 75.04), (-97.05, 75.28), (-97.25, 75.50),
    ],
]

# Additional islands and the northern Ellesmere outline.
LAND.extend([
    # Little Cornwallis Island.
    [
        (-96.75, 76.00), (-96.4, 76.09), (-96.08, 76.05),
        (-95.95, 75.94), (-96.23, 75.86), (-96.62, 75.88),
    ],
    # Devon Island.
    [
        (-93.75, 76.96), (-93.10, 77.03), (-92.45, 76.96),
        (-91.72, 76.98), (-91.08, 76.87), (-90.45, 76.91),
        (-89.78, 76.81), (-89.12, 76.81), (-88.45, 76.73),
        (-87.75, 76.75), (-87.10, 76.63), (-86.43, 76.66),
        (-85.78, 76.56), (-85.10, 76.52), (-84.45, 76.44),
        (-83.77, 76.40), (-83.12, 76.28), (-82.45, 76.28),
        (-81.82, 76.16), (-80.95, 76.14), (-80.60, 75.98),
        (-81.15, 75.87), (-81.95, 75.83), (-82.65, 75.74),
        (-83.30, 75.73), (-83.85, 75.61), (-84.38, 75.56),
        (-85.03, 75.53), (-85.65, 75.47), (-86.30, 75.45),
        (-86.90, 75.43), (-87.55, 75.39), (-88.10, 75.46),
        (-88.62, 75.53), (-89.05, 75.64), (-89.55, 75.72),
        (-90.05, 75.73), (-90.45, 75.82), (-90.95, 75.79),
        (-91.45, 75.84), (-91.90, 75.94), (-92.35, 76.00),
        (-92.85, 76.13), (-93.17, 76.30), (-93.62, 76.40),
        (-93.91, 76.60), (-94.10, 76.78),
    ],
    # Southwestern Ellesmere Island.
    [
        (-95.1, 79.3), (-94.6, 79.00), (-94.15, 78.82),
        (-93.75, 78.64), (-93.20, 78.53), (-92.72, 78.37),
        (-92.20, 78.30), (-91.75, 78.13), (-91.15, 78.05),
        (-90.75, 77.87), (-90.15, 77.82), (-89.62, 77.65),
        (-89.15, 77.53), (-88.55, 77.44), (-88.15, 77.27),
        (-87.62, 77.17), (-87.10, 77.04), (-86.55, 76.96),
        (-86.02, 76.83), (-85.48, 76.77), (-84.90, 76.69),
        (-84.25, 76.64), (-83.68, 76.61), (-83.10, 76.57),
        (-82.50, 76.57), (-81.95, 76.50), (-81.40, 76.49),
        (-80.70, 76.41), (-80.40, 79.3),
    ],
])

LAND.extend([
    # Cornwall Island.
    [
        (-96.75, 77.86), (-96.15, 78.02), (-95.48, 78.03),
        (-94.90, 77.91), (-94.60, 77.72), (-94.95, 77.55),
        (-95.52, 77.46), (-96.10, 77.52), (-96.58, 77.64),
    ],
    # Graham Island.
    [
        (-91.85, 77.82), (-91.38, 77.92), (-90.78, 77.89),
        (-90.40, 77.76), (-90.65, 77.60), (-91.17, 77.53),
        (-91.65, 77.61),
    ],
    # Small islands between Bathurst and Devon.
    [
        (-94.85, 76.78), (-94.45, 76.88), (-94.10, 76.83),
        (-94.05, 76.68), (-94.45, 76.62), (-94.80, 76.65),
    ],
    [
        (-95.85, 76.90), (-95.45, 77.02), (-95.05, 76.96),
        (-95.08, 76.83), (-95.45, 76.78), (-95.80, 76.81),
    ],
    # Prince of Wales Island.
    [
        (-101.0, 72.5), (-100.95, 73.10), (-100.72, 73.43),
        (-100.35, 73.73), (-100.12, 74.02), (-99.77, 74.20),
        (-99.58, 74.43), (-99.15, 74.56), (-98.78, 74.66),
        (-98.35, 74.61), (-98.02, 74.50), (-97.65, 74.53),
        (-97.30, 74.40), (-96.98, 74.21), (-96.63, 74.13),
        (-96.38, 73.92), (-96.13, 73.77), (-96.10, 73.52),
        (-95.85, 73.30), (-95.93, 73.03), (-96.0, 72.5),
    ],
    # Somerset Island.
    [
        (-95.35, 72.5), (-95.25, 73.05), (-95.12, 73.33),
        (-95.02, 73.58), (-94.73, 74.06), (-94.48, 74.25),
        (-94.23, 74.49), (-93.85, 74.61), (-93.45, 74.70),
        (-93.02, 74.70), (-92.64, 74.60), (-92.24, 74.52),
        (-91.87, 74.34), (-91.65, 74.12), (-91.50, 73.90),
        (-91.56, 73.68), (-91.42, 73.47), (-91.50, 73.25),
        (-91.42, 73.00), (-91.32, 72.5),
    ],
])

LAND.extend([
    # Russell Island.
    [
        (-100.35, 74.40), (-99.98, 74.55), (-99.68, 74.53),
        (-99.48, 74.42), (-99.72, 74.31), (-100.12, 74.30),
    ],
    # Northern Baffin Island within the requested box.
    [
        (-90.55, 72.5), (-90.40, 73.02), (-90.18, 73.24),
        (-90.05, 73.47), (-89.78, 73.65), (-89.38, 73.79),
        (-88.92, 73.83), (-88.42, 73.80), (-87.98, 73.72),
        (-87.60, 73.61), (-87.23, 73.57), (-86.82, 73.49),
        (-86.45, 73.45), (-86.08, 73.36), (-85.65, 73.34),
        (-85.27, 73.27), (-84.88, 73.23), (-84.50, 73.16),
        (-84.12, 73.13), (-83.70, 73.12), (-83.32, 73.08),
        (-82.93, 73.12), (-82.54, 73.07), (-82.17, 73.03),
        (-81.76, 72.98), (-81.34, 72.96), (-80.60, 72.88),
        (-80.60, 72.5),
    ],
    # Small islands in Barrow Strait and nearby channels.
    [
        (-98.37, 74.93), (-98.05, 75.02), (-97.75, 74.97),
        (-97.68, 74.87), (-97.98, 74.81), (-98.27, 74.84),
    ],
    [
        (-93.97, 75.26), (-93.72, 75.34), (-93.48, 75.29),
        (-93.47, 75.19), (-93.74, 75.15), (-93.94, 75.18),
    ],
])


def on_segment(x, y, ax, ay, bx, by):
    cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
    if abs(cross) > 1e-10:
        return False
    return (min(ax, bx) - 1e-10 <= x <= max(ax, bx) + 1e-10 and
            min(ay, by) - 1e-10 <= y <= max(ay, by) + 1e-10)


def inside_polygon(x, y, polygon):
    inside = False
    previous = polygon[-1]
    for current in polygon:
        ax, ay = previous
        bx, by = current
        if on_segment(x, y, ax, ay, bx, by):
            return True
        if (ay > y) != (by > y):
            crossing_x = ax + (y - ay) * (bx - ax) / (by - ay)
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def build_matrix():
    matrix = []
    for row in range(61):
        latitude = (790 - row) / 10.0
        values = []
        for column in range(241):
            longitude = (-1050 + column) / 10.0
            land = any(
                inside_polygon(longitude, latitude, polygon)
                for polygon in LAND
            )
            values.append(int(land))
        matrix.append(values)
    return matrix


def validate_matrix(matrix):
    assert len(matrix) == 61
    assert all(len(row) == 241 for row in matrix)
    assert sum(len(row) for row in matrix) == 14701
    assert all(value in (0, 1) for row in matrix for value in row)


def write_csv(matrix, path):
    with path.open("w", newline="", encoding="ascii") as file:
        csv.writer(file, lineterminator="\n").writerows(matrix)


def validate_csv(matrix, path):
    with path.open("r", newline="", encoding="ascii") as file:
        rows = list(csv.reader(file))
    assert len(rows) == 61
    assert all(len(row) == 241 for row in rows)
    assert all(value in ("0", "1") for row in rows for value in row)
    assert [[int(value) for value in row] for row in rows] == matrix


def write_images(matrix, output):
    image = Image.new("L", (241, 61))
    image.putdata([255 * value for row in matrix for value in row])
    image.save(output / "candidate.png")
    try:
        nearest = Image.Resampling.NEAREST
    except AttributeError:
        nearest = Image.NEAREST
    image.resize((1928, 488), resample=nearest).save(
        output / "candidate_8x.png"
    )


def validate_images(matrix, output):
    with Image.open(output / "candidate.png") as source_file:
        source = source_file.convert("L")
    with Image.open(output / "candidate_8x.png") as scaled_file:
        scaled = scaled_file.convert("L")
    assert source.size == (241, 61)
    assert scaled.size == (1928, 488)
    assert set(source.getdata()) <= {0, 255}
    assert set(scaled.getdata()) <= {0, 255}
    expected = [255 * value for row in matrix for value in row]
    assert list(source.getdata()) == expected
    source_pixels = source.load()
    scaled_pixels = scaled.load()
    for y in range(488):
        for x in range(1928):
            assert scaled_pixels[x, y] == source_pixels[x // 8, y // 8]


def main():
    output = Path("outputs")
    output.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()
    validate_matrix(matrix)
    csv_path = output / "candidate.csv"
    write_csv(matrix, csv_path)
    validate_csv(matrix, csv_path)
    write_images(matrix, output)
    validate_images(matrix, output)
    print("Validated 61 x 241 binary samples.")
    print("Wrote outputs/candidate.csv")
    print("Wrote outputs/candidate.png")
    print("Wrote outputs/candidate_8x.png")


if __name__ == "__main__":
    main()
