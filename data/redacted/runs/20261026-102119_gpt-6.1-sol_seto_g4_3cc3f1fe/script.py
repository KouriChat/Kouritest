from pathlib import Path
import csv
import numpy as np
from PIL import Image

ROWS = 64
COLS = 253
LAT_NORTH = 35.0
LON_WEST = 129.0
LAT_STEP = 2.0 / 63.0
LON_STEP = 8.0 / 252.0


def polygon(points):
    return tuple((float(x), float(y)) for x, y in points)


def ellipse(cx, cy, rx, ry, n=24, phase=0.0):
    points = []
    for i in range(n):
        angle = phase + 2.0 * np.pi * i / n
        points.append((cx + rx * np.cos(angle),
                       cy + ry * np.sin(angle)))
    return polygon(points)


def point_in_polygon(x, y, poly):
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)):
            crossing = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < crossing:
                inside = not inside
        j = i
    return inside


LAND_POLYGONS = []

LAND_POLYGONS.append(polygon([
    (129.25, 33.00), (129.32, 33.28), (129.52, 33.58),
    (129.82, 33.83), (130.18, 34.00), (130.50, 33.96),
    (130.78, 33.80), (131.08, 33.80), (131.35, 33.70),
    (131.60, 33.72), (131.78, 33.54), (131.86, 33.25),
    (131.82, 33.00),
]))

LAND_POLYGONS.append(polygon([
    (130.62, 34.00), (130.66, 34.28), (130.76, 34.57),
    (130.88, 34.84), (131.05, 35.00), (136.56, 35.00),
    (136.56, 34.86), (136.47, 34.73), (136.27, 34.71),
    (136.02, 34.65), (135.77, 34.62), (135.53, 34.62),
    (135.32, 34.64), (135.10, 34.68), (134.88, 34.67),
    (134.67, 34.59), (134.44, 34.55), (134.21, 34.51),
    (133.98, 34.47), (133.76, 34.42), (133.54, 34.35),
    (133.30, 34.30), (133.05, 34.25), (132.80, 34.20),
    (132.58, 34.15), (132.36, 34.10), (132.12, 34.06),
    (131.88, 34.02), (131.62, 33.98), (131.35, 33.94),
    (131.06, 33.90), (130.78, 33.88), (130.62, 34.00),
]))
LAND_POLYGONS.append(polygon([
    (134.78, 34.55), (135.04, 34.67), (135.34, 34.62),
    (135.64, 34.49), (135.94, 34.36), (136.23, 34.21),
    (136.44, 34.02), (136.46, 33.72), (136.23, 33.42),
    (135.92, 33.16), (135.60, 33.02), (135.12, 33.00),
    (134.96, 33.20), (134.82, 33.46), (134.72, 33.73),
    (134.72, 34.00), (134.78, 34.30),
]))

LAND_POLYGONS.append(polygon([
    (136.72, 35.00), (137.00, 35.00), (137.00, 34.71),
    (136.89, 34.60), (136.72, 34.70), (136.67, 34.91),
]))

LAND_POLYGONS.append(polygon([
    (137.00, 34.72), (137.00, 34.25), (136.82, 34.30),
    (136.66, 34.50), (136.70, 34.70),
]))

LAND_POLYGONS.append(polygon([
    (131.76, 33.55), (131.95, 33.80), (132.11, 34.00),
    (132.35, 34.15), (132.70, 34.25), (133.15, 34.30),
    (133.55, 34.35), (133.95, 34.32), (134.35, 34.22),
    (134.64, 34.10), (134.72, 33.85), (134.55, 33.60),
    (134.30, 33.40), (134.00, 33.25), (133.65, 33.10),
    (133.25, 33.00), (132.80, 33.00), (132.40, 33.05),
    (132.10, 33.20), (131.90, 33.40),
]))

LAND_POLYGONS.append(polygon([
    (129.18, 34.10), (129.22, 34.38), (129.30, 34.68),
    (129.43, 34.70), (129.51, 34.47), (129.46, 34.20),
    (129.34, 34.02),
]))

LAND_POLYGONS.append(polygon([
    (129.63, 33.70), (129.72, 33.86), (129.88, 33.88),
    (129.96, 33.73), (129.88, 33.59), (129.70, 33.58),
]))

LAND_POLYGONS.append(ellipse(129.63, 33.03, 0.10, 0.08, 18))
LAND_POLYGONS.append(ellipse(129.82, 33.02, 0.07, 0.055, 16))
LAND_POLYGONS.append(polygon([
    (134.66, 34.18), (134.76, 34.42), (134.92, 34.67),
    (135.08, 34.58), (135.12, 34.32), (135.02, 34.10),
    (134.84, 34.02), (134.70, 34.08),
]))

LAND_POLYGONS.append(polygon([
    (133.99, 34.43), (134.10, 34.57), (134.32, 34.60),
    (134.48, 34.49), (134.45, 34.31), (134.27, 34.22),
    (134.07, 34.26),
]))

LAND_POLYGONS.append(ellipse(133.98, 34.47, 0.08, 0.06, 18))
LAND_POLYGONS.append(ellipse(134.07, 34.42, 0.06, 0.05, 18))
LAND_POLYGONS.append(ellipse(134.24, 34.45, 0.045, 0.035, 16))
LAND_POLYGONS.append(ellipse(134.39, 34.34, 0.055, 0.04, 16))

LAND_POLYGONS.append(ellipse(133.72, 34.34, 0.10, 0.07, 20))
LAND_POLYGONS.append(ellipse(133.59, 34.28, 0.075, 0.055, 18))
LAND_POLYGONS.append(ellipse(133.40, 34.27, 0.075, 0.06, 18))
LAND_POLYGONS.append(ellipse(133.22, 34.30, 0.15, 0.12, 24))
LAND_POLYGONS.append(ellipse(133.08, 34.27, 0.12, 0.10, 22))
LAND_POLYGONS.append(ellipse(132.94, 34.25, 0.11, 0.09, 20))
LAND_POLYGONS.append(ellipse(132.82, 34.18, 0.13, 0.08, 20))
LAND_POLYGONS.append(ellipse(132.67, 34.15, 0.08, 0.06, 18))
LAND_POLYGONS.append(ellipse(132.54, 34.18, 0.12, 0.10, 22))
LAND_POLYGONS.append(ellipse(132.43, 34.27, 0.08, 0.07, 18))
LAND_POLYGONS.append(ellipse(132.31, 34.29, 0.09, 0.075, 20))
LAND_POLYGONS.append(ellipse(132.53, 34.04, 0.14, 0.10, 22))
LAND_POLYGONS.append(ellipse(132.35, 33.96, 0.19, 0.10, 24))
LAND_POLYGONS.append(ellipse(132.22, 33.89, 0.07, 0.05, 16))
LAND_POLYGONS.append(ellipse(131.91, 33.94, 0.06, 0.045, 16))
LAND_POLYGONS.append(ellipse(132.02, 33.86, 0.055, 0.04, 16))
LAND_POLYGONS.append(ellipse(132.16, 33.78, 0.07, 0.045, 16))
LAND_POLYGONS.append(ellipse(132.48, 33.79, 0.055, 0.04, 16))
LAND_POLYGONS.append(ellipse(132.74, 33.86, 0.075, 0.05, 18))
LAND_POLYGONS.append(ellipse(132.98, 33.91, 0.065, 0.05, 18))
LAND_POLYGONS.append(ellipse(133.18, 33.96, 0.06, 0.045, 16))
LAND_POLYGONS.append(ellipse(133.40, 34.00, 0.06, 0.045, 16))
LAND_POLYGONS.append(ellipse(133.61, 34.03, 0.05, 0.04, 16))
LAND_POLYGONS.append(ellipse(133.83, 34.08, 0.06, 0.04, 16))
LAND_POLYGONS.append(ellipse(134.08, 34.08, 0.06, 0.045, 16))

LAND_POLYGONS.append(ellipse(132.98, 34.43, 0.055, 0.04, 16))
LAND_POLYGONS.append(ellipse(133.26, 34.45, 0.065, 0.045, 16))
LAND_POLYGONS.append(ellipse(133.52, 34.47, 0.055, 0.04, 16))
LAND_POLYGONS.append(ellipse(133.73, 34.48, 0.05, 0.038, 16))
LAND_POLYGONS.append(ellipse(133.91, 34.52, 0.055, 0.04, 16))
LAND_POLYGONS.append(ellipse(134.19, 34.57, 0.06, 0.045, 16))
LAND_POLYGONS.append(ellipse(134.56, 34.47, 0.055, 0.04, 16))

LAND_POLYGONS.append(ellipse(136.58, 34.64, 0.06, 0.045, 16))
LAND_POLYGONS.append(ellipse(136.70, 34.58, 0.05, 0.04, 16))
LAND_POLYGONS.append(ellipse(136.82, 34.82, 0.05, 0.04, 16))
LAND_POLYGONS.append(ellipse(136.92, 34.72, 0.04, 0.03, 14))


def classify(lon, lat):
    for poly in LAND_POLYGONS:
        if point_in_polygon(lon, lat, poly):
            return 1
    return 0


def build_matrix():
    matrix = np.zeros((ROWS, COLS), dtype=np.uint8)
    for row in range(ROWS):
        lat = LAT_NORTH - row * LAT_STEP
        for col in range(COLS):
            lon = LON_WEST + col * LON_STEP
            matrix[row, col] = classify(lon, lat)
    return matrix
def write_csv(matrix, path):
    with path.open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        for row in matrix:
            writer.writerow(int(value) for value in row)


def write_images(matrix, output_dir):
    image = Image.fromarray((matrix * 255).astype(np.uint8), mode="L")
    if image.size != (253, 64):
        raise AssertionError(image.size)
    image.save(output_dir / "candidate.png")

    resampling = getattr(Image, "Resampling", Image)
    enlarged = image.resize((2024, 512), resample=resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")


def validate(matrix, output_dir):
    assert matrix.shape == (64, 253)
    assert matrix.size == 16192
    assert np.all((matrix == 0) | (matrix == 1))

    with (output_dir / "candidate.csv").open(
        "r", encoding="ascii", newline=""
    ) as f:
        rows = list(csv.reader(f))
    assert len(rows) == 64
    assert all(len(row) == 253 for row in rows)
    assert all(value in ("0", "1") for row in rows for value in row)

    raw = Image.open(output_dir / "candidate.png").convert("L")
    enlarged = Image.open(output_dir / "candidate_8x.png").convert("L")
    assert raw.size == (253, 64)
    assert enlarged.size == (2024, 512)
    assert set(raw.getdata()).issubset({0, 255})
    assert set(enlarged.getdata()).issubset({0, 255})

    raw_array = np.asarray(raw, dtype=np.uint8)
    expected = np.repeat(np.repeat(raw_array, 8, axis=0), 8, axis=1)
    assert np.array_equal(np.asarray(enlarged), expected)
    assert np.array_equal(raw_array, matrix * 255)


def main():
    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()
    write_csv(matrix, output_dir / "candidate.csv")
    write_images(matrix, output_dir)
    validate(matrix, output_dir)
    print("validation: passed")


if __name__ == "__main__":
    main()
