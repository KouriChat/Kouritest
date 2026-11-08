import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image


SPAIN = [
    (-9.0, 43.8), (-8.5, 43.6), (-7.5, 43.5), (-6.5, 43.6),
    (-5.5, 43.4), (-4.5, 43.4), (-3.5, 43.3), (-2.5, 43.3),
    (-1.5, 43.4), (-0.8, 43.1), (-0.3, 42.8), (-0.2, 42.1),
    (0.3, 41.5), (0.7, 41.0), (0.7, 40.5), (0.3, 40.0),
    (0.4, 39.5), (0.1, 39.0), (-0.2, 38.6), (-0.3, 38.1),
    (-0.8, 37.5), (-1.4, 37.2), (-2.2, 36.9), (-3.0, 36.75),
    (-4.0, 36.7), (-5.0, 36.7), (-5.5, 36.2), (-6.2, 36.0),
    (-6.8, 36.1), (-7.4, 36.3), (-8.0, 36.5), (-8.7, 36.7),
    (-9.0, 37.0),
]

FRANCE = [
    (-1.5, 43.4), (0.0, 42.8), (1.0, 42.7), (2.0, 42.7),
    (3.0, 42.8), (4.0, 43.0), (5.0, 43.0), (6.0, 43.0),
    (6.5, 43.1), (7.0, 43.4), (7.5, 43.7), (8.0, 44.1),
    (9.0, 45.0), (14.0, 45.0), (14.0, 45.1), (-1.5, 45.1),
]

ITALY = [
    (7.5, 45.0), (13.5, 45.0), (14.5, 44.0), (15.0, 43.0),
    (15.5, 42.5), (16.0, 41.8), (16.5, 41.2), (17.0, 41.0),
    (18.0, 40.8), (18.5, 40.5), (18.4, 40.0), (17.8, 40.0),
    (17.2, 40.4), (16.8, 39.8), (16.5, 39.2), (16.0, 38.8),
    (15.5, 38.8), (15.0, 39.2), (14.7, 40.0), (14.0, 40.4),
    (13.4, 40.5), (13.0, 40.0), (12.5, 40.0), (12.0, 40.5),
    (11.5, 41.0), (11.2, 41.6), (10.5, 42.0), (10.0, 42.5),
    (9.5, 43.0), (8.8, 43.5), (8.0, 44.0),
]

BALKANS = [
    (13.0, 45.0), (29.5, 45.0), (29.5, 42.0), (29.0, 41.5),
    (28.5, 41.0), (28.0, 40.7), (27.5, 40.4), (27.0, 40.3),
    (26.5, 40.0), (26.0, 39.8), (25.5, 39.5), (25.0, 39.0),
    (24.5, 38.8), (24.0, 38.7), (23.5, 39.0), (23.0, 39.0),
    (22.5, 38.5), (22.0, 38.3), (21.5, 38.3), (21.0, 38.5),
    (20.5, 39.0), (20.0, 39.5), (19.5, 40.0), (19.0, 40.5),
    (18.5, 40.5), (18.0, 40.8), (17.5, 41.0), (17.0, 41.5),
    (16.5, 42.0), (16.0, 42.5), (15.5, 43.0), (15.0, 44.0),
    (14.0, 45.0),
]
TURKEY = [
    (25.3, 41.1), (26.0, 41.4), (27.0, 41.4), (28.0, 41.3),
    (29.0, 41.2), (30.0, 41.2), (31.0, 41.3), (32.0, 41.4),
    (33.0, 41.6), (34.0, 41.7), (35.0, 41.5), (36.0, 41.2),
    (37.0, 41.1), (37.0, 36.5), (36.5, 36.4), (36.0, 36.2),
    (35.5, 36.1), (35.0, 36.1), (34.5, 36.2), (34.0, 36.2),
    (33.5, 36.0), (33.0, 36.0), (32.5, 36.2), (32.0, 36.3),
    (31.5, 36.2), (31.0, 36.3), (30.5, 36.4), (30.0, 36.5),
    (29.5, 36.6), (29.0, 36.7), (28.5, 36.8), (28.0, 37.0),
    (27.5, 37.3), (27.0, 37.7), (26.5, 38.0), (26.0, 38.5),
    (25.5, 39.0), (25.0, 39.5), (24.8, 40.0), (25.0, 40.5),
]

NORTH_AFRICA = [
    (-9.0, 31.0), (-9.0, 34.5), (-8.8, 35.0), (-8.0, 35.1),
    (-7.0, 35.3), (-6.0, 35.6), (-5.5, 35.9), (-5.0, 35.8),
    (-4.5, 35.7), (-4.0, 35.5), (-3.5, 35.3), (-3.0, 35.2),
    (-2.5, 35.1), (-2.0, 35.0), (-1.5, 35.0), (-1.0, 35.1),
    (-0.5, 35.2), (0.0, 35.4), (0.5, 35.7), (1.0, 36.0),
    (1.5, 36.2), (2.0, 36.5), (2.5, 36.6), (3.0, 36.7),
    (3.5, 36.8), (4.0, 36.8), (4.5, 36.8), (5.0, 36.8),
    (5.5, 36.9), (6.0, 37.0), (6.5, 37.1), (7.0, 37.1),
    (7.5, 37.1), (8.0, 37.0), (8.5, 36.9), (9.0, 37.0),
    (9.5, 37.1), (10.0, 37.2), (10.5, 37.2), (11.0, 37.1),
    (11.5, 37.0), (12.0, 36.8), (12.5, 36.7), (13.0, 36.7),
    (13.5, 36.5), (14.0, 36.3), (14.5, 36.0), (15.0, 35.8),
    (15.5, 35.5), (16.0, 35.2), (16.5, 34.9), (17.0, 34.5),
    (17.5, 34.0), (18.0, 33.5), (18.5, 33.0), (19.0, 32.7),
    (20.0, 32.5), (21.0, 32.4), (22.0, 32.2), (23.0, 32.0),
    (24.0, 31.8), (25.0, 31.5), (26.0, 31.3), (27.0, 31.1),
    (37.0, 31.0),
]

LEVANT = [
    (35.8, 37.0), (37.0, 37.0), (37.0, 31.0), (34.0, 31.0),
    (34.2, 31.3), (34.4, 31.7), (34.5, 32.1), (34.6, 32.5),
    (34.8, 33.0), (35.0, 33.3), (35.2, 33.6), (35.4, 34.0),
    (35.7, 34.5), (36.0, 35.0), (36.3, 35.5), (36.5, 36.0),
    (36.0, 36.5),
]

CAUCASUS_EDGE = [
    (35.0, 45.0), (37.0, 45.0), (37.0, 42.0), (36.5, 42.0),
    (36.0, 42.2), (35.5, 42.5), (35.0, 43.0), (35.0, 44.0),
]

LAND_POLYGONS = [
    SPAIN, FRANCE, ITALY, BALKANS, TURKEY,
    NORTH_AFRICA, LEVANT, CAUCASUS_EDGE,
]
def ellipse_polygon(cx, cy, rx, ry, count=32, rotation=0.0):
    points = []
    cr = math.cos(rotation)
    sr = math.sin(rotation)
    for i in range(count):
        angle = 2.0 * math.pi * i / count
        ex = rx * math.cos(angle)
        ey = ry * math.sin(angle)
        points.append((
            cx + ex * cr - ey * sr,
            cy + ex * sr + ey * cr,
        ))
    return points


ISLANDS = [
    ellipse_polygon(9.0, 40.1, 0.80, 1.35, rotation=0.10),
    ellipse_polygon(9.0, 42.1, 0.50, 0.90, rotation=-0.12),
    ellipse_polygon(14.1, 37.6, 1.35, 0.55, rotation=-0.10),
    ellipse_polygon(24.9, 35.2, 1.30, 0.30, rotation=-0.08),
    ellipse_polygon(33.0, 35.1, 1.05, 0.55, rotation=-0.12),
    ellipse_polygon(3.0, 39.6, 0.65, 0.55, rotation=0.10),
    ellipse_polygon(4.1, 39.95, 0.35, 0.25),
    ellipse_polygon(1.4, 38.98, 0.35, 0.22, rotation=-0.15),
    ellipse_polygon(28.1, 36.15, 0.30, 0.60, rotation=-0.20),
    ellipse_polygon(23.7, 38.5, 0.25, 0.90, rotation=0.08),
    ellipse_polygon(26.0, 38.4, 0.25, 0.45, rotation=-0.10),
    ellipse_polygon(26.3, 39.2, 0.50, 0.35, rotation=0.15),
    ellipse_polygon(27.0, 37.75, 0.25, 0.35),
    ellipse_polygon(27.3, 36.85, 0.25, 0.20),
    ellipse_polygon(19.9, 39.6, 0.18, 0.65),
    ellipse_polygon(20.55, 38.25, 0.25, 0.50, rotation=-0.25),
    ellipse_polygon(20.7, 37.8, 0.22, 0.30),
    ellipse_polygon(20.7, 38.75, 0.16, 0.35),
    ellipse_polygon(23.0, 36.2, 0.30, 0.25),
    ellipse_polygon(10.9, 33.8, 0.25, 0.20),
]

LAND_POLYGONS.extend(ISLANDS)

LAKES = [
    ellipse_polygon(20.70, 41.05, 0.18, 0.35),
    ellipse_polygon(21.00, 40.90, 0.18, 0.25),
    ellipse_polygon(19.30, 42.20, 0.30, 0.20),
    ellipse_polygon(11.95, 42.60, 0.20, 0.18),
    ellipse_polygon(12.10, 43.10, 0.25, 0.18),
]
def point_in_polygon(x, y, polygon):
    inside = False
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        crosses = (y1 > y) != (y2 > y)
        if crosses:
            boundary_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < boundary_x:
                inside = not inside
    return inside


def is_land(lon, lat):
    in_land = any(
        point_in_polygon(lon, lat, polygon)
        for polygon in LAND_POLYGONS
    )
    if not in_land:
        return False
    return not any(
        point_in_polygon(lon, lat, lake)
        for lake in LAKES
    )


def build_matrix():
    matrix = np.zeros((71, 231), dtype=np.uint8)
    for row in range(71):
        lat = 45.0 - 0.2 * row
        for col in range(231):
            lon = -9.0 + 0.2 * col
            matrix[row, col] = int(is_land(lon, lat))
    return matrix


def write_outputs(matrix):
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        for row in matrix:
            writer.writerow([int(value) for value in row])

    raw_pixels = (matrix * 255).astype(np.uint8)
    raw_path = output_dir / "candidate.png"
    Image.fromarray(raw_pixels, mode="L").save(raw_path)

    enlarged_pixels = np.repeat(np.repeat(raw_pixels, 8, axis=0), 8, axis=1)
    enlarged_path = output_dir / "candidate_8x.png"
    Image.fromarray(enlarged_pixels, mode="L").save(enlarged_path)
    return csv_path, raw_path, enlarged_path
def validate(matrix, paths):
    assert matrix.shape == (71, 231)
    assert matrix.size == 16401
    assert set(np.unique(matrix).tolist()) <= {0, 1}

    csv_path, raw_path, enlarged_path = paths
    with csv_path.open("r", newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 71
    assert all(len(row) == 231 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    with Image.open(raw_path) as image:
        assert image.size == (231, 71)
        assert set(image.convert("L").getdata()) <= {0, 255}

    expected = np.repeat(
        np.repeat((matrix * 255).astype(np.uint8), 8, axis=0),
        8,
        axis=1,
    )
    with Image.open(enlarged_path) as image:
        assert image.size == (1848, 568)
        actual = np.asarray(image.convert("L"))
        assert set(actual.reshape(-1).tolist()) <= {0, 255}
        assert np.array_equal(actual, expected)


def main():
    matrix = build_matrix()
    paths = write_outputs(matrix)
    validate(matrix, paths)
    print("matrix:", matrix.shape, "values:", matrix.size)
    print("csv:", paths[0])
    print("raw png:", paths[1], "size=231x71")
    print("nearest png:", paths[2], "size=1848x568")
    print("validation: OK")


if __name__ == "__main__":
    main()
