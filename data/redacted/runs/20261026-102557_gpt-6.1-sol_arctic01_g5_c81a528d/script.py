from pathlib import Path
import csv

import numpy as np
from PIL import Image


ROWS = 61
COLS = 241
LAT0 = 79.0
LON0 = -105.0
STEP = 0.1


def poly(*points):
    return tuple((float(x), float(y)) for x, y in points)


LAND_POLYGONS = [
    poly(
        (-104.5, 79.0), (-101.8, 79.6), (-98.0, 79.45),
        (-95.6, 78.7), (-96.0, 77.8), (-99.0, 77.2),
        (-103.1, 77.1), (-105.0, 77.8)
    ),
    poly(
        (-101.5, 78.5), (-98.2, 79.0), (-95.0, 78.4),
        (-94.0, 77.3), (-96.1, 76.8), (-99.5, 77.1),
        (-101.8, 77.7)
    ),
    poly(
        (-96.8, 79.3), (-93.5, 79.9), (-89.5, 79.8),
        (-86.0, 79.1), (-86.2, 78.1), (-89.0, 77.5),
        (-93.2, 77.7), (-96.4, 78.2)
    ),
    poly(
        (-95.0, 78.0), (-91.5, 77.6), (-87.0, 78.5),
        (-82.0, 78.4), (-79.4, 77.1), (-80.0, 75.7),
        (-84.0, 74.2), (-89.0, 74.1), (-93.5, 74.8)
    ),
    poly(
        (-104.0, 74.8), (-100.5, 75.0), (-96.2, 74.6),
        (-94.5, 73.4), (-96.4, 72.7), (-101.5, 72.6),
        (-104.8, 73.4)
    ),
    poly(
        (-97.8, 74.2), (-94.7, 74.5), (-90.0, 73.9),
        (-87.2, 73.0), (-89.0, 72.3), (-93.0, 72.2),
        (-97.2, 72.7)
    ),
    poly(
        (-96.0, 73.2), (-93.0, 73.5), (-87.0, 73.1),
        (-87.2, 72.4), (-91.0, 72.1), (-95.0, 72.4)
    ),
]
WATER_CUTS = [
    poly(
        (-104.8, 74.7), (-101.0, 74.55), (-97.0, 74.35),
        (-92.0, 74.15), (-87.0, 73.75), (-88.0, 73.3),
        (-93.0, 73.55), (-98.0, 73.8), (-102.5, 74.0)
    ),
    poly(
        (-104.2, 78.2), (-101.0, 78.0), (-98.0, 77.4),
        (-96.0, 76.7), (-97.0, 75.8), (-99.5, 75.6),
        (-102.5, 76.2), (-104.7, 77.2)
    ),
    poly(
        (-96.0, 77.6), (-91.0, 77.2), (-86.0, 77.0),
        (-81.0, 77.2), (-80.0, 76.6), (-85.0, 76.4),
        (-90.5, 76.45), (-94.5, 76.8)
    ),
    poly(
        (-98.0, 74.1), (-96.0, 74.0), (-94.0, 73.5),
        (-95.0, 73.15), (-97.3, 73.35), (-99.0, 73.75)
    ),
    poly(
        (-96.0, 73.1), (-94.0, 73.2), (-92.0, 72.9),
        (-92.2, 72.45), (-94.5, 72.35), (-96.0, 72.65)
    ),
]


def boxed(polygons):
    result = []
    for points in polygons:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        result.append((min(xs), max(xs), min(ys), max(ys), points))
    return result


LAND = boxed(LAND_POLYGONS)
WATER = boxed(WATER_CUTS)


def contains(lon, lat, shape):
    min_x, max_x, min_y, max_y, points = shape
    if not (min_x <= lon <= max_x and min_y <= lat <= max_y):
        return False

    inside = False
    previous = len(points) - 1
    for current, (x, y) in enumerate(points):
        px, py = points[previous]
        if (py > lat) != (y > lat):
            crossing = (x - px) * (lat - py) / (y - py) + px
            if lon < crossing:
                inside = not inside
        previous = current
    return inside
def is_land(lon, lat):
    if not any(contains(lon, lat, shape) for shape in LAND):
        return False
    return not any(contains(lon, lat, shape) for shape in WATER)


def build_matrix():
    matrix = np.zeros((ROWS, COLS), dtype=np.uint8)
    for row in range(ROWS):
        lat = LAT0 - row * STEP
        for col in range(COLS):
            lon = LON0 + col * STEP
            matrix[row, col] = int(is_land(lon, lat))
    return matrix


def write_outputs(matrix):
    output = Path("outputs")
    output.mkdir(exist_ok=True)
    csv_path = output / "candidate.csv"
    png_path = output / "candidate.png"
    large_path = output / "candidate_8x.png"

    with csv_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(matrix.tolist())

    image = Image.fromarray(matrix * 255, mode="L")
    image.save(png_path)
    image.resize((COLS * 8, ROWS * 8), Image.Resampling.NEAREST).save(large_path)
    return csv_path, png_path, large_path


def validate(matrix, csv_path, png_path, large_path):
    assert matrix.shape == (61, 241)
    assert matrix.size == 14701
    assert set(np.unique(matrix)) <= {0, 1}

    with csv_path.open(encoding="ascii", newline="") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 61
    assert all(len(row) == 241 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    with Image.open(png_path) as image:
        assert image.size == (241, 61)
        assert set(np.unique(np.asarray(image))) <= {0, 255}

    with Image.open(large_path) as image:
        assert image.size == (1928, 488)
        enlarged = np.asarray(image)
        assert set(np.unique(enlarged)) <= {0, 255}
        expected = np.repeat(np.repeat(matrix * 255, 8, axis=0), 8, axis=1)
        assert np.array_equal(enlarged, expected)


def main():
    matrix = build_matrix()
    paths = write_outputs(matrix)
    validate(matrix, *paths)
    print("validated", matrix.shape, matrix.size)


if __name__ == "__main__":
    main()
