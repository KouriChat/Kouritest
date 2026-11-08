from pathlib import Path
import csv
import numpy as np
from PIL import Image

N_LAT = 111
N_LON = 141
LAT_NORTH = 23.0
LON_WEST = 99.0
STEP = 0.2

polygons = []
ellipses = []


def add_polygon(points):
    polygons.append([(float(x), float(y)) for x, y in points])


def add_ellipse(cx, cy, rx, ry=None):
    if ry is None:
        ry = rx
    ellipses.append((float(cx), float(cy), float(rx), float(ry)))


add_polygon([
    (99.0, 23.0), (115.0, 23.0), (115.6, 22.7), (116.6, 22.5),
    (117.3, 22.0), (117.5, 21.5), (117.0, 21.0), (116.0, 20.8),
    (114.8, 21.1), (113.8, 21.5), (112.5, 21.8), (111.3, 21.7),
    (110.2, 21.5), (109.3, 21.3), (108.4, 21.5), (107.2, 22.1),
    (105.5, 22.8), (104.5, 23.0), (99.0, 23.0)
])

add_polygon([
    (99.0, 23.0), (105.0, 23.0), (108.2, 21.5), (107.2, 20.8),
    (108.6, 19.5), (109.5, 17.0), (109.5, 15.0), (109.2, 13.0),
    (109.0, 11.0), (108.5, 9.5), (107.3, 8.4), (104.8, 8.5),
    (104.5, 10.2), (103.5, 10.7), (102.5, 11.8), (101.0, 12.5),
    (100.2, 13.7), (99.0, 15.0), (99.0, 23.0)
])

add_polygon([
    (109.4, 21.5), (110.8, 21.5), (111.3, 21.0), (111.0, 20.2),
    (110.0, 20.0), (109.3, 20.7)
])

add_polygon([
    (102.0, 13.0), (102.5, 11.0), (103.1, 9.0), (103.7, 7.0),
    (103.9, 5.0), (103.6, 3.0), (103.8, 1.0), (102.5, 1.0),
    (101.5, 2.0), (100.8, 4.0), (100.0, 6.0), (99.7, 8.0),
    (99.7, 10.0), (100.5, 12.0)
])

add_polygon([
    (108.6, 20.1), (109.3, 20.4), (110.4, 20.2), (111.2, 19.5),
    (111.0, 18.7), (110.3, 18.2), (109.2, 18.2), (108.6, 18.8),
    (108.4, 19.5)
])
add_polygon([
    (120.0, 25.0), (120.6, 25.2), (121.4, 24.7), (121.9, 23.7),
    (121.8, 22.7), (121.3, 22.0), (120.6, 21.9), (120.0, 22.5),
    (119.8, 23.5)
])

add_polygon([
    (99.0, 5.8), (100.0, 5.8), (101.5, 5.5), (103.3, 5.2),
    (104.5, 4.5), (105.3, 3.2), (105.5, 2.0), (105.0, 1.0),
    (101.0, 1.0), (100.5, 2.0), (99.5, 3.5)
])

add_polygon([
    (109.4, 7.2), (110.2, 6.5), (110.0, 5.5), (109.1, 4.0),
    (109.0, 2.0), (110.0, 1.0), (116.5, 1.0), (117.0, 2.0),
    (118.0, 3.2), (119.0, 4.5), (119.4, 6.0), (118.8, 7.0),
    (117.5, 7.4), (116.0, 7.2), (114.2, 7.5), (112.8, 7.2),
    (111.4, 7.0), (110.4, 7.5)
])

add_polygon([
    (119.3, 1.0), (120.0, 2.0), (119.6, 3.0), (120.5, 3.6),
    (121.5, 3.0), (122.0, 2.0), (123.0, 1.5), (124.5, 1.0),
    (124.8, 2.0), (124.0, 3.0), (123.0, 3.8), (122.0, 4.0),
    (121.4, 5.0), (120.5, 5.5), (120.0, 4.5), (120.4, 3.5),
    (119.5, 3.0)
])

add_polygon([
    (119.2, 18.5), (120.2, 18.8), (121.4, 18.5), (122.2, 17.5),
    (122.1, 16.2), (121.4, 15.2), (121.0, 14.2), (120.4, 13.4),
    (119.7, 13.7), (119.2, 14.7), (119.0, 16.2), (119.0, 17.5)
])

add_polygon([
    (120.0, 13.5), (120.7, 13.8), (121.4, 13.3), (121.5, 12.5),
    (121.0, 11.8), (120.3, 12.0), (119.9, 12.8)
])

add_polygon([
    (117.0, 12.0), (117.8, 12.7), (119.0, 12.3), (119.5, 11.5),
    (119.0, 10.7), (118.4, 10.0), (117.5, 10.2), (117.0, 11.0)
])
add_polygon([
    (117.0, 11.0), (117.5, 10.2), (118.0, 9.2), (118.6, 8.7),
    (119.0, 9.2), (119.2, 10.2), (119.0, 11.2), (118.5, 12.0),
    (117.8, 12.5), (117.2, 12.0)
])

add_polygon([
    (121.2, 11.8), (122.0, 12.0), (122.8, 11.5), (122.7, 10.7),
    (122.0, 10.2), (121.3, 10.7)
])

add_polygon([
    (122.1, 11.1), (122.7, 11.5), (123.0, 10.5), (122.8, 9.5),
    (122.2, 9.3), (121.9, 10.2)
])

add_polygon([
    (123.2, 11.6), (123.7, 11.2), (123.7, 10.0), (123.3, 9.5),
    (123.0, 10.3)
])

add_polygon([
    (124.0, 13.4), (124.8, 13.2), (125.5, 12.5), (125.4, 11.5),
    (124.8, 10.8), (124.2, 11.2), (123.9, 12.2)
])

add_polygon([
    (124.0, 11.5), (124.6, 11.4), (125.0, 10.8), (124.8, 10.2),
    (124.1, 10.0), (123.8, 10.6)
])

add_polygon([
    (122.2, 10.0), (123.4, 9.8), (124.2, 9.4), (125.3, 9.5),
    (126.2, 8.5), (126.5, 7.0), (126.0, 6.0), (125.0, 5.2),
    (124.0, 5.0), (123.0, 5.2), (122.3, 6.0), (122.0, 7.0),
    (121.7, 8.0), (122.0, 9.0)
])

add_polygon([
    (120.7, 6.5), (121.5, 6.2), (122.1, 5.7), (121.8, 5.0),
    (121.0, 4.8), (120.5, 5.5)
])
add_ellipse(120.9, 5.0, 0.18, 0.12)
add_ellipse(121.3, 5.8, 0.16, 0.12)
add_ellipse(103.85, 1.32, 0.16, 0.10)
add_ellipse(104.10, 1.15, 0.18, 0.12)
add_ellipse(104.65, 1.05, 0.28, 0.18)
add_ellipse(105.55, 1.10, 0.15, 0.10)
add_ellipse(106.00, 2.70, 0.30, 0.20)
add_ellipse(108.00, 3.95, 0.35, 0.25)
add_ellipse(108.30, 4.00, 0.18, 0.12)

add_ellipse(103.95, 10.25, 0.18, 0.14)
add_ellipse(106.60, 8.70, 0.12, 0.10)
add_ellipse(116.75, 20.70, 0.18, 0.12)
add_ellipse(112.33, 16.83, 0.14, 0.10)
add_ellipse(112.00, 16.45, 0.10, 0.08)
add_ellipse(112.65, 16.05, 0.10, 0.08)
add_ellipse(111.45, 16.40, 0.08, 0.07)
add_ellipse(111.70, 15.85, 0.08, 0.06)

spratly = [
    (114.30, 11.05, 0.11, 0.08),
    (113.95, 10.90, 0.09, 0.07),
    (114.35, 10.38, 0.13, 0.09),
    (114.50, 10.38, 0.08, 0.06),
    (114.15, 10.18, 0.08, 0.06),
    (113.10, 10.20, 0.10, 0.07),
    (112.90, 9.55, 0.12, 0.08),
    (114.20, 9.70, 0.08, 0.06),
    (115.80, 9.90, 0.10, 0.07),
    (115.55, 10.00, 0.08, 0.06),
    (117.10, 10.00, 0.08, 0.06),
    (117.50, 9.80, 0.08, 0.06),
    (111.90, 9.55, 0.08, 0.06),
    (112.25, 8.65, 0.08, 0.06),
    (114.05, 8.65, 0.08, 0.06),
    (114.65, 8.20, 0.08, 0.06),
    (116.00, 8.00, 0.08, 0.06),
]
for item in spratly:
    add_ellipse(*item)


def point_on_segment(x, y, x1, y1, x2, y2):
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > 1e-10:
        return False
    return min(x1, x2) - 1e-10 <= x <= max(x1, x2) + 1e-10 and min(y1, y2) - 1e-10 <= y <= max(y1, y2) + 1e-10


def inside_polygon(x, y, polygon):
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[j]
        if point_on_segment(x, y, x1, y1, x2, y2):
            return True
        if (y1 > y) != (y2 > y):
            if x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside
        j = i
    return inside
def is_land(lon, lat):
    for polygon in polygons:
        if inside_polygon(lon, lat, polygon):
            return True
    for cx, cy, rx, ry in ellipses:
        dx = (lon - cx) / rx
        dy = (lat - cy) / ry
        if dx * dx + dy * dy <= 1.0:
            return True
    return False


def build_matrix():
    matrix = np.zeros((N_LAT, N_LON), dtype=np.uint8)
    for row in range(N_LAT):
        lat = LAT_NORTH - row * STEP
        for col in range(N_LON):
            lon = LON_WEST + col * STEP
            matrix[row, col] = int(is_land(lon, lat))
    return matrix


def validate_matrix(matrix):
    assert matrix.shape == (111, 141)
    assert matrix.size == 15651
    assert np.all((matrix == 0) | (matrix == 1))


def write_csv(matrix, path):
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for row in matrix:
            writer.writerow([int(value) for value in row])


def write_images(matrix, output_dir):
    raw = (matrix * 255).astype(np.uint8)
    image = Image.fromarray(raw, mode="L")
    image.save(output_dir / "candidate.png")
    enlarged = image.resize((1128, 888), Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")


def validate_outputs(output_dir):
    with (output_dir / "candidate.csv").open(
        "r", newline="", encoding="ascii"
    ) as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 111
    assert all(len(row) == 141 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    image = Image.open(output_dir / "candidate.png")
    assert image.size == (141, 111)
    assert set(image.getdata()).issubset({0, 255})

    enlarged = Image.open(output_dir / "candidate_8x.png")
    assert enlarged.size == (1128, 888)
    assert set(enlarged.getdata()).issubset({0, 255})
def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()
    validate_matrix(matrix)
    write_csv(matrix, output_dir / "candidate.csv")
    write_images(matrix, output_dir)
    validate_outputs(output_dir)
    print("Validated 111 x 141 matrix with 15651 binary values.")


if __name__ == "__main__":
    main()
