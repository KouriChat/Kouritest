from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image


# Coordinates are (degrees west, degrees north). All outlines below are
# hand-authored geographic approximations; no external map data is read.
LAND_POLYGONS = [
    # Ellef Ringnes Island
    [(105.0, 78.35), (104.4, 78.78), (103.3, 78.98),
     (101.8, 79.08), (100.2, 79.04), (98.9, 78.88),
     (98.0, 78.59), (97.7, 78.25), (98.4, 77.98),
     (99.8, 77.79), (101.5, 77.76), (102.9, 77.92),
     (104.1, 78.04)],

    # Amund Ringnes Island
    [(98.0, 77.62), (97.6, 78.05), (97.0, 78.40),
     (96.0, 78.66), (94.8, 78.72), (93.5, 78.58),
     (92.4, 78.31), (91.6, 77.99), (91.9, 77.66),
     (92.8, 77.39), (94.0, 77.22), (95.4, 77.25),
     (96.7, 77.37)],

    # Axel Heiberg Island
    [(96.4, 79.10), (94.9, 78.98), (94.1, 78.69),
     (93.6, 78.31), (92.8, 77.95), (91.8, 77.62),
     (91.0, 77.25), (90.3, 77.08), (89.3, 77.26),
     (88.6, 77.62), (87.9, 77.98), (87.2, 78.35),
     (86.3, 78.68), (85.1, 79.10)],

    # Ellesmere Island, southwest edge
    [(84.3, 79.10), (81.0, 79.10), (81.0, 76.35),
     (81.8, 76.40), (82.6, 76.68), (83.2, 77.13),
     (83.8, 77.67), (84.0, 78.20)],

    # Devon Island
    [(94.8, 74.34), (94.6, 74.79), (94.2, 75.25),
     (93.8, 75.75), (93.0, 76.19), (91.8, 76.48),
     (90.2, 76.65), (88.2, 76.70), (86.2, 76.65),
     (84.1, 76.55), (82.4, 76.38), (81.0, 76.16),
     (81.0, 73.94), (82.4, 73.98), (84.0, 74.05),
     (85.7, 74.10), (87.4, 74.15), (89.0, 74.24),
     (90.7, 74.31), (92.2, 74.38), (93.5, 74.31)],

    # Bathurst Island, main body
    [(102.5, 76.72), (101.5, 76.92), (100.3, 76.89),
     (99.2, 76.65), (98.3, 76.43), (97.5, 76.16),
     (97.2, 75.82), (97.5, 75.47), (98.2, 75.16),
     (99.2, 75.05), (100.2, 75.18), (101.0, 75.38),
     (101.7, 75.67), (102.4, 75.96)],

    # Bathurst western body
    [(104.1, 76.14), (103.2, 76.42), (102.3, 76.46),
     (101.8, 76.20), (101.8, 75.90), (102.3, 75.68),
     (103.2, 75.60), (104.0, 75.73), (104.5, 75.94)],

    # Alexander, Cameron, Vanier, and Massey islands
    [(105.0, 76.12), (104.3, 76.14), (103.6, 76.32),
     (103.0, 76.63), (103.1, 76.86), (104.0, 76.96),
     (105.0, 76.91)],
    [(105.0, 76.34), (104.3, 76.39), (103.8, 76.57),
     (103.9, 76.80), (104.5, 76.91), (105.0, 76.89)],
    [(105.0, 75.55), (104.1, 75.62), (103.4, 75.82),
     (103.2, 76.05), (103.8, 76.24), (104.6, 76.20),
     (105.0, 76.10)],
    [(105.0, 76.78), (104.3, 76.79), (103.7, 76.94),
     (103.8, 77.08), (105.0, 77.10)],

    # Eastern Melville Island
    [(105.0, 74.66), (104.5, 74.77), (104.0, 75.01),
     (104.2, 75.27), (104.7, 75.45), (105.0, 75.46)],

    # Cornwallis Island
    [(97.1, 74.75), (96.8, 75.08), (96.5, 75.43),
     (95.9, 75.70), (95.0, 75.73), (94.1, 75.57),
     (93.6, 75.28), (93.7, 74.91), (94.1, 74.62),
     (94.9, 74.45), (95.8, 74.48), (96.6, 74.58)],

    # Little Cornwallis Island
    [(97.2, 75.34), (96.8, 75.51), (96.5, 75.75),
     (96.8, 75.93), (97.4, 75.90), (97.8, 75.67),
     (97.7, 75.44)],

    # Baillie-Hamilton Island
    [(95.6, 75.72), (95.1, 75.90), (94.7, 76.14),
     (94.9, 76.34), (95.5, 76.31), (95.9, 76.10),
     (96.0, 75.87)],

    # Lowther and Griffith islands
    [(98.2, 74.37), (97.7, 74.44), (97.2, 74.64),
     (97.3, 74.84), (97.9, 74.93), (98.5, 74.79),
     (98.7, 74.58)],
    [(96.4, 74.34), (95.9, 74.42), (95.6, 74.59),
     (95.8, 74.75), (96.3, 74.77), (96.7, 74.59)],

    # Prince of Wales Island, northern portion
    [(105.0, 73.00), (105.0, 73.87), (104.2, 74.08),
     (103.2, 74.24), (102.1, 74.32), (101.0, 74.22),
     (100.1, 74.02), (99.2, 73.78), (98.5, 73.47),
     (97.9, 73.16), (97.7, 73.00)],

    # Somerset Island
    [(96.2, 73.00), (96.0, 73.36), (95.7, 73.68),
     (95.2, 73.91), (94.5, 74.04), (93.7, 74.02),
     (92.9, 73.88), (92.1, 73.67), (91.3, 73.39),
     (90.7, 73.10), (90.6, 73.00)],

    # Brodeur Peninsula, northwestern Baffin Island
    [(90.0, 73.00), (89.6, 73.28), (89.0, 73.52),
     (88.1, 73.70), (87.0, 73.77), (86.0, 73.71),
     (85.1, 73.54), (84.7, 73.30), (84.6, 73.00)],

    # Borden Peninsula, east of Admiralty Inlet
    [(83.5, 73.00), (83.4, 73.27), (83.0, 73.48),
     (82.3, 73.61), (81.5, 73.60), (81.0, 73.52),
     (81.0, 73.00)],
]


# (center west longitude, center latitude, longitude radius,
# latitude radius, rotation in degrees)
ELLIPTICAL_ISLANDS = [
    (102.0, 77.17, 0.42, 0.16, -10),
    (99.1, 77.14, 0.28, 0.12, 5),
    (92.1, 77.00, 0.34, 0.12, -15),
    (96.7, 74.25, 0.27, 0.10, 10),
    (95.1, 74.24, 0.20, 0.09, -5),
    (92.0, 74.72, 0.34, 0.10, 8),
    (90.0, 74.02, 0.23, 0.09, 0),
    (99.0, 74.15, 0.29, 0.10, -8),
    (100.1, 74.50, 0.21, 0.09, 12),
]


WATER_CUTOUTS = [
    # Bathurst western inlet
    [(104.3, 75.96), (103.6, 76.00), (102.9, 76.10),
     (102.1, 76.18), (101.4, 76.23), (101.3, 76.11),
     (102.0, 75.99), (102.9, 75.89), (103.7, 75.84),
     (104.3, 75.84)],

    # Bathurst northern inlet
    [(101.4, 76.96), (101.1, 76.65), (100.9, 76.37),
     (100.6, 76.17), (100.3, 76.22), (100.4, 76.52),
     (100.6, 76.77), (100.8, 76.96)],

    # Devon western inlet
    [(94.5, 75.02), (93.9, 75.11), (93.2, 75.15),
     (92.7, 75.09), (93.2, 74.99), (93.9, 74.93),
     (94.5, 74.91)],

    # Devon north-coast fjords
    [(89.8, 76.75), (89.5, 76.51), (89.1, 76.29),
     (88.7, 76.14), (88.5, 76.20), (88.7, 76.43),
     (89.0, 76.66), (89.1, 76.75)],
    [(85.8, 76.67), (85.5, 76.45), (85.1, 76.25),
     (84.7, 76.15), (84.6, 76.25), (84.9, 76.48),
     (85.1, 76.64)],

    # Cornwallis northern bay
    [(95.7, 75.76), (95.5, 75.53), (95.2, 75.37),
     (94.9, 75.42), (95.0, 75.61), (95.1, 75.76)],

    # Prince of Wales northern indentation
    [(102.7, 74.34), (102.4, 74.08), (102.0, 73.91),
     (101.6, 73.84), (101.5, 73.94), (101.8, 74.12),
     (102.0, 74.30)],

    # Somerset western indentation
    [(95.9, 73.56), (95.5, 73.48), (95.0, 73.45),
     (94.7, 73.52), (95.0, 73.62), (95.5, 73.68),
     (95.9, 73.69)],
]


def point_on_segment(x, y, x1, y1, x2, y2, eps=1e-10):
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > eps:
        return False
    return (min(x1, x2) - eps <= x <= max(x1, x2) + eps and
            min(y1, y2) - eps <= y <= max(y1, y2) + eps)


def point_in_polygon(x, y, polygon):
    inside = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        if point_on_segment(x, y, x1, y1, x2, y2):
            return True
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if crossing_x > x:
                inside = not inside
    return inside


def point_in_rotated_ellipse(x, y, spec):
    cx, cy, rx, ry, angle_degrees = spec
    angle = math.radians(angle_degrees)
    dx, dy = x - cx, y - cy
    xr = dx * math.cos(angle) + dy * math.sin(angle)
    yr = -dx * math.sin(angle) + dy * math.cos(angle)
    return (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0


def classify_point(west_longitude, latitude):
    land = any(point_in_polygon(west_longitude, latitude, polygon)
               for polygon in LAND_POLYGONS)
    if not land:
        land = any(point_in_rotated_ellipse(west_longitude, latitude, item)
                   for item in ELLIPTICAL_ISLANDS)
    if land and any(point_in_polygon(west_longitude, latitude, polygon)
                    for polygon in WATER_CUTOUTS):
        land = False
    return int(land)


def generate_matrix():
    matrix = np.zeros((61, 241), dtype=np.uint8)
    for row in range(61):
        latitude = 79.0 - row * 0.1
        for column in range(241):
            west_longitude = 105.0 - column * 0.1
            matrix[row, column] = classify_point(west_longitude, latitude)
    return matrix


def validate_csv(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 61
    assert all(len(row) == 241 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert sum(map(len, rows)) == 14701


def main():
    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = generate_matrix()
    assert matrix.shape == (61, 241)
    assert matrix.size == 14701
    assert np.all((matrix == 0) | (matrix == 1))

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(matrix.tolist())

    pixels = (matrix * 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    image.save(output_dir / "candidate.png")
    enlarged = image.resize((1928, 488), Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")

    validate_csv(csv_path)
    assert image.size == (241, 61)
    assert enlarged.size == (1928, 488)
    assert set(image.getdata()) <= {0, 255}
    assert set(enlarged.getdata()) <= {0, 255}
    large = np.asarray(enlarged)
    assert np.all(large.reshape(61, 8, 241, 8) == pixels[:, None, :, None])


if __name__ == "__main__":
    main()
