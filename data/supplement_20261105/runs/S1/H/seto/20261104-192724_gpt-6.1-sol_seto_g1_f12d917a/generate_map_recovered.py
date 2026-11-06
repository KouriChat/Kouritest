#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hand-built land/water reconstruction for Seto Inland Sea and surroundings.

Outputs:
  outputs/candidate.csv
  outputs/candidate.png
  outputs/candidate_8x.png

No network access, map data, GIS database, or external coastline data is used.
"""

from pathlib import Path
import csv
from PIL import Image


# Grid definition.
ROWS = 64
COLS = 253
LAT_NORTH = 35.0
LON_WEST = 129.0
STEP = 2.0 / 63.0


def point_on_segment(x, y, ax, ay, bx, by, eps=1.0e-10):
    """Return True if a point lies on a polygon edge."""
    cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
    if abs(cross) > eps:
        return False

    return (
        min(ax, bx) - eps <= x <= max(ax, bx) + eps
        and min(ay, by) - eps <= y <= max(ay, by) + eps
    )


def point_in_polygon(x, y, polygon):
    """
    Ray-casting point-in-polygon test.

    Boundary points are treated as land.
    """
    inside = False
    n = len(polygon)

    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]

        if point_on_segment(x, y, ax, ay, bx, by):
            return True

        crosses = ((ay > y) != (by > y))
        if crosses:
            x_intersection = ax + (y - ay) * (bx - ax) / (by - ay)
            if x < x_intersection:
                inside = not inside

    return inside


def point_in_ellipse(x, y, cx, cy, rx, ry):
    """Axis-aligned ellipse used for small rounded islands."""
    if rx <= 0.0 or ry <= 0.0:
        return False

    dx = (x - cx) / rx
    dy = (y - cy) / ry
    return dx * dx + dy * dy <= 1.0


# Approximate mainland and large-island polygons.
#
# Polygon coordinates are (longitude, latitude).
# The polygons intentionally extend outside the requested frame where useful;
# only points inside the 129E..137E, 33N..35N sampling frame are evaluated.
LAND_POLYGONS = [
    # Southeastern Korean Peninsula, clipped by the northern frame edge.
    [
        (129.00, 35.00),
        (129.30, 34.95),
        (129.55, 35.05),
        (129.80, 35.20),
        (130.10, 35.45),
        (129.00, 35.45),
    ],

    # Kyushu.
    [
        (129.38, 33.00),
        (129.38, 33.25),
        (129.30, 33.50),
        (129.20, 33.75),
        (129.25, 33.98),
        (129.52, 34.03),
        (129.75, 33.87),
        (130.00, 33.75),
        (130.30, 33.70),
        (130.50, 33.82),
        (130.78, 33.93),
        (131.00, 33.92),
        (131.20, 33.70),
        (131.45, 33.55),
        (131.70, 33.30),
        (131.82, 33.00),
    ],

    # Western and central Honshu, including the northern shore of the
    # Seto Inland Sea and the land north of Ise Bay.
    [
        (130.78, 34.02),
        (130.90, 34.08),
        (131.15, 34.12),
        (131.35, 34.18),
        (131.55, 34.20),
        (131.75, 34.25),
        (131.95, 34.25),
        (132.15, 34.30),
        (132.35, 34.30),
        (132.55, 34.35),
        (132.75, 34.35),
        (132.95, 34.40),
        (133.15, 34.40),
        (133.35, 34.40),
        (133.55, 34.45),
        (133.75, 34.50),
        (133.95, 34.55),
        (134.15, 34.60),
        (134.35, 34.63),
        (134.55, 34.67),
        (134.75, 34.68),
        (134.95, 34.65),
        (135.15, 34.72),
        (135.45, 34.77),
        (135.72, 34.84),
        (136.00, 34.86),
        (136.30, 34.84),
        (136.55, 34.78),
        (136.72, 34.68),
        (136.84, 34.50),
        (137.00, 34.45),
        (137.00, 35.45),
        (130.78, 35.45),
    ],

    # Shikoku.
    [
        (131.95, 33.33),
        (132.15, 33.50),
        (132.16, 33.75),
        (132.35, 33.95),
        (132.55, 34.15),
        (132.80, 34.30),
        (133.10, 34.40),
        (133.40, 34.42),
        (133.70, 34.30),
        (134.00, 34.15),
        (134.35, 34.05),
        (134.62, 33.90),
        (134.72, 33.65),
        (134.55, 33.40),
        (134.35, 33.25),
        (134.00, 33.20),
        (133.65, 33.15),
        (133.30, 33.10),
        (132.90, 33.12),
        (132.55, 33.20),
        (132.25, 33.25),
    ],

    # Kii Peninsula.
    [
        (135.05, 34.62),
        (135.20, 34.55),
        (135.25, 34.35),
        (135.15, 34.15),
        (135.30, 33.90),
        (135.45, 33.65),
        (135.65, 33.45),
        (135.90, 33.30),
        (136.15, 33.23),
        (136.35, 33.28),
        (136.55, 33.45),
        (136.68, 33.72),
        (136.78, 34.02),
        (136.82, 34.30),
        (136.80, 34.50),
        (136.50, 34.55),
        (136.10, 34.52),
        (135.70, 34.55),
        (135.30, 34.62),
    ],

    # Awaji Island.
    [
        (134.62, 34.60),
        (134.82, 34.64),
        (135.02, 34.56),
        (135.15, 34.35),
        (135.08, 34.15),
        (134.90, 34.08),
        (134.72, 34.15),
        (134.60, 34.35),
        (134.55, 34.50),
    ],

    # Shodoshima.
    [
        (133.90, 34.55),
        (134.12, 34.58),
        (134.35, 34.50),
        (134.42, 34.33),
        (134.30, 34.22),
        (134.05, 34.24),
        (133.90, 34.38),
    ],

    # Chita Peninsula in Ise Bay.
    [
        (136.75, 35.45),
        (137.00, 35.45),
        (137.00, 34.48),
        (136.88, 34.55),
        (136.78, 34.75),
        (136.72, 35.00),
    ],

    # Small visible part of the Atsumi side at the eastern edge.
    [
        (136.94, 34.82),
        (137.00, 34.82),
        (137.00, 34.45),
        (136.92, 34.55),
    ],

    # Northern Tsushima.
    [
        (129.25, 34.62),
        (129.38, 34.68),
        (129.50, 34.62),
        (129.52, 34.50),
        (129.45, 34.38),
        (129.30, 34.38),
        (129.20, 34.50),
    ],

    # Southern Tsushima.
    [
        (129.15, 34.28),
        (129.30, 34.35),
        (129.38, 34.25),
        (129.35, 34.08),
        (129.18, 34.08),
        (129.08, 34.18),
    ],
]


# Rounded approximations for smaller islands and reef groups.
# Each tuple is (center_lon, center_lat, radius_lon, radius_lat).
LAND_ELLIPSES = [
    # Iki Island.
    (129.75, 33.80, 0.13, 0.105),

    # Hirado and nearby islands.
    (129.55, 33.35, 0.075, 0.14),
    (129.43, 33.48, 0.045, 0.07),

    # Islands between northern Kyushu and Shikoku.
    (131.62, 33.72, 0.075, 0.135),
    (131.84, 33.78, 0.045, 0.06),
    (132.04, 33.87, 0.045, 0.085),

    # Suo / Hiroshima side.
    (132.22, 34.02, 0.10, 0.13),
    (132.28, 34.31, 0.075, 0.12),
    (132.43, 34.22, 0.13, 0.11),
    (132.62, 34.15, 0.09, 0.10),
    (132.73, 34.30, 0.055, 0.075),
    (132.90, 34.23, 0.095, 0.095),

    # Geiyo Islands.
    (133.00, 34.31, 0.12, 0.12),
    (133.12, 34.23, 0.09, 0.105),
    (133.18, 34.13, 0.12, 0.095),
    (133.33, 34.18, 0.105, 0.08),
    (133.48, 34.24, 0.07, 0.07),
    (133.58, 34.15, 0.075, 0.085),
    (133.73, 34.22, 0.07, 0.07),
    (133.82, 34.30, 0.05, 0.06),

    # Islands near Kagawa and the eastern Seto Inland Sea.
    (133.98, 34.46, 0.065, 0.065),
    (134.04, 34.40, 0.045, 0.055),
    (134.18, 34.43, 0.07, 0.065),
    (134.38, 34.43, 0.06, 0.06),
    (134.55, 34.38, 0.055, 0.06),
    (134.78, 34.43, 0.065, 0.055),

    # Islands around Osaka Bay.
    (135.02, 34.42, 0.045, 0.045),
    (135.30, 34.38, 0.04, 0.045),
    (135.48, 34.35, 0.04, 0.04),

    # Islands near the Kii channel and Shikoku coast.
    (134.62, 34.02, 0.07, 0.05),
    (134.82, 33.98, 0.05, 0.045),
    (134.98, 33.92, 0.045, 0.04),
    (135.20, 33.85, 0.04, 0.045),

    # Small islands off Shikoku's northern and western sides.
    (132.42, 33.72, 0.045, 0.06),
    (132.68, 33.75, 0.055, 0.045),
    (132.92, 33.82, 0.045, 0.05),
    (133.32, 33.86, 0.045, 0.045),
]


def is_land(lon, lat):
    """
    Point-only land test.

    The test is a union of hand-written polygons and ellipses.
    There is no cell-area averaging or neighborhood voting.
    """
    for polygon in LAND_POLYGONS:
        if point_in_polygon(lon, lat, polygon):
            return True

    for cx, cy, rx, ry in LAND_ELLIPSES:
        if point_in_ellipse(lon, lat, cx, cy, rx, ry):
            return True

    return False


def build_matrix():
    """Build rows from north to south and columns from west to east."""
    matrix = []

    for row in range(ROWS):
        lat = LAT_NORTH - row * STEP
        current_row = []

        for col in range(COLS):
            lon = LON_WEST + col * STEP
            current_row.append(1 if is_land(lon, lat) else 0)

        matrix.append(current_row)

    return matrix


def write_csv(matrix, path):
    """Write an unheaded CSV containing only 0 and 1."""
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for row in matrix:
            writer.writerow(row)


def write_images(matrix, original_path, enlarged_path):
    """
    Create the original binary-color PNG and an exact nearest-neighbor 8x copy.
    """
    pixels = []
    for row in matrix:
        for value in row:
            pixels.append(255 if value == 1 else 0)

    original = Image.new("L", (COLS, ROWS))
    original.putdata(pixels)
    original.save(original_path, format="PNG")

    enlarged = original.resize(
        (COLS * 8, ROWS * 8),
        resample=Image.Resampling.NEAREST,
    )
    enlarged.save(enlarged_path, format="PNG")


def verify_outputs(matrix, csv_path, original_path, enlarged_path):
    """Perform the requested structural and image checks."""
    assert len(matrix) == 64
    assert all(len(row) == 253 for row in matrix)
    assert sum(len(row) for row in matrix) == 16192
    assert all(value in (0, 1) for row in matrix for value in row)

    with csv_path.open("r", newline="", encoding="ascii") as handle:
        rows = list(csv.reader(handle))

    assert len(rows) == 64
    assert all(len(row) == 253 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert all(value != "" for row in rows for value in row)

    original = Image.open(original_path).convert("L")
    enlarged = Image.open(enlarged_path).convert("L")

    assert original.size == (253, 64)
    assert enlarged.size == (2024, 512)

    original_colors = set(original.getdata())
    enlarged_colors = set(enlarged.getdata())

    assert original_colors <= {0, 255}
    assert enlarged_colors <= {0, 255}

    # Verify exact nearest-neighbor replication.
    for y in range(64):
        for x in range(253):
            source_value = original.getpixel((x, y))
            for yy in range(y * 8, y * 8 + 8):
                for xx in range(x * 8, x * 8 + 8):
                    assert enlarged.getpixel((xx, yy)) == source_value


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "candidate.csv"
    original_path = output_dir / "candidate.png"
    enlarged_path = output_dir / "candidate_8x.png"

    matrix = build_matrix()
    write_csv(matrix, csv_path)
    write_images(matrix, original_path, enlarged_path)
    verify_outputs(matrix, csv_path, original_path, enlarged_path)

    land_count = sum(value for row in matrix for value in row)
    water_count = ROWS * COLS - land_count

    print("Generated:")
    print(f"  {csv_path}")
    print(f"  {original_path}")
    print(f"  {enlarged_path}")
    print(f"Matrix: {ROWS} rows x {COLS} columns = {ROWS * COLS} points")
    print(f"Land points: {land_count}")
    print(f"Water points: {water_count}")
    print("Validation: passed")


if __name__ == "__main__":
    main()
