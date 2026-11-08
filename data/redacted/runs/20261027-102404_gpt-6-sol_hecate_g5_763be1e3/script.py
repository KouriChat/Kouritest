"""Hand-drawn approximate land/water map for the Hecate Strait area.

Coordinates below are manually estimated (longitude, latitude); no geographic
data sources or map APIs are used.
"""

from pathlib import Path
import csv

import numpy as np
from PIL import Image


MAINLAND = [
    (-130.00, 55.00), (-129.96, 54.94), (-130.08, 54.88),
    (-130.19, 54.80), (-130.24, 54.72), (-130.17, 54.64),
    (-130.10, 54.58), (-130.08, 54.49), (-130.02, 54.43),
    (-130.09, 54.37), (-130.00, 54.32), (-129.98, 54.25),
    (-129.85, 54.20), (-129.79, 54.12), (-129.77, 54.04),
    (-129.65, 53.98), (-129.61, 53.88), (-129.53, 53.79),
    (-129.49, 53.70), (-129.37, 53.62), (-129.31, 53.52),
    (-129.18, 53.46), (-129.11, 53.37), (-129.14, 53.29),
    (-129.01, 53.21), (-128.80, 53.00),
    (-128.50, 53.00), (-128.50, 55.00),
]

ISLANDS = [
    # Islands by Dixon Entrance.
    [(-130.98, 54.57), (-130.92, 54.65), (-130.78, 54.69),
     (-130.66, 54.65), (-130.62, 54.56), (-130.72, 54.49),
     (-130.85, 54.47), (-130.96, 54.51)],
    [(-130.71, 54.69), (-130.62, 54.74), (-130.50, 54.72),
     (-130.47, 54.66), (-130.58, 54.62), (-130.68, 54.63)],
    # Islands near Prince Rupert.
    [(-130.60, 54.34), (-130.55, 54.41), (-130.47, 54.42),
     (-130.41, 54.38), (-130.44, 54.30), (-130.51, 54.26),
     (-130.58, 54.29)],
    [(-130.43, 54.34), (-130.38, 54.39), (-130.29, 54.39),
     (-130.25, 54.33), (-130.30, 54.27), (-130.38, 54.28)],
    [(-130.99, 54.17), (-130.94, 54.24), (-130.81, 54.28),
     (-130.69, 54.23), (-130.63, 54.16), (-130.71, 54.09),
     (-130.87, 54.08), (-130.97, 54.12)],
    # Porcher Island.
    [(-130.80, 53.94), (-130.75, 54.04), (-130.65, 54.12),
     (-130.54, 54.13), (-130.44, 54.08), (-130.38, 54.01),
     (-130.32, 53.95), (-130.37, 53.87), (-130.50, 53.82),
     (-130.65, 53.83), (-130.76, 53.88)],
    # Banks Island.
    [(-130.52, 53.72), (-130.47, 53.82), (-130.37, 53.88),
     (-130.27, 53.85), (-130.19, 53.77), (-130.16, 53.67),
     (-130.11, 53.59), (-130.14, 53.48), (-130.23, 53.37),
     (-130.34, 53.32), (-130.41, 53.39), (-130.44, 53.52)],
    # Pitt Island; channels remain between it and its neighbors.
    [(-130.13, 53.94), (-130.06, 54.00), (-129.99, 53.98),
     (-129.94, 53.91), (-129.92, 53.82), (-129.86, 53.74),
     (-129.83, 53.64), (-129.77, 53.56), (-129.77, 53.47),
     (-129.84, 53.41), (-129.91, 53.46), (-129.98, 53.56),
     (-130.04, 53.67), (-130.08, 53.78)],
    # Southern coastal islands.
    [(-129.82, 53.19), (-129.78, 53.32), (-129.70, 53.44),
     (-129.62, 53.52), (-129.54, 53.58), (-129.48, 53.56),
     (-129.43, 53.46), (-129.42, 53.36), (-129.46, 53.25),
     (-129.51, 53.14), (-129.55, 53.00), (-129.79, 53.00)],
    [(-129.66, 53.00), (-129.63, 53.11), (-129.55, 53.18),
     (-129.48, 53.15), (-129.45, 53.07), (-129.48, 53.00)],
    [(-131.08, 53.00), (-131.06, 53.10), (-131.01, 53.16),
     (-130.96, 53.13), (-130.94, 53.05), (-130.97, 53.00)],
]

# (longitude, latitude, longitude radius, latitude radius).
ISLETS = [
    (-130.88, 54.40, .036, .025), (-130.72, 54.38, .029, .020),
    (-130.39, 54.53, .031, .026), (-130.30, 54.47, .025, .020),
    (-130.89, 54.01, .037, .024), (-130.95, 53.79, .035, .027),
    (-130.81, 53.72, .036, .024), (-130.67, 53.70, .040, .029),
    (-130.55, 53.57, .037, .026), (-130.62, 53.40, .037, .025),
    (-130.48, 53.26, .036, .026), (-130.29, 53.18, .043, .028),
    (-130.10, 53.29, .033, .024), (-129.96, 53.22, .038, .027),
    (-129.89, 53.08, .038, .027), (-129.90, 53.80, .021, .033),
    (-129.72, 53.79, .029, .039), (-129.59, 53.71, .027, .038),
    (-129.32, 53.40, .026, .030), (-129.28, 53.16, .031, .040),
]

# Channels within the approximate mainland polygon are always water.
INLETS = [
    [(-130.27, 54.72), (-130.15, 54.76), (-130.08, 54.85),
     (-129.96, 54.92), (-129.84, 55.00), (-130.01, 55.00),
     (-130.15, 54.92), (-130.23, 54.82)],
    [(-130.18, 54.37), (-130.05, 54.40), (-129.92, 54.39),
     (-129.86, 54.35), (-129.97, 54.31), (-130.09, 54.32)],
]

def in_polygon(x, y, polygon):
    """Ray-casting point-in-polygon test."""
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        if (current_y > y) != (previous_y > y):
            crossing_x = current_x + (y - current_y) * (
                previous_x - current_x
            ) / (previous_y - current_y)
            if x < crossing_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def land_at(lon, lat):
    if any(in_polygon(lon, lat, inlet) for inlet in INLETS):
        return 0
    if in_polygon(lon, lat, MAINLAND):
        return 1
    if any(in_polygon(lon, lat, island) for island in ISLANDS):
        return 1
    for cx, cy, rx, ry in ISLETS:
        if ((lon - cx) / rx) ** 2 + ((lat - cy) / ry) ** 2 <= 1:
            return 1
    return 0


def main():
    output = Path("outputs")
    output.mkdir(parents=True, exist_ok=True)
    matrix = np.array([
        [land_at(-131 + col * 2 / 126, 55 - row * 2 / 126)
         for col in range(127)]
        for row in range(127)
    ], dtype=np.uint8)
    csv_path = output / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as file:
        csv.writer(file).writerows(matrix.tolist())
    pixels = matrix * 255
    image = Image.fromarray(pixels, mode="L")
    original_path = output / "candidate.png"
    enlarged_path = output / "candidate_8x.png"
    image.save(original_path)
    image.resize((1016, 1016), Image.Resampling.NEAREST).save(enlarged_path)
    with csv_path.open(newline="", encoding="ascii") as file:
        rows = list(csv.reader(file))
    assert len(rows) == 127 and all(len(row) == 127 for row in rows)
    assert all(value in ("0", "1") for row in rows for value in row)
    with Image.open(original_path) as saved:
        original = np.asarray(saved.convert("L"))
        assert saved.size == (127, 127)
    with Image.open(enlarged_path) as saved:
        enlarged = np.asarray(saved.convert("L"))
        assert saved.size == (1016, 1016)
    assert set(np.unique(original)) <= {0, 255}
    assert set(np.unique(enlarged)) <= {0, 255}
    assert np.array_equal(original, pixels)
    assert np.array_equal(enlarged, np.repeat(np.repeat(original, 8, 0), 8, 1))


if __name__ == "__main__":
    main()
