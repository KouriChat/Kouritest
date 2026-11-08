"""Hand-drawn, approximate land/water map of the Canadian Arctic Archipelago."""

from pathlib import Path
import csv

import numpy as np
from PIL import Image

# Vertices are (degrees west, degrees north). No external geographic data.
ISLANDS = {
    "Ellef Ringnes": [
        (104.5, 78.55), (104.0, 78.85), (103.0, 79.10),
        (101.8, 79.25), (100.7, 79.18), (99.5, 78.95),
        (98.9, 78.65), (99.3, 78.30), (100.2, 78.12),
        (101.0, 78.18), (101.5, 78.00), (102.5, 78.05),
        (103.3, 78.30), (104.0, 78.28),
    ],
    "King Christian": [
        (104.9, 77.40), (104.6, 77.68), (103.8, 77.94),
        (102.9, 77.99), (101.9, 77.78), (101.5, 77.46),
        (102.0, 77.23), (103.0, 77.25), (103.8, 77.36),
    ],
    "Amund Ringnes": [
        (98.0, 78.08), (97.5, 78.34), (96.7, 78.55),
        (95.7, 78.57), (94.8, 78.38), (94.5, 78.13),
        (95.0, 77.91), (96.0, 77.87), (97.0, 77.94),
    ],
    "Cornwall": [
        (97.1, 77.43), (96.8, 77.73), (96.0, 77.93),
        (95.0, 77.95), (94.0, 77.72), (93.7, 77.47),
        (94.2, 77.26), (95.2, 77.23), (96.3, 77.29),
    ],
    "Axel Heiberg": [
        (96.7, 79.30), (96.0, 79.13), (95.8, 78.91),
        (95.1, 78.82), (95.1, 78.55), (94.3, 78.42),
        (94.0, 78.17), (93.1, 78.08), (92.6, 78.23),
        (92.4, 78.51), (91.5, 78.69), (91.0, 78.98),
        (90.1, 79.13), (89.8, 79.35),
    ],
    "Ellesmere": [
        (90.2, 79.35), (89.4, 78.99), (88.8, 78.78),
        (88.0, 78.66), (87.8, 78.43), (86.6, 78.27),
        (86.2, 78.05), (84.6, 77.92), (83.8, 77.67),
        (82.8, 77.61), (82.0, 77.43), (81.0, 77.36),
        (80.5, 79.35),
    ],
    "Lougheed": [
        (105.4, 77.31), (104.9, 77.50), (104.3, 77.47),
        (103.8, 77.29), (104.2, 77.12), (105.3, 77.13),
    ],
    "Cameron": [
        (104.9, 76.58), (104.4, 76.94), (103.4, 77.10),
        (102.5, 76.99), (102.1, 76.70), (102.5, 76.38),
        (103.5, 76.31), (104.4, 76.38),
    ],
    "Vanier": [
        (104.9, 76.04), (104.2, 76.27), (103.4, 76.28),
        (103.0, 76.11), (103.3, 75.86), (104.2, 75.83),
    ],
    "Bathurst": [
        (101.8, 76.73), (101.2, 76.86), (100.9, 77.13),
        (100.0, 77.18), (99.2, 77.03), (99.1, 76.79),
        (98.3, 76.72), (98.0, 76.50), (97.4, 76.34),
        (97.4, 76.12), (98.1, 76.01), (98.5, 75.79),
        (98.1, 75.57), (98.7, 75.43), (99.5, 75.48),
        (99.9, 75.72), (100.5, 75.70), (100.9, 75.94),
        (101.6, 75.94), (101.9, 76.15), (101.5, 76.38),
    ],
    "Alexander": [
        (104.1, 75.84), (103.6, 76.05), (102.9, 76.05),
        (102.4, 75.87), (102.6, 75.64), (103.4, 75.60),
    ],
    "Byam Martin": [
        (105.4, 75.57), (104.8, 75.83), (104.1, 75.86),
        (103.9, 75.63), (104.3, 75.40), (105.2, 75.36),
    ],
    "Melville eastern edge": [
        (105.5, 76.91), (105.0, 76.90), (104.8, 76.66),
        (105.1, 76.38), (104.9, 76.08), (105.5, 75.94),
    ],
    "Graham": [
        (91.8, 77.32), (91.3, 77.57), (90.5, 77.72),
        (89.8, 77.61), (89.2, 77.39), (89.5, 77.20),
        (90.6, 77.16), (91.4, 77.21),
    ],
    "North Kent": [
        (91.4, 76.83), (90.9, 77.07), (90.1, 77.11),
        (89.6, 76.94), (89.9, 76.74), (90.8, 76.69),
    ],
    "Devon": [
        (95.0, 76.27), (94.7, 76.53), (93.8, 76.72),
        (92.8, 76.66), (92.3, 76.47), (91.6, 76.53),
        (91.0, 76.67), (90.1, 76.54), (89.3, 76.62),
        (88.5, 76.49), (87.9, 76.61), (86.8, 76.68),
        (85.9, 76.56), (85.2, 76.65), (84.1, 76.57),
        (83.3, 76.38), (82.4, 76.32), (81.7, 76.12),
        (80.5, 75.99), (80.5, 74.41), (81.4, 74.45),
        (82.2, 74.69), (83.0, 74.71), (83.7, 74.60),
        (84.5, 74.79), (85.4, 74.75), (86.0, 74.52),
        (86.8, 74.44), (87.6, 74.56), (88.4, 74.38),
        (89.3, 74.37), (90.2, 74.57), (90.6, 74.84),
        (91.1, 74.93), (91.0, 75.19), (91.7, 75.37),
        (92.5, 75.53), (93.0, 75.77), (93.8, 75.95),
        (94.7, 75.99),
    ],
    "Cornwallis": [
        (96.8, 75.29), (96.5, 75.61), (95.9, 75.83),
        (95.1, 75.88), (94.3, 75.75), (93.7, 75.84),
        (93.3, 75.66), (92.8, 75.52), (92.9, 75.23),
        (92.7, 74.96), (93.2, 74.74), (94.1, 74.62),
        (94.8, 74.48), (95.6, 74.62), (96.2, 74.85),
        (96.6, 75.02),
    ],
    "Little Cornwallis": [
        (97.9, 75.21), (97.6, 75.36), (97.1, 75.36),
        (96.9, 75.17), (97.2, 75.04), (97.7, 75.06),
    ],
    "Prince of Wales": [
        (99.7, 74.23), (99.0, 74.28), (98.4, 74.17),
        (97.7, 74.14), (97.2, 73.98), (96.8, 73.71),
        (96.2, 73.55), (96.3, 73.15), (96.6, 72.7),
        (102.1, 72.7), (102.0, 73.10), (101.7, 73.37),
        (101.1, 73.49), (100.7, 73.75), (100.0, 73.89),
    ],
    "Somerset": [
        (94.5, 73.93), (93.8, 74.07), (93.0, 74.10),
        (92.3, 73.99), (91.5, 74.04), (90.8, 73.96),
        (90.0, 74.02), (89.5, 73.91), (88.7, 73.83),
        (88.1, 73.63), (88.3, 73.31), (88.9, 73.15),
        (88.5, 72.7), (94.1, 72.7), (94.2, 73.15),
        (94.7, 73.41), (94.4, 73.64),
    ],
    "Baffin northern edge": [
        (87.8, 72.7), (87.0, 73.02), (86.1, 73.14),
        (85.5, 73.34), (84.7, 73.36), (83.9, 73.57),
        (83.0, 73.58), (82.4, 73.42), (81.9, 73.20),
        (81.5, 73.04), (80.5, 72.7),
    ],
    "Bylot western edge": [
        (82.3, 73.02), (82.0, 73.29), (81.7, 73.55),
        (81.1, 73.66), (80.5, 73.64), (80.5, 72.7),
        (81.4, 72.7),
    ],
}

# Selected smaller islands: longitude, latitude, longitudinal and latitudinal radii.
SMALL_ISLANDS = [
    (101.1, 77.57, 0.28, 0.11),
    (98.2, 77.39, 0.22, 0.10),
    (97.6, 77.72, 0.20, 0.10),
    (93.5, 77.77, 0.22, 0.09),
    (92.1, 77.64, 0.20, 0.08),
    (92.0, 76.91, 0.19, 0.08),
    (95.0, 76.17, 0.20, 0.08),
    (96.9, 76.63, 0.17, 0.07),
    (102.3, 75.58, 0.19, 0.08),
    (96.9, 75.73, 0.17, 0.07),
]


def inside_polygon(west, north, vertices):
    """Return point-in-polygon mask using the even/odd crossing rule."""
    inside = np.zeros(west.shape, dtype=bool)
    for i, (x1, y1) in enumerate(vertices):
        x2, y2 = vertices[(i + 1) % len(vertices)]
        if y1 == y2:
            continue
        crosses = (y1 > north) != (y2 > north)
        x_at_lat = x1 + (north - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (west < x_at_lat)
    return inside


def main():
    # Integer-based coordinates avoid accumulating floating-point step errors.
    west = (1050 - np.arange(241)) / 10.0
    north = (790 - np.arange(61)) / 10.0
    ww, nn = np.meshgrid(west, north)

    land = np.zeros((61, 241), dtype=bool)
    for vertices in ISLANDS.values():
        land |= inside_polygon(ww, nn, vertices)
    for x, y, rx, ry in SMALL_ISLANDS:
        land |= ((ww - x) / rx) ** 2 + ((nn - y) / ry) ** 2 <= 1

    matrix = land.astype(np.uint8)
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "candidate.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows(matrix.tolist())

    pixels = (matrix * 255).astype(np.uint8)
    original = Image.fromarray(pixels, mode="L")
    original.save(out / "candidate.png")
    enlarged = original.resize((1928, 488), Image.Resampling.NEAREST)
    enlarged.save(out / "candidate_8x.png")

    # Check the exported files, not just the in-memory matrix.
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    assert len(rows) == 61
    assert all(len(row) == 241 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert sum(map(len, rows)) == 14701
    assert np.array_equal(np.array(rows, dtype=np.uint8), matrix)

    with Image.open(out / "candidate.png") as image:
        base = np.asarray(image.convert("L"))
        assert image.size == (241, 61)
        assert set(np.unique(base).tolist()) <= {0, 255}

    with Image.open(out / "candidate_8x.png") as image:
        scaled = np.asarray(image.convert("L"))
        assert image.size == (1928, 488)
        assert set(np.unique(scaled).tolist()) <= {0, 255}
        assert np.array_equal(
            scaled, np.repeat(np.repeat(base, 8, axis=0), 8, axis=1)
        )

    print("Generated and checked outputs/candidate.csv and both PNG files")


if __name__ == "__main__":
    main()
