from pathlib import Path
import csv

import numpy as np
from PIL import Image


# Hand-estimated coastline coordinates only; no external geographic data.
# Coordinates use (longitude, latitude), with west longitudes negative.
MAINLAND_WEST_EDGE = [
    (-130.52, 55.00), (-130.42, 54.88), (-130.34, 54.76),
    (-130.28, 54.63), (-130.30, 54.50), (-130.20, 54.37),
    (-130.12, 54.22), (-130.06, 54.05), (-130.02, 53.88),
    (-129.95, 53.70), (-129.91, 53.52), (-129.83, 53.34),
    (-129.72, 53.16), (-129.63, 53.00),
]

# Major fjords and inlets cut into the mainland. These are deliberately
# simplified polygons, not surveyed shorelines.
INLETS = [
    [(-130.62, 55.00), (-130.39, 54.94), (-130.31, 54.83),
     (-130.37, 54.72), (-130.51, 54.65), (-130.73, 54.64),
     (-130.71, 54.78), (-130.62, 55.00)],
    [(-130.43, 54.78), (-130.33, 54.76), (-130.22, 54.62),
     (-130.20, 54.43), (-130.27, 54.34), (-130.38, 54.48),
     (-130.43, 54.78)],
    [(-130.28, 54.52), (-130.19, 54.45), (-130.08, 54.27),
     (-129.99, 54.10), (-129.94, 53.96), (-130.02, 53.91),
     (-130.14, 54.10), (-130.24, 54.31), (-130.28, 54.52)],
    [(-130.18, 54.20), (-130.10, 54.12), (-130.00, 53.95),
     (-129.91, 53.78), (-129.86, 53.62), (-129.93, 53.59),
     (-130.04, 53.77), (-130.13, 53.96), (-130.18, 54.20)],
    [(-129.97, 53.92), (-129.88, 53.85), (-129.78, 53.68),
     (-129.68, 53.50), (-129.62, 53.34), (-129.69, 53.31),
     (-129.81, 53.50), (-129.91, 53.70), (-129.97, 53.92)],
]

ISLANDS = [
    [(-130.38, 54.58), (-130.30, 54.62), (-130.25, 54.57),
     (-130.27, 54.48), (-130.34, 54.46), (-130.39, 54.51)],
    [(-130.20, 54.27), (-130.13, 54.31), (-130.08, 54.26),
     (-130.10, 54.18), (-130.17, 54.17), (-130.22, 54.21)],
    # Banks Island and neighboring islands in the eastern strait.
    [(-130.23, 54.22), (-130.10, 54.25), (-129.98, 54.19),
     (-129.96, 54.04), (-130.02, 53.91), (-130.14, 53.91),
     (-130.22, 54.04)],
    [(-130.31, 54.01), (-130.24, 54.05), (-130.18, 53.98),
     (-130.20, 53.89), (-130.28, 53.87), (-130.33, 53.93)],
    # Pitt Island and the cluster south of Banks Island.
    [(-129.86, 53.77), (-129.73, 53.80), (-129.61, 53.72),
     (-129.57, 53.56), (-129.62, 53.39), (-129.74, 53.34),
     (-129.84, 53.45), (-129.89, 53.61)],
    [(-130.02, 53.67), (-129.93, 53.70), (-129.88, 53.63),
     (-129.91, 53.54), (-129.99, 53.52), (-130.04, 53.59)],
    [(-129.55, 53.36), (-129.47, 53.40), (-129.41, 53.33),
     (-129.43, 53.24), (-129.51, 53.21), (-129.57, 53.27)],
    [(-129.89, 53.21), (-129.80, 53.24), (-129.73, 53.16),
     (-129.77, 53.07), (-129.85, 53.06), (-129.91, 53.13)],
]
def inside(point, polygon):
    """Ray-casting point-in-polygon test for a hand-drawn outline."""
    x, y = point
    hit = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            crossing_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < crossing_x:
                hit = not hit
        j = i
    return hit


def make_matrix():
    # Include both endpoints exactly: 127 samples across two degrees.
    latitudes = np.linspace(55.0, 53.0, 127)
    longitudes = np.linspace(-131.0, -129.0, 127)
    matrix = np.zeros((127, 127), dtype=np.uint8)
    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            point = (float(lon), float(lat))
            mainland = inside(
                point,
                MAINLAND_WEST_EDGE + [(-129.0, 53.0), (-129.0, 55.0)],
            )
            if mainland and not any(inside(point, inlet) for inlet in INLETS):
                matrix[row, col] = 1
            if any(inside(point, island) for island in ISLANDS):
                matrix[row, col] = 1
    return matrix


def main():
    output = Path("outputs")
    output.mkdir(parents=True, exist_ok=True)
    matrix = make_matrix()

    with (output / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Grayscale 0 is black water; 255 is white land.
    original = Image.fromarray(matrix * 255, mode="L")
    original.save(output / "candidate.png")
    enlarged = original.resize((1016, 1016), resample=Image.Resampling.NEAREST)
    enlarged.save(output / "candidate_8x.png")

    with (output / "candidate.csv").open("r", newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 127 and all(len(row) == 127 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert matrix.shape == (127, 127) and matrix.size == 16129
    assert original.size == (127, 127)
    assert set(original.getdata()) <= {0, 255}
    assert enlarged.size == (1016, 1016)
    assert set(enlarged.getdata()) <= {0, 255}
    for y in range(127):
        for x in range(127):
            assert enlarged.getpixel((8 * x, 8 * y)) == original.getpixel((x, y))


if __name__ == "__main__":
    main()
