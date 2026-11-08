"""Knowledge-only reconstruction of the Hecate Strait window.

Coordinates are written as (longitude, latitude).  They are deliberately
hand-drawn approximations, not data copied from a map or a geospatial file.
"""
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image

N = 127
WEST, EAST = -131.0, -129.0
NORTH, SOUTH = 55.0, 53.0


def inside(x, y, poly):
    """Ray-crossing point test, with boundary points treated as land."""
    hit = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        cross = ((yi > y) != (yj > y))
        if cross:
            xcross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x <= xcross:
                hit = not hit
        # Include a point exactly on a segment.
        dx, dy = xj - xi, yj - yi
        if abs(dx * (y - yi) - dy * (x - xi)) < 1e-12:
            if min(xi, xj) - 1e-12 <= x <= max(xi, xj) + 1e-12 and \
               min(yi, yj) - 1e-12 <= y <= max(yi, yj) + 1e-12:
                return True
        j = i
    return hit


# The mainland is represented by its western shore, followed by the
# east/north/south edges of the requested rectangle.  Small fjord mouths are
# intentionally approximated by the manually chosen shore vertices.
mainland = [
    (-129.0, 55.0), (-129.0, 54.93), (-129.18, 54.91),
    (-129.34, 54.84), (-129.48, 54.74), (-129.62, 54.63),
    (-129.78, 54.53), (-129.92, 54.42), (-130.08, 54.32),
    (-130.25, 54.27), (-130.18, 54.16), (-130.00, 54.08),
    (-129.86, 53.99), (-129.70, 53.89), (-129.57, 53.78),
    (-129.43, 53.67), (-129.29, 53.54), (-129.12, 53.40),
    (-128.96, 53.25), (-128.78, 53.0), (-129.0, 53.0),
]

# Major islands and island groups that intersect the window.
banks = [
    (-130.42, 54.70), (-130.18, 54.76), (-129.91, 54.69),
    (-129.73, 54.55), (-129.67, 54.36), (-129.72, 54.18),
    (-129.91, 54.08), (-130.17, 54.10), (-130.39, 54.25),
    (-130.55, 54.46), (-130.53, 54.60),
]

dundas = [
    (-130.93, 54.77), (-130.73, 54.88), (-130.43, 54.89),
    (-130.20, 54.82), (-130.10, 54.67), (-130.28, 54.60),
    (-130.55, 54.61), (-130.78, 54.67),
]

porcher = [
    (-130.55, 54.08), (-130.25, 54.13), (-129.91, 54.11),
    (-129.65, 53.99), (-129.59, 53.82), (-129.76, 53.71),
    (-130.08, 53.69), (-130.38, 53.78), (-130.57, 53.92),
]

pitt = [
    (-130.54, 53.62), (-130.39, 53.76), (-130.12, 53.78),
    (-129.87, 53.69), (-129.75, 53.52), (-129.84, 53.32),
    (-130.04, 53.20), (-130.34, 53.23), (-130.55, 53.39),
]
gil = [
    (-129.58, 53.72), (-129.35, 53.76), (-129.16, 53.66),
    (-129.08, 53.48), (-129.16, 53.30), (-129.38, 53.25),
    (-129.54, 53.39), (-129.62, 53.57),
]

# The north end of Princess Royal Island and a few adjacent shore islands
# enter only the southern edge of this two-degree window.
princess_royal = [
    (-129.72, 53.22), (-129.57, 53.38), (-129.39, 53.47),
    (-129.15, 53.40), (-128.93, 53.25), (-128.78, 53.0),
    (-129.72, 53.0),
]

haida_graham_edge = [
    (-131.0, 54.38), (-130.97, 54.27), (-130.98, 54.12),
    (-131.0, 54.03),
]

haida_moresby_edge = [
    (-131.0, 53.72), (-130.96, 53.56), (-130.97, 53.35),
    (-131.0, 53.17), (-131.0, 53.0),
]

# Small manually represented islands/rocks are included only where their
# approximate footprint is large enough to meet the sampling grid.
small_islands = [
    [(-130.78, 54.49), (-130.67, 54.53), (-130.60, 54.46),
     (-130.65, 54.38), (-130.77, 54.40)],
    [(-130.15, 54.47), (-130.07, 54.51), (-130.01, 54.46),
     (-130.06, 54.39), (-130.14, 54.40)],
    [(-130.72, 53.98), (-130.62, 54.02), (-130.56, 53.96),
     (-130.60, 53.90), (-130.70, 53.91)],
    [(-129.95, 53.92), (-129.86, 53.96), (-129.80, 53.89),
     (-129.84, 53.83), (-129.94, 53.85)],
    [(-129.78, 53.48), (-129.71, 53.52), (-129.65, 53.46),
     (-129.69, 53.40), (-129.76, 53.41)],
    [(-129.25, 53.12), (-129.16, 53.17), (-129.09, 53.10),
     (-129.13, 53.04), (-129.22, 53.05)],
]

land_polygons = [
    mainland, banks, dundas, porcher, pitt, gil, princess_royal,
    haida_graham_edge, haida_moresby_edge,
] + small_islands


def make_matrix():
    result = np.zeros((N, N), dtype=np.uint8)
    for row in range(N):
        lat = NORTH - (NORTH - SOUTH) * row / (N - 1)
        for col in range(N):
            lon = WEST + (EAST - WEST) * col / (N - 1)
            result[row, col] = any(inside(lon, lat, p) for p in land_polygons)
    return result


def write_outputs(matrix):
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # 0 is black water and 1 is white land.
    image = Image.fromarray((matrix * 255).astype(np.uint8), mode="L")
    image.save(out / "candidate.png")
    image.resize((N * 8, N * 8), Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )

def validate(matrix):
    assert matrix.shape == (127, 127)
    assert matrix.size == 16129
    assert set(np.unique(matrix)).issubset({0, 1})

    with open("outputs/candidate.csv", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 127
    assert all(len(row) == 127 for row in rows)
    assert all(v in {"0", "1"} for row in rows for v in row)

    original = Image.open("outputs/candidate.png")
    enlarged = Image.open("outputs/candidate_8x.png")
    assert original.size == (127, 127)
    assert enlarged.size == (1016, 1016)
    assert set(original.convert("L").getdata()).issubset({0, 255})
    assert set(enlarged.convert("L").getdata()).issubset({0, 255})
    assert enlarged.getpixel((8 * 17 + 3, 8 * 29 + 4)) == \
        original.getpixel((17, 29))


if __name__ == "__main__":
    matrix = make_matrix()
    write_outputs(matrix)
    validate(matrix)
    print("validated: 127x127, 16129 cells, binary CSV/PNGs, nearest-neighbor 8x")
