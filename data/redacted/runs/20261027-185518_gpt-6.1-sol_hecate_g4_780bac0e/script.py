"""Knowledge-only reconstruction of the Hecate Strait window.

The coastline is deliberately encoded as hand-written, approximate polygons.
No geographic data are read by this program.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

N = 127
WEST, EAST = -131.0, -129.0
NORTH, SOUTH = 55.0, 53.0


def poly(points):
    return tuple((float(x), float(y)) for x, y in points)


def inside(x, y, p):
    """Even-odd point-in-polygon test, with boundary treated as land."""
    hit = False
    n = len(p)
    for k in range(n):
        x1, y1 = p[k]
        x2, y2 = p[(k + 1) % n]
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) < 1.0e-12 and min(x1, x2) - 1e-12 <= x <= max(x1, x2) + 1e-12 and min(y1, y2) - 1e-12 <= y <= max(y1, y2) + 1e-12:
            return True
        if (y1 > y) != (y2 > y):
            at_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < at_x:
                hit = not hit
    return hit


# The main coast is represented by its west-facing shore followed by the
# eastern edge of the requested window.  The latter is intentionally allowed
# to run out of frame: only points in the 2 by 2 degree window are sampled.
MAINLAND = poly([
    (-129.97, 55.00), (-129.91, 54.88), (-130.00, 54.77),
    (-130.13, 54.66), (-130.27, 54.52), (-130.36, 54.39),
    (-130.29, 54.27), (-130.10, 54.19), (-129.92, 54.09),
    (-129.78, 53.96), (-129.62, 53.83), (-129.49, 53.68),
    (-129.37, 53.55), (-129.25, 53.40), (-129.15, 53.24),
    (-129.08, 53.00), (-129.00, 53.00), (-129.00, 55.00),
])

# Water bites in the mainland: Portland/Observatory inlets and the larger
# sounds around the Prince Rupert and Grenville Channel approaches.
MAINLAND_WATER = [
    poly([
        (-129.98, 55.01), (-129.72, 55.01), (-129.66, 54.91),
        (-129.73, 54.80), (-129.86, 54.72), (-129.98, 54.76),
    ]),
    poly([
        (-130.04, 54.73), (-129.79, 54.68), (-129.70, 54.57),
        (-129.77, 54.48), (-129.91, 54.46), (-130.05, 54.53),
    ]),
    poly([
        (-130.24, 54.40), (-130.07, 54.35), (-129.91, 54.28),
        (-129.88, 54.18), (-130.01, 54.12), (-130.17, 54.20),
    ]),
    poly([
        (-129.83, 54.08), (-129.66, 54.02), (-129.52, 53.91),
        (-129.53, 53.79), (-129.64, 53.75), (-129.75, 53.86),
    ]),
]

# Islands and island groups in the eastern Hecate Strait and Dixon Entrance.
# Coordinates are approximate sketches, not imported coastline data.
ISLANDS = [
    # Dundas Island and its small eastern satellites
    poly([
        (-131.02, 54.78), (-130.91, 54.86), (-130.73, 54.84),
        (-130.58, 54.76), (-130.56, 54.62), (-130.63, 54.49),
        (-130.78, 54.43), (-130.94, 54.47), (-131.02, 54.59),
    ]),
    poly([
        (-130.59, 54.72), (-130.52, 54.76), (-130.46, 54.71),
        (-130.46, 54.62), (-130.53, 54.59), (-130.59, 54.63),
    ]),
    poly([
        (-130.72, 54.43), (-130.66, 54.48), (-130.58, 54.45),
        (-130.57, 54.37), (-130.64, 54.34), (-130.72, 54.37),
    ]),
    # Stephens Island, at the mouth of Portland Inlet
    poly([
        (-130.99, 54.57), (-130.92, 54.63), (-130.82, 54.62),
        (-130.77, 54.54), (-130.81, 54.45), (-130.91, 54.42),
        (-130.99, 54.47),
    ]),
    # Porcher Island, with its indented north and south shores
    poly([
        (-130.73, 54.28), (-130.58, 54.34), (-130.39, 54.30),
        (-130.24, 54.19), (-130.07, 54.15), (-129.96, 54.02),
        (-129.99, 53.87), (-130.12, 53.79), (-130.28, 53.80),
        (-130.40, 53.91), (-130.57, 53.94), (-130.70, 54.06),
    ]),
    poly([
        (-130.38, 54.30), (-130.30, 54.37), (-130.19, 54.34),
        (-130.17, 54.26), (-130.25, 54.22), (-130.35, 54.24),
    ]),
    # Goschen / Spiller and the low islands off Porcher
    poly([
        (-130.02, 53.91), (-129.92, 53.96), (-129.82, 53.92),
        (-129.79, 53.84), (-129.87, 53.79), (-129.98, 53.82),
    ]),
    poly([
        (-130.49, 53.79), (-130.42, 53.84), (-130.33, 53.81),
        (-130.32, 53.74), (-130.41, 53.71), (-130.49, 53.74),
    ]),
    # Banks Island: the long island in the southern part of the frame
    poly([
        (-130.78, 53.75), (-130.64, 53.84), (-130.43, 53.86),
        (-130.25, 53.78), (-130.10, 53.62), (-130.00, 53.46),
        (-129.92, 53.28), (-129.96, 53.12), (-130.09, 53.01),
        (-130.32, 53.00), (-130.47, 53.10), (-130.60, 53.28),
        (-130.74, 53.47),
    ]),
    # Small islands along the east side of Banks Island
    poly([
        (-129.98, 53.68), (-129.87, 53.73), (-129.77, 53.69),
        (-129.73, 53.59), (-129.80, 53.51), (-129.92, 53.54),
    ]),
    poly([
        (-130.16, 53.05), (-130.09, 53.12), (-130.00, 53.10),
        (-129.98, 53.02), (-130.05, 52.98), (-130.14, 52.99),
    ]),
]

# The larger islands nearer the mainland and the southern frame edge.
ISLANDS += [
    # Gil Island
    poly([
        (-129.94, 54.10), (-129.79, 54.17), (-129.61, 54.13),
        (-129.48, 54.02), (-129.42, 53.87), (-129.48, 53.69),
        (-129.59, 53.57), (-129.76, 53.54), (-129.89, 53.64),
        (-129.96, 53.82),
    ]),
    # Campania Island, mostly cut by the eastern edge of the window
    poly([
        (-129.47, 53.68), (-129.32, 53.77), (-129.16, 53.72),
        (-129.04, 53.58), (-128.98, 53.37), (-129.04, 53.16),
        (-129.17, 53.04), (-129.30, 53.00), (-129.39, 53.16),
        (-129.45, 53.38),
    ]),
    # Pitt Island, entering the frame from the south-west
    poly([
        (-130.91, 53.48), (-130.73, 53.54), (-130.53, 53.52),
        (-130.35, 53.42), (-130.18, 53.24), (-130.03, 53.06),
        (-130.00, 53.00), (-130.91, 53.00),
    ]),
    poly([
        (-130.31, 53.45), (-130.22, 53.51), (-130.10, 53.48),
        (-130.05, 53.40), (-130.11, 53.33), (-130.24, 53.35),
    ]),
    # A few named-looking skerries are retained because the requested
    # resolution is fine enough for several of them to occupy a pixel.
    poly([
        (-130.82, 54.02), (-130.78, 54.06), (-130.72, 54.04),
        (-130.70, 53.98), (-130.75, 53.94), (-130.81, 53.96),
    ]),
    poly([
        (-130.20, 54.48), (-130.15, 54.52), (-130.09, 54.49),
        (-130.08, 54.43), (-130.14, 54.40), (-130.20, 54.43),
    ]),
    poly([
        (-129.84, 54.48), (-129.79, 54.52), (-129.72, 54.49),
        (-129.71, 54.43), (-129.77, 54.40), (-129.84, 54.43),
    ]),
]


def is_land(lon, lat):
    land = inside(lon, lat, MAINLAND)
    if land:
        for water in MAINLAND_WATER:
            if inside(lon, lat, water):
                land = False
                break
    if not land:
        for island in ISLANDS:
            if inside(lon, lat, island):
                land = True
                break
    return int(land)

def build_matrix():
    out = np.zeros((N, N), dtype=np.uint8)
    for row in range(N):
        lat = NORTH - row * (NORTH - SOUTH) / (N - 1)
        for col in range(N):
            lon = WEST + col * (EAST - WEST) / (N - 1)
            out[row, col] = is_land(lon, lat)
    return out


def write_outputs(matrix):
    target = Path("outputs")
    target.mkdir(parents=True, exist_ok=True)
    with (target / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        for row in matrix:
            writer.writerow([int(value) for value in row])

    # Pillow's L mode is used only as an exact two-colour raster: 0 is black,
    # 255 is white.  No interpolation or post-processing is applied.
    raw = (matrix * 255).astype(np.uint8)
    image = Image.fromarray(raw, mode="L")
    image.save(target / "candidate.png")
    enlarged = image.resize((N * 8, N * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(target / "candidate_8x.png")


def validate(matrix):
    assert matrix.shape == (127, 127)
    assert matrix.dtype == np.uint8
    assert int(matrix.sum()) + int((matrix == 0).sum()) == 16129
    assert set(np.unique(matrix).tolist()) <= {0, 1}
    with Path("outputs/candidate.csv").open(encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 127 and all(len(r) == 127 for r in rows)
    assert all(v in {"0", "1"} for r in rows for v in r)
    assert Image.open("outputs/candidate.png").size == (127, 127)
    check = Image.open("outputs/candidate_8x.png")
    assert check.size == (1016, 1016)
    assert set(check.getdata()) <= {0, 255}


if __name__ == "__main__":
    matrix = build_matrix()
    write_outputs(matrix)
    validate(matrix)
    print("validated", matrix.shape, "land=", int(matrix.sum()),
          "water=", int((matrix == 0).sum()))

