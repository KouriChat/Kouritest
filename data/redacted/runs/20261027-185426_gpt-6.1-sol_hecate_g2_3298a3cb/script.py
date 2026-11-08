"""Knowledge-based hand-drawn geometry for the Hecate Strait window."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

N = 127
WEST, EAST = -131.0, -129.0
NORTH, SOUTH = 55.0, 53.0

def inside(x, y, poly):
    hit = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        cross = ((yi > y) != (yj > y))
        if cross and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            hit = not hit
        j = i
    return hit

def any_shape(x, y, shapes):
    return any(inside(x, y, p) for p in shapes)

# Mainland coast, followed clockwise around the part visible in the window.
land = [
    [(-129.0,55.0),(-129.0,53.0),(-129.30,53.04),(-129.44,53.20),
     (-129.50,53.42),(-129.63,53.63),(-129.76,53.82),(-129.94,54.00),
     (-130.08,54.18),(-130.16,54.36),(-130.07,54.53),(-129.88,54.70),
     (-129.77,54.88),(-129.73,55.0)],

    # Northern mainland peninsula and the intricate shore around Portland Inlet.
    [(-129.0,55.0),(-129.0,54.73),(-129.26,54.75),(-129.42,54.86),
     (-129.55,55.0)],

    # Mainland-side pieces around Prince Rupert and Chatham Sound.
    [(-129.0,54.78),(-129.0,54.37),(-129.24,54.39),(-129.39,54.51),
     (-129.43,54.66),(-129.29,54.77)],

    # Lower mainland shoreline and the heads of several fjords.
    [(-129.0,53.62),(-129.0,53.0),(-129.30,53.04),(-129.44,53.20),
     (-129.50,53.42),(-129.63,53.62)],
]

# Small mainland-side water cuts.  These deliberately override land: they
# represent bays and narrow tidal channels, not holes inferred from a grid.
water_cuts = [
    [(-130.0,55.01),(-129.70,55.01),(-129.73,54.87),(-129.92,54.75),
     (-130.10,54.67),(-130.20,54.72),(-130.15,54.87)],
    [(-129.96,54.68),(-129.76,54.74),(-129.57,54.67),(-129.63,54.54),
     (-129.82,54.48),(-130.01,54.53)],
    [(-129.94,54.40),(-129.72,54.45),(-129.58,54.37),(-129.65,54.22),
     (-129.86,54.17),(-130.04,54.25)],
    [(-129.82,54.08),(-129.62,54.16),(-129.52,54.08),(-129.57,53.94),
     (-129.75,53.88),(-129.91,53.94)],
    [(-129.68,53.76),(-129.48,53.82),(-129.37,53.72),(-129.45,53.58),
     (-129.62,53.52),(-129.78,53.60)],
    [(-129.54,53.39),(-129.34,53.47),(-129.22,53.36),(-129.30,53.20),
     (-129.46,53.18),(-129.61,53.27)],
]
# Haida Gwaii edges visible at the western edge of this particular window.
land += [
    [(-131.01,54.98),(-130.91,54.91),(-130.86,54.78),(-130.91,54.66),
     (-130.82,54.53),(-130.78,54.39),(-130.88,54.25),(-131.01,54.17)],
    [(-131.01,53.38),(-130.89,53.30),(-130.82,53.16),(-130.83,53.0),
     (-131.01,53.0)],
]

# Dundas Island group and the islands in Dixon Entrance.
land += [
    [(-130.98,54.78),(-130.83,54.80),(-130.68,54.72),(-130.64,54.57),
     (-130.75,54.47),(-130.91,54.49),(-131.01,54.60)],
    [(-130.78,54.88),(-130.67,54.86),(-130.61,54.78),(-130.68,54.70),
     (-130.79,54.73)],
    [(-130.57,54.70),(-130.47,54.65),(-130.45,54.57),(-130.55,54.52),
     (-130.64,54.58)],
]

# Porcher Island and its long, indented north/east shore.
land += [
    [(-130.92,54.30),(-130.72,54.36),(-130.52,54.31),(-130.38,54.22),
     (-130.17,54.21),(-130.08,54.10),(-130.17,53.98),(-130.35,53.96),
     (-130.49,53.88),(-130.66,53.91),(-130.79,54.03),(-130.88,54.16)],
    [(-130.42,54.37),(-130.29,54.38),(-130.19,54.31),(-130.24,54.24),
     (-130.38,54.24)],
    [(-130.82,53.98),(-130.72,53.94),(-130.69,53.84),(-130.78,53.80),
     (-130.88,53.87)],
]

# Banks Island: a broad, low island at the south end of Hecate Strait.
land += [
    [(-130.14,54.42),(-129.91,54.46),(-129.67,54.40),(-129.46,54.28),
     (-129.33,54.09),(-129.34,53.91),(-129.48,53.76),(-129.70,53.69),
     (-129.93,53.76),(-130.08,53.91),(-130.18,54.10),(-130.24,54.29)],
    [(-129.41,54.31),(-129.28,54.26),(-129.21,54.14),(-129.28,54.06),
     (-129.40,54.10)],
]

# Pitt Island and the larger islands immediately south of it.
land += [
    [(-130.56,54.04),(-130.35,54.13),(-130.11,54.08),(-129.91,53.94),
     (-129.76,53.76),(-129.73,53.55),(-129.82,53.37),(-129.99,53.22),
     (-130.20,53.18),(-130.40,53.27),(-130.53,53.46),(-130.61,53.72)],
    [(-129.78,53.28),(-129.60,53.34),(-129.43,53.27),(-129.39,53.12),
     (-129.53,53.03),(-129.72,53.05)],
    [(-130.05,53.10),(-129.87,53.16),(-129.75,53.08),(-129.81,53.0),
     (-130.00,53.0)],
]

# A few named or unnamed islets are large enough to survive this sampling.
land += [
    [(-130.98,54.11),(-130.92,54.15),(-130.86,54.12),(-130.88,54.06),
     (-130.95,54.04)],
    [(-130.64,54.66),(-130.58,54.68),(-130.54,54.64),(-130.58,54.59),
     (-130.64,54.60)],
    [(-130.23,53.70),(-130.17,53.74),(-130.11,53.70),(-130.14,53.64),
     (-130.21,53.64)],
    [(-129.20,53.82),(-129.13,53.87),(-129.08,53.82),(-129.11,53.76),
     (-129.18,53.76)],
]
def classify(lon, lat):
    # The polygon model is evaluated at the point itself; no cell averaging
    # or neighborhood vote is used.
    value = 1 if any_shape(lon, lat, land) else 0
    if any_shape(lon, lat, water_cuts):
        value = 0
    return value

def build_matrix():
    matrix = np.zeros((N, N), dtype=np.uint8)
    for row in range(N):
        lat = NORTH - (NORTH - SOUTH) * row / (N - 1)
        for col in range(N):
            lon = WEST + (EAST - WEST) * col / (N - 1)
            matrix[row, col] = classify(lon, lat)
    return matrix

def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()

    assert matrix.shape == (127, 127)
    assert matrix.size == 16129
    assert set(np.unique(matrix)).issubset({0, 1})

    with (out / "candidate.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        for row in matrix:
            writer.writerow([int(v) for v in row])

    # Pillow receives 255 for land and 0 for water.
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(out / "candidate.png")
    enlarged = image.resize((N * 8, N * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(out / "candidate_8x.png")

    # Basic post-write checks make accidental format changes fail loudly.
    with (out / "candidate.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 127 and all(len(r) == 127 for r in rows)
    assert all(v in {"0", "1"} for r in rows for v in r)
    assert image.size == (127, 127)
    assert enlarged.size == (1016, 1016)
    assert set(image.getdata()).issubset({0, 255})
    assert set(enlarged.getdata()).issubset({0, 255})

if __name__ == "__main__":
    main()
