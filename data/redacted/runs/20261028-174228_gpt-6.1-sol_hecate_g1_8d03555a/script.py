from pathlib import Path
import csv
import numpy as np
from PIL import Image

# Hand-drawn WGS84 polygon approximations. Coordinates are (longitude, latitude).
WEST, EAST = -131.0, -129.0
NORTH, SOUTH = 55.0, 53.0
STEP = 2.0 / 126.0

def inside(lon, lat, ring):
    """Ray-crossing point-in-polygon test."""
    result = False
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat):
            cross = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < cross:
                result = not result
        j = i
    return result

# Mainland shore, with the outer coast sketched from north to south.
# The east side of the polygon intentionally extends beyond the sampled frame.
MAINLAND = [
    (-129.62, 55.10), (-129.66, 54.98), (-129.72, 54.88),
    (-129.78, 54.79), (-129.83, 54.69), (-129.87, 54.58),
    (-129.91, 54.48), (-129.89, 54.37), (-129.83, 54.28),
    (-129.79, 54.19), (-129.81, 54.10), (-129.76, 54.01),
    (-129.73, 53.92), (-129.68, 53.84), (-129.64, 53.75),
    (-129.61, 53.66), (-129.57, 53.57), (-129.53, 53.48),
    (-129.49, 53.39), (-129.45, 53.30), (-129.40, 53.20),
    (-129.35, 53.10), (-129.28, 53.00), (-128.80, 53.00),
    (-128.80, 55.10)
]
# Haida Gwaii eastern margin: Graham Island north, Moresby Island south.
GRAHAM = [
    (-131.02, 54.00), (-130.91, 54.03), (-130.79, 54.01),
    (-130.68, 53.96), (-130.58, 53.90), (-130.48, 53.84),
    (-130.42, 53.78), (-130.42, 53.72), (-130.48, 53.68),
    (-130.56, 53.66), (-130.67, 53.69), (-130.77, 53.73),
    (-130.87, 53.80), (-130.96, 53.88), (-131.02, 53.94)
]
MORESBY = [
    (-131.02, 53.62), (-130.91, 53.61), (-130.82, 53.57),
    (-130.74, 53.52), (-130.68, 53.45), (-130.62, 53.37),
    (-130.59, 53.29), (-130.61, 53.21), (-130.67, 53.15),
    (-130.75, 53.12), (-130.83, 53.17), (-130.90, 53.25),
    (-130.96, 53.35), (-131.02, 53.46)
]
# Larger islands along the mainland coast.
ISLANDS = [
    [(-130.34,54.57),(-130.25,54.61),(-130.15,54.58),(-130.12,54.52),(-130.18,54.48),(-130.29,54.49)],
    [(-130.42,54.35),(-130.34,54.39),(-130.25,54.36),(-130.23,54.30),(-130.30,54.27),(-130.39,54.29)],
    [(-130.36,54.08),(-130.28,54.12),(-130.19,54.08),(-130.18,54.02),(-130.25,53.99),(-130.34,54.01)],
    [(-130.48,53.91),(-130.41,53.95),(-130.33,53.92),(-130.32,53.87),(-130.39,53.84),(-130.47,53.86)],
    [(-130.35,53.53),(-130.27,53.56),(-130.20,53.52),(-130.20,53.47),(-130.28,53.45),(-130.34,53.48)],
    [(-130.18,53.32),(-130.10,53.35),(-130.02,53.32),(-130.02,53.27),(-130.09,53.25),(-130.16,53.27)],
]
def on_land(lon, lat):
    if inside(lon, lat, MAINLAND):
        return 1
    if inside(lon, lat, GRAHAM) or inside(lon, lat, MORESBY):
        return 1
    return int(any(inside(lon, lat, ring) for ring in ISLANDS))

matrix = np.zeros((127, 127), dtype=np.uint8)
for row in range(127):
    lat = NORTH - row * STEP
    for col in range(127):
        lon = WEST + col * STEP
        matrix[row, col] = on_land(lon, lat)

out = Path("outputs")
out.mkdir(parents=True, exist_ok=True)
with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(matrix.tolist())

# Grayscale 0 is black water; 255 is white land.
image = Image.fromarray((matrix * 255).astype(np.uint8), mode="L")
image.save(out / "candidate.png")
image.resize((1016, 1016), Image.Resampling.NEAREST).save(out / "candidate_8x.png")

assert matrix.shape == (127, 127) and matrix.size == 16129
assert set(np.unique(matrix).tolist()) <= {0, 1}
with (out / "candidate.csv").open(newline="", encoding="ascii") as f:
    rows = list(csv.reader(f))
assert len(rows) == 127 and all(len(row) == 127 for row in rows)
assert all(v in ("0", "1") for row in rows for v in row)
assert image.size == (127, 127) and set(image.getdata()) <= {0, 255}
large = Image.open(out / "candidate_8x.png")
assert large.size == (1016, 1016)
assert large.getpixel((8 * 10, 8 * 10)) == image.getpixel((10, 10))
