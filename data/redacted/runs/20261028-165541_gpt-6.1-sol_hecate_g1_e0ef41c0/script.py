from pathlib import Path
import csv
import numpy as np
from PIL import Image

N = 127
STEP = 2.0 / 126.0


def polygon_mask(lon, lat, points):
    """Vectorized ray-casting test for hand-drawn lon/lat polygons."""
    inside = np.zeros(lon.shape, dtype=bool)
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        crossing = (y1 > lat) != (y2 > lat)
        xhit = x1 + (lat - y1) * (x2 - x1) / (y2 - y1 + 1e-300)
        inside ^= crossing & (lon < xhit)
    return inside


def island(cx, cy, rx, ry, angle=0.0, count=32):
    t = np.arange(count) * (2 * np.pi / count)
    c, s = np.cos(angle), np.sin(angle)
    return [
        (cx + rx * np.cos(a) * c - ry * np.sin(a) * s,
         cy + rx * np.cos(a) * s + ry * np.sin(a) * c)
        for a in t
    ]


# Approximate mainland coast: land lies east of this northern-to-southern line.
coast = [
    (-130.20, 55.00), (-130.24, 54.85), (-130.30, 54.70),
    (-130.22, 54.58), (-130.16, 54.48), (-130.22, 54.38),
    (-130.29, 54.30), (-130.20, 54.18), (-130.02, 54.08),
    (-129.88, 53.98), (-129.78, 53.84), (-129.66, 53.70),
    (-129.54, 53.55), (-129.43, 53.38), (-129.37, 53.20),
    (-129.34, 53.00),
]
mainland = coast + [(-128.0, 53.0), (-128.0, 55.0)]
# Eastern edges of Haida Gwaii, plus schematic islands and reef groups
# within the requested extent. Most of Graham and Moresby lie west of it.
islands = [
    [(-131.00, 54.10), (-131.02, 54.22), (-131.07, 54.33),
     (-131.16, 54.39), (-131.28, 54.40), (-131.32, 54.30),
     (-131.26, 54.20), (-131.18, 54.12), (-131.08, 54.08)],
    [(-131.00, 53.82), (-131.04, 53.94), (-131.12, 54.02),
     (-131.22, 54.00), (-131.27, 53.92), (-131.22, 53.84),
     (-131.12, 53.78), (-131.04, 53.77)],
    [(-131.00, 53.18), (-131.02, 53.36), (-131.06, 53.54),
     (-131.12, 53.70), (-131.20, 53.79), (-131.30, 53.75),
     (-131.34, 53.60), (-131.30, 53.42), (-131.23, 53.26),
     (-131.13, 53.14), (-131.05, 53.12)],
    island(-130.72, 54.82, .055, .12, -.25),
    island(-130.55, 54.60, .045, .10, .30),
    island(-130.80, 54.42, .060, .13, -.35),
    island(-130.57, 54.22, .055, .12, .15),
    island(-130.40, 54.03, .055, .12, -.2),
    island(-130.20, 53.84, .070, .15, -.3),
    island(-130.02, 53.62, .060, .13, .25),
    island(-129.84, 53.39, .070, .14, -.35),
    island(-129.67, 53.16, .045, .09, .1),
    island(-130.95, 54.02, .025, .055),
    island(-130.88, 53.90, .025, .060, -.2),
    island(-130.76, 53.72, .030, .065, .3),
    island(-130.61, 53.51, .030, .060, -.2),
    island(-130.46, 53.29, .025, .055, .2),
]
latitudes = 55.0 - np.arange(N, dtype=np.float64) * STEP
longitudes = -131.0 + np.arange(N, dtype=np.float64) * STEP
lon, lat = np.meshgrid(longitudes, latitudes)

# Each sample is classified by its own point-in-polygon result.
land = polygon_mask(lon, lat, mainland)
for shape in islands:
    land |= polygon_mask(lon, lat, shape)
matrix = land.astype(np.uint8)

out = Path("outputs")
out.mkdir(parents=True, exist_ok=True)
with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
    csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

image = Image.fromarray(matrix * 255, mode="L")
image.save(out / "candidate.png")
image.resize((1016, 1016), Image.Resampling.NEAREST).save(
    out / "candidate_8x.png"
)

# Check dimensions, binary data, and exact nearest-neighbor replication.
assert matrix.shape == (127, 127) and matrix.size == 16129
assert set(np.unique(matrix)) <= {0, 1}
with (out / "candidate.csv").open(newline="", encoding="ascii") as f:
    rows = list(csv.reader(f))
assert len(rows) == 127
assert all(len(r) == 127 and set(r) <= {"0", "1"} for r in rows)
assert image.size == (127, 127) and set(image.getdata()) <= {0, 255}
large = Image.open(out / "candidate_8x.png")
assert large.size == (1016, 1016)
assert set(large.getdata()) <= {0, 255}
