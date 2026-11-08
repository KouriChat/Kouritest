"""Knowledge-based hand reconstruction of the Hecate Strait window.

The polygons below are deliberately an approximate sketch, not imported
cartography.  Coordinates are WGS84 degrees; x is longitude and y latitude.
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
    """Even/odd point-in-polygon test, including no external GIS package."""
    hit = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        crosses = ((yi > y) != (yj > y))
        if crosses and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            hit = not hit
        j = i
    return hit


def ellipse(cx, cy, rx, ry, count=32):
    import math
    return [
        (cx + rx * math.cos(2 * math.pi * k / count),
         cy + ry * math.sin(2 * math.pi * k / count))
        for k in range(count)
    ]


# Mainland coast, followed north-to-south.  Everything east of this line
# initially belongs to the mainland; the fjord polygons below cut it back.
mainland = [
    (-129.62, 55.00), (-129.73, 54.92), (-129.92, 54.84),
    (-129.98, 54.74), (-129.86, 54.64), (-129.74, 54.56),
    (-129.78, 54.46), (-129.86, 54.37), (-129.76, 54.27),
    (-129.80, 54.18), (-129.66, 54.08), (-129.55, 53.98),
    (-129.43, 53.88), (-129.30, 53.78), (-129.16, 53.66),
    (-129.04, 53.54), (-128.93, 53.43), (-128.83, 53.31),
    (-128.68, 53.00), (-129.00, 53.00), (-129.00, 55.00),
]

# Long tidal inlets which open on the west side of the mainland polygon.
fjords = [
    [(-129.98, 54.88), (-129.54, 54.88), (-129.42, 54.78),
     (-129.47, 54.66), (-129.64, 54.57), (-129.75, 54.51),
     (-129.83, 54.58), (-129.72, 54.69), (-129.72, 54.81)],
    [(-129.84, 54.39), (-129.48, 54.39), (-129.35, 54.30),
     (-129.40, 54.20), (-129.58, 54.15), (-129.73, 54.17),
     (-129.66, 54.27), (-129.72, 54.34)],
    [(-129.72, 54.10), (-129.33, 54.10), (-129.21, 54.02),
     (-129.27, 53.91), (-129.46, 53.86), (-129.57, 53.92),
     (-129.51, 54.00)],
    [(-129.37, 53.78), (-129.03, 53.78), (-128.88, 53.69),
     (-128.95, 53.59), (-129.14, 53.56), (-129.29, 53.62)],
]


def land_at(x, y):
    land = inside(x, y, mainland)
    if any(inside(x, y, p) for p in fjords):
        land = False
    return land

# Principal islands and island groups in the strait.  Small islands are
# represented by compact hand-drawn polygons so that they remain discrete at
# the requested sampling scale.
islands = [
    # eastern edge of Graham Island (most of that island is west of the box)
    [(-131.01, 54.22), (-130.94, 54.18), (-130.91, 54.08),
     (-130.96, 53.98), (-131.01, 53.96), (-131.01, 54.22)],
    # northern/eastern edge of Moresby Island
    [(-131.01, 53.48), (-130.94, 53.39), (-130.91, 53.24),
     (-130.96, 53.06), (-131.01, 53.02), (-131.01, 53.48)],
    # Dundas Island
    [(-131.00, 54.72), (-130.86, 54.77), (-130.68, 54.70),
     (-130.59, 54.59), (-130.66, 54.48), (-130.83, 54.45),
     (-130.98, 54.52), (-131.00, 54.72)],
    # Kaien Island (Prince Rupert)
    [(-130.51, 54.38), (-130.40, 54.43), (-130.25, 54.39),
     (-130.18, 54.29), (-130.23, 54.17), (-130.35, 54.12),
     (-130.49, 54.19), (-130.56, 54.29)],
    # Digby Island and the low islands west of Prince Rupert
    [(-130.72, 54.37), (-130.58, 54.40), (-130.48, 54.33),
     (-130.49, 54.22), (-130.60, 54.17), (-130.72, 54.23)],
    [(-130.70, 54.15), (-130.61, 54.18), (-130.53, 54.12),
     (-130.57, 54.04), (-130.69, 54.04), (-130.75, 54.10)],
    # Lelu and Ridley islands in the upper Skeena/Chatham Sound approach
    [(-130.29, 54.18), (-130.18, 54.23), (-130.08, 54.17),
     (-130.11, 54.05), (-130.23, 54.01), (-130.31, 54.08)],
    [(-130.10, 54.08), (-129.99, 54.13), (-129.91, 54.07),
     (-129.95, 53.98), (-130.07, 53.96), (-130.13, 54.02)],
    # Porcher Island
    [(-130.73, 54.01), (-130.53, 54.07), (-130.27, 54.04),
     (-130.03, 53.93), (-129.93, 53.77), (-130.04, 53.62),
     (-130.25, 53.56), (-130.49, 53.63), (-130.69, 53.78)],
    # Pitt Island and the islets around its northern end
    [(-130.25, 53.72), (-130.05, 53.75), (-129.84, 53.67),
     (-129.69, 53.53), (-129.65, 53.34), (-129.75, 53.18),
     (-129.96, 53.10), (-130.16, 53.19), (-130.29, 53.39)],
    [(-130.49, 53.66), (-130.38, 53.70), (-130.29, 53.64),
     (-130.32, 53.54), (-130.43, 53.51), (-130.51, 53.57)],
    # Smith Island, north of the Pitt group
    [(-130.72, 53.69), (-130.58, 53.76), (-130.43, 53.70),
     (-130.39, 53.57), (-130.50, 53.48), (-130.66, 53.52),
     (-130.75, 53.61)],
    # Banks Island's eastward projecting shore
    [(-129.04, 53.55), (-128.88, 53.61), (-128.72, 53.56),
     (-128.65, 53.41), (-128.76, 53.29), (-128.93, 53.31),
     (-129.03, 53.42)],
]

# A few recognizable small islands and reefs; their explicit inclusion is
# useful because the rule is point-based rather than area-based.
small_islands = [
    ellipse(-130.87, 54.34, .055, .045),
    ellipse(-130.79, 54.27, .045, .035),
    ellipse(-130.89, 54.10, .040, .030),
    ellipse(-130.40, 54.54, .050, .035),
    ellipse(-130.03, 54.38, .040, .025),
    ellipse(-129.79, 54.28, .035, .025),
    ellipse(-130.86, 53.96, .050, .035),
    ellipse(-130.11, 53.85, .045, .030),
    ellipse(-129.54, 53.72, .040, .025),
    ellipse(-129.37, 53.43, .035, .025),
    ellipse(-129.16, 53.20, .040, .025),
]


def land_at(x, y):
    land = inside(x, y, mainland)
    if any(inside(x, y, p) for p in fjords):
        land = False
    if any(inside(x, y, p) for p in islands):
        land = True
    if any(inside(x, y, p) for p in small_islands):
        land = True
    return land

def make_matrix():
    # The row/column construction is explicit so the first row is 55 N and
    # the first column is 131 W, exactly as requested.
    a = np.zeros((N, N), dtype=np.uint8)
    for r in range(N):
        lat = NORTH - r * (NORTH - SOUTH) / (N - 1)
        for c in range(N):
            lon = WEST + c * (EAST - WEST) / (N - 1)
            a[r, c] = 1 if land_at(lon, lat) else 0
    return a


def write_outputs(a):
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        for row in a:
            w.writerow([int(v) for v in row])

    # Pillow's L mode is an exact two-colour raster: 0 water, 255 land.
    raw = (a * 255).astype(np.uint8)
    image = Image.fromarray(raw, mode="L")
    image.save(out / "candidate.png")
    image.resize((N * 8, N * 8), resample=Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )


def validate(a):
    assert a.shape == (127, 127)
    assert int(a.size) == 16129
    assert set(np.unique(a).tolist()) <= {0, 1}
    with open("outputs/candidate.csv", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 127 and all(len(r) == 127 for r in rows)
    assert all(v in {"0", "1"} for r in rows for v in r)
    with Image.open("outputs/candidate.png") as im:
        assert im.size == (127, 127)
        assert set(im.convert("L").getdata()) <= {0, 255}
    with Image.open("outputs/candidate_8x.png") as im:
        assert im.size == (1016, 1016)
        assert set(im.convert("L").getdata()) <= {0, 255}


if __name__ == "__main__":
    matrix = make_matrix()
    write_outputs(matrix)
    validate(matrix)
    print("generated 127x127:", int(matrix.sum()), "land points")
