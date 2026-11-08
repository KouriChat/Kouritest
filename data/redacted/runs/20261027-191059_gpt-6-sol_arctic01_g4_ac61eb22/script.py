"""Knowledge-only reconstruction of the Canadian Arctic archipelago.

The coastline is deliberately represented by hand-written, low-resolution
polygons and a few small-island ellipses.  No geographic data is read.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

NROW, NCOL = 61, 241
LATS = [79.0 - 0.1 * j for j in range(NROW)]
LONS = [-105.0 + 0.1 * i for i in range(NCOL)]

def inside_polygon(x, y, poly):
    """Ray crossing test; boundary points are treated as land."""
    hit = False
    n = len(poly)
    for k in range(n):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n]
        cross = (x2-x1)*(y-y1) - (y2-y1)*(x-x1)
        if abs(cross) < 1e-10 and min(x1,x2)-1e-10 <= x <= max(x1,x2)+1e-10 and min(y1,y2)-1e-10 <= y <= max(y1,y2)+1e-10:
            return True
        if (y1 > y) != (y2 > y):
            xin = x1 + (y-y1) * (x2-x1) / (y2-y1)
            if x <= xin:
                hit = not hit
    return hit

def in_ellipse(x, y, cx, cy, rx, ry):
    return ((x-cx)/rx)**2 + ((y-cy)/ry)**2 <= 1.0

def in_any(x, y, shapes):
    for shape in shapes:
        if shape[0] == "p" and inside_polygon(x, y, shape[1]):
            return True
        if shape[0] == "e" and in_ellipse(x, y, *shape[1]):
            return True
    return False

def P(*points):
    return ("p", list(points))

def E(cx, cy, rx, ry):
    return ("e", (cx, cy, rx, ry))
SHAPES = []

# High Arctic islands: Axel Heiberg, Ellesmere, and the southern edge of
# the Ellesmere group.  The frame clips their northern and eastern parts.
SHAPES += [
    P((-97.0,79.0),(-96.2,79.8),(-93.8,80.4),(-89.5,80.3),
      (-87.0,79.5),(-86.2,78.5),(-87.2,77.4),(-89.0,76.7),
      (-91.5,76.1),(-94.8,76.0),(-96.5,76.8),(-97.2,78.0)),
    P((-89.5,79.0),(-88.2,80.0),(-84.7,80.7),(-80.8,80.1),
      (-78.0,79.2),(-77.0,78.1),(-78.1,77.0),(-79.7,76.1),
      (-81.5,75.1),(-84.0,74.9),(-86.0,75.7),(-87.7,77.0),
      (-89.0,78.0)),
]

# Devon Island and the substantial islands immediately to its southwest.
SHAPES += [
    P((-91.7,76.0),(-90.2,76.8),(-86.0,77.0),(-82.8,76.5),
      (-81.0,75.8),(-81.3,75.0),(-83.3,74.5),(-87.1,74.3),
      (-90.0,74.7),(-91.8,75.2)),
    P((-99.8,76.0),(-97.5,76.5),(-94.8,76.3),(-93.5,75.4),
      (-94.4,74.8),(-97.0,74.5),(-99.8,75.0)),
    P((-103.0,75.1),(-101.5,75.8),(-99.8,75.7),(-98.8,74.9),
      (-99.6,74.2),(-102.0,74.1),(-103.0,74.6)),
]

# Prince of Wales Island: broad, irregular island between Peel Sound and
# Franklin Strait.
SHAPES += [
    P((-101.5,75.4),(-99.6,76.0),(-96.5,76.2),(-93.5,75.6),
      (-91.3,74.8),(-90.4,73.8),(-90.8,72.6),(-93.0,71.9),
      (-96.2,72.1),(-99.1,72.7),(-101.0,73.7)),
]

# Somerset Island, Boothia's northwestern peninsula, and the mainland
# entering through the southern edge of the requested rectangle.
SHAPES += [
    P((-95.2,74.6),(-93.0,75.0),(-90.4,74.5),(-88.2,73.8),
      (-87.0,72.9),(-88.0,72.1),(-91.0,71.7),(-94.3,72.1),
      (-96.0,73.0)),
    P((-96.5,73.0),(-95.0,74.0),(-93.0,74.2),(-90.8,73.7),
      (-89.5,72.8),(-89.0,71.9),(-91.5,71.0),(-95.5,71.5),
      (-97.0,72.4)),
    P((-98.0,73.0),(-96.0,73.3),(-93.5,73.0),(-90.0,72.4),
      (-87.0,71.5),(-84.0,71.0),(-81.0,71.2),(-79.5,72.2),
      (-79.2,73.0),(-82.5,73.2),(-86.0,73.5),(-90.5,73.6),
      (-94.0,73.7),(-97.0,73.5)),
]
# Small islands, exposed rocks, and narrow island groups visible at this
# scale.  Ellipses are used only for features below a few tenths of a degree.
SHAPES += [
    # east end of Melville Island at the western frame edge
    P((-105.0,74.9),(-104.1,75.2),(-102.8,75.0),(-102.1,74.3),
      (-103.1,73.8),(-105.0,73.9)),
    # northwestern and central channel islands
    E(-101.6,75.0,0.35,0.28),
    E(-98.4,75.1,0.38,0.25),
    E(-96.1,75.6,0.45,0.32),
    E(-94.1,74.7,0.28,0.22),
    E(-92.0,74.7,0.22,0.17),
    E(-90.7,74.9,0.22,0.16),
    # Beechey and nearby small islands
    E(-90.6,74.72,0.18,0.15),
    E(-91.4,74.35,0.16,0.12),
    E(-88.8,75.1,0.20,0.14),
    E(-86.6,75.4,0.18,0.13),
    # islands in the sounds west of Ellesmere
    E(-94.8,77.2,0.26,0.18),
    E(-92.8,77.0,0.18,0.14),
    E(-89.8,76.6,0.20,0.14),
    E(-85.4,76.4,0.22,0.16),
    E(-81.8,76.5,0.18,0.14),
    # islands near Somerset, Peel Sound, and Franklin Strait
    E(-96.0,74.0,0.24,0.16),
    E(-94.7,73.6,0.19,0.13),
    E(-92.5,73.5,0.16,0.12),
    E(-88.1,73.4,0.20,0.13),
    E(-85.1,73.5,0.18,0.12),
    # southwestern Baffin Island / its coastal tip at the southeast edge
    P((-82.5,73.0),(-81.0,73.2),(-81.0,73.9),(-82.0,74.1),
      (-83.0,73.7)),
]

matrix = np.zeros((NROW, NCOL), dtype=np.uint8)
for r, lat in enumerate(LATS):
    for c, lon in enumerate(LONS):
        matrix[r, c] = 1 if in_any(lon, lat, SHAPES) else 0

assert matrix.shape == (61, 241)
assert int(matrix.size) == 14701
assert set(np.unique(matrix).tolist()) <= {0, 1}

with (OUT / "candidate.csv").open("w", newline="", encoding="ascii") as f:
    writer = csv.writer(f, lineterminator="\n")
    for row in matrix.tolist():
        writer.writerow([int(v) for v in row])

raw = (matrix * 255).astype(np.uint8)
im = Image.fromarray(raw, mode="L")
im.save(OUT / "candidate.png")
large = im.resize((NCOL * 8, NROW * 8), Image.Resampling.NEAREST)
large.save(OUT / "candidate_8x.png")

assert im.size == (241, 61)
assert large.size == (1928, 488)
assert set(im.getdata()) <= {0, 255}
assert set(large.getdata()) <= {0, 255}
print("generated", matrix.shape, "land points", int(matrix.sum()),
      "water points", int(matrix.size - matrix.sum()))
