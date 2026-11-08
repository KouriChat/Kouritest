from pathlib import Path
import math
import csv
import numpy as np
from PIL import Image

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

def poly(x, y):
    return list(zip(x, y))

def inside(x, y, p):
    hit = False
    j = len(p) - 1
    for i in range(len(p)):
        xi, yi = p[i]
        xj, yj = p[j]
        if ((yi > y) != (yj > y)):
            at = (xj-xi) * (y-yi) / (yj-yi) + xi
            if x < at:
                hit = not hit
        j = i
    return hit

# Broad mainland outlines, written from remembered geography.
LAND = []
LAND.append(poly(
    [129.78,130.05,130.35,130.65,131.0,131.45,131.9,132.35,
     132.8,133.25,133.75,134.2,134.7,135.0,135.18,135.3,135.5,
     135.75,136.05,136.32,136.52,136.62,136.62,136.72,137.0,137.0,129.78],
    [34.55,34.58,34.48,34.25,34.12,34.16,34.22,34.28,
     34.36,34.43,34.48,34.55,34.64,34.65,34.5,34.18,33.87,
     33.65,33.48,33.46,33.62,33.98,34.3,34.55,34.68,35.0,35.0]))

# Kyushu, including its irregular north coast and the part visible at 33 N.
LAND.append(poly(
    [129.0,129.18,129.4,129.62,129.8,130.0,130.2,130.45,130.7,
     130.9,131.1,131.35,131.58,131.82,131.95,132.0,131.72,131.35,
     130.9,130.45,130.0,129.55,129.0],
    [33.72,33.82,33.9,34.08,34.28,34.42,34.43,34.35,34.22,
     34.02,33.83,33.66,33.45,33.25,33.0,33.0,33.0,33.0,
     33.0,33.0,33.0,33.0,33.0]))

# Shikoku.
LAND.append(poly(
    [131.92,132.15,132.38,132.65,132.9,133.2,133.55,133.9,
     134.2,134.48,134.7,134.62,134.42,134.18,133.85,133.5,
     133.15,132.82,132.48,132.18,131.92],
    [34.08,34.18,34.28,34.34,34.34,34.34,34.38,34.4,
     34.39,34.3,34.16,33.9,33.7,33.52,33.35,33.2,
     33.08,33.0,33.0,33.04,34.08]))

# The land on the west and east sides of Ise Bay, kept separate from the bay.
LAND.append(poly(
    [136.53,136.56,136.62,136.66,136.72,136.8,137.0,137.0,136.58],
    [33.48,33.72,34.0,34.25,34.48,34.66,34.75,35.0,35.0]))
LAND.append(poly(
    [136.93,137.0,137.0,136.98,136.92,136.84,136.8,136.78,
     136.84,136.93],
    [35.0,35.0,34.42,34.52,34.7,34.82,34.94,35.0,35.0,35.0]))

# A few peninsulas which are important at this resolution.
LAND += [
    poly([129.0,129.0,129.2,129.28,129.2,129.0],
         [34.25,34.65,34.62,34.45,34.3,34.25]),
    poly([134.48,134.62,134.82,134.98,135.1,135.0,134.8,134.58],
         [34.05,34.0,34.1,34.28,34.5,34.65,34.57,34.35]),
    poly([136.74,136.9,137.0,137.0,136.88,136.72],
         [34.34,34.42,34.45,34.62,34.66,34.52]),
]
def island(cx, cy, rx, ry, tilt=0.0, n=18):
    q = []
    co, si = math.cos(tilt), math.sin(tilt)
    for k in range(n):
        a = 2.0 * math.pi * k / n
        xx, yy = rx * math.cos(a), ry * math.sin(a)
        q.append((cx + co*xx - si*yy, cy + si*xx + co*yy))
    LAND.append(q)

# Islands at the mouth of the Inland Sea and around western Japan.
island(129.28, 34.42, .18, .34, -.2)
island(129.72, 33.76, .16, .20, -.25)       # Iki
island(129.55, 33.38, .13, .18, -.4)
island(129.42, 33.62, .10, .15, -.25)
island(130.28, 33.65, .12, .16, .2)
island(130.45, 33.45, .10, .14, -.2)
island(130.58, 33.2, .07, .12, -.3)

# Northern and central Seto Inland Sea.
island(131.72, 34.15, .13, .18, .2)
island(132.06, 34.22, .12, .17, -.2)
island(132.28, 34.28, .14, .22, .25)       # Itsukushima
island(132.42, 34.25, .20, .18, .4)        # Etajima
island(132.58, 34.22, .13, .16, -.2)
island(132.72, 34.16, .10, .13, .1)
island(132.82, 34.29, .10, .15, -.5)
island(132.94, 34.27, .14, .20, .3)       # Kurahashi / western Geiyo
island(133.03, 34.18, .15, .18, .1)
island(133.08, 34.31, .16, .22, -.3)      # Omishima
island(133.18, 34.28, .14, .20, .35)       # Innoshima
island(133.25, 34.39, .11, .16, -.2)      # Mukojima
island(133.36, 34.34, .08, .13, .2)
island(133.43, 34.16, .12, .18, -.2)
island(133.56, 34.28, .09, .15, .25)
island(133.68, 34.23, .13, .17, -.2)
island(133.79, 34.31, .08, .11, .2)
island(133.92, 34.40, .12, .16, -.3)
island(134.02, 34.44, .11, .14, .1)       # Naoshima
island(134.13, 34.49, .10, .13, -.3)
island(134.28, 34.53, .20, .12, .2)       # Shodoshima
island(134.42, 34.48, .09, .11, -.4)
island(134.56, 34.47, .07, .10, .1)

# Islands between Shikoku and Kyushu.
island(132.02, 33.55, .12, .10, -.2)
island(132.2, 33.52, .09, .11, .1)
island(132.52, 33.55, .10, .13, .2)
island(132.74, 33.62, .08, .13, -.3)
island(133.0, 33.65, .07, .12, .2)
island(133.3, 33.72, .08, .12, -.2)
# Larger islands and the compact island groups around Awaji and Ise.
LAND += [
    poly([134.66,134.72,134.8,134.92,135.02,135.06,135.0,134.9,
          134.78,134.68,134.62],
         [34.18,34.24,34.43,34.62,34.62,34.45,34.25,34.12,34.1,
          34.12,34.18]),
    poly([135.52,135.65,135.8,135.92,136.02,136.0,135.86,135.7],
         [34.08,34.05,34.0,34.1,34.28,34.4,34.33,34.2]),
    poly([136.78,136.9,137.0,137.0,136.91,136.8],
         [34.62,34.58,34.58,34.78,34.8,34.72]),
]
island(130.12, 33.63, .10, .16, -.3)
island(130.42, 33.84, .09, .12, .1)
island(131.15, 33.72, .08, .12, -.4)
island(131.5, 33.86, .07, .11, .2)
island(131.95, 34.4, .08, .12, -.2)
island(132.12, 34.42, .08, .12, .3)
island(132.54, 34.48, .07, .11, -.2)
island(132.78, 34.46, .07, .10, .2)
island(133.02, 34.48, .06, .10, -.2)
island(133.31, 34.48, .08, .10, .1)
island(133.58, 34.5, .06, .09, -.2)
island(133.84, 34.52, .06, .10, .2)
island(134.54, 34.62, .06, .10, -.3)
island(134.74, 34.63, .07, .11, .2)
island(135.08, 34.5, .07, .10, -.2)
island(135.34, 34.38, .08, .11, .2)
island(135.58, 34.38, .06, .09, -.2)
island(135.78, 34.33, .07, .10, .3)
island(136.35, 34.28, .06, .10, -.2)
island(136.62, 34.25, .08, .10, .1)
island(136.72, 34.62, .07, .10, -.2)

def land_at(x, y):
    for p in LAND:
        if inside(x, y, p):
            return 1
    return 0

rows = []
for iy in range(64):
    lat = 35.0 - (2.0 / 63.0) * iy
    row = []
    for ix in range(253):
        lon = 129.0 + (8.0 / 252.0) * ix
        row.append(land_at(lon, lat))
    rows.append(row)

assert len(rows) == 64 and all(len(r) == 253 for r in rows)
assert all(v in (0, 1) for r in rows for v in r)
with (OUT / "candidate.csv").open("w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerows(rows)

arr = np.asarray(rows, dtype=np.uint8)
Image.fromarray(arr * 255, mode="L").save(OUT / "candidate.png")
Image.fromarray(arr * 255, mode="L").resize(
    (2024, 512), resample=Image.Resampling.NEAREST
).save(OUT / "candidate_8x.png")
