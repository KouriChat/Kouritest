import os
import csv
import math
import numpy as np
from PIL import Image

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

NROWS, NCOLS = 64, 253
LAT0, LON0 = 35.0, 129.0
STEP = 2.0 / 63.0

polys = []
holes = []

def poly(points, target=polys):
    target.append([(float(x), float(y)) for x, y in points])

def ellipse(cx, cy, rx, ry, n=20, target=polys):
    target.append([
        (cx + rx * math.cos(2 * math.pi * k / n),
         cy + ry * math.sin(2 * math.pi * k / n))
        for k in range(n)
    ])

def inside(x, y, p):
    # Standard ray crossing test; boundary is treated as land.
    hit = False
    j = len(p) - 1
    for i, (xi, yi) in enumerate(p):
        xj, yj = p[j]
        cross = ((yi > y) != (yj > y))
        if cross:
            xx = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x <= xx:
                hit = not hit
        if ((yi - y) * (xj - x) - (xi - x) * (yj - y)) == 0:
            if min(xi, xj) <= x <= max(xi, xj) and min(yi, yj) <= y <= max(yi, yj):
                return True
        j = i
    return hit

def is_land(lon, lat):
    value = any(inside(lon, lat, p) for p in polys)
    if value and any(inside(lon, lat, p) for p in holes):
        value = False
    return value
# Main land masses.  These are deliberately hand-drawn coastal traversals,
# rather than imported outlines.

# Southern Chugoku, Kinki, and the central Honshu shore.
poly([
    (130.65, 35.0), (137.0, 35.0), (137.0, 34.05),
    (136.75, 34.05), (136.55, 34.20), (136.30, 34.35),
    (136.00, 34.45), (135.70, 34.55), (135.45, 34.62),
    (135.20, 34.66), (134.80, 34.72), (134.40, 34.70),
    (134.00, 34.65), (133.60, 34.60), (133.20, 34.52),
    (132.80, 34.42), (132.50, 34.38), (132.20, 34.25),
    (132.00, 34.10), (131.70, 33.98), (131.30, 33.92),
    (130.90, 33.96), (130.70, 34.10), (130.65, 35.0)
])

# Northern and eastern Kyushu within the frame.
poly([
    (129.00, 33.00), (129.60, 33.00), (130.00, 32.96),
    (130.50, 32.88), (131.00, 32.84), (131.50, 32.88),
    (132.05, 33.08), (132.15, 33.38), (131.85, 33.68),
    (131.35, 33.84), (130.90, 34.00), (130.50, 34.00),
    (130.05, 33.86), (129.72, 33.66), (129.50, 33.42),
    (129.05, 33.32), (129.00, 33.00)
])

# Shikoku, with the Seto Inland Sea on its north side.
poly([
    (131.80, 33.65), (132.00, 33.90), (132.35, 34.00),
    (132.80, 34.08), (133.20, 34.12), (133.60, 34.22),
    (134.00, 34.32), (134.35, 34.34), (134.65, 34.25),
    (134.75, 34.10), (134.65, 33.85), (134.50, 33.60),
    (134.25, 33.35), (133.90, 33.15), (133.50, 33.10),
    (133.10, 33.15), (132.70, 33.20), (132.30, 33.35),
    (132.00, 33.50)
])

# Awaji Island and the larger islands at the eastern end of the Inland Sea.
poly([
    (134.65, 34.65), (134.95, 34.55), (135.00, 34.25),
    (134.82, 34.05), (134.62, 34.18), (134.58, 34.42)
])
poly([
    (133.55, 34.38), (133.90, 34.43), (134.10, 34.32),
    (134.02, 34.12), (133.70, 34.10), (133.52, 34.22)
])

# The Ise Bay water body is a conspicuous indentation in the Honshu outline.
poly([
    (136.48, 34.50), (136.70, 34.55), (136.86, 34.72),
    (136.95, 34.92), (137.00, 35.00), (136.76, 35.00),
    (136.68, 34.84), (136.55, 34.68)
], holes)
# Hand-placed islands and island groups (centres and radii are approximate).

# Western Japan and the approaches to the Inland Sea.
ellipse(129.28, 33.36, .13, .18)       # Hirado
ellipse(129.62, 33.82, .18, .20)       # Iki
ellipse(129.16, 33.08, .09, .16)
ellipse(129.42, 33.10, .07, .12)
ellipse(130.22, 33.72, .07, .09)
ellipse(130.62, 33.78, .06, .10)
ellipse(131.68, 33.72, .08, .13)       # Himeshima
ellipse(131.96, 33.82, .10, .08)
ellipse(132.24, 33.90, .17, .10)       # Suo-Oshima
ellipse(132.42, 33.96, .06, .07)

# Hiroshima Bay and the Geiyo island chain.
ellipse(132.33, 34.27, .09, .12)       # Miyajima
ellipse(132.47, 34.26, .13, .10)       # Etajima
ellipse(132.62, 34.18, .07, .08)
ellipse(132.78, 34.22, .08, .07)
ellipse(132.92, 34.30, .10, .10)
ellipse(133.08, 34.30, .14, .12)       # Innoshima / islands
ellipse(133.16, 34.23, .10, .12)
ellipse(133.02, 34.16, .10, .08)
ellipse(133.22, 34.15, .12, .08)
ellipse(133.36, 34.22, .07, .08)
ellipse(133.42, 34.33, .08, .07)
ellipse(133.55, 34.28, .10, .08)
ellipse(133.70, 34.30, .08, .06)
ellipse(133.86, 34.36, .06, .06)

# Kasaoka islands, the Bisan set, and the Kagawa side.
ellipse(133.50, 34.43, .08, .06)
ellipse(133.66, 34.48, .09, .07)
ellipse(133.82, 34.47, .09, .08)
ellipse(133.98, 34.46, .11, .08)       # Naoshima
ellipse(134.08, 34.52, .07, .06)       # Teshima
ellipse(134.22, 34.51, .18, .13)       # Shodoshima
ellipse(134.42, 34.48, .07, .07)
ellipse(134.57, 34.43, .06, .06)
ellipse(134.72, 34.36, .06, .06)
ellipse(134.70, 34.08, .07, .07)

# Small but map-visible pieces around Osaka Bay and the Kii coast.
ellipse(135.08, 34.38, .06, .05)
ellipse(135.28, 34.25, .07, .05)
ellipse(135.60, 34.30, .07, .06)
ellipse(135.92, 34.30, .08, .06)
ellipse(136.16, 34.18, .08, .06)
ellipse(136.68, 34.30, .06, .08)
# A few further island groups, including isolated rocks represented at this
# sampling scale by small ellipses.
ellipse(132.52, 34.12, .08, .06)
ellipse(132.55, 34.05, .06, .05)
ellipse(132.70, 34.04, .06, .05)
ellipse(132.86, 34.10, .06, .05)
ellipse(133.00, 34.05, .06, .05)
ellipse(133.30, 34.06, .07, .05)
ellipse(133.52, 34.10, .08, .06)       # Ibukijima vicinity
ellipse(133.78, 34.18, .08, .06)
ellipse(133.78, 34.34, .07, .07)
ellipse(133.95, 34.40, .06, .05)
ellipse(134.06, 34.40, .06, .06)       # Megijima / Ogijima vicinity
ellipse(134.33, 34.27, .06, .05)
ellipse(134.52, 34.30, .07, .06)
ellipse(134.55, 34.58, .09, .07)       # Ie-jima
ellipse(134.85, 34.28, .06, .05)
ellipse(134.96, 34.22, .06, .05)

# Some islands in the western mouth and along the Kyushu side.
ellipse(130.80, 33.78, .06, .08)
ellipse(131.05, 33.70, .07, .07)
ellipse(131.36, 33.78, .06, .06)
ellipse(131.68, 33.88, .06, .06)
ellipse(131.88, 33.70, .07, .07)
ellipse(132.08, 33.74, .06, .05)

rows = []
for r in range(NROWS):
    lat = LAT0 - r * STEP
    row = []
    for c in range(NCOLS):
        lon = LON0 + c * STEP
        row.append(1 if is_land(lon, lat) else 0)
    rows.append(row)

assert len(rows) == 64 and all(len(row) == 253 for row in rows)
assert all(v in (0, 1) for row in rows for v in row)

with open(os.path.join(OUT, "candidate.csv"), "w", newline="", encoding="ascii") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(rows)

arr = (np.asarray(rows, dtype=np.uint8) * 255)
raw = Image.fromarray(arr, mode="L")
raw.save(os.path.join(OUT, "candidate.png"))
large = raw.resize((NCOLS * 8, NROWS * 8), Image.Resampling.NEAREST)
large.save(os.path.join(OUT, "candidate_8x.png"))

assert raw.size == (253, 64)
assert large.size == (2024, 512)
assert set(raw.getdata()).issubset({0, 255})
print("generated", len(rows), "x", len(rows[0]), "matrix; land points:",
      int(arr.sum() // 255))
