"""Knowledge-only sketch of the Canadian Arctic Archipelago, 73--79 N.

All vertices below were entered by hand from geographic recollection.  No
map, gazetteer, coastline package, or external file is read by this program.
Coordinates in the lists are (longitude, latitude).
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image


def poly(*p):
    return ("poly", p)


def ellipse(cx, cy, rx, ry, angle=0.0):
    return ("ellipse", (cx, cy, rx, ry, angle))


def inside(shape, x, y):
    kind, q = shape
    if kind == "ellipse":
        cx, cy, rx, ry, a = q
        a = np.deg2rad(a)
        u = (x-cx)*np.cos(a) + (y-cy)*np.sin(a)
        v = -(x-cx)*np.sin(a) + (y-cy)*np.cos(a)
        return (u/rx)**2 + (v/ry)**2 <= 1.0
    hit = np.zeros(np.broadcast(x, y).shape, dtype=bool)
    xx, yy = np.broadcast_arrays(x, y)
    x1, y1 = q[-1]
    for x2, y2 in q:
        cross = ((y1 > yy) != (y2 > yy))
        xin = (x2-x1)*(yy-y1)/(y2-y1 + 1e-30) + x1
        hit ^= cross & (xx <= xin)
        x1, y1 = x2, y2
    return hit


LAND = []
CUT = []

# Prince of Wales Island, whose much larger southern part is below the frame.
LAND += [poly(
    (-105.0, 73.34), (-104.35, 73.55), (-103.55, 73.77),
    (-102.45, 73.91), (-101.35, 73.86), (-100.45, 73.70),
    (-99.45, 73.72), (-98.35, 73.58), (-97.48, 73.32),
    (-96.92, 73.00), (-105.0, 73.00))]

# Russell Island and the small Prescott islands in Peel Sound.
LAND += [poly(
    (-99.62, 73.84), (-99.25, 74.16), (-98.66, 74.34),
    (-97.94, 74.29), (-97.48, 74.03), (-97.62, 73.78),
    (-98.30, 73.65), (-99.10, 73.68))]
LAND += [ellipse(-96.78, 73.13, .34, .17, -18),
         ellipse(-96.42, 73.37, .17, .10, -20)]

# Somerset Island's north end.
LAND += [poly(
    (-95.70, 73.00), (-95.52, 73.43), (-95.02, 73.72),
    (-94.30, 73.94), (-93.55, 74.04), (-92.88, 73.91),
    (-92.25, 73.57), (-91.82, 73.00))]

# Bathurst Island: a deliberately angular, indented hand sketch.
LAND += [poly(
    (-103.05, 76.15), (-102.86, 76.56), (-102.40, 76.82),
    (-101.63, 76.96), (-100.73, 76.91), (-100.17, 76.74),
    (-99.48, 76.86), (-98.68, 76.78), (-97.95, 76.54),
    (-97.66, 76.22), (-98.06, 75.91), (-98.76, 75.72),
    (-99.56, 75.72), (-100.13, 75.88), (-100.70, 75.58),
    (-101.46, 75.48), (-102.05, 75.61), (-102.42, 75.83),
    (-102.96, 75.88))]

# The major remembered bays cutting into Bathurst.
CUT += [poly((-101.50, 77.02), (-100.93, 76.93), (-100.85, 76.56),
             (-101.02, 76.28), (-101.30, 76.48)),
        poly((-100.65, 75.45), (-100.28, 75.77), (-100.07, 76.15),
             (-100.34, 76.31), (-100.61, 75.91)),
        poly((-98.32, 75.77), (-98.65, 76.00), (-99.05, 76.14),
             (-98.88, 75.76))]

# Byam Martin and the Findlay-group chain west of Bathurst.
LAND += [poly((-105.0, 75.08), (-104.71, 75.40), (-104.17, 75.52),
             (-103.68, 75.36), (-103.55, 75.08), (-103.93, 74.91),
             (-104.56, 74.89), (-105.0, 75.01)),
         ellipse(-102.63, 75.86, .50, .20, -4),
         ellipse(-103.08, 76.03, .52, .18, 4),
         ellipse(-103.30, 76.20, .51, .20, 5),
         ellipse(-103.83, 76.47, .66, .25, 8),
         ellipse(-101.05, 76.66, .66, .26, -6)]

# Cornwallis Island and its close western satellites.
LAND += [poly(
    (-97.08, 75.19), (-96.75, 75.66), (-96.16, 75.91),
    (-95.36, 75.87), (-94.55, 75.72), (-93.70, 75.52),
    (-93.30, 75.24), (-93.57, 74.94), (-94.13, 74.70),
    (-94.92, 74.63), (-95.73, 74.78), (-96.51, 74.88)),
    poly((-98.08, 75.48), (-97.82, 75.77), (-97.26, 75.89),
         (-96.65, 75.76), (-96.43, 75.43), (-96.75, 75.18),
         (-97.39, 75.16), (-97.89, 75.28)),
    ellipse(-97.50, 74.55, .47, .20, 6),       # Lowther
    ellipse(-96.45, 74.75, .27, .13, -5),      # Browne
    ellipse(-95.50, 74.55, .42, .20, -8)]      # Griffith

# Baillie-Hamilton Island, north of Wellington Channel.
LAND += [poly((-95.53, 75.63), (-95.13, 76.03), (-94.52, 76.30),
             (-93.92, 76.20), (-93.72, 75.86), (-94.08, 75.57),
             (-94.78, 75.48))]

# Devon Island, extending beyond the eastern edge.  The western and southern
# shores are given more vertices because the channel pattern depends on them.
LAND += [poly(
    (-96.06, 75.03), (-95.78, 75.54), (-95.17, 75.95),
    (-94.25, 76.24), (-93.14, 76.44), (-91.84, 76.57),
    (-90.40, 76.64), (-88.83, 76.76), (-87.25, 76.74),
    (-85.65, 76.65), (-84.25, 76.53), (-82.72, 76.38),
    (-81.00, 76.16), (-81.00, 74.41), (-82.30, 74.39),
    (-83.55, 74.43), (-84.65, 74.53), (-85.54, 74.50),
    (-86.35, 74.38), (-87.15, 74.23), (-87.88, 74.20),
    (-88.70, 74.35), (-89.60, 74.48), (-90.36, 74.46),
    (-91.05, 74.57), (-91.78, 74.53), (-92.48, 74.64),
    (-93.28, 74.62), (-94.13, 74.72), (-94.92, 74.75),
    (-95.54, 74.82))]

# Remembered Devon embayments: Thomas Lee/Belcher from the north and
# Radstock/Croker bays from Lancaster Sound.
CUT += [poly((-92.40, 76.72), (-91.92, 76.52), (-91.88, 76.13),
             (-92.18, 75.93), (-92.48, 76.23)),
        poly((-90.62, 76.73), (-90.17, 76.59), (-90.02, 76.27),
             (-90.30, 76.08), (-90.54, 76.37)),
        poly((-91.56, 74.46), (-91.22, 74.78), (-90.82, 74.94),
             (-90.55, 74.80), (-90.80, 74.53)),
        poly((-87.98, 74.15), (-87.55, 74.45), (-87.04, 74.64),
             (-86.64, 74.55), (-87.13, 74.27))]

# Beechey and Prince Leopold Islands in Barrow Strait/Lancaster Sound.
LAND += [ellipse(-91.84, 74.71, .24, .10, -6),
         ellipse(-90.00, 74.04, .28, .13, -10)]

# The two northwestern Baffin peninsulas, clipped by the southern boundary.
# Admiralty Inlet remains the black opening between these polygons.
LAND += [poly((-89.27, 73.00), (-88.82, 73.31), (-88.15, 73.53),
             (-87.22, 73.66), (-86.28, 73.69), (-85.52, 73.56),
             (-85.07, 73.30), (-84.81, 73.00)),
         poly((-83.67, 73.00), (-83.38, 73.30), (-82.93, 73.57),
             (-82.28, 73.76), (-81.56, 73.85), (-81.00, 73.84),
             (-81.00, 73.00))]

# Lougheed's eastern edge and King Christian Island in the open northern sea.
LAND += [ellipse(-105.18, 77.31, .95, .29, 2),
         poly((-103.08, 77.49), (-102.58, 77.78), (-101.86, 77.91),
              (-101.15, 77.78), (-100.94, 77.50), (-101.42, 77.28),
              (-102.24, 77.25), (-102.86, 77.34))]

# Islands scattered across Norwegian Bay.
LAND += [poly((-96.25, 77.48), (-95.78, 77.76), (-95.09, 77.91),
             (-94.33, 77.85), (-93.71, 77.63), (-93.92, 77.38),
             (-94.55, 77.25), (-95.32, 77.28)),       # Cornwall
         poly((-91.76, 77.29), (-91.34, 77.57), (-90.68, 77.69),
              (-89.91, 77.62), (-89.34, 77.39), (-89.56, 77.17),
              (-90.25, 77.08), (-91.05, 77.10)),      # Graham
         ellipse(-91.03, 77.02, .36, .15, -7),        # Buckingham
         poly((-91.14, 76.49), (-90.69, 76.77), (-90.02, 76.88),
              (-89.49, 76.72), (-89.63, 76.43), (-90.18, 76.24),
              (-90.84, 76.28))]                       # North Kent

# Southern Axel Heiberg Island, clipped at 79 N.
LAND += [poly((-97.38, 79.00), (-96.70, 78.77), (-95.81, 78.57),
             (-94.67, 78.41), (-93.55, 78.38), (-92.52, 78.48),
             (-91.52, 78.63), (-90.48, 78.82), (-89.75, 79.00))]

# Southern Ellesmere Island.  This is the west and south coastline; the
# island continues north and east outside the requested rectangle.
LAND += [poly(
    (-86.08, 79.00), (-86.55, 78.69), (-87.37, 78.43),
    (-88.38, 78.18), (-89.22, 77.93), (-89.44, 77.68),
    (-89.05, 77.40), (-88.33, 77.12), (-87.41, 76.88),
    (-86.37, 76.74), (-85.50, 76.55), (-84.65, 76.41),
    (-83.69, 76.37), (-82.84, 76.45), (-81.93, 76.42),
    (-81.00, 76.34), (-81.00, 79.00))]

# Long west-facing fiords of Ellesmere.  They are subtracted after all land.
CUT += [poly((-89.36, 77.62), (-88.78, 77.69), (-88.16, 77.62),
             (-87.45, 77.43), (-86.44, 77.35), (-85.78, 77.48),
             (-86.52, 77.67), (-87.54, 77.79), (-88.55, 77.85)),
        poly((-88.46, 78.16), (-87.92, 78.15), (-87.27, 78.04),
             (-86.64, 77.89), (-85.93, 77.91), (-86.63, 78.17),
             (-87.61, 78.31)),
        poly((-86.73, 76.80), (-86.15, 76.93), (-85.39, 77.00),
             (-84.73, 76.93), (-85.32, 76.77))]

# A tiny remembered islet group in the channels; at this sampling scale these
# produce only a few white samples, but islands/reefs count as land.
LAND += [ellipse(-93.20, 76.77, .18, .08, 0),
         ellipse(-92.45, 77.05, .14, .07, 10),
         ellipse(-97.85, 77.12, .17, .08, -5),
         ellipse(-99.25, 77.65, .13, .07, 0)]


# Rasterize the point samples, not cells.  Array indices already have the
# requested north-to-south and west-to-east ordering.
lons = np.round(np.linspace(-105.0, -81.0, 241), 10)
lats = np.round(np.linspace(79.0, 73.0, 61), 10)
X, Y = np.meshgrid(lons, lats)
mask = np.zeros((61, 241), dtype=bool)
for shape in LAND:
    mask |= inside(shape, X, Y)
for shape in CUT:
    mask &= ~inside(shape, X, Y)
matrix = mask.astype(np.uint8)

out = Path("outputs")
out.mkdir(parents=True, exist_ok=True)
with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
    csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

raw = Image.fromarray(matrix * 255, mode="L")
raw.save(out / "candidate.png")
big = raw.resize((1928, 488), resample=Image.Resampling.NEAREST)
big.save(out / "candidate_8x.png")

# Fail loudly if an edit ever violates an output contract.
assert matrix.shape == (61, 241) and matrix.size == 14701
assert set(np.unique(matrix)).issubset({0, 1})
assert raw.size == (241, 61) and set(raw.getdata()).issubset({0, 255})
assert big.size == (1928, 488) and set(big.getdata()).issubset({0, 255})
assert np.array_equal(np.asarray(big), np.repeat(np.repeat(
    np.asarray(raw), 8, axis=0), 8, axis=1))
