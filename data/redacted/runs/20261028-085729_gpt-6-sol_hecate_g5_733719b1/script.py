"""Knowledge-based reconstruction of the Hecate Strait sample grid.

The coastline is deliberately hand-authored as approximate polygons and
small island/reef shapes; no geographic data are read or downloaded.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

N = 127
OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

# Grid coordinates: x is degrees east of 131 W, y is degrees south of 55 N.
def xy(west, north):
    return (131.0 - west, 55.0 - north)

def poly(points):
    return [xy(w, n) for w, n in points]

def inside(px, py, points):
    """Ray crossing test, with boundary points counted as land."""
    hit = False
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        cross = (y1 > py) != (y2 > py)
        if cross:
            at = (x2 - x1) * (py - y1) / (y2 - y1) + x1
            if px <= at:
                hit = not hit
    return hit

def add_polygon(mask, points):
    points = poly(points)
    for r in range(N):
        y = 2.0 * r / 126.0
        for c in range(N):
            if inside(2.0 * c / 126.0, y, points):
                mask[r, c] = 1

def add_ellipse(mask, west, north, rw, rh):
    cx, cy = xy(west, north)
    for r in range(N):
        y = 2.0 * r / 126.0
        for c in range(N):
            x = 2.0 * c / 126.0
            if ((x - cx) / rw) ** 2 + ((y - cy) / rh) ** 2 <= 1:
                mask[r, c] = 1

def cut_polygon(mask, points):
    points = poly(points)
    for r in range(N):
        y = 2.0 * r / 126.0
        for c in range(N):
            if inside(2.0 * c / 126.0, y, points):
                mask[r, c] = 0

land = np.zeros((N, N), dtype=np.uint8)

# The mainland: a deliberately irregular west-facing shore, with all land
# to its east.  The western edge is the Hecate Strait / coastal BC shore.
add_polygon(land, [
    (130.42, 55.00), (129.00, 55.00), (129.00, 53.00),
    (129.62, 53.00), (129.55, 53.16), (129.35, 53.28),
    (129.50, 53.40), (129.20, 53.55), (129.34, 53.68),
    (129.00, 53.84), (129.12, 54.00), (129.33, 54.12),
    (129.18, 54.28), (129.58, 54.42), (129.48, 54.57),
    (129.72, 54.72), (129.55, 54.86), (129.75, 55.00),
])

# A set of fjord mouths and coastal indentations.  These are water cut-outs,
# not a raster smoothing operation.
cut_polygon(land, [
    (129.00, 53.02), (129.00, 53.22), (129.22, 53.28),
    (129.38, 53.21), (129.52, 53.00),
])
cut_polygon(land, [
    (129.00, 53.40), (129.00, 53.62), (129.20, 53.70),
    (129.42, 53.64), (129.30, 53.50), (129.18, 53.40),
])
cut_polygon(land, [
    (129.00, 53.84), (129.00, 54.10), (129.25, 54.18),
    (129.45, 54.06), (129.32, 53.88), (129.20, 53.80),
])
cut_polygon(land, [
    (129.00, 54.30), (129.00, 54.58), (129.26, 54.66),
    (129.46, 54.53), (129.32, 54.38), (129.20, 54.28),
])
# Northern entrance islands: Dundas, Stephens, and the small islands around
# Prince Rupert.  Each outline is intentionally coarse at this grid scale.
add_polygon(land, [
    (130.98, 54.73), (130.76, 54.71), (130.61, 54.61),
    (130.65, 54.46), (130.83, 54.39), (131.00, 54.43),
    (131.00, 54.70),
])
add_polygon(land, [
    (130.98, 54.36), (130.84, 54.32), (130.77, 54.20),
    (130.88, 54.14), (131.00, 54.18), (131.00, 54.34),
])
add_ellipse(land, 130.63, 54.78, .035, .035)
add_ellipse(land, 130.54, 54.55, .025, .035)
add_ellipse(land, 130.73, 54.23, .030, .025)

# Kaien Island and the Prince Rupert island group.
add_polygon(land, [
    (130.48, 54.40), (130.34, 54.39), (130.23, 54.31),
    (130.25, 54.20), (130.40, 54.17), (130.53, 54.24),
    (130.57, 54.34),
])
add_ellipse(land, 130.16, 54.30, .045, .025)
add_ellipse(land, 130.02, 54.25, .026, .020)
add_ellipse(land, 130.46, 54.10, .035, .022)
add_ellipse(land, 130.67, 54.00, .025, .020)

# Porcher Island, a large elongate island south-west of the mainland.
add_polygon(land, [
    (130.73, 54.08), (130.48, 54.06), (130.24, 53.97),
    (130.10, 53.82), (130.18, 53.68), (130.42, 53.66),
    (130.67, 53.75), (130.80, 53.90),
])
add_ellipse(land, 130.18, 53.92, .055, .040)
add_ellipse(land, 130.54, 53.79, .042, .030)

# Islands in the inner side of Hecate Strait.
add_polygon(land, [
    (130.06, 53.78), (129.92, 53.73), (129.86, 53.60),
    (129.95, 53.50), (130.12, 53.53), (130.19, 53.66),
])
add_ellipse(land, 130.30, 53.60, .035, .025)
add_ellipse(land, 130.76, 53.52, .032, .025)
add_ellipse(land, 130.88, 53.43, .022, .018)

# Small named-island-scale fragments and reefs retained as land when sampled.
for args in [
    (130.92, 54.06, .018, .014), (130.79, 54.15, .022, .016),
    (130.60, 54.28, .018, .014), (130.35, 54.48, .020, .014),
    (130.05, 54.42, .016, .012), (129.83, 54.11, .016, .012),
    (129.72, 53.92, .020, .014), (129.50, 53.72, .017, .012),
    (129.42, 53.57, .018, .012), (129.24, 53.46, .014, .010),
]:
    add_ellipse(land, *args)
# Banks Island and Pitt Island, with their broken eastern shores.
add_polygon(land, [
    (130.12, 53.97), (129.83, 53.96), (129.57, 53.85),
    (129.42, 53.68), (129.50, 53.48), (129.76, 53.38),
    (130.02, 53.43), (130.19, 53.61), (130.27, 53.80),
])
add_polygon(land, [
    (129.98, 53.63), (129.78, 53.59), (129.64, 53.47),
    (129.68, 53.31), (129.87, 53.24), (130.04, 53.33),
    (130.11, 53.50),
])
add_ellipse(land, 129.38, 53.79, .035, .024)
add_ellipse(land, 129.27, 53.64, .027, .020)
add_ellipse(land, 129.18, 53.52, .020, .014)
add_ellipse(land, 129.44, 53.39, .025, .018)
add_ellipse(land, 129.25, 53.30, .020, .014)
add_ellipse(land, 129.06, 53.22, .022, .016)

# Gil Island and the northern end of Campania Island near the southern edge.
add_polygon(land, [
    (129.14, 53.73), (128.98, 53.68), (128.90, 53.55),
    (128.98, 53.43), (129.16, 53.42), (129.29, 53.53),
    (129.27, 53.66),
])
add_polygon(land, [
    (129.76, 53.10), (129.59, 53.07), (129.48, 53.00),
    (129.98, 53.00), (129.94, 53.08), (129.86, 53.13),
])
add_ellipse(land, 129.31, 53.12, .026, .018)
add_ellipse(land, 129.06, 53.07, .021, .014)

# Eastern fringe of Graham Island (Haida Gwaii).  Most of that island is
# west of the requested frame; only the edge touching 131 W is represented.
add_polygon(land, [
    (131.00, 54.92), (130.96, 54.87), (130.95, 54.72),
    (130.98, 54.60), (131.00, 54.55), (131.00, 54.92),
])
add_ellipse(land, 130.97, 54.49, .018, .035)
add_ellipse(land, 130.98, 54.35, .014, .028)

# Eastern fringe of Moresby Island and its nearshore islets.
add_polygon(land, [
    (131.00, 53.80), (130.96, 53.73), (130.95, 53.56),
    (130.98, 53.40), (130.96, 53.20), (130.99, 53.00),
    (131.00, 53.00),
])
add_ellipse(land, 130.96, 53.66, .018, .036)
add_ellipse(land, 130.97, 53.34, .014, .025)
add_ellipse(land, 130.98, 53.11, .012, .020)
# A few final skerries used to retain the island-rich character of the
# inner coast without attempting sub-grid detail.
for args in [
    (130.98, 54.82, .012, .010), (130.90, 54.67, .015, .010),
    (130.84, 54.53, .012, .010), (130.56, 54.01, .014, .010),
    (130.43, 53.91, .012, .009), (130.22, 53.76, .014, .010),
    (130.02, 53.89, .013, .009), (129.88, 53.76, .012, .009),
    (129.70, 53.66, .013, .009), (129.58, 53.47, .012, .009),
    (129.41, 53.26, .013, .009), (129.13, 53.36, .011, .008),
]:
    add_ellipse(land, *args)

# CSV rows are north-to-south and columns west-to-east by construction.
with (OUT / "candidate.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, lineterminator="\n")
    for row in land:
        writer.writerow([int(v) for v in row])

# Pillow's L mode gives exactly two grayscale values: black water, white land.
raw = Image.fromarray((land * 255).astype(np.uint8), mode="L")
raw.save(OUT / "candidate.png")
zoom = raw.resize((N * 8, N * 8), resample=Image.Resampling.NEAREST)
zoom.save(OUT / "candidate_8x.png")

# Lightweight self-checks required by the task.
assert land.shape == (127, 127)
assert land.size == 16129
assert set(np.unique(land)).issubset({0, 1})
assert raw.size == (127, 127)
assert zoom.size == (1016, 1016)
assert set(raw.getdata()).issubset({0, 255})
assert set(zoom.getdata()).issubset({0, 255})
print("generated", land.shape, "values=", land.size,
      "land=", int(land.sum()), "water=", int((land == 0).sum()))
