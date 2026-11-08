"""Knowledge-based reconstruction of the land/water raster.

Coordinates in the hand-written outlines use positive west longitude.
They are deliberately approximate: this is a rule-based reconstruction,
not a lookup of a coastline data set.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

LATS = [79.0 - 0.1 * i for i in range(61)]
LONS = [105.0 - 0.1 * i for i in range(241)]

# Polygon vertices are (degrees west, degrees north), in outline order.
# The larger outlines are the principal islands in this part of the
# Canadian Arctic Archipelago.
POLYGONS = [
    # Victoria Island: only its eastern edge enters the western frame.
    [(105.01, 73.02), (103.8, 73.12), (102.7, 73.55),
     (102.3, 74.25), (103.2, 74.95), (104.7, 75.28),
     (105.01, 75.35)],

    # Melville Island.
    [(105.05, 75.00), (106.1, 75.75), (106.0, 77.05),
     (105.2, 78.10), (105.0, 79.05), (102.8, 79.72),
     (98.0, 79.78), (94.0, 79.35), (91.5, 78.65),
     (90.2, 77.75), (91.2, 76.95), (94.2, 76.55),
     (96.2, 75.95), (100.3, 75.05)],

    # Bathurst Island, east of the Melville island channels.
    [(100.9, 77.42), (98.7, 77.78), (95.2, 77.72),
     (91.5, 77.25), (89.5, 76.55), (90.0, 75.75),
     (92.1, 75.12), (95.0, 75.05), (98.0, 75.45),
     (100.1, 76.25)],

    # Cornwallis Island.
    [(96.45, 75.90), (94.8, 76.20), (92.0, 76.08),
     (90.65, 75.60), (90.35, 74.88), (91.85, 74.35),
     (94.65, 74.40), (96.55, 75.02)],

    # Prince of Wales Island.
    [(103.05, 75.10), (101.2, 75.92), (98.1, 75.90),
     (95.3, 75.40), (92.4, 74.55), (90.35, 73.55),
     (90.05, 72.82), (93.3, 72.42), (97.0, 72.72),
     (100.2, 73.28), (102.3, 74.02)],

    # Somerset Island.
    [(95.05, 74.72), (93.0, 75.10), (89.4, 75.06),
     (86.0, 74.72), (82.75, 74.00), (82.25, 73.28),
     (84.0, 72.62), (87.2, 72.30), (90.3, 72.42),
     (93.2, 73.00), (94.75, 73.78)],
]

# Smaller islands and persistent rocky outliers.  These are included as
# separate outlines so that they do not get lost in the broad island shapes.
SMALL_ISLANDS = [
    # Borden Island
    [(96.0, 79.45), (94.7, 79.72), (92.2, 79.55),
     (90.7, 78.92), (91.3, 78.20), (93.8, 77.95),
     (96.0, 78.35)],
    # Mackenzie King Island
    [(100.7, 79.35), (99.0, 79.58), (97.0, 79.20),
     (96.4, 78.55), (97.2, 77.92), (99.4, 77.98),
     (100.9, 78.55)],
    # Lougheed Island
    [(98.0, 78.55), (96.5, 78.82), (94.9, 78.55),
     (94.5, 77.95), (95.7, 77.60), (97.5, 77.75)],
    # small islands near the Prince of Wales / Somerset channels
    [(88.45, 74.60), (87.65, 74.78), (87.15, 74.42),
     (87.55, 74.05), (88.35, 74.10)],
    [(84.35, 73.65), (83.65, 73.82), (83.15, 73.48),
     (83.55, 73.15), (84.25, 73.22)],
    [(99.1, 73.05), (98.5, 73.30), (98.0, 73.03),
     (98.35, 72.78), (98.95, 72.80)],
]

def inside(lon, lat, polygon):
    """Even-odd point-in-polygon test, with boundary treated as land."""
    hit = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # Explicit boundary test avoids holes along a hand-drawn edge.
        cross = (lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
        if abs(cross) < 1e-10 and min(x1, x2) - 1e-10 <= lon <= max(x1, x2) + 1e-10 and min(y1, y2) - 1e-10 <= lat <= max(y1, y2) + 1e-10:
            return True
        if (y1 > lat) != (y2 > lat):
            at_x = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < at_x:
                hit = not hit
    return hit


def land(lon, lat):
    return any(inside(lon, lat, p) for p in POLYGONS + SMALL_ISLANDS)


matrix = np.asarray(
    [[1 if land(lon, lat) else 0 for lon in LONS] for lat in LATS],
    dtype=np.uint8,
)

# The requested CSV is deliberately headerless and contains decimal 0/1.
with (OUT / "candidate.csv").open("w", newline="", encoding="ascii") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(matrix.tolist())

# Pillow's "L" mode gives exactly two possible pixel values here.
base = Image.fromarray(matrix * 255, mode="L")
base.save(OUT / "candidate.png")
enlarged = base.resize((base.width * 8, base.height * 8), Image.Resampling.NEAREST)
enlarged.save(OUT / "candidate_8x.png")

# Self-checks make accidental omissions or format changes fail loudly.
assert matrix.shape == (61, 241)
assert matrix.size == 14701
assert set(np.unique(matrix).tolist()) <= {0, 1}
assert base.size == (241, 61)
assert enlarged.size == (1928, 488)
assert set(base.getdata()) <= {0, 255}
