"""Knowledge-based reconstruction of the Canadian Arctic island region.

The geometry below is deliberately hand-written: it is not read from a map
or a geographic data source.  Coordinates are approximate WGS84 degrees.
"""
import csv
import os
import numpy as np
from PIL import Image

WEST, EAST = -105.0, -81.0
NORTH, SOUTH = 79.0, 73.0
STEP = 0.1

def inside(x, y, ring):
    """Ray crossing test for one hand-drawn coastline ring."""
    hit = False
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)):
            cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < cross:
                hit = not hit
        j = i
    return hit

def land(lon, lat):
    # The polygons are listed below in west-to-east geographic order.
    for ring in ISLANDS:
        if inside(lon, lat, ring):
            return True
    return False

# Victoria Island: only its northeastern/eastern shore enters this frame.
VICTORIA = [
    (-105.0, 73.00), (-105.0, 73.55), (-104.2, 73.92),
    (-103.0, 74.20), (-101.7, 74.42), (-100.2, 74.35),
    (-99.0, 73.90), (-98.6, 73.45), (-99.2, 73.12),
    (-100.5, 72.92), (-102.5, 72.90), (-104.0, 72.94),
]

# Melville Island, including its long southern shore and eastern headland.
MELVILLE = [
    (-105.0, 74.82), (-102.8, 74.70), (-100.0, 74.62),
    (-97.2, 74.75), (-94.2, 75.02), (-91.5, 75.45),
    (-90.2, 76.10), (-90.1, 77.45), (-91.8, 77.76),
    (-95.0, 77.92), (-98.4, 78.02), (-102.0, 78.02),
    (-105.0, 77.78),
]

# Bathurst Island is the compact island immediately southeast of Melville.
BATHURST = [
    (-105.0, 75.23), (-104.4, 75.72), (-103.3, 76.25),
    (-101.7, 76.64), (-99.6, 76.78), (-97.5, 76.69),
    (-96.0, 76.38), (-95.4, 75.88), (-95.7, 75.38),
    (-97.5, 75.10), (-100.0, 74.98), (-102.5, 75.02),
]

# Cornwallis Island, between Bathurst/Devon and Prince of Wales.
CORNWALLIS = [
    (-96.0, 74.92), (-95.6, 75.42), (-94.7, 75.91),
    (-93.2, 76.20), (-91.4, 76.17), (-90.4, 75.78),
    (-90.0, 75.25), (-90.6, 74.87), (-92.3, 74.68),
    (-94.3, 74.67),
]
# Devon Island: its western part is outside the frame; this is the
# southeastern/eastern portion visible along the right side of the map.
DEVON = [
    (-90.3, 74.65), (-88.7, 74.92), (-86.8, 75.18),
    (-84.8, 75.55), (-82.8, 75.87), (-81.0, 76.05),
    (-81.0, 78.05), (-83.3, 78.20), (-86.3, 78.05),
    (-89.0, 77.70), (-91.0, 77.15), (-91.7, 76.52),
    (-91.1, 75.75),
]

# Prince of Wales Island.  The irregular north and south coasts are
# represented by extra vertices rather than by a rectangular approximation.
PRINCE_OF_WALES = [
    (-103.8, 74.18), (-102.8, 74.78), (-101.0, 75.25),
    (-98.7, 75.55), (-96.2, 75.48), (-94.0, 75.10),
    (-92.8, 74.55), (-93.0, 73.82), (-93.8, 73.25),
    (-94.2, 72.92), (-95.4, 72.72), (-97.2, 72.69),
    (-99.0, 72.78), (-101.1, 72.64), (-102.5, 72.70),
    (-103.0, 73.03), (-103.8, 73.54),
]

# Somerset Island, separated from Prince of Wales by the narrow channel
# around its southern and western sides.
SOMERSET = [
    (-94.2, 74.35), (-93.7, 74.65), (-92.3, 74.72),
    (-90.6, 74.56), (-89.4, 74.05), (-89.1, 73.38),
    (-89.6, 72.72), (-90.4, 72.28), (-92.0, 72.12),
    (-93.8, 72.16), (-95.0, 72.48), (-94.8, 72.98),
    (-94.5, 73.67),
]

# Northern Boothia Peninsula: only the tip is in the southern edge of
# the requested box.
BOOTHIA_TIP = [
    (-99.4, 72.72), (-98.8, 73.35), (-97.3, 73.72),
    (-95.8, 73.82), (-94.4, 73.66), (-93.6, 73.30),
    (-93.4, 72.72),
]

# Northwest Baffin and the small eastern headland visible at the frame edge.
BAFFIN_EDGE = [
    (-81.0, 73.00), (-81.0, 74.10), (-81.8, 74.62),
    (-82.7, 75.05), (-83.8, 75.38), (-85.0, 75.53),
    (-85.5, 75.08), (-84.9, 74.55), (-83.8, 74.10),
    (-82.7, 73.62), (-82.2, 73.18),
]

ISLANDS = [
    VICTORIA, MELVILLE, BATHURST, CORNWALLIS, DEVON,
    PRINCE_OF_WALES, SOMERSET, BOOTHIA_TIP, BAFFIN_EDGE,
]
def build_matrix():
    rows = []
    for ri in range(61):
        lat = NORTH - ri * STEP
        row = []
        for ci in range(241):
            lon = WEST + ci * STEP
            row.append(1 if land(lon, lat) else 0)
        rows.append(row)
    return np.asarray(rows, dtype=np.uint8)

def write_outputs(matrix):
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/candidate.csv", "w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

    # A direct one-pixel-per-sample image: white land, black water.
    raw = Image.fromarray(np.where(matrix == 1, 255, 0).astype(np.uint8),
                          mode="L")
    raw.save("outputs/candidate.png")
    enlarged = raw.resize((241 * 8, 61 * 8), resample=Image.Resampling.NEAREST)
    enlarged.save("outputs/candidate_8x.png")

if __name__ == "__main__":
    result = build_matrix()
    assert result.shape == (61, 241)
    assert result.size == 14701
    assert set(np.unique(result)).issubset({0, 1})
    write_outputs(result)
