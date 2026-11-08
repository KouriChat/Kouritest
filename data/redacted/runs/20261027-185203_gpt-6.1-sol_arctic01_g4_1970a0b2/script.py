"""Knowledge-based reconstruction of the Canadian Arctic Archipelago.

This is deliberately a hand drawn model: the vertices below are approximate
coastlines remembered from general geographic knowledge, not imported data.
"""
import csv
import os
import numpy as np
from PIL import Image

NLAT, NLON = 61, 241
WEST, EAST = -105.0, -81.0
NORTH, SOUTH = 79.0, 73.0

# A polygon is listed counter-clockwise as (longitude, latitude).  The
# relatively generous vertex count is useful at the 0.1 degree sampling scale.
POLYGONS = [
    # Ellesmere Island, southern part visible in the north-east of the frame
    [(-95.2,78.1),(-94.0,78.6),(-92.0,78.9),(-89.0,79.0),
     (-84.7,79.0),(-81.0,79.0),(-81.0,77.0),(-82.2,76.3),
     (-84.0,76.0),(-86.5,75.8),(-89.0,76.0),(-91.5,76.4),
     (-94.0,77.0)],
    # Axel Heiberg Island
    [(-97.2,78.8),(-96.2,79.0),(-93.5,79.0),(-91.8,78.7),
     (-91.0,78.0),(-91.5,77.0),(-93.0,76.6),(-95.3,76.8),
     (-96.8,77.5)],
    # Devon Island
    [(-91.8,76.8),(-89.0,77.0),(-85.0,76.8),(-81.7,76.2),
     (-80.8,75.2),(-82.1,74.7),(-85.2,74.6),(-88.0,74.8),
     (-90.8,75.2),(-92.0,76.0)],
    # Cornwallis Island
    [(-96.0,76.6),(-93.6,76.7),(-91.7,76.1),(-91.2,75.2),
     (-92.3,74.8),(-95.0,75.0),(-96.5,75.7)],
    # Bathurst Island
    [(-100.7,76.2),(-98.2,76.6),(-95.8,76.1),(-95.0,75.2),
     (-96.0,74.7),(-99.0,74.8),(-100.8,75.3)],
    # Melville Island (only its eastern/central portion lies in the frame)
    [(-105.0,75.8),(-102.8,76.2),(-100.1,76.1),(-98.0,75.6),
     (-97.0,74.8),(-98.0,74.0),(-100.5,73.6),(-103.0,73.8),
     (-105.0,74.4)],
    # Prince of Wales Island
    [(-101.8,74.1),(-99.6,74.6),(-96.9,74.4),(-94.8,73.8),
     (-94.3,73.0),(-95.8,73.0),(-98.4,73.1),(-101.0,73.5)],
    # Somerset Island
    [(-94.0,74.4),(-91.0,74.6),(-87.0,74.3),(-84.1,73.7),
     (-84.1,73.0),(-94.2,73.0),(-95.0,73.6)],
]

def point_in_polygon(x, y, poly):
    """Ray casting, used only for the hand-written polygons above."""
    inside = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        crosses = ((yi > y) != (yj > y))
        if crosses:
            at_edge = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < at_edge:
                inside = not inside
        j = i
    return inside

def ellipse(x, y, cx, cy, rx, ry):
    return ((x-cx)/rx)**2 + ((y-cy)/ry)**2 <= 1.0

# Further islands and the pieces of larger land masses cut by the requested
# southern boundary.  These are separate polygons so that narrow channels
# remain water rather than being filled by a raster operation.
POLYGONS += [
    # Victoria Island: eastern end visible along the lower-left boundary
    [(-105.0,74.8),(-103.4,75.0),(-101.7,74.6),(-100.2,73.8),
     (-100.4,73.0),(-105.0,73.0)],
    # Boothia Peninsula, at the lower edge of the map
    [(-97.8,73.0),(-97.0,73.6),(-95.2,74.0),(-93.4,73.8),
     (-92.0,73.3),(-91.8,73.0)],
    # western Baffin Island coast entering at the right-hand lower edge
    [(-84.3,73.0),(-84.0,73.6),(-82.8,74.2),(-81.0,74.5),
     (-81.0,73.0)],
    # Byam Martin Island
    [(-94.8,76.3),(-93.4,76.5),(-91.9,76.2),(-92.1,75.7),
     (-93.8,75.6),(-95.0,75.9)],
    # Little Cornwallis Island
    [(-96.8,75.2),(-95.8,75.5),(-94.7,75.3),(-94.8,74.8),
     (-96.0,74.7),(-96.9,74.9)],
    # Eglinton Island
    [(-101.2,75.0),(-100.0,75.4),(-98.9,75.1),(-98.8,74.4),
     (-100.0,74.2),(-101.1,74.5)],
    # Alexander Island / small islands north of the Lancaster Sound shore
    [(-92.9,75.2),(-92.1,75.4),(-91.4,75.1),(-91.5,74.7),
     (-92.5,74.7)],
    # North Kent Island
    [(-90.8,76.9),(-89.8,77.1),(-89.1,76.7),(-89.4,76.2),
     (-90.5,76.2),(-91.0,76.5)],
    # Prince Leopold and nearby small islands
    [(-91.9,74.7),(-91.3,74.9),(-90.7,74.6),(-90.9,74.2),
     (-91.7,74.3)],
]

# A few tiny named or unnamed rocks/sand bars that survive at this sampling
# resolution.  Ellipses are used only for these isolated small features.
SMALL_ISLANDS = [
    (-99.0, 76.9, 0.18, 0.10),
    (-97.0, 76.8, 0.16, 0.12),
    (-88.0, 76.0, 0.20, 0.10),
    (-86.1, 75.6, 0.15, 0.09),
    (-103.0, 74.2, 0.14, 0.08),
    (-96.2, 74.1, 0.12, 0.09),
    (-87.5, 73.3, 0.20, 0.08),
]

def is_land(lon, lat):
    # No cell interpolation is performed: this is a direct point test.
    for poly in POLYGONS:
        if point_in_polygon(lon, lat, poly):
            return True
    for cx, cy, rx, ry in SMALL_ISLANDS:
        if ellipse(lon, lat, cx, cy, rx, ry):
            return True
    return False

def build_matrix():
    result = np.zeros((NLAT, NLON), dtype=np.uint8)
    for r in range(NLAT):
        lat = NORTH - 0.1 * r
        for c in range(NLON):
            lon = WEST + 0.1 * c
            result[r, c] = 1 if is_land(lon, lat) else 0
    return result

def main():
    matrix = build_matrix()
    assert matrix.shape == (61, 241)
    assert matrix.size == 14701
    assert set(np.unique(matrix)).issubset({0, 1})
    os.makedirs("outputs", exist_ok=True)

    with open("outputs/candidate.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        for row in matrix:
            writer.writerow([int(v) for v in row])

    # White is land (1), black is water (0); no antialiasing is involved.
    image_data = (matrix * 255).astype(np.uint8)
    image = Image.fromarray(image_data, mode="L")
    assert image.size == (241, 61)
    image.save("outputs/candidate.png")

    enlarged = image.resize((241 * 8, 61 * 8), resample=Image.Resampling.NEAREST)
    assert enlarged.size == (1928, 488)
    enlarged.save("outputs/candidate_8x.png")

    # Basic post-write checks requested for the deliverable.
    with open("outputs/candidate.csv", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 61 and all(len(row) == 241 for row in rows)
    assert all(v in ("0", "1") for row in rows for v in row)
    assert set(image.getdata()).issubset({0, 255})
    assert set(enlarged.getdata()).issubset({0, 255})
    print("validated:", matrix.shape, matrix.size, "samples")

if __name__ == "__main__":
    main()

