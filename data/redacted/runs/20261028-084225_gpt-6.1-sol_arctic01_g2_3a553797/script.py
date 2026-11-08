"""Offline, knowledge-based reconstruction of the Northwest Passage core area.

The geometry below is deliberately hand-authored.  It is not read from a map,
database, raster, coastline package, or online service.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

WEST, EAST = -105.0, -81.0
NORTH, SOUTH = 79.0, 73.0
STEP = 0.1


def polygon(points):
    return tuple((float(x), float(y)) for x, y in points)


# Large islands and continental fragments, in approximate clockwise order.
LAND = [
    # Axel Heiberg Island
    polygon([
        (-97.2, 78.0), (-96.6, 79.2), (-94.5, 80.5),
        (-91.0, 80.7), (-88.8, 79.7), (-88.5, 78.3),
        (-90.2, 77.2), (-93.0, 76.8), (-96.0, 77.1),
    ]),
    # Western and central Ellesmere Island
    polygon([
        (-91.6, 78.1), (-91.3, 79.4), (-89.2, 80.5),
        (-85.0, 81.0), (-80.0, 80.4), (-77.8, 79.0),
        (-78.8, 77.4), (-80.7, 76.2), (-83.1, 75.4),
        (-86.8, 75.8), (-89.8, 76.7),
    ]),
    # Devon Island
    polygon([
        (-96.0, 75.2), (-94.1, 77.0), (-91.7, 78.0),
        (-87.9, 78.3), (-84.8, 77.8), (-82.2, 76.7),
        (-81.2, 75.2), (-83.0, 74.1), (-87.0, 73.8),
        (-91.0, 74.0), (-94.5, 74.2),
    ]),
    # Melville Island
    polygon([
        (-106.0, 74.0), (-104.0, 75.5), (-103.0, 77.2),
        (-100.3, 78.0), (-96.8, 77.9), (-94.0, 77.0),
        (-92.0, 75.8), (-91.0, 74.7), (-92.4, 73.9),
        (-96.4, 73.6), (-100.2, 73.7), (-103.4, 73.5),
    ]),
    # Bathurst Island
    polygon([
        (-101.5, 75.0), (-100.6, 76.1), (-98.7, 76.8),
        (-96.4, 76.7), (-95.3, 75.8), (-95.7, 74.8),
        (-97.4, 74.4), (-100.2, 74.5),
    ]),
    # Cornwallis Island
    polygon([
        (-95.0, 74.8), (-94.0, 75.8), (-92.0, 76.2),
        (-90.6, 75.5), (-90.8, 74.6), (-92.3, 74.1),
        (-94.1, 74.2),
    ]),
    # Prince of Wales Island
    polygon([
        (-102.5, 72.9), (-102.0, 74.0), (-100.2, 74.6),
        (-98.0, 74.3), (-96.0, 73.7), (-95.0, 72.9),
        (-96.8, 72.6), (-99.5, 72.6), (-101.2, 72.7),
    ]),
    # Somerset Island
    polygon([
        (-95.2, 72.8), (-94.7, 74.1), (-92.7, 74.8),
        (-90.4, 74.7), (-88.9, 73.7), (-88.2, 72.9),
        (-90.2, 72.7), (-92.5, 72.6), (-94.0, 72.7),
    ]),
    # Baffin Island, western coast fragment in the eastern edge
    polygon([
        (-81.6, 73.0), (-81.8, 74.1), (-80.8, 75.5),
        (-79.5, 77.0), (-77.8, 78.2), (-76.0, 78.8),
        (-74.0, 78.5), (-73.5, 77.0), (-74.5, 75.2),
        (-76.5, 73.8), (-79.0, 72.8),
    ]),
    # Banks Island (western edge, northern portion)
    polygon([
        (-122.0, 71.0), (-121.0, 74.0), (-119.0, 76.0),
        (-116.0, 76.5), (-114.0, 75.0), (-114.0, 72.0),
        (-117.0, 71.0),
    ]),
    # Northern mainland coast, from Amundsen Gulf east to Boothia
    polygon([
        (-130.0, 69.0), (-117.0, 69.0), (-115.0, 70.3),
        (-113.0, 71.1), (-110.0, 71.4), (-107.0, 71.8),
        (-104.0, 72.2), (-101.0, 72.6), (-98.0, 72.8),
        (-95.0, 72.9), (-92.0, 72.8), (-89.0, 72.6),
        (-86.0, 72.4), (-83.0, 72.2), (-80.0, 72.0),
        (-78.0, 71.0), (-77.0, 69.0),
    ]),
    # Boothia Peninsula and northwestern mainland promontory
    polygon([
        (-96.0, 72.2), (-95.5, 73.0), (-94.3, 73.5),
        (-92.5, 73.4), (-91.5, 72.8), (-91.0, 72.0),
        (-92.5, 71.5), (-94.8, 71.7),
    ]),
]


# Smaller islands important at this grid scale: rough elliptical polygons.
ISLANDS = [
    (-99.8, 77.1, 0.45, 0.32), (-97.8, 77.0, 0.36, 0.24),
    (-100.8, 76.2, 0.28, 0.22), (-98.9, 75.9, 0.30, 0.22),
    (-97.2, 75.4, 0.25, 0.18), (-94.8, 76.6, 0.25, 0.16),
    (-93.5, 76.1, 0.28, 0.18), (-90.4, 76.9, 0.23, 0.16),
    (-88.2, 77.3, 0.25, 0.18), (-86.7, 76.7, 0.22, 0.15),
    (-84.3, 75.9, 0.22, 0.15), (-82.8, 76.2, 0.18, 0.14),
    (-101.8, 74.0, 0.30, 0.20), (-98.1, 73.9, 0.25, 0.16),
    (-96.0, 73.5, 0.22, 0.14), (-89.0, 74.0, 0.22, 0.14),
]
def point_in_polygon(x, y, poly):
    """Ray crossing test for one sampling point and one hand-written polygon."""
    inside = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        crosses = ((yi > y) != (yj > y))
        if crosses:
            boundary_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < boundary_x:
                inside = not inside
        j = i
    return inside


def in_ellipse(x, y, cx, cy, rx, ry):
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


# Lakes are holes in otherwise land-filled polygons.  They are included only
# where their approximate size can affect a 0.1 degree sample.
LAKES = [
    # Representative large inland waters on the main islands.
    polygon([
        (-99.3, 76.0), (-98.8, 76.3), (-97.9, 76.2),
        (-97.7, 75.9), (-98.3, 75.7), (-99.1, 75.7),
    ]),
    polygon([
        (-92.6, 75.7), (-92.1, 75.9), (-91.5, 75.7),
        (-91.6, 75.3), (-92.3, 75.2), (-92.8, 75.4),
    ]),
    polygon([
        (-87.2, 76.8), (-86.6, 77.0), (-85.8, 76.8),
        (-85.9, 76.4), (-86.7, 76.4),
    ]),
]


def is_land(lon, lat):
    for poly in LAND:
        if point_in_polygon(lon, lat, poly):
            for lake in LAKES:
                if point_in_polygon(lon, lat, lake):
                    return 0
            return 1
    for cx, cy, rx, ry in ISLANDS:
        if in_ellipse(lon, lat, cx, cy, rx, ry):
            return 1
    return 0


def build_matrix():
    longitudes = np.round(np.arange(WEST, EAST + STEP / 2, STEP), 1)
    latitudes = np.round(np.arange(NORTH, SOUTH - STEP / 2, -STEP), 1)
    matrix = np.array(
        [[is_land(float(lon), float(lat)) for lon in longitudes]
         for lat in latitudes],
        dtype=np.uint8,
    )
    if matrix.shape != (61, 241):
        raise RuntimeError(f"unexpected matrix shape: {matrix.shape}")
    if not np.all((matrix == 0) | (matrix == 1)):
        raise RuntimeError("matrix contains a value other than 0 or 1")
    return matrix
def write_outputs(matrix):
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)

    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # 0 is black water, 1 is white land.  "L" ensures exactly two values.
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(out / "candidate.png")
    enlarged = image.resize((241 * 8, 61 * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(out / "candidate_8x.png")


def validate_outputs(matrix):
    out = Path("outputs")
    rows = (out / "candidate.csv").read_text(encoding="ascii").splitlines()
    if len(rows) != 61 or any(len(row.split(",")) != 241 for row in rows):
        raise RuntimeError("CSV dimensions are invalid")
    if any(value not in {"0", "1"} for row in rows for value in row.split(",")):
        raise RuntimeError("CSV is not binary")
    original = Image.open(out / "candidate.png")
    enlarged = Image.open(out / "candidate_8x.png")
    if original.size != (241, 61) or enlarged.size != (1928, 488):
        raise RuntimeError("PNG dimensions are invalid")
    if set(original.convert("L").getdata()) != {0, 255}:
        raise RuntimeError("original PNG is not strictly black and white")
    if set(enlarged.convert("L").getdata()) != {0, 255}:
        raise RuntimeError("enlarged PNG is not strictly black and white")
    expected = np.repeat(np.repeat(np.asarray(original), 8, axis=0), 8, axis=1)
    if not np.array_equal(expected, np.asarray(enlarged)):
        raise RuntimeError("enlarged PNG is not nearest-neighbor equivalent")
    if int(matrix.size) != 14701:
        raise RuntimeError("wrong number of matrix values")


if __name__ == "__main__":
    matrix = build_matrix()
    write_outputs(matrix)
    validate_outputs(matrix)
    print("validated 61x241 matrix and both PNGs")
