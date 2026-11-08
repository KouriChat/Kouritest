import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image


# All coordinates below are hand-drawn longitude/latitude vertices.  They are
# deliberately a compact approximation, not data read from a map service.
LAND = []


def add(points):
    LAND.append([(float(x), float(y)) for x, y in points])


def inside(x, y, poly):
    """Even-odd point-in-polygon test for one hand-written coastline."""
    hit = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        crosses = ((yi > y) != (yj > y))
        if crosses:
            edge_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < edge_x:
                hit = not hit
        j = i
    return hit


# Victoria Island (only its eastern part falls in the requested window).
add([
    (-105.0, 73.05), (-103.5, 72.95), (-102.0, 73.35),
    (-100.6, 74.05), (-99.8, 74.9), (-100.1, 75.8),
    (-101.4, 76.35), (-103.0, 76.75), (-105.0, 76.55),
])

# Prince of Wales Island, including its broad northern and central body.
add([
    (-105.0, 74.2), (-103.4, 74.0), (-102.0, 74.25),
    (-100.8, 74.7), (-99.0, 74.55), (-97.3, 75.15),
    (-96.7, 76.0), (-97.9, 76.35), (-99.6, 76.15),
    (-101.4, 75.7), (-103.2, 75.25), (-105.0, 75.25),
])

# Melville Island, with its irregular south coast.
add([
    (-105.0, 75.3), (-103.5, 75.15), (-102.0, 74.8),
    (-100.3, 74.55), (-98.5, 74.55), (-97.0, 75.15),
    (-95.8, 76.0), (-95.45, 77.0), (-96.2, 77.9),
    (-98.0, 78.45), (-100.3, 78.7), (-102.5, 78.45),
    (-104.3, 77.75), (-105.0, 77.3),
])

# Bathurst Island, separated from Melville by narrow channels.
add([
    (-103.6, 75.45), (-102.3, 75.35), (-101.1, 75.65),
    (-100.2, 76.35), (-100.25, 77.35), (-101.1, 78.05),
    (-102.35, 78.45), (-103.45, 78.15), (-104.0, 77.35),
    (-103.7, 76.4),
])
# Cornwallis Island and the small islands around its western end.
add([
    (-96.8, 75.0), (-96.0, 75.55), (-95.6, 76.25),
    (-94.6, 76.55), (-93.3, 76.35), (-92.5, 75.75),
    (-92.7, 75.05), (-93.7, 74.65), (-95.1, 74.65),
    (-96.2, 74.8),
])

# Somerset Island, south of the Parry Channel.
add([
    (-96.7, 73.05), (-95.4, 73.25), (-94.1, 73.75),
    (-93.0, 74.55), (-91.8, 74.7), (-90.7, 74.35),
    (-89.6, 73.8), (-89.3, 73.0), (-90.5, 73.0),
    (-92.0, 73.15), (-93.7, 72.95), (-95.4, 72.85),
])

# The northern tip of the Boothia Peninsula at the bottom of the frame.
add([
    (-96.4, 73.0), (-95.2, 72.9), (-94.1, 72.8),
    (-92.9, 72.95), (-91.8, 73.25), (-90.8, 73.55),
    (-89.8, 73.5), (-89.5, 73.0),
])

# Devon Island: a large, high island occupying the northeastern part.
add([
    (-97.1, 75.35), (-96.3, 76.25), (-95.2, 77.2),
    (-94.5, 78.15), (-92.8, 78.9), (-90.3, 79.15),
    (-87.8, 78.95), (-85.7, 78.45), (-83.8, 77.75),
    (-82.0, 76.8), (-81.0, 75.65), (-81.0, 74.95),
    (-82.5, 74.55), (-84.8, 74.35), (-87.0, 74.25),
    (-89.3, 74.35), (-91.5, 74.55), (-93.5, 74.8),
    (-95.2, 75.05),
])

# Cornwallis' northern neighbour, Ellesmere's western/southern edge is
# represented only where it reaches the longitude window.
add([
    (-82.1, 79.0), (-81.0, 79.0), (-81.0, 78.2),
    (-81.5, 77.7), (-82.3, 77.55), (-83.0, 78.1),
    (-83.0, 78.7),
])
# Axel Heiberg Island, north of the channel west of Devon.
add([
    (-95.0, 78.1), (-94.0, 78.85), (-92.8, 79.25),
    (-90.8, 79.45), (-89.2, 79.05), (-88.1, 78.35),
    (-88.4, 77.65), (-89.8, 77.25), (-91.4, 77.35),
    (-93.0, 77.7), (-94.4, 77.65),
])

# Ellef Ringnes Island, with its western part near the top of the box.
add([
    (-101.0, 78.25), (-100.0, 78.85), (-98.7, 79.2),
    (-97.2, 79.15), (-96.0, 78.65), (-96.1, 78.0),
    (-97.1, 77.65), (-98.6, 77.75), (-100.0, 77.9),
])

# Amund Ringnes Island and Cornwall Island (small high-latitude islands).
add([
    (-97.0, 78.45), (-96.3, 79.05), (-95.0, 79.35),
    (-93.8, 79.2), (-93.3, 78.65), (-94.0, 78.15),
    (-95.4, 78.0), (-96.4, 78.1),
])
add([
    (-97.2, 78.85), (-96.4, 79.3), (-95.4, 79.35),
    (-94.8, 79.0), (-95.2, 78.65), (-96.3, 78.55),
])

# A few sizeable islands in the channels, kept as independent polygons so
# that the surrounding water remains zero.
add([
    (-104.8, 76.0), (-104.0, 76.35), (-103.2, 76.15),
    (-102.9, 75.7), (-103.5, 75.4), (-104.5, 75.55),
])
add([
    (-99.0, 76.45), (-98.3, 76.75), (-97.5, 76.55),
    (-97.3, 76.05), (-98.0, 75.85), (-98.8, 76.0),
])
add([
    (-88.8, 75.1), (-88.0, 75.35), (-87.3, 75.15),
    (-87.1, 74.7), (-87.8, 74.45), (-88.6, 74.55),
])

def is_land(lon, lat):
    return any(inside(lon, lat, polygon) for polygon in LAND)


def main():
    # Decimal integer arithmetic avoids accumulated 0.1 degree drift.
    lats = [79.0 - i / 10.0 for i in range(61)]
    lons = [-105.0 + j / 10.0 for j in range(241)]
    matrix = np.array(
        [[1 if is_land(lon, lat) else 0 for lon in lons] for lat in lats],
        dtype=np.uint8,
    )

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # White is land (1), black is water (0); no filtering or interpolation.
    raw = Image.fromarray(matrix * 255, mode="L")
    raw.save(out / "candidate.png")
    enlarged = raw.resize((241 * 8, 61 * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(out / "candidate_8x.png")

    assert matrix.shape == (61, 241)
    assert matrix.size == 14701
    assert set(np.unique(matrix)).issubset({0, 1})
    assert raw.size == (241, 61)
    assert enlarged.size == (1928, 488)
    print("matrix", matrix.shape, "values", matrix.size)
    print("land points", int(matrix.sum()), "water points", int(matrix.size - matrix.sum()))


if __name__ == "__main__":
    main()
