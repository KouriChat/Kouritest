"""Knowledge-only reconstruction of the requested Arctic map.

The geometry below is deliberately hand-authored: it is a compact set of
coastline polygons based on general geographic knowledge, not downloaded
cartography.  Coordinates are longitude, latitude in decimal degrees.
"""
from pathlib import Path
import csv

import numpy as np
from PIL import Image

LON0, LON1, DLON = -105.0, -81.0, 0.1
LAT0, LAT1, DLAT = 79.0, 73.0, -0.1
NROW, NCOL = 61, 241


def inside(x, y, poly):
    """Ray-crossing point test, with boundary treated as land."""
    hit = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cross = ((yi > y) != (yj > y))
        if cross:
            xcross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x <= xcross:
                hit = not hit
        # Explicitly retain coastline points rather than leaving ambiguity.
        dx1, dy1 = x - xi, y - yi
        dx2, dy2 = x - xj, y - yj
        if abs(dx1 * dy2 - dy1 * dx2) < 1e-10 and dx1 * dx2 + dy1 * dy2 <= 0:
            return True
        j = i
    return hit


def polygon(*points):
    return tuple(points)


# Hand-estimated outlines of islands and mainland portions intersecting the
# rectangle.  Narrow channels are intentionally represented as water by using
# separate island polygons instead of one enclosing polygon.
LAND = []


def add(*points):
    LAND.append(polygon(*points))

# Southern islands and peninsulas
add(
    (-105.0, 73.0), (-105.0, 74.15), (-104.2, 74.75),
    (-102.2, 75.25), (-100.2, 75.15), (-99.0, 74.35),
    (-99.3, 73.55), (-100.0, 73.0),
)
add(
    (-103.0, 73.0), (-102.5, 74.15), (-100.7, 75.15),
    (-98.0, 75.85), (-95.0, 76.15), (-93.0, 75.75),
    (-91.7, 74.65), (-90.0, 73.75), (-89.3, 73.0),
)
add(
    (-95.2, 73.0), (-95.0, 73.9), (-94.3, 74.8),
    (-92.7, 75.35), (-89.7, 75.25), (-87.2, 74.55),
    (-85.3, 73.55), (-85.2, 73.0),
)
# Northern tip of Boothia Peninsula
add(
    (-95.0, 73.0), (-94.0, 73.45), (-92.0, 73.7),
    (-89.0, 73.65), (-87.0, 73.25), (-87.0, 73.0),
)
# Western edge of Baffin Island in the eastern margin of the frame
add(
    (-81.0, 73.0), (-81.0, 76.15), (-82.1, 76.25),
    (-83.1, 75.55), (-83.0, 74.7), (-82.0, 73.75),
)

# Melville Island, including its eastern end
add(
    (-105.0, 74.05), (-104.0, 75.2), (-102.1, 76.25),
    (-99.7, 76.8), (-97.0, 76.7), (-94.4, 76.05),
    (-94.8, 75.15), (-96.7, 74.45), (-99.0, 73.9),
    (-102.2, 73.65),
)

# Bathurst Island and the nearby island groups
add(
    (-100.1, 75.35), (-99.3, 76.25), (-97.0, 76.6),
    (-94.5, 76.4), (-93.0, 75.85), (-94.0, 75.25),
    (-96.0, 75.0), (-98.4, 75.05),
)
add(
    (-99.0, 75.0), (-98.2, 75.45), (-96.8, 75.8),
    (-95.4, 75.55), (-95.0, 75.0),
)

# Cornwallis Island
add(
    (-96.2, 74.85), (-95.8, 75.85), (-94.5, 76.25),
    (-92.3, 76.15), (-90.7, 75.65), (-90.8, 74.95),
    (-92.5, 74.65), (-94.5, 74.55),
)

# Devon Island (the broad eastern island in the central/northern frame)
add(
    (-94.1, 76.15), (-94.4, 77.05), (-92.5, 77.75),
    (-89.5, 78.2), (-85.2, 78.25), (-82.0, 77.7),
    (-80.7, 76.7), (-81.2, 75.55), (-83.0, 74.95),
    (-86.3, 74.75), (-89.6, 75.15), (-91.8, 75.65),
)
# Ellesmere Island: only its southwestern and central portion is needed here.
add(
    (-91.2, 77.05), (-92.5, 77.7), (-92.2, 78.45),
    (-90.8, 79.0), (-81.0, 79.0), (-80.8, 78.15),
    (-81.0, 77.2), (-81.6, 76.45), (-83.1, 75.95),
    (-85.3, 76.05), (-87.1, 76.55), (-88.0, 76.95),
)

# Axel Heiberg Island
add(
    (-97.4, 77.15), (-97.1, 78.05), (-95.5, 78.8),
    (-93.8, 79.0), (-89.0, 79.0), (-88.2, 78.35),
    (-89.3, 77.6), (-91.4, 77.05), (-94.2, 76.85),
)

# Ellef Ringnes Island
add(
    (-103.5, 77.05), (-102.2, 78.0), (-100.1, 78.75),
    (-98.0, 79.0), (-96.0, 79.0), (-95.8, 78.0),
    (-97.0, 77.25), (-99.3, 76.9), (-101.5, 76.9),
)

# Amund Ringnes Island
add(
    (-99.2, 77.05), (-98.3, 77.85), (-96.2, 78.55),
    (-93.7, 78.45), (-92.0, 77.85), (-93.0, 77.2),
    (-95.2, 76.95), (-97.4, 76.85),
)

# Cornwall Island and the small islands around Norwegian Bay
add(
    (-96.2, 76.55), (-95.3, 77.0), (-93.8, 77.0),
    (-92.8, 76.6), (-93.4, 76.25), (-95.1, 76.25),
)
add(
    (-91.3, 77.25), (-90.5, 77.75), (-89.1, 77.7),
    (-88.7, 77.2), (-89.5, 76.85), (-90.7, 76.9),
)

# Lougheed and King Christian islands (small, but resolvable at this scale)
add(
    (-98.4, 77.0), (-97.6, 77.5), (-96.5, 77.55),
    (-96.0, 77.05), (-96.5, 76.75), (-97.7, 76.75),
)
add(
    (-101.3, 77.0), (-100.8, 77.45), (-99.6, 77.5),
    (-99.1, 77.15), (-99.5, 76.85), (-100.7, 76.8),
)

def land_at(lon, lat):
    # Union of the separately described islands; all unlisted locations are
    # water, including channels and the marine ice cover.
    return any(inside(lon, lat, p) for p in LAND)


def build_matrix():
    matrix = np.zeros((NROW, NCOL), dtype=np.uint8)
    for r in range(NROW):
        lat = LAT0 + r * DLAT
        for c in range(NCOL):
            lon = LON0 + c * DLON
            matrix[r, c] = 1 if land_at(lon, lat) else 0
    return matrix


def write_outputs(matrix):
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)

    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # 0 is water/black and 1 is land/white.
    raw = Image.fromarray((matrix * 255).astype(np.uint8), mode="L")
    raw.save(out / "candidate.png")
    enlarged = raw.resize((NCOL * 8, NROW * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(out / "candidate_8x.png")


def validate(matrix):
    assert matrix.shape == (61, 241)
    assert matrix.size == 14701
    assert set(np.unique(matrix)).issubset({0, 1})
    with Path("outputs/candidate.csv").open(encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 61 and all(len(row) == 241 for row in rows)
    assert all(v in {"0", "1"} for row in rows for v in row)
    raw = Image.open("outputs/candidate.png")
    big = Image.open("outputs/candidate_8x.png")
    assert raw.size == (241, 61) and big.size == (1928, 488)
    assert set(raw.getdata()).issubset({0, 255})
    assert raw.convert("L").resize(big.size, Image.Resampling.NEAREST).tobytes() == big.convert("L").tobytes()


if __name__ == "__main__":
    result = build_matrix()
    write_outputs(result)
    validate(result)
    print("validated", result.shape, int(result.sum()), "land samples")
