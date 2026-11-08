"""Knowledge-based Hecate Strait raster reconstruction.

All geographic shapes in this file are hand-written approximations.  No
external geographic data or online service is consulted.
"""
from pathlib import Path
import csv

import numpy as np
from PIL import Image

N = 127
LAT_NORTH = 55.0
LAT_SOUTH = 53.0
LON_WEST = -131.0
LON_EAST = -129.0
STEP = 2.0 / 126.0


def inside(point, polygon):
    """Return whether (lon, lat) lies inside a polygon."""
    x, y = point
    hit = False
    j = len(polygon) - 1
    for i, (px, py) in enumerate(polygon):
        qx, qy = polygon[j]
        crosses = ((py > y) != (qy > y))
        if crosses:
            at_x = (qx - px) * (y - py) / (qy - py) + px
            if x < at_x:
                hit = not hit
        j = i
    return hit


def polygon(coords):
    return [(float(lon), float(lat)) for lon, lat in coords]


# The eastern mainland coast is represented by the left edge of this
# polygon; the rest of the polygon extends to the eastern edge of the box.
MAINLAND = polygon([
    (-129.82, 55.00), (-129.76, 54.88), (-129.70, 54.76),
    (-129.62, 54.66), (-129.70, 54.56), (-129.58, 54.44),
    (-129.63, 54.34), (-129.54, 54.22), (-129.57, 54.10),
    (-129.48, 53.96), (-129.55, 53.83), (-129.51, 53.70),
    (-129.63, 53.58), (-129.58, 53.44), (-129.50, 53.30),
    (-129.45, 53.15), (-129.42, 53.00),
    (-129.00, 53.00), (-129.00, 55.00),
])

# The eastern fringe of Graham Island and the northern edge of Moresby
# Island reach the western boundary of this particular 2-degree frame.
GRAHAM_FRINGE = polygon([
    (-131.00, 54.73), (-130.94, 54.65), (-130.90, 54.54),
    (-130.93, 54.43), (-130.86, 54.33), (-130.89, 54.22),
    (-130.84, 54.10), (-130.90, 54.00), (-130.98, 53.91),
    (-131.00, 53.87), (-131.00, 54.73),
])
MORESBY_FRINGE = polygon([
    (-131.00, 53.48), (-130.96, 53.40), (-130.93, 53.31),
    (-130.96, 53.18), (-131.00, 53.10), (-131.00, 53.48),
])

# Larger islands in Hecate Strait and its northern approaches.
BANKS_ISLAND = polygon([
    (-130.48, 54.54), (-130.28, 54.48), (-130.12, 54.35),
    (-130.03, 54.17), (-130.08, 54.00), (-130.24, 53.91),
    (-130.42, 53.95), (-130.56, 54.08), (-130.60, 54.28),
    (-130.56, 54.45),
])
DUNDAS_ISLAND = polygon([
    (-130.70, 55.00), (-130.42, 55.00), (-130.28, 54.94),
    (-130.27, 54.82), (-130.40, 54.75), (-130.62, 54.78),
    (-130.77, 54.89),
])
ZAYAS_ISLAND = polygon([
    (-130.02, 54.98), (-129.82, 54.99), (-129.77, 54.92),
    (-129.86, 54.84), (-130.04, 54.86), (-130.10, 54.93),
])

LAND_POLYGONS = [
    MAINLAND, GRAHAM_FRINGE, MORESBY_FRINGE,
    BANKS_ISLAND, DUNDAS_ISLAND, ZAYAS_ISLAND,
]
# Southern and central island groups, written as separate polygons so that
# narrow channels remain water under the point-based rule.
PORCHER_ISLAND = polygon([
    (-130.64, 54.08), (-130.48, 54.02), (-130.34, 53.95),
    (-130.17, 53.96), (-130.04, 53.86), (-130.10, 53.72),
    (-130.26, 53.65), (-130.43, 53.68), (-130.59, 53.79),
    (-130.69, 53.94),
])
PITT_ISLAND = polygon([
    (-130.25, 53.73), (-130.10, 53.65), (-129.96, 53.52),
    (-129.86, 53.37), (-129.86, 53.22), (-130.00, 53.12),
    (-130.19, 53.18), (-130.31, 53.34), (-130.37, 53.51),
    (-130.34, 53.66),
])
GIL_ISLAND = polygon([
    (-129.77, 53.83), (-129.57, 53.79), (-129.43, 53.67),
    (-129.41, 53.50), (-129.51, 53.37), (-129.68, 53.31),
    (-129.80, 53.42), (-129.85, 53.59),
])
CAMPANIA_ISLAND = polygon([
    (-129.74, 53.31), (-129.55, 53.25), (-129.44, 53.13),
    (-129.45, 53.00), (-129.76, 53.00), (-129.83, 53.13),
])
PENROSE_ISLAND = polygon([
    (-130.82, 53.63), (-130.70, 53.58), (-130.64, 53.47),
    (-130.69, 53.38), (-130.82, 53.36), (-130.91, 53.45),
    (-130.90, 53.56),
])
DON_ISLAND = polygon([
    (-130.48, 53.38), (-130.36, 53.33), (-130.31, 53.24),
    (-130.38, 53.16), (-130.50, 53.17), (-130.58, 53.26),
])
PORCHER_NORTH = polygon([
    (-130.84, 54.18), (-130.70, 54.13), (-130.64, 54.04),
    (-130.72, 53.98), (-130.85, 54.03), (-130.91, 54.11),
])
LITTLE_ISLANDS = [
    polygon([(-130.85, 54.70), (-130.77, 54.72), (-130.73, 54.67),
             (-130.78, 54.62), (-130.86, 54.64)]),
    polygon([(-130.66, 54.60), (-130.57, 54.62), (-130.54, 54.56),
             (-130.59, 54.52), (-130.67, 54.54)]),
    polygon([(-130.20, 54.72), (-130.12, 54.75), (-130.07, 54.69),
             (-130.13, 54.64), (-130.21, 54.66)]),
    polygon([(-129.94, 54.36), (-129.84, 54.40), (-129.80, 54.34),
             (-129.85, 54.28), (-129.94, 54.30)]),
    polygon([(-130.91, 53.87), (-130.84, 53.90), (-130.81, 53.85),
             (-130.85, 53.80), (-130.92, 53.82)]),
    polygon([(-130.02, 53.87), (-129.92, 53.90), (-129.87, 53.84),
             (-129.93, 53.79), (-130.02, 53.81)]),
    polygon([(-129.29, 53.74), (-129.20, 53.77), (-129.16, 53.70),
             (-129.21, 53.64), (-129.29, 53.67)]),
]

LAND_POLYGONS.extend([
    PORCHER_ISLAND, PITT_ISLAND, GIL_ISLAND, CAMPANIA_ISLAND,
    PENROSE_ISLAND, DON_ISLAND, PORCHER_NORTH,
])
LAND_POLYGONS.extend(LITTLE_ISLANDS)

# Hand-placed narrow inlets and lakes.  They are subtracted after the
# land test so a point inside one of these remains water.
WATER_POLYGONS = [
    polygon([(-129.67, 54.89), (-129.52, 54.82), (-129.46, 54.70),
             (-129.51, 54.61), (-129.63, 54.68), (-129.72, 54.80)]),
    polygon([(-129.63, 54.26), (-129.49, 54.20), (-129.43, 54.08),
             (-129.50, 54.00), (-129.60, 54.08), (-129.68, 54.18)]),
    polygon([(-129.60, 53.73), (-129.47, 53.65), (-129.42, 53.54),
             (-129.50, 53.47), (-129.61, 53.54), (-129.68, 53.65)]),
    polygon([(-130.40, 54.28), (-130.27, 54.25), (-130.21, 54.13),
             (-130.26, 54.06), (-130.38, 54.13), (-130.44, 54.21)]),
]
def is_land(lon, lat):
    land = any(inside((lon, lat), shape) for shape in LAND_POLYGONS)
    if not land:
        return 0
    if any(inside((lon, lat), shape) for shape in WATER_POLYGONS):
        return 0
    return 1


def build_matrix():
    result = np.zeros((N, N), dtype=np.uint8)
    for row in range(N):
        lat = LAT_NORTH - row * STEP
        for col in range(N):
            lon = LON_WEST + col * STEP
            result[row, col] = is_land(lon, lat)
    return result


def write_outputs(matrix):
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "candidate.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # L mode gives exactly two grayscale values: 0 for water and 255 for land.
    raw = (matrix * 255).astype(np.uint8)
    original = Image.fromarray(raw, mode="L")
    original.save(output_dir / "candidate.png")
    enlarged = original.resize((N * 8, N * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")


def validate(matrix):
    assert matrix.shape == (127, 127)
    assert matrix.size == 16129
    assert set(np.unique(matrix).tolist()) <= {0, 1}

    csv_path = Path("outputs/candidate.csv")
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 127
    for line in lines:
        values = line.split(",")
        assert len(values) == 127
        assert all(value in {"0", "1"} for value in values)

    with Image.open("outputs/candidate.png") as image:
        assert image.size == (127, 127)
        assert set(image.convert("L").getdata()) <= {0, 255}
    with Image.open("outputs/candidate_8x.png") as image:
        assert image.size == (1016, 1016)
        assert set(image.convert("L").getdata()) <= {0, 255}


if __name__ == "__main__":
    matrix = build_matrix()
    write_outputs(matrix)
    validate(matrix)
    print("validated", matrix.shape, int(matrix.sum()), "land samples")
