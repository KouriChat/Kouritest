"""Approximate Hecate Strait land mask from hand-written geographic rules."""
from pathlib import Path
import csv

import numpy as np
from PIL import Image


N = 127
LAT_NORTH, LAT_SOUTH = 55.0, 53.0
LON_WEST, LON_EAST = -131.0, -129.0
STEP = 2.0 / 126.0

# Approximate mainland shoreline, traced north to south. The mask takes
# mainland to the east of this coast; vertices include broad bays/headlands.
MAINLAND_COAST = [
    (-130.53, 55.00), (-130.50, 54.91), (-130.63, 54.84),
    (-130.58, 54.76), (-130.42, 54.70), (-130.37, 54.62),
    (-130.50, 54.55), (-130.39, 54.48), (-130.26, 54.41),
    (-130.30, 54.33), (-130.17, 54.26), (-130.24, 54.18),
    (-130.12, 54.09), (-130.06, 54.00), (-130.17, 53.91),
    (-130.09, 53.83), (-129.96, 53.75), (-129.99, 53.67),
    (-129.87, 53.58), (-129.79, 53.49), (-129.83, 53.40),
    (-129.68, 53.31), (-129.59, 53.22), (-129.56, 53.13),
    (-129.42, 53.06), (-129.36, 53.00),
]

# Major islands and island groups inside the requested frame. These are
# schematic polygons, deliberately limited to known geographic features.
ISLANDS = [
    # Dundas Island and nearby islands near the Dixon Entrance.
    [(-130.86, 54.68), (-130.77, 54.73), (-130.63, 54.70),
     (-130.57, 54.64), (-130.63, 54.57), (-130.77, 54.55),
     (-130.87, 54.60)],
    [(-130.72, 54.52), (-130.64, 54.54), (-130.57, 54.50),
     (-130.60, 54.44), (-130.69, 54.43)],
    # Banks Island, near the mainland side of the strait.
    [(-130.12, 53.96), (-130.04, 54.03), (-129.91, 54.01),
     (-129.86, 53.94), (-129.89, 53.82), (-129.97, 53.76),
     (-130.07, 53.79), (-130.13, 53.87)],
    # Aristazabal Island and adjacent island mass.
    [(-130.62, 53.94), (-130.48, 54.02), (-130.34, 54.00),
     (-130.28, 53.93), (-130.33, 53.84), (-130.28, 53.73),
     (-130.37, 53.66), (-130.52, 53.71), (-130.61, 53.82)],
    # Smaller reefs/islets represented as compact land polygons.
    [(-130.93, 53.74), (-130.88, 53.78), (-130.82, 53.74),
     (-130.85, 53.68), (-130.91, 53.68)],
    [(-130.57, 53.48), (-130.51, 53.53), (-130.44, 53.49),
     (-130.47, 53.42), (-130.54, 53.42)],
]
def in_polygon(lon, lat, polygon):
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > lat) != (yj > lat):
            crossing = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing:
                inside = not inside
        j = i
    return inside


def shoreline_lon(lat):
    for (x1, y1), (x2, y2) in zip(MAINLAND_COAST, MAINLAND_COAST[1:]):
        if min(y1, y2) <= lat <= max(y1, y2):
            if y1 == y2:
                return min(x1, x2)
            return x1 + (x2 - x1) * (lat - y1) / (y2 - y1)
    return MAINLAND_COAST[0][0] if lat > 55 else MAINLAND_COAST[-1][0]


def land_at(lon, lat):
    if lon >= shoreline_lon(lat):
        return True
    return any(in_polygon(lon, lat, island) for island in ISLANDS)


def main():
    lats = LAT_NORTH - np.arange(N) * STEP
    lons = LON_WEST + np.arange(N) * STEP
    matrix = np.zeros((N, N), dtype=np.uint8)
    for row, lat in enumerate(lats):
        for col, lon in enumerate(lons):
            matrix[row, col] = int(land_at(float(lon), float(lat)))

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Grayscale 255 is land (white); 0 is water (black).
    image = Image.fromarray(matrix * np.uint8(255), mode="L")
    image.save(out / "candidate.png")
    image.resize((N * 8, N * 8), Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )
    validate(matrix, out)


def validate(matrix, out):
    assert matrix.shape == (127, 127) and matrix.size == 16129
    with (out / "candidate.csv").open(newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 127 and all(len(row) == 127 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert set(Image.open(out / "candidate.png").getdata()) <= {0, 255}
    enlarged = Image.open(out / "candidate_8x.png")
    assert enlarged.size == (1016, 1016)
    assert enlarged.resize((127, 127), Image.Resampling.NEAREST).tobytes() == (
        Image.open(out / "candidate.png").tobytes()
    )


if __name__ == "__main__":
    main()
