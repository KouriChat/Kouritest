#!/usr/bin/env python3
"""Create an approximate Hecate Strait land/water raster from hand-written rules."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image


SIZE = 127
STEP = 2.0 / 126.0
OUT_DIR = Path("outputs")

# Approximate west shoreline of the mainland, listed south to north as
# (latitude, longitude). Mainland points east of this interpolated boundary
# are treated as land.
MAINLAND_WEST_COAST = [
    (53.00, -130.05),
    (53.15, -129.96),
    (53.30, -129.82),
    (53.45, -129.72),
    (53.60, -129.78),
    (53.75, -129.90),
    (53.90, -130.00),
    (54.05, -130.12),
    (54.20, -130.20),
    (54.35, -130.29),
    (54.50, -130.36),
    (54.65, -130.27),
    (54.80, -130.22),
    (55.00, -130.43),
]

# Approximate island outlines in (longitude, latitude). These deliberately
# capture only the principal landforms and some larger islands and reefs.
ISLANDS = [
    # Eastern Graham Island, clipped by the western edge of the sampling box.
    [
        (-131.00, 53.34), (-131.04, 53.47), (-131.12, 53.58),
        (-131.08, 53.72), (-131.02, 53.86), (-131.04, 54.02),
        (-131.15, 54.13), (-131.34, 54.20), (-131.55, 54.19),
        (-131.73, 54.12), (-131.88, 53.99), (-131.96, 53.83),
        (-131.88, 53.70), (-131.74, 53.61), (-131.67, 53.47),
        (-131.54, 53.35), (-131.34, 53.29), (-131.16, 53.30),
    ],
    # Northern Moresby Island and the Graham-Moresby vicinity.
    [
        (-131.02, 53.26), (-131.12, 53.33), (-131.25, 53.35),
        (-131.40, 53.31), (-131.51, 53.24), (-131.57, 53.14),
        (-131.55, 53.03), (-131.43, 53.00), (-131.30, 53.03),
        (-131.19, 53.10), (-131.08, 53.17),
    ],
    # Banks Island, mostly east of the strait and partly clipped at 129 W.
    [
        (-129.50, 53.43), (-129.40, 53.52), (-129.28, 53.67),
        (-129.15, 53.83), (-129.00, 53.91), (-129.00, 53.46),
        (-129.12, 53.39), (-129.28, 53.35), (-129.42, 53.36),
    ],
    # Pitt Island, with its western shore facing Hecate Strait.
    [
        (-130.00, 53.23), (-129.88, 53.30), (-129.77, 53.43),
        (-129.69, 53.59), (-129.62, 53.76), (-129.51, 53.83),
        (-129.44, 53.75), (-129.48, 53.59), (-129.57, 53.43),
        (-129.68, 53.29), (-129.81, 53.18), (-129.95, 53.14),
    ],
    # Small islands in the eastern strait and near the island groups.
    [
        (-130.34, 53.78), (-130.29, 53.84), (-130.20, 53.86),
        (-130.15, 53.81), (-130.19, 53.75), (-130.29, 53.74),
    ],
    [
        (-130.67, 53.88), (-130.62, 53.94), (-130.54, 53.95),
        (-130.49, 53.90), (-130.54, 53.85), (-130.62, 53.84),
    ],
    [
        (-130.88, 54.04), (-130.84, 54.09), (-130.77, 54.10),
        (-130.73, 54.05), (-130.77, 54.00), (-130.84, 54.00),
    ],
    [
        (-130.93, 53.53), (-130.88, 53.58), (-130.81, 53.57),
        (-130.79, 53.52), (-130.84, 53.48), (-130.90, 53.49),
    ],
    [
        (-130.14, 54.20), (-130.08, 54.24), (-130.02, 54.21),
        (-130.02, 54.16), (-130.08, 54.14), (-130.13, 54.16),
    ],
    [
        (-129.82, 54.03), (-129.76, 54.08), (-129.69, 54.06),
        (-129.68, 54.00), (-129.74, 53.97), (-129.80, 53.99),
    ],
]


def mainland_west_boundary(latitude):
    """Linearly interpolate the hand-drawn mainland shoreline."""
    for (lat0, lon0), (lat1, lon1) in zip(
        MAINLAND_WEST_COAST, MAINLAND_WEST_COAST[1:]
    ):
        if lat0 <= latitude <= lat1:
            fraction = (latitude - lat0) / (lat1 - lat0)
            return lon0 + fraction * (lon1 - lon0)
    return None


def point_in_polygon(longitude, latitude, polygon):
    """Return whether a point lies inside a polygon using ray crossing."""
    inside = False
    count = len(polygon)
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]

        if (y1 > latitude) != (y2 > latitude):
            crossing_x = x1 + (latitude - y1) * (x2 - x1) / (y2 - y1)
            if longitude < crossing_x:
                inside = not inside
    return inside


def is_land(longitude, latitude):
    boundary = mainland_west_boundary(latitude)
    if boundary is not None and longitude >= boundary:
        return True

    return any(
        point_in_polygon(longitude, latitude, polygon)
        for polygon in ISLANDS
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Rows run north to south; columns run west to east.
    matrix = np.zeros((SIZE, SIZE), dtype=np.uint8)
    for row in range(SIZE):
        latitude = 55.0 - row * STEP
        for column in range(SIZE):
            longitude = -131.0 + column * STEP
            matrix[row, column] = int(is_land(longitude, latitude))

    csv_path = OUT_DIR / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Palette values are exactly black for water and white for land.
    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(OUT_DIR / "candidate.png")
    enlarged = image.resize(
        (SIZE * 8, SIZE * 8),
        resample=Image.Resampling.NEAREST,
    )
    enlarged.save(OUT_DIR / "candidate_8x.png")

    # Check CSV dimensions and values, and verify both image products.
    with csv_path.open("r", newline="", encoding="ascii") as csv_file:
        rows = list(csv.reader(csv_file))
    assert len(rows) == SIZE
    assert all(len(row) == SIZE for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert sum(map(len, rows)) == 16129

    original = Image.open(OUT_DIR / "candidate.png")
    zoomed = Image.open(OUT_DIR / "candidate_8x.png")
    assert original.size == (127, 127)
    assert zoomed.size == (1016, 1016)
    assert set(original.getdata()) <= {(0, 0, 0), (255, 255, 255)}
    assert set(zoomed.getdata()) <= {(0, 0, 0), (255, 255, 255)}
    assert zoomed == original.resize((1016, 1016), Image.Resampling.NEAREST)

    print(f"Wrote {csv_path}, {OUT_DIR / 'candidate.png'}, "
          f"and {OUT_DIR / 'candidate_8x.png'}")
    print("Validated 127 x 127 values and nearest-neighbor 8x image.")


if __name__ == "__main__":
    main()
