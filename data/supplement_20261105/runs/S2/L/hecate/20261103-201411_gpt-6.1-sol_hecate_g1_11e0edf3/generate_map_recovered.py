from pathlib import Path
import csv

import numpy as np
from PIL import Image

# Hand-estimated land outlines for the Hecate Strait region.
# Coordinates are (longitude, latitude), with west longitudes negative.
# Polygons are intentionally approximate; water gaps between Graham and
# Moresby Islands, and the open straits, are left unfilled.
LAND_POLYGONS = [
    # Mainland coast, from the northwestern frame edge around Portland Inlet
    # and the mainland side of Dixon Entrance, then south along the coast.
    [
        (-131.00, 55.00), (-129.00, 55.00), (-129.00, 53.00),
        (-129.28, 53.00), (-129.34, 53.10), (-129.42, 53.20),
        (-129.48, 53.30), (-129.50, 53.42), (-129.55, 53.52),
        (-129.64, 53.63), (-129.72, 53.74), (-129.77, 53.86),
        (-129.84, 53.98), (-129.93, 54.10), (-130.02, 54.22),
        (-130.10, 54.34), (-130.17, 54.46), (-130.25, 54.58),
        (-130.35, 54.68), (-130.44, 54.78), (-130.55, 54.86),
        (-130.67, 54.91), (-130.78, 54.93), (-130.87, 54.91),
        (-130.94, 54.86), (-131.00, 54.82),
    ],
    # Mainland north of the coastal indentation, including the Portland Inlet
    # side of the frame.
    [
        (-131.00, 55.00), (-129.00, 55.00), (-129.00, 54.92),
        (-129.18, 54.92), (-129.28, 54.95), (-129.39, 54.98),
        (-129.50, 54.97), (-129.62, 54.94), (-129.73, 54.90),
        (-129.84, 54.88), (-129.95, 54.90), (-130.06, 54.94),
        (-130.18, 54.97), (-130.30, 54.99), (-130.42, 54.98),
        (-130.54, 54.95), (-130.65, 54.91), (-130.76, 54.88),
        (-130.88, 54.88), (-131.00, 54.91),
    ],
    # Graham Island, northern and western part of Haida Gwaii.
    [
        (-131.00, 54.72), (-130.96, 54.72), (-130.91, 54.68),
        (-130.88, 54.61), (-130.86, 54.54), (-130.82, 54.47),
        (-130.78, 54.39), (-130.74, 54.31), (-130.69, 54.23),
        (-130.64, 54.15), (-130.58, 54.08), (-130.51, 54.03),
        (-130.44, 54.00), (-130.37, 54.00), (-130.30, 54.03),
        (-130.25, 54.09), (-130.24, 54.17), (-130.27, 54.24),
        (-130.33, 54.29), (-130.40, 54.33), (-130.46, 54.39),
        (-130.49, 54.47), (-130.50, 54.55), (-130.54, 54.63),
        (-130.60, 54.69), (-130.68, 54.73), (-130.78, 54.75),
        (-130.89, 54.75), (-131.00, 54.74),
    ],
    # Moresby Island: separated from Graham Island by narrow waterways.
    [
        (-131.00, 53.83), (-130.93, 53.84), (-130.88, 53.81),
        (-130.84, 53.76), (-130.81, 53.69), (-130.77, 53.61),
        (-130.72, 53.54), (-130.67, 53.48), (-130.62, 53.42),
        (-130.57, 53.36), (-130.53, 53.29), (-130.51, 53.22),
        (-130.52, 53.15), (-130.57, 53.09), (-130.64, 53.05),
        (-130.73, 53.03), (-130.84, 53.03), (-130.94, 53.05),
        (-131.00, 53.08),
    ],
    # Small offshore island groups and larger islands; simplified outlines.
    [
        (-130.58, 54.92), (-130.54, 54.94), (-130.49, 54.93),
        (-130.46, 54.90), (-130.48, 54.87), (-130.53, 54.87),
        (-130.57, 54.89),
    ],
    [
        (-130.20, 54.80), (-130.16, 54.83), (-130.11, 54.82),
        (-130.09, 54.78), (-130.12, 54.75), (-130.17, 54.76),
    ],
    [
        (-130.06, 54.57), (-130.02, 54.60), (-129.98, 54.58),
        (-129.97, 54.54), (-130.00, 54.51), (-130.04, 54.53),
    ],
    [
        (-129.94, 54.36), (-129.90, 54.39), (-129.86, 54.37),
        (-129.85, 54.33), (-129.89, 54.31), (-129.93, 54.33),
    ],
    [
        (-130.02, 53.95), (-129.98, 53.98), (-129.94, 53.96),
        (-129.93, 53.92), (-129.97, 53.90), (-130.01, 53.92),
    ],
    [
        (-130.30, 53.70), (-130.25, 53.73), (-130.20, 53.71),
        (-130.19, 53.67), (-130.23, 53.64), (-130.28, 53.66),
    ],
    [
        (-129.82, 53.55), (-129.78, 53.58), (-129.74, 53.56),
        (-129.73, 53.52), (-129.77, 53.50), (-129.81, 53.52),
    ],
]


def point_on_segment(x, y, x1, y1, x2, y2, eps=1e-10):
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > eps:
        return False
    return (
        min(x1, x2) - eps <= x <= max(x1, x2) + eps
        and min(y1, y2) - eps <= y <= max(y1, y2) + eps
    )


def point_in_polygon(lon, lat, polygon):
    """Ray-crossing point test; polygon boundary is classified as land."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        if point_on_segment(lon, lat, x1, y1, x2, y2):
            return True

        if (y1 > lat) != (y2 > lat):
            crossing_lon = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < crossing_lon:
                inside = not inside
    return inside


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    size = 127
    step = 2.0 / 126.0
    latitudes = [55.0 - row * step for row in range(size)]
    longitudes = [-131.0 + col * step for col in range(size)]

    matrix = np.zeros((size, size), dtype=np.uint8)
    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            if any(point_in_polygon(lon, lat, polygon) for polygon in LAND_POLYGONS):
                matrix[row, col] = 1

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Land is white (255); water is black (0).
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(output_dir / "candidate.png")

    enlarged = image.resize((1016, 1016), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")

    # Validate dimensions, values, CSV structure, palette, and exact nearest-neighbor scaling.
    assert matrix.shape == (127, 127)
    assert int(matrix.size) == 16129
    assert set(np.unique(matrix).tolist()) <= {0, 1}

    with csv_path.open("r", newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 127
    assert all(len(row) == 127 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    reopened = Image.open(output_dir / "candidate.png")
    assert reopened.size == (127, 127)
    assert set(reopened.getdata()) <= {0, 255}

    reopened_large = Image.open(output_dir / "candidate_8x.png")
    assert reopened_large.size == (1016, 1016)
    expected_large = image.resize((1016, 1016), resample=Image.Resampling.NEAREST)
    assert list(reopened_large.getdata()) == list(expected_large.getdata())


if __name__ == "__main__":
    main()
