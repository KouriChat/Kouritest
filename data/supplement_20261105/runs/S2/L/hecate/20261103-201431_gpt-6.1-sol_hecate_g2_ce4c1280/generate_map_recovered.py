from pathlib import Path
import csv

import numpy as np
from PIL import Image


# Approximate land outlines in (longitude, latitude), WGS84 degrees.
# These hand-written polygons are intended for this sampling window only.
POLYGONS = [
    # Mainland coast north and east of Hecate Strait
    [
        (-131.20, 55.15), (-130.95, 55.05), (-130.72, 54.92),
        (-130.58, 54.78), (-130.44, 54.67), (-130.32, 54.53),
        (-130.17, 54.37), (-130.03, 54.21), (-129.93, 54.07),
        (-129.78, 53.91), (-129.68, 53.74), (-129.58, 53.55),
        (-129.45, 53.33), (-129.32, 53.08), (-129.23, 52.90),
        (-128.95, 52.88), (-128.95, 55.15),
    ],
    # Graham Island, the northern and larger part of Haida Gwaii
    [
        (-131.12, 54.98), (-130.92, 55.03), (-130.72, 54.98),
        (-130.56, 54.89), (-130.49, 54.75), (-130.54, 54.62),
        (-130.62, 54.50), (-130.67, 54.35), (-130.71, 54.20),
        (-130.78, 54.08), (-130.90, 54.02), (-131.04, 54.06),
        (-131.15, 54.19), (-131.20, 54.36), (-131.23, 54.54),
        (-131.20, 54.72),
    ],
    # Moresby Island, separated from Graham Island by a narrow channel
    [
        (-131.17, 53.98), (-131.03, 54.00), (-130.91, 53.94),
        (-130.85, 53.82), (-130.83, 53.67), (-130.84, 53.50),
        (-130.90, 53.32), (-130.96, 53.14), (-131.04, 53.02),
        (-131.18, 52.98), (-131.26, 53.12), (-131.28, 53.31),
        (-131.27, 53.52), (-131.25, 53.72), (-131.22, 53.88),
    ],
    # Small islands and reefs around southern Moresby and Hecate Strait
    [
        (-130.70, 53.35), (-130.63, 53.39), (-130.58, 53.34),
        (-130.60, 53.27), (-130.67, 53.27),
    ],
    [
        (-130.52, 53.61), (-130.47, 53.64), (-130.43, 53.60),
        (-130.45, 53.55), (-130.50, 53.56),
    ],
    [
        (-130.33, 54.02), (-130.28, 54.05), (-130.23, 54.01),
        (-130.25, 53.96), (-130.31, 53.97),
    ],
    [
        (-130.12, 54.31), (-130.08, 54.34), (-130.04, 54.31),
        (-130.06, 54.27), (-130.11, 54.27),
    ],
]


def point_in_polygon(x, y, polygon):
    """Return whether (x, y) is inside a polygon, using ray crossing."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            crossing_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < crossing_x:
                inside = not inside
        j = i
    return inside


def main():
    rows = 127
    cols = 127
    step = 2.0 / 126.0

    # Rows run north to south; columns run west to east.
    latitudes = [55.0 - r * step for r in range(rows)]
    longitudes = [-131.0 + c * step for c in range(cols)]

    matrix = np.zeros((rows, cols), dtype=np.uint8)
    for r, lat in enumerate(latitudes):
        for c, lon in enumerate(longitudes):
            if any(point_in_polygon(lon, lat, polygon) for polygon in POLYGONS):
                matrix[r, c] = 1

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "candidate.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Land is white (255), water is black (0).
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(output_dir / "candidate.png")
    image.resize((cols * 8, rows * 8), resample=Image.Resampling.NEAREST).save(
        output_dir / "candidate_8x.png"
    )


if __name__ == "__main__":
    main()
