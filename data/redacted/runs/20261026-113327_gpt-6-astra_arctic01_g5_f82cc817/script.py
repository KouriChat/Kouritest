from pathlib import Path
import csv

import numpy as np
from PIL import Image

# Approximate hand-written island outlines as (longitude, latitude) polygons.
LAND_POLYGONS = [
    # Ellef Ringnes Island
    [
        (-103.7, 78.2), (-103.1, 78.8), (-101.8, 79.2), (-100.4, 79.15),
        (-99.4, 78.7), (-99.3, 78.1), (-100.0, 77.7), (-101.2, 77.45),
        (-102.5, 77.55), (-103.4, 77.85),
    ],
    # Amund Ringnes Island
    [
        (-99.8, 78.35), (-99.1, 78.75), (-97.9, 78.8), (-96.7, 78.45),
        (-96.0, 77.9), (-96.5, 77.45), (-97.3, 77.25), (-98.4, 77.35),
        (-99.3, 77.75),
    ],
    # Bathurst Island, eastern edge
    [
        (-105.0, 76.45), (-104.6, 76.8), (-104.1, 76.75), (-103.9, 76.35),
        (-104.1, 75.9), (-104.7, 75.65), (-105.0, 75.8),
    ],
    # Cornwall Island
    [
        (-96.5, 75.55), (-96.0, 75.9), (-95.3, 75.85), (-94.8, 75.55),
        (-95.1, 75.2), (-95.8, 75.12), (-96.3, 75.3),
    ],
    # Axel Heiberg Island
    [
        (-93.3, 79.6), (-92.6, 80.25), (-91.5, 80.65), (-90.3, 80.55),
        (-89.5, 80.05), (-89.4, 79.3), (-90.0, 78.75), (-90.8, 78.35),
        (-91.7, 78.25), (-92.7, 78.55), (-93.2, 79.0),
    ],
    # Ellesmere Island
    [
        (-90.4, 79.25), (-90.0, 80.0), (-89.0, 80.7), (-87.9, 81.2),
        (-86.0, 81.6), (-83.0, 81.4), (-81.0, 81.1), (-81.0, 76.0),
        (-82.0, 75.9), (-83.0, 76.15), (-84.0, 76.35), (-85.0, 76.7),
        (-86.0, 77.1), (-87.2, 77.45), (-88.2, 77.85), (-89.2, 78.25),
        (-90.0, 78.65),
    ],
    # Devon Island
    [
        (-90.2, 76.5), (-89.7, 77.0), (-88.9, 77.45), (-87.7, 77.7),
        (-86.3, 77.65), (-85.0, 77.3), (-83.7, 76.8), (-82.8, 76.2),
        (-82.6, 75.4), (-83.1, 74.8), (-84.2, 74.45), (-85.5, 74.4),
        (-86.7, 74.65), (-88.0, 74.95), (-89.2, 75.45), (-89.9, 75.95),
    ],
    # Prince of Wales Island
    [
        (-99.7, 75.7), (-99.0, 76.0), (-97.7, 76.15), (-96.5, 76.0),
        (-95.2, 75.65), (-94.5, 75.1), (-94.4, 74.3), (-94.7, 73.6),
        (-95.4, 73.0), (-96.5, 72.75), (-97.6, 72.85), (-98.5, 73.25),
        (-99.2, 73.9), (-99.6, 74.7),
    ],
    # Somerset Island
    [
        (-95.8, 75.45), (-95.0, 75.65), (-94.2, 75.45), (-93.4, 75.0),
        (-92.6, 74.4), (-92.0, 73.7), (-91.8, 73.0), (-92.3, 72.5),
        (-93.2, 72.2), (-94.2, 72.35), (-95.0, 72.8), (-95.5, 73.5),
        (-95.9, 74.4),
    ],
    # Northern Baffin Island entering the southeast edge
    [
        (-84.2, 74.4), (-83.6, 74.0), (-83.2, 73.5), (-83.0, 73.0),
        (-81.0, 73.0), (-81.0, 75.0), (-82.0, 75.0), (-83.0, 74.8),
    ],
    # Northeastern Victoria Island, near the southwest frame edge
    [
        (-105.0, 73.2), (-104.4, 73.55), (-103.5, 73.75), (-102.8, 73.55),
        (-102.5, 73.15), (-102.8, 72.75), (-103.7, 72.55), (-104.6, 72.7),
        (-105.0, 72.9),
    ],
]


def point_in_polygon(lon, lat, polygon):
    """Ray-crossing point test; points exactly on polygon edges count as land."""
    inside = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        cross = (lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
        if abs(cross) < 1e-10:
            if (min(x1, x2) - 1e-10 <= lon <= max(x1, x2) + 1e-10
                    and min(y1, y2) - 1e-10 <= lat <= max(y1, y2) + 1e-10):
                return True
        if (y1 > lat) != (y2 > lat):
            if lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
    return inside


def build_matrix():
    matrix = np.zeros((61, 241), dtype=np.uint8)
    for row in range(61):
        lat = 79.0 - row * 0.1
        for column in range(241):
            lon = -105.0 + column * 0.1
            matrix[row, column] = int(
                any(point_in_polygon(lon, lat, polygon) for polygon in LAND_POLYGONS)
            )
    return matrix


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()

    with (output_dir / "candidate.csv").open(
        "w", newline="", encoding="ascii"
    ) as csv_file:
        csv.writer(csv_file, lineterminator="\n").writerows(matrix.tolist())

    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(output_dir / "candidate.png")
    image.resize((1928, 488), Image.Resampling.NEAREST).save(
        output_dir / "candidate_8x.png"
    )


if __name__ == "__main__":
    main()
