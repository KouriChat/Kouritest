from pathlib import Path
import csv

import numpy as np
from PIL import Image


# Hand-drawn approximate island outlines, in (longitude, latitude) degrees.
# These are the complete land rules used by this script; points inside any
# outline are land, and all other points are water.
ISLANDS = [
    # Bathurst Island
    [
        (-101.2, 76.2), (-100.8, 77.0), (-100.1, 77.7), (-99.3, 78.0),
        (-98.4, 77.8), (-97.6, 77.3), (-96.4, 76.8), (-96.2, 76.2),
        (-96.8, 75.7), (-97.8, 75.4), (-98.8, 75.5), (-100.0, 75.7),
        (-100.8, 75.9),
    ],
    # Cornwallis Island
    [
        (-96.5, 75.0), (-96.0, 75.3), (-95.6, 75.9), (-95.1, 76.4),
        (-94.4, 76.6), (-93.8, 76.3), (-93.5, 75.7), (-93.8, 75.1),
        (-94.5, 74.9), (-95.3, 74.8), (-96.0, 74.8),
    ],
    # Devon Island
    [
        (-90.7, 74.7), (-90.0, 75.2), (-89.8, 75.8), (-89.2, 76.4),
        (-88.3, 76.9), (-87.3, 77.3), (-86.0, 77.5), (-84.9, 77.2),
        (-83.8, 76.9), (-82.8, 76.4), (-83.0, 75.8), (-83.6, 75.3),
        (-84.6, 74.9), (-86.0, 74.6), (-87.7, 74.4), (-89.2, 74.4),
    ],
    # Somerset Island
    [
        (-95.5, 73.0), (-95.0, 73.8), (-94.3, 74.4), (-93.5, 74.8),
        (-92.5, 74.9), (-91.8, 74.5), (-91.7, 73.9), (-92.0, 73.2),
        (-92.6, 72.8), (-93.5, 72.7), (-94.5, 72.8),
    ],
    # Prince of Wales Island
    [
        (-103.1, 73.1), (-102.7, 73.9), (-102.3, 74.7), (-101.7, 75.4),
        (-100.8, 76.0), (-99.8, 76.4), (-98.9, 76.5), (-98.1, 76.1),
        (-97.7, 75.5), (-98.0, 74.8), (-98.4, 74.1), (-98.8, 73.5),
        (-99.4, 73.0), (-100.2, 72.8), (-101.2, 72.7), (-102.2, 72.8),
    ],
    # Southern Melville Island, near the western edge of the sampling area
    [
        (-105.0, 75.5), (-104.7, 76.0), (-104.6, 76.7), (-104.8, 77.3),
        (-105.2, 77.7), (-106.0, 77.8), (-106.3, 77.3), (-106.2, 76.6),
        (-105.8, 76.0),
    ],
    # Axel Heiberg Island, with its southern end entering the northern rows
    [
        (-92.2, 78.1), (-91.8, 78.6), (-91.0, 79.2), (-90.0, 79.6),
        (-88.9, 79.8), (-87.5, 79.6), (-86.3, 79.2), (-85.5, 78.7),
        (-85.7, 78.2), (-86.5, 77.9), (-87.8, 77.8), (-89.2, 77.9),
        (-90.5, 78.0), (-91.5, 77.9),
    ],
    # Lougheed Island
    [
        (-93.5, 77.1), (-93.0, 77.6), (-92.4, 77.9), (-91.8, 77.7),
        (-91.5, 77.2), (-92.0, 76.9), (-92.8, 76.9),
    ],
    # Amund Ringnes Island, clipped by the northern edge
    [
        (-95.0, 78.0), (-94.7, 78.5), (-93.8, 78.9), (-92.8, 79.2),
        (-91.9, 79.1), (-91.4, 78.7), (-91.9, 78.2), (-92.7, 78.0),
        (-93.8, 77.9),
    ],
    # Small approximation of the western Ellesmere coast at the eastern edge
    [
        (-82.0, 77.6), (-81.7, 78.2), (-81.4, 78.9), (-81.0, 79.4),
        (-80.7, 79.0), (-80.8, 78.3), (-81.1, 77.7),
    ],
]


def point_on_segment(x, y, x1, y1, x2, y2):
    """Treat points on an outline as land."""
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > 1e-10:
        return False
    return (
        min(x1, x2) - 1e-10 <= x <= max(x1, x2) + 1e-10
        and min(y1, y2) - 1e-10 <= y <= max(y1, y2) + 1e-10
    )


def point_in_polygon(x, y, polygon):
    """Ray-crossing point-in-polygon test with boundary included."""
    inside = False
    count = len(polygon)
    for i in range(count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]

        if point_on_segment(x, y, x1, y1, x2, y2):
            return True

        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing_x:
                inside = not inside
    return inside


latitudes = [79.0 - 0.1 * row for row in range(61)]
longitudes = [-105.0 + 0.1 * col for col in range(241)]

matrix = np.zeros((61, 241), dtype=np.uint8)
for row, latitude in enumerate(latitudes):
    for col, longitude in enumerate(longitudes):
        matrix[row, col] = int(
            any(point_in_polygon(longitude, latitude, island) for island in ISLANDS)
        )

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

csv_path = output_dir / "candidate.csv"
with csv_path.open("w", newline="", encoding="ascii") as csv_file:
    writer = csv.writer(csv_file, lineterminator="\n")
    writer.writerows(matrix.tolist())

# 1 maps to white land; 0 maps to black water.
pixels = np.where(matrix == 1, 255, 0).astype(np.uint8)
image = Image.fromarray(pixels, mode="L")
image.save(output_dir / "candidate.png")
image.resize((241 * 8, 61 * 8), resample=Image.Resampling.NEAREST).save(
    output_dir / "candidate_8x.png"
)
