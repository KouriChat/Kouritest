import csv
import os

import numpy as np
from PIL import Image

# WGS84 sampling grid: north to south, west to east.
LATITUDES = [79.0 - 0.1 * i for i in range(61)]
LONGITUDES = [-105.0 + 0.1 * i for i in range(241)]

# Hand-drawn approximate land polygons, listed as (longitude, latitude).
# They represent the major islands and mainland portions in the sampled region.
LAND_POLYGONS = [
    # Axel Heiberg Island
    [(-96.0, 79.0), (-92.0, 79.0), (-90.8, 78.4), (-91.3, 77.7),
     (-89.9, 77.1), (-91.0, 76.3), (-93.4, 75.7), (-96.0, 76.1),
     (-98.0, 77.0), (-97.8, 78.0)],

    # Ellesmere Island, including northern and eastern coast
    [(-95.0, 79.0), (-81.0, 79.0), (-81.0, 77.8), (-79.5, 76.8),
     (-78.5, 75.8), (-79.2, 74.8), (-80.5, 74.0), (-82.4, 73.0),
     (-85.0, 73.0), (-87.0, 73.8), (-88.0, 74.8), (-89.4, 75.2),
     (-90.2, 76.0), (-91.0, 77.2), (-93.4, 77.7), (-95.5, 78.4)],

    # Devon Island
    [(-91.8, 75.9), (-88.2, 76.1), (-85.5, 75.8), (-84.2, 75.0),
     (-85.0, 74.4), (-87.0, 74.3), (-89.1, 74.5), (-90.8, 75.0)],

    # Cornwallis Island
    [(-96.2, 75.4), (-93.0, 75.5), (-92.7, 74.8), (-94.0, 74.6),
     (-95.9, 74.8)],

    # Bathurst Island
    [(-99.0, 75.8), (-96.0, 76.0), (-94.5, 75.4), (-95.0, 74.8),
     (-97.1, 74.6), (-99.0, 75.0)],

    # Melville Island
    [(-111.0, 75.9), (-104.0, 76.1), (-101.0, 75.6), (-99.2, 75.0),
     (-100.0, 74.5), (-103.4, 74.3), (-107.0, 74.7), (-110.0, 75.0)],

    # Prince of Wales Island
    [(-101.0, 73.8), (-98.7, 74.1), (-96.6, 73.8), (-96.9, 73.0),
     (-100.0, 73.0)],

    # Somerset Island
    [(-94.0, 74.8), (-91.2, 74.9), (-89.9, 74.1), (-90.3, 73.0),
     (-93.0, 73.0), (-94.5, 73.8)],

    # Baffin Island, western portion at the southeast edge
    [(-84.8, 73.5), (-81.0, 73.8), (-81.0, 73.0), (-85.2, 73.0)],

    # Banks Island, western edge of the frame
    [(-105.0, 74.5), (-103.5, 74.7), (-102.0, 74.2),
     (-102.5, 73.4), (-104.0, 73.0), (-105.0, 73.0)],
]


def point_on_segment(px, py, ax, ay, bx, by, eps=1e-10):
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > eps:
        return False
    return (min(ax, bx) - eps <= px <= max(ax, bx) + eps and
            min(ay, by) - eps <= py <= max(ay, by) + eps)


def point_in_polygon(lon, lat, polygon):
    """Ray-casting point test; boundary points count as land."""
    inside = False
    n = len(polygon)
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]
        if point_on_segment(lon, lat, ax, ay, bx, by):
            return True
        if (ay > lat) != (by > lat):
            crossing_lon = ax + (lat - ay) * (bx - ax) / (by - ay)
            if lon < crossing_lon:
                inside = not inside
    return inside


def is_land(lon, lat):
    return any(point_in_polygon(lon, lat, polygon)
               for polygon in LAND_POLYGONS)


matrix = np.zeros((len(LATITUDES), len(LONGITUDES)), dtype=np.uint8)
for row, lat in enumerate(LATITUDES):
    for col, lon in enumerate(LONGITUDES):
        matrix[row, col] = 1 if is_land(lon, lat) else 0

os.makedirs("outputs", exist_ok=True)

with open("outputs/candidate.csv", "w", newline="", encoding="ascii") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(matrix.tolist())

# Matrix value 1 is white land; 0 is black water.
pixels = np.where(matrix == 1, 255, 0).astype(np.uint8)
image = Image.fromarray(pixels, mode="L")
image.save("outputs/candidate.png")

enlarged = image.resize((241 * 8, 61 * 8), resample=Image.Resampling.NEAREST)
enlarged.save("outputs/candidate_8x.png")
