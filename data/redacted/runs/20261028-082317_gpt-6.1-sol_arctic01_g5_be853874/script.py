#!/usr/bin/env python3
"""Offline, hand-drawn approximation of land in the Canadian Arctic Archipelago.

Coordinates are WGS84 longitude/latitude in degrees. Polygon vertices encode
approximate coastlines from general geographic knowledge; no external data is
read or queried. The raster is classified by point-in-polygon tests.
"""
from pathlib import Path
import csv

import numpy as np
from PIL import Image


# Each polygon is an island outline, ordered around its coast. Longitudes are
# negative west. These coarse outlines deliberately omit tiny features.
ISLANDS = [
    # Melville Island: deeply indented north and south shores.
    [(-114.5, 74.0), (-112.7, 74.2), (-111.4, 74.7), (-109.2, 74.8),
     (-107.7, 75.1), (-106.2, 75.0), (-104.7, 75.5), (-103.4, 75.9),
     (-103.8, 76.5), (-105.0, 76.8), (-106.6, 76.9), (-108.0, 77.1),
     (-109.8, 76.8), (-111.1, 76.6), (-112.8, 76.5), (-114.3, 76.1),
     (-115.2, 75.5), (-115.0, 74.8)],
    # Prince Patrick Island
    [(-124.2, 76.1), (-122.8, 76.0), (-121.6, 76.3), (-120.4, 76.3),
     (-119.4, 76.8), (-118.7, 77.5), (-118.9, 78.1), (-120.0, 78.4),
     (-121.4, 78.4), (-122.4, 78.0), (-123.7, 77.8), (-124.4, 77.1)],
    # Eglinton Island
    [(-118.6, 75.7), (-117.2, 75.5), (-115.9, 75.8), (-115.1, 76.4),
     (-115.6, 77.0), (-116.6, 77.2), (-117.7, 76.9), (-118.5, 76.4)],
    # Mackenzie King Island
    [(-121.9, 77.5), (-120.9, 77.3), (-119.8, 77.6), (-119.2, 78.1),
     (-119.8, 78.5), (-121.0, 78.4), (-121.8, 78.1)],
    # Brock Island
    [(-119.7, 77.7), (-118.9, 77.7), (-118.3, 78.0), (-118.7, 78.3),
     (-119.5, 78.2)],
    # Borden Island
    [(-112.7, 77.1), (-111.3, 77.0), (-110.1, 77.3), (-109.9, 77.9),
     (-110.8, 78.3), (-112.2, 78.2), (-113.1, 77.8)],
    # Lougheed Island
    [(-106.6, 77.2), (-105.5, 77.1), (-104.6, 77.5), (-104.4, 78.0),
     (-105.1, 78.3), (-106.2, 78.1), (-106.8, 77.7)],
    # Ellef Ringnes Island
    [(-103.5, 78.0), (-102.1, 77.8), (-100.5, 77.9), (-99.2, 78.3),
     (-98.5, 78.8), (-99.2, 79.3), (-100.6, 79.4), (-102.2, 79.2),
     (-103.6, 78.8)],
    # Amund Ringnes Island
    [(-98.8, 78.1), (-97.4, 78.0), (-95.9, 78.2), (-94.9, 78.7),
     (-95.2, 79.2), (-96.4, 79.5), (-97.7, 79.3), (-98.6, 78.8)],
    # Axel Heiberg Island, truncated by northern frame.
    [(-96.0, 78.0), (-94.2, 77.9), (-92.7, 78.1), (-91.4, 78.5),
     (-90.0, 78.8), (-88.8, 79.1), (-88.0, 79.0), (-87.0, 79.0),
     (-87.0, 79.1), (-96.0, 79.1)],
    # Cornwallis Island
    [(-96.5, 74.7), (-95.4, 74.5), (-94.3, 74.7), (-93.4, 75.0),
     (-93.7, 75.5), (-94.8, 75.7), (-95.9, 75.4), (-96.5, 75.1)],
    # Bathurst Island
    [(-98.0, 75.0), (-96.8, 74.9), (-95.9, 75.2), (-95.7, 75.8),
     (-94.9, 76.0), (-94.5, 76.4), (-95.3, 76.7), (-96.6, 76.7),
     (-97.8, 76.4), (-98.2, 75.8)],
    # Devon Island: central/eastern part reaches the map's east edge.
    [(-92.8, 74.7), (-91.7, 74.5), (-90.6, 74.7), (-89.7, 74.9),
     (-88.7, 74.7), (-87.8, 74.9), (-86.6, 75.1), (-85.5, 75.3),
     (-84.2, 75.7), (-82.8, 76.0), (-81.0, 76.2), (-81.0, 78.2),
     (-82.2, 78.4), (-83.6, 78.2), (-84.5, 77.9), (-85.4, 77.4),
     (-86.5, 77.1), (-87.7, 76.8), (-88.6, 76.5), (-89.8, 76.3),
     (-90.8, 76.0), (-91.8, 75.7), (-92.8, 75.3)],
    # Somerset Island
    [(-95.7, 72.0), (-94.7, 72.0), (-93.8, 72.2), (-92.8, 72.3),
     (-92.0, 72.8), (-91.4, 73.3), (-91.5, 74.0), (-92.1, 74.5),
     (-93.2, 74.7), (-94.3, 74.5), (-95.1, 74.2), (-95.8, 73.7),
     (-96.0, 73.0)],
    # Prince of Wales Island
    [(-99.6, 72.0), (-98.6, 71.9), (-97.7, 72.2), (-96.9, 72.6),
     (-96.3, 73.2), (-96.4, 73.9), (-97.2, 74.4), (-98.3, 74.5),
     (-99.2, 74.2), (-99.9, 73.7), (-100.1, 73.0)],
    # King Christian Island
    [(-102.3, 77.4), (-101.5, 77.3), (-100.8, 77.6), (-100.7, 78.0),
     (-101.4, 78.2), (-102.1, 78.0)],
    # Graham Island, north of Prince Patrick
    [(-118.0, 78.3), (-117.1, 78.2), (-116.5, 78.5), (-116.7, 78.9),
     (-117.6, 79.0), (-118.2, 78.7)],
    # Little Cornwallis Island
    [(-96.4, 75.2), (-95.8, 75.1), (-95.3, 75.3), (-95.4, 75.6),
     (-96.1, 75.6)],
    # Small island groups in the Parry Islands; simplified as discrete land.
    [(-106.0, 76.0), (-105.5, 75.9), (-105.2, 76.2), (-105.5, 76.4),
     (-106.0, 76.3)],
    [(-109.4, 75.6), (-108.8, 75.5), (-108.4, 75.8), (-108.7, 76.0),
     (-109.3, 75.9)],
    [(-98.6, 76.7), (-98.0, 76.6), (-97.6, 76.9), (-97.9, 77.1),
     (-98.5, 77.0)],
]


def points_in_polygon(lon_grid, lat_grid, polygon):
    """Vectorized even-odd ray casting; boundary convention is immaterial at grid scale."""
    inside = np.zeros(lon_grid.shape, dtype=bool)
    vertices = polygon
    for i, (x1, y1) in enumerate(vertices):
        x2, y2 = vertices[(i + 1) % len(vertices)]
        crosses = (y1 > lat_grid) != (y2 > lat_grid)
        x_cross = (x2 - x1) * (lat_grid - y1) / (y2 - y1 + 1e-30) + x1
        inside ^= crosses & (lon_grid < x_cross)
    return inside


def main():
    # Axis order follows the requested display: west->east, north->south.
    longitudes = np.round(np.arange(-105.0, -81.0 + 0.0001, 0.1), 1)
    latitudes = np.round(np.arange(79.0, 73.0 - 0.0001, -0.1), 1)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    land = np.zeros((len(latitudes), len(longitudes)), dtype=bool)
    for polygon in ISLANDS:
        land |= points_in_polygon(lon_grid, lat_grid, polygon)
    return land


if __name__ == "__main__":
    matrix = main().astype(np.uint8)
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(matrix.tolist())
    image = Image.fromarray(np.where(matrix == 1, 255, 0).astype(np.uint8), mode="L")
    image.save(out / "candidate.png")
    image.resize((1928, 488), resample=Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )
