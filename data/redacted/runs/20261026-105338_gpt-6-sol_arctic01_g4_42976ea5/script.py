"""Knowledge-based approximation of land/water in the Canadian Arctic Archipelago."""
from pathlib import Path
import csv

import numpy as np
from PIL import Image


# Coordinates are (longitude, latitude), in degrees. These hand-drawn outlines
# intentionally approximate major island coastlines; no geographic data is read.
ISLANDS = [
    # Ellesmere Island: western coast and long northern/eastern extent
    [(-81.0, 78.25), (-81.4, 79.0), (-82.0, 79.8), (-83.5, 80.0),
     (-85.0, 80.2), (-87.5, 80.3), (-89.0, 80.0), (-89.2, 79.3),
     (-88.0, 78.8), (-86.5, 78.3), (-85.0, 77.8), (-83.5, 77.2),
     (-82.2, 76.4), (-81.4, 76.0), (-80.8, 75.7), (-80.6, 75.45),
     (-81.0, 75.3), (-81.7, 75.2), (-82.8, 75.3), (-83.7, 75.5),
     (-84.5, 75.8), (-85.4, 76.0), (-86.1, 76.2), (-87.2, 76.4),
     (-88.2, 76.7), (-89.0, 77.1), (-90.0, 77.5), (-90.2, 78.0),
     (-89.6, 78.4), (-88.8, 78.9), (-88.2, 79.4), (-87.5, 79.8),
     (-85.5, 79.9), (-83.0, 79.6), (-81.0, 78.25)],
    # Devon Island
    [(-91.0, 74.75), (-90.4, 75.0), (-89.4, 75.3), (-88.8, 75.8),
     (-88.3, 76.3), (-87.5, 76.8), (-86.5, 77.2), (-85.0, 77.5),
     (-83.5, 77.5), (-82.3, 77.2), (-81.4, 76.8), (-80.9, 76.3),
     (-81.2, 75.8), (-82.0, 75.4), (-83.0, 75.1), (-84.0, 74.8),
     (-85.0, 74.6), (-86.0, 74.45), (-87.0, 74.5), (-88.0, 74.55),
     (-89.0, 74.6), (-90.0, 74.65), (-91.0, 74.75)],
    # Axel Heiberg Island
    [(-91.3, 78.0), (-90.5, 78.5), (-90.2, 79.2), (-90.6, 79.9),
     (-91.5, 80.3), (-93.0, 80.2), (-94.3, 79.9), (-95.0, 79.4),
     (-94.8, 78.8), (-94.0, 78.4), (-93.0, 78.1), (-92.0, 77.9),
     (-91.3, 78.0)],
    # Bathurst Island
    [(-105.0, 75.0), (-104.2, 75.25), (-103.2, 75.4), (-102.2, 75.7),
     (-101.0, 76.0), (-100.2, 76.4), (-99.6, 76.9), (-99.5, 77.45),
     (-100.2, 77.8), (-101.4, 77.95), (-102.6, 77.8), (-103.6, 77.5),
     (-104.5, 77.2), (-105.0, 76.9), (-105.0, 75.0)],
]
ISLANDS += [
    # Cornwallis Island
    [(-96.3, 74.65), (-95.5, 74.75), (-94.6, 75.0), (-94.2, 75.5),
     (-94.6, 76.0), (-95.4, 76.25), (-96.2, 76.0), (-96.6, 75.5),
     (-96.3, 74.65)],
    # Somerset Island
    [(-97.0, 72.8), (-96.3, 73.2), (-95.5, 73.6), (-94.5, 74.1),
     (-93.8, 74.65), (-93.6, 75.15), (-94.2, 75.4), (-95.1, 75.25),
     (-96.0, 74.9), (-96.7, 74.5), (-97.2, 74.0), (-97.0, 72.8)],
    # Prince of Wales Island
    [(-99.2, 72.45), (-98.4, 72.55), (-97.5, 72.9), (-96.8, 73.35),
     (-96.4, 73.85), (-96.7, 74.3), (-97.5, 74.55), (-98.4, 74.4),
     (-99.2, 74.0), (-99.8, 73.5), (-99.8, 72.9), (-99.2, 72.45)],
    # King William Island
    [(-98.5, 68.8), (-97.5, 69.0), (-96.5, 69.4), (-95.8, 69.9),
     (-96.1, 70.5), (-97.0, 70.75), (-98.0, 70.5), (-98.8, 70.0),
     (-99.0, 69.4), (-98.5, 68.8)],
    # Victoria Island, only its eastern portion reaches the requested frame
    [(-105.0, 70.0), (-104.2, 70.3), (-103.2, 70.6), (-102.0, 70.9),
     (-101.0, 71.2), (-100.2, 71.65), (-99.6, 72.2), (-99.3, 72.8),
     (-99.5, 73.4), (-100.2, 73.8), (-101.0, 74.0), (-102.0, 73.9),
     (-103.0, 73.7), (-104.0, 73.45), (-105.0, 73.15), (-105.0, 70.0)],
    # Bathurst-area smaller islands and the Ringnes group
    [(-98.2, 77.5), (-97.6, 77.65), (-97.0, 77.55), (-96.8, 77.2),
     (-97.2, 76.95), (-97.9, 77.0), (-98.2, 77.5)],
    [(-96.8, 77.8), (-96.1, 78.05), (-95.4, 78.0), (-95.0, 77.65),
     (-95.4, 77.35), (-96.2, 77.4), (-96.8, 77.8)],
    [(-96.2, 78.6), (-95.5, 78.85), (-94.7, 78.8), (-94.2, 78.45),
     (-94.5, 78.1), (-95.3, 78.1), (-96.0, 78.3), (-96.2, 78.6)],
]


def inside_polygon(lon, lat, polygon):
    """Odd-even point-in-polygon test in longitude/latitude coordinates."""
    inside = False
    xj, yj = polygon[-1]
    for xi, yi in polygon:
        if (yi > lat) != (yj > lat):
            crossing_x = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing_x:
                inside = not inside
        xj, yj = xi, yi
    return inside
# Cornwallis-area islands and the eastern Sverdrup islands, all inside the frame.
ISLANDS += [
    # Little Cornwallis Island
    [(-96.0, 76.15), (-95.6, 76.3), (-95.1, 76.25), (-94.9, 75.95),
     (-95.2, 75.75), (-95.8, 75.85), (-96.0, 76.15)],
    # North Kent Island
    [(-89.5, 76.9), (-89.0, 77.15), (-88.4, 77.1), (-88.1, 76.8),
     (-88.5, 76.6), (-89.2, 76.65), (-89.5, 76.9)],
    # South Kent Island
    [(-88.5, 76.0), (-87.9, 76.2), (-87.3, 76.1), (-87.1, 75.8),
     (-87.6, 75.6), (-88.3, 75.7), (-88.5, 76.0)],
    # Grinnell Peninsula and nearby Ellesmere coastal islands
    [(-82.4, 78.4), (-81.9, 78.55), (-81.5, 78.4), (-81.5, 78.1),
     (-81.9, 77.95), (-82.4, 78.1), (-82.4, 78.4)],
    # Small islands north of Devon
    [(-84.6, 77.95), (-84.0, 78.1), (-83.5, 77.95), (-83.6, 77.65),
     (-84.2, 77.55), (-84.7, 77.7), (-84.6, 77.95)],
    # The western extremity of Melville Island reaches this map boundary.
    [(-105.0, 75.8), (-104.7, 75.85), (-104.5, 75.7), (-104.6, 75.5),
     (-105.0, 75.45), (-105.0, 75.8)],
]


def make_grid():
    latitudes = [79.0 - 0.1 * i for i in range(61)]
    longitudes = [-105.0 + 0.1 * j for j in range(241)]
    grid = np.zeros((61, 241), dtype=np.uint8)
    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            grid[row, col] = any(inside_polygon(lon, lat, p) for p in ISLANDS)
    return grid


def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    grid = make_grid()
    assert grid.shape == (61, 241)
    assert int(grid.size) == 14701
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(grid.tolist())
    # Encode land as white (255), water as black (0); rows run north to south.
    image = Image.fromarray(grid * 255, mode="L")
    image.save(out / "candidate.png")
    enlarged = image.resize((1928, 488), resample=Image.Resampling.NEAREST)
    enlarged.save(out / "candidate_8x.png")


if __name__ == "__main__":
    main()
