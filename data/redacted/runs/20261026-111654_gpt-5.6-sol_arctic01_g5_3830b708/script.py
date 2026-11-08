from pathlib import Path

import numpy as np
from PIL import Image


# Hand-estimated coastline polygons in (longitude, latitude), WGS84 degrees.
# Boundaries are deliberately simplified; no geographic data is consulted.
LAND = [
    # Ellesmere Island, with its southeast coast and northern clipped extent.
    [(-90.8, 79.0), (-90.0, 78.5), (-89.2, 78.0), (-87.8, 77.6),
     (-85.8, 77.3), (-83.6, 77.0), (-81.0, 76.8), (-81.0, 79.0)],
    # Axel Heiberg Island.
    [(-93.8, 79.0), (-93.1, 78.6), (-92.5, 78.0), (-92.7, 77.3),
     (-91.9, 76.9), (-90.7, 76.8), (-89.4, 77.2), (-88.6, 78.0),
     (-89.0, 78.7), (-90.0, 79.0)],
    # Amund Ringnes Island.
    [(-95.0, 79.0), (-94.0, 78.8), (-93.2, 78.4), (-93.5, 77.9),
     (-94.3, 77.7), (-95.1, 78.0), (-95.4, 78.5)],
    # Ellef Ringnes Island.
    [(-103.0, 79.0), (-100.5, 79.0), (-99.2, 78.6), (-99.5, 77.8),
     (-100.8, 77.4), (-102.5, 77.7), (-103.4, 78.3)],
    # Cornwall Island.
    [(-96.5, 77.1), (-95.6, 77.2), (-94.9, 76.8), (-95.3, 76.5),
     (-96.2, 76.6)],
    # Bathurst Island.
    [(-99.2, 76.9), (-98.1, 76.8), (-97.0, 76.4), (-96.8, 75.7),
     (-97.6, 75.2), (-98.9, 75.4), (-99.7, 76.0)],
    # Devon Island.
    [(-90.7, 76.7), (-88.8, 76.5), (-87.2, 76.0), (-85.7, 75.7),
     (-84.6, 75.1), (-83.5, 74.7), (-82.5, 74.7), (-81.0, 75.0),
     (-81.0, 76.0), (-83.0, 76.4), (-85.3, 76.7), (-87.5, 76.9),
     (-89.5, 76.9)],
    # Melville Island.
    [(-111.0, 76.0), (-108.0, 76.1), (-105.2, 75.8), (-102.5, 75.5),
     (-100.0, 75.8), (-98.8, 76.3), (-99.7, 76.8), (-102.5, 77.0),
     (-106.0, 77.0), (-109.0, 76.7)],
    # Smaller central Queen Elizabeth islands.
    [(-101.7, 77.1), (-100.8, 77.2), (-100.2, 76.9), (-100.7, 76.6),
     (-101.5, 76.7)],
    [(-97.0, 77.3), (-96.2, 77.2), (-95.7, 76.9), (-96.0, 76.6),
     (-96.8, 76.7)],
    # Victoria Island, eastern portion within the map frame.
    [(-105.0, 75.4), (-104.0, 75.1), (-103.1, 74.8), (-102.2, 74.4),
     (-101.2, 74.0), (-100.3, 73.5), (-99.4, 73.0), (-105.0, 73.0)],
    # Prince of Wales Island.
    [(-99.4, 76.1), (-98.5, 76.2), (-97.6, 75.8), (-96.7, 75.4),
     (-96.2, 74.8), (-96.6, 74.2), (-97.6, 73.8), (-98.7, 74.2),
     (-99.5, 74.8), (-100.0, 75.5)],
    # Somerset Island.
    [(-95.8, 74.9), (-94.6, 75.1), (-93.3, 74.8), (-92.5, 74.3),
     (-92.7, 73.7), (-93.5, 73.3), (-94.7, 73.4), (-95.6, 73.9)],
    # Boothia Peninsula, only its northern tip reaches this frame.
    [(-96.2, 73.6), (-95.2, 73.7), (-94.2, 73.5), (-93.7, 73.0),
     (-96.5, 73.0)],
    # Baffin Island's northwestern edge at the eastern border.
    [(-81.7, 74.4), (-81.0, 74.6), (-81.0, 73.0), (-82.0, 73.0),
     (-82.3, 73.6)],
]


def on_segment(x, y, a, b):
    cross = (x - a[0]) * (b[1] - a[1]) - (y - a[1]) * (b[0] - a[0])
    if abs(cross) > 1e-9:
        return False
    return (min(a[0], b[0]) - 1e-9 <= x <= max(a[0], b[0]) + 1e-9
            and min(a[1], b[1]) - 1e-9 <= y <= max(a[1], b[1]) + 1e-9)


def inside(lon, lat, polygon):
    hit = False
    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]
        if on_segment(lon, lat, a, b):
            return True
        if (a[1] > lat) != (b[1] > lat):
            crossing = a[0] + (lat - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if lon < crossing:
                hit = not hit
    return hit


def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    lats = [79.0 - 0.1 * i for i in range(61)]
    lons = [-105.0 + 0.1 * i for i in range(241)]
    grid = np.zeros((61, 241), dtype=np.uint8)
    for row, lat in enumerate(lats):
        for col, lon in enumerate(lons):
            grid[row, col] = int(any(inside(lon, lat, p) for p in LAND))

    with (out / "candidate.csv").open("w", encoding="ascii", newline="") as f:
        for row in grid:
            f.write(",".join(map(str, row.tolist())) + "\n")
    image = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
    image.save(out / "candidate.png")
    image.resize((1928, 488), Image.Resampling.NEAREST).save(out / "candidate_8x.png")


if __name__ == "__main__":
    main()

