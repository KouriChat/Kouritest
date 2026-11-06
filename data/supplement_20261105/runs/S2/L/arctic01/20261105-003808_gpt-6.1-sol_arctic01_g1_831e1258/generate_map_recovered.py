from pathlib import Path

import numpy as np
from PIL import Image


# Hand-estimated land outlines in (longitude, latitude) degrees.
# Coordinates describe simplified island boundaries, not map-derived data.
LAND_POLYGONS = [
    # Banks Island
    [
        (-124.8, 73.0), (-124.2, 74.0), (-123.9, 75.0),
        (-124.1, 76.0), (-124.8, 77.0), (-125.5, 78.0),
        (-126.5, 78.5), (-128.5, 78.6), (-130.0, 78.0),
        (-130.8, 77.0), (-130.4, 75.8), (-129.4, 74.8),
        (-127.5, 73.9), (-126.0, 73.3), (-124.8, 73.0),
    ],
    # Victoria Island
    [
        (-104.8, 73.0), (-104.0, 73.8), (-104.0, 74.8),
        (-105.0, 75.7), (-106.0, 76.4), (-108.0, 76.9),
        (-110.0, 77.4), (-112.0, 78.1), (-114.5, 78.4),
        (-117.0, 78.3), (-119.0, 77.8), (-120.8, 77.0),
        (-121.5, 76.0), (-120.8, 75.1), (-119.0, 74.5),
        (-117.0, 73.8), (-114.5, 73.2), (-111.0, 72.8),
        (-108.0, 72.8), (-104.8, 73.0),
    ],
    # Prince of Wales Island
    [
        (-98.0, 71.7), (-97.2, 72.3), (-97.0, 73.2),
        (-97.4, 74.2), (-98.0, 75.0), (-99.2, 75.8),
        (-100.8, 76.3), (-102.0, 76.0), (-102.5, 75.2),
        (-102.0, 74.2), (-101.0, 73.2), (-100.0, 72.2),
        (-98.0, 71.7),
    ],
    # Somerset Island
    [
        (-94.0, 72.0), (-93.0, 72.8), (-92.0, 73.8),
        (-91.5, 74.8), (-91.6, 75.8), (-92.4, 76.5),
        (-93.7, 76.9), (-95.0, 76.5), (-95.6, 75.7),
        (-95.0, 74.8), (-94.5, 73.8), (-94.0, 72.8),
        (-94.0, 72.0),
    ],
    # Boothia Peninsula and adjacent mainland
    [
        (-99.0, 72.0), (-98.2, 72.7), (-97.5, 73.5),
        (-96.5, 74.3), (-95.0, 75.0), (-93.5, 75.8),
        (-92.0, 76.5), (-90.0, 77.2), (-87.0, 78.0),
        (-84.0, 78.8), (-81.0, 79.2), (-81.0, 73.0),
        (-93.0, 73.0), (-96.0, 72.7), (-99.0, 72.0),
    ],
    # King William Island
    [
        (-97.5, 68.0), (-96.5, 68.8), (-96.2, 69.8),
        (-96.5, 70.8), (-97.5, 71.7), (-99.0, 72.0),
        (-100.5, 71.5), (-101.2, 70.5), (-100.5, 69.3),
        (-99.0, 68.4), (-97.5, 68.0),
    ],
    # Ellesmere Island, western and southern extent in this frame
    [
        (-81.0, 73.0), (-82.0, 74.0), (-83.0, 75.0),
        (-84.0, 76.0), (-85.0, 77.0), (-86.5, 78.0),
        (-88.0, 79.0), (-81.0, 79.0), (-81.0, 73.0),
    ],
    # Axel Heiberg Island
    [
        (-92.0, 78.0), (-91.0, 78.6), (-90.0, 79.0),
        (-86.5, 79.0), (-87.0, 78.4), (-88.5, 77.7),
        (-90.0, 77.4), (-91.5, 77.6), (-92.0, 78.0),
    ],
    # Bathurst Island
    [
        (-97.0, 77.0), (-96.0, 77.6), (-94.5, 78.0),
        (-93.0, 78.4), (-91.5, 78.2), (-92.0, 77.5),
        (-93.5, 77.0), (-95.0, 76.8), (-97.0, 77.0),
    ],
    # Melville Island
    [
        (-111.0, 76.0), (-109.0, 76.5), (-107.5, 77.2),
        (-106.0, 78.0), (-104.5, 78.3), (-103.5, 77.8),
        (-104.5, 77.0), (-106.0, 76.5), (-108.0, 75.8),
        (-111.0, 76.0),
    ],
    # Borden Island
    [
        (-114.0, 78.0), (-112.0, 78.5), (-110.0, 79.0),
        (-107.5, 79.0), (-108.5, 78.4), (-110.5, 77.9),
        (-112.5, 77.7), (-114.0, 78.0),
    ],
    # Mackenzie King Island
    [
        (-122.0, 77.0), (-120.0, 77.4), (-118.5, 77.8),
        (-117.0, 77.6), (-118.0, 77.0), (-120.0, 76.7),
        (-122.0, 77.0),
    ],
    # Prince Patrick Island
    [
        (-121.0, 76.0), (-119.0, 76.5), (-117.5, 77.0),
        (-116.0, 77.4), (-114.5, 77.2), (-115.5, 76.5),
        (-117.0, 76.0), (-119.0, 75.7), (-121.0, 76.0),
    ],
    # Cornwallis Island
    [
        (-96.0, 74.5), (-95.0, 75.0), (-94.0, 75.3),
        (-93.0, 75.0), (-93.5, 74.5), (-95.0, 74.2),
        (-96.0, 74.5),
    ],
    # Devon Island
    [
        (-96.0, 74.5), (-94.0, 75.0), (-92.0, 75.7),
        (-90.0, 76.5), (-88.0, 77.0), (-86.0, 77.5),
        (-84.0, 78.0), (-82.0, 78.5), (-81.0, 78.8),
        (-81.0, 75.2), (-83.0, 75.0), (-85.0, 74.7),
        (-88.0, 74.5), (-91.0, 74.3), (-94.0, 74.2),
        (-96.0, 74.5),
    ],
    # Cornwall Island
    [
        (-97.0, 76.0), (-96.0, 76.3), (-95.0, 76.1),
        (-95.5, 75.7), (-96.5, 75.6), (-97.0, 76.0),
    ],
    # Amund Ringnes Island
    [
        (-94.5, 78.0), (-93.0, 78.5), (-91.0, 78.7),
        (-89.5, 78.4), (-90.5, 77.8), (-92.5, 77.6),
        (-94.0, 77.7), (-94.5, 78.0),
    ],
    # Ellef Ringnes Island
    [
        (-103.0, 78.0), (-101.0, 78.5), (-99.0, 78.8),
        (-97.0, 78.5), (-98.0, 77.9), (-100.0, 77.6),
        (-102.0, 77.7), (-103.0, 78.0),
    ],
]


def point_in_polygon(lon, lat, polygon):
    """Ray-crossing point-in-polygon test."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        crosses = (yi > lat) != (yj > lat)
        if crosses:
            x_intersection = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_intersection:
                inside = not inside
        j = i
    return inside


def is_land(lon, lat):
    return any(point_in_polygon(lon, lat, polygon) for polygon in LAND_POLYGONS)


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    latitudes = [79.0 - 0.1 * i for i in range(61)]
    longitudes = [-105.0 + 0.1 * i for i in range(241)]

    matrix = np.zeros((61, 241), dtype=np.uint8)
    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            matrix[row, col] = 1 if is_land(lon, lat) else 0

    assert matrix.shape == (61, 241)
    assert matrix.size == 14701
    assert np.isin(matrix, (0, 1)).all()

    with (output_dir / "candidate.csv").open("w", newline="") as f:
        for row in matrix:
            f.write(",".join(str(int(value)) for value in row) + "\n")

    # Matrix value 1 is white land; 0 is black water.
    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    assert image.size == (241, 61)
    assert set(image.getdata()).issubset({(0, 0, 0), (255, 255, 255)})
    image.save(output_dir / "candidate.png")

    enlarged = image.resize((241 * 8, 61 * 8), resample=Image.Resampling.NEAREST)
    assert enlarged.size == (1928, 488)
    assert set(enlarged.getdata()).issubset({(0, 0, 0), (255, 255, 255)})
    enlarged.save(output_dir / "candidate_8x.png")


if __name__ == "__main__":
    main()
