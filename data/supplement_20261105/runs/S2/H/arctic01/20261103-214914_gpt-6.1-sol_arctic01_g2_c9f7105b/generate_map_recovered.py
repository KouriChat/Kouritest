from pathlib import Path
import csv

import numpy as np
from PIL import Image


# Sampling coordinates, ordered north-to-south and west-to-east.
LATITUDES = np.round(np.arange(79.0, 72.9, -0.1), 1)
LONGITUDES = np.round(np.arange(-105.0, -80.9, 0.1), 1)

# Approximate hand-drawn island outlines, in (longitude, latitude).
# These are intentionally coarse geographic estimates, not sourced map data.
ISLANDS = [
    # Victoria Island: only its northeastern/southeastern edge reaches this box.
    [
        (-105.8, 73.9), (-104.7, 73.8), (-103.8, 73.6), (-102.7, 73.4),
        (-101.8, 73.1), (-101.0, 72.7), (-101.0, 71.9), (-103.0, 71.8),
        (-105.0, 72.0), (-106.0, 72.4),
    ],

    # Melville Island: easternmost fringe at the western edge of the box.
    [
        (-106.0, 75.1), (-105.0, 75.0), (-104.5, 75.3), (-104.8, 75.9),
        (-105.2, 76.5), (-106.2, 76.9), (-107.0, 76.7), (-106.7, 76.0),
    ],

    # Lougheed Island.
    [
        (-106.7, 77.3), (-105.0, 77.2), (-104.8, 77.7), (-105.3, 78.0),
        (-106.2, 78.0), (-106.8, 77.7),
    ],

    # Ellef Ringnes Island.
    [
        (-104.7, 78.1), (-103.9, 78.0), (-102.7, 78.1), (-101.8, 78.4),
        (-101.1, 78.8), (-101.2, 79.3), (-102.0, 79.6), (-103.1, 79.5),
        (-104.0, 79.2), (-104.8, 78.7),
    ],

    # Amund Ringnes Island.
    [
        (-100.8, 77.9), (-99.8, 77.8), (-98.7, 78.0), (-97.6, 78.3),
        (-96.8, 78.7), (-97.2, 79.1), (-98.3, 79.3), (-99.4, 79.2),
        (-100.4, 78.8), (-100.9, 78.4),
    ],

    # Axel Heiberg Island.
    [
        (-93.0, 78.4), (-92.2, 78.2), (-91.2, 78.3), (-90.4, 78.6),
        (-89.6, 78.9), (-88.7, 79.0), (-87.5, 79.4), (-86.5, 79.9),
        (-85.7, 80.4), (-88.0, 80.5), (-89.2, 80.1), (-90.6, 79.8),
        (-91.8, 79.4), (-92.7, 78.9),
    ],

    # Bathurst Island.
    [
        (-101.5, 75.6), (-100.6, 75.4), (-99.8, 75.5), (-98.9, 75.8),
        (-98.0, 76.0), (-97.0, 76.1), (-96.4, 76.5), (-96.7, 76.9),
        (-97.7, 77.2), (-98.8, 77.3), (-99.8, 77.1), (-100.7, 76.8),
        (-101.4, 76.4),
    ],

    # Cornwallis Island.
    [
        (-96.3, 74.9), (-95.7, 74.8), (-94.9, 74.9), (-94.1, 75.2),
        (-94.0, 75.6), (-94.7, 75.9), (-95.6, 75.8), (-96.2, 75.5),
    ],

    # Little Cornwallis Island.
    [
        (-96.4, 75.3), (-96.0, 75.3), (-95.8, 75.5), (-96.1, 75.7),
        (-96.4, 75.6),
    ],

    # Prince of Wales Island.
    [
        (-100.4, 72.8), (-99.4, 72.7), (-98.4, 72.9), (-97.7, 73.2),
        (-97.0, 73.5), (-96.3, 73.8), (-96.1, 74.4), (-96.5, 75.0),
        (-97.2, 75.5), (-98.2, 75.7), (-99.0, 75.5), (-99.5, 75.0),
        (-100.0, 74.4), (-100.4, 73.7),
    ],

    # Somerset Island.
    [
        (-95.8, 73.2), (-95.1, 73.1), (-94.3, 73.3), (-93.8, 73.7),
        (-93.3, 74.1), (-92.6, 74.5), (-92.1, 74.9), (-92.4, 75.3),
        (-93.2, 75.4), (-94.0, 75.2), (-94.7, 74.8), (-95.2, 74.3),
        (-95.7, 73.8),
    ],

    # Boothia Peninsula, clipped naturally by the southern edge of the grid.
    [
        (-94.4, 73.0), (-93.9, 73.4), (-93.7, 74.0), (-93.2, 74.6),
        (-92.7, 75.1), (-92.0, 75.7), (-91.3, 76.2), (-90.8, 76.8),
        (-90.4, 77.0), (-90.5, 76.1), (-90.8, 75.3), (-91.1, 74.5),
        (-91.6, 73.8), (-92.2, 73.2),
    ],

    # Devon Island.
    [
        (-90.5, 74.6), (-89.5, 74.5), (-88.5, 74.7), (-87.6, 74.8),
        (-86.4, 74.7), (-85.2, 74.8), (-84.3, 75.2), (-83.1, 75.5),
        (-82.0, 75.8), (-82.2, 76.4), (-83.0, 76.9), (-84.2, 77.2),
        (-85.5, 77.4), (-86.8, 77.2), (-88.0, 77.0), (-89.2, 76.7),
        (-90.1, 76.2), (-90.7, 75.6),
    ],

    # Meighen Island, near the northeastern edge of the sampling box.
    [
        (-81.6, 79.7), (-81.0, 79.6), (-80.8, 79.9), (-81.0, 80.2),
        (-81.5, 80.2),
    ],

    # Western edge of Ellesmere Island, along the eastern boundary of the box.
    [
        (-83.4, 76.0), (-82.8, 76.2), (-82.3, 76.6), (-81.8, 77.0),
        (-81.3, 77.5), (-80.8, 78.0), (-80.7, 79.0), (-80.7, 79.8),
        (-81.0, 80.5), (-80.0, 80.5), (-80.0, 73.0), (-82.2, 73.0),
        (-82.5, 74.0), (-83.0, 75.0),
    ],

    # Small islands in the central archipelago.
    [
        (-97.1, 77.4), (-96.4, 77.3), (-96.0, 77.6), (-96.3, 77.9),
        (-96.9, 77.9),
    ],
    [
        (-94.4, 76.4), (-93.9, 76.3), (-93.6, 76.6), (-94.0, 76.8),
        (-94.4, 76.7),
    ],
    [
        (-92.2, 77.4), (-91.7, 77.3), (-91.4, 77.6), (-91.8, 77.8),
        (-92.2, 77.7),
    ],
    [
        (-88.8, 77.7), (-88.2, 77.6), (-87.9, 77.9), (-88.3, 78.1),
        (-88.8, 78.0),
    ],
    [
        (-84.0, 76.0), (-83.5, 75.9), (-83.2, 76.2), (-83.6, 76.4),
        (-84.0, 76.3),
    ],
]


def points_in_polygon(x_grid, y_grid, polygon):
    """Return a boolean mask using the even-odd point-in-polygon rule."""
    inside = np.zeros(x_grid.shape, dtype=bool)
    x_vertices = [point[0] for point in polygon]
    y_vertices = [point[1] for point in polygon]

    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = x_vertices[i], y_vertices[i]
        xj, yj = x_vertices[j], y_vertices[j]

        crosses_y = (yi > y_grid) != (yj > y_grid)
        denominator = yj - yi
        if denominator != 0:
            crossing_x = (xj - xi) * (y_grid - yi) / denominator + xi
            inside ^= crosses_y & (x_grid < crossing_x)
        j = i

    return inside


def main():
    longitude_grid, latitude_grid = np.meshgrid(LONGITUDES, LATITUDES)
    matrix = np.zeros((len(LATITUDES), len(LONGITUDES)), dtype=np.uint8)

    for island in ISLANDS:
        matrix[points_in_polygon(longitude_grid, latitude_grid, island)] = 1

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "candidate.csv").open("w", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # L mode with only 0 and 255 ensures strictly black-and-white pixels.
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(output_dir / "candidate.png")

    enlarged = image.resize(
        (image.width * 8, image.height * 8),
        resample=Image.Resampling.NEAREST,
    )
    enlarged.save(output_dir / "candidate_8x.png")

    assert matrix.shape == (61, 241)
    assert int(matrix.size) == 14701
    assert set(np.unique(matrix).tolist()) <= {0, 1}
    assert image.size == (241, 61)
    assert enlarged.size == (1928, 488)
    assert set(image.getdata()) <= {0, 255}
    assert set(enlarged.getdata()) <= {0, 255}


if __name__ == "__main__":
    main()
