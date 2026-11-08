from pathlib import Path

import numpy as np
from PIL import Image


# Hand-drawn approximate land polygons. Coordinates are (degrees west, north).
LAND_POLYGONS = [
    # Devon Island
    [
        (89.2, 74.6), (88.0, 74.9), (87.1, 75.4), (86.3, 76.0),
        (85.5, 76.8), (84.9, 77.4), (84.4, 77.8), (83.7, 77.5),
        (83.1, 77.0), (82.7, 76.4), (82.4, 75.8), (82.0, 75.2),
        (82.4, 74.8), (83.3, 74.5), (84.1, 74.2), (85.0, 74.0),
        (86.0, 74.0), (87.0, 74.1), (88.1, 74.3),
    ],
    # Axel Heiberg Island
    [
        (92.0, 77.4), (91.5, 77.9), (90.9, 78.5), (90.3, 79.2),
        (89.8, 79.8), (89.0, 80.2), (88.2, 80.0), (87.6, 79.5),
        (87.1, 78.9), (86.6, 78.3), (86.0, 77.8), (85.4, 77.4),
        (85.8, 77.1), (86.5, 77.0), (87.2, 77.1), (88.0, 77.2),
        (88.8, 77.1), (89.6, 77.2), (90.5, 77.2), (91.3, 77.3),
    ],
    # Western Ellesmere Island; eastern edge includes continuous land.
    [
        (83.0, 78.2), (82.7, 78.7), (82.5, 79.2), (82.2, 79.7),
        (81.8, 80.2), (80.5, 80.5), (79.0, 80.5), (77.0, 80.5),
        (75.0, 80.5), (75.0, 73.0), (79.0, 73.0), (80.0, 73.8),
        (80.8, 74.6), (81.1, 75.4), (81.4, 76.2), (81.7, 77.0),
        (82.2, 77.7),
    ],
    # Bathurst Island
    [
        (99.5, 75.0), (99.1, 75.6), (98.5, 76.2), (97.7, 76.7),
        (96.8, 76.9), (95.9, 76.6), (95.2, 76.1), (94.8, 75.5),
        (95.1, 75.0), (95.9, 74.8), (96.8, 74.8), (97.7, 74.9),
        (98.7, 74.9),
    ],
    # Cornwallis Island
    [
        (96.3, 74.8), (96.0, 75.2), (95.5, 75.5), (94.8, 75.4),
        (94.4, 75.0), (94.6, 74.6), (95.2, 74.4), (95.9, 74.5),
    ],
    # Northern Somerset Island
    [
        (96.4, 73.0), (96.1, 73.5), (95.8, 74.0), (95.1, 74.4),
        (94.2, 74.6), (93.3, 74.4), (92.9, 73.9), (93.1, 73.4),
        (93.6, 73.0),
    ],
    # Northern Prince of Wales Island
    [
        (101.0, 73.0), (100.7, 73.5), (100.1, 74.0), (99.5, 74.5),
        (98.7, 74.8), (97.8, 74.6), (97.0, 74.2), (96.6, 73.7),
        (96.9, 73.2), (97.6, 73.0),
    ],
    # Northernmost tip of Victoria Island
    [
        (109.0, 73.0), (108.4, 73.4), (107.7, 73.7), (106.8, 73.8),
        (105.9, 73.7), (105.2, 73.4), (104.6, 73.0),
    ],
    # Lougheed Island group
    [
        (106.0, 77.1), (105.8, 77.4), (105.4, 77.6), (105.0, 77.5),
        (104.9, 77.2), (105.2, 77.0), (105.7, 77.0),
    ],
    [
        (103.6, 77.0), (103.4, 77.3), (103.0, 77.5), (102.7, 77.3),
        (102.8, 77.0), (103.2, 76.9),
    ],
    # King Christian Island
    [
        (102.9, 77.0), (102.7, 77.4), (102.2, 77.8), (101.6, 77.9),
        (101.1, 77.6), (101.0, 77.2), (101.4, 76.9), (102.1, 76.8),
    ],
    # Ellef Ringnes Island
    [
        (103.8, 77.8), (103.7, 78.2), (103.3, 78.6), (102.7, 78.8),
        (102.1, 78.6), (101.6, 78.2), (101.7, 77.8), (102.3, 77.6),
        (103.0, 77.6),
    ],
    # Amund Ringnes Island
    [
        (99.3, 78.1), (99.1, 78.5), (98.7, 78.8), (98.1, 78.8),
        (97.7, 78.5), (97.8, 78.1), (98.2, 77.9), (98.8, 77.9),
    ],
    # Byam Martin Island
    [
        (105.0, 75.8), (104.7, 76.2), (104.2, 76.4), (103.6, 76.3),
        (103.3, 76.0), (103.5, 75.6), (104.1, 75.5), (104.7, 75.6),
    ],
    # Small island north of Bathurst
    [
        (97.2, 77.0), (97.0, 77.3), (96.6, 77.4), (96.3, 77.1),
        (96.5, 76.9), (96.9, 76.9),
    ],
]


def point_in_polygon(x, y, polygon):
    """Ray-casting test for a point and a polygon."""
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
    latitudes = [79.0 - 0.1 * row for row in range(61)]
    west_longitudes = [105.0 - 0.1 * col for col in range(241)]
    matrix = np.zeros((61, 241), dtype=np.uint8)

    for row, latitude in enumerate(latitudes):
        for col, west_longitude in enumerate(west_longitudes):
            matrix[row, col] = int(
                any(
                    point_in_polygon(west_longitude, latitude, polygon)
                    for polygon in LAND_POLYGONS
                )
            )

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "candidate.csv").open(
        "w", encoding="ascii", newline=""
    ) as csv_file:
        for row in matrix:
            csv_file.write(",".join(str(int(value)) for value in row) + "\n")

    image = Image.fromarray(matrix * 255, mode="L")
    image.save(output_dir / "candidate.png")
    enlarged = image.resize((1928, 488), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")

    assert matrix.shape == (61, 241)
    assert matrix.size == 14701
    assert set(np.unique(matrix)).issubset({0, 1})


if __name__ == "__main__":
    main()
