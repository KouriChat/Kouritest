from pathlib import Path

import numpy as np
from PIL import Image


# Hand-drawn approximate island outlines, (longitude, latitude).
LAND_POLYGONS = [
    # Ellesmere Island
    [(-93.8, 78.7), (-92.8, 79.0), (-91.0, 79.1), (-89.8, 78.8),
     (-88.2, 78.9), (-86.6, 78.5), (-85.0, 78.0), (-83.7, 77.3),
     (-82.6, 76.5), (-81.8, 75.6), (-82.8, 75.0), (-84.0, 75.2),
     (-85.2, 75.8), (-86.4, 76.4), (-87.3, 76.9), (-88.5, 77.1),
     (-89.6, 76.8), (-90.5, 76.5), (-91.8, 76.7), (-92.6, 77.2),
     (-93.4, 77.8)],
    # Axel Heiberg Island
    [(-91.5, 79.0), (-90.4, 79.2), (-89.1, 79.0), (-87.9, 78.7),
     (-86.8, 78.3), (-86.2, 77.7), (-86.7, 77.1), (-87.8, 76.8),
     (-88.8, 76.9), (-89.7, 77.3), (-90.7, 77.5), (-91.5, 78.0),
     (-92.0, 78.5)],
    # Devon Island
    [(-90.0, 75.9), (-89.3, 76.3), (-87.7, 76.4), (-85.9, 76.1),
     (-84.3, 75.6), (-83.0, 75.0), (-82.7, 74.5), (-83.8, 74.3),
     (-85.1, 74.5), (-86.3, 74.7), (-87.1, 75.0), (-88.4, 75.1),
     (-89.6, 75.4)],
    # Bathurst and Cornwallis Islands
    [(-99.6, 76.6), (-98.7, 77.1), (-97.6, 77.2), (-96.5, 76.9),
     (-95.6, 76.4), (-95.7, 75.7), (-96.4, 75.3), (-97.5, 75.4),
     (-98.4, 75.7), (-99.3, 76.0)],
    [(-96.5, 75.4), (-95.7, 75.6), (-94.8, 75.5), (-93.8, 75.2),
     (-93.6, 74.8), (-94.3, 74.6), (-95.2, 74.7), (-96.0, 74.9)],
    # Somerset and Prince of Wales Islands
    [(-95.0, 74.8), (-94.3, 75.0), (-93.1, 74.9), (-92.1, 74.5),
     (-91.9, 74.0), (-92.5, 73.5), (-93.4, 73.2), (-94.3, 73.4),
     (-95.0, 73.8), (-95.4, 74.3)],
    [(-102.2, 74.9), (-101.2, 75.2), (-100.0, 75.1), (-99.0, 74.8),
     (-98.3, 74.1), (-98.4, 73.5), (-99.1, 73.3), (-100.0, 73.5),
     (-101.1, 73.8), (-101.8, 74.2)],
    # Small islands
    [(-96.5, 77.4), (-96.0, 77.6), (-95.6, 77.4), (-95.8, 77.0), (-96.3, 77.0)],
    [(-94.7, 76.3), (-94.1, 76.5), (-93.7, 76.2), (-94.0, 75.9), (-94.5, 76.0)],
]


def point_in_polygon(lon, lat, polygon):
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > lat) != (yj > lat):
            crossing_lon = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing_lon:
                inside = not inside
        j = i
    return inside


def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    matrix = np.zeros((61, 241), dtype=np.uint8)
    for row in range(61):
        lat = 79.0 - row * 0.1
        for col in range(241):
            lon = -105.0 + col * 0.1
            matrix[row, col] = any(
                point_in_polygon(lon, lat, polygon)
                for polygon in LAND_POLYGONS
            )
    with (out / "candidate.csv").open("w", encoding="ascii", newline="") as f:
        for values in matrix:
            f.write(",".join(map(str, values.tolist())) + "\n")
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(out / "candidate.png")
    image.resize((1928, 488), Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )


if __name__ == "__main__":
    main()
