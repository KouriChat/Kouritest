from pathlib import Path

import numpy as np
from PIL import Image


# Hand-estimated coastline polygons, in (longitude, latitude) degrees.
# These are geographic approximations, not imported map data.
LAND = [
    # Victoria Island: northeastern tip within the western edge of this frame.
    [(-105.0, 74.30), (-104.0, 74.15), (-102.8, 73.85),
     (-101.5, 73.45), (-101.0, 73.05), (-102.0, 72.8),
     (-104.0, 73.0), (-105.0, 73.2)],
    # Prince of Wales Island, clipped at the southern boundary.
    [(-102.0, 74.05), (-100.6, 74.0), (-99.1, 73.65),
     (-97.4, 73.0), (-99.1, 72.8), (-101.0, 73.0)],
    # Melville Island, chiefly its eastern portion.
    [(-105.0, 75.75), (-104.4, 75.55), (-103.0, 75.5),
     (-101.8, 75.25), (-100.6, 75.05), (-99.8, 74.7),
     (-100.4, 74.25), (-102.1, 74.25), (-103.4, 74.55),
     (-104.7, 74.65), (-105.0, 74.9)],
    # Bathurst Island.
    [(-100.7, 76.55), (-99.7, 76.45), (-98.8, 76.0),
     (-97.8, 75.8), (-97.4, 75.35), (-98.0, 75.0),
     (-99.0, 75.05), (-99.8, 75.4), (-100.6, 75.85)],
    # Cornwallis Island.
    [(-96.7, 76.25), (-95.8, 76.2), (-94.9, 75.85),
     (-94.5, 75.45), (-95.0, 75.1), (-96.0, 75.15),
     (-96.8, 75.55)],
    # Little Cornwallis and nearby islands.
    [(-96.1, 75.0), (-95.5, 75.05), (-95.2, 74.85),
     (-95.7, 74.72), (-96.15, 74.82)],
    # Somerset Island.
    [(-94.9, 74.8), (-93.7, 74.85), (-92.4, 74.55),
     (-91.3, 74.15), (-90.8, 73.6), (-91.5, 73.2),
     (-93.0, 73.15), (-94.0, 73.55), (-94.8, 74.0)],
    # Devon Island.
    [(-90.7, 76.15), (-89.5, 76.2), (-87.9, 76.0),
     (-86.0, 75.7), (-84.2, 75.55), (-82.5, 75.1),
     (-82.0, 74.55), (-82.8, 74.2), (-84.0, 74.0),
     (-85.7, 74.15), (-87.2, 74.45), (-88.7, 74.75),
     (-90.0, 75.2), (-90.8, 75.65)],
    # Ellesmere Island: western/southwestern part and northern mainland.
    [(-81.0, 79.0), (-84.4, 79.0), (-86.2, 78.8),
     (-87.5, 78.35), (-88.5, 77.9), (-89.2, 77.4),
     (-88.5, 77.0), (-87.5, 76.8), (-86.7, 76.55),
     (-85.8, 76.2), (-84.5, 76.0), (-83.3, 76.0),
     (-81.0, 76.25)],
    # Axel Heiberg Island.
    [(-92.0, 79.0), (-93.5, 78.85), (-94.2, 78.45),
     (-93.8, 77.95), (-92.8, 77.65), (-91.2, 77.7),
     (-90.2, 78.05), (-90.1, 78.55), (-91.0, 78.9)],
    # Ellef Ringnes and Amund Ringnes islands.
    [(-99.4, 78.3), (-98.2, 78.25), (-97.1, 77.9),
     (-96.7, 77.45), (-97.3, 77.0), (-98.6, 76.85),
     (-99.8, 77.2), (-100.0, 77.75)],
    [(-96.5, 78.7), (-95.1, 78.55), (-94.3, 78.2),
     (-94.6, 77.8), (-95.7, 77.65), (-96.7, 78.0)],
    # Borden, Mackenzie King, and smaller islands.
    [(-101.1, 78.8), (-99.7, 78.75), (-99.2, 78.35),
     (-100.0, 78.05), (-101.3, 78.2)],
    [(-106.0, 77.5), (-104.0, 77.45), (-103.5, 77.1),
     (-104.4, 76.85), (-105.8, 77.0)],
    [(-98.0, 77.3), (-97.5, 77.25), (-97.3, 77.0),
     (-97.8, 76.9), (-98.2, 77.05)],
]
def inside_polygon(lon, lat, polygon):
    """Ray-crossing point-in-polygon test."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > lat) != (yj > lat):
            crossing_x = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing_x:
                inside = not inside
        j = i
    return inside


def main():
    latitudes = [79.0 - 0.1 * i for i in range(61)]
    longitudes = [-105.0 + 0.1 * i for i in range(241)]
    matrix = np.zeros((61, 241), dtype=np.uint8)
    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            matrix[row, col] = any(
                inside_polygon(lon, lat, polygon) for polygon in LAND
            )

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="") as f:
        for row in matrix:
            f.write(",".join(map(str, row.tolist())) + "\n")
    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(out / "candidate.png")
    image.resize((241 * 8, 61 * 8), Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )


if __name__ == "__main__":
    main()
