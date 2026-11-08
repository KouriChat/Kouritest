from pathlib import Path
import csv

import numpy as np
from PIL import Image


# Hand-drawn WGS84 approximations only. Coordinates are (longitude, latitude).
WEST, EAST = -131.0, -129.0
SOUTH, NORTH = 53.0, 55.0
STEP = 2.0 / 126.0

# Mainland east of the fjorded outer coast. The coast is intentionally
# generalized to the resolution of the sampling grid.
MAINLAND = [
    (-130.48, 55.00), (-130.38, 54.90), (-130.50, 54.82),
    (-130.31, 54.75), (-130.45, 54.64), (-130.27, 54.55),
    (-130.42, 54.43), (-130.18, 54.32), (-130.30, 54.20),
    (-130.13, 54.10), (-130.31, 53.99), (-130.06, 53.88),
    (-130.20, 53.77), (-129.99, 53.66), (-130.08, 53.55),
    (-129.89, 53.43), (-129.96, 53.32), (-129.75, 53.20),
    (-129.72, 53.00), (EAST, SOUTH), (EAST, NORTH), (-130.70, NORTH),
]

# Major offshore islands inside the boxed area.
ISLANDS = [
    # Banks Island
    [(-130.10, 54.59), (-129.91, 54.61), (-129.72, 54.53),
     (-129.64, 54.39), (-129.67, 54.22), (-129.78, 54.08),
     (-129.96, 54.02), (-130.12, 54.10), (-130.18, 54.27),
     (-130.20, 54.44)],
    # Dundas Island group
    [(-130.74, 54.73), (-130.56, 54.78), (-130.42, 54.71),
     (-130.45, 54.59), (-130.61, 54.56), (-130.76, 54.63)],
    [(-130.82, 54.57), (-130.70, 54.60), (-130.64, 54.51),
     (-130.72, 54.44), (-130.83, 54.47)],
    # Porcher Island
    [(-130.62, 53.96), (-130.45, 54.00), (-130.31, 53.94),
     (-130.30, 53.83), (-130.40, 53.75), (-130.56, 53.78),
     (-130.66, 53.86)],
    # Smaller islands along the outer coast and Banks Island channels
    [(-130.43, 54.32), (-130.32, 54.37), (-130.24, 54.30),
     (-130.28, 54.21), (-130.39, 54.22)],
    [(-130.26, 53.63), (-130.14, 53.67), (-130.05, 53.59),
     (-130.10, 53.50), (-130.22, 53.51)],
    [(-129.92, 53.88), (-129.82, 53.91), (-129.76, 53.84),
     (-129.80, 53.76), (-129.90, 53.78)],
]
# Water intrusions into the generalized mainland polygon (major inlet mouths).
INLETS = [
    # Portland Inlet and its approach
    [(-130.48, 54.96), (-130.25, 54.98), (-130.04, 54.94),
     (-129.92, 54.88), (-130.05, 54.84), (-130.27, 54.87),
     (-130.47, 54.86)],
    # Khutzeymateen / Work Channel approaches
    [(-130.42, 54.81), (-130.19, 54.83), (-130.02, 54.77),
     (-130.11, 54.72), (-130.34, 54.73)],
    # Skeena estuary and Chatham Sound indentations
    [(-130.35, 54.29), (-130.10, 54.31), (-129.94, 54.25),
     (-129.89, 54.18), (-130.02, 54.14), (-130.25, 54.18)],
    # Kitkatla / Estevan Sound
    [(-130.30, 53.98), (-130.07, 54.01), (-129.93, 53.95),
     (-129.91, 53.88), (-130.07, 53.85), (-130.25, 53.89)],
    # Principe Channel / narrow coastal arms
    [(-130.17, 53.73), (-129.94, 53.76), (-129.81, 53.70),
     (-129.78, 53.63), (-129.93, 53.60), (-130.14, 53.64)],
]


def inside(point, polygon):
    """Ray-casting point-in-polygon test; boundary points count as land."""
    x, y = point
    hit = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)):
            cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < cross:
                hit = not hit
        j = i
    return hit


def is_land(lon, lat):
    if any(inside((lon, lat), island) for island in ISLANDS):
        return True
    if not inside((lon, lat), MAINLAND):
        return False
    return not any(inside((lon, lat), inlet) for inlet in INLETS)


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    coords = [NORTH - i * STEP for i in range(127)]
    lons = [WEST + i * STEP for i in range(127)]
    matrix = np.array([[int(is_land(lon, lat)) for lon in lons]
                       for lat in coords], dtype=np.uint8)

    with (output_dir / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Mode 1 is black (water); mode 0 is white (land).
    image = Image.fromarray((1 - matrix) * 255, mode="L").convert("1")
    image.save(output_dir / "candidate.png")
    enlarged = image.resize((1016, 1016), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")


if __name__ == "__main__":
    main()
