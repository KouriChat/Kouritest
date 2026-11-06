from pathlib import Path

import numpy as np
from PIL import Image

# Coordinates are in degrees east. Longitudes west of Greenwich are negative.
# Hand-drawn geographic approximations for the Canadian Arctic Archipelago.
# Polygon vertices are (longitude, latitude); no external geographic data is used.
POLYGONS = [
    # Ellesmere Island
    [(-85.0, 79.0), (-75.0, 79.0), (-73.0, 78.2), (-69.0, 77.7),
     (-65.0, 76.8), (-64.0, 75.8), (-66.0, 74.8), (-71.0, 74.3),
     (-77.0, 75.0), (-81.0, 76.0), (-84.5, 77.0), (-85.0, 79.0)],
    # Axel Heiberg Island
    [(-94.0, 79.0), (-87.0, 79.0), (-84.0, 78.5), (-85.0, 77.4),
     (-88.0, 76.7), (-92.0, 76.9), (-94.0, 78.0), (-94.0, 79.0)],
    # North Devon Island
    [(-90.0, 75.8), (-85.0, 76.0), (-82.0, 75.5), (-82.0, 74.9),
     (-85.0, 74.5), (-89.0, 74.8), (-90.0, 75.8)],
    # South Devon Island
    [(-87.5, 74.6), (-82.5, 74.8), (-81.0, 74.5), (-81.0, 73.8),
     (-84.5, 73.5), (-87.0, 73.8), (-87.5, 74.6)],
    # Cornwallis Island
    [(-96.0, 75.5), (-93.0, 75.5), (-92.5, 74.9), (-94.5, 74.6),
     (-96.0, 75.0), (-96.0, 75.5)],
    # Bathurst Island
    [(-100.0, 76.5), (-96.0, 76.3), (-94.0, 75.6), (-95.0, 74.9),
     (-98.0, 74.7), (-101.0, 75.3), (-100.0, 76.5)],
    # Melville Island
    [(-111.0, 76.8), (-105.0, 77.0), (-101.0, 76.3), (-100.0, 75.5),
     (-103.0, 74.9), (-109.0, 75.1), (-112.0, 75.8), (-111.0, 76.8)],
    # Prince Patrick Island
    [(-124.0, 76.5), (-119.0, 76.6), (-117.0, 76.0), (-118.0, 75.2),
     (-122.0, 75.1), (-124.0, 75.7), (-124.0, 76.5)],
    # Banks Island
    [(-125.0, 74.8), (-121.0, 74.8), (-118.0, 73.8), (-119.0, 73.0),
     (-124.0, 73.0), (-126.0, 73.8), (-125.0, 74.8)],
    # Victoria Island, northern portion in the requested frame
    [(-115.0, 73.8), (-110.0, 73.9), (-106.0, 73.5), (-105.0, 73.0),
     (-115.0, 73.0), (-115.0, 73.8)],
]

# Mainland coast and Boothia Peninsula, represented as explicit approximate
# polygons so the western and southern frame edges are classified as well.
MAINLAND = [
    [(-105.0, 73.0), (-105.0, 72.0), (-81.0, 72.0), (-81.0, 73.0),
     (-88.0, 73.0), (-91.0, 73.3), (-95.0, 73.5), (-100.0, 73.3),
     (-105.0, 73.0)],
]
POLYGONS.extend(MAINLAND)

# Boothia Peninsula
POLYGONS.append([
    (-96.0, 73.0), (-94.0, 73.0), (-93.0, 72.0), (-94.0, 71.0),
    (-96.0, 71.0), (-97.0, 72.0), (-96.0, 73.0)
])

latitudes = np.round(np.arange(79.0, 73.0 - 1e-9, -0.1), 1)
longitudes = np.round(np.arange(-105.0, -81.0 + 1e-9, 0.1), 1)


def point_in_polygon(lon, lat, polygon):
    """Ray-casting point-in-polygon test."""
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


matrix = np.zeros((len(latitudes), len(longitudes)), dtype=np.uint8)
for row, lat in enumerate(latitudes):
    for col, lon in enumerate(longitudes):
        if any(point_in_polygon(lon, lat, polygon) for polygon in POLYGONS):
            matrix[row, col] = 1

out_dir = Path("outputs")
out_dir.mkdir(parents=True, exist_ok=True)

with (out_dir / "candidate.csv").open("w", encoding="ascii", newline="") as f:
    for row in matrix:
        f.write(",".join(str(int(value)) for value in row) + "\n")

# Convert land=1 to white and water=0 to black.
image = Image.fromarray(matrix * 255, mode="L")
image.save(out_dir / "candidate.png")
image.resize((241 * 8, 61 * 8), resample=Image.Resampling.NEAREST).save(
    out_dir / "candidate_8x.png"
)
