from pathlib import Path

import numpy as np
from PIL import Image

# Hand-drawn approximate outlines in (degrees west, latitude).
LAND = [
    # Victoria Island
    [(105, 74.15), (103.7, 74.45), (102, 74.55), (100.5, 74.2),
     (100, 73.7), (101.2, 73.2), (102, 72.6), (103, 72.0),
     (104.1, 71.8), (105, 72.1)],
    # Prince of Wales Island
    [(103.1, 73.9), (102, 73.7), (101.2, 73.25), (101.1, 72.6),
     (101.8, 72.2), (102.7, 72.1), (103.4, 72.5), (103.6, 73.2)],
    # Somerset Island
    [(99.8, 74.15), (98.9, 74.55), (97.3, 74.4), (96.5, 73.9),
     (95.8, 73.2), (95.1, 72.5), (95.7, 71.9), (97, 71.8),
     (98.2, 72.3), (99.1, 72.8), (99.7, 73.45)],
    # Bathurst Island
    [(101, 75.4), (100.1, 75.75), (99.4, 76.3), (98.1, 76.7),
     (97.2, 76.45), (96.8, 75.9), (97.3, 75.4), (98.2, 75.1),
     (99.1, 75), (100.1, 75.1)],
    # Cornwallis Island
    [(96.7, 75.35), (95.8, 75.45), (95.1, 75.1), (94.7, 74.7),
     (95.1, 74.55), (96.1, 74.6), (96.8, 74.9)],
    # Devon Island
    [(90.7, 76.55), (89.6, 76.8), (88.1, 76.7), (86.8, 76.45),
     (85.2, 76.35), (83.8, 76.6), (82.7, 76.9), (82, 77.35),
     (82, 78.1), (83, 78.25), (84.5, 78), (86, 77.9),
     (87.6, 78.05), (89, 77.75), (90.2, 77.25)],
    # Axel Heiberg Island
    [(93.4, 78.8), (92.8, 79), (91.4, 79), (89.8, 78.7),
     (88.8, 78.2), (88.5, 77.7), (89.3, 77.4), (90.5, 77.55),
     (91.8, 77.8), (93, 78.15)],
    # Ellesmere Island, west coast
    [(85.5, 79), (84.2, 78.8), (83.1, 78.45), (82.4, 78),
     (82, 77.4), (82.4, 76.95), (83.4, 76.8), (84, 77.15),
     (84.5, 77.6), (85.3, 78), (86, 78.5)],
    # Baffin Island northwestern edge
    [(83, 73), (82.8, 73.45), (83.2, 73.9), (83.5, 74.5),
     (83.7, 75.2), (84.2, 75.8), (84.1, 76.3), (83.5, 76.65),
     (82.8, 76.45), (82.4, 75.9), (82.2, 75.2), (81.8, 74.5),
     (81.5, 73.8), (81.2, 73)],
    # Mainland coast clipped at the southern frame edge.
    [(105, 73), (103.7, 73), (102.5, 73), (101, 73), (99.5, 73),
     (98, 73), (96.5, 73), (95, 73), (93.5, 73), (92, 73),
     (90.5, 73), (89, 73), (87.5, 73), (86, 73), (85, 73),
     (85, 72.6), (87, 72.35), (90, 72.1), (94, 71.9),
     (98, 71.8), (102, 71.8), (105, 71.9)],
]


def contains(x, y, polygon):
    """Ray cast with boundary points assigned to land."""
    inside = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) < 1e-10 and (
            min(x1, x2) - 1e-10 <= x <= max(x1, x2) + 1e-10
            and min(y1, y2) - 1e-10 <= y <= max(y1, y2) + 1e-10
        ):
            return True
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing_x:
                inside = not inside
    return inside


latitudes = [79.0 - 0.1 * row for row in range(61)]
longitudes_west = [105.0 - 0.1 * col for col in range(241)]
matrix = np.zeros((61, 241), dtype=np.uint8)
for row, lat in enumerate(latitudes):
    for col, lon_west in enumerate(longitudes_west):
        matrix[row, col] = int(any(
            contains(lon_west, lat, polygon) for polygon in LAND
        ))
assert matrix.shape == (61, 241)
assert matrix.size == 14701
assert np.isin(matrix, (0, 1)).all()

out = Path("outputs")
out.mkdir(parents=True, exist_ok=True)

with (out / "candidate.csv").open("w", encoding="ascii", newline="") as f:
    for row in matrix:
        f.write(",".join(str(int(value)) for value in row) + "\n")

pixels = np.where(matrix == 1, 255, 0).astype(np.uint8)
image = Image.fromarray(pixels, mode="L")
image.save(out / "candidate.png")
image.resize((1928, 488), Image.Resampling.NEAREST).save(
    out / "candidate_8x.png"
)
