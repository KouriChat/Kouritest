#!/usr/bin/env python3
"""Knowledge-based, hand-drawn approximation of land in the Canadian Arctic."""

from pathlib import Path

import numpy as np
from PIL import Image


# Coordinates are (degrees west, degrees north). Polygons are deliberately
# approximate and are the complete geography rules used by this generator.
LAND = [
    # Northern mainland coast, clipped by the task's southern frame.
    [(105, 73), (105, 72.4), (103.5, 72.2), (102, 72.5),
     (100.5, 72.6), (99, 72.7), (97, 72.5), (95, 72.4),
     (94, 72.6), (92.5, 72.7), (91, 72.5), (90, 72.4),
     (89, 72.8), (88, 72.8), (87, 72.5), (86, 72.3),
     (85, 72.1), (84, 71.8), (83, 71.6), (82, 71.3),
     (81, 71.2), (81, 73), (105, 73)],
    # Prince of Wales Island.
    [(99.8, 72.1), (98.8, 72.35), (97.7, 72.5), (96.8, 73.0),
     (97.0, 73.5), (97.8, 74.2), (98.2, 74.75), (99.0, 74.9),
     (99.8, 74.6), (100.0, 73.8), (100.3, 73.1), (100.2, 72.5)],
    # Somerset Island.
    [(94.6, 72.2), (93.6, 72.35), (92.6, 72.65), (92.0, 73.0),
     (92.1, 73.6), (92.4, 74.0), (93.1, 74.2), (93.8, 73.95),
     (94.5, 73.45), (94.8, 72.9)],
    # Boothia Peninsula, joined to mainland at its southern end.
    [(95.4, 72.25), (94.3, 72.45), (93.2, 72.65), (92.4, 73.0),
     (91.7, 73.4), (91.0, 73.8), (90.2, 73.95), (89.8, 73.65),
     (90.2, 73.3), (90.7, 72.9), (91.2, 72.5), (92.5, 72.25)],
    # Devon Island.
    [(89.8, 74.45), (88.8, 74.55), (87.7, 74.65), (86.8, 74.85),
     (85.7, 75.0), (84.9, 75.2), (84.3, 75.65), (83.8, 76.15),
     (83.6, 76.55), (84.0, 76.85), (84.8, 76.9), (85.7, 76.75),
     (86.6, 76.6), (87.7, 76.35), (88.6, 76.1), (89.3, 75.7),
     (89.8, 75.2)],
    # Ellesmere Island, eastern edge of the archipelago.
    [(89.6, 76.4), (88.7, 76.65), (87.7, 76.85), (86.8, 77.1),
     (85.7, 77.35), (84.7, 77.55), (83.8, 77.85), (82.9, 78.2),
     (82.1, 78.55), (81.5, 78.9), (81, 79.3), (81, 78.4),
     (80.4, 77.6), (80.2, 76.9), (80.7, 76.3), (81.5, 75.9),
     (82.5, 75.7), (83.4, 75.65), (84.2, 75.8), (85.2, 75.95),
     (86.2, 76.0), (87.3, 76.05), (88.5, 76.2), (89.2, 76.35),
     (90.0, 76.5)],
]
LAND.extend([
    # Axel Heiberg Island.
    [(93.8, 77.45), (92.8, 77.65), (91.8, 77.8), (90.8, 78.0),
     (89.8, 78.3), (89.0, 78.7), (88.8, 79.2), (89.1, 79.8),
     (89.0, 80.5), (89.4, 81.1), (90.2, 81.4), (91.1, 81.35),
     (92.0, 81.1), (92.8, 80.8), (93.5, 80.3), (94.0, 79.7),
     (94.2, 79.0), (94.1, 78.4)],
    # Ellef Ringnes Island.
    [(101.3, 78.15), (100.6, 78.25), (99.8, 78.5), (99.4, 78.9),
     (99.7, 79.35), (100.3, 79.55), (101.0, 79.45),
     (101.6, 79.15), (101.8, 78.75), (101.6, 78.4)],
    # Amund Ringnes Island.
    [(96.5, 78.05), (95.7, 78.25), (94.9, 78.55), (94.4, 78.95),
     (94.6, 79.35), (95.2, 79.6), (96.0, 79.5),
     (96.8, 79.2), (97.1, 78.8), (96.9, 78.4)],
    # Lougheed Island.
    [(105.0, 78.0), (104.5, 78.15), (104.0, 78.45),
     (103.8, 78.8), (104.2, 79.05), (104.8, 78.95)],
    # Meighen Island.
    [(99.5, 79.9), (98.9, 80.05), (98.6, 80.35),
     (99.0, 80.55), (99.6, 80.5), (100.0, 80.25)],
    # Bathurst Island.
    [(99.7, 74.95), (98.8, 75.05), (97.8, 75.25), (96.9, 75.6),
     (96.4, 76.05), (96.7, 76.5), (97.3, 76.7), (98.1, 76.55),
     (98.9, 76.35), (99.7, 76.1), (100.3, 75.75), (100.4, 75.35)],
    # Cornwallis Island.
    [(96.2, 74.95), (95.5, 75.05), (95.0, 75.3), (95.1, 75.65),
     (95.6, 75.85), (96.2, 75.75), (96.6, 75.45), (96.6, 75.15)],
    # North Devon satellite islands.
    [(91.8, 75.0), (91.3, 75.15), (91.1, 75.45),
     (91.5, 75.65), (92.0, 75.45)],
    # Small central archipelago islands.
    [(101.0, 76.0), (100.2, 76.2), (99.8, 76.55), (100.1, 76.85),
     (100.8, 76.75), (101.4, 76.45)],
    [(103.0, 76.7), (102.4, 76.85), (102.1, 77.15),
     (102.5, 77.4), (103.1, 77.25)],
])


def inside_polygon(x, y, polygon):
    """Ray-crossing point-in-polygon test; boundary points are land."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        cross = (yi > y) != (yj > y)
        if cross and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside
def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    rows, cols = 61, 241
    matrix = np.zeros((rows, cols), dtype=np.uint8)

    # Integer tenths avoid floating-point drift in the sampling coordinates.
    for r in range(rows):
        latitude = (790 - r) / 10
        for c in range(cols):
            west = (1050 - c) / 10
            matrix[r, c] = any(
                inside_polygon(west, latitude, polygon) for polygon in LAND
            )

    with (out / "candidate.csv").open("w", encoding="ascii", newline="") as f:
        for row in matrix:
            f.write(",".join(str(int(value)) for value in row) + "\n")

    image = Image.fromarray(matrix * 255, mode="L")
    image.save(out / "candidate.png")
    image.resize((1928, 488), resample=Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )

    assert matrix.shape == (61, 241) and matrix.size == 14701
    assert set(np.unique(matrix)).issubset({0, 1})
    assert image.size == (241, 61) and set(image.getdata()).issubset({0, 255})
    with (out / "candidate.csv").open("r", encoding="ascii") as f:
        lines = f.read().splitlines()
    assert len(lines) == 61
    assert all(len(line.split(",")) == 241 for line in lines)
    assert all(set(line.split(",")) <= {"0", "1"} for line in lines)
    enlarged = Image.open(out / "candidate_8x.png")
    assert enlarged.size == (1928, 488)
    assert set(enlarged.getdata()).issubset({0, 255})
    print("validated: 61x241, 14701 binary values; PNGs 241x61 and 1928x488")


if __name__ == "__main__":
    main()
