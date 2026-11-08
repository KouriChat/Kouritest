import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image

# Hand-estimated WGS84 lon/lat outlines. Longitudes are west degrees.
# Vertices are intentionally approximate and restricted to the requested area.
LAND = [
    # Victoria Island: eastern half and south-eastern coast.
    [(105, 79), (102.0, 78.7), (101.0, 78.0), (100.4, 77.3),
     (100.0, 76.5), (99.5, 75.8), (98.7, 75.0), (97.7, 74.5),
     (96.3, 74.0), (96.0, 73.0), (105, 73)],
    # Prince of Wales Island.
    [(98.0, 76.9), (96.7, 77.0), (95.3, 76.5), (94.8, 75.8),
     (95.2, 74.9), (96.1, 74.1), (97.1, 73.5), (98.2, 73.5),
     (98.7, 74.2), (98.5, 75.3), (99.0, 76.2)],
    # Somerset Island.
    [(94.2, 75.9), (92.8, 76.1), (91.5, 75.8), (91.0, 75.0),
     (91.2, 74.1), (92.1, 73.4), (93.4, 72.9), (94.8, 73.1),
     (95.0, 74.0), (94.8, 75.0)],
    # Boothia Peninsula (the northern peninsula within this frame).
    [(93.2, 73.9), (92.0, 73.6), (90.9, 73.2), (89.8, 72.8),
     (88.4, 72.5), (87.0, 72.3), (85.5, 72.2), (84.0, 72.1),
     (82.5, 72.0), (81.0, 72.0), (81.0, 73.0), (85.0, 73.0),
     (88.0, 73.4), (90.0, 73.8), (92.0, 74.0)],
    # Cornwallis Island.
    [(96.1, 76.5), (95.2, 76.7), (94.4, 76.4), (94.0, 75.9),
     (94.4, 75.5), (95.2, 75.4), (96.0, 75.8)],
    # Bathurst Island, eastern portion in the box.
    [(100.5, 78.5), (98.8, 78.4), (97.4, 78.0), (96.8, 77.4),
     (97.4, 76.9), (98.8, 76.8), (100.1, 77.2)],
    # Devon Island.
    [(91.7, 78.7), (89.8, 78.6), (88.5, 78.1), (87.4, 77.6),
     (86.4, 76.7), (85.5, 75.7), (84.2, 75.1), (82.6, 74.8),
     (81.0, 74.7), (81.0, 77.2), (83.0, 77.7), (85.0, 78.2),
     (88.0, 78.8), (90.0, 79.0)],
    # Ellesmere Island, western and southern outline.
    [(81.0, 79.0), (79.5, 78.9), (78.0, 78.4), (77.0, 77.8),
     (76.0, 77.1), (75.0, 76.2), (74.8, 75.3), (75.5, 74.7),
     (77.0, 74.3), (78.7, 74.1), (80.2, 74.0), (81.0, 74.2)],
    # Axel Heiberg Island.
    [(93.0, 80.0), (91.5, 80.0), (90.0, 79.6), (88.7, 79.0),
     (88.2, 78.3), (89.1, 77.8), (90.5, 77.6), (91.9, 78.0),
     (93.0, 78.8), (94.0, 79.5)],
    # Amund Ringnes Island.
    [(96.2, 79.8), (94.7, 79.7), (93.5, 79.3), (93.1, 78.8),
     (94.0, 78.4), (95.3, 78.5), (96.4, 78.9)],
    # Small islands in the central archipelago.
    [(97.3, 79.4), (96.7, 79.5), (96.3, 79.2), (96.7, 78.9), (97.3, 79.0)],
    [(98.8, 78.9), (98.2, 79.0), (97.9, 78.7), (98.4, 78.5), (98.9, 78.6)],
    [(95.0, 77.9), (94.4, 78.0), (94.2, 77.7), (94.7, 77.5), (95.1, 77.6)],
    [(92.5, 77.4), (91.9, 77.5), (91.6, 77.2), (92.1, 77.0), (92.6, 77.1)],
    [(99.5, 76.2), (99.1, 76.3), (98.9, 76.0), (99.3, 75.9)],
    [(88.1, 76.0), (87.8, 76.1), (87.6, 75.8), (87.9, 75.7)],
    [(85.0, 76.0), (84.6, 76.1), (84.4, 75.8), (84.8, 75.7)],
]


def on_segment(x, y, a, b):
    cross = (x-a[0])*(b[1]-a[1]) - (y-a[1])*(b[0]-a[0])
    return abs(cross) < 1e-9 and min(a[0], b[0])-1e-9 <= x <= max(a[0], b[0])+1e-9 and min(a[1], b[1])-1e-9 <= y <= max(a[1], b[1])+1e-9


def inside(lon_w, lat, polygon):
    hit = False
    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]
        if on_segment(lon_w, lat, a, b):
            return True
        if (a[1] > lat) != (b[1] > lat):
            cross_lon = a[0] + (lat-a[1]) * (b[0]-a[0]) / (b[1]-a[1])
            if lon_w < cross_lon:
                hit = not hit
    return hit
def main():
    lats = [79.0 - 0.1 * i for i in range(61)]
    lons_w = [105.0 - 0.1 * i for i in range(241)]
    matrix = np.zeros((61, 241), dtype=np.uint8)
    for r, lat in enumerate(lats):
        for c, lon_w in enumerate(lons_w):
            matrix[r, c] = int(any(inside(lon_w, lat, p) for p in LAND))

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # 1 is white land; 0 is black water.
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(out / "candidate.png")
    image.resize((241 * 8, 61 * 8), Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )

    assert matrix.shape == (61, 241)
    assert matrix.size == 14701 and np.isin(matrix, (0, 1)).all()
    assert image.size == (241, 61)
    assert set(image.getdata()) == {0, 255}
    enlarged = Image.open(out / "candidate_8x.png")
    assert enlarged.size == (1928, 488)
    assert set(enlarged.getdata()) == {0, 255}
    for row in csv.reader((out / "candidate.csv").open(newline="")):
        assert len(row) == 241 and set(row) <= {"0", "1"}


if __name__ == "__main__":
    main()
