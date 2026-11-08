import csv
from pathlib import Path

import numpy as np
from PIL import Image

# Approximate hand-drawn outlines in (longitude, latitude).
LAND = [
    # Ellesmere Island.
    [(-91.8, 76.1), (-91.0, 76.5), (-90.4, 77.0), (-89.4, 77.2),
     (-88.9, 77.8), (-87.7, 78.1), (-86.7, 78.7), (-85.5, 79.0),
     (-81.0, 79.4), (-81.0, 76.2), (-82.0, 76.1), (-83.0, 76.4),
     (-84.1, 76.5), (-85.0, 76.2), (-86.0, 75.8), (-87.2, 75.6),
     (-88.2, 75.7), (-89.1, 76.0), (-90.0, 75.9), (-90.8, 75.8)],
    # Devon Island.
    [(-90.2, 74.6), (-89.5, 74.8), (-88.7, 75.1), (-87.6, 75.0),
     (-86.7, 75.2), (-85.8, 75.1), (-84.8, 75.4), (-83.8, 75.5),
     (-82.8, 75.4), (-81.0, 75.9), (-81.0, 77.2), (-82.4, 77.0),
     (-83.3, 76.7), (-84.3, 76.8), (-85.1, 76.5), (-86.1, 76.6),
     (-87.0, 76.3), (-88.0, 76.5), (-89.0, 76.2), (-89.7, 75.9),
     (-90.4, 75.5)],
    # Axel Heiberg Island.
    [(-94.2, 78.0), (-93.4, 78.4), (-92.5, 78.5), (-91.8, 78.9),
     (-90.9, 79.0), (-90.0, 79.5), (-88.9, 79.4), (-87.7, 79.7),
     (-87.2, 79.2), (-88.0, 78.9), (-88.3, 78.4), (-89.4, 78.2),
     (-90.1, 77.8), (-91.2, 77.8), (-92.0, 77.5), (-93.1, 77.6),
     (-93.8, 77.8)],
    # Amund Ringnes Island.
    [(-95.0, 77.1), (-94.2, 77.4), (-93.2, 77.4), (-92.7, 77.8),
     (-91.7, 78.0), (-90.8, 77.8), (-90.0, 77.6), (-90.3, 77.2),
     (-91.2, 77.1), (-92.0, 76.9), (-93.2, 77.0), (-94.1, 76.9)],
    # Ellef Ringnes Island.
    [(-103.0, 77.0), (-102.7, 77.5), (-101.8, 77.7), (-101.0, 78.0),
     (-100.1, 78.0), (-99.3, 77.8), (-98.7, 77.5), (-98.8, 77.1),
     (-99.5, 76.9), (-100.4, 76.8), (-101.1, 76.6), (-102.0, 76.7)],
    # Bathurst Island.
    [(-98.6, 75.7), (-98.0, 76.1), (-97.7, 76.6), (-96.9, 76.9),
     (-96.0, 76.8), (-95.2, 76.5), (-94.7, 76.1), (-94.4, 75.7),
     (-95.2, 75.5), (-96.0, 75.6), (-96.8, 75.4), (-97.8, 75.5)],
    # Melville Island eastern tip, clipped by the western frame.
    [(-105.0, 75.0), (-104.4, 75.2), (-104.0, 75.7), (-104.1, 76.3),
     (-104.7, 76.8), (-105.0, 76.9)],
    # Prince of Wales Island.
    [(-99.8, 73.9), (-99.5, 74.5), (-99.0, 74.9), (-98.6, 75.5),
     (-98.0, 76.0), (-97.3, 76.2), (-96.6, 76.0), (-96.0, 75.6),
     (-96.0, 75.0), (-96.5, 74.6), (-96.9, 74.2), (-97.5, 73.9),
     (-98.6, 73.8), (-99.3, 73.7)],
    # Somerset Island.
    [(-96.2, 73.0), (-96.0, 73.5), (-95.5, 74.0), (-95.4, 74.5),
     (-94.8, 74.8), (-94.0, 74.9), (-93.1, 74.7), (-92.5, 74.3),
     (-92.2, 73.8), (-92.4, 73.0)],
    # Boothia Peninsula.
    [(-96.0, 73.0), (-95.4, 73.5), (-95.0, 74.0), (-94.7, 74.5),
     (-94.1, 75.0), (-93.7, 75.5), (-92.8, 75.8), (-91.9, 75.7),
     (-91.2, 75.3), (-91.0, 74.8), (-91.7, 74.3), (-92.2, 73.8),
     (-92.4, 73.0)],
    # Mainland north coast, with the land continuing south of the frame.
    [(-105.0, 73.0), (-105.0, 72.0), (-102.0, 71.7), (-99.0, 71.9),
     (-96.0, 72.0), (-93.0, 71.8), (-90.0, 71.9), (-87.0, 72.1),
     (-84.0, 72.4), (-81.0, 72.5), (-81.0, 73.0)],
    # Baffin Island western edge.
    [(-82.0, 73.0), (-82.4, 73.5), (-82.6, 74.0), (-82.5, 74.7),
     (-82.2, 75.2), (-81.7, 75.6), (-81.0, 75.8), (-81.0, 73.0)],
    # Cornwallis Island and several coarse small-island outlines.
    [(-96.6, 74.9), (-96.3, 75.2), (-95.8, 75.4), (-95.2, 75.3),
     (-94.8, 75.0), (-95.0, 74.7), (-95.7, 74.6), (-96.3, 74.7)],
    [(-100.0, 75.0), (-99.8, 75.2), (-99.5, 75.2), (-99.4, 74.9),
     (-99.7, 74.8)],
    [(-94.4, 76.4), (-94.2, 76.6), (-93.9, 76.5), (-94.0, 76.3)],
    [(-91.0, 76.8), (-90.8, 77.0), (-90.5, 76.9), (-90.6, 76.7)],
]


def on_segment(lon, lat, a, b, eps=1e-10):
    ax, ay = a
    bx, by = b
    cross = (lon - ax) * (by - ay) - (lat - ay) * (bx - ax)
    if abs(cross) > eps:
        return False
    return (min(ax, bx) - eps <= lon <= max(ax, bx) + eps and
            min(ay, by) - eps <= lat <= max(ay, by) + eps)


def in_polygon(lon, lat, polygon):
    inside = False
    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]
        if on_segment(lon, lat, a, b):
            return True
        ax, ay = a
        bx, by = b
        if (ay > lat) != (by > lat):
            crossing_lon = ax + (lat - ay) * (bx - ax) / (by - ay)
            if lon < crossing_lon:
                inside = not inside
    return inside


def is_land(lon, lat):
    return any(in_polygon(lon, lat, polygon) for polygon in LAND)


def main():
    rows, cols = 61, 241
    matrix = np.zeros((rows, cols), dtype=np.uint8)
    for r in range(rows):
        lat = 79.0 - 0.1 * r
        for c in range(cols):
            lon = -105.0 + 0.1 * c
            matrix[r, c] = int(is_land(lon, lat))

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(out / "candidate.png")
    image.resize((1928, 488), resample=Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )

    assert matrix.shape == (61, 241) and matrix.size == 14701
    assert set(np.unique(matrix)).issubset({0, 1})


if __name__ == "__main__":
    main()
