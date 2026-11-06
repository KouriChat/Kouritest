from pathlib import Path
import csv

import numpy as np
from PIL import Image


LAT_NORTH = 45.0
LAT_SOUTH = 31.0
LON_WEST = -9.0
LON_EAST = 37.0
STEP = 0.2

ROWS = 71
COLS = 231

# Hand-authored approximate coastlines. Each ring is an ordered sequence
# of (longitude, latitude) vertices. The script uses these rules to classify
# all sample points; it does not perform independent point-by-point reasoning.
LAND_RINGS = [
    # Iberian Peninsula
    [(-9, 43.8), (-8.2, 43.9), (-7, 43.7), (-5.8, 43.5), (-3.2, 43.4),
     (-1.8, 43.4), (0.1, 42.7), (1.5, 42.4), (3.2, 42.4), (3.3, 41.9),
     (2.5, 41.3), (2.0, 40.5), (0.7, 39.8), (0.1, 38.7), (-0.7, 37.7),
     (-1.6, 36.7), (-3.0, 36.7), (-4.4, 36.7), (-5.3, 36.0), (-6.2, 36.0),
     (-7.4, 37.1), (-7.4, 38.0), (-9, 38.7)],
    # Southern France and western Alps
    [(-1.8, 43.4), (0.1, 42.7), (1.5, 42.4), (3.3, 42.4), (4.8, 43.0),
     (6.0, 43.1), (7.0, 43.7), (7.5, 44.1), (7.6, 45.0), (6.0, 45.0),
     (4.5, 45.0), (2.5, 45.0), (0.0, 45.0), (-1.8, 45.0)],
    # Italy and the Italian peninsula
    [(6.6, 45.0), (7.5, 45.0), (8.8, 45.0), (10.5, 45.0), (12.5, 45.0),
     (13.7, 45.0), (13.7, 44.5), (13.1, 43.7), (13.5, 42.8), (13.2, 42.0),
     (14.0, 41.5), (15.2, 41.0), (16.0, 40.0), (17.0, 40.0), (18.0, 40.4),
     (18.5, 39.8), (17.8, 39.0), (17.0, 38.8), (16.2, 38.2), (15.6, 38.3),
     (15.1, 37.9), (15.6, 37.2), (16.3, 36.8), (15.5, 36.7), (14.8, 37.0),
     (13.0, 38.0), (12.5, 38.2), (12.4, 39.0), (13.0, 40.0), (13.0, 41.0),
     (12.4, 41.8), (12.0, 42.5), (11.0, 43.0), (10.5, 43.8), (9.0, 44.0),
     (8.0, 44.3), (7.0, 44.2)],
    # Balkans, Greece, and western Turkey
    [(13.5, 45.0), (16.0, 45.0), (18.0, 45.0), (20.0, 45.0), (22.0, 45.0),
     (24.0, 45.0), (26.0, 45.0), (28.0, 45.0), (30.0, 45.0), (30.0, 41.0),
     (29.0, 40.5), (28.2, 40.3), (27.2, 40.5), (26.5, 40.0), (26.0, 39.5),
     (25.0, 39.0), (24.5, 38.2), (24.0, 37.5), (23.5, 37.0), (23.0, 36.5),
     (22.5, 36.3), (22.0, 37.0), (21.5, 37.8), (21.0, 38.0), (20.5, 37.5),
     (20.0, 37.0), (19.5, 37.5), (19.0, 38.0), (19.0, 39.0), (19.5, 40.0),
     (19.0, 41.0), (18.5, 42.0), (17.5, 42.8), (16.0, 43.0), (15.0, 44.0)],
    # North Africa, from Morocco through Egypt
    [(-9.0, 31.0), (37.0, 31.0), (34.0, 31.5), (32.5, 31.5), (31.5, 30.0),
     (29.0, 30.0), (27.0, 31.0), (25.0, 31.5), (23.0, 32.0), (21.0, 32.0),
     (19.0, 32.5), (17.0, 32.9), (15.0, 32.7), (13.0, 32.5), (11.0, 33.0),
     (10.0, 33.5), (9.0, 34.0), (8.0, 33.9), (7.0, 33.8), (6.0, 33.8),
     (5.0, 34.2), (4.0, 35.0), (3.0, 35.5), (2.0, 36.0), (1.0, 36.7),
     (0.0, 36.7), (-1.0, 36.0), (-2.0, 35.0), (-3.0, 34.0), (-4.0, 33.0),
     (-5.0, 32.0), (-7.0, 31.5)],
    # Corsica
    [(8.55, 41.35), (9.4, 41.35), (9.55, 42.0), (9.4, 42.8), (9.0, 43.0),
     (8.7, 42.5)],
    # Sardinia
    [(8.1, 38.8), (8.7, 38.8), (9.7, 39.2), (9.8, 40.0), (9.6, 40.8),
     (9.0, 41.2), (8.4, 40.8), (8.1, 39.8)],
    # Sicily
    [(12.4, 36.6), (15.7, 36.6), (15.2, 37.3), (14.5, 38.2), (13.0, 38.2)],
    # Balearic Islands
    [(1.2, 38.6), (1.7, 38.6), (1.7, 39.1), (1.2, 39.1)],
    [(2.3, 39.2), (4.4, 39.2), (4.4, 40.1), (3.3, 40.1), (2.3, 39.8)],
    # Malta
    [(14.1, 35.7), (14.6, 35.7), (14.6, 36.1), (14.1, 36.1)],
    # Crete
    [(23.4, 34.8), (26.3, 34.8), (26.3, 35.7), (25.2, 35.5), (24.0, 35.7)],
    # Cyprus
    [(32.2, 34.5), (34.7, 34.5), (34.7, 35.7), (33.5, 35.8), (32.2, 35.2)],
    # Rhodes
    [(27.7, 35.7), (28.3, 35.7), (28.3, 36.3), (27.7, 36.3)],
    # Corsican/Sicilian small coastal additions omitted where below sample scale
]

# Islands and coastal land groups represented by compact approximate rings.
EXTRA_RINGS = [
    # Adriatic east coast and the Ionian islands are mostly captured by the Balkans ring.
    # Djerba
    [(10.3, 33.4), (11.0, 33.4), (11.0, 33.9), (10.3, 33.9)],
    # Gulf of Gabes islands / coastal land, and eastern Aegean larger islands
    [(26.0, 39.0), (26.6, 39.0), (26.6, 39.5), (26.0, 39.5)],
    [(25.0, 37.5), (25.6, 37.5), (25.6, 38.0), (25.0, 38.0)],
    [(26.5, 37.5), (27.1, 37.5), (27.1, 38.0), (26.5, 38.0)],
]


def point_on_segment(x, y, x1, y1, x2, y2, eps=1e-10):
    dx = x2 - x1
    dy = y2 - y1
    cross = (x - x1) * dy - (y - y1) * dx
    if abs(cross) > eps:
        return False
    dot = (x - x1) * dx + (y - y1) * dy
    return -eps <= dot <= dx * dx + dy * dy + eps


def in_ring(lon, lat, ring):
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if point_on_segment(lon, lat, x1, y1, x2, y2):
            return True
        if (y1 > lat) != (y2 > lat):
            crossing_lon = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < crossing_lon:
                inside = not inside
    return inside


def main():
    lats = [LAT_NORTH - i * STEP for i in range(ROWS)]
    lons = [LON_WEST + j * STEP for j in range(COLS)]
    rings = LAND_RINGS + EXTRA_RINGS

    matrix = np.zeros((ROWS, COLS), dtype=np.uint8)
    for row, lat in enumerate(lats):
        for col, lon in enumerate(lons):
            if any(in_ring(lon, lat, ring) for ring in rings):
                matrix[row, col] = 1

    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    image = Image.fromarray(matrix * 255, mode="L")
    image.save(output_dir / "candidate.png")
    image.resize((COLS * 8, ROWS * 8), resample=Image.Resampling.NEAREST).save(
        output_dir / "candidate_8x.png"
    )


if __name__ == "__main__":
    main()
