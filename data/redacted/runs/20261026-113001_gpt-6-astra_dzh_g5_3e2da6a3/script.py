"""Knowledge-only Mediterranean land/water reconstruction.

The coordinates below are deliberately hand-authored approximations.  No
geographic data or network access is used by this generator.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

MIN_LON, MAX_LON = -9.0, 37.0
MIN_LAT, MAX_LAT = 31.0, 45.0
STEP = 0.2
WIDTH, HEIGHT = 231, 71

# Each ring is a rough coastline, in (longitude, latitude) order.
LAND = []
WATER = []

LAND.append([
    (-9.0, 45.0), (26.0, 45.0), (26.0, 39.0), (25.0, 38.0),
    (24.0, 37.4), (23.0, 37.0), (22.1, 37.2), (21.2, 38.0),
    (20.0, 39.0), (18.5, 40.2), (17.0, 41.0), (15.8, 41.2),
    (14.5, 42.0), (13.0, 42.5), (12.0, 43.0), (11.0, 43.6),
    (10.0, 44.5), (8.7, 44.5), (7.5, 43.8), (6.5, 43.3),
    (5.2, 43.0), (4.0, 43.3), (3.0, 43.5), (2.0, 43.3),
    (1.0, 42.8), (0.0, 42.7), (-1.0, 43.0), (-2.0, 43.3),
    (-3.0, 43.6), (-4.0, 43.5), (-5.0, 43.6), (-6.0, 43.8),
    (-7.0, 44.0), (-8.0, 44.4), (-9.0, 44.6)
])

# Iberia: Atlantic/northern coast, then Mediterranean and back along the north.
LAND.append([
    (-9.0, 43.8), (-8.0, 43.5), (-7.4, 43.0), (-6.8, 42.4),
    (-6.0, 41.6), (-5.3, 40.2), (-4.5, 39.0), (-3.4, 37.5),
    (-2.0, 36.7), (-1.0, 36.5), (0.0, 37.0), (1.0, 37.6),
    (2.0, 38.5), (3.0, 40.0), (3.2, 41.5), (3.0, 43.5),
    (2.0, 43.5), (1.0, 43.3), (0.0, 43.0), (-1.0, 43.0),
    (-2.0, 43.3), (-3.0, 43.5), (-4.0, 43.5), (-5.0, 43.7),
    (-6.0, 43.8), (-7.0, 44.0), (-8.0, 44.2)
])

# Italy, including the mainland boot and the toe.
LAND.append([
    (7.0, 44.3), (8.5, 44.1), (10.0, 44.0), (11.5, 43.2),
    (12.8, 42.0), (13.8, 41.2), (14.5, 40.5), (15.3, 39.8),
    (16.2, 39.5), (16.5, 38.8), (15.8, 38.2), (15.0, 38.0),
    (14.3, 38.5), (13.5, 39.0), (13.0, 40.0), (12.0, 41.0),
    (11.0, 42.0), (10.0, 43.0), (8.5, 43.5)
])

# Turkey and the eastern Mediterranean mainland.
LAND.append([
    (26.0, 39.0), (27.0, 38.0), (28.0, 37.2), (29.0, 36.8),
    (30.0, 36.0), (31.0, 36.0), (32.0, 36.2), (33.0, 36.0),
    (34.0, 36.1), (35.0, 36.5), (36.0, 36.6), (37.0, 36.5),
    (37.0, 45.0), (26.0, 45.0)
])

# North Africa, from Morocco across the Maghreb and Libya.
LAND.append([
    (-5.5, 31.0), (-5.5, 35.8), (-4.5, 35.9), (-3.0, 35.7),
    (-1.5, 35.2), (0.0, 35.0), (1.0, 35.1), (2.0, 36.0),
    (3.0, 36.8), (4.0, 36.8), (5.0, 37.0), (6.0, 36.7),
    (7.0, 37.0), (8.0, 37.1), (9.0, 37.2), (10.0, 37.0),
    (11.0, 37.0), (11.5, 36.5), (12.0, 35.7), (13.0, 34.8),
    (14.0, 33.5), (15.0, 32.5), (16.0, 32.1), (17.0, 32.5),
    (18.0, 32.5), (19.0, 31.8), (21.0, 31.0)
])

# Egypt, the Levant, and the Sinai edge visible inside the requested frame.
LAND.append([
    (25.0, 31.0), (25.0, 32.5), (26.0, 32.3), (27.0, 31.8),
    (28.0, 31.4), (29.0, 31.3), (30.0, 31.2), (31.0, 31.3),
    (32.0, 31.5), (33.0, 31.5), (34.0, 31.2), (35.0, 31.0),
    (37.0, 31.0), (37.0, 33.0), (36.0, 33.5), (35.0, 34.0),
    (34.0, 34.5), (33.0, 35.0), (32.0, 35.5), (31.0, 35.8),
    (30.0, 35.5), (29.0, 35.0), (28.0, 34.5), (27.0, 34.0),
    (26.0, 33.0)
])
# Major islands and island groups, hand-drawn as small closed rings.
LAND.extend([
    # Sicily, Sardinia, Corsica
    [(12.4, 38.2), (13.5, 38.3), (15.2, 37.2), (15.1, 36.6),
     (13.8, 36.7), (12.5, 37.2)],
    [(8.1, 41.2), (9.8, 41.2), (9.7, 39.0), (8.5, 38.8),
     (8.1, 39.8)],
    [(8.5, 43.0), (9.6, 43.0), (9.5, 41.4), (8.8, 41.3)],
    # Balearics, Malta, Pantelleria, Lampedusa
    [(1.1, 39.9), (4.4, 40.1), (4.3, 38.6), (1.2, 38.7)],
    [(14.1, 36.1), (14.6, 36.1), (14.6, 35.8), (14.2, 35.8)],
    [(11.9, 36.9), (12.0, 36.9), (12.0, 36.7), (11.9, 36.7)],
    [(12.5, 35.6), (12.7, 35.6), (12.7, 35.4), (12.5, 35.4)],
    # Crete, Rhodes, Dodecanese and Aegean island scatter
    [(23.4, 35.7), (26.3, 35.7), (26.2, 34.8), (23.5, 34.9)],
    [(27.7, 36.5), (28.3, 36.5), (28.3, 35.8), (27.8, 35.8)],
    [(25.0, 39.0), (25.8, 39.0), (25.8, 38.4), (25.1, 38.4)],
    [(24.0, 40.0), (24.5, 40.0), (24.5, 39.4), (24.1, 39.4)],
    [(22.5, 39.0), (23.0, 39.0), (23.0, 38.3), (22.5, 38.3)],
    [(26.1, 40.2), (26.7, 40.2), (26.7, 39.5), (26.2, 39.5)],
    [(20.5, 38.0), (21.0, 38.0), (21.0, 37.5), (20.5, 37.5)],
    [(19.4, 39.0), (20.0, 39.0), (20.0, 38.4), (19.4, 38.4)],
    [(24.4, 37.8), (25.0, 37.8), (25.0, 37.2), (24.4, 37.2)],
    # Cyprus
    [(32.2, 35.7), (34.6, 35.7), (34.5, 34.5), (32.3, 34.5)],
    # Adriatic and Ionian islands
    [(14.0, 44.0), (15.0, 44.0), (15.0, 42.0), (14.0, 42.0)],
    [(18.3, 40.5), (19.0, 40.5), (19.0, 39.8), (18.3, 39.8)],
    # Balearic smaller islands and Alboran
    [(3.0, 39.8), (3.5, 39.8), (3.5, 39.4), (3.0, 39.4)],
    [(-3.1, 35.0), (-2.8, 35.0), (-2.8, 34.8), (-3.1, 34.8)],
])

# Inland/near-inland water bodies visibly inside the frame.
WATER.extend([
    # Black Sea
    [(27.0, 42.0), (29.0, 42.8), (32.0, 43.0), (35.0, 42.5),
     (37.0, 41.8), (37.0, 45.0), (27.0, 45.0)],
])
def inside_ring(lon, lat, ring):
    """Ray-crossing point-in-polygon test."""
    inside = False
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat):
            crossing_lon = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing_lon:
                inside = not inside
        j = i
    return inside


def main():
    lons = np.round(np.arange(MIN_LON, MAX_LON + 0.001, STEP), 1)
    lats = np.round(np.arange(MAX_LAT, MIN_LAT - 0.001, -STEP), 1)
    matrix = np.zeros((len(lats), len(lons)), dtype=np.uint8)
    for row, lat in enumerate(lats):
        for col, lon in enumerate(lons):
            matrix[row, col] = int(
                any(inside_ring(lon, lat, ring) for ring in LAND)
                and not any(inside_ring(lon, lat, ring) for ring in WATER)
            )

    assert matrix.shape == (71, 231) and matrix.size == 16401
    assert np.isin(matrix, (0, 1)).all()
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(matrix.tolist())
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(out / "candidate.png")
    enlarged = image.resize((1848, 568), Image.Resampling.NEAREST)
    enlarged.save(out / "candidate_8x.png")

    with (out / "candidate.csv").open(newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 71 and all(len(r) == 231 for r in rows)
    assert all(v in {"0", "1"} for r in rows for v in r)
    assert image.size == (231, 71) and set(image.getdata()) <= {0, 255}
    assert enlarged.size == (1848, 568)


if __name__ == "__main__":
    main()
