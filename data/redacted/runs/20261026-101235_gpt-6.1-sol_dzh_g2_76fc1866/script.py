"""Knowledge-based Mediterranean land/water reconstruction.

All geometry below is an intentionally hand-written approximation.  The
classifier evaluates each requested coordinate against these polygons.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

WEST, EAST = -9.0, 37.0
NORTH, SOUTH = 45.0, 31.0
DX = DY = 0.2
COLS, ROWS = 231, 71


def inside(x, y, poly):
    """Ray crossing test, with boundary points treated as land."""
    hit = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        cross = ((yi > y) != (yj > y))
        if cross:
            at_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x <= at_x:
                hit = not hit
        if ((yi == y and yj == y and min(xi, xj) <= x <= max(xi, xj))
                or (xi == x and min(yi, yj) <= y <= max(yi, yj))):
            return True
        j = i
    return hit


def polygon(points):
    return tuple(points)


# Mainland and large peninsulas.  Polygons are deliberately dense near
# familiar Mediterranean capes and gulfs, while the far inland edge is broad.
LAND = [
    # Iberia, including the southern Spanish coast and Gibraltar.
    polygon([
        (-9, 43.25), (-8.7, 43.65), (-7.5, 43.75), (-5.8, 43.55),
        (-3.4, 43.45), (-1.8, 43.35), (-0.1, 42.7), (0.2, 41.5),
        (0.7, 40.5), (0.2, 39.2), (0.3, 38.5), (-0.4, 37.7),
        (-1.3, 37.2), (-2.5, 36.75), (-3.5, 36.7), (-4.8, 36.8),
        (-5.5, 36.5), (-6.0, 36.0), (-7.3, 36.0), (-8.1, 36.55),
        (-8.9, 37.2), (-9.0, 43.25),
    ]),
    # Southern France and the land north of the Ligurian coast.
    polygon([
        (-1.2, 45), (7.5, 45), (7.7, 44.35), (7.45, 43.85),
        (7.0, 43.65), (6.4, 43.45), (5.7, 43.25), (4.9, 43.2),
        (4.0, 43.25), (3.2, 43.35), (2.2, 43.3), (1.1, 43.45),
        (0.3, 43.5), (-0.1, 44.0), (-1.2, 45),
    ]),
    # Alpine and northern Italian mainland, down the Adriatic side.
    polygon([
        (6.4, 45), (13.8, 45), (14.2, 44.6), (14.0, 44.1),
        (14.3, 43.7), (14.5, 43.1), (15.0, 42.6), (15.5, 42.1),
        (16.0, 41.6), (16.2, 41.1), (16.8, 40.6), (17.1, 40.1),
        (16.9, 39.6), (16.5, 39.4), (16.0, 39.6), (15.6, 40.0),
        (15.1, 40.2), (14.7, 40.6), (14.1, 40.8), (13.8, 41.2),
        (13.4, 41.5), (13.1, 42.0), (12.7, 42.35), (12.2, 42.6),
        (11.8, 42.8), (11.4, 42.9), (10.9, 42.8), (10.3, 43.0),
        (9.8, 43.3), (9.4, 43.7), (8.8, 44.1), (8.0, 44.4),
        (7.2, 44.7), (6.4, 45),
    ]),
    # Balkans and Greek mainland, with the Black Sea side cut at the north.
    polygon([
        (13.5, 45), (19.2, 45), (20.4, 44.9), (21.6, 44.7),
        (22.8, 44.9), (24.0, 45), (28.0, 45), (28.6, 44.6),
        (28.3, 43.8), (27.6, 43.4), (26.8, 43.1), (26.1, 42.6),
        (25.6, 41.9), (24.9, 41.6), (24.3, 41.3), (23.8, 41.1),
        (23.5, 40.6), (23.1, 40.3), (22.6, 40.1), (22.1, 39.8),
        (21.6, 39.5), (21.2, 39.1), (20.7, 38.9), (20.0, 39.0),
        (19.5, 39.3), (19.2, 39.7), (19.0, 40.1), (18.5, 40.3),
        (18.1, 40.6), (17.7, 41.0), (17.4, 41.4), (17.0, 41.8),
        (16.4, 42.0), (15.9, 42.4), (15.5, 42.8), (15.0, 43.2),
        (14.4, 43.8), (13.5, 45),
    ]),
]
# continuation of _part01.py
LAND += [
    # Anatolia and the eastern Mediterranean coast.
    polygon([
        (26.0, 41.3), (26.8, 41.5), (27.5, 41.8), (28.2, 42.0),
        (29.0, 42.1), (30.0, 41.9), (31.0, 41.8), (32.0, 41.9),
        (33.0, 42.0), (34.0, 42.3), (35.0, 42.3), (36.0, 42.4),
        (37.0, 42.5), (37.0, 35.8), (36.5, 36.0),
        (35.8, 36.2), (35.1, 36.4), (34.6, 36.5), (34.0, 36.6),
        (33.4, 36.3), (32.7, 36.0), (32.0, 36.1), (31.5, 36.4),
        (31.0, 36.7), (30.4, 36.8), (29.8, 36.9), (29.2, 36.8),
        (28.7, 36.6), (28.1, 36.5), (27.7, 36.7), (27.2, 37.0),
        (26.8, 37.3), (26.2, 37.6), (25.8, 38.0), (25.4, 38.5),
        (25.2, 39.0), (25.0, 39.5), (25.3, 40.0), (25.6, 40.4),
        (26.0, 40.7), (26.5, 41.0), (26.9, 41.4), (27.5, 41.7),
        (28.1, 42.0), (28.6, 42.4), (29.2, 42.7), (29.8, 43.0),
        (30.5, 41.9), (31.2, 41.9), (31.8, 41.9), (32.6, 41.8),
        (33.5, 41.7), (34.5, 41.5), (35.5, 41.4), (36.5, 41.1),
        (37.0, 40.8),
    ]),
    # Maghreb: Morocco, Algeria, Tunisia, and the coastal Libyan edge.
    polygon([
        (-9.0, 35.9), (-8.5, 35.7), (-7.4, 35.3), (-6.2, 35.0),
        (-5.1, 35.1), (-4.2, 35.2), (-3.4, 35.4), (-2.6, 35.3),
        (-1.7, 35.1), (-0.8, 35.0), (0.2, 35.0), (1.0, 35.1),
        (1.9, 35.3), (2.7, 36.0), (3.5, 36.7), (4.3, 37.0),
        (5.0, 37.0), (5.7, 36.7), (6.3, 36.4), (7.0, 36.1),
        (7.7, 36.0), (8.3, 36.2), (9.0, 36.7), (9.8, 37.1),
        (10.5, 37.2), (11.2, 37.0), (11.8, 36.5), (12.3, 36.0),
        (12.8, 35.5), (13.5, 35.1), (14.2, 34.8), (15.0, 34.5),
        (16.0, 34.2), (17.2, 33.9), (18.5, 33.7), (20.0, 33.5),
        (22.0, 33.2), (24.0, 32.9), (25.0, 32.4), (25.0, 31.0),
        (-9.0, 31.0),
    ]),
    # Levant and the northern Sinai/Delta land.
    polygon([
        (32.5, 36.6), (33.2, 36.4), (34.0, 36.6), (34.7, 36.8),
        (35.1, 36.5), (35.4, 36.0), (35.5, 35.4), (35.7, 34.8),
        (35.9, 34.3), (36.2, 33.7), (36.5, 33.2), (36.9, 32.7),
        (37.0, 32.0), (37.0, 31.0), (32.5, 31.0), (32.5, 32.0),
        (32.4, 32.7), (32.2, 33.4), (32.2, 34.0), (32.3, 34.6),
        (32.4, 35.2), (32.4, 35.8), (32.5, 36.6),
    ]),
]


# Small land pieces that matter at this resolution.  Keeping these separate
# prevents broad mainland polygons from turning nearby sea into land.
ISLANDS = [
    # Balearics
    polygon([(2.3, 39.8), (2.8, 40.0), (3.4, 40.1), (3.6, 39.7),
             (3.4, 39.3), (2.8, 39.2), (2.4, 39.4)]),
    polygon([(3.8, 39.0), (4.1, 39.2), (4.4, 39.0), (4.3, 38.7),
             (4.0, 38.7)]),
    # Corsica and Sardinia
    polygon([(8.55, 41.9), (8.8, 42.2), (9.1, 42.8), (9.4, 42.9),
             (9.55, 42.4), (9.4, 41.7), (9.3, 41.0), (9.0, 40.4),
             (8.7, 40.5), (8.55, 41.2)]),
    polygon([(8.1, 39.2), (8.4, 39.7), (8.6, 40.3), (9.0, 40.5),
             (9.4, 40.1), (9.5, 39.5), (9.3, 38.9), (9.0, 38.3),
             (8.5, 38.0), (8.1, 38.4)]),
    # Sicily and Malta
    polygon([(12.4, 38.2), (13.0, 38.4), (13.7, 38.2), (14.6, 37.9),
             (15.2, 37.5), (15.0, 37.1), (14.2, 36.7), (13.5, 36.7),
             (12.8, 37.0), (12.3, 37.5)]),
    polygon([(14.2, 36.1), (14.6, 36.1), (14.6, 35.8), (14.2, 35.8)]),
]
# Major Mediterranean and Black Sea islands.
ISLANDS += [
    # Crete, Rhodes, and Cyprus
    polygon([(23.4, 35.6), (24.2, 35.5), (25.1, 35.3), (26.2, 35.2),
             (26.5, 35.0), (25.8, 34.8), (24.8, 34.9), (23.8, 35.1),
             (23.3, 35.4)]),
    polygon([(27.6, 36.5), (28.2, 36.5), (28.3, 36.1), (28.0, 35.8),
             (27.6, 36.0)]),
    polygon([(32.2, 35.5), (33.0, 35.7), (34.0, 35.5), (34.6, 35.0),
             (34.3, 34.5), (33.7, 34.2), (32.9, 34.4), (32.2, 34.9)]),
    # Euboea, Lesbos, Chios, Samos and the Dodecanese.
    polygon([(23.4, 39.0), (23.8, 39.2), (24.0, 38.5), (23.9, 38.0),
             (23.5, 38.2), (23.3, 38.6)]),
    polygon([(25.8, 39.3), (26.4, 39.4), (26.8, 39.0), (26.4, 38.7),
             (25.8, 38.8)]),
    polygon([(26.0, 38.5), (26.4, 38.6), (26.5, 38.2), (26.1, 38.1)]),
    polygon([(26.5, 37.8), (27.0, 37.9), (27.2, 37.6), (26.8, 37.4)]),
    polygon([(27.0, 36.9), (27.4, 37.0), (27.5, 36.7), (27.1, 36.6)]),
    # Ionian islands and a few clearly resolved Aegean islands.
    polygon([(19.7, 39.8), (20.1, 39.9), (20.3, 39.5), (20.0, 39.2),
             (19.7, 39.4)]),
    polygon([(20.3, 38.4), (20.7, 38.7), (20.8, 38.2), (20.5, 37.9),
             (20.2, 38.1)]),
    polygon([(20.6, 37.9), (21.0, 38.0), (21.1, 37.7), (20.7, 37.6)]),
    polygon([(24.1, 37.8), (24.5, 38.0), (24.5, 37.6), (24.1, 37.5)]),
    # Balearic and Tyrrhenian small islands visible at 0.2 degrees.
    polygon([(1.4, 38.9), (1.7, 39.1), (1.8, 38.8), (1.5, 38.6)]),
    polygon([(10.0, 39.2), (10.4, 39.4), (10.5, 39.0), (10.2, 38.8)]),
    polygon([(11.8, 38.7), (12.2, 38.9), (12.4, 38.5), (12.0, 38.3)]),
    # Crimea, whose southern shore reaches the northern edge of this map.
    polygon([(32.4, 45.0), (33.5, 45.2), (35.0, 45.4), (36.4, 45.1),
             (36.5, 44.7), (36.0, 44.3), (35.5, 44.0), (34.7, 44.1),
             (34.0, 44.5), (33.2, 44.6), (32.4, 44.8)]),
]

# Cyprus is already represented above; the rule remains uniformly point based.
def is_land(lon, lat):
    return any(inside(lon, lat, shape) for shape in LAND + ISLANDS)
def main():
    matrix = np.zeros((ROWS, COLS), dtype=np.uint8)
    for r in range(ROWS):
        lat = round(NORTH - r * DY, 10)
        for c in range(COLS):
            lon = round(WEST + c * DX, 10)
            matrix[r, c] = 1 if is_land(lon, lat) else 0

    csv_path = OUT / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(matrix.tolist())

    raw = Image.fromarray(matrix * 255, mode="L")
    raw.save(OUT / "candidate.png")
    enlarged = raw.resize((COLS * 8, ROWS * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(OUT / "candidate_8x.png")

    assert matrix.shape == (71, 231)
    assert matrix.size == 16401
    assert set(np.unique(matrix)) <= {0, 1}
    assert raw.size == (231, 71)
    assert enlarged.size == (1848, 568)
    assert set(raw.getdata()) <= {0, 255}
    assert set(enlarged.getdata()) <= {0, 255}


if __name__ == "__main__":
    main()
