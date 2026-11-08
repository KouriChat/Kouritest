import csv
from pathlib import Path
import numpy as np
from PIL import Image

LAND = []
WATER = []

def add(dst, pts):
    dst.append([(float(x), float(y)) for x, y in pts])

# Iberia and the European mainland west of the Alps.
add(LAND, [(-9,45.5), (3.2,45.5), (3.4,44.5), (3.1,43.4),
    (2.4,42.5), (1.5,42.3), (0.4,41.4), (0.2,40.2),
    (-0.3,39.2), (-0.7,38.3), (-1.2,37.4), (-2.2,36.8),
    (-3.2,36.5), (-4.4,36.0), (-5.4,36.0), (-6.0,36.5),
    (-7.5,37.0), (-8.8,38.0), (-9,45.5)])

# France, northern Italy's western approach, and the Alps.
add(LAND, [(-1.5,46), (15.0,46), (15.0,45.0), (13.7,44.7),
    (12.5,44.2), (11.0,43.7), (9.7,43.5), (8.0,43.4),
    (7.0,43.7), (6.1,43.1), (5.3,43.2), (4.8,43.4),
    (3.5,43.5), (2.8,44.3), (2.0,45.0), (-1.5,45.0)])

# Italy, including the long peninsula but not Sicily.
add(LAND, [(6.5,46), (14.0,46), (13.8,44.8), (13.7,43.8),
    (14.2,42.7), (15.3,41.8), (16.5,41.1), (17.8,40.8),
    (18.5,40.5), (18.4,39.8), (17.5,39.6), (16.7,39.0),
    (16.2,38.2), (15.7,37.7), (15.0,38.0), (14.4,39.0),
    (13.5,40.0), (12.8,41.2), (12.0,42.2), (11.2,43.0),
    (10.0,43.7), (8.7,44.0), (7.5,44.0)])

# The Balkans and Greek mainland.  The southern edge follows the main coast.
add(LAND, [(12.5,46), (29.0,46), (29.0,41.4), (27.8,41.0),
    (26.2,40.7), (24.8,40.3), (24.0,39.7), (23.4,38.8),
    (22.8,38.0), (22.0,37.6), (21.2,37.7), (20.5,38.5),
    (20.0,39.5), (19.0,40.4), (18.0,41.2), (17.0,42.0),
    (16.0,42.8), (15.0,43.7), (14.0,44.5), (12.5,45.0)])

# Anatolia.
add(LAND, [(26.0,46), (37.0,46), (37.0,36.7), (36.2,36.5),
    (35.1,36.4), (34.2,36.1), (33.0,36.0), (31.7,36.1),
    (30.5,36.3), (29.5,36.3), (28.7,36.7), (28.0,37.1),
    (27.2,37.8), (26.4,38.4), (26.0,39.1), (26.6,40.0),
    (27.5,40.8), (28.4,41.5)])

# North Africa, with its Mediterranean shore as the northern edge.
add(LAND, [(-9,31), (37,31), (37,31.5), (34.5,31.2),
    (32.5,31.2), (30.0,31.3), (28.0,31.5), (26.0,31.1),
    (24.0,31.2), (22.0,31.7), (20.0,32.1), (18.0,32.2),
    (16.0,32.5), (14.8,33.5), (13.5,36.0), (12.2,37.0),
    (10.5,37.2), (9.0,37.1), (7.2,37.0), (5.5,36.8),
    (4.0,36.6), (2.5,36.2), (1.2,36.0), (0.0,35.5),
    (-1.5,35.2), (-3.0,35.2), (-4.5,35.5), (-5.5,35.9),
    (-6.5,35.7), (-7.8,35.1), (-9,35.0)])

# Levant coast and the land east of it.
add(LAND, [(34.5,31), (37,31), (37,37.5), (36.4,36.8),
    (36.0,35.8), (35.3,34.8), (34.8,33.8), (34.4,32.8)])

def inside(x, y, poly):
    hit = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)):
            cross = (xj-xi) * (y-yi) / (yj-yi) + xi
            if x < cross:
                hit = not hit
        j = i
    return hit
# Major western Mediterranean islands.
add(LAND, [(2.25,39.5), (2.8,39.2), (3.4,39.4), (3.5,40.0),
    (3.1,40.2), (2.4,40.1), (2.1,39.8)])       # Mallorca
add(LAND, [(3.75,39.9), (4.2,39.8), (4.35,40.0), (4.25,40.2),
    (3.8,40.2)])                                # Menorca
add(LAND, [(1.15,38.7), (1.45,38.6), (1.65,38.9), (1.45,39.1),
    (1.1,39.0)])                                # Ibiza
add(LAND, [(8.55,41.35), (9.15,41.4), (9.35,42.0), (9.25,42.7),
    (8.8,43.0), (8.55,42.5), (8.45,41.9)])      # Corsica
add(LAND, [(8.0,38.9), (8.7,38.8), (9.3,39.3), (9.65,40.0),
    (9.55,40.8), (9.0,41.25), (8.45,41.15), (8.05,40.4)]) # Sardinia

# Sicily, Malta, and the nearby Aeolian land.
add(LAND, [(12.4,37.0), (13.0,36.7), (14.0,36.6), (15.1,36.7),
    (15.7,37.3), (15.3,38.0), (14.5,38.2), (13.5,38.1),
    (12.7,37.8)])
add(LAND, [(14.20,35.78), (14.55,35.80), (14.65,36.05),
    (14.35,36.10), (14.15,35.95)])
add(LAND, [(15.0,38.4), (15.2,38.35), (15.25,38.55), (15.05,38.6)])

# Crete and the larger islands of the Aegean.
add(LAND, [(23.45,35.35), (24.2,35.0), (25.2,35.0), (26.3,35.25),
    (26.1,35.65), (25.0,35.75), (24.0,35.65)])
add(LAND, [(22.4,37.65), (22.9,37.55), (23.3,37.8), (23.0,38.05),
    (22.5,37.95)])                              # Euboea
add(LAND, [(26.0,38.9), (26.6,38.85), (26.9,39.1), (26.6,39.35),
    (26.1,39.3)])                               # Chios
add(LAND, [(25.8,39.9), (26.4,39.85), (26.7,40.1), (26.3,40.35),
    (25.8,40.2)])                               # Lesbos
add(LAND, [(26.7,37.65), (27.0,37.5), (27.25,37.7), (27.1,38.0),
    (26.75,38.0)])                              # Samos
add(LAND, [(27.55,35.8), (27.9,35.85), (28.15,36.15), (27.9,36.45),
    (27.55,36.25)])                             # Rhodes
add(LAND, [(23.35,38.0), (23.7,37.9), (23.85,38.15), (23.5,38.35)])
add(LAND, [(24.55,37.3), (24.9,37.2), (25.05,37.45), (24.7,37.55)])
add(LAND, [(25.3,36.65), (25.6,36.6), (25.75,36.85), (25.45,36.95)])

# Ionian and Adriatic islands that survive the 0.2 degree sampling.
add(LAND, [(19.65,39.7), (20.05,39.55), (20.2,39.85), (20.0,40.1),
    (19.65,40.0)])                              # Corfu
add(LAND, [(20.5,38.15), (20.75,38.0), (20.9,38.25), (20.65,38.45)])
add(LAND, [(20.65,37.65), (21.0,37.5), (21.15,37.8), (20.85,37.95)])
add(LAND, [(14.65,42.8), (15.0,42.6), (15.2,42.8), (14.95,43.05)])
add(LAND, [(15.7,43.0), (16.05,42.9), (16.1,43.25), (15.8,43.35)])

# Cyprus.
add(LAND, [(32.25,34.65), (32.8,34.5), (33.7,34.55), (34.55,35.0),
    (34.65,35.45), (34.0,35.7), (33.1,35.65), (32.4,35.35)])

# Small island groups near the Levant and southern Turkey.
add(LAND, [(34.0,35.9), (34.2,35.85), (34.25,36.05), (34.05,36.1)])
add(LAND, [(28.05,36.2), (28.25,36.15), (28.3,36.35), (28.1,36.4)])
def land_at(x, y):
    # Polygon rules are deliberately evaluated only at the requested points.
    return any(inside(x, y, p) for p in LAND) and not any(inside(x, y, p) for p in WATER)

def main():
    # Coordinates are generated arithmetically to avoid accumulated decimal drift.
    lats = [45.0 - 0.2*i for i in range(71)]
    lons = [-9.0 + 0.2*j for j in range(231)]
    matrix = [[1 if land_at(round(x, 10), round(y, 10)) else 0
               for x in lons] for y in lats]

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerows(matrix)

    # 0 is black water; 1 is white land.
    a = np.asarray(matrix, dtype=np.uint8) * 255
    Image.fromarray(a, mode="L").save(out / "candidate.png")
    Image.fromarray(a, mode="L").resize((1848, 568), Image.Resampling.NEAREST).save(
        out / "candidate_8x.png")

    assert len(matrix) == 71 and all(len(r) == 231 for r in matrix)
    assert sum(len(r) for r in matrix) == 16401
    assert set(v for r in matrix for v in r) <= {0, 1}

if __name__ == "__main__":
    main()
