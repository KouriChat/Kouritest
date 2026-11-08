"""Knowledge-based, deliberately approximate Mediterranean land mask."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

W, H = 231, 71
LONS = [-9.0 + 0.2*i for i in range(W)]
LATS = [45.0 - 0.2*i for i in range(H)]

def poly(points):
    return tuple((float(x), float(y)) for x, y in points)

def inside(x, y, p):
    # Standard ray crossing test; boundary points are treated as land.
    hit = False
    j = len(p) - 1
    for i, (xi, yi) in enumerate(p):
        xj, yj = p[j]
        if ((yi > y) != (yj > y)):
            cross = (xj-xi) * (y-yi) / (yj-yi) + xi
            if x < cross:
                hit = not hit
        if ((x-xi)*(yj-yi) - (y-yi)*(xj-xi) == 0 and
                min(xi, xj) <= x <= max(xi, xj) and
                min(yi, yj) <= y <= max(yi, yj)):
            return True
        j = i
    return hit

# Mainland polygons.  Their clipped northern/southern edges deliberately
# extend beyond the requested frame where the real coast is outside it.
LAND = [
    # Iberia
    poly([(-9,45),(3.6,45),(3.6,42.5),(3.0,42.2),(2.5,41.8),
          (2.1,41.1),(1.4,40.5),(0.5,40.0),(-0.2,39.4),
          (-0.8,38.5),(-1.5,37.5),(-2.5,36.8),(-4,36.5),
          (-5.5,36.0),(-7.1,36.0),(-8.2,36.5),(-9,37.2)]),
    # France and the northern Mediterranean rim
    poly([(2.8,45),(13.5,45),(13.5,46),(7.0,46),(5.8,44.2),
          (5.2,43.2),(4.5,43.1),(3.7,42.5),(3.2,42.4)]),
    # Italian peninsula
    poly([(6.2,45),(13.7,46),(14.2,45.4),(14.0,44.8),
          (14.2,43.8),(14.8,42.8),(15.5,41.9),(16.8,41.1),
          (18.4,40.7),(18.3,40.1),(17.4,39.7),(16.6,39.0),
          (16.0,38.3),(15.3,37.7),(14.8,38.6),(14.2,39.2),
          (13.2,39.7),(12.5,40.8),(11.6,41.8),(10.8,42.7),
          (10.0,43.6),(8.8,44.3),(7.5,44.5)]),
    # Balkans and the European side of the Aegean
    poly([(13.0,45),(37,45),(37,42.0),(34,42.0),(31,42.1),
          (28.0,41.9),(27.0,41.3),(26.2,40.8),(25.9,40.1),
          (25.4,39.3),(24.6,38.6),(23.5,38.2),(22.5,38.1),
          (21.7,38.5),(21.0,39.0),(20.4,39.8),(20.0,40.6),
          (19.3,41.6),(18.6,42.2),(17.7,43.0),(16.8,44.0),
          (15.7,45.0)]),
    # Anatolia
    poly([(26.0,42.0),(37,42.0),(37,36.0),(35.8,36.0),(34.7,36.2),
          (33.6,36.1),(32.2,36.0),(31.1,36.1),(30.0,36.2),
          (29.0,36.5),(28.1,37.0),(27.2,38.0),(26.2,38.7),
          (25.7,39.5),(26.1,40.2),(26.0,41.0)]),
    # Levant, Sinai, and the eastern edge of the frame
    poly([(34.4,36.2),(37,36.0),(37,31),(33.8,31),(33.8,32.0),
          (34.5,33.0),(35.0,34.0),(34.7,35.0),(34.0,35.5)]),
    # North Africa, following the Mediterranean coast east to west
    poly([(-9,31), (37,31), (37,31.3),(35.0,31.2),(33.0,31.2),
          (31.0,31.7),(29.0,32.1),(27.0,32.5),(25.0,32.7),
          (23.0,32.8),(21.0,32.8),(19.0,32.8),(17.0,33.0),
          (15.0,33.2),(13.2,34.0),(12.0,35.0),(11.0,36.3),
          (10.0,36.8),(9.0,37.0),(7.5,36.9),(6.0,36.8),
          (4.5,36.5),(3.0,36.2),(1.0,35.7),(-1.0,35.1),
          (-3.0,35.0),(-5.0,35.4),(-6.2,35.8),(-7.0,35.0),
          (-8.0,34.0),(-9,33.2)])
]

def add(p):
    LAND.append(poly(p))
# Major and minor islands.  Ellipses are used only where a sub-degree island
# is too small for a useful hand-drawn vertex list.
import math

def island(points):
    LAND.append(poly(points))

def ellipse(cx, cy, rx, ry, n=20, phase=0.0):
    island([(cx + rx*math.cos(phase + 2*math.pi*k/n),
             cy + ry*math.sin(phase + 2*math.pi*k/n)) for k in range(n)])

# Balearic Islands
ellipse(3.0, 39.6, .42, .28, 18, .1)       # Mallorca
ellipse(2.8, 39.9, .18, .12, 14)
ellipse(1.43, 38.98, .22, .16, 14)          # Ibiza
ellipse(1.42, 39.1, .10, .07, 12)

# Corsica, Sardinia, Sicily and the Tyrrhenian islands
island([(8.55,41.9),(8.75,42.5),(9.25,42.9),(9.55,42.7),
        (9.55,42.1),(9.35,41.4),(9.0,41.2),(8.65,41.5)])
island([(8.1,41.3),(8.5,41.0),(9.0,40.6),(9.2,39.8),
        (9.0,39.2),(8.5,38.9),(8.2,39.2),(8.0,40.0)])
island([(12.4,38.2),(13.0,38.3),(13.8,37.9),(14.8,37.5),
        (15.2,37.0),(14.7,36.7),(13.9,36.7),(13.0,37.0),
        (12.5,37.4)])
ellipse(14.25,38.4, .12, .08, 12)           # Stromboli vicinity
ellipse(12.7,38.5, .10, .06, 12)
ellipse(11.95,36.8, .18, .13, 14)           # Pantelleria
ellipse(10.0,42.75, .22, .12, 14)           # Elba
ellipse(9.0,42.75, .07, .05, 10)

# Ligurian, Tuscan and Adriatic islands
ellipse(7.7,43.0, .12, .06, 12)
ellipse(10.3,42.7, .12, .08, 12)
ellipse(10.1,42.55, .08, .05, 10)
ellipse(11.1,42.55, .08, .05, 10)
ellipse(13.3,44.0, .12, .05, 12)
ellipse(14.0,43.7, .10, .05, 12)
ellipse(15.1,44.8, .18, .06, 12)
ellipse(15.7,44.9, .12, .05, 12)

# Malta, Gozo and nearby Pelagian islands
ellipse(14.38,35.9, .16, .09, 16)
ellipse(14.25,36.05, .08, .05, 12)
ellipse(12.0,35.5, .14, .08, 12)            # Lampedusa
ellipse(12.33,35.85, .06, .04, 10)

# Crete and the southern Aegean
island([(23.5,35.6),(24.2,35.2),(25.2,35.0),(26.2,35.1),
        (26.3,35.5),(25.5,35.7),(24.5,35.8)])
ellipse(23.0,35.4, .13, .06, 12)
ellipse(25.8,35.4, .10, .05, 12)
ellipse(26.6,35.3, .11, .06, 12)
ellipse(22.6,36.4, .13, .10, 12)

# Cyprus
island([(32.2,35.6),(33.0,35.4),(34.1,35.0),(34.6,34.7),
        (34.4,34.3),(33.7,34.5),(32.7,34.6),(32.0,35.0)])

# Ionian islands and the Aegean archipelago
ellipse(19.9,39.7, .13, .30, 16)
ellipse(20.7,38.5, .11, .25, 14)
ellipse(20.6,37.8, .08, .18, 14)
ellipse(21.1,38.9, .07, .12, 12)
ellipse(22.0,39.2, .10, .20, 14)
ellipse(22.7,39.0, .10, .18, 14)
ellipse(23.7,39.1, .13, .24, 14)
ellipse(24.3,39.7, .10, .17, 12)
ellipse(25.0,39.9, .10, .16, 12)
ellipse(25.4,38.9, .14, .25, 14)
ellipse(25.8,38.2, .10, .15, 12)
ellipse(26.9,39.1, .10, .20, 12)
ellipse(27.1,37.8, .10, .20, 12)
ellipse(27.9,37.8, .11, .16, 12)
ellipse(28.1,39.4, .10, .18, 12)
ellipse(28.4,36.8, .09, .15, 12)
# Larger eastern Mediterranean islands omitted from the coarse mainland
# outlines, plus a few islands visible at the requested 0.2-degree sampling.
island([(26.0,39.2),(26.3,39.5),(26.8,39.4),(27.0,39.0),
        (26.7,38.7),(26.2,38.8)])             # Rhodes
island([(26.0,39.9),(26.3,40.0),(26.5,39.8),(26.3,39.6),
        (26.0,39.7)])                          # Kos
island([(25.6,38.5),(25.9,38.8),(26.0,39.1),(25.8,39.2),
        (25.5,38.9)])                          # Karpathos
island([(23.0,38.3),(23.3,38.5),(23.5,38.3),(23.4,38.0),
        (23.1,38.0)])                          # Euboea
island([(22.4,37.8),(22.7,38.0),(23.0,37.8),(22.9,37.5),
        (22.6,37.5)])                          # Peloponnese offshore
island([(26.0,38.7),(26.3,38.8),(26.4,38.5),(26.1,38.4)])
island([(23.4,39.1),(23.7,39.3),(23.9,39.1),(23.8,38.9)])
island([(24.1,40.0),(24.4,40.2),(24.6,40.0),(24.4,39.8)])
island([(25.9,40.2),(26.2,40.5),(26.5,40.3),(26.3,40.0)])
island([(27.0,40.0),(27.3,40.2),(27.5,40.0),(27.3,39.8)])
island([(22.7,40.0),(22.9,40.2),(23.1,40.0),(22.9,39.8)])
island([(26.8,37.0),(27.1,37.2),(27.3,37.0),(27.1,36.8)])

# North Aegean islands
island([(25.8,39.9),(26.3,40.0),(26.5,39.7),(26.2,39.5),
        (25.8,39.6)])                          # Lesbos
island([(26.0,38.4),(26.2,38.6),(26.4,38.4),(26.2,38.2)]) # Chios
island([(26.7,37.8),(27.0,38.0),(27.2,37.8),(27.0,37.6)]) # Samos

# Inland waters: these are holes in otherwise terrestrial polygons.
LAKES = [
    poly([(20.6,41.2),(20.9,41.2),(21.1,40.9),(20.9,40.7),
          (20.6,40.8)]),                       # Lake Ohrid
    poly([(20.8,41.0),(21.1,41.0),(21.2,40.8),(20.9,40.7)]),
]

def land_at(x, y):
    result = any(inside(x, y, p) for p in LAND)
    if result and any(inside(x, y, p) for p in LAKES):
        return False
    return result

def main():
    matrix = np.array([[1 if land_at(x, y) else 0 for x in LONS]
                       for y in LATS], dtype=np.uint8)
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())
    raw = Image.fromarray(np.where(matrix == 1, 255, 0).astype(np.uint8),
                          mode="L")
    raw.save(out / "candidate.png")
    raw.resize((W*8, H*8), Image.Resampling.NEAREST).save(out / "candidate_8x.png")
    assert matrix.shape == (71, 231)
    assert set(np.unique(matrix)) <= {0, 1}

if __name__ == "__main__":
    main()
