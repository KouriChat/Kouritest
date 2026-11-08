"""Hand-reconstructed Mediterranean land/water raster.

All coordinates in this file are handwritten approximations.  No map data is
read: each sample is classified by point-in-polygon tests below.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

LATS = [45.0 - 0.2*i for i in range(71)]
LONS = [-9.0 + 0.2*i for i in range(231)]

def inside(x, y, poly):
    """Ray crossing test for one point and a closed polygon."""
    hit = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        crosses = ((yi > y) != (yj > y))
        if crosses:
            at_x = (xj-xi) * (y-yi) / (yj-yi) + xi
            if x < at_x:
                hit = not hit
        j = i
    return hit

def any_poly(x, y, polygons):
    return any(inside(x, y, p) for p in polygons)

def add_ellipse(target, cx, cy, rx, ry, n=48):
    """Add a deliberately simple island/reef ellipse as a polygon."""
    import math
    target.append([(cx + rx*math.cos(2*math.pi*k/n),
                    cy + ry*math.sin(2*math.pi*k/n)) for k in range(n)])

# Large land masses.  The boundary is followed counter-clockwise from the
# eastern edge of the frame along the Mediterranean coast.
land = []

# Europe, the Balkans, Anatolia, and the Levant north of its coast.
land.append([
    (37.0,31.0),(37.0,45.0),(-9.0,45.0),(-9.0,37.2),
    (-8.5,37.0),(-7.5,36.8),(-6.5,36.4),(-5.7,36.0),
    (-5.1,36.3),(-4.4,36.6),(-3.3,36.6),(-2.3,37.0),
    (-1.3,37.4),(-0.5,38.0),(0.2,39.0),(0.6,40.0),
    (1.4,40.9),(2.4,41.2),(3.1,41.3),(3.3,42.0),
    (4.2,43.0),(5.3,43.4),(6.0,43.2),(6.5,43.5),
    (7.4,43.7),(7.7,44.0),(8.5,44.1),(9.5,44.0),
    (10.1,43.7),(10.4,43.2),(10.0,42.5),(9.9,42.0),
    (10.1,41.5),(10.3,40.9),(10.5,40.4),(10.9,39.9),
    (11.5,39.5),(12.0,39.0),(12.8,38.5),(13.6,38.2),
    (14.6,38.0),(15.5,37.7),(16.0,38.1),(16.1,38.7),
    (16.1,39.2),(16.5,40.0),(17.2,40.5),(18.0,40.8),
    (18.5,40.1),(18.3,40.8),(17.5,41.1),(16.5,41.3),
    (15.7,41.7),(15.0,42.1),(14.4,42.7),(13.8,43.2),
    (13.6,43.7),(13.8,44.2),(14.8,44.5),(15.8,44.6),
    (16.8,44.4),(17.8,43.9),(18.5,43.6),(19.1,43.1),
    (19.4,42.5),(19.6,41.8),(20.0,41.2),(20.3,40.5),
    (20.5,39.8),(20.7,39.1),(21.2,38.4),(21.5,37.8),
    (22.0,36.8),(22.7,36.4),(23.1,36.6),(23.3,37.2),
    (23.7,37.6),(24.2,37.8),(24.7,38.1),(25.1,38.5),
    (25.4,39.1),(25.8,39.7),(26.3,40.0),(26.7,40.7),
    (27.0,39.7),(27.4,39.1),(27.9,38.6),(28.4,38.1),
    (29.0,37.7),(29.8,37.3),(30.5,36.9),(31.3,36.6),
    (32.2,36.2),(33.0,36.1),(34.0,36.2),(35.0,36.3),
    (36.0,36.6),(36.6,36.9),(37.0,36.8)
])

# North Africa, including the land south of the Mediterranean coast.
land.append([
    (-9.0,31.0),(-9.0,35.2),(-8.0,35.5),(-7.0,35.8),
    (-6.2,35.8),(-5.7,35.5),(-5.1,35.1),(-4.2,35.0),
    (-3.2,35.0),(-2.2,35.1),(-1.2,35.2),(-0.3,35.1),
    (0.5,35.3),(1.2,35.8),(2.0,36.3),(2.8,36.7),
    (3.7,36.8),(4.7,36.8),(5.6,37.0),(6.4,37.1),
    (7.2,37.1),(8.0,37.0),(8.8,37.0),(9.6,36.9),
    (10.4,36.8),(11.1,37.0),(11.6,37.0),(12.0,36.7),
    (12.4,36.1),(12.6,35.2),(12.8,34.4),(13.3,33.7),
    (14.2,33.2),(15.0,32.9),(16.0,32.8),(17.0,32.9),
    (18.0,32.5),(19.0,32.3),(20.0,32.2),(21.0,32.0),
    (22.0,31.5),(23.0,31.2),(24.0,31.0),(-9.0,31.0)
])

# The Levant and Sinai/Egypt coast, where the broad frame edge cuts the land.
land.append([(29.5,31.0),(30.0,31.2),(31.0,31.3),(32.0,31.3),
             (33.0,31.4),(34.0,31.5),(34.5,32.2),(34.8,33.0),
             (35.0,33.8),(35.4,34.3),(36.0,34.8),(36.5,35.5),
             (37.0,36.0),(37.0,31.0)])
# Principal islands and island groups.  Small polygons are intentional: at
# this resolution a narrow island only contributes when a sample point falls
# on its approximate centreline.
islands = []
islands += [
 [(8.55,41.9),(8.75,42.2),(9.25,42.35),(9.55,42.05),
  (9.45,41.5),(9.65,40.9),(9.55,40.35),(9.2,39.7),
  (8.75,39.15),(8.35,39.0),(8.05,39.35),(8.05,40.0),
  (8.25,40.65),(8.2,41.3)], # Sardinia
 [(8.55,42.95),(8.65,43.15),(9.0,43.35),(9.35,43.2),
  (9.55,42.75),(9.45,42.3),(9.2,42.15),(8.8,42.25)], # Corsica
 [(1.15,39.9),(1.45,40.1),(2.0,40.05),(2.65,39.7),
  (3.15,39.55),(3.35,39.85),(3.15,40.15),(2.6,40.25),
  (1.9,40.55),(1.35,40.45)], # Mallorca
 [(3.75,40.0),(4.0,40.1),(4.35,40.05),(4.3,39.7),(4.0,39.6)], # Menorca
 [(1.15,38.9),(1.5,39.1),(1.7,38.9),(1.55,38.6),(1.2,38.6)], # Ibiza
 [(12.4,37.9),(12.7,38.05),(13.1,37.8),(13.3,37.45),
  (13.0,37.1),(12.5,36.75),(12.1,36.85),(12.2,37.35)], # Sicily
 [(32.25,35.05),(32.8,35.45),(33.5,35.55),(34.3,35.25),
  (34.65,34.75),(34.4,34.4),(33.7,34.45),(33.0,34.55),
  (32.45,34.75)], # Cyprus
 [(23.45,35.55),(24.0,35.7),(24.7,35.55),(25.3,35.2),
  (26.0,35.0),(26.3,35.15),(26.0,35.45),(25.2,35.65),
  (24.4,35.85),(23.7,35.8)], # Crete
 [(14.1,36.0),(14.35,36.1),(14.6,36.05),(14.5,35.8),(14.2,35.8)], # Malta
]

# Islands of the western and central Greek seas.
islands += [
 [(19.35,39.6),(19.55,39.9),(19.85,39.95),(20.0,39.65),
  (19.8,39.35),(19.5,39.35)], # Corfu
 [(20.6,38.0),(20.9,38.4),(21.3,38.55),(21.55,38.3),
  (21.45,37.9),(21.0,37.75)], # Kefalonia/Zakynthos area
 [(22.65,38.4),(22.9,38.7),(23.25,38.8),(23.55,38.55),
  (23.5,38.25),(23.1,38.15)], # Euboea rough
 [(22.35,37.3),(22.65,37.55),(23.0,37.5),(23.15,37.2),
  (22.9,36.95),(22.55,37.0)], # Peloponnese island-like outline
 [(24.3,37.7),(24.6,37.95),(24.9,37.8),(25.0,37.45),
  (24.7,37.25),(24.4,37.4)], # Andros/Tinos
 [(25.0,37.8),(25.25,38.0),(25.5,37.9),(25.55,37.6),
  (25.3,37.45),(25.05,37.55)], # Syros
 [(25.25,36.8),(25.55,37.0),(25.85,36.85),(25.9,36.5),
  (25.6,36.3),(25.3,36.45)], # Naxos
 [(26.0,36.8),(26.25,37.0),(26.55,36.9),(26.65,36.6),
  (26.4,36.35),(26.1,36.45)], # Paros/Amorgos
 [(26.7,36.5),(27.0,36.75),(27.35,36.7),(27.45,36.4),
  (27.2,36.15),(26.85,36.2)], # Kos
 [(27.55,36.0),(27.8,36.3),(28.0,36.25),(28.1,35.9),
  (27.85,35.6),(27.6,35.7)], # Rhodes
 [(26.0,39.0),(26.2,39.35),(26.5,39.45),(26.75,39.2),
  (26.65,38.85),(26.35,38.7)], # Lesbos
 [(26.9,38.35),(27.1,38.65),(27.4,38.7),(27.55,38.4),
  (27.35,38.1),(27.05,38.1)], # Chios
 [(27.65,37.65),(27.85,37.9),(28.05,37.85),(28.15,37.5),
  (27.95,37.3),(27.7,37.4)], # Samos
]

# Northern Adriatic and Tyrrhenian minor islands visible at 0.2 degrees.
islands += [
 [(12.2,42.9),(12.4,43.15),(12.65,43.1),(12.75,42.8),
  (12.55,42.6),(12.3,42.7)], # Elba
 [(10.2,42.35),(10.4,42.55),(10.65,42.5),(10.7,42.25),
  (10.5,42.1),(10.3,42.15)], # Capraia/Giglio approximation
 [(15.9,39.7),(16.1,40.0),(16.3,39.95),(16.35,39.65),
  (16.1,39.5)], # small Calabrian island
 [(13.8,38.55),(14.0,38.75),(14.25,38.7),(14.3,38.45),
  (14.05,38.3)], # Aeolian group
 [(14.7,38.5),(14.9,38.7),(15.1,38.6),(15.1,38.35),
  (14.85,38.3)],
]

# A few exposed Greek/Aegean rocks and the Gulf of Gabes islands.
for args in [(23.0,36.4,.08,.10),(23.55,36.65,.07,.10),
             (24.1,35.95,.08,.09),(25.8,35.5,.07,.09),
             (26.55,35.55,.07,.09),(28.35,36.8,.08,.10),
             (10.1,34.7,.18,.12),(10.8,34.8,.10,.08)]:
    add_ellipse(islands, *args)
def make_matrix():
    """Return rows north-to-south, columns west-to-east, using point tests."""
    polygons = land + islands
    rows = []
    for lat in LATS:
        row = []
        for lon in LONS:
            row.append(1 if any_poly(lon, lat, polygons) else 0)
        rows.append(row)
    return rows

def main():
    rows = make_matrix()
    assert len(rows) == 71 and all(len(r) == 231 for r in rows)
    assert all(v in (0, 1) for r in rows for v in r)
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)

    # Pillow's one-bit image gives exactly two pixel values.  Coordinates are
    # already in required image order: row 0 is 45N and column 0 is 9W.
    a = np.asarray(rows, dtype=np.uint8)
    image = Image.fromarray(a * 255, mode="L")
    image.save(out / "candidate.png")
    enlarged = image.resize((231*8, 71*8), resample=Image.Resampling.NEAREST)
    enlarged.save(out / "candidate_8x.png")

    # Internal validation of files and dimensions, without reading any map.
    with (out / "candidate.csv").open(newline="", encoding="utf-8") as f:
        check = list(csv.reader(f))
    assert len(check) == 71 and all(len(r) == 231 for r in check)
    assert set(v for r in check for v in r) <= {"0", "1"}
    assert Image.open(out / "candidate.png").size == (231, 71)
    assert Image.open(out / "candidate_8x.png").size == (1848, 568)
    assert set(image.getdata()) <= {0, 255}
    assert set(enlarged.getdata()) <= {0, 255}
    print("validated: 71x231=16401 cells; CSV and PNG outputs written")

if __name__ == "__main__":
    main()
