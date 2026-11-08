"""Knowledge-based, deliberately approximate Mediterranean raster reconstruction."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

def inside(x, y, poly):
    """Ray crossing test for a hand-written longitude/latitude polygon."""
    hit = False
    n = len(poly)
    for k in range(n):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n]
        if (y1 > y) != (y2 > y):
            xx = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xx:
                hit = not hit
    return hit

def any_inside(x, y, polygons):
    return any(inside(x, y, p) for p in polygons)

# The large European polygon follows the Mediterranean-facing coast from
# Gibraltar eastward, then closes around the northern edge of the frame.
EUROPE = [[
    (-9,45), (-9,42), (-8.8,40), (-7,38), (-5.6,36),
    (-4,36.5), (-2,36.7), (0,38), (1.5,40), (2.5,42),
    (3,43), (4.5,43.4), (6,43.4), (7,43.3), (8,44),
    (9,44.1), (10,44.4), (11.5,44.6), (12.5,45),
    (13,45.5), (14,45.6), (15,45.5), (16,45), (17,44.5),
    (18,44), (19,43.5), (20,43), (21,42.5), (22,42),
    (23,41.5), (24,41), (25,41.5), (26,41.8), (27,41.8),
    (28,42), (29,42), (30,42.2), (31,42), (32,41.5),
    (33,41.5), (34,42), (35,42), (36,42.5), (37,43),
    (37,45), (-9,45)
]]

# Iberia is separately outlined so that its Atlantic side is not inferred
# from the coarse European mainland polygon alone.
IBERIA = [[
    (-9,43.8), (-8.8,42.5), (-8.7,41), (-9,39),
    (-8,38), (-7,37), (-5.6,36), (-3.5,36.5),
    (-1.5,36.8), (0,38), (1.5,40), (3.1,42),
    (2.8,43.1), (1,43.5), (-1,43.8), (-3,43.7),
    (-5,43.8), (-7,43.8)
]]

# Italy, including the long peninsula but excluding the surrounding seas.
ITALY = [[
    (6.8,45.7), (9,46), (12,46), (13.5,45.3), (13.5,44),
    (14,43), (15,42), (16.5,40), (16.4,38.8), (15.5,37.3),
    (14.8,37.5), (14.2,39), (13.2,40.5), (12.3,42),
    (12.4,43), (11.5,44), (10,44.5), (9,44), (8,43),
    (7,44)
]]

# Anatolia and the eastern Mediterranean land bridge.
TURKEY = [[
    (26,41), (28,41.5), (30,42), (32,42), (34,42),
    (36,41.5), (37,41), (37,36.5), (36,35.5),
    (34,36), (32,36), (30,36), (28,36), (26,37),
    (25.5,38), (26,39), (25.5,40)
]]

LEVANT = [[
    (35.8,36.8), (37,37), (37,31), (34.5,31),
    (34.2,32), (34.5,33), (35,34), (35.5,35),
    (34.8,36), (35.2,36.5)
]]

# North Africa is traced along the Mediterranean coast and then follows the
# southern frame boundary. The Gulf of Gabes and Gulf of Sirte are represented
# by the northward bends in this coastline.
AFRICA = [[
    (-9,31), (-9,35), (-8,35), (-6.5,35.5), (-5.6,35.9),
    (-4,35.2), (-2,35.1), (0,35.8), (1.5,36.8),
    (3,36.7), (5,36), (7,35.9), (8.5,36.3), (9.5,36.8),
    (10.5,37), (11.5,37.1), (11.4,35.5), (10.8,34.5),
    (11,33.5), (11.8,33), (13,32.7), (15,32.9),
    (16,33.2), (18,33.5), (20,33.2), (21,32.5),
    (23,32.3), (25,31.6), (27,31.3), (30,31.2),
    (33,31.2), (35,31.3), (37,31.4), (37,31), (-9,31)
]]

LAND = EUROPE + IBERIA + ITALY + TURKEY + LEVANT + AFRICA
# Major islands and island groups, all represented by hand-drawn polygons.
ISLANDS = [
    # Balearics
    [(2.2,39.9),(2.4,39.5),(3.2,39.3),(3.5,39.7),(3.4,40.1),(2.8,40.2)],
    [(3.7,39.2),(3.8,38.8),(4.3,38.7),(4.5,39.1),(4.2,39.5)],
    # Corsica and Sardinia
    [(8.55,43.0),(9.2,42.8),(9.5,42.2),(9.4,41.5),(9.0,41.1),
     (8.6,41.4),(8.5,42.2)],
    [(8.1,41.2),(8.9,40.9),(9.1,40.2),(9.0,39.4),(8.5,38.9),
     (8.0,39.2),(7.8,40.1)],
    # Sicily and Malta
    [(12.4,38.2),(13.4,38.3),(14.7,38.1),(15.6,37.5),(15.1,36.7),
     (13.8,36.6),(12.6,37.0),(12.0,37.6)],
    [(14.2,36.1),(14.6,36.0),(14.5,35.8),(14.1,35.8)],
    # Ionian islands and Crete
    [(19.7,39.7),(20.1,39.5),(20.3,38.9),(20.0,38.3),(19.7,38.5)],
    [(20.5,39.8),(20.9,39.8),(21.0,39.3),(20.7,38.8),(20.5,39.1)],
    [(23.4,35.6),(24.6,35.3),(26.0,35.2),(26.3,35.7),(25.4,36.0),
     (24.0,36.1)],
    # Euboea, Rhodes and Cyprus
    [(23.3,39.0),(24.0,39.2),(24.7,38.8),(24.5,38.2),(23.9,38.3)],
    [(27.6,36.4),(28.2,36.4),(28.3,35.8),(27.8,35.6),(27.5,36.0)],
    [(32.2,35.7),(33.4,35.2),(34.5,35.0),(34.7,35.5),(34.0,35.8),
     (32.7,36.0)],
]

# The numerous Aegean islands are small at this resolution but several are
# large enough to affect the requested 0.2-degree point grid.
AEGEAN = [
    [(25.6,40.0),(26.2,40.1),(26.4,39.7),(26.0,39.4),(25.6,39.6)],
    [(26.4,39.2),(27.0,39.1),(27.2,38.7),(26.8,38.5),(26.4,38.8)],
    [(24.0,37.8),(24.5,37.9),(24.7,37.5),(24.3,37.2),(23.9,37.4)],
    [(25.0,37.8),(25.5,37.8),(25.8,37.3),(25.5,36.9),(25.1,37.1)],
    [(26.8,37.8),(27.5,37.8),(27.8,37.4),(27.3,37.1),(26.9,37.3)],
    [(22.5,37.0),(23.0,37.1),(23.2,36.7),(22.8,36.4),(22.4,36.6)],
]

# A few Adriatic islands (the remainder are narrower than a sample spacing).
ADRIATIC = [
    [(14.7,44.0),(15.2,44.0),(15.4,43.5),(15.0,43.2),(14.7,43.5)],
    [(15.2,43.0),(15.8,43.1),(16.0,42.6),(15.5,42.5),(15.2,42.7)],
    [(16.7,43.0),(17.2,43.0),(17.5,42.5),(17.0,42.3),(16.7,42.6)],
]

LAND += ISLANDS + AEGEAN + ADRIATIC
# Water cut-outs prevent coarse mainland outlines from filling enclosed seas
# where two nearby peninsulas approach one another.
WATER = [
    # Black Sea, between the Balkans, Anatolia and the Caucasus.
    [(27.0,44.2),(29.5,44.8),(33,44.8),(36.5,44.2),(37,43.0),
     (36.5,42.2),(35,41.9),(33,41.8),(31,41.8),(29,41.7),
     (27.8,42.0),(27.2,42.8)],
    # Sea of Marmara and the two straits.
    [(26.0,41.2),(27.0,41.5),(28.2,41.3),(29.0,40.8),(28.0,40.4),
     (26.8,40.5),(26.0,40.8)],
    [(28.0,41.0),(29.0,41.2),(29.5,41.0),(29.3,40.5),(28.8,40.4)],
    # Lake Ohrid and Lake Prespa.
    [(20.55,41.2),(20.85,41.2),(21.1,40.9),(20.9,40.6),(20.5,40.7)],
    [(20.9,40.9),(21.3,41.0),(21.5,40.8),(21.2,40.5),(20.9,40.6)],
    # Lake Skadar.
    [(19.1,42.3),(19.5,42.4),(19.5,42.0),(19.2,41.9),(19.0,42.1)],
    # A small representative set of Alpine/Italian lakes visible at this scale.
    [(10.7,46.0),(11.0,46.0),(11.1,45.7),(10.8,45.5),(10.6,45.7)],
    [(10.0,45.9),(10.3,45.9),(10.4,45.6),(10.1,45.4),(9.9,45.6)],
]

def is_land(lon, lat):
    # Point classification is the union of the hand-written land shapes,
    # followed by explicit water holes. No raster-cell averaging is used.
    return any_inside(lon, lat, LAND) and not any_inside(lon, lat, WATER)

def build_matrix():
    lats = [45.0 - 0.2 * i for i in range(71)]
    lons = [-9.0 + 0.2 * j for j in range(231)]
    return np.array(
        [[1 if is_land(x, y) else 0 for x in lons] for y in lats],
        dtype=np.uint8,
    )

def write_outputs(a):
    if a.shape != (71, 231):
        raise AssertionError(a.shape)
    if not np.isin(a, (0, 1)).all():
        raise AssertionError("non-binary matrix")
    with (OUT / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(a.tolist())
    # Pillow's mode L image is deliberately only 0 or 255.
    raw = (a * 255).astype(np.uint8)
    image = Image.fromarray(raw, mode="L")
    image.save(OUT / "candidate.png")
    enlarged = image.resize((1848, 568), resample=Image.Resampling.NEAREST)
    enlarged.save(OUT / "candidate_8x.png")

if __name__ == "__main__":
    matrix = build_matrix()
    write_outputs(matrix)
    assert matrix.size == 16401
    assert set(np.unique(matrix)) <= {0, 1}
