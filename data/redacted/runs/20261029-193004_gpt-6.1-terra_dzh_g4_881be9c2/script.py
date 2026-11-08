import os, csv, math
import numpy as np
from PIL import Image

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

def poly(points):
    return [(float(x), float(y)) for x, y in points]

def inside(x, y, p):
    hit = False
    j = len(p) - 1
    for i in range(len(p)):
        xi, yi = p[i]; xj, yj = p[j]
        if ((yi > y) != (yj > y)) and (x < (xj-xi)*(y-yi)/(yj-yi)+xi):
            hit = not hit
        j = i
    return hit

def ellipse(cx, cy, rx, ry, n=24, phase=0.0):
    return [(cx+rx*math.cos(phase+2*math.pi*i/n),
             cy+ry*math.sin(phase+2*math.pi*i/n)) for i in range(n)]

# The large continental outlines are deliberately hand-written.  They are
# perimeter approximations, not data read from a map.
EUROPE = poly([
 (-9,45),(37,45),(37,41.3),(36,40.8),(35,40.2),(34,39.5),
 (33,39.0),(31.5,39.0),(30,39.4),(29,39.0),(28,38.8),(27,38.3),
 (26,37.8),(25,37.7),(24,37.4),(23.5,38.0),(23,39.0),(22,39.2),
 (21,39.4),(20,40.0),(19,40.6),(18,41.2),(17.5,42.0),(16.5,43.5),
 (15.5,44.5),(14.2,45),(13.5,44.7),(13.3,43.8),(13.0,42.8),
 (13.5,41.8),(14.5,40.8),(15.8,39.4),(17.2,38.0),(18.0,37.0),
 (17.0,37.2),(16.2,38.5),(15.2,40.0),(14.0,41.5),(12.5,43.0),
 (11.0,44.0),(10.0,44.6),(9.0,44.0),(8.0,43.4),(6.5,43.2),
 (5.0,43.1),(3.5,43.2),(2.0,43.0),(0.5,42.8),(-1,43.0),
 (-2.5,43.2),(-4,43.5),(-5.5,44.0),(-7,44.6),(-9,45)
])

AFRICA = poly([
 (-9,31),(37,31),(35,31.1),(34,31.3),(33,32.0),(32,32.5),
 (31,32.8),(30,32.8),(29,32.7),(28,32.5),(27,32.5),(26,32.8),
 (25,33.0),(24,33.0),(23,32.9),(22,32.8),(21,32.5),(20,32.3),
 (19,32.0),(18,32.5),(17,33.0),(16,33.6),(15,34.2),(14,35.0),
 (13,36.0),(12,36.5),(11,37.0),(10,37.2),(9,37.0),(8,37.0),
 (7,36.8),(6,36.5),(5,36.2),(4,36.0),(3,35.8),(2,35.5),
 (1,35.0),(0,35.0),(-1,35.5),(-2,35.7),(-3,35.8),(-4,35.5),
 (-5,35.0),(-6,34.0),(-7,32.5),(-8,31.7),(-9,31)
])

TURKEY = poly([
 (26.0,40.0),(26.7,41.0),(28.0,41.7),(29.5,42.0),(31,42.2),
 (33,42.0),(35,42.3),(37,41.7),(37,31),(35.5,31.0),(35,32.0),
 (35.5,33.0),(36.0,34.0),(36.2,35.0),(35.5,36.0),(34.0,36.2),
 (32.5,36.1),(31.0,36.0),(29.5,36.3),(28.0,36.7),(27.0,37.2),
 (26.0,38.0)
])

# Syria, Lebanon, Israel, and the small land continuation at the east edge.
LEVANT = poly([(34.5,36.2),(37,36.2),(37,31),(34,31),(34.1,32),
               (34.5,33),(34.4,34),(35,35),(34.5,36.2)])

LAND = [EUROPE, AFRICA, TURKEY, LEVANT]

# Large marine cut-outs prevent the coarse continental envelopes from filling
# the Black Sea, Marmara, and the principal Aegean water body.
WATER = [
 poly([(27,45),(37,45),(37,41.0),(35,41.3),(33,41.5),(31,41.7),
       (29,42.0),(27,42.0)]),
 poly([(26.0,41.0),(28,41.0),(29.5,40.5),(30.5,39.5),(29.5,38.5),
       (28,38.0),(27,37.5),(26,37.0),(25,37.0),(24,37.5),(23,38.5),
       (22,39.0),(21,39.5),(20,40.0),(20,41.0)]),
 poly([(8.5,45),(14.5,45),(14.0,44.0),(13.5,43.0),(13.0,42.0),
       (12.0,41.0),(11.0,40.0),(10.0,39.0),(9.0,38.0),(8.5,37.0),
       (7.5,37.0),(7.0,39.0),(7.5,41.0),(8.0,43.0)])
]
# Iberian, Italian, Balkan and Anatolian islands, plus the principal western
# Mediterranean islands.  Ellipses keep the construction compact while still
# representing islands at the requested sampling scale.
ISLANDS = [
 poly([(-9,36.0),(-8,36.0),(-7,36.5),(-6,36.7),(-5,36.5),(-4.5,37.0),
       (-3.5,37.0),(-2.5,36.8),(-1.5,37.0),(-.5,37.5),(.3,38.0),
       (1.0,39.0),(1.5,40.0),(2.0,41.0),(1.0,42.0),(-1,43),(-3,43),
       (-5,43),(-7,42),(-8,40),(-9,38)]),
 # Corsica and Sardinia
 poly([(8.55,43.0),(9.1,43.1),(9.5,42.5),(9.6,41.5),(9.3,41.0),
       (9.4,40.2),(9.0,39.3),(8.4,39.0),(8.2,40.0),(8.4,41.0),
       (8.2,42.0)]),
 poly([(8.2,41.2),(9.0,41.3),(9.6,40.8),(9.7,39.7),(9.5,38.8),
       (9.0,38.2),(8.3,38.8),(8.0,39.8)]),
 # Sicily, Malta, Balearics
 poly([(12.5,38.2),(13.5,38.4),(14.5,38.2),(15.3,37.7),(15.2,37.2),
       (14.0,36.7),(13.0,36.7),(12.2,37.2)]),
 ellipse(14.35,35.9,.16,.08,16),
 ellipse(3.0,39.6,.7,.35,20), ellipse(1.4,38.9,.28,.16,16),
 ellipse(2.7,39.0,.12,.08,12),
 # Adriatic and Ionian islands (a representative chain at 0.2 degree scale)
 ellipse(15.0,44.9,.18,.06,12), ellipse(15.3,44.7,.20,.06,12),
 ellipse(15.5,44.4,.16,.06,12), ellipse(15.7,44.0,.16,.06,12),
 ellipse(16.0,43.7,.16,.06,12), ellipse(16.2,43.3,.13,.06,12),
 ellipse(16.5,42.8,.14,.06,12), ellipse(17.0,42.4,.12,.05,12),
 ellipse(18.0,39.8,.12,.06,12), ellipse(19.0,39.5,.14,.06,12),
 # Greek mainland/island groups
 ellipse(20.7,39.1,.25,.12,16), ellipse(21.4,38.8,.20,.09,14),
 ellipse(22.4,38.5,.18,.10,14), ellipse(23.0,37.8,.20,.10,14),
 ellipse(24.0,37.7,.20,.12,14), ellipse(24.6,36.9,.35,.18,18),
 ellipse(25.3,35.3,.75,.45,24), ellipse(26.6,35.1,.15,.08,12),
 ellipse(27.0,36.8,.22,.12,14), ellipse(28.0,36.8,.25,.12,14),
 # Cyprus and Crete
 poly([(32.2,35.6),(33.2,35.4),(34.2,35.0),(34.7,34.6),(34,34.2),
       (33,34.3),(32,34.8),(32.0,35.3)]),
 poly([(23.4,35.6),(24.5,35.5),(25.5,35.1),(26.3,35.0),(26.0,34.7),
       (24.8,34.7),(23.8,35.0)])
]

def is_land(lon, lat):
    # Start with the broad outlines and subtract hand-defined marine basins.
    result = any(inside(lon, lat, p) for p in LAND)
    if any(inside(lon, lat, p) for p in WATER):
        result = False
    if any(inside(lon, lat, p) for p in ISLANDS):
        result = True
    return result
def make_matrix():
    # Coordinate construction is integer-based to avoid accumulated decimal
    # rounding changing a boundary column.
    rows = []
    for ri in range(71):
        lat = 45.0 - 0.2 * ri
        row = []
        for ci in range(231):
            lon = -9.0 + 0.2 * ci
            row.append(1 if is_land(lon, lat) else 0)
        rows.append(row)
    return rows

def write_all(rows):
    assert len(rows) == 71 and all(len(r) == 231 for r in rows)
    assert all(v in (0, 1) for r in rows for v in r)
    with open(os.path.join(OUT, "candidate.csv"), "w", newline="", encoding="ascii") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerows(rows)
    a = np.asarray(rows, dtype=np.uint8)
    # 0 is black water and 1 is white land.
    img = Image.fromarray(a * 255, mode="L")
    assert img.size == (231, 71)
    img.save(os.path.join(OUT, "candidate.png"))
    big = img.resize((1848, 568), resample=Image.Resampling.NEAREST)
    assert big.size == (1848, 568)
    big.save(os.path.join(OUT, "candidate_8x.png"))

if __name__ == "__main__":
    matrix = make_matrix()
    write_all(matrix)
    print("wrote", len(matrix), "x", len(matrix[0]), "=", sum(map(len, matrix)))
