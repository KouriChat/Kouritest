import os, csv
import numpy as np
from PIL import Image

# Hand-drawn geographic reconstruction.  Coordinates are longitude, latitude.
def inside(x, y, p):
    hit = False
    j = len(p) - 1
    for i in range(len(p)):
        xi, yi = p[i]; xj, yj = p[j]
        if ((yi > y) != (yj > y)):
            q = (xj-xi) * (y-yi) / (yj-yi) + xi
            if x < q:
                hit = not hit
        j = i
    return hit

def add(poly):
    land.append(poly)

land = []

# Iberia, France and the European mainland (the lower edge follows the
# Mediterranean shore in deliberately sparse, hand-written vertices).
add([(-9,45), (37,45), (37,41.7), (35,41.4), (33,41.2), (31,40.8),
     (29,40.4), (27,40.4), (25,40.8), (23,41.2), (21,41.7), (19,42.1),
     (17,42.5), (15,42.9), (13,43.3), (11,43.8), (10,44.1), (9,44.1),
     (8,44.0), (7,43.7), (6,43.2), (5,43.0), (4,42.6), (3,42.5),
     (2,42.6), (1,42.8), (0,42.8), (-1,43.0), (-3,43.4), (-5,43.3),
     (-7,43.5), (-9,43.2)])

# Italy, including the characteristic peninsula and the Po plain.
add([(7.0,45.0),(13.0,46.0),(14.0,45.5),(13.7,44.5),(13.5,43.5),
     (14.0,42.8),(14.5,42.0),(15.0,41.2),(15.5,40.5),(16.0,39.6),
     (16.0,38.8),(15.5,38.0),(15.0,37.5),(14.2,37.3),(13.5,38.0),
     (13.0,39.0),(12.5,40.0),(12.0,41.0),(11.5,42.0),(10.5,43.0),
     (9.5,44.0),(7.0,44.3)])

# The mainland Balkans and Greece, with an irregular Aegean-facing edge.
add([(13,46),(29,45),(30,43),(29,42),(28,41.5),(27,41.5),(26,41.0),
     (25,40.4),(24.5,39.6),(24,39.0),(23.3,38.7),(22.7,39.0),
     (22.2,39.7),(21.5,40.0),(20.7,39.7),(20.0,39.5),(19.5,40.0),
     (19.0,40.8),(18.0,41.3),(17.0,41.5),(16.2,42.0),(15.5,43.0),
     (14.5,44.0)])

# North Africa, from Morocco across the Maghreb to Egypt.
add([(-9,37), (0,37), (4,36.7), (7,37.0), (9,37.2), (11,37.1),
     (13,37.0), (15,36.7), (17,36.2), (18.5,35.5), (20,35.0),
     (22,34.2), (24,33.2), (26,32.0), (28,31.0), (37,31.0),
     (37,20), (-9,20)])

# Anatolia and the eastern Mediterranean mainland.
add([(26,42),(37,42),(37,36.5),(36,36.0),(35,36.2),(34,36.2),
     (33,36.0),(32,36.1),(31,36.3),(30,36.0),(29,36.2),(28,36.5),
     (27,36.6),(26,37.0),(25.5,38.0)])
add([(34.8,36.3),(37,36.5),(37,31),(34,31),(34.4,32),(34.6,33),
     (34.3,34.5)])
# Islands and small peninsulas.  These are intentionally polygonal because
# the requested raster itself has only 0.2 degree sampling.
add([(-1.9,36.0),(-1.0,36.8),(0.3,37.1),(1.5,37.0),(3.0,36.8),
     (3.6,36.4),(3.0,36.0),(1.8,35.7),(0.5,35.5),(-0.8,35.6)])
add([(8.1,41.3),(9.0,41.4),(9.8,40.8),(9.8,39.8),(9.3,39.0),
     (8.5,38.9),(8.0,39.6),(7.9,40.5)])
add([(12.0,38.2),(12.8,38.3),(13.4,37.8),(13.3,36.7),(12.6,36.5),
     (12.0,37.0),(11.8,37.7)])
add([(23.4,35.7),(24.4,35.7),(25.3,35.2),(26.2,35.0),(26.4,34.4),
     (25.7,34.0),(24.8,34.6),(23.8,34.8)])
add([(32.2,35.7),(33.2,35.7),(34.5,35.2),(34.5,34.3),(33.7,34.0),
     (32.7,34.5),(32.1,35.1)])
add([(14.2,39.2),(15.0,39.4),(15.5,39.0),(15.4,38.5),(14.7,38.3)])
add([(18.3,39.0),(19.0,39.2),(19.8,39.0),(20.1,38.5),(19.3,38.3),
     (18.5,38.4)])
add([(20.7,35.8),(21.5,36.0),(22.2,35.7),(22.7,35.1),(22.1,34.8),
     (21.2,35.0)])

# Larger islands and island groups in the western Mediterranean.
add([(4.8,40.1),(5.5,40.4),(6.4,40.3),(6.8,39.8),(6.5,39.3),
     (5.8,39.0),(5.0,39.2),(4.6,39.7)])
add([(2.3,39.0),(3.1,39.2),(3.5,39.0),(3.4,38.7),(2.6,38.7)])
add([(2.4,40.1),(3.0,40.2),(3.3,39.9),(3.0,39.6),(2.5,39.7)])

# Adriatic islands and the Ionian islands (small but visible at this scale).
for p in [
    [(14.0,44.9),(14.4,45.0),(14.5,44.6),(14.1,44.5)],
    [(14.7,44.5),(15.0,44.6),(15.0,44.1),(14.7,44.1)],
    [(15.4,43.8),(15.8,43.9),(15.7,43.5),(15.4,43.5)],
    [(16.0,43.4),(16.4,43.5),(16.3,43.1),(16.0,43.1)],
    [(18.0,40.8),(18.5,40.9),(18.6,40.4),(18.1,40.3)],
    [(19.5,39.9),(19.9,40.0),(20.0,39.5),(19.6,39.5)],
    [(20.4,39.0),(20.8,39.1),(20.8,38.7),(20.4,38.7)],
]:
    add(p)

# Malta and a few island fragments near the Aegean.
add([(14.15,36.1),(14.6,36.1),(14.6,35.8),(14.1,35.8)])
for p in [
    [(25.4,36.0),(26.0,36.1),(26.2,35.8),(25.7,35.6)],
    [(26.8,36.8),(27.3,37.0),(27.5,36.7),(27.0,36.5)],
    [(27.8,36.4),(28.3,36.5),(28.4,36.1),(27.9,36.0)],
    [(28.8,36.9),(29.2,37.0),(29.3,36.7),(28.8,36.6)],
    [(24.0,37.5),(24.5,37.7),(24.7,37.3),(24.2,37.2)],
]:
    add(p)

# Local water cuts prevent the coarse mainland polygons from filling major
# gulfs and straits.  A point in a cut is water even if in a land polygon.
water = [
    # Adriatic
    [(12.2,45.8),(14.0,45.7),(15.0,44.8),(16.0,43.7),(17.5,42.5),
     (18.5,41.6),(18.0,41.0),(16.5,41.5),(15.5,42.2),(14.5,43.4),
     (13.5,44.5)],
    # Aegean basin
    [(22,41.5),(24,41.5),(26,41.0),(28.5,41.2),(29.5,39.8),(29,37.0),
     (27.5,36.5),(26,37),(25,38),(24,39),(23,40),(22,40.5)],
    # Gulf of Lion / Gulf of Valencia nicks
    [(1.5,43.0),(4.8,43.0),(5.2,42.5),(4.2,41.9),(2.0,41.8)],
]
def is_land(x, y):
    # The explicit water cuts are evaluated last, like bays and sea channels.
    result = any(inside(x, y, p) for p in land)
    if result and any(inside(x, y, p) for p in water):
        result = False
    return int(result)

def main():
    # WGS84 sample centres as specified: north-to-south and west-to-east.
    lats = [45.0 - 0.2*i for i in range(71)]
    lons = [-9.0 + 0.2*j for j in range(231)]
    rows = [[is_land(x, y) for x in lons] for y in lats]
    assert len(rows) == 71 and all(len(r) == 231 for r in rows)
    assert all(v in (0, 1) for r in rows for v in r)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/candidate.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerows(rows)

    # Pillow mode '1' is a one-bit image: black water, white land.
    a = np.asarray(rows, dtype=np.uint8) * 255
    im = Image.fromarray(a, mode="L").convert("1")
    im.save("outputs/candidate.png")
    big = im.resize((231*8, 71*8), resample=Image.Resampling.NEAREST)
    big.save("outputs/candidate_8x.png")

    # Internal checks required by the experiment.
    assert a.shape == (71, 231)
    assert im.size == (231, 71) and big.size == (1848, 568)
    assert set(np.asarray(im).ravel()).issubset({False, True})
    assert np.array_equal(np.asarray(big)[::8, ::8], np.asarray(im))

if __name__ == "__main__":
    main()
