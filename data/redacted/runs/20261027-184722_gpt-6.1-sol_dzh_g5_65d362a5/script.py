import os, csv, math
import numpy as np
from PIL import Image

W, H = 231, 71
LONS = [-9.0 + 0.2*i for i in range(W)]
LATS = [45.0 - 0.2*i for i in range(H)]

def poly(points):
    return [(float(x), float(y)) for x, y in points]

def ellipse(cx, cy, rx, ry, n=20):
    return [(cx + rx*math.cos(2*math.pi*i/n),
             cy + ry*math.sin(2*math.pi*i/n)) for i in range(n)]

def inside(x, y, p):
    hit = False
    j = len(p)-1
    for i in range(len(p)):
        xi, yi = p[i]; xj, yj = p[j]
        cross = ((yi > y) != (yj > y))
        if cross and x < (xj-xi)*(y-yi)/(yj-yi) + xi:
            hit = not hit
        j = i
    return hit

# Broad mainland outlines.  These are deliberately hand-drawn geographical
# approximations, rather than data imported from a coastline file.
LAND = []
LAND.append(poly([
 (-9,45), (3.2,45), (3.3,43.7), (2.2,43.0), (1.0,42.6),
 (-0.5,42.5), (-1.8,43.0), (-3.0,43.4), (-4.4,43.5),
 (-5.5,42.7), (-6.7,42.3), (-8.8,42.0), (-9,41.0),
 (-9,37.2), (-8.2,37.0), (-7.0,37.0), (-6.5,36.3),
 (-5.7,36.0), (-5.1,36.2), (-4.2,36.5), (-3.0,36.7),
 (-1.8,36.6), (-0.5,37.2), (0.6,38.0), (1.4,39.0),
 (2.3,40.2), (3.3,42.0), (4.0,44.0), (4.5,45)
]))

LAND.append(poly([
 (-5,45), (9.2,45), (9.0,44.3), (8.2,43.8), (7.4,43.6),
 (6.8,43.1), (5.8,43.0), (4.8,43.1), (3.7,42.8),
 (2.8,42.5), (1.3,42.5), (0.1,42.8), (-1.2,43.2),
 (-2.8,43.5), (-4.5,43.8), (-5,45)
]))

# Italy, including the long peninsula and its northern connection.
LAND.append(poly([
 (6.5,45), (13.8,45), (13.7,44.0), (13.5,43.0),
 (13.4,42.0), (13.8,41.2), (14.7,40.5), (15.8,39.2),
 (16.5,38.2), (16.0,37.6), (15.0,37.5), (14.0,38.0),
 (13.0,38.7), (12.0,39.5), (11.3,40.5), (10.7,41.5),
 (10.0,42.3), (9.0,43.2), (8.0,44.0), (6.5,45)
]))

# Balkans and the northern Aegean mainland.
LAND.append(poly([
 (13.2,45), (29.0,45), (29.2,43.8), (28.8,42.5),
 (27.8,41.8), (26.5,41.0), (25.0,40.0), (24.0,39.0),
 (23.0,38.2), (22.0,38.0), (21.0,39.0), (20.0,40.0),
 (19.0,41.0), (18.0,42.0), (17.0,43.0), (16.0,44.0),
 (14.5,44.8), (13.2,45)
]))

# Anatolia, with the Black Sea coast on top and Mediterranean coast below.
LAND.append(poly([
 (26.0,42.2), (28.0,42.4), (30.0,42.1), (32.0,41.8),
 (34.0,41.5), (36.0,41.2), (37.0,41.0), (37.0,31.0),
 (36.0,31.0), (35.0,32.0), (34.0,33.0), (33.0,35.8),
 (31.5,36.0), (30.0,36.3), (28.5,37.0), (27.0,38.0),
 (25.5,39.0), (25.0,40.0), (26.0,42.2)
]))

# North Africa: the lower edge of the drawing is entirely continental.
LAND.append(poly([
 (-9,31), (37,31), (37,33.0), (35.5,33.2), (34.0,32.8),
 (32.5,32.0), (31.0,31.5), (29.0,31.3), (27.0,31.5),
 (25.0,32.0), (23.0,32.7), (21.0,32.8), (19.0,32.7),
 (17.0,32.8), (15.0,32.9), (13.0,33.0), (11.0,33.4),
 (9.0,33.8), (7.0,34.3), (5.0,35.0), (3.0,35.8),
 (1.0,36.4), (-1.0,36.5), (-3.0,35.8), (-5.0,35.5),
 (-7.0,35.0), (-9,34.5), (-9,31)
]))

# A small correction for the eastern Nile/Levant land around the frame edge.
LAND.append(poly([(32.0,31.0),(37.0,31.0),(37.0,34.0),(35.0,33.4),
                  (33.0,32.2),(32.0,31.0)]))
# Major islands and island groups, expressed as hand-written polygons/ellipses.
ISLANDS = []

# Balearics
ISLANDS += [
 poly([(2.3,39.9),(2.7,40.1),(3.4,40.0),(3.9,39.6),
       (3.8,39.2),(3.2,39.0),(2.7,39.2),(2.3,39.9)]),
 poly([(1.2,38.9),(1.6,39.0),(1.7,38.7),(1.4,38.5),
       (1.0,38.6),(1.2,38.9)]),
 poly([(4.1,40.1),(4.4,40.2),(4.4,39.9),(4.1,39.8)])
]

# Corsica and Sardinia
ISLANDS += [
 poly([(8.6,42.0),(9.1,42.2),(9.4,41.8),(9.3,41.0),
       (9.0,40.2),(8.8,41.0),(8.6,42.0)]),
 poly([(8.1,41.3),(8.5,41.0),(9.0,40.1),(9.2,39.3),
       (9.0,38.6),(8.5,38.9),(8.2,39.8),(8.1,41.3)])
]

# Sicily, Malta, and the toe-side minor islands
ISLANDS += [
 poly([(12.3,38.2),(13.0,38.4),(14.0,38.2),(15.0,37.8),
       (15.2,37.3),(14.5,36.8),(13.5,36.6),(12.6,36.8),
       (12.1,37.3),(12.3,38.2)]),
 poly([(14.2,36.1),(14.5,36.2),(14.5,35.8),(14.1,35.8),
       (14.2,36.1)]),
 poly([(15.7,38.4),(16.0,38.5),(16.0,38.2),(15.7,38.4)])
]

# Crete and the long Euboean island
ISLANDS += [
 poly([(23.4,35.6),(24.3,35.4),(25.5,35.2),(26.3,35.3),
       (26.5,35.0),(25.6,34.8),(24.4,35.0),(23.4,35.3)]),
 poly([(23.5,39.0),(24.0,39.3),(24.1,38.4),(23.8,37.8),
       (23.6,38.2),(23.5,39.0)])
]

# Cyprus
ISLANDS += [poly([(32.2,35.7),(33.0,35.6),(34.0,35.2),(34.6,35.0),
                   (34.8,34.5),(34.0,34.4),(33.0,34.6),(32.2,35.2),
                   (32.2,35.7)])]

# Larger Greek islands and compact groups; small polygons are intentionally
# retained because a 0.2-degree sample can hit them.
ISLANDS += [
 ellipse(26.0,38.8,.35,.18), ellipse(26.4,36.9,.25,.55),
 ellipse(27.1,36.2,.25,.18), ellipse(27.9,36.4,.18,.42),
 ellipse(28.1,35.9,.16,.22), ellipse(25.5,36.8,.17,.28),
 ellipse(24.0,37.8,.14,.22), ellipse(22.6,37.2,.18,.32),
 ellipse(21.7,37.1,.18,.15), ellipse(20.7,38.8,.20,.18),
 ellipse(20.0,39.4,.16,.22), ellipse(19.6,39.9,.12,.25),
 ellipse(19.9,40.6,.16,.32), ellipse(22.7,39.4,.12,.30),
 ellipse(24.7,39.0,.13,.22), ellipse(24.6,38.2,.13,.25),
 ellipse(25.9,37.6,.12,.22), ellipse(26.8,37.7,.10,.18),
 ellipse(27.3,37.6,.12,.18), ellipse(28.0,37.3,.10,.16),
 ellipse(28.1,36.7,.11,.17)
]

# Adriatic islands and a few Dalmatian fragments.
ISLANDS += [
 ellipse(14.7,43.7,.10,.38), ellipse(15.1,43.6,.10,.30),
 ellipse(15.5,43.5,.08,.26), ellipse(15.9,43.4,.08,.22),
 ellipse(16.3,43.3,.07,.18), ellipse(16.7,43.2,.06,.16),
 ellipse(17.0,42.9,.07,.18), ellipse(17.4,42.8,.06,.16),
 ellipse(18.0,42.6,.08,.18), ellipse(18.4,42.5,.06,.14)
]

# Tiny but easily sampled islands around the western Mediterranean.
ISLANDS += [
 ellipse(1.4,39.3,.08,.06), ellipse(2.0,39.5,.08,.05),
 ellipse(5.3,39.9,.08,.10), ellipse(7.0,39.2,.08,.08),
 ellipse(10.1,37.6,.07,.06), ellipse(11.2,36.7,.07,.06),
 ellipse(17.0,35.9,.08,.06), ellipse(18.0,35.5,.08,.07)
]
# Interior water bodies are subtracted after testing the surrounding land.
LAKES = [
    ellipse(10.0, 45.7, .45, .25),   # upper Italian lakes, mostly outside frame
    ellipse(7.3, 45.8, .35, .22),
    ellipse(26.0, 41.0, .20, .12),
    ellipse(14.0, 40.8, .13, .08),
]

def is_land(lon, lat):
    # Point-in-polygon testing is the sole rasterization operation.  No cell
    # averaging or coastline smoothing is performed.
    value = any(inside(lon, lat, p) for p in LAND)
    if not value:
        value = any(inside(lon, lat, p) for p in ISLANDS)
    if value and any(inside(lon, lat, p) for p in LAKES):
        value = False
    return 1 if value else 0

def make_matrix():
    return np.array([[is_land(x, y) for x in LONS] for y in LATS],
                    dtype=np.uint8)

def write_outputs(a):
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/candidate.csv", "w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(a.tolist())
    # 0 -> black water, 1 -> white land.
    raw = Image.fromarray((a * 255).astype(np.uint8), mode="L")
    raw.save("outputs/candidate.png")
    zoom = raw.resize((W * 8, H * 8), Image.Resampling.NEAREST)
    zoom.save("outputs/candidate_8x.png")
    return raw, zoom

if __name__ == "__main__":
    matrix = make_matrix()
    assert matrix.shape == (71, 231)
    assert matrix.size == 16401
    assert set(np.unique(matrix)).issubset({0, 1})
    write_outputs(matrix)
