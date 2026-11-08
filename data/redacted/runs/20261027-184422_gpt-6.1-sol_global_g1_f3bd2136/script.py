"""Knowledge-based coarse reconstruction of world land at a 2 degree grid.
No geographic files or network access are used; all shapes below are hand written.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
LATS = list(range(89, -90, -2))
LONS = list(range(-179, 180, 2))

def poly_contains(x, y, p):
    inside = False
    j = len(p) - 1
    for i, (xi, yi) in enumerate(p):
        xj, yj = p[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside

def ellipse(x, y, cx, cy, rx, ry):
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1

def any_poly(x, y, shapes):
    return any(poly_contains(x, y, p) for p in shapes)

# Outer coastlines, intentionally simplified but with enough bays and peninsulas
# to represent the geography visible at 2 degree sampling.
NORTH_AMERICA = [[
    (-168,72),(-153,71),(-142,69),(-136,62),(-130,57),(-126,50),
    (-124,43),(-117,32),(-110,31),(-105,29),(-98,26),(-92,28),
    (-88,25),(-83,25),(-80,29),(-80,35),(-75,40),(-70,44),
    (-64,46),(-59,52),(-63,58),(-75,61),(-80,68),(-95,70),
    (-111,73),(-130,75),(-151,74)
]]
CENTRAL_AMERICA = [[
    (-97,19),(-92,18),(-88,16),(-85,12),(-82,10),(-79,8),
    (-77,10),(-79,15),(-84,18),(-88,20),(-93,22),(-97,22)
]]
SOUTH_AMERICA = [[
    (-81,11),(-76,9),(-71,12),(-64,10),(-57,7),(-51,3),(-48,-2),
    (-50,-7),(-46,-12),(-44,-20),(-47,-29),(-53,-36),(-61,-40),
    (-68,-54),(-75,-51),(-73,-43),(-77,-34),(-79,-22),(-81,-9)
]]
AFRICA = [[
    (-17,15),(-16,25),(-10,32),(-5,36),(7,37),(17,35),(27,34),
    (35,30),(43,14),(50,7),(46,-2),(41,-10),(38,-18),(32,-27),
    (24,-34),(14,-35),(4,-33),(-4,-29),(-11,-19),(-16,-7),(-17,7)
]]
AUSTRALIA = [[
    (113,-14),(115,-20),(120,-23),(125,-24),(130,-27),(136,-26),
    (141,-19),(146,-18),(151,-24),(154,-28),(153,-35),(147,-39),
    (137,-39),(128,-36),(121,-35),(115,-31),(112,-25)
]]
EURASIA = [[
    (-10,36),(-5,43),(2,44),(8,43),(15,45),(20,40),(27,38),(32,36),
    (36,35),(40,37),(45,36),(50,32),(55,27),(60,26),(65,24),(68,22),
    (70,18),(73,8),(78,7),(83,15),(88,21),(94,21),(98,14),(102,8),
    (108,10),(111,20),(117,22),(122,18),(127,23),(130,31),(136,35),
    (141,39),(145,44),(151,46),(158,52),(165,55),(172,60),(179,65),
    (179,78),(165,77),(150,75),(135,73),(120,72),(105,73),(90,74),
    (75,74),(60,72),(45,70),(32,68),(20,69),(8,67),(-2,62),
    (-8,55),(-10,48)
]]
ARABIA = [[
    (34,31),(42,30),(50,29),(56,25),(59,20),(56,14),(51,12),
    (47,13),(43,17),(39,21),(35,25)
]]
ANTARCTICA = [[
    (-180,-60),(-165,-64),(-150,-67),(-135,-68),(-120,-70),(-105,-72),
    (-90,-70),(-75,-72),(-60,-74),(-45,-69),(-30,-70),(-15,-74),
    (0,-75),(15,-73),(30,-71),(45,-69),(60,-68),(75,-70),(90,-71),
    (105,-71),(120,-68),(135,-67),(150,-66),(165,-64),(180,-60),
    (180,-90),(-180,-90)
]]
GREENLAND = [[
    (-74,59),(-63,60),(-45,59),(-28,65),(-20,72),(-22,80),
    (-32,84),(-48,83),(-61,78),(-68,71)
]]

# Large islands are polygons; smaller islands are represented by ellipses below.
ISLAND_POLYS = [
    [(-6,50),(-4,58),(0,59),(2,54),(1,51),(-2,50)],       # Great Britain
    [(-11,52),(-10,55),(-7,56),(-6,53),(-8,51)],          # Ireland
    [(4,51),(7,54),(9,57),(14,56),(15,53),(11,50)],       # Denmark/low countries
    [(12,46),(14,42),(18,39),(23,38),(25,41),(22,45),(18,47)], # Italy
    [(8,44),(12,44),(15,43),(14,39),(10,39),(7,41)],      # Iberia
    [(24,40),(27,36),(29,35),(34,35),(37,39),(35,42),(29,43)], # Greece/Balkans
    [(35,35),(39,34),(42,36),(42,39),(38,40),(35,38)],    # Cyprus region
    [(139,35),(142,34),(145,37),(145,42),(142,45),(139,42)], # Japan main
    [(128,31),(131,32),(131,35),(128,36)],                # Kyushu/Shikoku
    [(121,25),(123,22),(122,20),(120,22)],                # Taiwan
    [(104,1),(108,4),(114,2),(117,-2),(114,-7),(108,-8),(103,-4)], # Borneo
    [(95,5),(101,4),(104,-1),(102,-6),(98,-5),(95,-1)],   # Sumatra
    [(105,-6),(114,-6),(116,-8),(113,-9),(106,-8)],       # Java
    [(130,-2),(138,-2),(143,-5),(145,-9),(142,-12),(135,-11),
     (130,-7)],                                           # New Guinea
    [(122,19),(126,18),(127,14),(124,10),(120,12)],       # Philippines
    [(79,10),(82,9),(82,6),(80,6),(79,8)],                # Sri Lanka
    [(43,-13),(47,-15),(50,-20),(49,-26),(45,-26),(43,-20)], # Madagascar
]
ISLAND_ELLIPSES = [
    (-19,65,4.5,2.3), (18,79,5,2.2), (55,73,3,6), (143,51,2,6),
    (-64,66,8,5), (-72,74,5,5), (-62,77,4,4), (-67,53,2.5,3),
    (-56,50,2.5,3), (-78,22,5,2), (-71,19,3,2), (-77,18,2,2),
    (-77,-12,1.5,1), (-62,-52,3,2), (-37,65,2,2),
    (172,-41,3,5), (177,-44,2,3), (147,-42,5,3),
    (141,-17,2.5,2), (151,-6,3,3), (122,2,2,2),
    (120,7,2,3), (118,15,2,2), (124,1,2,2),
    (100,15,1.5,2), (114,0,2,2), (134,35,2,2),
    (146,20,2,2), (138,19,2,2), (156,23,2,2),
    (-155,20,3,2), (-157,21,2,1.5), (-170,-14,2,1.5),
    (166,-14,2,2), (174,-18,2,2), (128,26,1.5,1.5),
    (35,31,1.5,1), (7,39,1.5,1), (12,38,1.5,1),
]

# Inland water is explicitly removed from land.  These are broad lake outlines,
# not a database lookup, and only matter where a 2 degree sample falls in them.
LAKES = [
    (-85,57,11,6), (-120,65,4,2.5), (-115,62,4,3),
    (-88,47,4,2), (-87,44,2,4), (-82,45,3,3), (-81,42,3,1.2),
    (-77,44,2.5,1.5), (51,41,5,2.5), (60,45,3,2),
    (76,46,5,2), (31,60,3,2), (33,-1,4,3), (30,-6,1.3,4),
    (34,-12,1.5,4), (14,14,4,2), (-69,-16,2,1.2),
]

BASE_SHAPES = NORTH_AMERICA + CENTRAL_AMERICA + SOUTH_AMERICA + AFRICA
BASE_SHAPES += AUSTRALIA + EURASIA + ARABIA + ANTARCTICA + GREENLAND

def is_land(lon, lat):
    land = any_poly(lon, lat, BASE_SHAPES)
    if not land:
        land = any_poly(lon, lat, ISLAND_POLYS)
    if not land:
        land = any(ellipse(lon, lat, *e) for e in ISLAND_ELLIPSES)
    if land and any(ellipse(lon, lat, *lake) for lake in LAKES):
        land = False
    return int(land)

def build_matrix():
    return [[is_land(lon, lat) for lon in LONS] for lat in LATS]

def write_outputs(matrix):
    with (OUT / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(matrix)
    raw = np.asarray(matrix, dtype=np.uint8) * 255
    Image.fromarray(raw, mode="L").save(OUT / "candidate.png")
    Image.fromarray(raw, mode="L").resize((1440, 720), Image.Resampling.NEAREST).save(
        OUT / "candidate_8x.png"
    )

if __name__ == "__main__":
    matrix = build_matrix()
    assert len(matrix) == 90 and all(len(row) == 180 for row in matrix)
    assert all(v in (0, 1) for row in matrix for v in row)
    write_outputs(matrix)
