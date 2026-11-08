import csv
import os
from math import cos, radians
from PIL import Image

NL, NW = 90, 180
LATS = list(range(89, -90, -2))
LONS = list(range(-179, 180, 2))

def pip(x, y, p):
    inside = False
    j = len(p) - 1
    for i in range(len(p)):
        xi, yi = p[i]
        xj, yj = p[j]
        if ((yi > y) != (yj > y)):
            q = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < q:
                inside = not inside
        j = i
    return inside

def ellipse(x, y, cx, cy, rx, ry):
    return ((x-cx) / rx) ** 2 + ((y-cy) / ry) ** 2 <= 1

def box(x, y, x0, x1, y0, y1):
    return x0 <= x <= x1 and y0 <= y <= y1

def any_poly(x, y, shapes):
    return any(pip(x, y, p) for p in shapes)

# The polygons below are deliberately hand-written geographic approximations.
# Longitudes are east-positive and latitudes are north-positive.
N_AMERICA = [[
    (-168,72),(-160,70),(-151,70),(-143,70),(-136,70),(-132,69),
    (-128,67),(-124,64),(-124,58),(-126,53),(-124,49),(-123,46),
    (-117,43),(-112,37),(-108,32),(-104,29),(-98,26),(-92,26),
    (-88,28),(-85,30),(-82,25),(-80,25),(-80,29),(-77,32),
    (-75,35),(-75,39),(-72,42),(-68,44),(-65,47),(-61,50),
    (-57,52),(-58,56),(-62,58),(-61,62),(-66,65),(-72,67),
    (-80,68),(-88,70),(-96,72),(-105,73),(-116,73),(-128,73),
    (-140,73),(-151,74),(-160,74)
], [
    (-91,18),(-88,17),(-86,15),(-84,12),(-83,10),(-80,9),
    (-78,9),(-77,12),(-79,15),(-82,18),(-86,20),(-90,20)
]]

S_AMERICA = [[
    (-81,12),(-77,12),(-73,11),(-70,9),(-66,10),(-62,9),
    (-59,6),(-54,5),(-51,2),(-49,-2),(-45,-6),(-40,-10),
    (-37,-16),(-35,-22),(-38,-29),(-43,-34),(-48,-38),
    (-53,-42),(-58,-46),(-63,-51),(-68,-55),(-71,-54),
    (-73,-49),(-75,-43),(-74,-37),(-76,-30),(-78,-23),
    (-80,-16),(-80,-8),(-79,-1),(-81,5)
]]

AFRICA = [[
    (-17,36),(-8,37),(0,36),(9,37),(16,35),(24,33),(32,31),
    (36,27),(42,13),(49,12),(51,5),(48,-2),(43,-12),(40,-18),
    (35,-25),(31,-30),(25,-34),(18,-35),(12,-35),(8,-34),
    (3,-30),(-2,-25),(-8,-19),(-12,-10),(-15,-1),(-16,8),
    (-17,17),(-16,25)
]]

EUROPE = [[
    (-10,36),(-5,36),(0,38),(4,43),(8,44),(10,47),(14,46),
    (18,44),(22,41),(25,40),(27,42),(29,45),(28,49),(24,52),
    (25,55),(23,58),(19,60),(14,59),(11,57),(7,58),(5,55),
    (2,53),(-2,51),(-5,49),(-9,45),(-10,41)
]]

ASIA = [[
    (27,42),(32,42),(36,39),(40,38),(44,39),(48,38),(52,36),
    (56,34),(60,32),(64,30),(68,27),(72,25),(77,24),(80,27),
    (84,29),(88,27),(92,27),(96,25),(100,22),(104,20),(108,20),
    (111,23),(115,22),(119,25),(122,29),(126,32),(130,35),
    (135,38),(140,42),(145,45),(150,47),(156,50),(162,53),
    (170,55),(179,57),(179,72),(170,74),(160,73),(150,71),
    (140,72),(130,70),(122,70),(115,71),(108,70),(101,72),
    (94,72),(87,72),(80,70),(73,69),(67,70),(61,69),(55,68),
    (49,67),(43,65),(37,62),(33,58),(30,54),(28,50)
], [
    (68,30),(72,29),(76,30),(78,33),(75,36),(70,37),(66,35)
]]

AUSTRALIA = [[
    (113,-11),(119,-13),(126,-12),(132,-12),(137,-15),(142,-14),
    (147,-17),(151,-22),(153,-27),(153,-32),(150,-36),(145,-39),
    (138,-38),(132,-35),(126,-34),(121,-31),(116,-28),(114,-23)
]]
GREENLAND = [[
    (-73,60),(-61,59),(-46,60),(-37,64),(-30,70),(-22,76),
    (-20,82),(-28,85),(-40,84),(-52,83),(-62,80),(-68,75),
    (-73,69)
]]

MADAGASCAR = [[
    (43,-12),(49,-13),(51,-18),(50,-25),(47,-31),(44,-26),(43,-19)
]]

# Additional large islands and compact island groups, represented by ellipses
# or small polygons at the resolution of the requested sampling grid.
ISLAND_POLYS = [
    [(-25,64),(-21,63),(-17,64),(-14,66),(-16,67),(-22,67)],
    [(-11,51),(-7,51),(-6,54),(-9,58),(-13,57),(-14,54)],
    [(129,31),(132,32),(134,35),(136,38),(140,40),(143,43),
     (141,46),(137,44),(134,40),(131,37)],
    [(120,18),(123,14),(125,10),(122,7),(119,11)],
    [(95,6),(99,5),(101,2),(99,-2),(96,-4),(94,-1)],
    [(105,-6),(111,-7),(115,-5),(114,-2),(109,0),(105,-1)],
    [(-82,23),(-77,22),(-74,19),(-76,17),(-80,18)],
    [(-72,19),(-68,18),(-65,15),(-67,12),(-71,13)],
    [(-61,15),(-58,14),(-57,12),(-60,11),(-63,13)],
    [(-47,-1),(-44,-1),(-43,2),(-46,4),(-49,2)],
]

def land(x, y):
    # Main continental bodies.
    if any_poly(x, y, N_AMERICA + S_AMERICA + AFRICA + EUROPE + ASIA +
                AUSTRALIA + GREENLAND + MADAGASCAR):
        return True

    # Arctic islands: coarse but intentionally conservative approximations.
    if y >= 67 and (
        ellipse(x, y, -103, 76, 10, 7) or
        ellipse(x, y, -90, 74, 8, 6) or
        ellipse(x, y, -80, 74, 7, 5) or
        ellipse(x, y, -62, 75, 6, 8) or
        ellipse(x, y, -50, 76, 7, 8) or
        ellipse(x, y, 15, 78, 8, 7) or
        ellipse(x, y, 35, 76, 10, 6) or
        ellipse(x, y, 55, 77, 7, 6) or
        ellipse(x, y, 80, 77, 8, 5) or
        ellipse(x, y, 100, 76, 8, 5) or
        ellipse(x, y, 125, 74, 8, 5) or
        ellipse(x, y, 145, 73, 8, 5) or
        ellipse(x, y, 170, 72, 8, 5)
    ):
        return True

    # Major islands visible on a two-degree grid.
    if any(pip(x, y, p) for p in ISLAND_POLYS):
        return True
    if ellipse(x, y, 10, 56, 3, 5):       # Great Britain
        return True
    if ellipse(x, y, 138, -18, 2, 5):     # Cape York / Torres region
        return True
    if ellipse(x, y, 170, -42, 5, 8):     # New Zealand
        return True
    if ellipse(x, y, -151, 61, 5, 4):     # Aleutian / Alaska islands
        return True
    if ellipse(x, y, -155, 19, 2, 1):     # Hawaii
        return True
    if ellipse(x, y, 36, 35, 1, 1):       # Cyprus
        return True
    if ellipse(x, y, 14, 39, 2, 1):       # Sicily / nearby
        return True
    if ellipse(x, y, 18, 36, 1, 1):       # Sardinia/Corsica area
        return True
    if ellipse(x, y, 48, -13, 2, 2):      # Comoros / western Indian islands
        return True
    if ellipse(x, y, 73, 4, 3, 3):        # Sri Lanka
        return True
    if ellipse(x, y, 127, 26, 3, 2):      # Ryukyu / Taiwan area
        return True
    if ellipse(x, y, 142, 43, 4, 3):      # Hokkaido
        return True

    # Antarctica is ice-covered land; the irregular southern edge is a
    # hand-drawn approximation rather than a latitude cutoff.
    if y <= -70:
        edge = -67 + 5*cos(radians(x*1.7)) + 3*cos(radians(x*4.0))
        if y <= edge:
            return True
    return False

def write_outputs():
    os.makedirs("outputs", exist_ok=True)
    rows = [[1 if land(lon, lat) else 0 for lon in LONS] for lat in LATS]
    with open("outputs/candidate.csv", "w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)
    raw = Image.new("1", (NW, NL))
    raw.putdata([255 if v else 0 for row in rows for v in row])
    raw.save("outputs/candidate.png")
    raw.resize((NW*8, NL*8), Image.Resampling.NEAREST).save(
        "outputs/candidate_8x.png")

if __name__ == "__main__":
    write_outputs()
