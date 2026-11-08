"""Knowledge-based coarse world land mask, on a two-degree WGS84 grid."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

def poly(*points):
    return tuple(points)

def ellipse(cx, cy, rx, ry):
    return ("ellipse", cx, cy, rx, ry)

def inside_poly(x, y, p):
    hit = False
    j = len(p) - 1
    for i in range(len(p)):
        xi, yi = p[i]
        xj, yj = p[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi) + xi
        ):
            hit = not hit
        j = i
    return hit

def inside(x, y, shape):
    if shape[0] == "ellipse":
        _, cx, cy, rx, ry = shape
        return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1
    return inside_poly(x, y, shape)

LAND = []
WATER = []

# North America, with a deliberately simplified northern coastline.
LAND.append(poly(
    (-168,71),(-158,72),(-150,70),(-143,73),(-136,72),(-130,69),
    (-120,70),(-112,68),(-103,70),(-96,68),(-88,69),(-80,66),
    (-73,61),(-62,57),(-57,52),(-62,48),(-70,45),(-77,45),
    (-82,38),(-87,30),(-94,25),(-101,22),(-108,24),(-114,29),
    (-120,32),(-125,40),(-130,48),(-141,54),(-153,57),(-163,61)
))
# Alaska's western and northern projection.
LAND.append(poly((-170,63),(-168,70),(-160,72),(-150,70),(-145,64),
                 (-150,58),(-160,58),(-168,60)))
LAND.append(poly((-106,25),(-102,22),(-97,20),(-93,18),(-88,16),
                 (-85,13),(-89,11),(-94,15),(-99,18),(-104,20)))

# Greenland (land ice is explicitly land in this experiment).
LAND.append(ellipse(-42,73,20,14))
LAND.append(poly((-57,76),(-52,82),(-45,84),(-35,83),(-25,78),
                 (-24,69),(-31,62),(-42,60),(-52,65)))

# South America.
LAND.append(poly(
    (-81,12),(-75,12),(-70,10),(-65,8),(-60,6),(-54,4),(-50,1),
    (-48,-4),(-45,-10),(-43,-16),(-39,-21),(-37,-28),(-40,-35),
    (-45,-39),(-50,-45),(-57,-52),(-64,-55),(-69,-52),(-73,-46),
    (-76,-39),(-78,-32),(-80,-24),(-80,-16),(-78,-7)
))

# Europe and the western Mediterranean land mass.
LAND.append(poly(
    (-10,36),(-5,43),(0,44),(5,48),(10,51),(17,54),(25,55),
    (30,59),(35,65),(40,69),(30,70),(20,69),(12,67),(5,63),
    (-2,60),(-8,58),(-12,54),(-10,50),(-17,48),(-15,43)
))
LAND.append(poly(
    (-10,36),(-2,36),(5,38),(12,40),(18,39),(25,40),(30,43),
    (31,48),(27,51),(22,48),(17,46),(12,43),(7,43),(2,41)
))

# Scandinavia and the Baltic northern peninsula.
LAND.append(poly((5,58),(10,62),(12,68),(18,71),(25,71),(31,68),
                 (30,62),(25,58),(20,56),(14,56)))

# Continental Asia, excluding the major southern peninsulas.
LAND.append(poly(
    (27,40),(34,42),(40,48),(44,55),(49,62),(57,68),(66,72),
    (78,75),(92,76),(106,74),(120,72),(134,70),(147,67),(158,63),
    (170,58),(177,52),(170,48),(160,45),(151,43),(145,39),
    (138,35),(130,34),(123,31),(117,27),(110,23),(103,21),
    (97,24),(91,27),(85,29),(79,28),(73,30),(67,27),(60,25),
    (54,27),(48,30),(44,35),(37,37),(31,38)
))

# Arabian Peninsula and the Indian subcontinent.
LAND.append(poly((35,31),(42,30),(48,27),(55,25),(59,20),(57,15),
                 (52,12),(47,13),(43,17),(39,22)))
LAND.append(poly((67,25),(73,27),(78,31),(83,30),(88,27),(91,22),
                 (88,17),(84,12),(80,8),(75,9),(72,15),(69,21)))

# Africa, Madagascar, and the large islands of the Indian Ocean.
LAND.append(poly(
    (-17,15),(-10,20),(-3,23),(5,23),(12,21),(20,20),(27,17),
    (32,12),(37,7),(42,1),(43,-8),(40,-17),(36,-25),(31,-31),
    (24,-35),(17,-35),(10,-34),(4,-30),(-1,-24),(-6,-16),
    (-11,-7),(-15,2)
))
LAND.append(poly((43,-12),(48,-14),(50,-20),(49,-27),(46,-34),
                 (42,-26),(41,-19)))
LAND.append(poly((48,-13),(51,-12),(54,-16),(54,-23),(52,-26),
                 (49,-23),(47,-18)))
LAND.append(poly((-17,15),(-13,21),(-5,27),(5,30),(14,31),
                 (20,28),(25,25),(30,20),(28,16),(20,17),(12,20),
                 (5,23),(-4,23),(-10,20)))

# Australia, New Guinea, and Tasmania.
LAND.append(poly(
    (113,-11),(119,-14),(128,-13),(136,-15),(145,-18),(151,-24),
    (153,-31),(150,-38),(143,-40),(134,-38),(126,-39),(118,-36),
    (114,-31),(112,-24)
))
LAND.append(poly((141,-10),(146,-7),(153,-6),(157,-3),(154,0),
                 (149,-2),(145,-5)))
LAND.append(poly((144,-2),(149,-3),(154,-7),(153,-11),(148,-10),
                 (144,-7)))
LAND.append(poly((146,-41),(151,-40),(153,-43),(149,-47),(145,-44)))

# Antarctica is land ice.  The northern edge is a coarse approximation to
# the grounded coast; ocean ice is not included in this polygon.
LAND.append(poly(
    (-180,-73),(-165,-72),(-150,-70),(-137,-73),(-123,-71),
    (-110,-72),(-96,-74),(-83,-75),(-73,-71),(-67,-66),(-61,-64),
    (-55,-68),(-47,-73),(-35,-74),(-22,-72),(-10,-70),(0,-71),
    (14,-72),(27,-70),(40,-71),(53,-72),(67,-71),(80,-69),
    (95,-70),(110,-72),(125,-71),(140,-73),(157,-71),(173,-72),
    (180,-73),(180,-90),(-180,-90)
))
# Antarctic Peninsula reaches substantially farther north.
LAND.append(poly((-75,-65),(-68,-64),(-63,-61),(-58,-63),(-55,-67),
                 (-60,-70),(-70,-71)))

# Major Asian islands and archipelagos.
LAND.append(poly((128,42),(132,44),(138,43),(141,39),(140,35),
                 (136,34),(132,36),(129,39)))       # Japan main islands
LAND.append(poly((141,46),(146,47),(146,43),(143,41),(141,43))) # Hokkaido
LAND.append(poly((121,25),(124,26),(123,22),(120,21),(119,24))) # Taiwan
LAND.append(poly((95,6),(101,3),(104,-2),(103,-7),(98,-5),
                 (96,0)))                              # Sumatra
LAND.append(poly((105,-6),(114,-6),(115,-9),(111,-9),(106,-8))) # Java
LAND.append(poly((109,7),(117,7),(119,2),(116,-1),(111,1),(108,5))) # Borneo
LAND.append(poly((119,1),(124,4),(126,0),(124,-5),(120,-4),(118,-1))) # Sulawesi
LAND.append(poly((130,-2),(137,-2),(142,-5),(145,-9),(142,-10),
                 (136,-8),(131,-6)))                  # New Guinea west/east
LAND.append(poly((80,10),(82,9),(82,6),(80,5),(79,8)))  # Sri Lanka
LAND.append(poly((122,20),(126,20),(127,16),(124,14),(122,17))) # Luzon
LAND.append(poly((123,13),(127,12),(128,8),(125,7),(122,10)))  # Visayas/Mindanao

# European and North Atlantic islands.
LAND.append(ellipse(-18,65,3.5,2.5))   # Iceland
LAND.append(poly((-11,58),(-8,58),(-6,55),(-8,51),(-11,52),(-13,55)))
LAND.append(poly((-6,50),(-2,51),(1,49),(-1,45),(-5,43),(-8,44),(-9,47)))
LAND.append(poly((8,43),(13,43),(15,40),(13,37),(9,38),(7,40))) # Italy
LAND.append(ellipse(9,40,1.5,2.5))      # Sardinia
LAND.append(ellipse(12,38,1.5,2.0))     # Sicily
LAND.append(ellipse(9,42,1.5,1.2))      # Corsica
LAND.append(ellipse(25,35,3.5,1.3))     # Crete

# Islands around the Americas.
LAND.append(poly((-83,24),(-79,23),(-75,21),(-73,20),(-75,18),
                 (-80,20),(-83,22)))                  # Cuba
LAND.append(poly((-74,20),(-69,20),(-68,18),(-71,17),(-74,18))) # Hispaniola
LAND.append(poly((-67,18),(-65,18),(-62,16),(-61,14),(-64,15),
                 (-67,16)))                           # Puerto Rico / Antilles
LAND.append(poly((-88,18),(-86,18),(-84,16),(-83,14),(-86,14),
                 (-89,16)))                            # Central America
LAND.append(ellipse(-67,-54,2.5,1.4))                  # Falkland/Malvinas
LAND.append(ellipse(-90,-1,1.5,1.2))                   # Galapagos
LAND.append(ellipse(-71,-33,1.0,2.0))                  # Juan Fernandez area

# Arctic islands and the northeastern edge of the Russian land mass.
LAND.append(poly((-81,76),(-75,78),(-70,76),(-68,72),(-73,69),
                 (-80,70),(-83,73)))                  # Ellesmere
LAND.append(poly((-88,73),(-82,74),(-78,70),(-80,66),(-87,67),
                 (-91,70)))                            # Baffin
LAND.append(poly((-106,74),(-101,75),(-98,71),(-101,68),(-106,69)))
LAND.append(poly((-116,73),(-110,75),(-106,72),(-108,68),(-114,69)))
LAND.append(poly((64,72),(70,75),(76,74),(76,70),(70,68),(65,69)))
LAND.append(poly((80,76),(87,78),(93,76),(94,72),(88,70),(82,72)))
LAND.append(poly((51,75),(56,77),(59,74),(57,70),(53,71))) # Novaya Zemlya
LAND.append(poly((47,80),(54,81),(58,78),(55,76),(49,77))) # Franz Josef
LAND.append(poly((140,54),(146,55),(150,52),(149,48),(145,46),
                 (141,48)))                             # Sakhalin

# Southeast Asian island chains and Pacific islands visible on this grid.
LAND.append(ellipse(135,-17,2.0,1.0))                    # New Caledonia
LAND.append(ellipse(178,-18,1.2,1.0))                    # Fiji
LAND.append(ellipse(-155,19,1.8,1.2))                    # Hawaii
LAND.append(ellipse(147,-6,1.3,1.0))                     # Bismarck
LAND.append(ellipse(172,-43,2.0,4.0))                    # New Zealand
LAND.append(ellipse(174,-37,1.5,3.0))
LAND.append(ellipse(166,-46,1.1,1.2))

# A few additional islands whose size survives the two-degree sampling.
LAND.append(ellipse(73,4,1.5,1.0))                       # Maldives (coarse)
LAND.append(ellipse(110,1,1.4,1.0))                      # Bangka/Batam area
LAND.append(ellipse(8,40,0.9,1.0))
LAND.append(ellipse(35,36,1.2,0.8))                      # Cyprus
LAND.append(ellipse(32,36,1.4,1.0))                      # Cyprus west/east extent
LAND.append(ellipse(-61,13,1.0,1.2))                     # Lesser Antilles

# Inland water: these holes are applied after land.  They represent only
# sizeable lakes/bays that affect a two-degree sample, not every small lake.
WATER.append(ellipse(-82,58,17,10))       # Hudson Bay
WATER.append(ellipse(-90,25,12,7))        # Gulf of Mexico
WATER.append(poly((-117,34),(-114,35),(-112,30),(-113,24),(-115,27)))
WATER.append(ellipse(-85,44,6,2.0))       # Great Lakes (combined coarse hole)
WATER.append(ellipse(50,42,4.5,2.0))      # Caspian Sea
WATER.append(ellipse(47,31,2.0,1.0))      # Persian Gulf
WATER.append(ellipse(31,30,2.0,1.0))      # Lake Nasser / Nile reservoir
WATER.append(ellipse(34,0,2.0,4.0))       # Lake Victoria
WATER.append(ellipse(29,-12,1.7,3.0))     # Lake Tanganyika
WATER.append(ellipse(117,31,2.0,1.5))     # Yellow Sea
WATER.append(ellipse(120,13,2.0,2.0))    # South China Sea opening

def is_land(lon, lat):
    result = any(inside(lon, lat, s) for s in LAND)
    if result and any(inside(lon, lat, s) for s in WATER):
        result = False
    return int(result)

def main():
    # Coordinates are intentionally generated, rather than stored as a
    # point-by-point answer: 89..-89 and -179..179 at two-degree spacing.
    lats = list(range(89, -90, -2))
    lons = list(range(-179, 180, 2))
    matrix = np.array([[is_land(lon, lat) for lon in lons] for lat in lats],
                      dtype=np.uint8)
    assert matrix.shape == (90, 180)
    assert np.all((matrix == 0) | (matrix == 1))

    with (OUT / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # White is land, black is water.  No interpolation or post-processing.
    original = Image.fromarray(matrix * 255, mode="L")
    original.save(OUT / "candidate.png")
    enlarged = original.resize((1440, 720), resample=Image.Resampling.NEAREST)
    enlarged.save(OUT / "candidate_8x.png")

    # Lightweight self-checks also make accidental malformed output obvious.
    assert original.size == (180, 90)
    assert enlarged.size == (1440, 720)
    assert set(original.getdata()).issubset({0, 255})
    with (OUT / "candidate.csv").open(encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 90 and all(len(r) == 180 for r in rows)
    assert all(v in {"0", "1"} for r in rows for v in r)

if __name__ == "__main__":
    main()

