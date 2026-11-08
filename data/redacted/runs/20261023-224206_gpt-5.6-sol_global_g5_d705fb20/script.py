from pathlib import Path
import csv
import numpy as np
from PIL import Image

# All geometry below is a hand-written, deliberately approximate recollection of
# geography.  Coordinates are (longitude, latitude); no external data is read.
lons = np.arange(-179, 180, 2, dtype=float)
lats = np.arange(89, -90, -2, dtype=float)
X, Y = np.meshgrid(lons, lats)
land = np.zeros(X.shape, dtype=bool)

def poly(points):
    """Rasterise one simple polygon by the even/odd rule."""
    p = np.asarray(points, dtype=float)
    inside = np.zeros(X.shape, dtype=bool)
    xj, yj = p[-1]
    for xi, yi in p:
        crossing = ((yi > Y) != (yj > Y))
        xcut = (xj-xi) * (Y-yi) / ((yj-yi) if yj != yi else 1e-20) + xi
        inside ^= crossing & (X < xcut)
        xj, yj = xi, yi
    return inside

def add(points):
    global land
    land |= poly(points)

def cut(points):
    global land
    land &= ~poly(points)

def ellipse(cx, cy, rx, ry, angle=0):
    a = np.deg2rad(angle)
    xx = (X-cx)*np.cos(a) + (Y-cy)*np.sin(a)
    yy = -(X-cx)*np.sin(a) + (Y-cy)*np.cos(a)
    return (xx/rx)**2 + (yy/ry)**2 <= 1

def add_ellipse(cx, cy, rx, ry, angle=0):
    global land
    land |= ellipse(cx, cy, rx, ry, angle)

# North American mainland, including Mexico and the Central American isthmus.
add([
 (-168,66),(-166,62),(-158,59),(-153,57),(-147,59),(-141,59),
 (-136,57),(-134,54),(-130,53),(-128,50),(-125,48),(-124,43),
 (-122,38),(-118,34),(-115,32),(-112,29),(-110,24),(-107,23),
 (-105,20),(-101,18),(-97,16),(-93,16),(-91,15),(-88,15),
 (-86,12),(-83,10),(-80,8),(-78,9),(-81,10),(-84,15),
 (-87,19),(-87,21),(-90,22),(-92,19),(-96,20),(-98,23),
 (-97,27),(-95,29),(-90,29),(-88,30),(-85,30),(-83,28),
 (-81,25),(-80,26),(-81,30),(-80,32),(-77,35),(-75,39),
 (-72,41),(-70,43),(-67,44),(-65,47),(-61,47),(-59,50),
 (-56,51),(-57,54),(-60,56),(-61,60),(-66,61),(-70,63),
 (-76,63),(-80,65),(-87,66),(-96,68),(-106,68),(-115,69),
 (-124,70),(-133,69),(-141,69),(-148,71),(-156,71),(-164,69)
])
# Alaska's Seward peninsula and broad southwest lobe.
add([(-169,66),(-166,64),(-164,61),(-158,60),(-153,59),(-150,61),
     (-153,64),(-160,66)])
# Hudson Bay and its long southern appendage are ocean, not an inland lake.
cut([(-95,64),(-86,64),(-78,61),(-79,57),(-82,53),(-81,51),
     (-84,51),(-88,54),(-94,57)])

# Greenland.
add([(-73,60),(-65,60),(-52,61),(-45,60),(-42,64),(-38,69),
     (-22,76),(-20,81),(-28,84),(-42,84),(-55,82),(-63,78),
     (-70,75),(-73,70),(-67,66)])

# Major Canadian Arctic islands.
add([(-125,72),(-119,69),(-111,70),(-109,74),(-115,76),(-123,75)])
add([(-110,76),(-101,75),(-96,78),(-101,81),(-109,80)])
add([(-100,68),(-91,69),(-88,73),(-96,75),(-105,74)])
add([(-93,76),(-84,76),(-80,80),(-87,83),(-96,82)])
add([(-80,63),(-72,64),(-64,68),(-62,73),(-69,76),(-78,73)])
add([(-80,75),(-72,75),(-69,79),(-76,82),(-84,80)])
# South America.
add([
 (-80,10),(-76,11),(-72,11),(-68,10),(-63,10),(-59,8),
 (-56,6),(-52,4),(-50,1),(-48,-1),(-44,-2),(-41,-3),
 (-37,-5),(-35,-8),(-36,-12),(-39,-17),(-40,-22),(-44,-24),
 (-48,-28),(-50,-31),(-53,-33),(-57,-36),(-62,-39),(-65,-43),
 (-67,-47),(-69,-51),(-72,-54),(-74,-51),(-73,-46),(-75,-42),
 (-73,-38),(-72,-34),(-71,-30),(-71,-25),(-70,-20),(-72,-17),
 (-75,-14),(-77,-10),(-80,-7),(-81,-3),(-80,1),(-77,5)
])

# Eurasian mainland.  Seas lying inside this enclosing outline are cut below.
add([
 (-10,36),(-9,40),(-9,43),(-5,44),(-2,47),(-5,48),(-2,50),
 (2,51),(5,53),(8,55),(9,58),(6,59),(5,62),(9,65),
 (14,68),(20,70),(27,71),(32,70),(38,68),(45,68),(53,70),
 (61,70),(69,72),(82,73),(98,76),(113,75),(127,73),(140,72),
 (153,72),(164,70),(177,68),(179,66),(174,64),(169,61),
 (163,60),(160,56),(157,52),(153,49),(146,46),(141,47),
 (137,50),(135,48),(131,44),(129,40),(126,39),(124,40),
 (121,39),(120,35),(122,31),(120,27),(116,24),(111,21),
 (108,18),(107,14),(104,10),(104,5),(102,1),(100,5),
 (98,8),(98,13),(95,16),(94,20),(91,22),(89,22),(87,21),
 (85,19),(82,15),(80,10),(77,8),(75,10),(73,16),(70,20),
 (67,23),(63,25),(59,25),(57,27),(54,26),(51,25),(49,27),
 (48,30),(45,31),(42,35),(39,37),(36,36),(35,34),(33,35),
 (31,36),(29,36),(27,38),(25,39),(22,40),(20,42),(17,43),
 (14,45),(10,44),(7,44),(4,43),(1,42),(-2,43),(-5,41)
])

# Arabia protrudes south of the general Asian coastline.
add([(34,30),(39,31),(43,29),(48,29),(52,26),(56,25),(58,22),
     (55,17),(51,15),(46,13),(43,13),(42,17),(39,20),(37,24)])

# Peninsular Europe: Italy, Denmark and Peloponnese.
add([(7,45),(10,46),(13,45),(14,42),(17,40),(16,38),(13,38),
     (12,41),(10,43)])
add([(8,54),(9,57),(11,58),(12,56),(11,54)])
add([(20,42),(24,42),(26,40),(24,37),(21,36),(19,39)])

# Black Sea and Baltic are connected marine waters.
cut([(27,41),(29,46),(34,47),(39,45),(42,42),(40,40),(34,40)])
cut([(10,54),(13,55),(16,56),(18,59),(22,60),(25,59),(24,55),
     (20,54),(16,54)])

# The British Isles, Ireland and Iceland.
add([(-6,50),(-3,50),(1,52),(0,56),(-3,59),(-6,58),(-7,55)])
add([(-10,51),(-6,51),(-5,54),(-7,56),(-10,55)])
add([(-24,63),(-14,63),(-13,66),(-18,67),(-23,66)])

# Mediterranean islands large enough to affect this grid.
add_ellipse(9, 40, 1.6, 2.5, -10)       # Sardinia
add_ellipse(15, 37.5, 2.5, 1.3, 5)      # Sicily
add_ellipse(24.8, 35.2, 3.0, .75, 5)    # Crete
add_ellipse(33, 35, 1.8, .8, 10)        # Cyprus
add_ellipse(9, 42, 1.0, 1.8, -10)       # Corsica

# Northern Eurasian islands.
add([(-9,71),(0,70),(9,72),(18,77),(16,80),(7,81),(-3,78)])
add([(43,69),(52,70),(58,74),(56,77),(49,77),(44,74)]) # Novaya Zemlya
add([(75,73),(88,74),(95,79),(83,81),(72,78)])          # N. Zemlya
add([(95,77),(106,77),(113,80),(108,82),(98,81)])
add([(136,74),(146,74),(150,77),(143,79),(136,77)])
add_ellipse(18, 78.5, 8, 2.7, 5)                       # Svalbard
# Africa, kept separate from Asia at Suez.
add([
 (-17,37),(-10,36),(-6,35),(-2,36),(4,37),(10,37),(12,34),
 (15,32),(20,32),(25,32),(30,31),(33,30),(35,28),(35,23),
 (37,18),(40,15),(43,12),(51,11),(50,7),(46,3),(43,-1),
 (41,-5),(40,-10),(37,-15),(35,-20),(33,-26),(30,-31),
 (27,-34),(22,-35),(18,-34),(15,-30),(12,-25),(9,-20),
 (7,-15),(4,-9),(1,-5),(-5,-1),(-9,4),(-14,5),(-17,10),
 (-17,15),(-16,20),(-14,25),(-13,29),(-10,32),(-6,35)
])

# Australia.
add([
 (113,-22),(114,-17),(121,-14),(126,-14),(129,-12),(136,-12),
 (138,-16),(141,-17),(143,-12),(148,-18),(153,-24),(153,-29),
 (150,-35),(146,-39),(140,-38),(136,-35),(131,-34),(125,-34),
 (117,-35),(114,-32),(113,-27)
])
# Cape York is narrow and better represented explicitly.
add([(137,-18),(139,-12),(142,-10),(145,-14),(145,-18)])

# Madagascar and nearby substantial islands.
add([(49,-12),(51,-16),(50,-22),(47,-26),(44,-25),(44,-20),(46,-15)])
add_ellipse(55.5, -4.5, 1.0, 2.3, -15)
add_ellipse(43.5, -12.3, .9, 1.2)

# Japan: four approximate elongated islands.
add([(130,31),(132,31),(134,33),(132,34),(130,33)])          # Kyushu
add([(133,33),(136,33),(135,35),(133,35)])                   # Shikoku
add([(135,34),(139,35),(141,39),(140,41),(137,37)])          # Honshu
add([(140,41),(144,43),(145,45),(142,46),(140,44)])          # Hokkaido

# Sri Lanka, Taiwan and Hainan.
add_ellipse(80.7, 7.5, 1.5, 2.8, -15)
add_ellipse(121, 23.6, 1.2, 2.4, -15)
add_ellipse(109.5, 19.2, 1.8, 1.5)

# Sumatra, Java, Borneo, Sulawesi, Timor, and New Guinea.
add([(94,6),(98,5),(102,1),(106,-5),(104,-7),(100,-5),(96,-1)])
add([(105,-6),(113,-6),(115,-8),(111,-9),(106,-8)])
add([(108,7),(115,7),(119,3),(117,-4),(111,-4),(108,0)])
add([(119,2),(122,3),(122,-2),(125,-4),(123,-6),(120,-4)])
add([(124,-8),(129,-8),(130,-10),(125,-10)])
add([(130,-3),(136,-2),(141,-4),(147,-6),(151,-9),(147,-10),
     (141,-8),(136,-7),(131,-8)])

# The principal Philippine islands (grouped at this sampling resolution).
add([(119,19),(122,19),(124,16),(123,13),(120,13)])
add([(122,13),(126,12),(126,8),(124,6),(122,9)])
add_ellipse(120, 10, 1.5, 2.2, 15)

# New Zealand.
add([(166,-34),(170,-37),(174,-40),(173,-42),(170,-41),(168,-38)])
add([(166,-40),(169,-43),(167,-47),(165,-46),(164,-43)])

# Korea and Sakhalin are sharpened/additions to the mainland.
add([(125,39),(127,42),(130,43),(130,39),(128,35),(126,34)])
add([(142,46),(144,49),(143,54),(141,53),(140,49)])
# Newfoundland is separated from the mainland by the Gulf of St Lawrence.
cut([(-69,47),(-64,46),(-58,47),(-57,50),(-61,51),(-65,50)])
add([(-59,47),(-53,46),(-52,49),(-55,52),(-59,51)])
add_ellipse(-126, 49.5, 1.2, 3.0, 20)  # Vancouver Island

# Caribbean.
add([(-85,23),(-80,23),(-75,20),(-77,19),(-82,21)])       # Cuba
add([(-74,20),(-68,19),(-69,17),(-73,18)])                # Hispaniola
add_ellipse(-77, 18, 1.8, .65, 5)                          # Jamaica
add_ellipse(-66.3, 18.2, 1.5, .55, 5)                     # Puerto Rico
add_ellipse(-61, 15, .65, 2.8, -10)                       # Lesser Antilles

# Aleutians and a few large islands of the northeast Pacific.
add_ellipse(-174, 52, 4.5, .65, -8)
add_ellipse(-165, 54, 4.0, .8, -12)
add_ellipse(-153, 57, 2.0, 1.2)

# Other sizeable Atlantic islands.
add_ellipse(-6, 62, .7, 1.4)                              # Faroes
add_ellipse(-28, 38, .8, 2.0, -20)                        # Azores group
add_ellipse(-16, 28.2, .8, 2.3, -10)                      # Canaries
add_ellipse(-24, 16, .7, 1.8, 15)                         # Cape Verde
add_ellipse(-60, -51.5, 2.6, 1.3)                         # Falklands

# Indian Ocean islands.
add_ellipse(73, -0.5, .7, 2.8, 5)                         # Maldives
add_ellipse(57.5, -20.2, .8, 1.2)                         # Mauritius
add_ellipse(55.5, -21.1, .8, 1.4)                         # Reunion
add_ellipse(47, -10, .7, 1.7)                             # Aldabra/Comoros
add_ellipse(93, 8, 1.0, 4.2, -15)                         # Andaman/Nicobar

# Pacific islands that can contain at least one of the two-degree sample points.
add_ellipse(-155.4, 19.5, 1.2, 1.1)                       # Hawaii
add_ellipse(-157.8, 21.2, 1.3, .65, -20)                  # Oahu group
add_ellipse(-90.5, -0.7, 1.6, .8)                         # Galapagos
add_ellipse(166.5, -21, 1.0, 3.0, -25)                    # New Caledonia
add([(154,-5),(158,-7),(160,-10),(158,-11),(155,-8)])     # Solomons
add_ellipse(178, -17, 1.1, 2.3, 20)                       # Fiji
add_ellipse(-172, -14, 1.4, 2.2, 20)                      # Samoa

# Tasmania.
add([(144,-39),(149,-40),(148,-44),(145,-44),(144,-42)])

# Antarctica.  The table is a remembered coarse grounded-coast latitude by
# longitude.  Linear interpolation gives the main polar body; the peninsula is
# added separately.  Deep Ross/Weddell embayments reflect floating shelves as
# water rather than silently counting them as continental land.
ak_lon = np.array([-179,-165,-150,-135,-120,-105,-90,-75,-60,-45,-30,-15,
                     0, 15, 30, 45, 60, 75, 90,105,120,135,150,165,179])
ak_lat = np.array([ -82, -78, -76, -74, -74, -74,-73,-75,-78,-77,-74,-71,
                    -70, -70,-68,-67,-67,-68,-66,-66,-67,-67,-69,-74,-82])
coast = np.interp(X, ak_lon, ak_lat)
land |= Y <= coast
add([(-78,-74),(-74,-70),(-69,-66),(-64,-62),(-58,-63),(-57,-67),
     (-62,-72),(-68,-75)])
# Finalise the binary matrix and write all requested artefacts.
matrix = land.astype(np.uint8)
assert matrix.shape == (90, 180)
assert set(np.unique(matrix)).issubset({0, 1})

out = Path("outputs")
out.mkdir(parents=True, exist_ok=True)
with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
    csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

pixels = matrix * np.uint8(255)
raw = Image.fromarray(pixels, mode="L")
assert raw.size == (180, 90)
raw.save(out / "candidate.png")
large = raw.resize((1440, 720), resample=Image.Resampling.NEAREST)
large.save(out / "candidate_8x.png")

# Self-check files after serialisation.
with (out / "candidate.csv").open(newline="", encoding="ascii") as f:
    rows = list(csv.reader(f))
assert len(rows) == 90 and all(len(r) == 180 for r in rows)
assert all(v in ("0", "1") for r in rows for v in r)
with Image.open(out / "candidate.png") as im:
    assert im.size == (180, 90) and set(im.getdata()) <= {0, 255}
with Image.open(out / "candidate_8x.png") as im:
    assert im.size == (1440, 720) and set(im.getdata()) <= {0, 255}

print("Generated 90x180 matrix (16,200 values) and both binary PNG images.")
