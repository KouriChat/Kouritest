from pathlib import Path
import csv
import math
import numpy as np
from PIL import Image

# All geometry below is a hand-written, deliberately approximate reconstruction.
# Coordinates are (longitude, latitude); no geographic files are read.
LONS = np.arange(-179, 180, 2)
LATS = np.arange(89, -90, -2)
land = np.zeros((len(LATS), len(LONS)), dtype=bool)

def inside_poly(x, y, p):
    c = False
    j = len(p) - 1
    for i in range(len(p)):
        xi, yi = p[i]; xj, yj = p[j]
        if ((yi > y) != (yj > y)):
            xx = (xj-xi)*(y-yi)/(yj-yi) + xi
            if x < xx: c = not c
        j = i
    return c

def add_poly(p):
    for r,y in enumerate(LATS):
        for c,x in enumerate(LONS):
            if inside_poly(x, y, p): land[r,c] = True

def cut_poly(p):
    for r,y in enumerate(LATS):
        for c,x in enumerate(LONS):
            if inside_poly(x, y, p): land[r,c] = False

def add_ellipse(cx, cy, rx, ry, angle=0):
    a=math.radians(angle); ca=math.cos(a); sa=math.sin(a)
    for r,y in enumerate(LATS):
        for c,x in enumerate(LONS):
            dx=x-cx; dy=y-cy
            u=dx*ca+dy*sa; v=-dx*sa+dy*ca
            if (u/rx)**2+(v/ry)**2 <= 1: land[r,c]=True

# North American mainland, including Alaska, Mexico and the Labrador peninsula.
add_poly([
 (-168,72),(-160,70),(-151,71),(-142,70),(-136,68),(-130,70),
 (-120,73),(-108,75),(-96,77),(-84,77),(-74,72),(-68,68),
 (-62,60),(-58,55),(-61,50),(-67,47),(-70,44),(-74,41),
 (-76,38),(-80,35),(-81,31),(-80,25),(-82,24),(-87,30),
 (-91,29),(-97,26),(-97,21),(-91,21),(-87,18),(-89,16),
 (-94,16),(-98,18),(-104,19),(-106,23),(-110,25),(-113,29),
 (-117,32),(-119,34),(-121,37),(-124,40),(-124,46),(-128,50),
 (-132,54),(-137,58),(-145,60),(-153,59),(-158,57),(-164,58),
 (-168,61),(-165,65),(-168,68)
])
# Hudson Bay and its broad northern outlet.
cut_poly([(-96,64),(-91,67),(-84,68),(-78,66),(-77,62),(-80,57),
          (-84,53),(-91,52),(-96,56)])
# Gulf of St Lawrence.
cut_poly([(-70,50),(-65,51),(-59,49),(-61,46),(-67,47)])
# Canadian Arctic islands.
for p in [
 [(-125,74),(-119,72),(-111,73),(-108,77),(-113,80),(-122,79)],
 [(-107,77),(-98,76),(-90,78),(-92,82),(-103,81)],
 [(-85,76),(-77,76),(-68,80),(-72,83),(-82,82)],
 [(-97,69),(-90,68),(-84,71),(-88,74),(-96,73)],
 [(-81,67),(-75,68),(-76,73),(-82,73)],
 [(-128,69),(-121,68),(-117,72),(-124,74)]
]: add_poly(p)
# Greenland
add_poly([(-52,60),(-44,61),(-39,65),(-30,69),(-22,75),(-25,81),
          (-34,84),(-47,83),(-58,78),(-62,72),(-58,65)])
# Aleutians and nearby sizeable islands.
add_ellipse(-173,52.5,4.5,1.0,-5)
add_ellipse(-163,54.5,4.0,1.3,10)
add_ellipse(-135,57,2.2,3.3,-15)
add_ellipse(-127,50.5,2.0,4.0,-15)
add_ellipse(-72,62,2.0,3.0)
add_ellipse(-64,46.5,2.0,1.2,-20)
add_ellipse(-56,49,2.5,2.5)
# Central America (kept separate to retain the narrow isthmus at this sampling).
add_poly([(-92,18),(-87,18),(-84,15),(-83,10),(-79,9),(-77,7),
          (-79,6),(-83,8),(-86,11),(-90,14)])
# Caribbean islands that are resolvable on a two-degree grid.
add_ellipse(-79,21,4.5,1.2,-8)     # Cuba
add_ellipse(-72.5,19,2.6,1.2,-12)  # Hispaniola
add_ellipse(-66.2,18.2,1.5,0.8)
add_ellipse(-77,18,1.7,0.7,-20)

# South America
add_poly([
 (-79,10),(-74,11),(-70,12),(-62,10),(-57,8),(-52,5),(-50,1),
 (-48,-1),(-44,-2),(-41,-5),(-38,-7),(-35,-6),(-35,-11),
 (-38,-15),(-39,-20),(-42,-23),(-46,-25),(-49,-29),(-52,-33),
 (-57,-36),(-62,-39),(-65,-43),(-67,-48),(-69,-53),(-72,-55),
 (-74,-51),(-73,-45),(-72,-40),(-72,-34),(-71,-29),(-70,-24),
 (-70,-19),(-73,-15),(-76,-12),(-78,-8),(-80,-4),(-81,1),
 (-78,5)
])
# Tierra del Fuego
add_ellipse(-69,-54.5,4.0,1.6,-10)

# Africa, with the Horn and Cape represented explicitly.
add_poly([
 (-17,37),(-8,36),(-1,36),(8,37),(11,34),(20,33),(25,31),
 (31,31),(33,28),(35,24),(38,18),(42,13),(51,12),(49,8),
 (44,4),(41,-1),(40,-6),(37,-11),(36,-17),(33,-22),(31,-29),
 (27,-34),(20,-35),(16,-29),(12,-25),(9,-18),(5,-10),(1,-5),
 (-5,0),(-8,5),(-13,9),(-16,15),(-17,21),(-15,27),(-10,31)
])
# Madagascar and nearby substantial islands.
add_poly([(49,-12),(51,-16),(50,-22),(47,-26),(44,-24),(44,-18),(46,-13)])
add_ellipse(43.5,-12,1.2,1.8)

# Western and northern Eurasia.  This is one coastline polygon; seas connected
# to the ocean are subsequently cut out, while inland lakes remain "land".
add_poly([
 (-10,36),(-9,43),(-5,48),(-5,54),(0,58),(5,59),(5,63),(10,64),
 (15,68),(20,70),(24,71),(29,70),(34,67),(40,68),(45,69),
 (51,70),(59,70),(66,72),(72,72),(78,73),(86,74),(94,77),
 (105,78),(116,76),(127,73),(139,72),(151,71),(163,68),(174,66),
 (180,65),(180,48),(173,51),(166,54),(160,58),(153,59),(143,59),
 (137,55),(135,50),(131,46),(126,43),(121,40),(121,35),(124,31),
 (121,28),(118,25),(112,22),(108,20),(106,16),(109,12),(105,8),
 (103,2),(101,1),(100,7),(97,15),(94,20),(90,22),(87,22),
 (82,25),(80,30),(77,35),(73,38),(67,38),(63,40),(58,43),
 (52,45),(48,42),(44,41),(41,38),(36,36),(30,36),(26,39),
 (21,40),(16,39),(12,41),(8,43),(3,43),(-1,40),(-6,37)
])
# Scandinavia is better defined as a projecting peninsula.
add_poly([(5,58),(8,63),(12,68),(17,71),(23,71),(28,69),(25,64),
          (21,60),(16,57),(11,56)])
# Iberia, Italy, Balkans/Greece.
add_poly([(-10,36),(-9,43),(-1,44),(3,42),(3,39),(-1,36)])
add_poly([(7,44),(12,46),(14,42),(18,40),(17,37),(14,38),(11,42)])
add_poly([(14,45),(21,46),(27,42),(27,38),(23,36),(19,39),(15,41)])
# Britain, Ireland, Iceland and Svalbard
add_ellipse(-3,54.5,3.2,5.0,-12)
add_ellipse(-8,53,2.0,3.0)
add_ellipse(-18,65,4.5,2.5,-10)
add_ellipse(17,79,4.0,2.5)
# Ocean-connected seas in Europe.
cut_poly([(10,54),(14,54),(18,55),(22,58),(24,62),(20,66),
          (16,65),(13,59)])  # Baltic
cut_poly([(27,71),(34,71),(40,68),(39,65),(34,64),(30,67)]) # White Sea
cut_poly([(27,40),(31,42),(37,42),(42,45),(40,47),(34,47),
          (29,45)]) # Black Sea
# Sea of Azov is subsumed in the coarse Black Sea cut.

# Arabian peninsula and India.
add_poly([(34,31),(39,32),(44,29),(49,27),(56,25),(55,20),(51,15),
          (46,13),(43,16),(40,20),(36,23)])
add_poly([(67,24),(72,24),(77,22),(82,22),(87,21),(88,17),(84,10),
          (80,7),(77,9),(74,15),(70,20)])
# Sri Lanka
add_ellipse(80.8,7.2,1.2,2.2,-5)
# Sinai is land but keep the Red Sea and Persian Gulf as water.
add_poly([(32,31),(35,31),(34,27),(32,28)])
cut_poly([(36,28),(39,25),(43,19),(44,14),(42,13),(39,18),(36,23)])
cut_poly([(48,30),(54,30),(57,27),(54,24),(49,25),(47,28)])

# Kamchatka, Korea, Sakhalin and Japan.
add_poly([(157,51),(159,57),(163,61),(166,59),(164,53),(160,49)])
add_poly([(126,42),(130,42),(130,34),(127,34),(125,38)])
add_ellipse(143,50,1.5,4.3,-8)
add_ellipse(141,38,1.5,4.5,-18)
add_ellipse(134,34,3.5,1.1,18)
add_ellipse(130,32,1.2,2.0,-15)
add_ellipse(142.5,44,1.8,2.8,-25)
# Taiwan and Hainan.
add_ellipse(121,24,1.0,2.0,-10)
add_ellipse(110,19,1.6,1.2)

# Philippines (coarse island groups)
add_ellipse(121,17,1.4,3.0,-10)
add_ellipse(124,12,1.8,3.0,-15)
add_ellipse(122,8,1.5,2.0)

# Sumatra, Java, Borneo, Sulawesi and Lesser Sundas.
add_ellipse(101,-1,2.2,6.5,-42)
add_ellipse(110,-7.2,5.7,1.1,-5)
add_poly([(109,7),(116,7),(119,3),(118,-3),(114,-5),(109,-2)])
add_poly([(119,2),(124,2),(125,-2),(122,-5),(120,-3)])
add_ellipse(120,-8.5,5.0,0.8,0)
add_ellipse(127,-3,2.0,2.0)
add_ellipse(129,1,1.0,2.5)

# New Guinea and its eastern island arc.
add_poly([(130,-2),(136,-1),(141,-3),(147,-6),(151,-6),(151,-9),
          (146,-10),(140,-8),(135,-7),(130,-5)])
add_ellipse(153,-5,1.0,2.5,-25)
add_ellipse(156,-7,1.0,2.0,-35)
add_ellipse(160,-9,1.0,2.0,-35)
add_ellipse(166,-16,1.0,3.0,-35)

# Australia
add_poly([(113,-12),(119,-14),(125,-14),(129,-12),(136,-12),(142,-11),
          (146,-15),(148,-20),(153,-25),(153,-30),(150,-35),(145,-39),
          (138,-38),(132,-35),(125,-34),(117,-35),(114,-31),(113,-25),
          (111,-20)])
add_ellipse(147,-42.5,2.4,2.4)

# New Zealand
add_ellipse(174,-40,1.3,4.2,-30)
add_ellipse(169,-45,1.4,4.0,-35)

# Miscellaneous large, potentially sampled islands.
add_ellipse(48,-5,1.2,2.0)      # Seychelles region (coarse)
add_ellipse(55.5,-21,1.0,1.6)  # Reunion/Mauritius vicinity
add_ellipse(-25,38,2.0,1.0)    # Azores
add_ellipse(-16,28,1.0,1.8)    # Canaries
add_ellipse(-6,-16,1.0,1.0)    # St Helena
add_ellipse(-60,-52,2.5,1.0)   # Falklands
add_ellipse(167,-22,1.0,2.0)   # New Caledonia
add_ellipse(-157,20.5,1.5,1.0) # Hawaii
# Antarctica: hand-written grounding-coast approximation.  The deep Ross and
# Weddell embayments are water (including their floating shelves), while the
# high interior and grounded ice are land.
def antarctic_edge(x):
    knots = [
      (-180,-78),(-165,-78),(-150,-75),(-130,-74),(-110,-73),
      (-90,-72),(-75,-73),(-62,-75),(-45,-78),(-25,-76),
      (-10,-71),(15,-70),(35,-69),(55,-67),(75,-67),(95,-66),
      (115,-66),(135,-67),(150,-69),(165,-75),(180,-78)]
    for (x0,y0),(x1,y1) in zip(knots,knots[1:]):
        if x0 <= x <= x1:
            return y0+(y1-y0)*(x-x0)/(x1-x0)
    return -78

for r,y in enumerate(LATS):
    for c,x in enumerate(LONS):
        if y <= antarctic_edge(x): land[r,c] = True
# Antarctic Peninsula and attached grounded coast.
add_poly([(-75,-74),(-71,-69),(-66,-64),(-61,-63),(-57,-65),
          (-62,-70),(-67,-74)])
# A few near-coastal Antarctic islands large enough to affect this grid.
add_ellipse(-60,-62,1.8,1.0)
add_ellipse(-45,-60.5,1.5,1.0)

# Validation and required output products.
assert land.shape == (90,180)
arr = land.astype(np.uint8)
assert set(np.unique(arr)).issubset({0,1})
out = Path("outputs")
out.mkdir(parents=True, exist_ok=True)
with (out/"candidate.csv").open("w", newline="", encoding="ascii") as f:
    csv.writer(f, lineterminator="\n").writerows(arr.tolist())

pixels = arr * 255
im = Image.fromarray(pixels, mode="L")
assert im.size == (180,90)
im.save(out/"candidate.png")
big = im.resize((1440,720), resample=Image.Resampling.NEAREST)
big.save(out/"candidate_8x.png")

# Re-open and verify the delivered files, including exact 8x block replication.
with (out/"candidate.csv").open(newline="", encoding="ascii") as f:
    rows = list(csv.reader(f))
assert len(rows)==90 and all(len(row)==180 for row in rows)
assert all(v in ("0","1") for row in rows for v in row)
check = np.array(Image.open(out/"candidate.png"))
check8 = np.array(Image.open(out/"candidate_8x.png"))
assert check.shape==(90,180) and set(np.unique(check)) <= {0,255}
assert check8.shape==(720,1440) and set(np.unique(check8)) <= {0,255}
assert np.array_equal(check8, np.repeat(np.repeat(check,8,0),8,1))
print("Generated and verified 90x180 matrix and 180x90 / 1440x720 images.")
