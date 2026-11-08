from pathlib import Path
import csv
import numpy as np
from PIL import Image

# The grid is sampled at odd integer degree coordinates, north to south.
lats = np.arange(89, -90, -2, dtype=float)
lons = np.arange(-179, 180, 2, dtype=float)
LON, LAT = np.meshgrid(lons, lats)
land = np.zeros(LON.shape, dtype=bool)

def poly(points):
    """Union a hand-written lon/lat polygon into the land mask."""
    global land
    x, y = LON, LAT
    inside = np.zeros(x.shape, dtype=bool)
    p = list(points)
    for (x1, y1), (x2, y2) in zip(p, p[1:] + p[:1]):
        if y1 == y2:
            continue
        cross = ((y1 > y) != (y2 > y)) & (x < x1 + (x2-x1)*(y-y1)/(y2-y1))
        inside ^= cross
    land |= inside

def water_poly(points):
    """Carve a sea or ocean-connected bay from an existing polygon."""
    global land
    x, y = LON, LAT
    inside = np.zeros(x.shape, dtype=bool)
    p = list(points)
    for (x1, y1), (x2, y2) in zip(p, p[1:] + p[:1]):
        if y1 == y2:
            continue
        cross = ((y1 > y) != (y2 > y)) & (x < x1 + (x2-x1)*(y-y1)/(y2-y1))
        inside ^= cross
    land &= ~inside

def ellipse(cx, cy, rx, ry, angle=0):
    """Union a rotated ellipse, useful for compact islands."""
    global land
    a = np.deg2rad(angle)
    dx, dy = LON-cx, LAT-cy
    u = dx*np.cos(a) + dy*np.sin(a)
    v = -dx*np.sin(a) + dy*np.cos(a)
    land |= (u/rx)**2 + (v/ry)**2 <= 1.0

# North and Central American mainland (coastline intentionally approximate).
poly([(-168,66),(-164,61),(-158,58),(-152,58),(-147,60),(-140,60),
      (-136,57),(-132,55),(-128,51),(-125,47),(-124,42),(-121,37),
      (-117,33),(-112,31),(-108,25),(-105,20),(-99,17),(-94,16),
      (-90,14),(-87,12),(-83,8),(-78,8),(-77,10),(-81,11),
      (-83,15),(-88,16),(-87,21),(-90,22),(-97,25),(-97,29),
      (-91,30),(-86,30),(-83,29),(-81,25),(-80,27),(-81,31),
      (-79,34),(-76,36),(-75,39),(-71,42),(-67,44),(-66,47),
      (-61,49),(-57,53),(-60,57),(-64,60),(-69,62),(-78,63),
      (-82,66),(-89,68),(-96,69),(-105,69),(-114,70),(-123,72),
      (-132,70),(-141,69),(-151,71),(-161,70)])
# Baja California and Florida are narrower than the mainland polygon permits.
poly([(-117,32),(-114,32),(-111,27),(-109,23),(-110,22),(-113,27)])
poly([(-84,30),(-81,30),(-80,26),(-81,24),(-83,27)])
# Ocean-connected Hudson/James Bay and Gulf of California.
water_poly([(-95,64),(-86,64),(-79,61),(-78,55),(-81,51),(-86,52),
            (-88,56),(-94,58)])
water_poly([(-114,31),(-111,29),(-109,24),(-111,25)])

# Greenland and the larger Canadian Arctic islands.
poly([(-52,59),(-44,60),(-39,64),(-32,68),(-25,72),(-18,77),
      (-21,81),(-35,83),(-50,82),(-61,78),(-68,73),(-61,66)])
poly([(-79,63),(-69,63),(-62,66),(-62,70),(-67,74),(-72,77),
      (-78,75),(-82,70)])                         # Baffin
poly([(-85,76),(-75,76),(-62,80),(-66,83),(-82,83),(-91,80)]) # Ellesmere
poly([(-121,70),(-113,68),(-105,69),(-104,73),(-112,76),(-122,74)])
poly([(-126,70),(-120,74),(-123,77),(-133,75),(-134,71)])     # Banks
poly([(-107,74),(-96,74),(-91,77),(-96,80),(-108,79)])        # Devon
poly([(-101,69),(-94,68),(-92,72),(-101,73)])                  # Boothia

# South America.
poly([(-79,10),(-73,12),(-66,11),(-60,9),(-54,6),(-50,2),
      (-43,-2),(-35,-6),(-37,-12),(-40,-20),(-46,-27),
      (-51,-32),(-57,-36),(-63,-40),(-66,-46),(-68,-52),
      (-67,-55),(-72,-56),(-75,-50),(-74,-42),(-73,-35),
      (-71,-27),(-70,-19),(-75,-14),(-78,-9),(-81,-5),(-80,2)])
poly([(-74,-52),(-68,-53),(-65,-55),(-69,-56),(-73,-55)])     # Tierra del Fuego

# Eurasian mainland.  Peninsulas are included in the same coarse outer contour;
# major ocean-connected seas accidentally enclosed by it are carved below.
poly([(-10,36),(-9,43),(-5,44),(-2,48),(2,51),(7,54),(9,58),
      (6,62),(10,67),(17,70),(26,71),(35,70),(42,68),(52,69),
      (62,70),(72,72),(83,73),(95,76),(108,76),(120,74),(132,72),
      (145,72),(158,70),(171,67),(178,65),(170,62),(164,60),
      (160,55),(157,51),(151,49),(143,47),(137,44),(131,43),
      (130,39),(126,36),(123,34),(121,29),(121,23),(116,21),
      (110,21),(108,17),(109,12),(105,8),(104,2),(101,1),(100,6),
      (97,11),(96,16),(92,21),(88,22),(86,20),(82,17),(79,10),
      (77,7),(74,15),(70,21),(66,24),(61,25),(57,25),(55,27),
      (51,27),(49,30),(47,30),(49,26),(54,24),(58,22),(57,18),
      (53,15),(51,12),(45,12),(42,16),(40,21),(37,25),(35,30),
      (35,33),(30,36),(26,40),(20,41),(16,43),(12,44),(8,43),
      (4,43),(1,42),(-2,43),(-7,44),(-9,40)])
# Scandinavia's east side and Finland, preventing an oversized Baltic coast.
poly([(9,55),(13,55),(17,58),(20,60),(24,65),(30,69),(25,71),
      (17,70),(10,67),(6,62)])
# Italy, Greece, and the west coast of the Black Sea region.
poly([(7,44),(12,46),(14,43),(16,40),(16,38),(13,38),(12,41),(9,43)])
poly([(19,42),(23,42),(25,39),(23,36),(21,38)])

# Large marginal seas (inland lakes are deliberately not removed).
water_poly([(12,54),(18,54),(22,57),(23,61),(20,65),(16,62),(13,58)]) # Baltic
water_poly([(142,48),(150,48),(157,51),(160,56),(157,59),(150,58),
            (143,54)])                                                   # Okhotsk
water_poly([(130,35),(135,35),(140,39),(141,44),(138,47),(132,43)])     # Japan Sea
water_poly([(48,30),(52,30),(56,27),(55,25),(51,25),(49,27)])           # Persian Gulf
water_poly([(35,30),(38,27),(41,21),(44,15),(42,13),(39,18),(36,25)])   # Red Sea
water_poly([(26,41),(30,42),(36,42),(41,45),(40,47),(34,46),(29,44)])   # Black Sea

# Africa.
poly([(-17,28),(-11,34),(-6,36),(-1,35),(5,37),(10,37),(12,33),
      (20,32),(27,31),(33,31),(35,28),(37,23),(40,18),(43,12),
      (51,12),(50,7),(45,2),(42,-3),(41,-10),(39,-16),(35,-23),
      (32,-30),(27,-34),(20,-35),(16,-32),(13,-26),(11,-18),
      (9,-8),(10,-2),(8,4),(3,5),(-2,5),(-6,4),(-11,7),
      (-15,13),(-17,21)])

# Australia, including the Gulf of Carpentaria indentation.
poly([(113,-22),(114,-16),(120,-14),(126,-14),(130,-12),(135,-12),
      (137,-15),(137,-18),(140,-18),(141,-11),(144,-14),(147,-19),
      (151,-24),(153,-29),(151,-35),(146,-39),(139,-39),(135,-36),
      (130,-33),(124,-34),(117,-35),(113,-29)])

# Antarctica: the top edge approximates the grounded coast; the Ross and
# Weddell embayments are kept farther south than nearby continental coasts.
poly([(-180,-78),(-170,-78),(-160,-77),(-150,-75),(-138,-74),
      (-125,-74),(-112,-73),(-101,-73),(-91,-74),(-82,-75),
      (-76,-72),(-71,-68),(-66,-64),(-61,-63),(-58,-65),
      (-62,-69),(-69,-73),(-75,-76),(-64,-78),(-50,-78),
      (-40,-76),(-30,-73),(-18,-71),(0,-70),(20,-69),(40,-67),
      (60,-67),(80,-66),(100,-66),(120,-67),(140,-67),(155,-69),
      (170,-72),(180,-74),(180,-90),(-180,-90)])

# Atlantic and American islands.
poly([(-59,47),(-55,46),(-52,48),(-54,51),(-57,52),(-60,50)]) # Newfoundland
poly([(-67,43),(-62,44),(-60,46),(-64,46)])                    # Nova Scotia
ellipse(-126,49.2,2.8,1.0,25)                                  # Vancouver Island
ellipse(-153.5,57.5,2.0,1.2,25)                                # Kodiak
ellipse(-79.5,21.5,5.6,1.25,7)                                 # Cuba
ellipse(-71.1,19.0,3.2,1.25,5)                                 # Hispaniola
ellipse(-77.3,18.1,1.3,.55,10)                                 # Jamaica
ellipse(-66.4,18.2,1.1,.50,5)                                  # Puerto Rico
ellipse(-91.1,-.6,.65,.65)                                     # Galapagos/Isabela
ellipse(-59.0,-51.7,2.0,1.1)                                   # Falkland Islands

# European and Mediterranean islands.
poly([(-6,50),(-4,50),(-3,53),(-5,55),(-4,58),(0,58),(1,54),(1,51)]) # Britain
ellipse(-8,53,1.8,2.7,-10)                                     # Ireland
poly([(-24,64),(-20,63),(-14,64),(-13,66),(-18,67),(-23,66)])  # Iceland
ellipse(17,78.5,5.5,2.7,10)                                    # Svalbard
ellipse(58.5,74,3.0,7.0,-35)                                   # Novaya Zemlya
ellipse(100,79,5.0,2.1,5)                                      # Severnaya Zemlya
ellipse(141,75,3.0,1.5)                                        # New Siberian Islands
ellipse(179,71,2.3,1.2,-20)                                    # Wrangel
ellipse(9,40,1.6,2.4,-8)                                       # Sardinia
ellipse(9,42.5,.8,1.4)                                         # Corsica
ellipse(14,37.5,2.2,1.1)                                       # Sicily
ellipse(3,39.5,1.4,.8)                                         # Mallorca
ellipse(24.5,35.2,3.0,.65)                                     # Crete
ellipse(33.2,35.0,1.8,.85)                                     # Cyprus

# African islands.
poly([(49,-12),(50,-16),(48,-21),(45,-25),(44,-21),(45,-16),(47,-13)]) # Madagascar
ellipse(-15.5,28.3,.7,.8)                                      # Canary group footprint
ellipse(55.5,-4.5,.6,.8)                                       # Seychelles footprint

# East and South Asian islands.
ellipse(143,50,1.8,4.8,-8)                                     # Sakhalin
ellipse(143,43.3,2.5,1.8,10)                                   # Hokkaido
ellipse(138.2,36.7,6.2,1.15,-32)                               # Honshu
ellipse(133.3,33.7,2.0,.75,-18)                                # Shikoku
ellipse(130.5,32.5,2.5,1.3,-35)                                # Kyushu
ellipse(121,23.7,1.0,2.2,-12)                                  # Taiwan
ellipse(110,19.2,2.0,1.3)                                      # Hainan
poly([(120,19),(122,18),(122,14),(120,13),(119,16)])            # Luzon
poly([(123,12),(125,12),(126,10),(125,8),(122,9)])              # central Philippines
ellipse(124.5,7.5,2.2,2.0,10)                                  # Mindanao
ellipse(118.8,9.5,.7,3.1,-20)                                  # Palawan
poly([(95,6),(98,6),(102,2),(106,-4),(104,-6),(100,-3),(97,1)]) # Sumatra
poly([(105,-6),(110,-6),(115,-8),(113,-9),(107,-8)])            # Java
poly([(109,6),(114,7),(118,5),(119,1),(116,-4),(111,-4),(108,0)]) # Borneo
poly([(120,1),(124,2),(122,-2),(125,-4),(122,-6),(120,-3)])     # Sulawesi
ellipse(127,-3,2.2,3.0,25)                                     # Moluccas/Halmahera
poly([(130,-1),(137,-1),(143,-3),(151,-5),(150,-8),(145,-10),
      (138,-8),(132,-6)])                                      # New Guinea
ellipse(120,-9,6.5,.75,-5)                                     # Lesser Sundas
poly([(79,10),(80,6),(81,5),(82,8),(81,10)])                   # Sri Lanka
ellipse(93,12,1.0,2.8,-10)                                     # Andaman/Nicobar

# Australia-region islands and remote large islands.
ellipse(147,-42.2,2.4,3.2,-12)                                 # Tasmania
poly([(172,-34),(176,-37),(177,-40),(174,-41),(172,-38)])       # New Zealand North
poly([(166,-41),(171,-42),(174,-45),(171,-47),(167,-46)])       # New Zealand South
ellipse(167,-16.5,1.0,3.2,-18)                                 # Vanuatu chain
ellipse(162,-10,1.0,3.2,-25)                                   # Solomon chain
ellipse(145,-6,1.0,2.8,-30)                                    # Bismarck region
ellipse(-155.5,19.5,.8,.9)                                     # Hawaii (large island)
ellipse(-172.5,-14.2,.7,.7)                                    # Samoa footprint
ellipse(178,-17.8,.7,1.4,-30)                                  # Fiji footprint
ellipse(166,-22,.7,.9)                                         # New Caledonia vicinity

# Write and independently validate all requested artifacts.
matrix = land.astype(np.uint8)
assert matrix.shape == (90, 180)
assert matrix.size == 16200 and np.isin(matrix, (0, 1)).all()
out = Path("outputs")
out.mkdir(parents=True, exist_ok=True)
with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
    csv.writer(f, lineterminator="\n").writerows(matrix.tolist())
raw = Image.fromarray(matrix * 255, mode="L")
raw.save(out / "candidate.png")
large = raw.resize((1440, 720), resample=Image.Resampling.NEAREST)
large.save(out / "candidate_8x.png")
assert raw.size == (180, 90) and large.size == (1440, 720)
assert set(raw.getdata()) <= {0, 255} and set(large.getdata()) <= {0, 255}

