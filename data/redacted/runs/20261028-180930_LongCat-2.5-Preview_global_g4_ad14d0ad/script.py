from pathlib import Path
import math
import numpy as np
from PIL import Image

# Knowledge-based world map generator.  All shapes below are hand-written
# geographic approximations; no external map or geographic database is used.
LATS = np.arange(89, -90, -2, dtype=float)
LONS = np.arange(-179, 180, 2, dtype=float)


def in_poly(lon, lat, pts):
    """Even-odd ray casting for a lon/lat polygon."""
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if ((yi > lat) != (yj > lat)):
            x = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x:
                inside = not inside
        j = i
    return inside


def in_ellipse(lon, lat, cx, cy, rx, ry, angle=0.0):
    a = math.radians(angle)
    dx, dy = lon - cx, lat - cy
    u = dx * math.cos(a) + dy * math.sin(a)
    v = -dx * math.sin(a) + dy * math.cos(a)
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1.0


def polygon(*pts):
    return ("poly", pts)


def ellipse(cx, cy, rx, ry, angle=0.0):
    return ("ellipse", (cx, cy, rx, ry, angle))


def any_shape(lon, lat, shapes):
    for kind, data in shapes:
        if kind == "poly" and in_poly(lon, lat, data):
            return True
        if kind == "ellipse" and in_ellipse(lon, lat, *data):
            return True
    return False


def coast_land(lon, lat, table):
    """Land lies south of an interpolated, hand-written coastline."""
    xs = [p[0] for p in table]
    ys = [p[1] for p in table]
    if lon <= xs[0]:
        c = ys[0]
    elif lon >= xs[-1]:
        c = ys[-1]
    else:
        i = 0
        while lon > xs[i + 1]:
            i += 1
        f = (lon - xs[i]) / (xs[i + 1] - xs[i])
        c = ys[i] + f * (ys[i + 1] - ys[i])
    return lat <= c

north_america = [polygon(
    (-168,66),(-166,69),(-160,71),(-155,71),(-145,70),(-135,69),
    (-125,70),(-115,69),(-105,69),(-95,68),(-85,70),(-80,73),
    (-70,73),(-62,67),(-64,60),(-78,62),(-78,56),(-82,55),
    (-85,52),(-80,45),(-74,45),(-70,42),(-75,37),(-81,32),
    (-80,25),(-83,25),(-85,30),(-90,29),(-94,29),(-97,26),
    (-97,21),(-92,18),(-88,21),(-87,16),(-83,9),(-78,9),
    (-79,7),(-83,8),(-87,13),(-92,15),(-96,16),(-105,20),
    (-110,24),(-114,28),(-117,32),(-122,37),(-124,42),(-124,48),
    (-130,54),(-140,60),(-150,59),(-158,56),(-162,59),(-165,62)
)]

# Seas that are enclosed by the deliberately broad continental outline.
water_masks = [
    polygon((-96,60),(-90,64),(-82,64),(-78,60),(-80,55),(-86,52),
            (-93,55),(-96,60)),                       # Hudson Bay
    polygon((10,54),(13,56),(20,56),(25,60),(30,65),(25,66),(20,63),
            (17,58),(12,56),(10,54)),                 # Baltic Sea
    polygon((27,41),(30,47),(38,47),(42,43),(40,40),(34,40),(27,41)),
                                                      # Black Sea
    polygon((47,29),(50,30),(56,27),(57,24),(54,23),(50,25),(47,29)),
                                                      # Persian Gulf
]

south_america = [polygon(
    (-81,8),(-77,8),(-75,11),(-71,12),(-63,11),(-60,8),(-52,5),
    (-50,0),(-44,-2),(-38,-5),(-35,-8),(-39,-13),(-40,-20),
    (-48,-25),(-54,-34),(-58,-38),(-62,-41),(-65,-45),(-68,-50),
    (-70,-55),(-75,-53),(-74,-45),(-73,-37),(-71,-30),(-70,-23),
    (-70,-18),(-75,-15),(-77,-10),(-81,-6),(-80,0),(-78,3)
)]

africa = [polygon(
    (-17,15),(-16,21),(-13,28),(-10,30),(-6,36),(0,36),(10,37),
    (11,34),(20,32),(25,32),(32,31),(34,28),(37,22),(43,12),
    (51,7),(47,2),(42,-2),(40,-10),(40,-18),(35,-24),(32,-28),
    (28,-33),(20,-35),(18,-34),(15,-27),(12,-18),(9,-2),(5,4),
    (-4,5),(-8,4),(-13,8)
)]

# Broad continental fill is trimmed at the Red Sea and Gulf of Guinea.
water_masks += [
    polygon((32,30),(35,28),(43,12),(40,11),(37,18),(32,30)),  # Red Sea
]

eurasia = [polygon(
    (-9,43),(-9,38),(-6,36),(-2,37),(0,40),(3,42),(4,43),(8,44),
    (11,44),(13,45),(15,45),(18,42),(20,40),(22,37),(24,38),
    (26,40),(29,40),(31,37),(35,36),(36,32),(40,30),(43,29),
    (45,30),(48,30),(52,28),(55,25),(58,23),(62,25),(65,25),
    (68,24),(70,20),(74,12),(77,8),(80,15),(82,19),(87,21),
    (90,22),(94,20),(97,16),(98,8),(100,5),(103,1),(105,10),
    (108,15),(110,20),(118,22),(121,25),(122,31),(126,35),
    (130,35),(135,40),(140,42),(142,46),(150,46),(155,50),
    (158,52),(160,57),(165,60),(172,60),(180,62),(180,66),
    (170,68),(160,70),(150,72),(140,73),(130,73),(120,74),
    (110,76),(100,77),(90,76),(80,74),(70,73),(60,70),(50,69),
    (40,68),(30,70),(25,65),(21,60),(18,57),(14,55),(12,55),
    (10,58),(8,57),(8,54),(4,53),(2,51),(-4,48),(0,47),(-2,43)
)]

# Mediterranean and North Sea are outside this outline; these masks clean
# small overhangs and the enclosed European seas.
water_masks += [
    polygon((-4,51),(4,53),(8,57),(5,61),(-2,59),(-4,55),(-4,51)),
                                                       # North Sea
]

australia = [polygon(
    (114,-22),(113,-26),(115,-34),(119,-34),(123,-34),(129,-32),
    (135,-35),(138,-35),(141,-38),(146,-39),(150,-37),(153,-29),
    (153,-25),(146,-19),(143,-14),(142,-11),(137,-12),(132,-11),
    (130,-14),(125,-16),(122,-18)
)]

greenland = [polygon(
    (-73,78),(-60,82),(-40,83),(-20,82),(-18,76),(-22,70),(-32,68),
    (-42,60),(-50,61),(-53,66),(-58,70),(-62,75)
)]

# Northern limit of the grounded Antarctic continent, interpolated by
# longitude.  Floating ice shelves are deliberately not included.
antarctic_coast = [
    (-180,-78),(-170,-78),(-160,-78),(-150,-75),(-140,-74),(-130,-74),
    (-120,-73),(-110,-74),(-100,-73),(-90,-73),(-80,-72),(-70,-70),
    (-65,-66),(-60,-63),(-55,-65),(-50,-70),(-45,-75),(-40,-78),
    (-30,-77),(-20,-75),(-10,-72),(0,-70),(10,-70),(20,-70),(30,-69),
    (40,-68),(50,-66),(60,-67),(70,-69),(80,-67),(90,-66),(100,-66),
    (110,-66),(120,-66),(130,-66),(140,-66),(150,-69),(160,-71),
    (170,-73),(180,-78)
]

islands = [
    polygon((-5,50),(-3,50),(1,51),(-1,53),(-3,55),(-5,58),(-6,57),
            (-5,55)),                                  # Great Britain
    ellipse(-8,53,2.1,2.6),                            # Ireland
    ellipse(-19,65,4.0,1.6,20),                        # Iceland
    ellipse(16,78,3.0,2.2),                            # Svalbard
    polygon((52,70),(55,71),(57,74),(56,77),(53,76),(51,73)), # Novaya Zemlya
    ellipse(98,79,3.2,1.8),                            # Severnaya Zemlya
    ellipse(140,74,4.2,1.8),                           # New Siberian Is.
    ellipse(179,71,1.1,1.0),                           # Wrangel I.
    ellipse(142,43,2.4,2.0,25),                        # Hokkaido
    polygon((136,34),(139,35),(142,39),(141,41),(138,37),(135,35)),
                                                       # Honshu
    ellipse(131,32,1.4,1.4),                           # Kyushu
    ellipse(134,34,1.5,0.7,35),                        # Shikoku
    polygon((142,46),(144,46),(144,52),(142,54),(141,50)), # Sakhalin
    ellipse(121,24,0.9,1.9),                           # Taiwan
    ellipse(110,19,1.3,0.9),                           # Hainan
    ellipse(121,16,2.0,3.5,-10),                       # Luzon
    ellipse(123,8,2.1,2.2),                            # Mindanao
    ellipse(123,11,1.2,2.0),                           # Visayas
    ellipse(81,7.5,1.5,2.5),                           # Sri Lanka
    ellipse(47,-19,2.6,6.0,10),                        # Madagascar
    ellipse(101,-1,4.0,7.2,-32),                       # Sumatra
    ellipse(110,-7,7.0,1.0,5),                         # Java
    ellipse(114,0,6.2,5.0),                            # Borneo
    polygon((119,-5),(123,-5),(125,-1),(123,2),(120,1),(118,-2)),
                                                       # Sulawesi
    polygon((131,-1),(141,-2),(151,-6),(150,-9),(141,-8),(133,-4)),
                                                       # New Guinea
    ellipse(175,-38,2.1,4.0,-15),                      # North Island NZ
    ellipse(170,-44,2.0,5.0,10),                       # South Island NZ
    ellipse(147,-42,1.2,1.4),                          # Tasmania
]

islands += [
    ellipse(-79,22,6.0,1.1,-8),                        # Cuba
    ellipse(-71,19,4.0,1.2,-12),                       # Hispaniola
    ellipse(-77,18,1.2,0.7),                           # Jamaica
    ellipse(-66,18,1.3,0.6),                           # Puerto Rico
    ellipse(-74,24,3.0,0.7),                           # Bahamas
    ellipse(-56,49,3.0,2.2,-20),                       # Newfoundland
    ellipse(-60,45,2.0,1.0),                           # Nova Scotia
    ellipse(-72,69,8.0,5.0,10),                        # Baffin Island
    ellipse(-110,70,6.5,4.0),                          # Victoria Island
    ellipse(-80,80,5.0,4.0),                           # Ellesmere Island
    ellipse(-122,73,5.0,3.0,-10),                      # Banks Island
    ellipse(-92,75,4.0,3.0),                           # Devon Island
    ellipse(-100,73,4.0,2.2),                          # Prince of Wales
    ellipse(-84,73,4.0,2.0),                           # Somerset Island
    ellipse(-84,64,5.0,4.0),                           # Southampton I.
    ellipse(-106,76,3.0,2.0),                          # Melville Island
    ellipse(-40,72,5.0,7.0,-15),                       # East Greenland is.
    ellipse(19,70,3.5,2.2,20),                         # Norway islands
    ellipse(13,56,1.4,0.8),                            # Denmark islands
    ellipse(15,38,1.5,0.8,30),                         # Sicily
    ellipse(9,40,1.0,2.0),                             # Sardinia
    ellipse(9,42,0.8,1.1),                             # Corsica
    ellipse(25,35,1.8,0.7,20),                         # Crete
    ellipse(33,35,1.0,0.6),                            # Cyprus
    ellipse(2,39,1.2,0.7),                             # Balearic Is.
]

islands += [
    ellipse(-16,28,2.5,1.0,25),                        # Canary Islands
    ellipse(-25,38,1.5,0.8),                           # Azores
    ellipse(-17,33,1.2,0.6),                           # Madeira
    ellipse(-24,16,1.8,0.8),                           # Cape Verde
    ellipse(-64,32,0.7,0.5),                           # Bermuda
    ellipse(-59,-52,2.2,1.2),                          # Falkland Islands
    ellipse(-37,-54,1.2,0.8),                          # South Georgia
    ellipse(-68,-55,2.0,0.8,-30),                      # Tierra del Fuego
    ellipse(70,-49,2.2,1.2),                           # Kerguelen
    ellipse(55,-21,1.2,0.8),                           # Reunion
    ellipse(57,-20,0.8,0.6),                           # Mauritius
    ellipse(44,-12,0.8,1.2),                           # Comoros
    ellipse(55,-4,0.8,0.8),                            # Seychelles
    ellipse(54,12,0.9,0.6),                            # Socotra
    ellipse(93,12,1.4,3.0),                            # Andaman Islands
    ellipse(94,7,0.9,1.8),                             # Nicobar Islands
    ellipse(73,4,0.6,1.2),                             # Maldives
    ellipse(166,-16,1.0,2.2,-20),                      # Vanuatu
    ellipse(160,-9,2.0,0.9),                           # Solomon Islands
    ellipse(178,-18,1.2,1.1),                          # Fiji
    ellipse(165,-21,1.2,1.5),                          # New Caledonia
    ellipse(152,-5,2.5,1.2),                           # New Britain
    ellipse(147,-6,2.8,1.5),                           # New Britain/Bismarck
    ellipse(134,7,0.8,1.0),                            # Palau
    ellipse(145,14,0.8,0.6),                           # Guam
    ellipse(-157,20,1.0,0.7),                          # Hawaii
    ellipse(-149,-17,1.0,0.8),                         # Tahiti
    ellipse(-172,-14,0.9,0.7),                         # Samoa
    ellipse(-90,-1,1.2,1.0),                           # Galapagos
    ellipse(105,-10,0.8,0.6),                          # Christmas I.
]

def is_land(lon, lat):
    base = (any_shape(lon, lat, north_america) or
            any_shape(lon, lat, south_america) or
            any_shape(lon, lat, africa) or
            any_shape(lon, lat, eurasia) or
            any_shape(lon, lat, australia) or
            any_shape(lon, lat, greenland))
    if any_shape(lon, lat, water_masks):
        base = False
    return (base or any_shape(lon, lat, islands) or
            coast_land(lon, lat, antarctic_coast))


def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    matrix = np.zeros((len(LATS), len(LONS)), dtype=np.uint8)
    for i, lat in enumerate(LATS):
        for j, lon in enumerate(LONS):
            matrix[i, j] = 1 if is_land(float(lon), float(lat)) else 0
    assert matrix.shape == (90, 180)
    assert matrix.size == 16200
    assert np.isin(matrix, (0, 1)).all()
    np.savetxt(out / "candidate.csv", matrix, fmt="%d", delimiter=",")
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(out / "candidate.png")
    image.resize((1440, 720), resample=Image.Resampling.NEAREST).save(
        out / "candidate_8x.png")
    print("shape:", matrix.shape, "land:", int(matrix.sum()),
          "water:", int(matrix.size - matrix.sum()))


if __name__ == "__main__":
    main()
