"""Knowledge-based coarse world land/water reconstruction.
No geographic data are read or downloaded; all shapes below are hand-written.
"""
import csv
import os
import numpy as np
from PIL import Image

OUT = "outputs"

# Coordinates are (longitude, latitude), in degrees.
LAND = []
WATER = []


def poly(points):
    LAND.append(points)


def water_poly(points):
    WATER.append(points)


def ellipse(lon, lat, rx, ry):
    """An intentionally simple island/lake outline."""
    pts = []
    for k in range(24):
        a = 2.0 * np.pi * k / 24.0
        pts.append((lon + rx * np.cos(a), lat + ry * np.sin(a)))
    return pts


def island(lon, lat, rx, ry):
    LAND.append(ellipse(lon, lat, rx, ry))


def lake(lon, lat, rx, ry):
    WATER.append(ellipse(lon, lat, rx, ry))


def inside(x, y, points):
    """Even-odd point-in-polygon test."""
    hit = False
    j = len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        cross = ((yi > y) != (yj > y))
        if cross and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            hit = not hit
        j = i
    return hit


def land_at(lon, lat):
    return any(inside(lon, lat, p) for p in LAND) and not any(
        inside(lon, lat, p) for p in WATER
    )
# North America: Arctic Canada, the United States, Mexico and Central America.
poly([
    (-168, 71), (-160, 70), (-150, 72), (-140, 70), (-132, 69),
    (-125, 70), (-118, 72), (-105, 74), (-92, 73), (-82, 70),
    (-72, 69), (-62, 65), (-55, 61), (-58, 55), (-62, 52),
    (-67, 48), (-66, 45), (-61, 46), (-58, 50), (-54, 52),
    (-52, 48), (-58, 44), (-66, 43), (-70, 42), (-75, 40),
    (-81, 31), (-80, 26), (-84, 25), (-88, 29), (-93, 29),
    (-97, 26), (-102, 23), (-107, 21), (-111, 18), (-116, 18),
    (-118, 23), (-114, 28), (-112, 31), (-117, 32), (-122, 31),
    (-125, 35), (-124, 40), (-130, 45), (-137, 51), (-143, 57),
    (-151, 59), (-157, 63), (-164, 64), (-170, 68)
])

# Alaska is separated by the Bering Strait in this coarse outline.
poly([
    (-170, 68), (-166, 65), (-162, 62), (-155, 59), (-150, 58),
    (-145, 60), (-140, 63), (-136, 67), (-141, 70), (-150, 71),
    (-160, 70)
])

# The narrow isthmus of Central America.
poly([
    (-92, 18), (-88, 16), (-84, 11), (-81, 9), (-78, 8),
    (-77, 5), (-81, 5), (-85, 8), (-88, 11), (-92, 14)
])

# Greenland, whose ice sheet is land under the requested convention.
poly([
    (-74, 60), (-64, 59), (-51, 60), (-42, 64), (-35, 70),
    (-24, 76), (-20, 82), (-27, 86), (-42, 88), (-57, 86),
    (-68, 81), (-73, 73), (-75, 66)
])

# South America.
poly([
    (-81, 12), (-76, 9), (-71, 11), (-66, 10), (-61, 8),
    (-55, 6), (-50, 2), (-45, -2), (-40, -8), (-36, -14),
    (-39, -21), (-43, -25), (-47, -30), (-52, -34), (-58, -38),
    (-64, -42), (-68, -48), (-73, -54), (-70, -56), (-64, -53),
    (-60, -48), (-57, -42), (-53, -36), (-50, -29), (-47, -23),
    (-45, -16), (-48, -10), (-54, -6), (-61, -4), (-68, -1),
    (-74, 2), (-79, 6)
])

# Africa, from the Mediterranean coast around the Cape and back up the Red Sea.
poly([
    (-17, 35), (-8, 37), (1, 37), (10, 36), (16, 33), (25, 32),
    (32, 31), (36, 28), (43, 13), (48, 12), (51, 8), (48, 2),
    (44, -4), (42, -12), (38, -20), (35, -27), (32, -34),
    (25, -35), (18, -34), (12, -34), (5, -35), (-1, -31),
    (-6, -25), (-10, -17), (-14, -7), (-16, 2), (-17, 10),
    (-13, 17), (-17, 25)
])
# Europe and Asia as one broad connected mainland.
poly([
    (-10, 36), (-9, 43), (-5, 48), (-9, 54), (-5, 58),
    (2, 58), (6, 62), (11, 65), (16, 70), (27, 71), (38, 73),
    (52, 76), (70, 78), (94, 79), (118, 77), (140, 73),
    (160, 70), (180, 67), (180, 55), (171, 54), (165, 50),
    (156, 48), (149, 44), (143, 39), (138, 35), (132, 32),
    (127, 30), (123, 33), (119, 38), (114, 39), (109, 35),
    (105, 30), (101, 25), (97, 23), (92, 27), (87, 28),
    (82, 25), (78, 22), (74, 21), (70, 24), (65, 25),
    (60, 27), (54, 28), (49, 31), (45, 35), (40, 38),
    (35, 39), (30, 40), (25, 42), (20, 40), (15, 38),
    (10, 37), (4, 36), (-2, 36)
])

# The Arabian peninsula.
poly([
    (35, 32), (42, 31), (48, 29), (55, 25), (57, 19),
    (55, 14), (51, 12), (46, 12), (43, 16), (39, 20),
    (37, 24), (34, 27)
])

# Indian subcontinent and the mainland of southeast Asia.
poly([
    (68, 24), (73, 27), (78, 30), (83, 29), (88, 27),
    (92, 24), (96, 21), (98, 17), (96, 12), (92, 9),
    (87, 9), (83, 14), (79, 18), (74, 20), (70, 20)
])
poly([
    (97, 23), (102, 26), (106, 23), (109, 19), (108, 14),
    (104, 10), (101, 7), (98, 9), (96, 14), (94, 19)
])

# Australia and Tasmania.
poly([
    (113, -11), (122, -12), (132, -12), (141, -14), (151, -19),
    (153, -25), (150, -31), (145, -36), (137, -38), (128, -36),
    (121, -34), (115, -31), (113, -25), (110, -20)
])
island(146, -42, 2.0, 1.3)

# Antarctica: ice-covered continent, with a deliberately approximate coast.
poly([
    (-180, -74), (-165, -75), (-150, -73), (-135, -72), (-120, -74),
    (-105, -76), (-90, -75), (-78, -71), (-66, -67), (-55, -64),
    (-43, -65), (-30, -69), (-15, -71), (0, -70), (15, -72),
    (30, -68), (45, -68), (60, -70), (75, -68), (90, -67),
    (108, -65), (125, -66), (140, -68), (155, -70), (170, -68),
    (180, -68), (180, -90), (-180, -90)
])
# Arctic and North Atlantic islands.
island(-42, 73, 3.0, 5.0)       # Greenland's western/northern bulk supplement
island(-62, 67, 4.0, 6.0)        # Baffin Island
island(-82, 74, 4.0, 5.0)        # Ellesmere/Axel Heiberg region
island(-18, 65, 2.5, 1.3)        # Iceland
island(-3, 54, 1.0, 5.5)          # Great Britain
island(-8, 53, 1.5, 3.0)          # Ireland
island(16, 78, 4.0, 2.0)          # Svalbard
island(52, 73, 2.5, 5.0)          # Novaya Zemlya

# Mediterranean islands.
island(9, 40, 1.7, 3.5)           # Sardinia
island(13, 38, 1.8, 1.2)          # Sicily
island(8, 42, 1.0, 1.0)           # Corsica
island(25, 35, 2.0, 1.0)          # Cyprus
island(24, 36, 3.0, 1.0)          # Crete
island(3, 39, 2.0, 1.0)           # Balearics

# East Asian island arcs.
island(141, 39, 1.4, 5.0)         # Honshu
island(135, 34, 1.2, 2.2)         # Shikoku
island(131, 33, 1.7, 2.0)         # Kyushu
island(142, 45, 1.5, 3.0)         # Hokkaido
island(121, 23.5, 1.5, 2.2)       # Taiwan
island(125, 7, 2.0, 4.5)          # Philippines (main cluster)
island(123, 13, 2.0, 3.0)
island(119, 18, 1.0, 2.0)

# Indonesia and New Guinea.
island(101, 0, 3.0, 10.0)         # Sumatra
island(110, -2, 5.0, 4.0)         # Borneo
island(113, -7, 5.0, 1.3)         # Java
island(120, -2, 3.0, 5.0)         # Sulawesi
island(141, -5, 10.0, 5.0)        # New Guinea
island(147, -9, 3.0, 2.0)

# Indian Ocean and Pacific islands.
island(81, 7, 1.4, 3.0)            # Sri Lanka
island(48, -19, 3.0, 9.0)         # Madagascar
island(174, -41, 2.0, 6.0)        # New Zealand North/South
island(168, -44, 1.8, 4.0)
island(134, -5, 1.0, 2.0)         # Halmahera
island(159, -8, 2.0, 3.0)         # Solomon group
island(171, -13, 1.2, 2.5)        # Vanuatu/Fiji vicinity
# Caribbean, Central American and nearby islands.
island(-74, 20, 5.0, 3.0)         # Cuba
island(-70, 19, 3.0, 2.0)         # Hispaniola
island(-66, 18, 1.5, 1.3)         # Puerto Rico
island(-77, 18, 1.0, 1.0)         # Jamaica
island(-61, 15, 1.0, 2.0)         # Lesser Antilles
island(-63, 12, 0.8, 1.2)
island(-60, 10, 0.8, 1.0)
island(-80, 9, 1.0, 1.0)          # Panama offshore
island(-87, 13, 1.0, 2.0)         # Gulf of Honduras islands

# Pacific and Atlantic islands.
island(-90, 0, 1.5, 1.0)           # Galapagos
island(-157, 21, 2.0, 1.0)         # Hawaii
island(-155, 20, 1.0, 1.0)
island(-25, 65, 0.8, 0.8)          # Faroes vicinity
island(-29, 39, 1.0, 0.8)          # Azores vicinity
island(-17, 28, 1.5, 1.2)          # Canaries
island(-5, -16, 1.0, 0.8)          # Ascension/South Atlantic vicinity

# Large Arctic Canadian islands and the Aleutian chain.
island(-95, 69, 4.0, 7.0)
island(-105, 73, 4.0, 5.0)
island(-88, 76, 3.0, 4.0)
island(-177, 52, 2.0, 0.7)
island(-169, 52, 2.0, 0.7)
island(-160, 53, 1.5, 0.7)

# A few major European and Asian islands omitted by the mainland outlines.
island(18, 58, 2.5, 4.0)          # Sweden's Baltic island region
island(35, 65, 2.0, 5.0)          # White Sea islands
island(143, 52, 1.5, 4.0)         # Sakhalin
island(153, 45, 1.0, 1.5)         # Kurils
# Inland seas and lakes, plus bays accidentally enclosed by broad continental
# outlines.  They are water under the experiment's classification.
water_poly([
    (-96, 64), (-88, 63), (-80, 60), (-78, 55), (-84, 52),
    (-94, 53), (-101, 57), (-101, 62)
])                                      # Hudson Bay
water_poly([
    (-98, 30), (-91, 28), (-84, 26), (-80, 25), (-82, 22),
    (-91, 23), (-99, 26)
])                                      # Gulf of Mexico
water_poly([
    (-177, 66), (-168, 62), (-160, 58), (-166, 54), (-176, 55),
    (-180, 60)
])                                      # Bering Sea
water_poly([
    (-7, 38), (2, 36), (12, 34), (22, 33), (30, 34), (36, 37),
    (31, 40), (22, 40), (12, 39), (3, 39), (-4, 40)
])                                      # Mediterranean
water_poly([
    (27, 41), (34, 41), (42, 42), (42, 46), (35, 47), (29, 45)
])                                      # Black Sea
lake(51, 42, 4.0, 5.5)                  # Caspian Sea
lake(60, 45, 3.0, 2.0)                  # Aral Sea
lake(108, 53, 1.8, 3.0)                 # Lake Baikal
lake(31, 60, 1.5, 2.2)                  # Lake Ladoga
lake(-91, 51, 4.0, 3.0)                 # Lake Winnipeg
lake(-84, 45, 2.5, 1.2)                 # Great Lakes, broad group
lake(-112, 41, 1.7, 1.0)                # Great Salt Lake
lake(-71, -15, 2.0, 1.0)                # Lake Titicaca
lake(33, -1, 4.0, 3.0)                  # Lake Victoria
lake(29, -5, 1.0, 5.0)                  # Lake Tanganyika
lake(34, -13, 1.0, 3.0)                 # Lake Malawi
lake(35, 31, 0.5, 0.5)                  # Dead Sea
lake(143, -29, 2.0, 1.2)                # Lake Eyre

# Major narrow seas/gulfs represented explicitly where the mainland polygon
# would otherwise make them land.
water_poly([
    (33, 29), (36, 25), (39, 20), (42, 15), (44, 12),
    (40, 12), (37, 18), (34, 24)
])                                      # Red Sea
water_poly([
    (48, 30), (53, 29), (57, 26), (55, 23), (50, 24), (47, 27)
])                                      # Persian Gulf
# Generate the prescribed 2-degree sample grid and deliver all artifacts.
def main():
    lats = list(range(89, -90, -2))
    lons = list(range(-179, 180, 2))
    matrix = np.zeros((90, 180), dtype=np.uint8)
    for row, lat in enumerate(lats):
        for col, lon in enumerate(lons):
            matrix[row, col] = 1 if land_at(lon, lat) else 0

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "candidate.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # 1 is white land and 0 is black water.
    pixels = (matrix * 255).astype(np.uint8)
    raw = Image.fromarray(pixels, mode="L")
    raw.save(os.path.join(OUT, "candidate.png"))
    large = raw.resize((1440, 720), resample=Image.Resampling.NEAREST)
    large.save(os.path.join(OUT, "candidate_8x.png"))

    assert matrix.shape == (90, 180)
    assert int(matrix.sum()) + int((1 - matrix).sum()) == 16200
    assert set(np.unique(matrix).tolist()) <= {0, 1}
    assert raw.size == (180, 90) and large.size == (1440, 720)


if __name__ == "__main__":
    main()
