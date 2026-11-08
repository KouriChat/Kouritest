import csv
import os
import numpy as np
from PIL import Image

POLYS = []
ELLIPSES = []

def poly(points):
    POLYS.append(points)

def ellipse(cx, cy, rx, ry):
    ELLIPSES.append((cx, cy, rx, ry))

def inside(x, y, p):
    c = False
    j = len(p) - 1
    for i in range(len(p)):
        xi, yi = p[i]
        xj, yj = p[j]
        if ((yi > y) != (yj > y)):
            xx = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < xx:
                c = not c
        j = i
    return c

# North America, including the Canadian Arctic mainland.
poly([
    (-168, 66), (-160, 61), (-150, 60), (-143, 55), (-136, 54),
    (-130, 49), (-124, 48), (-121, 42), (-116, 32), (-108, 25),
    (-99, 19), (-91, 18), (-84, 24), (-80, 29), (-78, 35),
    (-74, 40), (-67, 44), (-59, 49), (-57, 56), (-66, 62),
    (-78, 66), (-92, 70), (-108, 73), (-126, 73), (-145, 72),
    (-160, 70)
])
# Alaska's western peninsula and the large islands are separate below.
poly([(-180, 70), (-168, 72), (-150, 70), (-140, 65), (-132, 60),
      (-145, 56), (-160, 58), (-174, 62)])
poly([(-117, 32), (-114, 24), (-110, 18), (-103, 15), (-97, 16),
      (-91, 19), (-99, 25), (-108, 30)])
# Central America.
poly([(-92, 18), (-87, 18), (-83, 13), (-78, 10), (-77, 7),
      (-82, 8), (-87, 12), (-92, 14)])
poly([(-117, 33), (-112, 28), (-106, 25), (-103, 20), (-97, 18),
      (-91, 18), (-98, 25), (-108, 31)])

# South America.
poly([
    (-81, 10), (-76, 12), (-70, 11), (-62, 10), (-53, 6),
    (-45, 2), (-36, -5), (-35, -14), (-40, -24), (-47, -31),
    (-50, -40), (-56, -51), (-66, -55), (-73, -46), (-76, -31),
    (-73, -20), (-77, -9), (-81, 0)
])

# Africa.
poly([
    (-17, 37), (-5, 36), (8, 37), (20, 34), (31, 31), (36, 24),
    (43, 13), (50, 11), (49, 1), (43, -10), (37, -21),
    (31, -34), (20, -35), (9, -34), (1, -30), (-7, -25),
    (-13, -17), (-17, -5), (-16, 10), (-10, 21)
])

# Europe west of the main Russian plain.
poly([
    (-10, 36), (-5, 43), (3, 44), (8, 48), (15, 45), (20, 41),
    (28, 41), (34, 45), (31, 51), (36, 56), (30, 60), (25, 66),
    (15, 70), (5, 67), (-3, 60), (-9, 57), (-6, 51), (-2, 47),
    (-10, 44)
])
poly([(-10, 36), (-5, 43), (3, 44), (8, 48), (4, 52),
      (-5, 51), (-10, 45)])

# Asia, with the Siberian and Chinese landmass.
poly([
    (25, 40), (38, 41), (50, 40), (61, 37), (70, 34), (76, 29),
    (79, 21), (87, 20), (95, 17), (101, 9), (108, 7), (115, 14),
    (122, 9), (130, 11), (137, 25), (146, 35), (159, 45),
    (173, 51), (179, 54), (179, 70), (163, 71), (145, 70),
    (125, 74), (105, 76), (84, 75), (62, 74), (43, 70),
    (31, 65), (25, 57), (25, 49)
])

# Arabia and the Indian subcontinent.
poly([(34, 31), (44, 30), (54, 26), (58, 20), (55, 13),
      (49, 12), (43, 16), (38, 22)])
poly([(68, 35), (78, 34), (88, 28), (91, 20), (87, 8),
      (81, 7), (74, 15), (70, 24)])

# Australia and Tasmania.
poly([(113, -11), (126, -11), (136, -14), (145, -19), (153, -26),
      (153, -34), (146, -39), (136, -38), (126, -35), (116, -32),
      (112, -24), (113, -17)])
poly([(144, -40), (151, -40), (148, -44), (143, -44), (140, -42)])

# Antarctica: a deliberately coarse ice-covered land outline.
poly([
    (-180, -69), (-160, -71), (-140, -73), (-120, -71), (-100, -73),
    (-80, -70), (-60, -72), (-40, -68), (-20, -72), (0, -70),
    (20, -71), (40, -68), (60, -70), (80, -68), (100, -70),
    (120, -68), (140, -70), (160, -68), (180, -70),
    (180, -89), (-180, -89)
])
# Greenland and the principal Arctic islands.
poly([(-73, 59), (-62, 60), (-48, 59), (-35, 65), (-24, 72),
      (-22, 80), (-30, 84), (-45, 83), (-58, 80), (-70, 74)])
poly([(-110, 70), (-92, 71), (-83, 77), (-88, 82), (-103, 82),
      (-116, 77)])
poly([(-80, 68), (-67, 68), (-62, 73), (-69, 77), (-80, 76)])
poly([(-105, 77), (-92, 76), (-87, 80), (-94, 83), (-106, 82)])
poly([(-86, 67), (-78, 65), (-75, 70), (-79, 73), (-87, 72)])

# Atlantic and European islands.
poly([(-25, 63), (-17, 63), (-14, 66), (-18, 67), (-24, 66)])  # Iceland
poly([(-8, 58), (-5, 51), (0, 50), (2, 54), (-1, 58)])          # Britain
poly([(-11, 56), (-6, 54), (-6, 51), (-10, 51), (-11, 53)])     # Ireland
poly([(7, 44), (13, 44), (16, 40), (12, 37), (8, 39)])           # Italy
poly([(19, 40), (26, 40), (28, 36), (24, 34), (20, 36)])         # Greece
poly([(35, 36), (36, 41), (42, 42), (44, 39), (41, 35)])         # Anatolia edge
poly([(12, 38), (15, 38), (15, 36), (12, 36)])                   # Sicily
poly([(13, 41), (16, 41), (16, 39), (13, 39)])                   # Sardinia/Corsica
poly([(28, 36), (34, 36), (34, 31), (29, 31)])                  # Cyprus

# Japan, Sakhalin, Kamchatka, and East Asian island groups.
poly([(130, 32), (134, 34), (136, 38), (141, 41), (145, 44),
      (142, 35), (137, 31), (132, 30)])
poly([(141, 46), (146, 50), (145, 57), (142, 55), (140, 50)])
poly([(143, 61), (151, 60), (160, 55), (168, 54), (169, 58),
      (160, 62), (151, 64)])
poly([(120, 19), (124, 18), (126, 14), (122, 11), (119, 14)])    # Luzon
poly([(122, 12), (127, 10), (126, 6), (121, 5), (119, 9)])       # Visayas
poly([(120, 6), (125, 5), (124, 0), (118, 0), (117, 4)])         # Mindanao

# Southeast Asian peninsula and islands.
poly([(95, 21), (102, 20), (105, 14), (103, 8), (99, 7),
      (97, 12), (98, 17)])
poly([(103, 3), (108, 7), (113, 5), (116, 1), (113, -3),
      (108, -2), (104, 0)])
poly([(95, 5), (101, 5), (104, 0), (101, -5), (97, -4)])          # Sumatra
poly([(105, -6), (114, -6), (119, -9), (116, -12), (108, -10)])  # Java
poly([(113, -1), (120, 3), (124, 0), (122, -5), (117, -4)])      # Borneo
poly([(120, -5), (128, -5), (132, -9), (127, -11), (122, -9)])   # Sulawesi

# Indian Ocean and western Pacific islands.
poly([(43, -12), (50, -13), (50, -22), (46, -26), (43, -20)])    # Madagascar
poly([(79, 10), (82, 10), (82, 6), (80, 5), (79, 7)])            # Sri Lanka
poly([(166, -34), (179, -36), (178, -47), (170, -47), (166, -42)])
poly([(174, -35), (179, -36), (179, -41), (175, -41)])            # North Island

# Caribbean islands, using intentionally broad knowledge-based blobs.
ellipse(-75, 20, 7, 2)
ellipse(-70, 18, 4, 2)
ellipse(-66, 18, 2, 1)
poly([(-86, 22), (-80, 23), (-77, 20), (-80, 18), (-85, 19)])
poly([(-78, 18), (-74, 18), (-73, 15), (-77, 14), (-80, 16)])

# Pacific island groups large enough to occur on the two-degree sampling grid.
ellipse(158, 6, 5, 3)
ellipse(147, -6, 4, 3)
ellipse(134, 7, 2, 2)

# A few isolated high-latitude islands.
ellipse(69, 78, 6, 4)
ellipse(95, 78, 6, 4)
ellipse(55, 74, 4, 3)
WATER_POLYS = []
WATER_ELLIPSES = []

def water_poly(points):
    WATER_POLYS.append(points)

def water_ellipse(cx, cy, rx, ry):
    WATER_ELLIPSES.append((cx, cy, rx, ry))

# Inland waters are explicitly removed from otherwise coarse continental blobs.
water_ellipse(45, 42, 5, 2)       # Caspian Sea
water_ellipse(51, 37, 3, 2)       # south Caspian extension
water_ellipse(59, 46, 2, 1)       # Aral Sea
water_ellipse(34, 0, 3, 2)        # Lake Victoria
water_ellipse(29, -3, 2, 4)       # Lake Tanganyika
water_ellipse(31, 7, 2, 2)        # Lake Chad
water_ellipse(108, 31, 3, 2)      # Poyang / central China lakes
water_ellipse(29, 61, 3, 2)       # Lake Ladoga
water_ellipse(61, 54, 2, 2)       # broad Russian lake area
water_ellipse(-85, 46, 5, 2)      # Great Lakes, western group
water_ellipse(-80, 44, 3, 2)
water_ellipse(-77, 45, 2, 1)
water_ellipse(-82, 42, 2, 1)
water_ellipse(-97, 56, 3, 2)      # Lake Winnipeg
water_ellipse(-114, 51, 2, 2)     # Alberta lakes
water_ellipse(-111, 45, 2, 1)     # Great Salt Lake
water_ellipse(-71, -41, 2, 3)     # Patagonian fjord/lake approximation
water_ellipse(84, 30, 2, 2)       # Himalayan high lake
water_ellipse(117, 30, 2, 1)

# The Mediterranean and connected marginal seas, kept as water despite
# intentionally simplified European and Asian coast polygons.
water_poly([(-6, 36), (4, 36), (13, 35), (22, 35), (29, 36),
            (34, 33), (35, 30), (27, 30), (18, 31), (8, 32),
            (0, 33), (-6, 34)])
water_poly([(27, 30), (37, 30), (43, 28), (49, 25), (50, 20),
            (43, 20), (36, 23), (31, 27)])
water_poly([(22, 41), (29, 42), (34, 39), (35, 36), (28, 36)])

# Persian Gulf, Red Sea, and Gulf of Aden.
water_poly([(48, 30), (56, 29), (57, 25), (52, 24), (48, 26)])
water_poly([(32, 30), (38, 28), (40, 20), (43, 13), (40, 12),
            (36, 20), (34, 26)])
water_poly([(42, 14), (52, 13), (55, 10), (48, 10), (43, 11)])

# Large simple lakes in North America.
water_poly([(-95, 50), (-88, 49), (-84, 46), (-89, 44), (-96, 46)])
water_poly([(-84, 47), (-79, 46), (-76, 44), (-80, 42), (-84, 44)])

def is_water_override(x, y):
    for p in WATER_POLYS:
        if inside(x, y, p):
            return True
    for cx, cy, rx, ry in WATER_ELLIPSES:
        if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1:
            return True
    return False

def is_land(x, y):
    hit = False
    for p in POLYS:
        if inside(x, y, p):
            hit = True
            break
    if not hit:
        for cx, cy, rx, ry in ELLIPSES:
            if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1:
                hit = True
                break
    return hit and not is_water_override(x, y)

def generate():
    rows = []
    for lat in range(89, -90, -2):
        rows.append([1 if is_land(lon, lat) else 0
                     for lon in range(-179, 180, 2)])
    return rows
def write_outputs(rows):
    os.makedirs("outputs", exist_ok=True)
    if len(rows) != 90 or any(len(row) != 180 for row in rows):
        raise ValueError("matrix dimensions are not 90 by 180")
    if any(v not in (0, 1) for row in rows for v in row):
        raise ValueError("matrix contains a value other than 0 or 1")

    with open("outputs/candidate.csv", "w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(rows)

    # 1 is white land and 0 is black water.  No interpolation is used.
    raw = np.asarray(rows, dtype=np.uint8) * 255
    image = Image.fromarray(raw, mode="L")
    image.save("outputs/candidate.png")
    image.resize((1440, 720), Image.Resampling.NEAREST).save(
        "outputs/candidate_8x.png"
    )

if __name__ == "__main__":
    write_outputs(generate())
