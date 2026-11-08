"""Knowledge-based coarse reconstruction of world land and water.
No geographic data is read: the polygons below are hand-written approximations.
"""
import csv
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

# A deliberately simple ray-casting test is adequate at the requested 2-degree
# sampling.  Longitudes are kept in the ordinary -180..180 convention.
def inside(x, y, ring):
    hit = False
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi) + xi
        ):
            hit = not hit
        j = i
    return hit


land = []
def poly(*points):
    land.append(points)

# Africa, including the broad continental ice-free outline.
poly((-17, 15), (-16, 22), (-8, 29), (-4, 35), (5, 37), (15, 35),
     (25, 33), (32, 31), (36, 25), (43, 13), (50, 12), (50, 3),
     (43, -2), (42, -12), (36, -20), (32, -29), (25, -35), (15, -35),
     (7, -34), (-1, -29), (-9, -18), (-15, -6), (-17, 4))
# Europe and the main Asian landmass.
poly((-11, 36), (-9, 43), (-5, 43), (-10, 48), (-5, 51), (2, 51),
     (5, 55), (9, 55), (9, 58), (15, 57), (19, 54), (23, 54),
     (28, 58), (35, 58), (41, 55), (43, 50), (40, 45), (34, 43),
     (27, 40), (22, 38), (14, 37), (9, 39), (2, 38), (-5, 36))
poly((-5, 45), (2, 50), (15, 55), (25, 62), (38, 65), (48, 70),
     (63, 72), (78, 75), (95, 75), (112, 73), (127, 70), (140, 72),
     (151, 70), (165, 72), (180, 70), (180, 55), (169, 54),
     (158, 51), (150, 47), (143, 40), (137, 33), (130, 30),
     (125, 19), (120, 6), (113, 2), (105, 5), (98, 8), (91, 11),
     (84, 8), (79, 14), (73, 20), (66, 22), (60, 25), (53, 25),
     (46, 29), (40, 30), (34, 32), (29, 35), (22, 38), (14, 37),
     (8, 39), (0, 38), (-5, 45))
# Arabian Peninsula and the Indian subcontinent are made explicit so that
# their distinctive southward projections survive the coarse grid.
poly((35, 31), (43, 30), (52, 27), (59, 23), (56, 16), (50, 12),
     (43, 13), (40, 20), (35, 23))
poly((67, 24), (73, 22), (78, 17), (82, 8), (88, 21), (87, 27),
     (80, 30), (73, 29))

# North America (the Alaska extension is separate to avoid a dateline ring).
poly((-168, 70), (-158, 66), (-150, 61), (-143, 57), (-135, 56),
     (-130, 51), (-124, 49), (-122, 42), (-117, 32), (-110, 28),
     (-101, 25), (-92, 19), (-85, 18), (-82, 25), (-79, 29),
     (-75, 35), (-70, 42), (-64, 46), (-58, 51), (-61, 56),
     (-70, 60), (-80, 65), (-91, 70), (-104, 72), (-116, 75),
     (-132, 75), (-148, 75), (-161, 73))
poly((-170, 52), (-161, 54), (-151, 59), (-141, 60), (-130, 55),
     (-130, 64), (-140, 70), (-154, 72), (-168, 68))

# South America.
poly((-80, 10), (-75, 5), (-77, -5), (-75, -15), (-72, -26),
     (-68, -40), (-64, -52), (-55, -55), (-45, -50), (-39, -40),
     (-35, -28), (-34, -15), (-36, -5), (-45, 4), (-55, 9),
     (-66, 12), (-73, 12))

# Antarctica: the northern ice-covered coast is land, while the ocean and
# floating sea ice north of this hand-drawn boundary remain water.
poly((-180, -68), (-165, -69), (-150, -72), (-135, -73), (-120, -74),
     (-105, -74), (-90, -73), (-78, -72), (-68, -69), (-62, -65),
     (-58, -61), (-64, -64), (-70, -70), (-55, -72), (-40, -72),
     (-25, -71), (-10, -70), (5, -69), (20, -68), (40, -68),
     (60, -67), (80, -66), (100, -66), (120, -66), (140, -68),
     (160, -70), (180, -68), (180, -90), (-180, -90))
 # Greenland and the larger North Atlantic islands.
poly((-73, 60), (-61, 59), (-47, 60), (-31, 64), (-23, 70),
     (-21, 77), (-28, 82), (-40, 84), (-55, 83), (-65, 78),
     (-71, 70))
poly((-25, 63), (-20, 64), (-14, 66), (-14, 69), (-20, 67))
poly((-11, 50), (-6, 51), (1, 52), (2, 55), (-3, 59), (-6, 58),
     (-8, 54))
poly((-10, 51), (-6, 52), (-6, 55), (-8, 55))
poly((4, 59), (8, 58), (9, 62), (7, 68), (4, 70), (3, 64))
poly((10, 76), (18, 76), (26, 79), (30, 82), (20, 81), (12, 79))

# Australia and nearby islands.
poly((113, -35), (114, -25), (117, -20), (121, -18), (126, -14),
     (132, -12), (138, -12), (144, -14), (151, -20), (154, -27),
     (153, -35), (147, -39), (137, -39), (126, -37))
poly((144, -41), (149, -42), (148, -47), (143, -44))
poly((166, -34), (178, -37), (178, -46), (171, -47), (166, -42))
poly((172, -36), (176, -37), (176, -41), (172, -40))
poly((141, -2), (148, -2), (153, -6), (151, -10), (145, -9))

# Madagascar and the Indian Ocean islands large enough for this sampling.
poly((43, -12), (50, -13), (50, -22), (47, -26), (44, -25), (43, -18))
poly((79, 10), (82, 9), (82, 6), (80, 5), (79, 7))
poly((72, -7), (78, -6), (82, -1), (96, 5), (103, 2), (105, -5),
     (100, -9), (94, -9), (88, -8), (82, -7), (77, -9))
poly((105, -6), (114, -7), (114, -9), (106, -9))
poly((114, -8), (123, -8), (124, -5), (118, -4))
poly((120, -5), (127, -5), (131, -1), (130, 3), (124, 1))
poly((130, -4), (138, -4), (145, -7), (148, -3), (146, 0),
     (140, 0), (136, -2))
poly((140, -2), (147, -1), (151, -4), (147, -7), (140, -5))

# Southeast Asian peninsulas and islands.
poly((95, 20), (101, 18), (105, 14), (104, 8), (101, 2), (98, 1),
     (96, 7), (98, 13))
poly((105, 21), (110, 21), (114, 17), (117, 12), (115, 7), (111, 7),
     (108, 12), (104, 16))
poly((114, 7), (120, 8), (125, 6), (127, 2), (124, 0), (119, 2))
poly((125, 18), (130, 20), (135, 18), (140, 14), (140, 9),
     (135, 6), (130, 8), (127, 13))
poly((126, 7), (130, 6), (130, 3), (126, 3))
poly((121, 19), (124, 19), (124, 13), (120, 13), (118, 16))
poly((117, 22), (120, 22), (121, 18), (118, 17))
poly((105, 10), (108, 10), (109, 6), (106, 5))
 # East Asian and western Pacific islands.
poly((130, 31), (135, 33), (141, 39), (145, 43), (145, 35),
     (141, 31), (136, 30))
poly((139, 45), (146, 46), (146, 42), (142, 40))
poly((123, 25), (123, 22), (120, 22), (119, 25), (121, 26))
poly((128, 36), (130, 36), (129, 34), (127, 34))
poly((121, 25), (123, 25), (122, 21), (120, 21))
poly((145, 13), (146, 14), (147, 13), (146, 12))
poly((155, 49), (157, 51), (159, 51), (159, 49))
poly((167, 53), (170, 54), (170, 52), (168, 52))

# Large islands of the Mediterranean and the Middle East.
poly((26, 36), (34, 36), (35, 34), (30, 34), (26, 35))
poly((32, 35), (34, 36), (34, 38), (32, 38))
poly((8, 44), (13, 44), (17, 42), (18, 39), (14, 38), (10, 39))
poly((7, 40), (10, 40), (10, 37), (8, 37))
poly((13, 38), (16, 38), (16, 36), (13, 36))
poly((35, 33), (36, 35), (39, 35), (39, 33), (37, 32))
poly((47, 40), (51, 42), (51, 39), (48, 38))

# Caribbean and Central American stepping stones.
poly((-92, 18), (-88, 22), (-83, 23), (-78, 21), (-76, 19),
     (-80, 17), (-86, 17))
poly((-74, 20), (-69, 21), (-68, 18), (-73, 17), (-76, 19))
poly((-86, 16), (-83, 16), (-82, 13), (-85, 12))
poly((-78, 18), (-75, 19), (-73, 17), (-76, 16))
poly((-79, 9), (-77, 9), (-77, 7), (-79, 7))
poly((-88, 16), (-86, 16), (-86, 14), (-88, 14))
poly((-67, 18), (-65, 19), (-64, 18), (-65, 17))
# Isthmus is intentionally broad at this resolution.
poly((-92, 18), (-88, 17), (-86, 15), (-84, 12), (-82, 10),
     (-79, 9), (-77, 8), (-78, 13), (-82, 17), (-86, 20))

# A few sub-Arctic and North Pacific islands.
poly((-55, 52), (-50, 54), (-46, 53), (-48, 50), (-53, 49))
poly((-74, 63), (-68, 64), (-65, 62), (-69, 60), (-74, 61))
poly((-167, 54), (-162, 55), (-160, 53), (-164, 52))
poly((-156, 20), (-154, 22), (-156, 23), (-158, 21))
poly((-161, 21), (-159, 22), (-160, 20))

# South Atlantic islands and the southern tip islands.
poly((-62, -52), (-57, -51), (-57, -53), (-61, -54))
poly((-69, -54), (-65, -54), (-65, -56), (-69, -56))
poly((-47, -24), (-44, -23), (-44, -25), (-47, -26))
 # Water holes override broad continental rings for the principal inland seas and
# lakes.  They are still hand-written rules, not a lookup or a downloaded map.
water = []
def lake(*points):
    water.append(points)

lake((46, 36), (54, 37), (54, 43), (51, 47), (47, 46), (46, 41))
lake((57, 43), (61, 43), (61, 46), (58, 46))
lake((27, 41), (36, 41), (42, 44), (40, 47), (31, 47), (27, 45))
lake((32, 28), (35, 26), (40, 23), (43, 16), (41, 12),
     (37, 16), (35, 22))
lake((48, 24), (56, 25), (58, 28), (54, 30), (49, 29))
lake((-98, 18), (-90, 18), (-84, 22), (-82, 27), (-87, 30),
     (-94, 29), (-98, 25))
lake((-96, 51), (-78, 51), (-76, 59), (-82, 64), (-92, 63),
     (-98, 58))
lake((-82, 43), (-76, 43), (-76, 47), (-80, 49), (-84, 47))
lake((-90, 46), (-83, 46), (-83, 49), (-89, 49))
lake((-94, 45), (-89, 45), (-89, 49), (-94, 49))
lake((-86, 41), (-81, 41), (-79, 43), (-84, 44))
lake((29, -2), (34, -2), (34, -5), (31, -5))
lake((32, -9), (35, -9), (35, -14), (32, -14))
lake((33, -15), (35, -15), (35, -18), (33, -18))
lake((105, 49), (110, 49), (110, 54), (105, 54))
lake((88, 44), (93, 44), (93, 47), (88, 47))

def is_land(lon, lat):
    value = any(inside(lon, lat, ring) for ring in land)
    if value and any(inside(lon, lat, ring) for ring in water):
        return 0
    return int(value)


def main():
    # The order and coordinates here exactly implement the requested sample:
    # latitude 89 down to -89, longitude -179 up to 179, both odd integers.
    lats = list(range(89, -90, -2))
    lons = list(range(-179, 180, 2))
    matrix = np.array(
        [[is_land(lon, lat) for lon in lons] for lat in lats], dtype=np.uint8
    )
    assert matrix.shape == (90, 180)
    assert int(matrix.sum()) + int((1 - matrix).sum()) == 16200

    with (OUT / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # L is exact black/white: 0 is water and 255 is land.
    base = Image.fromarray(matrix * 255, mode="L")
    assert base.size == (180, 90)
    colors = set(base.getdata())
    assert colors <= {0, 255}
    base.save(OUT / "candidate.png")
    enlarged = base.resize((1440, 720), resample=Image.Resampling.NEAREST)
    enlarged.save(OUT / "candidate_8x.png")
    assert enlarged.size == (1440, 720)


if __name__ == "__main__":
    main()
