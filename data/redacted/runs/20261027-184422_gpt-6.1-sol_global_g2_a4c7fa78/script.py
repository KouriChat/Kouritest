"""Knowledge-based coarse world land mask on the requested 2 degree grid."""
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image


def inside(x, y, poly):
    """Ray casting; polygons are hand-written geographic approximations."""
    hit = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi) + xi
        ):
            hit = not hit
        j = i
    return hit


def ellipse(x, y, cx, cy, rx, ry):
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


def any_poly(x, y, polygons):
    return any(inside(x, y, p) for p in polygons)


# Mainland outlines.  These are intentionally broad, low-resolution outlines,
# not a lookup of coastlines.
POLYS = [
    # Africa
    [(-17, 15), (-14, 10), (-17, 5), (-12, 0), (-10, -6),
     (-4, -11), (2, -16), (9, -18), (13, -28), (18, -35),
     (27, -35), (33, -30), (35, -22), (40, -16), (43, -7),
     (50, 0), (45, 10), (41, 13), (39, 19), (34, 23),
     (30, 31), (20, 33), (12, 32), (5, 37), (-5, 35),
     (-10, 30), (-17, 25)],
    # Europe, including the Scandinavian peninsula
    [(-10, 36), (-9, 43), (-5, 44), (-2, 48), (3, 51),
     (7, 54), (9, 58), (6, 60), (8, 63), (14, 66),
     (19, 69), (26, 70), (31, 67), (34, 62), (35, 57),
     (41, 53), (40, 48), (35, 45), (30, 45), (27, 42),
     (22, 40), (17, 38), (12, 37), (9, 43), (4, 44),
     (2, 40), (-2, 38)],
    # The continuous northern part of Asia
    [(25, 40), (33, 36), (42, 32), (50, 27), (57, 27),
     (63, 30), (68, 27), (74, 25), (80, 25), (86, 27),
     (92, 24), (98, 19), (104, 17), (110, 18), (116, 22),
     (122, 24), (128, 29), (134, 35), (140, 39), (147, 44),
     (155, 48), (164, 51), (173, 55), (180, 55), (180, 80),
     (165, 79), (150, 78), (135, 77), (120, 77), (105, 78),
     (90, 78), (75, 77), (60, 75), (45, 73), (34, 69),
     (27, 63), (21, 58), (25, 53), (29, 48), (25, 44)],
    # North America
    [(-168, 52), (-160, 58), (-150, 60), (-141, 67), (-132, 70),
     (-122, 72), (-112, 70), (-103, 72), (-94, 70), (-84, 74),
     (-70, 76), (-58, 73), (-53, 68), (-59, 63), (-68, 59),
     (-65, 55), (-75, 50), (-80, 45), (-77, 40), (-82, 33),
     (-91, 30), (-99, 26), (-105, 21), (-111, 18), (-116, 23),
     (-121, 30), (-127, 36), (-131, 43), (-140, 48), (-151, 50),
     (-160, 49)],
    # South America
    [(-81, 12), (-75, 10), (-70, 12), (-65, 10), (-58, 7),
     (-51, 4), (-45, 0), (-40, -6), (-35, -11), (-35, -20),
     (-39, -29), (-45, -35), (-50, -45), (-57, -52), (-67, -55),
     (-72, -49), (-76, -40), (-75, -31), (-78, -22), (-80, -11)],
    # Australia
    [(113, -11), (121, -13), (130, -12), (138, -14), (145, -15),
     (153, -20), (153, -27), (148, -34), (141, -39), (131, -39),
     (122, -36), (115, -34), (113, -28), (115, -22), (111, -17)],
]
# Additional connected land regions and polar ice-covered land.
POLYS += [
    # Arabian Peninsula
    [(34, 30), (39, 32), (46, 30), (52, 26), (56, 22),
     (59, 18), (56, 13), (51, 12), (47, 14), (43, 17),
     (39, 20), (36, 24)],
    # Indian subcontinent
    [(61, 25), (67, 28), (73, 28), (78, 30), (84, 28),
     (89, 26), (94, 23), (94, 18), (88, 20), (84, 16),
     (81, 8), (76, 7), (73, 12), (69, 19), (64, 21)],
    # Indochina and the Malay peninsula
    [(96, 22), (102, 23), (108, 21), (110, 17), (108, 13),
     (103, 10), (101, 5), (103, 1), (101, -1), (97, 5),
     (98, 12)],
    # Korean peninsula
    [(126, 39), (129, 43), (132, 42), (130, 37), (127, 34),
     (126, 36)],
    # Alaska and its western extension, kept separate from the main polygon
    [(-170, 52), (-166, 57), (-160, 60), (-153, 59), (-146, 61),
     (-141, 58), (-145, 54), (-153, 52), (-160, 50)],
    [(-180, 65), (-174, 69), (-168, 68), (-165, 64), (-170, 61),
     (-180, 61)],
    # Antarctic land ice: a polar cap with a deliberately uneven coastline.
    [(-180, -70), (-160, -70), (-140, -72), (-120, -72),
     (-100, -72), (-80, -71), (-62, -70), (-45, -74),
     (-25, -72), (-8, -73), (10, -71), (28, -70), (47, -70),
     (64, -68), (82, -68), (101, -66), (120, -68), (140, -70),
     (160, -68), (180, -70), (180, -90), (-180, -90)],
    # Antarctic Peninsula and a few attached ice shelves
    [(-66, -70), (-64, -64), (-61, -61), (-59, -63),
     (-60, -68), (-58, -72)],
]


# Large islands, all written as simple outlines or geographic ellipses.
ISLAND_POLYS = [
    # Greenland
    [(-73, 60), (-67, 64), (-57, 65), (-48, 60), (-39, 59),
     (-27, 64), (-19, 70), (-21, 78), (-30, 83), (-43, 84),
     (-55, 82), (-64, 76), (-70, 68)],
    # Madagascar
    [(43, -12), (48, -12), (51, -18), (50, -26), (46, -27),
     (43, -22), (42, -16)],
    # Hispaniola and Cuba
    [(-85, 23), (-80, 21), (-75, 20), (-73, 18), (-77, 17),
     (-83, 19)],
    [(-74, 20), (-70, 22), (-65, 22), (-62, 20), (-67, 19)],
    # Borneo
    [(109, 7), (117, 7), (119, 2), (117, -3), (112, -4),
     (108, 0), (106, 4)],
    # Sumatra
    [(95, 5), (102, 5), (106, 0), (104, -6), (99, -5),
     (96, 0)],
    # Java
    [(105, -6), (114, -6), (114, -9), (106, -9)],
    # New Guinea
    [(130, -2), (137, 0), (145, -2), (153, -5), (148, -10),
     (140, -9), (134, -6)],
    # New Zealand North and South Islands
    [(172, -34), (178, -37), (177, -42), (173, -41), (171, -37)],
    [(166, -45), (174, -46), (172, -49), (166, -47)],
    # Taiwan and Hainan
    [(120, 25), (122, 25), (122, 22), (120, 21)],
    [(109, 20), (111, 20), (111, 18), (109, 18)],
    # Sri Lanka
    [(80, 10), (82, 8), (81, 5), (79, 6), (79, 9)],
    # Iceland
    [(-24, 66), (-20, 64), (-15, 64), (-13, 66), (-16, 67),
     (-22, 67)],
    # Ireland and Great Britain
    [(-11, 54), (-10, 52), (-7, 51), (-6, 54), (-8, 56)],
    [(-6, 50), (1, 51), (2, 55), (0, 58), (-4, 58), (-6, 55)],
    # Japan's main islands
    [(130, 31), (134, 33), (136, 35), (135, 37), (131, 35)],
    [(137, 34), (141, 34), (145, 38), (143, 42), (139, 41)],
]
# Compact islands and island groups represented by approximate ellipses.
ELLIPSES = [
    (-43, -73, 17, 10),       # Greenland interior supplement
    (18, 78, 9, 4),           # Svalbard
    (12, 67, 4, 5),           # northern Norway islands
    (10, 39, 1, 2), (14, 38, 2, 2),  # Balearic / Mediterranean
    (9, 41, 2, 2), (12, 38, 2, 2),    # Corsica/Sardinia/Sicily area
    (25, 35, 4, 2),            # Cyprus
    (-17, 28, 2, 2),           # Canaries
    (-26, 38, 2, 1),           # Azores
    (-5, 36, 2, 1),            # Gibraltar vicinity
    (-78, 26, 2, 1),           # Bahamas
    (-76, 18, 2, 1), (-72, 18, 2, 1),  # Caribbean
    (-66, 18, 1, 1), (-61, 15, 1, 2),  # Puerto Rico / Lesser Antilles
    (-77, 18, 2, 1),            # Jamaica
    (-114, 29, 3, 8),           # Baja California
    (-89, 15, 3, 5),            # Central America (also reinforced below)
    (-91, 17, 3, 2),            # Yucatan
    (-157, 20, 2, 1), (-155, 20, 1, 1),  # Hawaii
    (-70, -53, 3, 2),           # Tierra del Fuego
    (-59, -52, 4, 2),           # Falklands
    (-37, 55, 2, 1),            # Faroe vicinity
    (47, -20, 2, 2),             # Madagascar supplement
    (57, -20, 1, 1), (55, -21, 1, 1),    # Mascarenes
    (38, -4, 1, 2),              # Zanzibar / nearby
    (12, -1, 1, 1),              # Gulf of Guinea islands
    (113, -1, 3, 3), (123, -1, 2, 3),    # Borneo/Sulawesi supplements
    (119, -4, 2, 2), (125, -4, 2, 2),    # Indonesian islands
    (135, -4, 2, 2), (124, 6, 2, 2),     # Philippines
    (120, 12, 2, 3), (125, 10, 2, 3),
    (138, 36, 1, 2), (142, 43, 2, 2),    # Japan supplements
    (155, -6, 2, 2),             # eastern New Guinea
    (160, -9, 2, 2), (166, -16, 2, 2),   # Melanesia
    (178, -18, 2, 2), (175, -21, 2, 2),  # Fiji / Tonga vicinity
    (147, -42, 5, 4),             # Tasmania
    (69, 8, 2, 2),                # Maldives (coarse)
    (92, 11, 4, 2),               # Andaman/Nicobar group
    (107, 16, 1, 1),              # Gulf of Tonkin island
    (141, 44, 4, 3),              # Sakhalin
    (143, 53, 2, 3),              # Kurils
    (-25, 72, 2, 3),              # Jan Mayen region
    (-2, 72, 3, 2),               # northern British islands
    (-28, 81, 4, 2),              # Greenland high Arctic islands
]

# Narrow peninsulas which are easy to lose at a two-degree sample spacing.
ISLAND_POLYS += [
    [(-93, 15), (-88, 18), (-84, 15), (-82, 10), (-78, 8),
     (-80, 5), (-85, 9), (-90, 12)],
    [(-118, 33), (-114, 32), (-110, 27), (-112, 23), (-117, 27)],
    [(140, 47), (146, 45), (150, 47), (149, 51), (143, 52)],
    [(152, -23), (154, -27), (153, -31), (150, -28)],
    [(-61, -63), (-58, -62), (-57, -65), (-60, -67)],
]


# Lakes are water under the experiment's definition.  These are only applied
# at grid samples and do not attempt to draw lake shorelines.
LAKE_ELLIPSES = [
    (-86, 46, 3, 2), (-84, 44, 2, 2), (-82, 45, 2, 1),
    (-79, 44, 2, 1),                         # Great Lakes
    (50, 42, 4, 2), (53, 46, 2, 2),          # Caspian and north shore
    (33, -1, 3, 2), (32, -5, 2, 2),          # Victoria/Tanganyika vicinity
    (107, 53, 2, 4), (109, 48, 2, 2),        # Baikal and nearby
    (62, 45, 2, 2),                          # Aral Sea
]
def is_land(lon, lat):
    if any_poly(lon, lat, POLYS) or any_poly(lon, lat, ISLAND_POLYS):
        land = True
    else:
        land = any(ellipse(lon, lat, *e) for e in ELLIPSES)
    if any(ellipse(lon, lat, *e) for e in LAKE_ELLIPSES):
        land = False
    return bool(land)


def build_mask():
    # Row order is north to south; column order is west to east.
    lats = [89 - 2 * r for r in range(90)]
    lons = [-179 + 2 * c for c in range(180)]
    return np.array(
        [[1 if is_land(lon, lat) else 0 for lon in lons] for lat in lats],
        dtype=np.uint8,
    )


def write_outputs(mask):
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(mask.tolist())

    # L mode is a one-channel, exact black/white PNG.
    raw = (mask * 255).astype(np.uint8)
    Image.fromarray(raw, mode="L").save(out / "candidate.png")
    Image.fromarray(raw, mode="L").resize(
        (1440, 720), Image.Resampling.NEAREST
    ).save(out / "candidate_8x.png")


def validate(mask):
    assert mask.shape == (90, 180)
    assert mask.size == 16200
    assert set(np.unique(mask)).issubset({0, 1})
    with Image.open("outputs/candidate.png") as im:
        assert im.size == (180, 90)
        assert set(im.convert("L").getdata()).issubset({0, 255})
    with Image.open("outputs/candidate_8x.png") as im:
        assert im.size == (1440, 720)
        assert set(im.convert("L").getdata()).issubset({0, 255})


if __name__ == "__main__":
    matrix = build_mask()
    write_outputs(matrix)
    validate(matrix)
    print("wrote 90x180 matrix and exact nearest-neighbor PNGs")
