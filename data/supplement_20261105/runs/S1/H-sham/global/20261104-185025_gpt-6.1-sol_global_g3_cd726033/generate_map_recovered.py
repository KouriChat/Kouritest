from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------

def ellipse(cx, cy, rx, ry, n=32):
    """Return a hand-authored ellipse approximation in lon/lat coordinates."""
    return [
        (
            cx + rx * math.cos(2.0 * math.pi * i / n),
            cy + ry * math.sin(2.0 * math.pi * i / n),
        )
        for i in range(n)
    ]


def point_in_polygon(x, y, polygon):
    """Ray-casting point-in-polygon test, with boundary treated as inside."""
    inside = False
    j = len(polygon) - 1

    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # Boundary test.
        dx = xj - xi
        dy = yj - yi
        cross = (x - xi) * dy - (y - yi) * dx
        if abs(cross) < 1e-10:
            if (
                min(xi, xj) - 1e-10 <= x <= max(xi, xj) + 1e-10
                and min(yi, yj) - 1e-10 <= y <= max(yi, yj) + 1e-10
            ):
                return True

        if (yi > y) != (yj > y):
            crossing_x = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < crossing_x:
                inside = not inside

        j = i

    return inside


# ----------------------------------------------------------------------
# Hand-written approximate land regions
# ----------------------------------------------------------------------

outer_regions = [
    # North America, including the continental United States and Canada.
    [
        (-168, 71), (-150, 72), (-135, 70), (-125, 61),
        (-123, 49), (-117, 32), (-108, 25), (-98, 24),
        (-90, 28), (-82, 25), (-79, 31), (-73, 35),
        (-68, 44), (-60, 48), (-61, 53), (-72, 58),
        (-87, 59), (-95, 64), (-110, 69), (-130, 73),
        (-150, 73), (-168, 70),
    ],

    # Alaska and the far northwest of North America.
    [
        (-170, 70), (-160, 72), (-145, 71), (-135, 66),
        (-130, 60), (-140, 55), (-151, 56), (-163, 59),
        (-173, 64),
    ],

    # Eastern Canada and Labrador.
    [
        (-96, 58), (-88, 61), (-76, 60), (-65, 55),
        (-57, 51), (-59, 57), (-68, 63), (-80, 66),
        (-92, 68), (-108, 70), (-110, 64),
    ],

    # Eurasian mainland, from the Middle East across Siberia.
    [
        (26, 41), (35, 42), (44, 46), (53, 51),
        (62, 58), (72, 63), (82, 70), (99, 77),
        (122, 76), (145, 71), (161, 66), (178, 62),
        (179, 51), (169, 45), (158, 41), (149, 38),
        (143, 34), (135, 35), (130, 31), (124, 26),
        (119, 21), (112, 18), (108, 13), (102, 10),
        (96, 8), (90, 10), (84, 15), (78, 21),
        (69, 23), (62, 24), (56, 27), (49, 28),
        (43, 32), (36, 36), (30, 38),
    ],

    # Far eastern Russia and Chukotka near the date line.
    [
        (158, 66), (170, 67), (179, 66), (179, 51),
        (171, 51), (164, 56), (157, 60),
    ],

    # Europe.
    [
        (-18, 58), (-12, 51), (-10, 44), (-10, 37),
        (2, 36), (8, 39), (11, 44), (16, 41),
        (22, 40), (28, 42), (33, 45), (40, 47),
        (40, 55), (34, 59), (30, 65), (24, 70),
        (15, 72), (5, 70), (-5, 67), (-12, 63),
    ],

    # Africa.
    [
        (-17, 15), (-17, 28), (-10, 34), (0, 37),
        (10, 37), (20, 34), (30, 32), (36, 28),
        (38, 21), (43, 13), (51, 11), (50, 1),
        (45, -11), (40, -21), (33, -34), (21, -35),
        (10, -35), (3, -30), (-5, -30), (-12, -18),
        (-17, -5),
    ],

    # South America.
    [
        (-81, 12), (-72, 12), (-66, 9), (-60, 5),
        (-52, 4), (-49, -2), (-44, -8), (-40, -15),
        (-35, -22), (-38, -30), (-48, -34), (-53, -42),
        (-62, -52), (-70, -55), (-75, -52), (-72, -40),
        (-76, -25), (-80, -10), (-82, 0),
    ],

    # Australia.
    [
        (113, -10), (126, -11), (137, -12), (145, -15),
        (153, -25), (153, -34), (145, -39), (134, -39),
        (124, -35), (115, -35), (111, -28), (109, -21),
    ],

    # Antarctica. The northern boundary is intentionally approximate.
    [
        (-180, -72), (-160, -70), (-140, -70), (-120, -72),
        (-100, -73), (-82, -75), (-67, -73), (-53, -70),
        (-43, -67), (-30, -66), (-17, -70), (0, -70),
        (17, -70), (32, -69), (46, -68), (61, -70),
        (77, -68), (94, -67), (110, -69), (127, -71),
        (145, -72), (162, -70), (180, -70), (180, -90),
        (-180, -90),
    ],

    # Arabian Peninsula.
    [
        (34, 31), (42, 30), (50, 29), (57, 26),
        (59, 20), (56, 14), (50, 12), (43, 13),
        (39, 18), (36, 24),
    ],

    # Indian subcontinent.
    [
        (68, 24), (75, 23), (80, 27), (88, 27),
        (93, 22), (97, 20), (95, 13), (90, 8),
        (83, 7), (77, 10), (73, 15), (69, 20),
    ],

    # Mainland Southeast Asia.
    [
        (96, 23), (104, 23), (110, 19), (110, 14),
        (107, 10), (103, 7), (98, 9), (95, 15),
    ],
]


# Land regions added after lake and sea exclusions. These include islands,
# peninsulas which were simplified by the broad mainland polygons, and
# small but geographically recognizable island groups.
restored_regions = [
    # Central America.
    [
        (-117, 32), (-110, 25), (-105, 20), (-98, 18),
        (-91, 15), (-87, 12), (-83, 9), (-80, 8),
        (-84, 16), (-89, 19), (-94, 22), (-103, 26),
        (-111, 29),
    ],

    # Antarctic Peninsula.
    [
        (-75, -63), (-68, -63), (-62, -66), (-57, -69),
        (-54, -73), (-60, -76), (-69, -73), (-75, -69),
    ],

    # Turkey and the southern Balkans.
    [
        (26, 36), (35, 36), (42, 39), (43, 42),
        (36, 43), (30, 42), (26, 40),
    ],
    [
        (14, 42), (20, 40), (27, 40), (30, 44),
        (25, 46), (18, 46), (14, 44),
    ],

    # Iberian Peninsula.
    [
        (-10, 36), (3, 36), (4, 43), (-1, 44),
        (-8, 43), (-10, 40),
    ],

    # Italy and Sicily.
    [
        (7, 44), (14, 46), (18, 43), (17, 39),
        (14, 37), (13, 41), (10, 40),
    ],
    [(12, 38), (16, 38), (16, 36), (13, 36)],

    # India and Sri Lanka.
    [(79, 8), (82, 8), (82, 5), (79, 5)],
    [(80, 10), (82, 9), (82, 6), (80, 6)],

    # Japan's major islands.
    [(129, 31), (132, 33), (136, 35), (141, 39), (145, 42),
     (143, 45), (138, 43), (134, 38), (130, 35)],
    [(140, 43), (145, 45), (146, 42), (143, 40)],
    [(142, 39), (145, 40), (146, 37), (143, 35)],
    [(130, 34), (132, 35), (134, 34), (133, 32)],

    # Korean Peninsula.
    [(126, 34), (130, 35), (130, 39), (127, 43), (125, 40)],

    # Taiwan and Hainan.
    [(119, 22), (121, 25), (123, 24), (122, 22)],
    [(108, 18), (111, 20), (111, 18), (109, 17)],

    # Borneo.
    [(109, 7), (116, 7), (119, 3), (117, -2),
     (111, -4), (109, 0)],
    # Sumatra.
    [(95, 5), (103, 5), (106, 1), (103, -5),
     (98, -5), (95, 0)],
    # Java.
    [(105, -6), (114, -6), (115, -8), (106, -9)],
    # Sulawesi.
    [(119, 1), (123, 4), (124, 0), (121, -4),
     (118, -3), (120, 0)],
    # Lesser Sunda islands.
    [(115, -8), (124, -8), (127, -9), (124, -10),
     (116, -10)],
    # New Guinea.
    [(130, -2), (141, -2), (151, -5), (154, -9),
     (148, -11), (138, -9), (132, -6)],

    # Madagascar.
    [(43, -12), (50, -13), (50, -23), (46, -26),
     (43, -22), (42, -16)],

    # New Zealand.
    [(166, -34), (174, -36), (178, -42), (174, -47),
     (168, -46), (165, -40)],
    [(172, -40), (178, -39), (179, -45), (174, -47)],

    # British Isles and Ireland.
    [(-8, 50), (2, 51), (2, 58), (-3, 59), (-6, 55)],
    [(-11, 52), (-6, 52), (-6, 56), (-10, 55)],
    [(-9, 51), (-6, 51), (-7, 49), (-10, 50)],

    # Iceland, Svalbard and Jan Mayen.
    [(-25, 63), (-14, 63), (-13, 67), (-20, 67), (-25, 65)],
    [(-18, 70), (-14, 70), (-13, 73), (-18, 73)],
    [(10, 76), (25, 76), (28, 81), (18, 81), (10, 79)],
    [(50, 71), (58, 72), (59, 77), (52, 78)],

    # Arctic islands.
    [(-96, 76), (-80, 76), (-60, 78), (-61, 84),
     (-76, 84), (-95, 81)],
    [(-86, 62), (-66, 62), (-60, 68), (-66, 74),
     (-80, 73), (-87, 68)],
    [(-118, 69), (-102, 69), (-98, 74), (-109, 77),
     (-121, 75)],
    [(-132, 70), (-118, 72), (-115, 77), (-126, 78),
     (-138, 75)],
    [(88, 76), (105, 76), (108, 80), (98, 82), (88, 80)],
    [(135, 75), (150, 75), (153, 79), (144, 80), (136, 78)],
    [(160, 70), (170, 70), (172, 74), (166, 76), (160, 73)],
    [(179, 71), (180, 71), (180, 76), (177, 76)],

    # Sakhalin and Kamchatka.
    [(141, 46), (145, 47), (145, 54), (142, 59), (140, 54)],
    [(157, 51), (163, 52), (166, 58), (163, 62), (158, 59)],

    # Philippines.
    [(120, 18), (123, 19), (125, 16), (124, 12),
     (121, 10), (119, 13)],
    [(125, 11), (128, 13), (128, 8), (125, 5), (123, 8)],
    [(117, 8), (120, 10), (120, 6), (118, 4)],

    # Caribbean.
    [(-85, 21), (-77, 23), (-74, 20), (-77, 18), (-83, 19)],
    [(-74, 19), (-68, 20), (-68, 17), (-73, 17)],
    [(-67, 18), (-64, 19), (-65, 16), (-67, 16)],
    [(-78, 16), (-75, 16), (-75, 12), (-78, 13)],

    # Bermuda.
    [(-65, 32), (-63, 32), (-63, 33), (-65, 33)],

    # Pacific islands.
    [(-161, 21), (-155, 21), (-154, 19), (-160, 18)],
    [(-174, -14), (-170, -13), (-170, -18), (-174, -18)],
    [(-178, -22), (-174, -20), (-174, -24), (-178, -24)],
    [(166, -14), (170, -14), (170, -19), (166, -20)],
    [(159, -7), (163, -6), (163, -11), (158, -11)],
    [(176, -17), (180, -17), (180, -21), (176, -21)],
    [(177, -4), (180, -4), (180, -8), (177, -8)],
    [(140, -8), (145, -7), (145, -10), (140, -10)],

    # Indian Ocean islands with a chance of appearing on the 2-degree grid.
    [(55, -4), (57, -4), (57, -6), (55, -6)],
    [(73, 5), (74, 5), (74, 3), (73, 3)],
]


# Add smooth approximations for several irregular island groups.
restored_regions.extend([
    ellipse(-42, 73, 2.8, 4.0),       # Greenland's northern extension
    ellipse(-83, 74, 5.0, 4.0),       # Ellesmere/Baffin transition
    ellipse(-91, 66, 4.0, 3.0),       # western Canadian Arctic island
    ellipse(-61, 54, 2.0, 2.0),       # Newfoundland
    ellipse(-45, 60, 3.0, 2.0),       # southern Greenland fringe
    ellipse(138, -35, 4.0, 3.0),      # Tasmania
    ellipse(72, 7, 1.0, 2.0),         # Sri Lanka
    ellipse(25, 39, 1.7, 1.2),        # Crete
    ellipse(20, 38, 1.3, 1.0),        # Aegean island group
    ellipse(14, 39, 0.8, 1.2),        # Sicily
    ellipse(-1, 53, 1.7, 1.8),        # Great Britain interior
    ellipse(-9, 53, 1.5, 1.8),        # Ireland interior
    ellipse(134, 35, 1.0, 1.0),       # Shikoku
    ellipse(130, 33, 1.0, 1.0),       # Kyushu
    ellipse(141, 43, 2.0, 2.5),       # Hokkaido
    ellipse(120, 13, 1.5, 3.5),       # Philippines central
    ellipse(123, 7, 1.5, 3.0),        # Philippines south
    ellipse(114, 1, 3.5, 3.5),        # Borneo interior
    ellipse(137, -5, 7.0, 4.0),       # New Guinea interior
    ellipse(47, -19, 3.5, 7.0),       # Madagascar interior
])


# Water bodies that fall inside broad hand-written land polygons.
water_regions = [
    # Hudson Bay.
    [
        (-96, 52), (-91, 57), (-83, 59), (-77, 56),
        (-79, 51), (-87, 50), (-94, 50),
    ],

    # Great Lakes.
    [(-93, 46), (-88, 47), (-84, 46), (-87, 44), (-92, 44)],
    [(-88, 45), (-85, 46), (-84, 42), (-87, 41)],
    [(-84, 46), (-80, 46), (-79, 43), (-83, 42)],
    [(-84, 43), (-79, 43), (-78, 41), (-82, 41)],
    [(-80, 44), (-76, 44), (-76, 42), (-80, 42)],

    # Great Bear and Great Slave lakes.
    [(-125, 65), (-118, 66), (-117, 63), (-124, 63)],
    [(-121, 62), (-110, 62), (-108, 59), (-118, 59)],

    # Black Sea.
    [(27, 41), (35, 41), (42, 43), (42, 47),
     (34, 47), (28, 45)],

    # Caspian Sea.
    [(47, 37), (51, 36), (54, 40), (53, 46),
     (49, 47), (47, 42)],

    # Aral Sea.
    [(58, 41), (62, 41), (62, 46), (58, 46)],

    # Persian Gulf.
    [(48, 24), (55, 24), (57, 27), (53, 30),
     (49, 29)],

    # Red Sea.
    [(32, 28), (36, 28), (43, 17), (43, 12),
     (39, 13), (34, 22)],

    # Gulf of Aden.
    [(43, 12), (52, 12), (52, 16), (44, 16)],

    # Lake Victoria.
    [(32, -1), (36, -1), (36, 3), (32, 3)],

    # Lake Tanganyika.
    [(29, -4), (31, -4), (31, -8), (29, -8)],

    # Lake Malawi.
    [(33, -9), (35, -9), (35, -15), (33, -15)],

    # Lake Chad.
    [(13, 12), (15, 12), (15, 15), (13, 15)],

    # Lake Titicaca.
    [(-70, -14), (-68, -14), (-68, -16), (-70, -16)],

    # Lake Baikal.
    [(103, 51), (108, 51), (110, 54), (106, 56), (103, 54)],

    # Lake Ladoga.
    [(30, 59), (33, 59), (33, 62), (30, 62)],
]


def is_land(lon, lat):
    base_land = any(point_in_polygon(lon, lat, p) for p in outer_regions)

    if base_land:
        if any(point_in_polygon(lon, lat, p) for p in water_regions):
            base_land = False

    # These are deliberately applied after water exclusions because some
    # broad water polygons overlap simplified coastlines or peninsulas.
    if any(point_in_polygon(lon, lat, p) for p in restored_regions):
        base_land = True

    return 1 if base_land else 0


# ----------------------------------------------------------------------
# Generate the 90 x 180 sample matrix
# ----------------------------------------------------------------------

latitudes = list(range(89, -90, -2))
longitudes = list(range(-179, 180, 2))

assert len(latitudes) == 90
assert len(longitudes) == 180

matrix = np.zeros((90, 180), dtype=np.uint8)

for row, lat in enumerate(latitudes):
    for col, lon in enumerate(longitudes):
        matrix[row, col] = is_land(lon, lat)

assert matrix.shape == (90, 180)
assert matrix.size == 16200
assert set(np.unique(matrix)).issubset({0, 1})


# ----------------------------------------------------------------------
# Write CSV and PNG files
# ----------------------------------------------------------------------

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

csv_path = output_dir / "candidate.csv"
with csv_path.open("w", newline="", encoding="ascii") as f:
    writer = csv.writer(f, lineterminator="\n")
    for row in matrix:
        writer.writerow([str(int(value)) for value in row])

# Black = water, white = land.
image_array = matrix * 255
image = Image.fromarray(image_array.astype(np.uint8), mode="L")

assert image.size == (180, 90)
assert set(image.getdata()).issubset({0, 255})

png_path = output_dir / "candidate.png"
image.save(png_path, format="PNG")

resampling = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
large_image = image.resize((1440, 720), resample=resampling)

assert large_image.size == (1440, 720)
assert set(large_image.getdata()).issubset({0, 255})

large_png_path = output_dir / "candidate_8x.png"
large_image.save(large_png_path, format="PNG")

# Additional runtime checks for the requested deliverables.
with csv_path.open("r", encoding="ascii") as f:
    csv_rows = list(csv.reader(f))

assert len(csv_rows) == 90
assert all(len(row) == 180 for row in csv_rows)
assert all(value in {"0", "1"} for row in csv_rows for value in row)

print("Generated:")
print(f"  {csv_path}       : 90 rows x 180 columns")
print(f"  {png_path}       : {image.size[0]} x {image.size[1]}")
print(f"  {large_png_path} : {large_image.size[0]} x {large_image.size[1]}")
print(f"  land samples     : {int(matrix.sum())}")
print(f"  water samples    : {int(matrix.size - matrix.sum())}")
