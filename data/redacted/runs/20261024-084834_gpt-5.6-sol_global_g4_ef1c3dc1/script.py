from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image


# 坐标顺序严格按照题目要求：
# 纬度：89, 87, ..., -89
# 经度：-179, -177, ..., 179
LATS = np.arange(89, -90, -2, dtype=float)
LONS = np.arange(-179, 180, 2, dtype=float)
X, Y = np.meshgrid(LONS, LATS)


def polygon_mask(vertices):
    """Ray-casting point-in-polygon test over the complete sampling grid."""
    result = np.zeros(X.shape, dtype=bool)
    n = len(vertices)

    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]

        if y1 == y2:
            continue

        crosses = (y1 > Y) != (y2 > Y)
        intersection_x = x1 + (Y - y1) * (x2 - x1) / (y2 - y1)
        result ^= crosses & (X < intersection_x)

    return result


def ellipse_mask(cx, cy, rx, ry, angle=0.0):
    """Rotated ellipse, used only for manually specified island outlines."""
    a = math.radians(angle)
    ca = math.cos(a)
    sa = math.sin(a)

    dx = X - cx
    dy = Y - cy
    xr = ca * dx + sa * dy
    yr = -sa * dx + ca * dy

    return (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0


land = np.zeros(X.shape, dtype=bool)


def add_polygon(vertices):
    global land
    land |= polygon_mask(vertices)


def remove_polygon(vertices):
    global land
    land &= ~polygon_mask(vertices)


def add_ellipse(cx, cy, rx, ry, angle=0.0):
    global land
    land |= ellipse_mask(cx, cy, rx, ry, angle)


def remove_ellipse(cx, cy, rx, ry, angle=0.0):
    global land
    land &= ~ellipse_mask(cx, cy, rx, ry, angle)


# ---------------------------------------------------------------------------
# North America: hand-written, deliberately approximate coastline.
# ---------------------------------------------------------------------------
add_polygon([
    (-168, 72), (-158, 71), (-151, 70), (-143, 70),
    (-136, 68), (-130, 70), (-121, 74), (-111, 75),
    (-101, 74), (-94, 72), (-87, 70), (-82, 66),
    (-76, 64), (-69, 61), (-62, 59), (-58, 55),
    (-56, 51), (-60, 47), (-66, 45), (-70, 43),
    (-71, 41), (-75, 39), (-77, 36), (-80, 33),
    (-81, 29), (-80, 25), (-83, 25), (-87, 29),
    (-91, 29), (-94, 28), (-97, 26), (-99, 23),
    (-103, 20), (-106, 22), (-109, 25), (-112, 29),
    (-115, 32), (-118, 34), (-121, 38), (-124, 42),
    (-125, 48), (-129, 52), (-134, 55), (-139, 59),
    (-146, 60), (-151, 58), (-157, 56), (-163, 58),
    (-166, 62), (-164, 66)
])

# Baja California
add_polygon([
    (-117, 33), (-114, 31), (-112, 27), (-110, 23),
    (-111, 22), (-114, 26), (-116, 29)
])

# Central America
add_polygon([
    (-105, 21), (-99, 18), (-94, 17), (-91, 14),
    (-87, 12), (-84, 10), (-82, 8), (-78, 9),
    (-80, 11), (-84, 12), (-88, 16), (-94, 19),
    (-100, 22)
])

# Hudson Bay and ocean-connected James Bay.
remove_polygon([
    (-96, 64), (-89, 65), (-82, 63), (-77, 59),
    (-79, 54), (-82, 51), (-87, 51), (-91, 54),
    (-94, 58)
])
remove_ellipse(-80.5, 50.8, 2.4, 4.0, -10)

# Gulf of St. Lawrence.
remove_polygon([
    (-67, 50), (-63, 51), (-58, 50), (-59, 46),
    (-64, 46), (-68, 48)
])

# Gulf of California.
remove_polygon([
    (-114.7, 31.5), (-112.0, 29.0), (-109.0, 24.0),
    (-110.1, 22.7), (-113.0, 27.0)
])

# Greenland.
add_polygon([
    (-73, 78), (-68, 82), (-55, 84), (-39, 83),
    (-25, 80), (-20, 75), (-25, 69), (-32, 64),
    (-42, 60), (-50, 61), (-57, 66), (-61, 72)
])

# Canadian Arctic Archipelago.
add_polygon([
    (-90, 83), (-78, 83), (-62, 82), (-59, 77),
    (-68, 74), (-77, 75), (-84, 78)
])
add_polygon([
    (-82, 74), (-72, 74), (-62, 70), (-65, 64),
    (-73, 63), (-79, 67)
])
add_polygon([
    (-126, 75), (-117, 77), (-110, 75), (-113, 70),
    (-122, 70), (-128, 72)
])
add_polygon([
    (-112, 76), (-102, 78), (-96, 76), (-99, 71),
    (-107, 70), (-113, 72)
])
add_polygon([
    (-100, 81), (-90, 82), (-84, 79), (-91, 76),
    (-100, 77)
])
add_ellipse(-122, 73, 5.5, 3.5, -15)
add_ellipse(-105, 73, 4.5, 2.8, 10)

# Newfoundland and nearby large islands.
add_ellipse(-56.5, 49.0, 3.7, 5.0, -22)
add_ellipse(-63.0, 46.5, 2.0, 1.1, 5)
add_ellipse(-127.0, 50.5, 2.0, 4.0, -18)

# Aleutian Islands.
for item in [
    (-171, 52.5, 2.2, 0.8, -8),
    (-176, 51.5, 1.8, 0.7, 0),
    (176, 52.0, 2.0, 0.8, 5),
    (169, 53.0, 2.2, 0.8, 15),
]:
    add_ellipse(*item)


# ---------------------------------------------------------------------------
# South America.
# ---------------------------------------------------------------------------
add_polygon([
    (-81, 12), (-75, 11), (-71, 12), (-66, 10),
    (-61, 10), (-57, 7), (-52, 5), (-49, 1),
    (-47, -5), (-43, -9), (-39, -14), (-38, -19),
    (-40, -23), (-44, -24), (-47, -28), (-49, -33),
    (-53, -38), (-58, -42), (-64, -46), (-67, -51),
    (-71, -55), (-74, -52), (-74, -47), (-72, -43),
    (-73, -39), (-71, -34), (-71, -29), (-70, -24),
    (-71, -19), (-75, -15), (-78, -10), (-80, -5),
    (-81, 0), (-79, 5)
])

# Tierra del Fuego and Falkland Islands.
add_ellipse(-69, -54, 4.2, 2.1, -10)
add_ellipse(-59, -51.5, 2.0, 1.2, 5)


# ---------------------------------------------------------------------------
# Africa.
# ---------------------------------------------------------------------------
add_polygon([
    (-17, 37), (-8, 36), (-1, 37), (8, 37),
    (14, 35), (21, 33), (27, 32), (32, 31),
    (35, 29), (36, 24), (38, 18), (43, 13),
    (51, 12), (50, 8), (46, 4), (43, 0),
    (42, -5), (40, -10), (37, -16), (35, -22),
    (32, -27), (29, -33), (24, -35), (18, -34),
    (14, -30), (11, -24), (8, -18), (4, -10),
    (1, -5), (-5, 1), (-8, 5), (-14, 10),
    (-17, 15), (-17, 21), (-14, 27), (-10, 31)
])

# Sinai: approximate land bridge.
add_polygon([
    (32, 31), (35, 31), (34, 27), (32, 28)
])

# Madagascar and notable nearby islands.
add_polygon([
    (49, -12), (51, -16), (50, -22), (47, -26),
    (44, -24), (43, -19), (45, -14)
])
add_ellipse(55.5, -4.5, 1.1, 1.1)
add_ellipse(43.3, -11.7, 1.2, 1.0)


# ---------------------------------------------------------------------------
# Eurasia main landmass.
# ---------------------------------------------------------------------------
add_polygon([
    (179, 67), (170, 69), (160, 70), (149, 72),
    (138, 73), (126, 73), (114, 75), (101, 77),
    (88, 77), (76, 75), (65, 75), (55, 72),
    (47, 69), (40, 68), (33, 69), (27, 71),
    (21, 70), (15, 68), (10, 64), (5, 61),
    (5, 58), (0, 57), (-5, 58), (-10, 55),
    (-9, 50), (-10, 44), (-9, 38), (-5, 36),
    (0, 36), (4, 40), (7, 44), (12, 45),
    (17, 45), (21, 42), (26, 41), (30, 41),
    (36, 37), (37, 35), (35, 32), (36, 29),
    (39, 24), (42, 18), (46, 13), (51, 12),
    (56, 16), (57, 21), (54, 25), (51, 27),
    (55, 27), (59, 25), (63, 25), (67, 24),
    (70, 22), (72, 19), (72, 15), (75, 10),
    (78, 7), (81, 8), (82, 14), (86, 20),
    (89, 22), (92, 22), (94, 18), (97, 14),
    (98, 8), (100, 2), (103, 1), (104, 5),
    (104, 10), (108, 11), (109, 16), (106, 21),
    (110, 22), (114, 22), (118, 25), (121, 29),
    (122, 34), (122, 38), (126, 40), (126, 36),
    (128, 34), (130, 35), (130, 41), (133, 44),
    (138, 46), (141, 51), (142, 55), (151, 59),
    (156, 58), (162, 61), (166, 64)
])

# Chukotka on the western side of the date line.
add_polygon([
    (-180, 66), (-176, 67), (-171, 66), (-168, 64),
    (-172, 62), (-178, 63), (-180, 64)
])

# Scandinavian southward detail.
add_polygon([
    (5, 58), (7, 55), (12, 55), (13, 58),
    (18, 60), (22, 64), (25, 69), (20, 71),
    (14, 68), (8, 63)
])

# Italy.
add_polygon([
    (7, 45), (12, 46), (14, 43), (17, 40),
    (16, 38), (13, 40), (11, 43)
])
add_ellipse(15.0, 37.5, 2.2, 1.4, -10)   # Sicily
add_ellipse(9.0, 40.0, 1.5, 2.5, 5)     # Sardinia
add_ellipse(9.0, 42.3, 1.0, 1.8, 5)     # Corsica

# Greece and Peloponnese.
add_polygon([
    (20, 42), (24, 42), (26, 39), (24, 35),
    (21, 37), (22, 39), (19, 39)
])

# Ocean-connected European and Asian seas.
remove_polygon([
    (12, 55), (15, 55), (18, 58), (23, 60),
    (29, 60), (30, 64), (24, 66), (19, 64),
    (15, 61)
])  # Baltic

remove_polygon([
    (27, 46), (32, 47), (38, 46), (42, 43),
    (41, 40), (36, 41), (31, 41), (28, 43)
])  # Black Sea

remove_polygon([
    (12, 45), (15, 45), (19, 43), (20, 40),
    (18, 39), (16, 42), (14, 43)
])  # Adriatic

remove_polygon([
    (47, 31), (50, 30), (54, 27), (57, 25),
    (55, 23), (51, 25), (48, 27)
])  # Persian Gulf

remove_polygon([
    (99, 14), (103, 13), (106, 10), (105, 5),
    (102, 3), (100, 7)
])  # Gulf of Thailand

remove_polygon([
    (34, 66), (39, 67), (42, 65), (40, 63),
    (36, 64)
])  # White Sea

# British Isles.
add_polygon([
    (-6, 50), (-3, 50), (0, 52), (1, 55),
    (-2, 59), (-5, 58), (-6, 55)
])
add_ellipse(-8.0, 53.3, 2.4, 3.1, 8)

# Iceland, Faroe.
add_polygon([
    (-24, 64), (-21, 63), (-16, 64), (-14, 66),
    (-18, 67), (-23, 66)
])
add_ellipse(-6.7, 62.0, 0.8, 0.8)

# Arctic Russian islands.
add_polygon([
    (45, 80), (58, 81), (66, 80), (62, 77),
    (51, 77)
])
add_polygon([
    (53, 76), (59, 76), (67, 73), (65, 70),
    (58, 71)
])
add_polygon([
    (75, 79), (95, 81), (103, 79), (93, 77),
    (80, 77)
])
add_polygon([
    (18, 81), (29, 81), (34, 78), (25, 76),
    (16, 77)
])
add_ellipse(106, 78, 5.0, 2.0, 5)
add_ellipse(140, 76, 3.0, 2.0, 0)

# Cyprus and Crete.
add_ellipse(33.0, 35.0, 2.2, 0.8, 5)
add_ellipse(25.0, 35.2, 3.6, 0.8, 5)

# Sri Lanka.
add_polygon([
    (80, 10), (82, 9), (82, 6), (80, 5), (79, 8)
])

# Japan.
add_polygon([
    (129, 31), (131, 31), (133, 33), (135, 34),
    (138, 35), (141, 39), (142, 43), (140, 45),
    (138, 41), (136, 37), (132, 35)
])
add_ellipse(143, 43, 3.0, 2.0, 15)  # Hokkaido
add_ellipse(130.5, 32.2, 1.7, 2.0, -10)
add_ellipse(134.2, 33.7, 2.1, 1.0, 5)

# Taiwan.
add_ellipse(121.0, 23.5, 1.3, 2.5, -12)

# Sakhalin.
add_ellipse(143.0, 50.5, 1.7, 5.5, -8)

# Kuril chain.
for cx, cy in [(146, 45), (149, 47), (153, 49)]:
    add_ellipse(cx, cy, 1.0, 1.8, -30)


# ---------------------------------------------------------------------------
# Maritime Southeast Asia.
# ---------------------------------------------------------------------------
# Sumatra.
add_polygon([
    (94, 6), (97, 6), (102, 1), (106, -5),
    (104, -6), (100, -3), (96, 1)
])

# Java.
add_polygon([
    (105, -6), (111, -6), (115, -8), (113, -9),
    (108, -8)
])

# Borneo.
add_polygon([
    (109, 7), (115, 7), (119, 4), (119, -2),
    (116, -5), (111, -4), (108, 0)
])

# Sulawesi.
add_polygon([
    (119, 2), (122, 2), (122, -2), (125, -4),
    (123, -6), (120, -4), (118, -5), (119, -1)
])
add_polygon([
    (121, 0), (125, 1), (126, 0), (123, -2)
])

# Philippines.
add_polygon([
    (120, 19), (123, 19), (124, 15), (122, 12),
    (120, 14)
])
add_polygon([
    (123, 13), (126, 12), (126, 8), (124, 6),
    (122, 8)
])
add_ellipse(118.5, 10.0, 1.2, 2.5, -20)
add_ellipse(125.5, 7.0, 1.8, 2.2, 5)

# Moluccas and Lesser Sunda.
for item in [
    (128, -3, 1.3, 2.7, 10),
    (130, 0, 1.1, 2.0, 0),
    (118, -8, 2.2, 0.8, 0),
    (122, -8, 2.0, 0.8, 0),
    (126, -8, 2.0, 0.8, 0),
    (130, -8, 2.0, 0.8, 0),
]:
    add_ellipse(*item)

# Andaman and Nicobar chain.
add_ellipse(93, 11, 0.8, 3.2, -10)
add_ellipse(94, 7, 0.7, 1.8, -10)


# ---------------------------------------------------------------------------
# New Guinea, Australia, New Zealand and Pacific islands.
# ---------------------------------------------------------------------------
add_polygon([
    (130, -2), (136, -1), (141, -3), (147, -5),
    (151, -7), (151, -10), (146, -9), (141, -7),
    (136, -7), (132, -5)
])

add_polygon([
    (113, -22), (114, -15), (121, -11), (129, -12),
    (136, -12), (142, -11), (146, -16), (153, -24),
    (153, -29), (149, -35), (143, -39), (136, -36),
    (130, -34), (124, -35), (117, -32), (113, -27)
])

# Cape York refinement.
add_polygon([
    (136, -17), (139, -11), (142, -10), (145, -15),
    (143, -19)
])

# Tasmania.
add_ellipse(147, -42.5, 2.8, 3.2, -5)

# New Zealand.
add_polygon([
    (166, -46), (169, -47), (174, -42), (173, -40),
    (170, -42)
])
add_polygon([
    (173, -41), (176, -39), (178, -35), (176, -34),
    (173, -37)
])

# New Caledonia.
add_ellipse(165.5, -21.5, 1.0, 3.4, -35)

# Hawaiian Islands; only the large-island-scale samples matter at 2 degrees.
add_ellipse(-155.5, 19.5, 1.1, 1.0, -10)
add_ellipse(-157.5, 21.0, 1.0, 0.8, -20)

# A few geographically significant Pacific islands/archipelagos.
add_ellipse(178, -17, 1.0, 2.0, 5)       # Fiji area
add_ellipse(-172, -14, 1.0, 1.4, 0)     # Samoa area
add_ellipse(-149, -17.5, 0.9, 0.9, 0)   # Tahiti
add_ellipse(167, -16, 0.9, 2.0, -20)     # Vanuatu


# ---------------------------------------------------------------------------
# Caribbean islands.
# ---------------------------------------------------------------------------
add_polygon([
    (-85, 22), (-82, 23), (-77, 22), (-74, 20),
    (-78, 19), (-83, 20)
])  # Cuba

add_ellipse(-70.5, 19.0, 3.7, 1.7, -5)  # Hispaniola
add_ellipse(-77.0, 18.2, 2.2, 1.0, 5)   # Jamaica
add_ellipse(-66.3, 18.2, 1.5, 0.8, 0)   # Puerto Rico
add_ellipse(-77.5, 24.5, 2.0, 3.7, -25) # Bahamas approximation

for cx, cy in [
    (-61.5, 16.5), (-61.5, 14.5), (-61, 12.5),
    (-62, 11), (-64, 10.5)
]:
    add_ellipse(cx, cy, 0.7, 0.9, 0)


# ---------------------------------------------------------------------------
# Antarctica. The northern edge is a hand-written approximate grounded-ice
# coastline. Floating Ross and Filchner-Ronne ice shelves are then removed.
# ---------------------------------------------------------------------------
add_polygon([
    (-180, -72), (-168, -74), (-155, -75), (-140, -74),
    (-125, -73), (-110, -73), (-94, -74), (-80, -74),
    (-71, -71), (-66, -67), (-62, -63), (-58, -64),
    (-60, -69), (-52, -72), (-42, -73), (-30, -72),
    (-15, -71), (0, -70), (20, -69), (39, -67),
    (58, -66), (78, -67), (98, -66), (119, -66),
    (139, -67), (158, -69), (180, -72),
    (180, -90), (-180, -90)
])

# Floating Ross Ice Shelf.
remove_polygon([
    (158, -77), (176, -77), (180, -78), (180, -86),
    (170, -86), (160, -83)
])
remove_polygon([
    (-180, -78), (-165, -77), (-150, -78), (-155, -83),
    (-170, -86), (-180, -86)
])

# Floating Filchner-Ronne Ice Shelf.
remove_polygon([
    (-61, -75), (-50, -74), (-35, -75), (-28, -78),
    (-35, -82), (-50, -83), (-60, -80)
])


# ---------------------------------------------------------------------------
# Write and validate all requested artifacts.
# ---------------------------------------------------------------------------
matrix = land.astype(np.uint8)

assert matrix.shape == (90, 180)
assert matrix.size == 16200
assert np.all((matrix == 0) | (matrix == 1))

output_dir = Path.cwd() / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

csv_path = output_dir / "candidate.csv"
png_path = output_dir / "candidate.png"
large_path = output_dir / "candidate_8x.png"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(matrix.tolist())

# 0 -> black, 1 -> white.
pixels = matrix * 255
image = Image.fromarray(pixels, mode="L")
image.save(png_path)

try:
    nearest = Image.Resampling.NEAREST
except AttributeError:
    nearest = Image.NEAREST

large_image = image.resize((1440, 720), resample=nearest)
large_image.save(large_path)

# Independent post-write checks.
with csv_path.open("r", newline="", encoding="utf-8") as f:
    rows = list(csv.reader(f))

assert len(rows) == 90
assert all(len(row) == 180 for row in rows)
assert all(value in {"0", "1"} for row in rows for value in row)
assert sum(len(row) for row in rows) == 16200

with Image.open(png_path) as check:
    check.load()
    assert check.size == (180, 90)
    assert set(check.convert("L").getdata()).issubset({0, 255})

with Image.open(large_path) as check:
    check.load()
    assert check.size == (1440, 720)
    assert set(check.convert("L").getdata()).issubset({0, 255})

# Verify the enlarged image is exactly an 8x nearest-neighbor replication.
with Image.open(large_path) as check:
    enlarged_array = np.asarray(check.convert("L"))

expected_enlarged = np.repeat(np.repeat(pixels, 8, axis=0), 8, axis=1)
assert np.array_equal(enlarged_array, expected_enlarged)

print(f"Generated: {csv_path}")
print(f"Generated: {png_path}")
print(f"Generated: {large_path}")
print(f"Matrix: {matrix.shape[0]}x{matrix.shape[1]} ({matrix.size} values)")
