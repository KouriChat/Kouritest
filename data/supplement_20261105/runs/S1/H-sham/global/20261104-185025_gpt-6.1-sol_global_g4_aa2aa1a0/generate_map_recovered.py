from pathlib import Path
import math

import numpy as np
from PIL import Image


# -----------------------------
# Basic geometry helpers
# -----------------------------

def polygon(*points):
    return tuple((float(x), float(y)) for x, y in points)


def rectangle(lon1, lat1, lon2, lat2):
    return polygon(
        (lon1, lat1),
        (lon2, lat1),
        (lon2, lat2),
        (lon1, lat2),
    )


def ellipse(cx, cy, rx, ry, count=24):
    points = []
    for i in range(count):
        angle = 2.0 * math.pi * i / count
        points.append(
            (
                cx + rx * math.cos(angle),
                cy + ry * math.sin(angle),
            )
        )
    return tuple(points)


def point_in_polygon(lon, lat, points):
    inside = False
    j = len(points) - 1

    for i in range(len(points)):
        xi, yi = points[i]
        xj, yj = points[j]

        crosses = ((yi > lat) != (yj > lat))
        if crosses:
            x_at_lat = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_at_lat:
                inside = not inside

        j = i

    return inside


# -----------------------------
# Handwritten land rules
# -----------------------------

LAND_POLYGONS = []

def add_land(*points):
    LAND_POLYGONS.append(polygon(*points))


# Antarctica, including the permanent continental ice sheet.
add_land(
    (-180, -61), (-160, -66), (-140, -70), (-120, -71),
    (-100, -73), (-80, -74), (-65, -69), (-55, -66),
    (-40, -69), (-20, -70), (0, -70), (20, -69),
    (40, -70), (60, -68), (80, -69), (100, -67),
    (120, -66), (140, -69), (160, -66), (180, -61),
    (180, -90), (-180, -90)
)


# Eurasia.
add_land(
    (-10, 36), (-8, 43), (-3, 44), (2, 48), (8, 50),
    (13, 54), (20, 57), (28, 60), (38, 64), (50, 67),
    (65, 69), (80, 72), (95, 74), (110, 75), (125, 73),
    (140, 70), (155, 68), (170, 65), (179, 61), (178, 55),
    (170, 52), (160, 50), (150, 47), (145, 43), (140, 39),
    (135, 36), (130, 33), (125, 30), (120, 26), (115, 22),
    (110, 18), (105, 15), (100, 11), (95, 9), (90, 8),
    (85, 9), (80, 8), (76, 11), (72, 18), (68, 22),
    (60, 24), (54, 27), (48, 30), (42, 34), (37, 37),
    (32, 40), (28, 42), (23, 42), (18, 40), (12, 38),
    (6, 36), (0, 36), (-5, 35)
)


# Scandinavia and the northern European peninsula.
add_land(
    (4, 57), (5, 64), (7, 70), (13, 72), (20, 71),
    (27, 70), (32, 67), (31, 62), (27, 60), (20, 58),
    (13, 57)
)


# Arabian Peninsula.
add_land(
    (34, 30), (40, 31), (48, 29), (56, 25), (56, 18),
    (52, 13), (46, 12), (40, 16), (36, 22)
)


# India.
add_land(
    (68, 23), (72, 27), (77, 30), (83, 29), (88, 27),
    (93, 24), (95, 19), (91, 15), (87, 12), (82, 8),
    (76, 8), (73, 13), (70, 18)
)


# Indochina.
add_land(
    (97, 22), (102, 25), (108, 22), (108, 17), (105, 12),
    (101, 9), (97, 10), (94, 16)
)


# Korea and the eastern mainland extensions.
add_land(
    (124, 39), (129, 43), (132, 42), (130, 36), (126, 34)
)

add_land(
    (104, 30), (111, 32), (120, 30), (124, 26),
    (120, 22), (113, 21), (108, 24)
)


# Africa.
add_land(
    (-17, 15), (-15, 21), (-10, 28), (-5, 35), (5, 37),
    (12, 37), (20, 33), (30, 33), (35, 30), (40, 20),
    (43, 12), (50, 2), (45, -5), (42, -12), (35, -20),
    (32, -30), (25, -35), (16, -35), (10, -34), (3, -30),
    (-2, -25), (-8, -18), (-12, -5), (-16, 5)
)


# North America, mainly Canada, the United States and Alaska.
add_land(
    (-168, 55), (-160, 60), (-150, 62), (-140, 68),
    (-130, 70), (-120, 72), (-110, 70), (-100, 72),
    (-90, 70), (-80, 72), (-65, 68), (-55, 62), (-52, 55),
    (-60, 50), (-65, 45), (-70, 43), (-75, 40), (-80, 32),
    (-82, 25), (-88, 25), (-92, 28), (-98, 29), (-105, 28),
    (-110, 30), (-115, 32), (-120, 35), (-125, 40),
    (-130, 45), (-140, 50), (-150, 52)
)


# Mexico and Central America.
add_land(
    (-117, 32), (-110, 30), (-105, 28), (-97, 25),
    (-90, 21), (-88, 18), (-92, 16), (-100, 17),
    (-108, 21), (-115, 25)
)

add_land(
    (-92, 18), (-88, 18), (-84, 14), (-80, 9),
    (-84, 10), (-90, 14), (-94, 17)
)


# South America.
add_land(
    (-81, 12), (-75, 12), (-70, 10), (-62, 10), (-55, 6),
    (-50, 2), (-45, 0), (-40, -5), (-35, -10), (-35, -20),
    (-38, -25), (-42, -32), (-48, -38), (-55, -43),
    (-65, -52), (-70, -55), (-74, -52), (-75, -45),
    (-72, -38), (-75, -30), (-78, -20), (-80, -10)
)


# Australia.
add_land(
    (113, -21), (115, -16), (122, -14), (130, -12),
    (137, -12), (145, -14), (153, -20), (154, -25),
    (151, -31), (153, -35), (147, -39), (140, -38),
    (135, -35), (128, -34), (123, -35), (117, -34),
    (114, -30), (112, -25)
)


# -----------------------------
# Major islands and island groups
# -----------------------------

ISLAND_POLYGONS = []

def add_island(*points):
    ISLAND_POLYGONS.append(polygon(*points))


# Greenland.
add_island(
    (-73, 60), (-63, 58), (-48, 60), (-32, 66), (-23, 73),
    (-21, 80), (-30, 84), (-45, 85), (-58, 82), (-68, 76),
    (-73, 68)
)

# Canadian Arctic islands.
add_island(
    (-84, 62), (-75, 61), (-66, 66), (-63, 72),
    (-70, 75), (-80, 73), (-85, 68)
)

add_island(
    (-118, 68), (-105, 68), (-99, 73), (-104, 77),
    (-115, 77), (-122, 73)
)

add_island(
    (-97, 68), (-88, 67), (-83, 72), (-87, 76),
    (-97, 76), (-103, 72)
)

# Newfoundland.
add_island(
    (-59, 47), (-54, 47), (-52, 50), (-55, 52),
    (-60, 51), (-62, 49)
)

# Cuba, Hispaniola and Puerto Rico.
add_island(
    (-85, 23), (-80, 23), (-75, 22), (-74, 20),
    (-78, 20), (-84, 21)
)

add_island(
    (-74, 20), (-69, 20), (-68, 18), (-72, 17),
    (-75, 19)
)

add_island(
    (-67, 18), (-65, 18), (-65, 17), (-67, 17)
)

# Greenland-adjacent and North Atlantic islands.
ISLAND_POLYGONS.append(ellipse(-19, 65, 5.5, 2.8, 28))
ISLAND_POLYGONS.append(ellipse(-8, 71, 0.8, 1.4, 16))
ISLAND_POLYGONS.append(ellipse(18, 79, 5.0, 3.0, 28))
ISLAND_POLYGONS.append(ellipse(-25, 39, 1.5, 1.0, 18))
ISLAND_POLYGONS.append(ellipse(-17, 29, 1.3, 1.0, 18))
ISLAND_POLYGONS.append(ellipse(-25, 15, 1.0, 0.8, 16))

# British Isles and Ireland.
add_island(
    (-11, 51), (-9, 55), (-6, 59), (-2, 59), (1, 55),
    (0, 51), (-3, 50), (-7, 50)
)

add_island(
    (-10, 51), (-8, 54), (-6, 55), (-7, 52), (-9, 51)
)

# Mediterranean islands.
ISLAND_POLYGONS.append(ellipse(12, 38, 1.4, 0.8, 18))
ISLAND_POLYGONS.append(ellipse(15, 37, 1.0, 0.5, 16))
ISLAND_POLYGONS.append(ellipse(33, 35, 2.0, 1.0, 20))
ISLAND_POLYGONS.append(ellipse(7, 39, 0.7, 0.5, 16))

# Madagascar.
add_island(
    (43, -12), (49, -13), (51, -19), (49, -25),
    (45, -27), (43, -22), (42, -17)
)

# Arabian and Indian Ocean islands.
ISLAND_POLYGONS.append(ellipse(54, 24, 1.7, 0.8, 18))
ISLAND_POLYGONS.append(ellipse(73, 7, 1.2, 2.0, 20))
ISLAND_POLYGONS.append(ellipse(74, 4, 0.8, 1.1, 18))
ISLAND_POLYGONS.append(ellipse(43, -11, 0.9, 1.0, 16))

# Japan.
ISLAND_POLYGONS.append(ellipse(141, 43, 3.7, 1.8, 24))
ISLAND_POLYGONS.append(ellipse(138, 36, 4.0, 1.4, 26))
ISLAND_POLYGONS.append(ellipse(134, 34, 1.8, 0.7, 18))
ISLAND_POLYGONS.append(ellipse(130, 33, 1.8, 1.0, 18))

# Taiwan.
ISLAND_POLYGONS.append(ellipse(121, 24, 1.0, 2.0, 20))

# Philippines.
ISLAND_POLYGONS.append(ellipse(121, 16, 2.0, 3.5, 24))
ISLAND_POLYGONS.append(ellipse(124, 10, 2.5, 2.8, 24))
ISLAND_POLYGONS.append(ellipse(123, 7, 1.8, 2.2, 20))

# Indonesia.
add_island(
    (95, 5), (100, 5), (105, 1), (104, -5),
    (100, -5), (96, -1)
)

add_island(
    (105, -6), (112, -6), (115, -8), (114, -9),
    (108, -9), (105, -8)
)

add_island(
    (108, 7), (117, 7), (119, 2), (116, -2),
    (110, -1), (107, 3)
)

add_island(
    (120, 1), (125, 4), (124, -2), (120, -5),
    (118, -2)
)

# New Guinea.
add_island(
    (130, -2), (138, -1), (146, -2), (153, -4),
    (153, -9), (147, -10), (141, -8), (136, -6),
    (132, -5)
)

# Tasmania and New Zealand.
ISLAND_POLYGONS.append(ellipse(147, -42, 3.5, 2.5, 22))
ISLAND_POLYGONS.append(ellipse(173, -39, 2.3, 7.0, 24))
ISLAND_POLYGONS.append(ellipse(168, -44, 2.2, 3.5, 20))

# Pacific islands.
ISLAND_POLYGONS.append(ellipse(-155, 20, 1.2, 0.8, 18))
ISLAND_POLYGONS.append(ellipse(165, -21, 2.2, 1.2, 20))
ISLAND_POLYGONS.append(ellipse(167, -16, 1.8, 2.4, 20))
ISLAND_POLYGONS.append(ellipse(160, -9, 1.8, 1.6, 20))
ISLAND_POLYGONS.append(ellipse(178, -18, 1.8, 1.2, 20))
ISLAND_POLYGONS.append(ellipse(145, 13, 0.8, 0.7, 18))
ISLAND_POLYGONS.append(ellipse(135, 7, 0.8, 0.7, 18))
ISLAND_POLYGONS.append(ellipse(149, -17, 0.8, 0.7, 18))

# South Atlantic islands.
ISLAND_POLYGONS.append(ellipse(-59, -51, 2.3, 1.2, 18))
ISLAND_POLYGONS.append(ellipse(-91, -1, 2.0, 1.4, 20))


# -----------------------------
# Handwritten marine holes
# -----------------------------

WATER_HOLES = []

def add_water(*points):
    WATER_HOLES.append(polygon(*points))


# Hudson Bay and James Bay.
add_water(
    (-98, 57), (-95, 52), (-86, 51), (-79, 54),
    (-77, 60), (-81, 64), (-89, 65), (-96, 62)
)

add_water(
    (-84, 51), (-78, 52), (-78, 56), (-82, 58),
    (-86, 56)
)

# Gulf of Mexico.
add_water(
    (-98, 18), (-91, 18), (-84, 22), (-80, 27),
    (-85, 30), (-93, 29), (-98, 25)
)

# Mediterranean Sea.
add_water(
    (-7, 35), (0, 35), (10, 36), (20, 38),
    (30, 37), (36, 34), (36, 30), (27, 30),
    (17, 31), (7, 31), (-3, 32)
)

# Black Sea.
add_water(
    (27, 42), (34, 41), (42, 43), (42, 47),
    (35, 47), (28, 45)
)

# Baltic Sea.
add_water(
    (10, 54), (16, 54), (24, 56), (27, 60),
    (22, 61), (16, 59), (11, 57)
)

# Persian Gulf.
add_water(
    (48, 30), (56, 27), (55, 24), (51, 24),
    (48, 26), (46, 29)
)

# Red Sea.
add_water(
    (33, 29), (38, 29), (43, 20), (43, 12),
    (40, 12), (37, 20)
)

# Bay of Bengal and Arabian Sea indentation.
add_water(
    (80, 20), (87, 20), (94, 17), (94, 10),
    (88, 9), (82, 12)
)

# Java Sea and parts of the South China Sea.
add_water(
    (105, 0), (115, 0), (122, 2), (124, 7),
    (116, 8), (108, 6)
)

add_water(
    (115, 12), (123, 12), (127, 17), (124, 22),
    (117, 20)
)


def is_land(lon, lat):
    # Islands are tested first so that island polygons remain land even
    # when they are close to a broad marine hole.
    for shape in ISLAND_POLYGONS:
        if point_in_polygon(lon, lat, shape):
            return 1

    in_land = False
    for shape in LAND_POLYGONS:
        if point_in_polygon(lon, lat, shape):
            in_land = True
            break

    if not in_land:
        return 0

    for shape in WATER_HOLES:
        if point_in_polygon(lon, lat, shape):
            return 0

    return 1


# -----------------------------
# Generate matrix and outputs
# -----------------------------

def main():
    latitudes = list(range(89, -90, -2))
    longitudes = list(range(-179, 180, 2))

    matrix = np.zeros((len(latitudes), len(longitudes)), dtype=np.uint8)

    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            matrix[row, col] = is_land(lon, lat)

    assert matrix.shape == (90, 180)
    assert matrix.size == 16200
    assert set(np.unique(matrix).tolist()).issubset({0, 1})

    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "candidate.csv"
    raw_png_path = output_dir / "candidate.png"
    enlarged_png_path = output_dir / "candidate_8x.png"

    # No header and no index.
    np.savetxt(
        csv_path,
        matrix,
        fmt="%d",
        delimiter=",",
        newline="\n",
    )

    # Land is white; water is black.
    image_array = (matrix * 255).astype(np.uint8)
    raw_image = Image.fromarray(image_array, mode="L")

    assert raw_image.size == (180, 90)
    assert set(raw_image.getdata()).issubset({0, 255})

    raw_image.save(raw_png_path, format="PNG")

    enlarged_image = raw_image.resize(
        (1440, 720),
        resample=Image.Resampling.NEAREST,
    )

    assert enlarged_image.size == (1440, 720)
    assert set(enlarged_image.getdata()).issubset({0, 255})

    enlarged_image.save(enlarged_png_path, format="PNG")

    print(f"matrix: {matrix.shape[0]} rows x {matrix.shape[1]} columns")
    print(f"values: {matrix.size}")
    print(f"land cells: {int(matrix.sum())}")
    print(f"water cells: {int(matrix.size - matrix.sum())}")
    print(f"wrote: {csv_path}")
    print(f"wrote: {raw_png_path}")
    print(f"wrote: {enlarged_png_path}")


if __name__ == "__main__":
    main()
