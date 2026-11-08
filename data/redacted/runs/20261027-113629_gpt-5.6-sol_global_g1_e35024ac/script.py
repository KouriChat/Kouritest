from pathlib import Path
import csv

import numpy as np
from PIL import Image


# Sampling coordinates:
# rows: 89, 87, ..., -89
# columns: -179, -177, ..., 179
LATITUDES = np.arange(89, -90, -2, dtype=float)
LONGITUDES = np.arange(-179, 180, 2, dtype=float)


def point_in_polygon(x, y, polygon):
    """Ray-casting test for a polygon expressed as (longitude, latitude)."""
    inside = False
    j = len(polygon) - 1

    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # Treat points on an edge as inside.
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


def polygon_mask(polygon):
    mask = np.zeros((len(LATITUDES), len(LONGITUDES)), dtype=bool)
    min_lon = min(p[0] for p in polygon)
    max_lon = max(p[0] for p in polygon)
    min_lat = min(p[1] for p in polygon)
    max_lat = max(p[1] for p in polygon)

    candidate_rows = np.where(
        (LATITUDES >= min_lat) & (LATITUDES <= max_lat)
    )[0]
    candidate_cols = np.where(
        (LONGITUDES >= min_lon) & (LONGITUDES <= max_lon)
    )[0]

    for row in candidate_rows:
        lat = LATITUDES[row]
        for col in candidate_cols:
            lon = LONGITUDES[col]
            if point_in_polygon(lon, lat, polygon):
                mask[row, col] = True

    return mask


def ellipse_mask(center_lon, center_lat, radius_lon, radius_lat):
    lon_term = ((LONGITUDES[None, :] - center_lon) / radius_lon) ** 2
    lat_term = ((LATITUDES[:, None] - center_lat) / radius_lat) ** 2
    return lon_term + lat_term <= 1.0


def build_matrix():
    land = np.zeros((90, 180), dtype=bool)

    # Hand-authored approximate land polygons.
    polygons = [
        # North American mainland
        [
            (-168, 72), (-157, 72), (-147, 70), (-139, 69),
            (-130, 70), (-123, 73), (-112, 75), (-102, 74),
            (-95, 71), (-88, 69), (-82, 65), (-73, 62),
            (-64, 60), (-58, 55), (-57, 51), (-63, 48),
            (-68, 47), (-67, 44), (-72, 42), (-75, 39),
            (-76, 35), (-81, 30), (-80, 25), (-83, 24),
            (-86, 29), (-91, 29), (-97, 26), (-96, 21),
            (-91, 19), (-87, 21), (-87, 17), (-84, 15),
            (-83, 10), (-78, 9), (-82, 8), (-85, 10),
            (-87, 13), (-92, 15), (-97, 16), (-103, 19),
            (-107, 23), (-111, 24), (-112, 29), (-117, 32),
            (-122, 37), (-124, 43), (-125, 49), (-132, 54),
            (-138, 58), (-145, 59), (-152, 57), (-160, 59),
            (-166, 64),
        ],
        # Baja California
        [
            (-117, 32), (-114, 31), (-111, 24),
            (-109, 22), (-112, 27), (-115, 30),
        ],
        # Greenland
        [
            (-73, 60), (-62, 60), (-49, 62), (-42, 66),
            (-29, 70), (-20, 76), (-24, 82), (-42, 84),
            (-58, 82), (-68, 77), (-73, 70),
        ],
        # Baffin Island
        [
            (-90, 64), (-84, 70), (-80, 74), (-68, 74),
            (-61, 69), (-65, 63), (-75, 62),
        ],
        # Ellesmere Island
        [
            (-86, 76), (-78, 82), (-62, 83), (-59, 78),
            (-69, 76),
        ],
        # Victoria Island
        [
            (-116, 69), (-108, 73), (-98, 73),
            (-96, 68), (-105, 67),
        ],
        # Banks Island
        [
            (-126, 70), (-124, 75), (-116, 76),
            (-111, 72), (-117, 69),
        ],
        # Southampton Island
        [
            (-87, 62), (-84, 66), (-78, 65), (-80, 62),
        ],
        # Newfoundland
        [
            (-59, 47), (-54, 47), (-53, 51),
            (-57, 52), (-60, 50),
        ],
        # Cuba
        [
            (-85, 23), (-80, 23), (-74, 20),
            (-77, 19), (-83, 21),
        ],
        # Hispaniola
        [
            (-74, 20), (-68, 19), (-69, 18), (-73, 18),
        ],
        # South America
        [
            (-79, 10), (-74, 12), (-67, 11), (-61, 9),
            (-55, 6), (-50, 2), (-44, 0), (-37, -4),
            (-35, -9), (-38, -15), (-40, -20), (-45, -24),
            (-49, -29), (-54, -34), (-58, -38), (-64, -42),
            (-67, -48), (-68, -55), (-73, -56), (-76, -51),
            (-75, -45), (-73, -38), (-72, -31), (-71, -23),
            (-75, -17), (-76, -12), (-81, -6), (-80, 1),
            (-77, 7),
        ],
        # Iberian Peninsula
        [
            (-10, 36), (-9, 43), (-2, 44), (3, 42),
            (0, 38), (-5, 36),
        ],
        # Continental Europe
        [
            (-5, 43), (-5, 48), (0, 51), (5, 53),
            (9, 55), (14, 54), (19, 55), (22, 57),
            (27, 59), (30, 64), (35, 68), (42, 66),
            (48, 61), (51, 55), (49, 51), (44, 49),
            (40, 46), (35, 45), (30, 46), (29, 42),
            (26, 40), (23, 39), (20, 41), (17, 42),
            (14, 45), (10, 44), (7, 43), (3, 43),
        ],
        # Scandinavia and Finland
        [
            (5, 58), (7, 62), (12, 67), (17, 71),
            (25, 71), (30, 69), (32, 65), (29, 60),
            (25, 57), (20, 56), (15, 57), (11, 59),
        ],
        # Great Britain
        [
            (-6, 50), (-5, 56), (-3, 59), (1, 58),
            (0, 53), (2, 51),
        ],
        # Ireland
        [
            (-11, 51), (-10, 55), (-6, 56),
            (-5, 52), (-8, 51),
        ],
        # Italy
        [
            (7, 44), (12, 46), (14, 43), (16, 40),
            (18, 39), (16, 37), (13, 39), (11, 43),
        ],
        # Greece and southern Balkans
        [
            (19, 42), (23, 42), (27, 40), (25, 37),
            (22, 36), (21, 39),
        ],
        # Sicily
        [
            (12, 39), (16, 39), (15, 37), (12, 37),
        ],
        # Main Asian landmass
        [
            (28, 69), (38, 71), (51, 72), (65, 74),
            (82, 77), (101, 77), (121, 75), (143, 73),
            (160, 70), (178, 67), (179, 64), (171, 61),
            (164, 60), (161, 55), (151, 59), (144, 57),
            (140, 52), (135, 48), (131, 43), (127, 42),
            (129, 38), (126, 34), (122, 39), (120, 32),
            (122, 25), (116, 22), (109, 21), (106, 17),
            (109, 11), (105, 8), (104, 2), (101, 1),
            (99, 7), (98, 15), (94, 18), (92, 22),
            (88, 22), (85, 20), (81, 16), (79, 9),
            (76, 7), (73, 14), (72, 19), (67, 24),
            (62, 25), (58, 27), (54, 27), (51, 29),
            (48, 30), (45, 35), (41, 37), (36, 36),
            (30, 40), (27, 41), (31, 43), (37, 43),
            (42, 47), (49, 51), (51, 57), (46, 62),
            (38, 66),
        ],
        # Chukotka across the antimeridian
        [
            (-180, 64), (-174, 64), (-169, 66),
            (-171, 69), (-180, 72),
        ],
        # Arabian Peninsula
        [
            (35, 32), (39, 31), (44, 31), (48, 29),
            (50, 26), (54, 24), (56, 26), (59, 24),
            (59, 20), (55, 17), (52, 13), (46, 12),
            (43, 16), (40, 21), (37, 27),
        ],
        # Sri Lanka
        [
            (79, 10), (82, 9), (82, 6), (80, 5), (79, 7),
        ],
        # Sumatra
        [
            (94, 6), (98, 6), (105, -5), (104, -7),
            (100, -4), (96, 1),
        ],
        # Java
        [
            (105, -6), (114, -8), (113, -9), (106, -8),
        ],
        # Borneo
        [
            (109, 7), (116, 7), (119, 3), (118, -3),
            (113, -4), (109, 0),
        ],
        # Sulawesi
        [
            (119, 2), (122, 1), (121, -3), (125, -2),
            (124, -5), (121, -5), (120, -2),
        ],
        # New Guinea
        [
            (130, -3), (137, -2), (143, -4), (151, -6),
            (150, -10), (143, -9), (137, -7), (131, -8),
        ],
        # Japan - Honshu, Shikoku, Kyushu
        [
            (129, 31), (132, 34), (136, 35), (140, 40),
            (142, 41), (141, 36), (137, 33), (132, 31),
        ],
        # Hokkaido
        [
            (139, 41), (145, 44), (144, 46),
            (141, 45),
        ],
        # Sakhalin
        [
            (142, 46), (144, 51), (143, 54),
            (141, 50),
        ],
        # Taiwan
        [
            (120, 22), (121, 26), (123, 25), (122, 22),
        ],
        # Luzon
        [
            (119, 14), (121, 19), (123, 18),
            (123, 14), (121, 12),
        ],
        # Mindanao
        [
            (124, 10), (127, 9), (126, 6), (123, 6),
        ],
        # Africa
        [
            (-17, 37), (-7, 36), (0, 37), (10, 37),
            (12, 33), (20, 32), (25, 31), (32, 31),
            (34, 27), (36, 22), (40, 16), (44, 12),
            (51, 11), (49, 7), (43, 3), (40, -4),
            (41, -11), (38, -18), (35, -24), (32, -30),
            (27, -34), (20, -35), (16, -33), (12, -29),
            (10, -21), (7, -12), (4, -5), (-2, 3),
            (-8, 6), (-15, 11), (-17, 20), (-17, 28),
        ],
        # Madagascar
        [
            (48, -12), (50, -16), (49, -25),
            (46, -26), (44, -20), (45, -15),
        ],
        # Australia
        [
            (113, -22), (114, -15), (121, -14),
            (128, -14), (131, -11), (136, -12),
            (138, -17), (142, -11), (145, -15),
            (147, -20), (153, -27), (153, -31),
            (148, -38), (141, -39), (136, -36),
            (130, -33), (124, -34), (116, -35),
            (113, -29),
        ],
        # Tasmania
        [
            (144, -40), (149, -41), (148, -44), (145, -44),
        ],
        # New Zealand, South Island
        [
            (166, -47), (170, -45), (174, -41),
            (172, -40), (168, -43),
        ],
        # New Zealand, North Island
        [
            (173, -41), (177, -39), (178, -36),
            (175, -35), (173, -38),
        ],
    ]

    for polygon in polygons:
        land |= polygon_mask(polygon)

    # Small islands and archipelagos represented by simple ellipses.
    islands = [
        (-19, 65, 4.5, 2.7),    # Iceland
        (20, 78, 5.0, 2.5),     # Svalbard
        (-8, 62, 1.2, 1.2),     # Faroe Islands
        (9, 40, 2.0, 2.2),      # Sardinia
        (9, 42, 1.5, 2.0),      # Corsica
        (25, 35, 3.5, 1.2),     # Crete
        (33, 35, 2.0, 1.0),     # Cyprus
        (3, 39, 2.0, 1.2),      # Balearic Islands
        (-66, 18, 1.5, 1.0),    # Puerto Rico
        (-77, 18, 1.6, 1.0),    # Jamaica
        (-79, 25, 2.0, 2.0),    # Bahamas
        (-61, 15, 1.2, 2.5),    # Lesser Antilles
        (-60, -52, 2.5, 1.5),   # Falkland Islands
        (-27, 39, 1.0, 1.0),    # Azores
        (-16, 28, 1.0, 1.5),    # Canary Islands
        (-24, 16, 1.2, 1.5),    # Cape Verde
        (73, 4, 1.0, 2.0),      # Maldives
        (121, 10, 2.0, 3.0),    # Central Philippines
        (126, 1, 1.5, 2.5),     # Moluccas
        (128, -4, 1.5, 2.0),    # Moluccas
        (124, -9, 4.0, 1.2),    # Lesser Sunda Islands
        (130, -9, 2.5, 1.2),    # Timor region
        (157, -8, 3.0, 1.5),    # Solomon Islands
        (166, -16, 2.0, 3.0),   # Vanuatu
        (165, -21, 1.3, 3.0),   # New Caledonia
        (178, -17, 1.5, 2.5),   # Fiji
        (-155, 19, 1.4, 1.3),   # Hawaii, Big Island
        (-157, 21, 1.2, 1.2),   # Hawaiian Islands
        (147, 45, 1.2, 1.5),    # Kuril Islands
        (150, 47, 1.2, 1.8),
        (153, 50, 1.2, 1.8),
        (-172, 52, 2.0, 1.0),   # Aleutian Islands
        (-166, 53, 2.0, 1.0),
        (-160, 55, 2.0, 1.0),
    ]

    for island in islands:
        land |= ellipse_mask(*island)

    # Aleutians on the eastern side of the antimeridian.
    for lon, lat in [
        (178, 52), (173, 52), (168, 53), (163, 54), (158, 55)
    ]:
        land |= ellipse_mask(lon, lat, 1.6, 0.9)

    # Antarctica: longitude-dependent approximate grounded coastline.
    antarctic_lon = np.array(
        [
            -180, -160, -140, -120, -100, -82, -72, -66, -62,
            -58, -52, -45, -25, 0, 30, 60, 90, 120, 150, 180
        ],
        dtype=float,
    )
    antarctic_coast = np.array(
        [
            -72, -75, -75, -73, -72, -74, -70, -66, -63,
            -64, -68, -74, -72, -70, -69, -67, -66, -66, -68, -72
        ],
        dtype=float,
    )
    coast_at_columns = np.interp(
        LONGITUDES, antarctic_lon, antarctic_coast
    )
    land |= LATITUDES[:, None] <= coast_at_columns[None, :]

    # Ocean-connected bays and seas removed from otherwise broad polygons.
    water_polygons = [
        # Hudson Bay and James Bay
        [
            (-95, 64), (-87, 66), (-80, 63), (-77, 58),
            (-80, 52), (-84, 50), (-89, 53), (-93, 57),
        ],
        # Black Sea
        [
            (27, 46), (34, 47), (40, 44), (42, 42),
            (39, 40), (31, 41), (28, 43),
        ],
        # Baltic Sea and Gulf of Bothnia
        [
            (10, 55), (13, 57), (17, 59), (20, 63),
            (23, 66), (27, 65), (29, 61), (25, 58),
            (21, 56), (18, 54), (14, 54),
        ],
        # Red Sea
        [
            (32, 30), (35, 29), (39, 22), (44, 13),
            (42, 12), (38, 18), (34, 25),
        ],
        # Persian Gulf
        [
            (48, 30), (51, 30), (56, 27), (57, 25),
            (55, 24), (51, 26), (49, 28),
        ],
        # Gulf of California
        [
            (-115, 31), (-112, 29), (-109, 24),
            (-110, 22), (-113, 25), (-116, 30),
        ],
    ]

    for polygon in water_polygons:
        land &= ~polygon_mask(polygon)

    # Restore a few islands situated inside coarse sea masks.
    land |= ellipse_mask(20, 57, 1.3, 1.3)   # Gotland
    land |= ellipse_mask(24, 60, 1.0, 1.0)   # Baltic archipelago approximation

    return land.astype(np.uint8)


def save_outputs(matrix):
    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "candidate.csv"
    raw_png_path = output_dir / "candidate.png"
    large_png_path = output_dir / "candidate_8x.png"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(matrix.tolist())

    pixels = (matrix * 255).astype(np.uint8)
    raw_image = Image.fromarray(pixels, mode="L")
    raw_image.save(raw_png_path)

    try:
        nearest = Image.Resampling.NEAREST
    except AttributeError:
        nearest = Image.NEAREST

    large_image = raw_image.resize((1440, 720), resample=nearest)
    large_image.save(large_png_path)

    return csv_path, raw_png_path, large_png_path


def validate(matrix, paths):
    csv_path, raw_png_path, large_png_path = paths

    assert matrix.shape == (90, 180)
    assert matrix.size == 16200
    assert set(np.unique(matrix).tolist()) <= {0, 1}

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert len(rows) == 90
    assert all(len(row) == 180 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    with Image.open(raw_png_path) as image:
        raw = np.asarray(image.convert("L"))
        assert image.size == (180, 90)

    with Image.open(large_png_path) as image:
        enlarged = np.asarray(image.convert("L"))
        assert image.size == (1440, 720)

    assert set(np.unique(raw).tolist()) <= {0, 255}
    assert set(np.unique(enlarged).tolist()) <= {0, 255}
    assert np.array_equal(raw, matrix * 255)
    assert np.array_equal(
        enlarged,
        np.repeat(np.repeat(raw, 8, axis=0), 8, axis=1),
    )


def main():
    matrix = build_matrix()
    paths = save_outputs(matrix)
    validate(matrix, paths)

    print("Generated and validated:")
    for path in paths:
        print(path)
    print(f"Matrix: {matrix.shape[0]}x{matrix.shape[1]} = {matrix.size} values")
    print(f"Land samples: {int(matrix.sum())}")
    print(f"Water samples: {int(matrix.size - matrix.sum())}")


if __name__ == "__main__":
    main()
