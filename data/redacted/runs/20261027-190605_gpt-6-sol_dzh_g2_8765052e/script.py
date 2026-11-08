"""Knowledge-based Mediterranean land/water reconstruction.

All geometry below is deliberately approximate and hand-written.  No map,
geodata file, network access, or land-mask service is used.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

WEST, EAST = -9.0, 37.0
NORTH, SOUTH = 45.0, 31.0
STEP = 0.2
ROWS, COLS = 71, 231


def polygon_contains(x, y, polygon):
    """Ray-casting point test for one closed polygon."""
    inside = False
    j = len(polygon) - 1
    for i, (px, py) in enumerate(polygon):
        qx, qy = polygon[j]
        crosses = ((py > y) != (qy > y))
        if crosses:
            hit_x = (qx - px) * (y - py) / (qy - py) + px
            if x < hit_x:
                inside = not inside
        j = i
    return inside


def poly(*points):
    return tuple(points)


# Large land blocks.  Their southern/eastern edges follow the remembered
# Mediterranean shoreline; the other edges are safely inside the frame.
LAND_BLOCKS = [
    # Iberian Peninsula
    poly((-9.0, 45.0), (3.2, 45.0), (3.4, 43.8), (3.1, 42.5),
         (2.8, 41.5), (3.1, 41.2), (2.5, 40.6), (1.5, 39.5),
         (0.5, 39.0), (-0.2, 38.8), (-0.8, 37.8), (-1.5, 37.2),
         (-2.8, 36.7), (-5.0, 36.0), (-6.8, 36.0), (-7.6, 36.7),
         (-8.8, 37.0), (-9.0, 38.4), (-9.3, 40.0), (-9.1, 42.0),
         (-9.0, 44.0)),
    # France and the northern Mediterranean coast
    poly((-2.0, 45.0), (7.0, 45.0), (7.6, 44.2), (7.7, 43.8),
         (7.1, 43.7), (6.5, 43.4), (5.6, 43.1), (4.7, 43.2),
         (3.8, 43.3), (3.1, 42.5), (2.2, 42.5), (1.1, 43.0),
         (-0.2, 43.5), (-2.0, 44.0)),
    # Italy, traced down the Adriatic and back up the Tyrrhenian coast
    poly((6.5, 45.0), (13.7, 46.0), (15.0, 45.4), (16.2, 45.0),
         (17.3, 44.5), (18.5, 43.7), (18.5, 42.0), (17.8, 41.4),
         (17.3, 40.5), (16.7, 40.4), (16.2, 39.3), (16.0, 38.0),
         (15.7, 37.5), (15.0, 38.0), (14.6, 39.0), (13.6, 40.0),
         (12.6, 41.0), (12.0, 42.0), (11.0, 43.3), (10.0, 43.7),
         (9.0, 44.0), (7.7, 44.5), (6.8, 44.6)),
]
# Balkans and Greece.  This outline includes the mainland behind the
# irregular Adriatic and Aegean shores.
LAND_BLOCKS += [
    poly((13.0, 45.0), (26.5, 45.0), (26.5, 41.8), (25.7, 41.0),
         (24.8, 40.8), (24.1, 40.5), (23.6, 40.0), (23.0, 39.4),
         (22.5, 38.7), (21.9, 38.0), (21.5, 38.0), (21.0, 39.0),
         (20.4, 39.7), (19.8, 40.5), (19.3, 41.1), (18.8, 42.0),
         (18.1, 42.8), (17.5, 43.5), (16.7, 44.0), (15.5, 44.5)),
    # Southern Greece and the Peloponnese
    poly((20.5, 40.8), (22.2, 41.0), (23.8, 40.8), (25.4, 40.9),
         (26.2, 40.3), (25.6, 39.5), (24.8, 39.0), (24.5, 38.4),
         (23.8, 38.2), (23.5, 37.7), (23.0, 37.5), (22.8, 37.0),
         (22.3, 36.8), (21.8, 37.0), (21.4, 37.5), (20.9, 37.8),
         (20.5, 38.5), (20.0, 39.2), (20.2, 40.1)),
    # Anatolian mainland
    poly((26.0, 45.0), (37.0, 45.0), (37.0, 36.5), (36.5, 36.4),
         (35.8, 36.5), (34.8, 36.2), (33.8, 36.1), (32.5, 36.0),
         (31.0, 36.2), (29.7, 36.4), (28.6, 36.6), (27.5, 36.7),
         (26.8, 37.1), (26.6, 37.7), (26.3, 38.3), (26.6, 38.9),
         (26.2, 39.5), (26.0, 40.2)),
    # Syria, Lebanon, Israel and the eastern edge of the frame
    poly((36.2, 37.0), (37.0, 37.0), (37.0, 31.0), (34.4, 31.0),
         (34.5, 32.0), (34.6, 33.0), (35.0, 33.8), (35.5, 34.7),
         (36.0, 35.6)),
]

# North African mainland.  The upper edge is a coarse west-to-east trace of
# the Maghreb, Gulf of Gabes, Tripolitania, Cyrenaica and Egypt.
LAND_BLOCKS += [
    poly((-9.0, 31.0), (-9.0, 35.1), (-7.0, 35.8), (-5.3, 35.8),
         (-4.8, 35.2), (-3.0, 35.1), (-1.5, 35.2), (0.0, 35.1),
         (1.4, 36.0), (3.0, 36.6), (4.5, 36.8), (5.5, 37.1),
         (7.0, 37.1), (8.5, 37.2), (9.5, 37.0), (10.5, 36.8),
         (11.2, 37.0), (11.4, 36.7), (11.1, 35.9), (10.5, 35.2),
         (10.2, 34.8), (10.5, 34.4), (11.2, 34.0), (12.2, 33.7),
         (13.2, 33.2), (14.5, 32.9), (15.8, 32.8), (17.0, 32.9),
         (18.5, 32.8), (20.0, 32.5), (22.0, 32.3), (24.0, 31.7),
         (25.0, 31.3), (26.0, 31.2), (27.0, 31.0)),
]
# Principal islands and island groups, represented as hand-drawn polygons.
ISLANDS = [
    # Balearics
    poly((2.2, 39.8), (3.5, 39.8), (4.4, 39.9), (4.3, 40.2),
         (3.6, 40.4), (2.7, 40.1)),
    poly((1.2, 38.7), (1.7, 38.7), (1.7, 39.1), (1.2, 39.1)),
    # Corsica and Sardinia
    poly((8.5, 41.4), (9.5, 41.4), (9.6, 42.0), (9.4, 42.8),
         (9.1, 43.0), (8.8, 42.5)),
    poly((8.1, 38.8), (9.8, 38.8), (9.8, 40.9), (9.2, 41.1),
         (8.6, 40.8), (8.2, 40.0)),
    # Sicily, Malta
    poly((12.3, 37.0), (15.2, 37.0), (15.3, 37.6), (14.5, 38.2),
         (13.4, 38.2), (12.5, 37.8)),
    poly((14.1, 35.7), (14.6, 35.7), (14.6, 36.1), (14.1, 36.1)),
    # Crete and nearby southern Aegean islands
    poly((23.4, 34.8), (26.3, 34.8), (26.3, 35.5), (25.2, 35.7),
         (24.0, 35.5)),
    # Ionian islands / Euboea / Cyclades / Dodecanese
    poly((19.7, 39.1), (20.1, 39.1), (20.1, 40.0), (19.7, 40.0)),
    poly((23.2, 38.0), (23.8, 38.0), (23.8, 39.0), (23.2, 39.0)),
    poly((24.0, 36.3), (27.0, 36.3), (27.0, 37.0), (24.0, 37.0)),
    # Cyprus
    poly((32.2, 34.5), (34.7, 34.5), (34.7, 35.7), (33.3, 35.8),
         (32.3, 35.4)),
    # Smaller but frequently sampled Aegean islands
    poly((24.3, 39.8), (25.1, 39.8), (25.1, 40.1), (24.3, 40.1)),
    poly((25.5, 36.4), (26.0, 36.4), (26.0, 36.9), (25.5, 36.9)),
]

# Inland lakes override land.  Coordinates are deliberately approximate.
LAKES = [
    poly((27.7, 40.6), (29.0, 40.6), (29.0, 41.0), (27.7, 41.0)),
    poly((29.5, 38.0), (30.0, 38.0), (30.0, 38.4), (29.5, 38.4)),
]
ISLANDS += [
    # Adriatic islands and the Gulf of Taranto islands
    poly((14.4, 42.7), (15.0, 42.7), (15.0, 43.1), (14.4, 43.1)),
    poly((15.6, 43.5), (16.2, 43.5), (16.2, 44.0), (15.6, 44.0)),
    poly((16.2, 43.0), (16.8, 43.0), (16.8, 43.4), (16.2, 43.4)),
    poly((18.0, 40.0), (18.6, 40.0), (18.6, 40.5), (18.0, 40.5)),
    # Gulf of Lion / Ligurian and Tyrrhenian islands
    poly((5.8, 42.9), (6.3, 42.9), (6.3, 43.2), (5.8, 43.2)),
    poly((9.3, 42.7), (9.8, 42.7), (9.8, 43.1), (9.3, 43.1)),
    poly((10.0, 42.7), (10.6, 42.7), (10.6, 43.0), (10.0, 43.0)),
    poly((12.3, 40.5), (12.7, 40.5), (12.7, 40.8), (12.3, 40.8)),
    poly((12.5, 40.0), (13.0, 40.0), (13.0, 40.5), (12.5, 40.5)),
    # Tiny Mediterranean islands visible at this sampling scale
    poly((7.6, 35.7), (8.1, 35.7), (8.1, 36.1), (7.6, 36.1)),
    poly((26.0, 35.4), (26.5, 35.4), (26.5, 35.8), (26.0, 35.8)),
    poly((27.5, 35.5), (28.2, 35.5), (28.2, 36.1), (27.5, 36.1)),
]


def is_land(lon, lat):
    land = any(polygon_contains(lon, lat, shape)
               for shape in LAND_BLOCKS + ISLANDS)
    if land and any(polygon_contains(lon, lat, lake) for lake in LAKES):
        return 0
    return int(land)


def make_matrix():
    matrix = np.zeros((ROWS, COLS), dtype=np.uint8)
    for row in range(ROWS):
        lat = NORTH - row * STEP
        for col in range(COLS):
            lon = WEST + col * STEP
            matrix[row, col] = is_land(lon, lat)
    return matrix


def write_outputs(matrix):
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Pillow's mode "1" is a strictly two-colour, one-bit source image.
    original = Image.fromarray(np.where(matrix == 1, 255, 0).astype(np.uint8),
                               mode="L")
    original.save(out / "candidate.png")
    enlarged = original.resize((COLS * 8, ROWS * 8), Image.Resampling.NEAREST)
    enlarged.save(out / "candidate_8x.png")


if __name__ == "__main__":
    result = make_matrix()
    write_outputs(result)
