import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image


# Coordinates are (longitude, latitude), in WGS84 decimal degrees.
# These hand-written outlines are deliberately approximate.
EUROPE = [
    (-12, 48), (40, 48), (40, 42.4), (37, 42.0), (34, 41.7),
    (31, 41.3), (29, 41.1), (28.5, 40.8), (27.5, 40.6),
    (26.5, 40.3), (26.0, 39.8), (25.5, 39.1), (25.0, 38.7),
    (24.4, 38.3), (23.8, 38.0), (23.4, 37.6), (23.0, 37.1),
    (22.4, 36.7), (21.8, 36.4), (21.2, 36.5), (20.5, 37.0),
    (20.0, 37.7), (19.5, 38.5), (19.4, 39.3), (19.2, 40.1),
    (19.0, 40.8), (18.5, 41.2), (18.0, 41.5), (17.5, 42.0),
    (17.0, 42.5), (16.5, 43.0), (16.0, 43.5), (16.2, 44.0),
    (15.5, 44.5), (15.0, 45.0), (14.0, 45.5), (13.0, 46.0),
    (12.0, 46.5), (10.0, 47.0), (7.0, 47.5), (4.0, 48.0),
    (-12, 48),
]

AFRICA = [
    (-12, 30), (40, 30), (40, 34), (37, 34), (35.5, 33.2),
    (34.9, 32.5), (34.8, 31.8), (34.5, 31.3), (33.5, 31.2),
    (32.5, 31.3), (31.5, 31.5), (30.5, 31.7), (29.5, 31.9),
    (28.0, 32.0), (26.0, 32.5), (24.0, 32.8), (22.0, 32.9),
    (20.0, 32.7), (18.0, 32.5), (16.0, 32.5), (14.0, 32.5),
    (12.0, 32.3), (10.0, 32.0), (8.0, 33.0), (6.0, 35.0),
    (4.0, 36.0), (2.0, 36.7), (0.0, 36.8), (-2.0, 36.5),
    (-4.0, 36.0), (-5.6, 35.9), (-7.0, 35.8), (-9.0, 36.0),
    (-12, 36.0), (-12, 30),
]

ASIA = [
    (25.0, 42.0), (40.0, 42.0), (40.0, 30.0), (35.0, 30.0),
    (35.5, 31.0), (35.0, 31.3), (34.5, 31.5), (34.0, 32.0),
    (33.5, 33.0), (32.5, 34.0), (31.5, 35.0), (30.5, 36.0),
    (29.0, 36.5), (27.0, 36.5), (26.0, 37.0), (26.0, 38.0),
    (26.0, 39.0), (26.5, 40.0), (27.5, 40.5), (28.5, 40.8),
    (29.0, 41.2), (31.0, 41.5), (34.0, 41.7), (37.0, 42.0),
    (40.0, 42.0),
]

# Approximate connected Mediterranean water area, including major gulfs.
MEDITERRANEAN = [
    (-6.0, 36.2), (-5.0, 36.0), (-3.5, 36.0), (-1.0, 36.2),
    (1.0, 36.5), (3.0, 36.5), (5.0, 36.2), (7.0, 36.0),
    (9.0, 36.0), (11.0, 36.0), (12.0, 36.5), (13.0, 37.2),
    (13.5, 38.0), (14.5, 38.2), (15.5, 38.0), (16.0, 37.5),
    (16.5, 37.0), (17.0, 36.5), (18.0, 36.2), (19.0, 35.8),
    (20.0, 35.4), (21.0, 35.0), (22.0, 34.8), (23.0, 34.8),
    (24.0, 35.0), (25.0, 35.5), (26.0, 35.8), (27.0, 36.0),
    (28.0, 36.0), (29.0, 35.8), (30.0, 35.5), (31.0, 35.0),
    (32.0, 34.5), (33.0, 34.0), (34.0, 33.5), (35.0, 33.0),
    (36.0, 32.5), (37.0, 32.0), (38.0, 31.0), (39.0, 30.0),
    (40.0, 30.0), (40.0, 45.0), (-9.0, 45.0),
    (-9.0, 38.0), (-7.0, 37.0), (-6.0, 36.2),
]

# Additional sea cuts refine the Adriatic and Aegean where the broad
# Mediterranean outline would otherwise leave too much land.
ADRIATIC = [
    (12.0, 45.8), (19.0, 42.0), (19.7, 40.0), (19.0, 39.0),
    (18.0, 40.0), (16.5, 41.5), (15.5, 43.0), (14.0, 44.5),
    (12.0, 45.8),
]
AEGEAN = [
    (22.0, 41.0), (29.0, 41.0), (28.0, 39.0), (27.0, 37.0),
    (25.0, 35.0), (23.0, 35.0), (21.0, 36.0), (20.0, 38.0),
    (22.0, 41.0),
]

# Major islands as hand-drawn coarse polygons, followed by compact ellipse
# rules for smaller island groups.
ISLAND_POLYGONS = [
    # Balearic Islands
    [(-1.0, 39.8), (0.3, 39.7), (1.6, 39.4), (2.0, 39.0),
     (1.4, 38.7), (0.3, 38.7), (-0.8, 39.0), (-1.0, 39.8)],
    # Corsica
    [(8.5, 42.9), (9.5, 42.9), (9.6, 42.0), (9.4, 41.4),
     (9.0, 41.3), (8.7, 41.8), (8.5, 42.9)],
    # Sardinia
    [(8.1, 41.3), (9.7, 41.2), (9.8, 40.0), (9.4, 39.0),
     (8.5, 38.9), (8.0, 39.8), (8.1, 41.3)],
    # Sicily
    [(12.4, 38.2), (15.7, 38.2), (15.3, 37.0), (14.0, 36.6),
     (12.5, 37.0), (12.4, 38.2)],
    # Crete
    [(23.4, 35.7), (26.4, 35.7), (26.3, 35.0), (24.8, 34.9),
     (23.5, 35.1), (23.4, 35.7)],
    # Cyprus
    [(32.2, 35.7), (34.7, 35.7), (34.6, 34.5), (33.0, 34.5),
     (32.2, 35.0), (32.2, 35.7)],
    # Malta
    [(14.1, 36.1), (14.6, 36.1), (14.6, 35.8), (14.2, 35.8),
     (14.1, 36.1)],
    # Rhodes
    [(27.7, 36.5), (28.3, 36.5), (28.2, 35.8), (27.8, 35.8),
     (27.7, 36.5)],
    # Euboea
    [(23.2, 39.1), (24.0, 39.1), (24.0, 38.0), (23.5, 37.8),
     (23.2, 39.1)],
    # Peloponnese
    [(21.0, 37.8), (22.6, 37.8), (22.4, 36.4), (21.5, 36.3),
     (20.9, 37.0), (21.0, 37.8)],
    # Gulf of Corinth separates peninsula from mainland approximately;
    # this polygon is land and is added after the sea cuts.
    [(21.7, 38.4), (22.2, 38.4), (22.1, 37.9), (21.8, 37.9),
     (21.7, 38.4)],
]

# (longitude, latitude, radius_lon, radius_lat), approximate small islands.
ISLAND_ELLIPSES = [
    (12.6, 43.0, 0.30, 0.18),  # Elba
    (10.2, 42.4, 0.16, 0.12),  # Tuscan Archipelago
    (8.3, 39.2, 0.22, 0.14),   # southwestern Sardinian islets
    (15.0, 38.5, 0.20, 0.10),  # Aeolian Islands
    (12.0, 35.5, 0.18, 0.12),  # Lampedusa
    (25.0, 36.4, 0.18, 0.10),  # Cyclades
    (26.0, 37.5, 0.30, 0.14),  # Cyclades
    (24.5, 37.0, 0.18, 0.12),  # Cyclades
    (25.7, 38.8, 0.20, 0.12),  # Sporades
    (26.5, 39.0, 0.18, 0.12),  # Sporades
    (27.1, 36.8, 0.20, 0.12),  # Dodecanese
    (26.9, 36.6, 0.16, 0.10),  # Kos
    (24.0, 35.2, 0.14, 0.10),  # Gavdos
    (19.9, 39.7, 0.16, 0.10),  # Corfu
    (20.7, 38.7, 0.12, 0.16),  # Kefalonia
    (21.7, 38.2, 0.12, 0.10),  # Gulf of Corinth islands
    (18.8, 42.5, 0.10, 0.10),  # Adriatic islands
    (16.0, 42.8, 0.25, 0.10),  # Dalmatian islands
    (17.0, 42.5, 0.25, 0.10),
    (18.0, 42.4, 0.20, 0.10),
]


def point_in_polygon(lon, lat, polygon):
    """Ray-casting point-in-polygon test."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        crosses = (yi > lat) != (yj > lat)
        if crosses:
            x_cross = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_cross:
                inside = not inside
        j = i
    return inside


def is_island(lon, lat):
    if any(point_in_polygon(lon, lat, poly) for poly in ISLAND_POLYGONS):
        return True
    for cx, cy, rx, ry in ISLAND_ELLIPSES:
        if ((lon - cx) / rx) ** 2 + ((lat - cy) / ry) ** 2 <= 1.0:
            return True
    return False


def is_land(lon, lat):
    broad_land = (
        point_in_polygon(lon, lat, EUROPE)
        or point_in_polygon(lon, lat, AFRICA)
        or point_in_polygon(lon, lat, ASIA)
    )

    # Water overrides the broad continental outlines.
    mediterranean_water = (
        point_in_polygon(lon, lat, MEDITERRANEAN)
        or point_in_polygon(lon, lat, ADRIATIC)
        or point_in_polygon(lon, lat, AEGEAN)
    )
    return is_island(lon, lat) or (broad_land and not mediterranean_water)


def main():
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    latitudes = [45.0 - 0.2 * i for i in range(71)]
    longitudes = [-9.0 + 0.2 * i for i in range(231)]

    matrix = np.array(
        [[1 if is_land(lon, lat) else 0 for lon in longitudes]
         for lat in latitudes],
        dtype=np.uint8,
    )

    assert matrix.shape == (71, 231)
    assert int(matrix.sum() + (matrix == 0).sum()) == 16401

    with (out_dir / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # 1 is white land; 0 is black water.
    pixels = matrix * 255
    image = Image.fromarray(pixels, mode="L")
    image.save(out_dir / "candidate.png")

    enlarged = image.resize((231 * 8, 71 * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(out_dir / "candidate_8x.png")

    # Verify CSV dimensions/content and both image sizes/colors.
    with (out_dir / "candidate.csv").open("r", newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 71
    assert all(len(row) == 231 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    original = Image.open(out_dir / "candidate.png")
    zoomed = Image.open(out_dir / "candidate_8x.png")
    assert original.size == (231, 71)
    assert set(original.getdata()) <= {0, 255}
    assert zoomed.size == (1848, 568)
    assert set(zoomed.getdata()) <= {0, 255}

    print("Generated 71x231 matrix (16401 values).")
    print("outputs/candidate.csv")
    print("outputs/candidate.png (231x71)")
    print("outputs/candidate_8x.png (1848x568, nearest-neighbor)")


if __name__ == "__main__":
    main()
