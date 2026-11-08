from pathlib import Path
import csv

import numpy as np
from PIL import Image


# Hand-written approximate mainland and island outlines in (lon, lat).
LAND_POLYGONS = [
    # Iberian Peninsula
    [(-9.5, 36.0), (-9.5, 43.8), (-8.0, 43.7), (-7.0, 43.8),
     (-5.5, 43.5), (-3.0, 43.4), (-1.8, 43.4), (0.0, 42.8),
     (1.5, 42.4), (3.2, 42.4), (3.3, 41.8), (2.5, 41.3),
     (1.4, 41.0), (0.7, 40.6), (0.0, 39.7), (-0.2, 38.7),
     (-0.7, 37.9), (-1.5, 37.2), (-2.0, 36.7), (-5.5, 36.0)],
    # France and continental Europe north of the Mediterranean
    [(-1.8, 43.4), (0.0, 42.8), (1.5, 42.4), (3.2, 42.4),
     (4.5, 43.0), (5.5, 43.2), (6.5, 43.4), (7.5, 43.7),
     (8.5, 43.7), (9.5, 44.1), (10.5, 44.2), (11.5, 44.7),
     (12.5, 45.5), (13.2, 46.5), (15.0, 47.0), (17.0, 48.0),
     (20.0, 48.5), (23.0, 48.5), (25.0, 48.0), (28.0, 46.5),
     (31.0, 46.0), (37.5, 46.0), (37.5, 60.0), (-1.8, 60.0)],
    # Italian Peninsula
    [(6.5, 45.0), (7.0, 44.0), (8.0, 43.8), (9.0, 44.0),
     (10.5, 43.8), (11.5, 43.0), (12.5, 42.5), (13.5, 41.8),
     (15.0, 41.0), (16.5, 40.0), (17.5, 39.8), (18.5, 40.0),
     (17.5, 39.0), (16.5, 38.5), (15.5, 38.2), (16.0, 37.8),
     (15.5, 37.0), (14.5, 38.0), (13.5, 38.2), (12.8, 39.0),
     (12.5, 40.0), (12.0, 41.0), (11.0, 42.0), (10.0, 42.5),
     (9.0, 43.0), (8.0, 44.0)],
    # Western Balkans and Greek mainland
    [(13.0, 46.5), (14.0, 45.5), (15.0, 45.2), (16.0, 45.0),
     (17.5, 44.8), (18.5, 44.0), (19.0, 43.5), (19.5, 42.5),
     (20.0, 41.5), (20.5, 40.8), (21.5, 40.0), (22.5, 39.5),
     (23.5, 38.5), (24.0, 37.5), (23.0, 36.5), (22.0, 36.5),
     (21.0, 37.0), (20.0, 37.5), (19.5, 38.5), (19.0, 39.5),
     (18.5, 40.0), (17.5, 39.8), (16.5, 40.0), (15.0, 41.0),
     (14.0, 42.0), (13.5, 43.0), (13.0, 44.0)],
]
# Anatolia, North Africa, and the Levant
LAND_POLYGONS += [
    [(26.0, 40.7), (27.0, 41.0), (29.0, 41.2), (31.0, 41.3),
     (33.0, 41.5), (35.0, 41.3), (37.5, 41.0), (37.5, 35.8),
     (36.0, 36.0), (34.0, 36.2), (32.0, 36.1), (30.0, 36.2),
     (28.5, 36.5), (27.0, 36.7), (26.0, 37.0), (27.0, 38.0),
     (27.5, 39.0), (26.5, 40.0)],
    [(-9.5, 31.0), (-9.5, 36.2), (-6.0, 35.8), (-4.0, 35.2),
     (-2.0, 35.1), (0.0, 35.5), (2.0, 36.0), (4.0, 36.8),
     (6.0, 37.1), (8.0, 37.0), (10.0, 37.0), (11.0, 37.2),
     (11.5, 33.0), (13.0, 32.8), (15.0, 32.5), (18.0, 32.8),
     (20.0, 32.9), (22.0, 32.7), (24.0, 31.5), (25.0, 31.0)],
    [(34.5, 31.0), (34.5, 33.0), (35.0, 33.5), (36.0, 34.0),
     (36.5, 35.0), (37.5, 35.5), (37.5, 31.0)],
]

# Additional southern Iberian outline where it meets the Mediterranean.
LAND_POLYGONS.append([
    (-6.0, 35.8), (-5.5, 36.0), (-2.0, 36.7), (-1.5, 37.2),
    (-0.7, 37.9), (-0.2, 38.7), (0.0, 39.7), (0.7, 40.6),
    (1.4, 41.0), (2.5, 41.3), (3.3, 41.8), (3.2, 42.4),
    (1.5, 42.4), (0.0, 42.8), (-1.8, 43.4), (-3.0, 43.4),
    (-5.5, 43.5), (-7.0, 43.8), (-8.0, 43.7), (-9.0, 43.5),
    (-9.0, 36.0),
])
ISLAND_POLYGONS = [
    # Balearic Islands
    [(1.2, 38.6), (1.3, 39.0), (2.0, 39.9), (3.4, 40.1),
     (4.3, 39.8), (4.3, 39.3), (3.0, 38.7), (2.0, 38.5)],
    # Corsica and Sardinia
    [(8.5, 41.3), (8.6, 42.0), (8.7, 42.8), (9.3, 43.0),
     (9.6, 42.4), (9.5, 41.5), (9.0, 41.3)],
    [(8.1, 38.8), (8.2, 39.7), (8.4, 40.8), (9.0, 41.2),
     (9.6, 40.8), (9.7, 39.5), (9.2, 38.9), (8.6, 38.7)],
    # Sicily, Crete, Cyprus, Malta
    [(12.4, 38.0), (13.0, 38.2), (14.5, 38.2), (15.6, 37.5),
     (15.2, 36.7), (13.5, 36.6), (12.5, 37.2)],
    [(23.4, 35.6), (24.0, 35.7), (25.5, 35.5), (26.3, 35.2),
     (25.0, 34.9), (23.5, 35.0)],
    [(32.2, 35.6), (32.4, 35.2), (33.5, 34.6), (34.6, 35.0),
     (34.7, 35.6), (33.4, 35.8)],
    [(14.1, 35.8), (14.6, 35.8), (14.6, 35.9), (14.1, 36.0)],
    # Ionian and larger Aegean islands
    [(19.6, 39.2), (20.0, 39.4), (20.8, 39.8), (20.9, 39.2),
     (20.3, 38.8), (19.8, 38.9)],
    [(20.4, 38.0), (21.0, 38.4), (21.8, 38.4), (21.7, 37.8),
     (21.0, 37.5), (20.5, 37.6)],
    [(22.5, 36.3), (23.0, 36.5), (23.7, 36.4), (23.8, 35.9),
     (23.0, 35.8)],
]
ISLAND_POLYGONS += [
    # Rhodes, Lesbos, Chios, Samos, and other Aegean islands
    [(27.7, 36.4), (28.2, 36.5), (28.3, 36.1), (27.9, 35.9)],
    [(25.8, 39.4), (26.5, 39.5), (26.6, 39.0), (26.0, 38.9)],
    [(25.9, 38.5), (26.2, 38.6), (26.3, 38.1), (26.0, 38.0)],
    [(26.6, 37.8), (27.0, 37.9), (27.1, 37.5), (26.7, 37.4)],
    [(24.3, 39.0), (24.8, 39.2), (25.0, 38.8), (24.5, 38.6)],
    [(25.0, 37.8), (25.5, 37.9), (25.6, 37.4), (25.1, 37.3)],
]


def point_in_polygon(lon, lat, polygon):
    """Return whether a point lies inside a polygon using the even-odd rule."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > lat) != (yj > lat):
            crossing_lon = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing_lon:
                inside = not inside
        j = i
    return inside


def is_land(lon, lat):
    return any(
        point_in_polygon(lon, lat, polygon)
        for polygon in LAND_POLYGONS + ISLAND_POLYGONS
    )
def main():
    latitudes = [45.0 - 0.2 * row for row in range(71)]
    longitudes = [-9.0 + 0.2 * col for col in range(231)]
    matrix = np.zeros((71, 231), dtype=np.uint8)
    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            matrix[row, col] = int(is_land(lon, lat))

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "candidate.csv").open(
        "w", newline="", encoding="ascii"
    ) as f:
        csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(output_dir / "candidate.png")
    image.resize(
        (1848, 568), resample=Image.Resampling.NEAREST
    ).save(output_dir / "candidate_8x.png")

    assert matrix.shape == (71, 231) and matrix.size == 16401
    assert set(np.unique(matrix)).issubset({0, 1})
    rows = list(csv.reader(
        (output_dir / "candidate.csv").open(encoding="ascii")
    ))
    assert len(rows) == 71 and all(len(row) == 231 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert image.size == (231, 71)
    assert set(image.getdata()).issubset({(0, 0, 0), (255, 255, 255)})
    enlarged = Image.open(output_dir / "candidate_8x.png")
    assert enlarged.size == (1848, 568)
    assert enlarged.getpixel((0, 0)) == image.getpixel((0, 0))


if __name__ == "__main__":
    main()
