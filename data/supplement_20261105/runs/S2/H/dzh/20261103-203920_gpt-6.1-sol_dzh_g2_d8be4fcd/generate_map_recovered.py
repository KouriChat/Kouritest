from pathlib import Path
import csv

import numpy as np
from PIL import Image


# Approximate coastlines sketched from general geographic knowledge.
# Coordinates are (longitude, latitude), in WGS84 decimal degrees.
LAND_POLYGONS = [
    # Iberia, southern France, Italy, the Balkans, Greece, and Anatolia.
    [
        (-9.0, 43.8), (-8.7, 42.8), (-8.9, 41.8), (-8.8, 39.8),
        (-9.0, 38.5), (-8.8, 37.0), (-7.5, 37.0), (-7.2, 36.8),
        (-6.0, 36.0), (-5.6, 36.0), (-5.3, 36.2), (-4.5, 36.5),
        (-3.0, 36.7), (-1.5, 36.7), (-0.5, 37.0), (0.5, 37.5),
        (1.5, 37.8), (2.0, 39.0), (3.0, 41.0), (3.2, 42.4),
        (4.5, 43.4), (6.0, 43.5), (7.5, 43.7), (8.5, 43.8),
        (9.5, 44.0), (10.5, 43.8), (10.0, 43.0), (10.5, 42.5),
        (11.5, 42.0), (12.0, 41.0), (13.0, 40.7), (14.5, 40.7),
        (15.5, 39.8), (16.0, 38.8), (15.5, 38.0), (15.0, 37.2),
        (15.5, 36.7), (16.5, 37.0), (17.0, 38.0), (17.2, 39.0),
        (18.5, 40.0), (18.0, 40.7), (17.0, 41.0), (16.5, 41.8),
        (17.0, 42.0), (17.8, 40.8), (18.5, 40.0), (19.3, 39.7),
        (19.5, 40.0), (19.3, 40.5), (19.0, 41.0), (18.5, 41.5),
        (18.5, 42.0), (19.0, 42.5), (19.5, 42.7), (19.5, 43.0),
        (19.0, 43.5), (18.5, 43.8), (18.0, 44.0), (17.5, 44.5),
        (16.5, 45.0), (37.0, 45.0), (37.0, 36.5), (35.0, 36.5),
        (33.0, 36.0), (31.0, 36.0), (29.0, 36.2), (27.0, 36.5),
        (26.5, 37.0), (26.0, 38.0), (26.5, 39.0), (26.0, 40.0),
        (27.0, 40.3), (28.0, 40.5), (29.0, 40.7), (30.0, 41.0),
        (31.0, 41.3), (32.0, 41.5), (33.0, 42.0), (34.0, 42.5),
        (35.0, 43.0), (37.0, 43.5), (37.0, 45.0), (-9.0, 45.0),
    ],

    # North Africa, Egypt, Sinai, and the southern Levant.
    [
        (-5.8, 35.8), (-5.2, 35.3), (-3.0, 35.2), (-1.0, 35.0),
        (0.0, 35.3), (1.0, 36.0), (2.0, 36.7), (3.0, 36.8),
        (4.0, 36.8), (5.0, 37.0), (6.0, 37.1), (7.0, 37.1),
        (8.0, 37.0), (9.0, 37.0), (10.0, 36.8), (10.8, 36.5),
        (11.0, 35.5), (11.0, 34.0), (10.5, 33.0), (11.0, 32.5),
        (13.0, 32.3), (15.0, 32.5), (18.0, 32.8), (20.0, 32.5),
        (22.0, 32.0), (24.0, 31.5), (25.0, 31.4), (27.0, 31.4),
        (29.0, 31.4), (31.0, 31.4), (32.5, 31.3), (33.5, 31.3),
        (34.3, 31.5), (34.5, 31.8), (34.8, 32.3), (35.0, 33.0),
        (35.5, 33.5), (36.0, 34.0), (36.5, 34.5), (37.0, 35.0),
        (37.0, 29.0), (-9.0, 29.0), (-9.0, 34.0),
    ],

    # Balearic Islands.
    [(-1.0, 38.5), (0.0, 38.5), (1.5, 39.0), (2.0, 39.8),
     (1.0, 40.0), (0.0, 39.7), (-1.0, 39.5)],
    [(3.0, 39.5), (4.5, 39.5), (4.5, 40.2), (3.5, 40.1)],

    # Corsica and Sardinia.
    [(8.5, 41.3), (9.5, 41.3), (9.6, 42.0), (9.3, 43.0),
     (8.8, 42.8), (8.5, 42.0)],
    [(8.1, 38.8), (9.7, 38.8), (9.8, 40.0), (9.5, 41.0),
     (8.5, 41.0), (8.2, 40.0)],

    # Sicily, Malta, and nearby islands.
    [(12.3, 37.0), (13.0, 36.6), (14.5, 37.0), (15.2, 37.3),
     (15.0, 38.2), (14.0, 38.2), (13.0, 38.1), (12.5, 37.7)],
    [(14.1, 35.8), (14.7, 35.8), (14.7, 36.1), (14.2, 36.1)],

    # Crete and Cyprus.
    [(23.4, 35.0), (24.8, 34.9), (26.3, 35.0), (26.2, 35.4),
     (24.8, 35.6), (23.5, 35.5)],
    [(32.2, 34.5), (34.6, 34.5), (34.7, 35.2), (33.2, 35.7),
     (32.3, 35.2)],

    # Rhodes and larger Dodecanese islands.
    [(27.6, 35.8), (28.3, 35.8), (28.2, 36.5), (27.7, 36.5)],
    [(26.0, 36.5), (26.5, 36.5), (26.5, 37.0), (26.0, 37.0)],
    [(25.0, 36.5), (25.5, 36.5), (25.5, 37.0), (25.0, 37.0)],

    # Selected Aegean islands and island groups.
    [(23.3, 37.5), (24.2, 37.5), (24.4, 38.0), (23.5, 38.2)],
    [(24.5, 38.0), (25.2, 38.0), (25.4, 38.5), (24.7, 38.6)],
    [(25.0, 39.0), (25.7, 39.0), (25.7, 39.5), (25.0, 39.5)],
    [(24.0, 39.0), (24.7, 39.0), (24.8, 39.5), (24.1, 39.5)],
    [(22.5, 39.0), (23.2, 39.0), (23.2, 39.5), (22.5, 39.5)],
    [(26.0, 38.5), (26.5, 38.5), (26.5, 39.0), (26.0, 39.0)],

    # Other islands in the western and central Mediterranean.
    [(10.0, 35.8), (11.0, 35.8), (11.1, 36.2), (10.2, 36.3)],
    [(7.5, 39.0), (8.0, 39.0), (8.0, 39.5), (7.5, 39.5)],
    [(5.8, 43.0), (6.0, 43.0), (6.0, 43.2), (5.8, 43.2)],
]


def point_in_polygon(lon, lat, polygon):
    """Ray-casting test for one point against a closed polygon."""
    inside = False
    j = len(polygon) - 1

    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        crosses = (yi > lat) != (yj > lat)
        if crosses:
            x_intersection = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_intersection:
                inside = not inside
        j = i

    return inside


def main():
    latitudes = 45.0 - 0.2 * np.arange(71)
    longitudes = -9.0 + 0.2 * np.arange(231)

    matrix = np.zeros((71, 231), dtype=np.uint8)

    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            if any(point_in_polygon(lon, lat, poly) for poly in LAND_POLYGONS):
                matrix[row, col] = 1

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # 1 is white land; 0 is black water.
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(output_dir / "candidate.png")

    enlarged = image.resize((231 * 8, 71 * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")

    # Validate dimensions, values, CSV shape, and image colors.
    assert matrix.shape == (71, 231)
    assert matrix.size == 16401
    assert np.isin(matrix, (0, 1)).all()

    with csv_path.open("r", newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 71
    assert all(len(row) == 231 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    check = Image.open(output_dir / "candidate.png")
    assert check.size == (231, 71)
    assert set(check.getdata()) <= {0, 255}

    check_large = Image.open(output_dir / "candidate_8x.png")
    assert check_large.size == (1848, 568)
    assert set(check_large.getdata()) <= {0, 255}

    print("Created outputs/candidate.csv, candidate.png, candidate_8x.png")
    print("Validated 71x231 matrix (16401 values), binary CSV, and PNG dimensions.")


if __name__ == "__main__":
    main()
