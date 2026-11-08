from pathlib import Path
import csv

import numpy as np
from PIL import Image

# Hand-authored approximate land outlines, coordinates are (longitude, latitude).
LAND_POLYGONS = [
    # Iberia, mainland Europe, Balkans, and Anatolia
    [
        (-9.0, 36.8), (-9.2, 38.0), (-9.5, 40.0), (-9.4, 41.5),
        (-8.8, 42.3), (-8.9, 43.7), (-7.0, 43.8), (-5.5, 43.4),
        (-2.0, 43.4), (0.0, 42.8), (1.5, 42.5), (3.0, 42.4),
        (4.5, 43.0), (6.0, 43.2), (7.5, 43.8), (9.5, 44.0),
        (10.5, 43.8), (12.0, 44.0), (13.5, 45.0), (15.0, 45.7),
        (16.5, 45.8), (18.0, 45.8), (19.5, 45.5), (21.0, 45.0),
        (22.5, 44.8), (24.0, 45.0), (26.0, 41.8), (27.5, 41.5),
        (29.0, 41.3), (31.0, 41.2), (33.0, 41.5), (35.0, 41.5),
        (37.2, 42.0), (39.0, 44.0), (39.0, 60.0), (-12.0, 60.0),
        (-12.0, 36.0), (-9.0, 36.8),
        # Trace the north coast of the Mediterranean back westward.
        (36.0, 36.0), (34.5, 36.0), (32.5, 36.1), (30.5, 36.2),
        (29.0, 36.2), (27.5, 36.4), (26.0, 36.5), (25.0, 36.8),
        (24.5, 37.0), (23.5, 37.5), (22.8, 38.0), (23.0, 38.5),
        (22.5, 39.0), (22.0, 39.5), (21.5, 40.0), (21.0, 40.5),
        (20.5, 40.8), (19.5, 40.9), (19.0, 40.5), (19.3, 39.8),
        (20.0, 39.0), (20.5, 38.5), (21.0, 38.0), (21.5, 37.5),
        (22.0, 37.0), (22.5, 36.8), (23.0, 36.5), (23.5, 36.3),
        (24.0, 36.2), (24.5, 36.0), (25.0, 35.8),
        (24.0, 35.0), (23.0, 34.8), (22.0, 35.0), (21.0, 35.2),
        (20.0, 35.5), (19.0, 36.0), (18.0, 36.2), (17.0, 36.5),
        (16.0, 37.0), (15.0, 37.3), (14.0, 37.5), (13.0, 38.0),
        (12.5, 38.2), (13.0, 39.0), (14.0, 40.0), (15.0, 40.5),
        (16.0, 41.0), (17.0, 41.5), (18.0, 42.0), (18.5, 42.5),
        (18.0, 43.0), (17.0, 43.5), (16.0, 44.0), (15.0, 44.5),
        (14.0, 45.0), (13.0, 45.5), (12.0, 46.0), (11.0, 46.0),
        (10.0, 45.8), (9.0, 45.5), (8.5, 44.8), (8.0, 44.0),
        (7.5, 43.5), (7.0, 43.0), (6.5, 43.0), (5.5, 43.2),
        (4.5, 43.4), (3.5, 43.4), (2.5, 42.8), (1.5, 42.5),
        (0.5, 42.5), (-0.5, 42.7), (-1.5, 43.0), (-2.5, 43.3),
        (-3.5, 43.4), (-4.5, 43.5), (-5.5, 43.4), (-6.0, 42.5),
        (-7.0, 41.5), (-8.0, 40.5), (-8.8, 39.5), (-9.2, 38.5),
        (-9.0, 36.8),
    ],

    # North African Mediterranean coast.
    [
        (-12.0, 30.0), (37.0, 30.0), (37.0, 32.5), (34.0, 31.5),
        (32.0, 31.2), (30.0, 31.2), (29.0, 31.3), (27.0, 31.5),
        (25.0, 31.5), (23.0, 32.0), (21.0, 32.5), (19.0, 33.0),
        (17.0, 33.0), (15.0, 32.5), (13.0, 32.8), (11.0, 33.0),
        (10.0, 33.5), (9.0, 34.0), (8.0, 34.5), (7.0, 35.0),
        (6.0, 36.0), (5.0, 36.5), (4.0, 36.8), (3.0, 36.8),
        (2.0, 36.7), (1.0, 36.7), (0.0, 36.7), (-1.0, 36.8),
        (-2.0, 36.8), (-3.0, 36.7), (-4.0, 36.5), (-5.0, 36.0),
        (-6.0, 35.0), (-7.0, 33.0), (-9.0, 30.0), (-12.0, 30.0),
    ],

    # The southwest of Great Britain, clipped by the requested frame.
    [
        (-6.5, 49.5), (-5.0, 50.0), (-4.0, 50.5), (-3.0, 51.0),
        (-3.0, 55.0), (-6.5, 55.0), (-6.5, 49.5),
    ],

    # Mallorca and neighboring Balearic islands as one coarse outline.
    [
        (-1.8, 38.6), (0.0, 38.5), (1.5, 39.0), (2.5, 39.5),
        (3.3, 40.0), (3.2, 40.8), (2.0, 41.0), (0.5, 40.8),
        (-0.5, 40.2), (-1.5, 39.5), (-1.8, 38.6),
    ],

    # Corsica and Sardinia.
    [
        (8.0, 41.0), (8.2, 42.0), (8.5, 43.0), (9.2, 43.9),
        (9.7, 43.8), (9.6, 42.5), (9.5, 41.5), (9.0, 40.5),
        (8.3, 40.5), (8.0, 41.0),
    ],
    [
        (8.0, 38.8), (8.2, 40.0), (8.3, 41.0), (9.0, 41.3),
        (9.7, 40.8), (9.7, 39.5), (9.5, 38.8), (9.0, 38.5),
        (8.0, 38.8),
    ],

    # Sicily, Malta, and Pantelleria.
    [
        (12.3, 36.6), (13.0, 37.0), (13.8, 37.2), (15.3, 38.0),
        (15.5, 37.0), (15.0, 36.6), (14.0, 36.6), (12.3, 36.6),
    ],
    [(11.8, 35.7), (12.0, 36.0), (12.2, 35.8), (11.8, 35.7)],
    [(14.1, 35.8), (14.6, 36.1), (14.6, 35.8), (14.1, 35.8)],

    # Crete and Cyprus.
    [
        (23.4, 35.0), (24.0, 35.2), (25.0, 35.3), (26.3, 35.0),
        (26.2, 34.7), (25.0, 34.6), (24.0, 34.7), (23.4, 35.0),
    ],
    [
        (32.2, 34.6), (33.0, 34.6), (34.0, 35.0), (34.6, 35.2),
        (34.6, 35.7), (33.5, 35.7), (32.5, 35.3), (32.2, 34.6),
    ],

    # Larger Aegean islands.
    [(26.0, 39.0), (26.5, 39.2), (27.0, 39.0), (26.5, 38.8), (26.0, 39.0)],
    [(25.0, 36.5), (25.5, 36.8), (26.0, 36.6), (25.5, 36.3), (25.0, 36.5)],
    [(24.0, 38.0), (24.5, 38.3), (25.0, 38.0), (24.5, 37.8), (24.0, 38.0)],
]


def point_in_polygon(lon, lat, polygon):
    """Ray-casting point test, with polygon edges assigned to land."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > lat) != (yj > lat):
            crossing_x = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing_x:
                inside = not inside
        dx, dy = xj - xi, yj - yi
        cross = (lon - xi) * dy - (lat - yi) * dx
        if (abs(cross) < 1e-10 and min(xi, xj) <= lon <= max(xi, xj)
                and min(yi, yj) <= lat <= max(yi, yj)):
            return True
        j = i
    return inside


def main():
    latitudes = [45.0 - 0.2 * r for r in range(71)]
    longitudes = [-9.0 + 0.2 * c for c in range(231)]
    matrix = np.zeros((71, 231), dtype=np.uint8)

    for r, lat in enumerate(latitudes):
        for c, lon in enumerate(longitudes):
            matrix[r, c] = int(any(
                point_in_polygon(lon, lat, polygon)
                for polygon in LAND_POLYGONS
            ))

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    image = Image.fromarray(matrix * 255, mode="L")
    image.save(out / "candidate.png")
    image.resize((1848, 568), resample=Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )

    assert matrix.shape == (71, 231) and matrix.size == 16401
    with (out / "candidate.csv").open(newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 71 and all(len(row) == 231 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert set(image.getdata()) <= {0, 255}
    enlarged = Image.open(out / "candidate_8x.png")
    expected = image.resize((1848, 568), resample=Image.Resampling.NEAREST)
    assert enlarged.size == (1848, 568)
    assert list(enlarged.getdata()) == list(expected.getdata())


if __name__ == "__main__":
    main()
