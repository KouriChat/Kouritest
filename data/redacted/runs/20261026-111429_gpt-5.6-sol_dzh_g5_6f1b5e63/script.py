from pathlib import Path
import csv

import numpy as np
from PIL import Image


# Approximate hand-written outlines in (longitude, latitude).
POLYGONS = [
    # Iberian Peninsula
    [
        (-9.5, 43.8), (-8.2, 43.7), (-7.0, 43.6), (-5.5, 43.5),
        (-3.3, 43.4), (-1.8, 43.4), (-1.5, 43.0), (-1.8, 42.6),
        (-1.4, 42.5), (-1.2, 41.9), (-0.7, 41.4), (0.2, 41.0),
        (1.0, 41.0), (1.7, 41.2), (2.2, 41.3), (3.2, 41.0),
        (3.2, 40.0), (2.8, 39.5), (1.5, 39.0), (0.5, 38.7),
        (-0.2, 38.4), (-0.7, 37.9), (-1.3, 37.5), (-2.2, 36.8),
        (-3.0, 36.7), (-4.4, 36.7), (-5.3, 36.0), (-5.6, 36.0),
        (-6.0, 36.2), (-6.4, 36.5), (-7.4, 37.2), (-7.5, 37.8),
        (-7.4, 38.5), (-7.0, 39.0), (-7.0, 40.0), (-7.0, 41.0),
        (-7.0, 41.9), (-8.0, 42.0), (-8.9, 42.1), (-9.5, 42.2),
    ],
]


def on_segment(x, y, a, b, eps=1e-10):
    ax, ay = a
    bx, by = b
    cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
    if abs(cross) > eps:
        return False
    return (min(ax, bx) - eps <= x <= max(ax, bx) + eps
            and min(ay, by) - eps <= y <= max(ay, by) + eps)


def inside_polygon(x, y, polygon):
    inside = False
    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]
        if on_segment(x, y, a, b):
            return True
        ax, ay = a
        bx, by = b
        if (ay > y) != (by > y):
            crossing_x = ax + (y - ay) * (bx - ax) / (by - ay)
            if x < crossing_x:
                inside = not inside
    return inside
# France and adjoining European mainland
POLYGONS.append([
    (-1.8, 42.6), (-1.5, 43.0), (-1.2, 43.4), (-1.0, 44.0),
    (-1.5, 44.6), (-1.6, 45.0), (8.0, 45.0), (9.0, 45.0),
    (10.0, 45.0), (12.0, 45.0), (15.0, 45.0), (18.0, 45.0),
    (21.0, 45.0), (24.0, 45.0), (27.0, 45.0), (30.0, 45.0),
    (33.0, 45.0), (37.2, 45.0), (37.2, 43.0), (36.5, 42.5),
    (35.5, 42.0), (34.0, 41.8), (32.0, 41.7), (30.0, 41.7),
    (29.0, 41.5), (28.0, 41.3), (27.0, 41.2), (26.0, 41.1),
    (25.5, 40.9), (25.0, 40.8), (24.0, 40.9), (22.5, 40.7),
    (21.5, 40.7), (20.5, 40.8), (19.5, 40.9), (19.0, 41.0),
    (18.0, 41.4), (17.0, 41.8), (16.0, 42.2), (15.0, 42.5),
    (14.0, 42.7), (13.0, 43.0), (12.0, 43.2), (11.0, 43.4),
    (10.0, 43.7), (9.0, 44.2), (8.0, 44.6), (7.0, 44.8),
    (6.0, 45.0), (4.0, 45.0), (2.0, 45.0), (0.0, 45.0),
    (-1.0, 45.0), (-1.5, 44.8), (-1.2, 44.4), (-1.0, 44.0),
    (-1.4, 43.5), (-1.8, 43.0),
])

# Italian mainland
POLYGONS.append([
    (6.5, 45.0), (7.5, 44.7), (8.5, 44.4), (9.5, 44.0),
    (10.0, 43.7), (11.0, 43.3), (12.0, 42.7), (13.0, 42.0),
    (14.0, 41.4), (15.0, 40.8), (16.0, 40.2), (17.0, 39.8),
    (17.5, 40.5), (18.0, 40.7), (18.5, 40.5), (18.4, 40.0),
    (17.8, 39.7), (17.0, 39.6), (16.5, 39.0), (16.0, 38.5),
    (15.7, 38.0), (15.6, 37.5), (15.2, 37.1), (14.8, 37.0),
    (14.3, 38.0), (13.5, 38.0), (12.5, 38.2), (11.5, 39.0),
    (11.0, 40.0), (10.5, 41.0), (10.0, 42.0), (9.0, 44.0),
    (8.0, 44.5), (7.0, 44.8),
])
# North Africa and Anatolia
POLYGONS.append([
    (-9.5, 30.5), (-9.5, 35.0), (-6.0, 35.9), (-5.5, 35.8),
    (-5.2, 35.3), (-4.8, 35.1), (-3.5, 35.2), (-2.0, 35.1),
    (0.0, 35.0), (1.0, 35.0), (2.0, 36.0), (3.0, 36.8),
    (4.0, 36.8), (5.0, 37.0), (6.0, 37.0), (7.0, 37.1),
    (8.0, 37.0), (9.0, 37.2), (9.8, 37.3), (10.5, 37.0),
    (11.0, 36.8), (11.5, 36.5), (11.0, 35.8), (10.5, 34.8),
    (10.0, 34.0), (10.5, 33.0), (11.0, 32.0), (12.0, 31.5),
    (15.0, 31.3), (19.0, 31.5), (23.0, 31.5), (27.0, 31.5),
    (30.0, 31.5), (32.0, 31.0), (37.2, 31.0),
])
POLYGONS.append([
    (26.0, 40.2), (26.0, 40.8), (26.5, 41.0), (27.0, 41.2),
    (28.0, 41.3), (29.0, 41.5), (30.0, 41.7), (32.0, 41.7),
    (34.0, 41.8), (36.5, 42.5), (37.2, 42.5), (37.2, 36.0),
    (36.0, 36.0), (35.0, 36.2), (34.0, 36.2), (33.0, 36.0),
    (31.0, 36.0), (29.0, 36.0), (27.0, 36.0), (26.0, 36.0),
    (26.0, 36.5), (27.0, 37.0), (27.5, 37.5), (28.0, 38.0),
    (27.5, 38.5), (27.0, 39.0), (26.5, 39.5),
])
POLYGONS.append([
    (34.0, 36.2), (37.2, 36.0), (37.2, 31.0), (32.0, 31.0),
    (32.0, 31.5), (34.0, 31.5), (35.0, 32.0), (35.5, 33.0),
    (35.8, 34.0), (36.0, 34.8), (35.8, 35.5),
])

# Major islands
POLYGONS.extend([
    [(12.4, 38.2), (13.6, 38.1), (15.2, 38.3), (15.4, 37.9),
     (15.0, 37.2), (14.5, 36.7), (13.0, 36.7), (12.2, 37.7)],
    [(8.1, 41.3), (9.7, 40.8), (9.7, 39.2), (9.3, 38.8),
     (8.4, 39.0), (8.0, 40.0)],
    [(8.5, 43.0), (9.5, 43.0), (9.5, 42.0), (9.3, 41.3),
     (8.7, 41.4), (8.4, 42.0)],
    [(23.4, 35.7), (26.3, 35.7), (26.3, 35.0), (23.4, 35.0)],
    [(32.2, 35.7), (34.7, 35.7), (34.7, 34.5), (32.2, 34.5)],
])
# Balearic, Maltese, Ionian, and Aegean islands
POLYGONS.extend([
    [(1.1, 39.9), (1.8, 39.7), (2.0, 39.2), (1.5, 38.7), (1.0, 39.0)],
    [(2.2, 40.1), (4.4, 40.1), (4.4, 39.7), (2.2, 39.7)],
    [(3.8, 40.1), (4.4, 40.1), (4.4, 39.8), (3.8, 39.8)],
    [(14.1, 36.1), (14.6, 36.1), (14.6, 35.8), (14.1, 35.8)],
    [(19.5, 39.9), (20.0, 39.9), (20.0, 39.3), (19.5, 39.3)],
    [(20.5, 38.0), (21.0, 38.0), (21.0, 37.5), (20.5, 37.5)],
    [(22.0, 39.0), (22.8, 39.0), (22.8, 38.3), (22.0, 38.3)],
    [(24.0, 38.0), (25.0, 38.0), (25.0, 37.0), (24.0, 37.0)],
    [(25.0, 36.9), (26.0, 36.9), (26.0, 36.3), (25.0, 36.3)],
    [(26.0, 39.0), (26.8, 39.0), (26.8, 38.4), (26.0, 38.4)],
])


def is_land(lon, lat):
    return any(inside_polygon(lon, lat, polygon) for polygon in POLYGONS)


def main():
    lats = [45.0 - 0.2 * i for i in range(71)]
    lons = [-9.0 + 0.2 * i for i in range(231)]
    matrix = np.zeros((71, 231), dtype=np.uint8)
    for row, lat in enumerate(lats):
        for col, lon in enumerate(lons):
            matrix[row, col] = int(is_land(lon, lat))

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open(
            "w", newline="", encoding="ascii") as stream:
        csv.writer(stream, lineterminator="\n").writerows(matrix.tolist())

    image = Image.fromarray(matrix * 255, mode="L")
    image.save(out / "candidate.png")
    enlarged = image.resize((1848, 568), Image.Resampling.NEAREST)
    enlarged.save(out / "candidate_8x.png")

if __name__ == "__main__":
    main()
    data = np.loadtxt("outputs/candidate.csv", delimiter=",", dtype=np.uint8)
    assert data.shape == (71, 231)
    assert data.size == 16401
    assert set(np.unique(data)).issubset({0, 1})

    with open("outputs/candidate.csv", newline="", encoding="ascii") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 71
    assert all(len(row) == 231 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    original = Image.open("outputs/candidate.png")
    enlarged = Image.open("outputs/candidate_8x.png")
    assert original.size == (231, 71)
    assert enlarged.size == (1848, 568)
    assert set(original.getdata()).issubset({0, 255})
    assert set(enlarged.getdata()).issubset({0, 255})
    assert enlarged.resize((231, 71), Image.Resampling.NEAREST).tobytes() == (
        original.tobytes()
    )
