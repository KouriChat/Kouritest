from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image

WEST, EAST = -9.0, 37.0
NORTH, SOUTH = 45.0, 31.0
STEP = 0.2
ROWS, COLS = 71, 231


def polygon(points):
    return [(float(x), float(y)) for x, y in points]


def inside(x, y, points):
    hit = False
    j = len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        crosses = (yi > y) != (yj > y)
        if crosses and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            hit = not hit
        j = i
    return hit


def ellipse(cx, cy, rx, ry, n=24):
    return polygon([
        (cx + rx * math.cos(2 * math.pi * i / n),
         cy + ry * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ])


# Mainland outlines are intentionally compact, hand-written approximations.
LAND = [
    # Iberian Peninsula
    polygon([
        (-9, 45), (4.3, 45), (4.0, 43.4), (3.2, 42.4),
        (2.7, 41.6), (1.8, 40.8), (0.5, 40.2), (-0.5, 39.7),
        (-1.2, 38.0), (-2.0, 37.2), (-3.0, 36.8), (-4.5, 36.0),
        (-6.0, 36.0), (-7.5, 36.5), (-8.5, 37.3), (-9, 38.5),
    ]),
    # Southern France and the north-western Mediterranean coast
    polygon([
        (-2, 45), (8.2, 45), (7.7, 44.0), (6.7, 43.2),
        (5.5, 42.5), (4.4, 41.7), (3.2, 41.1), (2.0, 40.6),
        (0.3, 40.3), (-1.0, 40.5), (-2.0, 41.4),
    ]),
    # Italian peninsula and the Alpine edge
    polygon([
        (6.0, 45), (13.0, 45), (13.3, 44.0), (13.7, 43.0),
        (14.3, 42.4), (15.2, 41.9), (16.2, 41.4), (17.2, 40.9),
        (18.5, 40.2), (18.0, 39.5), (17.0, 39.5), (16.0, 39.8),
        (15.0, 40.2), (14.6, 39.5), (14.1, 38.5), (13.5, 38.0),
        (12.6, 38.5), (12.0, 39.2), (11.0, 39.8), (10.0, 40.3),
        (9.0, 41.0), (8.0, 42.0), (7.0, 43.0), (6.5, 44.0),
    ]),
    # Balkan mainland, including the northern Adriatic and Thrace
    polygon([
        (13.0, 45), (37, 45), (37, 36.2), (35.8, 36.3),
        (34.4, 36.7), (33.0, 36.8), (31.4, 36.3), (30.0, 36.2),
        (28.5, 36.8), (27.2, 37.0), (26.0, 37.5), (24.8, 38.0),
        (23.5, 38.3), (22.4, 38.4), (21.5, 38.3), (20.7, 38.0),
        (19.5, 38.0), (18.5, 38.4), (17.5, 39.0), (17.0, 40.0),
        (16.3, 41.0), (15.5, 42.0), (14.8, 43.0), (14.2, 44.0),
    ]),
    # Anatolian mainland
    polygon([
        (26.0, 45), (37, 45), (37, 36.2), (36.3, 36.0),
        (35.5, 36.2), (34.5, 36.5), (33.5, 36.7), (32.0, 36.2),
        (30.5, 36.1), (29.0, 36.3), (27.8, 36.8), (26.8, 37.3),
        (26.0, 38.0), (26.6, 39.0), (27.4, 39.8), (26.8, 40.6),
    ]),
    # Maghreb and the north coast of Libya and Egypt
    polygon([
        (-9, 31), (-9, 35.8), (-7.5, 35.8), (-6.3, 35.4),
        (-5.0, 35.1), (-3.5, 35.0), (-2.0, 35.0), (-0.5, 35.1),
        (1.0, 35.4), (2.5, 35.5), (4.0, 35.8), (5.2, 36.2),
        (6.5, 36.6), (7.8, 36.6), (9.0, 36.4), (10.0, 36.8),
        (11.5, 37.0), (12.8, 37.0), (14.0, 36.7), (15.0, 36.3),
        (16.0, 36.0), (17.2, 35.7), (18.5, 35.5), (20.0, 35.4),
        (21.5, 35.3), (23.0, 35.1), (24.5, 34.8), (26.0, 34.5),
        (27.5, 34.0), (29.0, 33.5), (30.5, 33.0), (32.0, 32.2),
        (33.5, 31.5), (35.0, 31.0), (37, 31),
    ]),
]
# Replace the coarse mainland list with separate hand-traced coastal regions.
# Keeping each region independent makes the Mediterranean bays readable.
LAND = [
    polygon([
        (-9, 45), (-1.8, 45), (-1.1, 44.5), (0.0, 43.8),
        (1.0, 43.4), (2.0, 43.1), (3.1, 42.8), (3.4, 42.2),
        (2.5, 41.8), (1.5, 41.5), (0.5, 41.3), (-0.5, 41.0),
        (-1.2, 40.2), (-1.5, 39.3), (-2.4, 38.9), (-3.0, 37.8),
        (-4.0, 36.9), (-5.4, 36.0), (-6.7, 36.0), (-8.1, 36.5),
        (-9, 37.2),
    ]),
    polygon([
        (-2.0, 45), (8.2, 45), (7.8, 44.4), (7.3, 43.8),
        (6.7, 43.4), (6.0, 43.1), (5.2, 43.0), (4.4, 42.8),
        (3.5, 42.4), (2.8, 42.0), (2.2, 41.7), (1.4, 41.4),
        (0.3, 41.2), (-0.4, 41.4), (-1.2, 42.0), (-2.0, 43.0),
    ]),
    polygon([
        (6.0, 45), (13.0, 45), (13.5, 44.5), (13.7, 43.7),
        (13.8, 43.0), (14.4, 42.5), (15.5, 42.0), (16.5, 41.5),
        (17.5, 41.0), (18.5, 40.5), (18.1, 39.8), (17.2, 39.5),
        (16.3, 39.8), (15.5, 40.1), (14.8, 40.4), (14.5, 39.7),
        (14.0, 38.8), (13.5, 38.0), (12.6, 38.4), (12.0, 39.1),
        (11.1, 39.7), (10.0, 40.3), (9.0, 41.1), (8.1, 42.0),
        (7.0, 43.0), (6.5, 44.0),
    ]),
    # Adriatic and Balkan mainland, with the Greek mainland at the south.
    polygon([
        (13.0, 45), (15.0, 45), (15.0, 44.5), (15.7, 44.0),
        (16.5, 43.7), (17.5, 43.5), (18.5, 43.6), (19.5, 43.3),
        (20.5, 43.0), (21.5, 42.8), (22.2, 42.3), (22.8, 41.8),
        (23.4, 41.3), (24.0, 40.8), (24.8, 40.4), (25.5, 40.0),
        (26.5, 39.5), (27.2, 39.0), (28.0, 38.5), (29.0, 38.0),
        (30.0, 37.5), (31.0, 37.0), (32.0, 36.5), (33.0, 36.2),
        (34.0, 36.0), (35.0, 36.2), (36.0, 36.4), (37, 36.2),
        (37, 45), (15.0, 45),
    ]),
    # Anatolia, kept separate so the Aegean remains water.
    polygon([
        (26.0, 45), (37, 45), (37, 36.2), (36.0, 36.4),
        (35.0, 36.2), (34.0, 36.0), (33.0, 36.2), (32.0, 36.1),
        (31.0, 36.0), (30.0, 36.1), (29.0, 36.4), (28.0, 36.8),
        (27.0, 37.3), (26.1, 38.0), (26.5, 39.0), (27.2, 39.8),
        (27.0, 40.7), (26.4, 41.5), (26.0, 42.5),
    ]),
    polygon([
        (-9, 31), (-9, 35.9), (-8.0, 35.9), (-7.0, 35.7),
        (-6.0, 35.3), (-4.5, 35.1), (-3.0, 35.0), (-1.5, 35.1),
        (0.0, 35.3), (1.5, 35.5), (3.0, 35.6), (4.5, 36.0),
        (5.8, 36.4), (7.0, 36.7), (8.2, 36.7), (9.5, 36.5),
        (10.5, 36.8), (11.5, 37.0), (12.5, 37.0), (13.5, 36.7),
        (14.5, 36.4), (15.5, 36.0), (17.0, 35.7), (18.5, 35.5),
        (20.0, 35.4), (22.0, 35.2), (24.0, 34.8), (26.0, 34.4),
        (28.0, 33.7), (30.0, 32.8), (32.0, 32.0), (34.0, 31.2),
        (37, 31), 
    ]),
]
# Islands and island groups that can affect the 0.2 degree samples.
ISLANDS = [
    ellipse(3.05, 39.65, 0.62, 0.48, 28),   # Mallorca
    ellipse(4.12, 40.05, 0.36, 0.20, 20),   # Menorca
    ellipse(1.43, 38.98, 0.25, 0.18, 18),   # Ibiza
    ellipse(1.45, 38.70, 0.12, 0.08, 14),   # Formentera
    polygon([(8.55, 43.0), (8.75, 42.5), (8.65, 41.8), (8.9, 41.2),
             (9.15, 40.5), (9.3, 39.7), (9.0, 39.0), (8.4, 39.1),
             (8.1, 40.0), (8.0, 41.0), (8.25, 42.0)]),  # Corsica
    polygon([(8.2, 41.2), (8.8, 41.1), (9.3, 40.5), (9.7, 39.8),
             (9.7, 39.0), (9.4, 38.2), (9.0, 38.0), (8.4, 38.4),
             (8.1, 39.2)]),  # Sardinia
    polygon([(12.4, 38.2), (13.0, 38.0), (13.8, 38.0), (14.7, 37.6),
             (15.2, 37.0), (15.0, 36.5), (14.2, 36.6), (13.5, 37.0),
             (12.6, 37.4)]),  # Sicily
    ellipse(14.35, 35.88, 0.28, 0.14, 18),   # Malta
    polygon([(32.3, 35.7), (33.0, 35.6), (34.0, 35.4), (34.8, 35.0),
             (34.5, 34.4), (33.8, 34.0), (32.8, 34.5), (32.2, 35.1)]),
    # Crete and Rhodes
    polygon([(23.4, 35.5), (24.5, 35.7), (25.5, 35.6), (26.4, 35.3),
             (26.2, 34.9), (25.2, 34.8), (24.2, 35.0)]),
    ellipse(28.2, 36.17, 0.34, 0.18, 18),
    # Larger Aegean islands
    ellipse(24.5, 39.1, 0.35, 0.20, 18),   # Lemnos
    ellipse(24.2, 38.5, 0.22, 0.13, 16),   # Lesbos
    ellipse(26.0, 38.4, 0.42, 0.20, 20),   # Chios
    ellipse(26.9, 37.75, 0.28, 0.13, 16),  # Samos
    ellipse(24.65, 37.0, 0.30, 0.14, 16),  # Milos
    ellipse(25.45, 37.1, 0.30, 0.16, 16),  # Naxos
    ellipse(25.25, 36.4, 0.25, 0.15, 16),  # Santorini
    ellipse(25.9, 36.85, 0.23, 0.13, 16),  # Amorgos
    ellipse(27.15, 36.8, 0.28, 0.15, 16),  # Kos
    # Ionian islands
    ellipse(19.92, 39.65, 0.18, 0.40, 18),  # Corfu
    ellipse(20.7, 38.7, 0.18, 0.55, 18),   # Kefalonia
    ellipse(20.65, 38.45, 0.14, 0.25, 16),  # Zakynthos
    # Adriatic islands; only the larger ones survive this sampling scale.
    ellipse(15.95, 43.6, 0.13, 0.32, 16),
    ellipse(16.3, 43.75, 0.12, 0.28, 16),
    ellipse(16.9, 43.45, 0.12, 0.27, 16),
    ellipse(17.35, 43.05, 0.12, 0.25, 16),
]

WATER_HOLES = [
    ellipse(6.25, 46.45, 0.28, 0.10, 18),  # Lake Geneva edge in the frame
    ellipse(10.0, 45.8, 0.18, 0.45, 18),    # Lake Garda / northern Italian lakes
    ellipse(20.8, 41.05, 0.16, 0.20, 16),   # Lake Ohrid
    ellipse(21.05, 40.95, 0.10, 0.12, 14),  # Lake Prespa
]
def is_land(lon, lat):
    if any(inside(lon, lat, shape) for shape in WATER_HOLES):
        return False
    if any(inside(lon, lat, shape) for shape in LAND):
        return True
    return any(inside(lon, lat, shape) for shape in ISLANDS)


def make_matrix():
    values = []
    for row in range(ROWS):
        lat = NORTH - STEP * row
        values.append([
            1 if is_land(WEST + STEP * col, lat) else 0
            for col in range(COLS)
        ])
    matrix = np.asarray(values, dtype=np.uint8)
    assert matrix.shape == (71, 231)
    assert matrix.size == 16401
    assert set(np.unique(matrix)).issubset({0, 1})
    return matrix


def save_outputs(matrix):
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(matrix.tolist())

    image = Image.fromarray((matrix * 255).astype(np.uint8), mode="L")
    assert image.size == (231, 71)
    image.save(out / "candidate.png")

    enlarged = image.resize((1848, 568), resample=Image.Resampling.NEAREST)
    assert enlarged.size == (1848, 568)
    enlarged.save(out / "candidate_8x.png")

    # Validate the exact output contract after serialization.
    rows = (out / "candidate.csv").read_text(encoding="ascii").splitlines()
    assert len(rows) == 71
    assert all(len(line.split(",")) == 231 for line in rows)
    assert all(cell in {"0", "1"} for line in rows for cell in line.split(","))
    assert set(image.getdata()).issubset({0, 255})
    assert enlarged.getpixel((8 * 20, 8 * 20)) == image.getpixel((20, 20))


if __name__ == "__main__":
    matrix = make_matrix()
    save_outputs(matrix)
    print("generated 71x231 matrix and 8x nearest-neighbor image")
