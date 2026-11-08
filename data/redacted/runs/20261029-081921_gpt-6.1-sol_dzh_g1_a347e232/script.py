from pathlib import Path

import numpy as np
from PIL import Image


LAT_TOP = 45.0
LAT_BOTTOM = 31.0
LON_LEFT = -9.0
LON_RIGHT = 37.0
STEP = 0.2

ROWS = 71
COLS = 231


def polygon_contains(lon, lat, polygon):
    """Ray-casting point-in-polygon test."""
    inside = False
    n = len(polygon)

    x1, y1 = polygon[-1]
    for x2, y2 in polygon:
        crosses = ((y1 > lat) != (y2 > lat))
        if crosses:
            x_intersection = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
            if lon < x_intersection:
                inside = not inside
        x1, y1 = x2, y2

    return inside


def ellipse_polygon(cx, cy, rx, ry, points=40):
    """Create a small hand-approximated elliptical island."""
    angles = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    return [
        (cx + rx * float(np.cos(a)), cy + ry * float(np.sin(a)))
        for a in angles
    ]


# Mainland Europe west of the Adriatic and Greece.
EUROPE_WEST = [
    (-9.0, 45.0),
    (15.0, 45.0),
    (14.5, 44.0),
    (13.5, 43.0),
    (13.0, 42.0),
    (13.7, 41.0),
    (15.0, 40.0),
    (16.5, 39.0),
    (18.5, 39.0),
    (17.5, 38.5),
    (16.0, 38.3),
    (15.0, 39.0),
    (14.0, 40.0),
    (13.0, 41.0),
    (12.0, 42.0),
    (11.0, 43.0),
    (9.0, 43.5),
    (7.0, 43.4),
    (5.0, 43.3),
    (3.5, 42.8),
    (3.0, 42.0),
    (1.0, 41.5),
    (-1.0, 42.0),
    (-3.0, 43.0),
    (-5.0, 44.0),
    (-9.0, 44.0),
]

# Iberian Peninsula.
IBERIA = [
    (-9.0, 45.0),
    (3.0, 45.0),
    (3.3, 42.3),
    (2.3, 40.0),
    (1.2, 38.5),
    (0.0, 37.5),
    (-2.0, 36.7),
    (-3.0, 36.1),
    (-5.0, 36.0),
    (-6.3, 36.0),
    (-7.5, 36.8),
    (-8.9, 37.0),
    (-8.9, 41.0),
    (-8.6, 41.5),
    (-8.8, 42.2),
    (-8.7, 43.4),
]

# Balkan Peninsula and mainland Greece.
BALKANS = [
    (18.0, 39.0),
    (19.0, 39.2),
    (20.0, 39.5),
    (21.0, 40.0),
    (22.0, 40.5),
    (23.0, 41.0),
    (24.0, 41.1),
    (25.0, 41.3),
    (26.0, 41.5),
    (27.0, 42.0),
    (28.5, 43.0),
    (29.0, 43.5),
    (29.0, 45.0),
    (15.0, 45.0),
    (15.0, 44.0),
    (14.5, 43.0),
    (14.5, 42.0),
    (15.0, 41.0),
    (16.0, 40.0),
    (17.0, 39.0),
]

# Anatolia.
ANATOLIA = [
    (26.0, 41.0),
    (27.0, 41.5),
    (29.0, 42.0),
    (31.0, 42.5),
    (33.0, 42.4),
    (35.0, 42.5),
    (37.0, 41.5),
    (37.0, 36.0),
    (36.0, 36.0),
    (35.0, 36.0),
    (34.0, 36.0),
    (33.0, 36.0),
    (32.0, 36.2),
    (30.0, 36.5),
    (28.0, 36.5),
    (27.0, 37.5),
    (26.0, 38.5),
]

# North Africa, including Morocco, Algeria, Tunisia, Libya and Egypt.
NORTH_AFRICA = [
    (-9.0, 35.5),
    (-8.0, 35.2),
    (-7.0, 35.0),
    (-6.0, 35.8),
    (-5.5, 35.9),
    (-4.5, 35.3),
    (-3.0, 35.0),
    (-1.0, 35.0),
    (0.0, 35.2),
    (1.0, 35.5),
    (2.0, 35.8),
    (3.0, 36.0),
    (4.0, 36.2),
    (5.0, 36.3),
    (6.0, 36.7),
    (7.0, 37.0),
    (8.0, 37.1),
    (9.0, 37.0),
    (10.0, 37.2),
    (11.0, 37.2),
    (11.5, 36.5),
    (12.0, 35.5),
    (13.0, 34.8),
    (14.0, 33.8),
    (16.0, 33.3),
    (18.0, 32.8),
    (20.0, 32.5),
    (22.0, 32.5),
    (24.0, 32.8),
    (25.0, 31.5),
    (37.0, 31.0),
    (37.0, 31.0),
    (-9.0, 31.0),
]

# Levantine coast and inland land east of the Mediterranean.
LEVANT = [
    (36.2, 36.5),
    (35.7, 35.9),
    (35.8, 35.2),
    (35.1, 34.5),
    (34.5, 33.5),
    (34.3, 32.8),
    (34.7, 31.2),
    (37.0, 31.0),
    (37.0, 37.0),
]

# Italian Peninsula.
ITALY = [
    (7.5, 44.0),
    (9.0, 44.0),
    (10.0, 43.5),
    (11.0, 42.5),
    (12.0, 42.0),
    (13.0, 41.5),
    (14.5, 41.0),
    (16.0, 40.5),
    (16.8, 39.0),
    (16.2, 38.0),
    (15.5, 37.0),
    (15.0, 38.0),
    (14.0, 39.0),
    (13.0, 40.0),
    (12.0, 41.0),
    (11.0, 41.5),
    (10.0, 42.0),
    (9.0, 42.5),
    (8.5, 43.0),
]

LAND_POLYGONS = [
    EUROPE_WEST,
    IBERIA,
    BALKANS,
    ANATOLIA,
    NORTH_AFRICA,
    LEVANT,
    ITALY,
]


# Large Mediterranean islands.
ISLAND_POLYGONS = [
    # Corsica
    [
        (8.55, 41.35),
        (8.65, 42.00),
        (8.75, 42.55),
        (9.05, 42.95),
        (9.35, 42.75),
        (9.45, 42.20),
        (9.35, 41.65),
        (9.15, 41.30),
        (8.80, 41.20),
    ],

    # Sardinia
    [
        (8.10, 38.85),
        (8.05, 39.50),
        (8.20, 40.20),
        (8.55, 40.85),
        (9.10, 41.25),
        (9.60, 41.05),
        (9.80, 40.45),
        (9.65, 39.75),
        (9.35, 39.15),
        (8.75, 38.85),
    ],

    # Sicily
    [
        (12.35, 37.05),
        (12.80, 37.45),
        (13.50, 37.85),
        (14.30, 38.25),
        (15.20, 38.25),
        (15.65, 37.80),
        (15.25, 37.35),
        (14.55, 36.75),
        (13.55, 36.60),
        (12.70, 36.75),
    ],

    # Crete
    [
        (23.40, 35.25),
        (24.00, 35.00),
        (25.00, 34.90),
        (26.00, 35.10),
        (26.30, 35.45),
        (25.80, 35.70),
        (24.70, 35.75),
        (23.80, 35.65),
    ],

    # Cyprus
    [
        (32.20, 34.75),
        (33.00, 34.50),
        (34.00, 34.55),
        (34.60, 35.00),
        (34.50, 35.45),
        (33.70, 35.70),
        (32.80, 35.55),
        (32.20, 35.10),
    ],

    # Mallorca
    [
        (2.30, 39.25),
        (2.40, 39.75),
        (2.80, 40.05),
        (3.40, 39.95),
        (3.55, 39.55),
        (3.35, 39.20),
        (2.80, 39.10),
    ],

    # Menorca
    [
        (3.80, 39.80),
        (4.15, 39.80),
        (4.35, 40.05),
        (4.20, 40.25),
        (3.90, 40.15),
    ],

    # Ibiza and Formentera
    [
        (1.15, 38.75),
        (1.35, 39.05),
        (1.65, 39.00),
        (1.75, 38.75),
        (1.50, 38.60),
    ],
    [
        (1.35, 38.60),
        (1.55, 38.65),
        (1.60, 38.50),
        (1.40, 38.45),
    ],

    # Rhodes
    [
        (27.70, 35.90),
        (28.00, 36.20),
        (28.25, 36.45),
        (28.25, 36.75),
        (27.95, 36.85),
        (27.70, 36.55),
        (27.60, 36.15),
    ],

    # Euboea
    [
        (23.35, 38.45),
        (23.70, 38.60),
        (24.20, 38.75),
        (24.65, 39.05),
        (24.55, 39.35),
        (24.10, 39.40),
        (23.75, 39.00),
        (23.45, 38.75),
    ],

    # Corfu
    [
        (19.65, 39.35),
        (19.80, 39.75),
        (20.05, 39.95),
        (20.25, 39.75),
        (20.25, 39.35),
        (20.00, 39.15),
    ],

    # Lesbos
    [
        (25.90, 39.00),
        (26.30, 39.20),
        (26.55, 39.45),
        (26.35, 39.75),
        (25.90, 39.75),
        (25.70, 39.45),
    ],

    # Chios
    [
        (25.90, 38.15),
        (26.05, 38.45),
        (26.25, 38.60),
        (26.35, 38.35),
        (26.20, 38.00),
    ],

    # Samos
    [
        (26.55, 37.65),
        (26.80, 37.85),
        (27.05, 37.80),
        (27.10, 37.55),
        (26.80, 37.45),
    ],

    # Karpathos
    [
        (26.95, 35.35),
        (27.15, 35.75),
        (27.35, 36.20),
        (27.50, 36.45),
        (27.35, 36.55),
        (27.10, 36.20),
        (26.90, 35.70),
    ],

    # Malta and Gozo
    [
        (14.15, 35.80),
        (14.35, 35.75),
        (14.55, 35.85),
        (14.55, 36.10),
        (14.30, 36.15),
        (14.10, 36.00),
    ],
    [
        (14.15, 36.20),
        (14.35, 36.20),
        (14.35, 36.35),
        (14.10, 36.35),
    ],

    # Djerba
    [
        (10.70, 33.70),
        (10.95, 33.55),
        (11.20, 33.65),
        (11.30, 33.90),
        (11.10, 34.10),
        (10.80, 34.00),
    ],

    # Lampedusa
    [
        (12.00, 35.45),
        (12.25, 35.45),
        (12.40, 35.60),
        (12.20, 35.75),
        (11.95, 35.65),
    ],
]

# Smaller islands and island groups represented by compact hand-made ellipses.
ISLAND_POLYGONS.extend([
    ellipse_polygon(23.75, 37.70, 0.13, 0.20),  # Cephalonia
    ellipse_polygon(20.65, 38.20, 0.12, 0.22),  # Zakynthos
    ellipse_polygon(24.00, 37.75, 0.09, 0.16),  # Ithaca
    ellipse_polygon(24.40, 37.45, 0.10, 0.16),  # Kythira
    ellipse_polygon(25.15, 36.75, 0.12, 0.20),  # Naxos
    ellipse_polygon(25.25, 37.10, 0.09, 0.13),  # Paros
    ellipse_polygon(24.70, 37.10, 0.08, 0.14),  # Serifos
    ellipse_polygon(25.90, 36.95, 0.08, 0.13),  # Kos
    ellipse_polygon(26.95, 36.65, 0.07, 0.13),  # Halki
    ellipse_polygon(27.15, 36.70, 0.06, 0.12),  # Symi
    ellipse_polygon(23.55, 35.15, 0.07, 0.12),  # Gavdos
    ellipse_polygon(11.95, 36.85, 0.10, 0.14),  # Pantelleria
    ellipse_polygon(8.15, 40.05, 0.08, 0.14),  # Asinara
    ellipse_polygon(9.75, 40.60, 0.06, 0.10),  # La Maddalena
    ellipse_polygon(10.20, 42.70, 0.08, 0.12),  # Elba
    ellipse_polygon(10.00, 42.80, 0.04, 0.08),  # Capraia
    ellipse_polygon(4.10, 39.90, 0.04, 0.07),  # small Balearic islet
])


def is_land(lon, lat):
    for polygon in LAND_POLYGONS:
        if polygon_contains(lon, lat, polygon):
            return True

    for polygon in ISLAND_POLYGONS:
        if polygon_contains(lon, lat, polygon):
            return True

    return False


def build_matrix():
    latitudes = LAT_TOP - STEP * np.arange(ROWS)
    longitudes = LON_LEFT + STEP * np.arange(COLS)

    matrix = np.zeros((ROWS, COLS), dtype=np.uint8)

    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            matrix[row, col] = 1 if is_land(float(lon), float(lat)) else 0

    return matrix


def write_outputs(matrix):
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "candidate.csv"
    png_path = output_dir / "candidate.png"
    enlarged_path = output_dir / "candidate_8x.png"

    np.savetxt(
        csv_path,
        matrix,
        fmt="%d",
        delimiter=",",
    )

    # 1 is land and is rendered white; 0 is water and is rendered black.
    image = Image.fromarray((matrix * 255).astype(np.uint8), mode="L")
    image.save(png_path)

    enlarged = image.resize(
        (COLS * 8, ROWS * 8),
        resample=Image.Resampling.NEAREST,
    )
    enlarged.save(enlarged_path)

    return csv_path, png_path, enlarged_path


def validate(matrix, paths):
    csv_path, png_path, enlarged_path = paths

    assert matrix.shape == (71, 231)
    assert matrix.size == 16401
    assert np.all((matrix == 0) | (matrix == 1))

    csv_rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(csv_rows) == 71
    assert all(len(row.split(",")) == 231 for row in csv_rows)
    assert all(value in {"0", "1"} for row in csv_rows for value in row.split(","))

    with Image.open(png_path) as image:
        assert image.size == (231, 71)
        assert image.mode == "L"
        assert set(image.getdata()).issubset({0, 255})

    with Image.open(enlarged_path) as image:
        assert image.size == (1848, 568)
        assert image.mode == "L"
        assert set(image.getdata()).issubset({0, 255})

    # Confirm every 8x8 block in the enlarged image equals one source pixel.
    with Image.open(png_path) as original:
        original_pixels = np.asarray(original, dtype=np.uint8)
    with Image.open(enlarged_path) as enlarged:
        enlarged_pixels = np.asarray(enlarged, dtype=np.uint8)

    expected = np.repeat(np.repeat(original_pixels, 8, axis=0), 8, axis=1)
    assert np.array_equal(enlarged_pixels, expected)


def main():
    matrix = build_matrix()
    paths = write_outputs(matrix)
    validate(matrix, paths)

    print(f"matrix: {matrix.shape[0]} x {matrix.shape[1]}")
    print(f"valid values: {matrix.size}")
    print(f"wrote: {paths[0]}")
    print(f"wrote: {paths[1]}")
    print(f"wrote: {paths[2]}")


if __name__ == "__main__":
    main()
