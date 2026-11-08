from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image


ROWS = 61
COLS = 241
LAT_NORTH = 79.0
LON_WEST = -105.0
STEP = 0.1


def ellipse(lat, lon, lat0, lon0, lat_radius, lon_radius):
    """Return whether a point is inside an axis-aligned geographic ellipse."""
    y = (lat - lat0) / lat_radius
    x = (lon - lon0) / lon_radius
    return x * x + y * y <= 1.0


def polygon_contains(lat, lon, polygon):
    """Ray-casting test for a hand-written island or mainland polygon."""
    inside = False
    j = len(polygon) - 1
    for i, (plat, plon) in enumerate(polygon):
        jlat, jlon = polygon[j]
        crosses = (plat > lat) != (jlat > lat)
        if crosses:
            crossing_lon = (jlon - plon) * (lat - plat) / (jlat - plat) + plon
            if lon < crossing_lon:
                inside = not inside
        j = i
    return inside


def in_any_ellipse(lat, lon, shapes):
    return any(ellipse(lat, lon, *shape) for shape in shapes)


def in_any_polygon(lat, lon, shapes):
    return any(polygon_contains(lat, lon, shape) for shape in shapes)
# Broad island bodies in the Canadian Arctic Archipelago. Coordinates are
# deliberately approximate, hand-authored geographic knowledge, not map data.
LARGE_ISLANDS = [
    # Banks Island
    (73.7, -121.0, 1.35, 1.7),
    # Victoria Island, split into overlapping lobes to retain its broad shape
    (72.2, -108.5, 2.6, 3.6),
    (70.5, -105.0, 1.8, 2.8),
    # Prince of Wales Island
    (73.0, -98.8, 1.8, 1.45),
    # Somerset Island
    (73.2, -94.8, 1.65, 1.0),
    # Boothia Peninsula and adjoining mainland
    (71.5, -95.0, 2.3, 1.05),
    # King William Island
    (69.1, -97.5, 1.35, 0.95),
    # Baffin Island, western margin within the frame
    (73.8, -81.2, 5.8, 1.25),
    # Melville Island
    (75.3, -111.0, 2.25, 2.8),
    # Bathurst Island
    (75.7, -101.8, 1.65, 1.45),
    # Devon Island
    (75.4, -84.8, 2.35, 2.05),
    # Ellesmere Island, southern and western portions
    (78.0, -82.8, 2.4, 2.5),
    (79.2, -85.5, 1.0, 2.1),
    # Axel Heiberg Island
    (78.1, -90.7, 1.9, 1.65),
    # Ellef Ringnes Island
    (78.8, -102.0, 1.15, 1.1),
    # Amund Ringnes Island
    (78.8, -96.7, 0.85, 0.75),
]
# Smaller islands and exposed land fragments visible at the requested scale.
SMALL_ISLANDS = [
    (77.9, -105.5, 0.45, 0.55),
    (77.6, -103.2, 0.35, 0.45),
    (77.3, -100.7, 0.32, 0.42),
    (76.9, -98.6, 0.28, 0.35),
    (76.8, -94.2, 0.48, 0.5),
    (76.4, -91.4, 0.32, 0.38),
    (76.0, -88.3, 0.25, 0.35),
    (75.4, -95.7, 0.32, 0.4),
    (74.7, -91.4, 0.25, 0.32),
    (74.3, -87.3, 0.3, 0.32),
    (73.9, -84.7, 0.22, 0.28),
    (73.0, -86.2, 0.35, 0.28),
    (72.4, -84.5, 0.25, 0.3),
    (71.9, -82.0, 0.2, 0.24),
    (71.2, -79.8, 0.35, 0.3),
    (70.2, -76.8, 0.28, 0.26),
    (69.5, -74.0, 0.2, 0.25),
    (74.8, -106.0, 0.22, 0.35),
    (74.2, -104.0, 0.2, 0.25),
    (73.8, -100.5, 0.2, 0.28),
    (72.6, -101.0, 0.26, 0.25),
    (71.3, -102.4, 0.22, 0.22),
    (70.5, -99.0, 0.2, 0.2),
]


def land_at(lat, lon):
    # Ellipse unions provide a compact deterministic approximation of the
    # remembered island silhouettes; water remains the default classification.
    if in_any_ellipse(lat, lon, LARGE_ISLANDS):
        return 1
    if in_any_ellipse(lat, lon, SMALL_ISLANDS):
        return 1
    return 0
def build_matrix():
    matrix = []
    for row in range(ROWS):
        lat = LAT_NORTH - row * STEP
        values = []
        for col in range(COLS):
            lon = LON_WEST + col * STEP
            values.append(land_at(lat, lon))
        matrix.append(values)
    return matrix


def write_csv(matrix, path):
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(matrix)


def write_images(matrix, output_dir):
    pixels = np.asarray(matrix, dtype=np.uint8) * 255
    image = Image.fromarray(pixels, mode="L")
    image.save(output_dir / "candidate.png")
    enlarged = image.resize((COLS * 8, ROWS * 8), Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")


def validate(matrix, output_dir):
    assert len(matrix) == ROWS
    assert all(len(row) == COLS for row in matrix)
    assert all(value in (0, 1) for row in matrix for value in row)

    with (output_dir / "candidate.csv").open(encoding="ascii") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == ROWS
    assert all(len(row) == COLS for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    original = Image.open(output_dir / "candidate.png")
    enlarged = Image.open(output_dir / "candidate_8x.png")
    assert original.size == (COLS, ROWS)
    assert enlarged.size == (COLS * 8, ROWS * 8)
    assert set(original.getdata()).issubset({0, 255})
    assert set(enlarged.getdata()).issubset({0, 255})


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()
    write_csv(matrix, output_dir / "candidate.csv")
    write_images(matrix, output_dir)
    validate(matrix, output_dir)
    print(f"validated {ROWS}x{COLS} matrix ({ROWS * COLS} cells)")


if __name__ == "__main__":
    main()
