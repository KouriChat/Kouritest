from pathlib import Path

import numpy as np
from PIL import Image


LAT_NORTH = 79.0
LAT_SOUTH = 73.0
LON_WEST = -105.0
LON_EAST = -81.0
STEP = 0.1

ROWS = 61
COLS = 241

# Each polygon is an approximate island outline, in (longitude, latitude)
# order. The outlines are intentionally hand-drawn approximations.
LAND_POLYGONS = [
    # Ellesmere Island, including its southern and western coasts.
    [
        (-90.0, 75.7), (-88.4, 76.1), (-87.1, 76.8), (-86.3, 77.5),
        (-85.6, 78.2), (-84.9, 79.0), (-83.7, 79.7), (-82.2, 80.4),
        (-80.1, 81.2), (-77.3, 82.2), (-73.0, 82.4), (-68.0, 81.5),
        (-65.5, 80.0), (-66.1, 78.5), (-68.0, 77.0), (-71.0, 75.9),
        (-75.0, 75.2), (-80.0, 75.1), (-85.5, 75.4),
    ],
    # Axel Heiberg Island.
    [
        (-97.0, 78.3), (-96.0, 79.1), (-95.2, 80.0), (-93.4, 80.5),
        (-91.2, 80.2), (-89.6, 79.4), (-89.8, 78.5), (-91.0, 77.7),
        (-93.0, 77.2), (-95.2, 77.5), (-96.5, 77.9),
    ],
    # Amund Ringnes Island.
    [
        (-100.0, 78.4), (-99.4, 79.0), (-98.0, 79.4), (-96.6, 79.1),
        (-96.1, 78.5), (-97.0, 78.0), (-98.4, 77.9), (-99.6, 78.1),
    ],
    # Ellef Ringnes Island.
    [
        (-105.0, 77.7), (-104.5, 78.4), (-103.4, 78.8), (-102.0, 78.7),
        (-100.9, 78.2), (-101.2, 77.5), (-102.4, 77.1), (-104.0, 77.2),
    ],
    # Bathurst Island.
    [
        (-101.0, 75.3), (-100.6, 76.0), (-99.7, 76.7), (-98.4, 76.9),
        (-97.0, 76.5), (-96.5, 75.8), (-97.2, 75.2), (-98.4, 74.9),
        (-100.0, 75.0),
    ],
    # Melville Island.
    [
        (-115.0, 75.0), (-114.3, 75.8), (-113.0, 76.6), (-111.4, 76.8),
        (-109.8, 76.4), (-108.5, 75.8), (-107.0, 75.4), (-106.0, 74.8),
        (-107.1, 74.1), (-109.0, 73.9), (-111.0, 74.2), (-113.0, 74.3),
        (-114.4, 74.6),
    ],
    # Borden Island.
    [
        (-115.6, 77.5), (-114.8, 78.0), (-113.5, 78.2), (-112.0, 77.9),
        (-111.7, 77.3), (-112.7, 76.9), (-114.2, 77.0), (-115.2, 77.2),
    ],
    # Mackenzie King Island.
    [
        (-121.5, 77.1), (-120.8, 77.8), (-119.2, 78.1), (-117.8, 77.8),
        (-117.3, 77.2), (-118.4, 76.8), (-120.0, 76.8),
    ],
    # Brock Island.
    [
        (-114.7, 77.8), (-114.2, 78.2), (-113.5, 78.1), (-113.2, 77.7),
        (-113.8, 77.4), (-114.4, 77.5),
    ],
    # Prince of Wales Island.
    [
        (-101.8, 73.2), (-101.0, 74.0), (-100.2, 74.7), (-98.8, 75.0),
        (-97.3, 74.8), (-96.5, 74.1), (-97.0, 73.4), (-98.4, 72.9),
        (-100.0, 72.8),
    ],
    # Somerset Island.
    [
        (-95.2, 73.3), (-94.7, 74.0), (-94.1, 74.9), (-92.9, 75.4),
        (-91.6, 75.2), (-91.0, 74.5), (-91.7, 73.8), (-92.7, 73.3),
        (-94.1, 73.1),
    ],
    # Devon Island.
    [
        (-90.4, 74.5), (-89.2, 75.1), (-88.6, 76.0), (-87.4, 76.7),
        (-85.4, 76.9), (-83.8, 76.4), (-83.0, 75.5), (-83.4, 74.7),
        (-85.0, 74.3), (-87.0, 74.2), (-89.0, 74.2),
    ],
    # Small islands in the western archipelago.
    [
        (-105.0, 74.0), (-104.4, 74.5), (-103.4, 74.7), (-102.4, 74.3),
        (-102.8, 73.7), (-104.0, 73.5),
    ],
]


def points_in_polygon(x_grid, y_grid, polygon):
    """Return a boolean mask for grid points inside one polygon."""
    inside = np.zeros(x_grid.shape, dtype=bool)
    vertices = polygon

    for index in range(len(vertices)):
        x1, y1 = vertices[index]
        x2, y2 = vertices[(index + 1) % len(vertices)]

        crosses_scanline = (y1 > y_grid) != (y2 > y_grid)
        x_intersection = (x2 - x1) * (y_grid - y1) / (
            (y2 - y1) if y2 != y1 else 1e-15
        ) + x1
        inside ^= crosses_scanline & (x_grid < x_intersection)

    return inside


def build_matrix():
    latitudes = LAT_NORTH - STEP * np.arange(ROWS)
    longitudes = LON_WEST + STEP * np.arange(COLS)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    matrix = np.zeros((ROWS, COLS), dtype=np.uint8)
    for polygon in LAND_POLYGONS:
        matrix[points_in_polygon(lon_grid, lat_grid, polygon)] = 1

    return matrix


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = build_matrix()

    if matrix.shape != (61, 241):
        raise RuntimeError(f"Unexpected matrix shape: {matrix.shape}")
    if matrix.size != 14701 or not np.isin(matrix, (0, 1)).all():
        raise RuntimeError("Matrix contains an invalid shape or value")

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", encoding="ascii", newline="") as handle:
        for row in matrix:
            handle.write(",".join(str(int(value)) for value in row) + "\n")

    # Matrix value 1 (land) is white; 0 (water) is black.
    pixels = matrix * 255
    image = Image.fromarray(pixels, mode="L")
    image.save(output_dir / "candidate.png")

    enlarged = image.resize((COLS * 8, ROWS * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")

    # Verify the written CSV and images.
    lines = csv_path.read_text(encoding="ascii").splitlines()
    if len(lines) != ROWS:
        raise RuntimeError("CSV row count is incorrect")
    for line in lines:
        values = line.split(",")
        if len(values) != COLS or any(value not in ("0", "1") for value in values):
            raise RuntimeError("CSV has an invalid row")

    if image.size != (241, 61):
        raise RuntimeError("Original PNG dimensions are incorrect")
    if enlarged.size != (1928, 488):
        raise RuntimeError("Enlarged PNG dimensions are incorrect")
    if set(image.getdata()) != {0, 255}:
        raise RuntimeError("Original PNG is not strictly black and white")

    print("Generated 61x241 matrix (14,701 values).")
    print(f"CSV: {csv_path}")
    print(f"PNG: {output_dir / 'candidate.png'} (241x61)")
    print(f"PNG: {output_dir / 'candidate_8x.png'} (1928x488, nearest neighbor)")


if __name__ == "__main__":
    main()
