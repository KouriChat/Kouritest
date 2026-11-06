import csv
import os

import numpy as np
from PIL import Image


ROWS = 64
COLS = 253
LON_MIN = 129.0
LON_MAX = 137.0
LAT_MAX = 35.0
LAT_MIN = 33.0

OUTPUT_DIR = "outputs"


def polygon_mask(lons, lats, polygon):
    """Return a point-in-polygon mask for a polygon in lon/lat coordinates."""
    polygon = np.asarray(polygon, dtype=np.float64)
    px = polygon[:, 0]
    py = polygon[:, 1]

    mask = np.zeros(lons.shape, dtype=bool)
    inside = np.zeros(lons.shape, dtype=bool)

    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi = px[i]
        yi = py[i]
        xj = px[j]
        yj = py[j]

        intersects = ((yi > lats) != (yj > lats))
        denominator = yj - yi
        if denominator == 0:
            denominator = 1e-15

        crossing_x = (xj - xi) * (lats - yi) / denominator + xi
        inside ^= intersects & (lons < crossing_x)
        j = i

    mask |= inside
    return mask


def add_polygon(land, lons, lats, points):
    land |= polygon_mask(lons, lats, points)


def subtract_polygon(land, lons, lats, points):
    land &= ~polygon_mask(lons, lats, points)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    latitudes = np.linspace(LAT_MAX, LAT_MIN, ROWS)
    longitudes = np.linspace(LON_MIN, LON_MAX, COLS)
    grid_lon, grid_lat = np.meshgrid(longitudes, latitudes)

    land = np.zeros((ROWS, COLS), dtype=bool)

    # Northern Kyushu, including the principal north-coast peninsulas.
    add_polygon(
        land,
        grid_lon,
        grid_lat,
        [
            (129.55, 33.55),
            (129.70, 33.78),
            (129.90, 33.98),
            (130.22, 34.04),
            (130.50, 34.00),
            (130.76, 33.88),
            (131.00, 33.70),
            (131.30, 33.58),
            (131.60, 33.48),
            (131.86, 33.30),
            (132.00, 33.00),
            (129.66, 33.00),
            (129.45, 33.20),
            (129.40, 33.40),
        ],
    )

    # Northern Kyushu coastal detail.
    add_polygon(
        land,
        grid_lon,
        grid_lat,
        [
            (129.72, 33.95),
            (129.92, 34.04),
            (130.16, 34.08),
            (130.38, 34.03),
            (130.52, 33.90),
            (130.42, 33.76),
            (130.18, 33.69),
            (129.96, 33.75),
            (129.76, 33.84),
        ],
    )

    # Tsushima.
    add_polygon(
        land,
        grid_lon,
        grid_lat,
        [
            (129.16, 34.18),
            (129.25, 34.30),
            (129.31, 34.52),
            (129.27, 34.70),
            (129.38, 34.74),
            (129.51, 34.58),
            (129.51, 34.37),
            (129.43, 34.18),
            (129.29, 34.08),
        ],
    )

    # Iki and nearby islands.
    add_polygon(
        land,
        grid_lon,
        grid_lat,
        [
            (129.62, 33.70),
            (129.76, 33.66),
            (129.91, 33.73),
            (129.96, 33.88),
            (129.86, 34.00),
            (129.70, 33.98),
            (129.61, 33.86),
        ],
    )
    add_polygon(
        land,
        grid_lon,
        grid_lat,
        [
            (129.42, 33.32),
            (129.51, 33.36),
            (129.57, 33.48),
            (129.53, 33.60),
            (129.43, 33.61),
            (129.35, 33.49),
        ],
    )
    add_polygon(
        land,
        grid_lon,
        grid_lat,
        [
            (129.32, 33.38),
            (129.37, 33.34),
            (129.40, 33.26),
            (129.34, 33.20),
            (129.28, 33.24),
            (129.27, 33.32),
        ],
    )

    # Western Honshu, with its Seto Inland Sea coastline.
    add_polygon(
        land,
        grid_lon,
        grid_lat,
        [
            (130.80, 35.00),
            (137.00, 35.00),
            (137.00, 34.96),
            (136.78, 34.93),
            (136.53, 34.88),
            (136.25, 34.84),
            (136.00, 34.80),
            (135.72, 34.76),
            (135.48, 34.72),
            (135.28, 34.68),
            (135.05, 34.67),
            (134.82, 34.65),
            (134.58, 34.63),
            (134.32, 34.60),
            (134.08, 34.56),
            (133.88, 34.53),
            (133.65, 34.46),
            (133.42, 34.45),
            (133.18, 34.43),
            (132.98, 34.40),
            (132.78, 34.38),
            (132.58, 34.34),
            (132.40, 34.36),
            (132.21, 34.34),
            (132.02, 34.26),
            (131.80, 34.21),
            (131.57, 34.17),
            (131.34, 34.12),
            (131.10, 34.08),
            (130.90, 34.13),
            (130.80, 34.30),
        ],
    )

    # Kii Peninsula and the eastern side of Osaka Bay.
    add_polygon(
        land,
        grid_lon,
        grid_lat,
        [
            (135.08, 34.30),
            (135.20, 34.42),
            (135.31, 34.53),
            (135.45, 34.65),
            (135.63, 34.73),
            (135.84, 34.79),
            (136.08, 34.83),
            (136.30, 34.88),
            (136.52, 34.95),
            (136.76, 35.00),
            (137.00, 35.00),
            (137.00, 33.00),
            (135.22, 33.00),
            (135.02, 33.32),
            (134.96, 33.65),
            (134.99, 33.96),
        ],
    )

    # Chita Peninsula and the eastern edge of Ise Bay.
    add_polygon(
        land,
        grid_lon,
        grid_lat,
        [
            (136.67, 35.00),
            (137.00, 35.00),
            (137.00, 34.42),
            (136.88, 34.48),
            (136.76, 34.61),
            (136.69, 34.78),
        ],
    )

    # Atsumi Peninsula at the eastern frame edge.
    add_polygon(
        land,
        grid_lon,
        grid_lat,
        [
            (136.88, 34.62),
            (137.00, 34.58),
            (137.00, 34.00),
            (136.88, 34.06),
            (136.78, 34.24),
            (136.78, 34.44),
        ],
    )

    # Shikoku.
    add_polygon(
        land,
        grid_lon,
        grid_lat,
        [
            (131.94, 33.72),
            (132.10, 33.86),
            (132.28, 34.00),
            (132.53, 34.07),
            (132.82, 34.08),
            (133.08, 34.17),
            (133.38, 34.22),
            (133.68, 34.23),
            (133.96, 34.29),
            (134.25, 34.30),
            (134.52, 34.25),
            (134.76, 34.12),
            (134.91, 33.94),
            (134.98, 33.70),
            (134.91, 33.46),
            (134.72, 33.31),
            (134.44, 33.20),
            (134.12, 33.13),
            (133.80, 33.13),
            (133.46, 33.18),
            (133.16, 33.25),
            (132.88, 33.31),
            (132.61, 33.30),
            (132.34, 33.24),
            (132.12, 33.31),
            (131.98, 33.48),
        ],
    )

    # Awaji Island.
    add_polygon(
        land,
        grid_lon,
        grid_lat,
        [
            (134.68, 34.24),
            (134.79, 34.37),
            (134.90, 34.56),
            (135.03, 34.65),
            (135.16, 34.57),
            (135.20, 34.40),
            (135.14, 34.21),
            (135.06, 34.08),
            (134.92, 34.04),
            (134.77, 34.10),
        ],
    )

    # Major islands of the western Seto Inland Sea.
    islands = [
        [
            (132.00, 33.86),
            (132.15, 33.80),
            (132.28, 33.83),
            (132.36, 33.94),
            (132.29, 34.04),
            (132.12, 34.05),
            (132.01, 33.97),
        ],
        [
            (132.20, 33.98),
            (132.35, 33.98),
            (132.46, 34.08),
            (132.42, 34.18),
            (132.28, 34.18),
            (132.18, 34.09),
        ],
        [
            (132.26, 34.22),
            (132.37, 34.20),
            (132.48, 34.27),
            (132.49, 34.38),
            (132.37, 34.42),
            (132.26, 34.34),
        ],
        [
            (132.42, 34.10),
            (132.57, 34.10),
            (132.65, 34.20),
            (132.60, 34.31),
            (132.47, 34.31),
            (132.39, 34.22),
        ],
        [
            (132.70, 34.08),
            (132.88, 34.09),
            (132.98, 34.18),
            (132.93, 34.29),
            (132.78, 34.31),
            (132.67, 34.22),
        ],
        [
            (132.88, 34.16),
            (133.03, 34.13),
            (133.13, 34.22),
            (133.09, 34.32),
            (132.97, 34.34),
            (132.87, 34.27),
        ],
        [
            (133.00, 34.24),
            (133.13, 34.21),
            (133.29, 34.27),
            (133.33, 34.38),
            (133.20, 34.45),
            (133.07, 34.39),
        ],
        [
            (133.18, 34.26),
            (133.36, 34.24),
            (133.49, 34.32),
            (133.49, 34.43),
            (133.34, 34.48),
            (133.20, 34.40),
        ],
        [
            (133.44, 34.17),
            (133.58, 34.15),
            (133.67, 34.23),
            (133.63, 34.32),
            (133.48, 34.34),
            (133.40, 34.26),
        ],
        [
            (133.55, 34.29),
            (133.70, 34.26),
            (133.82, 34.33),
            (133.80, 34.43),
            (133.66, 34.46),
            (133.54, 34.39),
        ],
        [
            (133.72, 34.16),
            (133.86, 34.15),
            (133.94, 34.24),
            (133.90, 34.33),
            (133.76, 34.34),
            (133.68, 34.26),
        ],
        [
            (133.88, 34.24),
            (134.06, 34.23),
            (134.15, 34.31),
            (134.10, 34.40),
            (133.96, 34.42),
            (133.86, 34.34),
        ],
        [
            (134.02, 34.40),
            (134.18, 34.40),
            (134.27, 34.48),
            (134.22, 34.56),
            (134.07, 34.56),
            (133.99, 34.49),
        ],
        [
            (134.25, 34.37),
            (134.39, 34.35),
            (134.49, 34.42),
            (134.47, 34.51),
            (134.32, 34.53),
            (134.23, 34.46),
        ],
        [
            (134.36, 34.50),
            (134.51, 34.48),
            (134.59, 34.55),
            (134.54, 34.64),
            (134.40, 34.64),
            (134.32, 34.57),
        ],
        [
            (134.72, 34.31),
            (134.87, 34.29),
            (134.95, 34.36),
            (134.92, 34.45),
            (134.79, 34.45),
            (134.70, 34.39),
        ],
        [
            (134.00, 34.08),
            (134.12, 34.08),
            (134.20, 34.16),
            (134.17, 34.25),
            (134.03, 34.24),
            (133.96, 34.16),
        ],
        [
            (133.70, 33.98),
            (133.84, 33.95),
            (133.91, 34.03),
            (133.88, 34.11),
            (133.75, 34.12),
            (133.68, 34.06),
        ],
        [
            (134.42, 34.12),
            (134.57, 34.10),
            (134.65, 34.19),
            (134.61, 34.28),
            (134.47, 34.29),
            (134.39, 34.21),
        ],
        [
            (134.06, 33.98),
            (134.18, 33.96),
            (134.25, 34.02),
            (134.22, 34.10),
            (134.10, 34.11),
            (134.03, 34.05),
        ],
    ]

    for island in islands:
        add_polygon(land, grid_lon, grid_lat, island)

    # Smaller islands in Hiroshima and Ehime waters.
    small_islands = [
        [(131.83, 34.06), (131.92, 34.05), (131.97, 34.12), (131.93, 34.19), (131.84, 34.18)],
        [(132.05, 34.22), (132.14, 34.20), (132.19, 34.27), (132.14, 34.34), (132.05, 34.31)],
        [(132.63, 34.32), (132.71, 34.31), (132.76, 34.38), (132.71, 34.45), (132.63, 34.42)],
        [(132.78, 34.36), (132.86, 34.35), (132.90, 34.42), (132.85, 34.48), (132.77, 34.45)],
        [(133.14, 34.43), (133.23, 34.42), (133.28, 34.50), (133.22, 34.56), (133.13, 34.52)],
        [(133.42, 34.47), (133.50, 34.46), (133.56, 34.53), (133.51, 34.60), (133.42, 34.56)],
        [(133.60, 34.49), (133.70, 34.48), (133.77, 34.55), (133.72, 34.63), (133.62, 34.59)],
        [(133.86, 34.45), (133.95, 34.44), (134.00, 34.51), (133.95, 34.59), (133.86, 34.55)],
        [(134.17, 34.27), (134.25, 34.25), (134.31, 34.32), (134.28, 34.39), (134.19, 34.37)],
        [(134.55, 34.27), (134.64, 34.25), (134.70, 34.32), (134.66, 34.39), (134.56, 34.36)],
        [(134.88, 34.00), (134.98, 33.98), (135.03, 34.05), (134.99, 34.12), (134.90, 34.10)],
        [(131.62, 33.92), (131.71, 33.91), (131.77, 33.98), (131.72, 34.05), (131.63, 34.02)],
    ]

    for island in small_islands:
        add_polygon(land, grid_lon, grid_lat, island)

    # Southern Japanese islands visible near the lower western frame.
    add_polygon(
        land,
        grid_lon,
        grid_lat,
        [
            (129.48, 33.00),
            (129.62, 33.00),
            (129.68, 33.08),
            (129.62, 33.16),
            (129.51, 33.15),
            (129.45, 33.08),
        ],
    )
    add_polygon(
        land,
        grid_lon,
        grid_lat,
        [
            (129.84, 33.00),
            (129.96, 33.00),
            (130.03, 33.09),
            (129.97, 33.17),
            (129.86, 33.14),
            (129.80, 33.07),
        ],
    )

    # Approximate inland-water correction for the southern end of Lake Biwa.
    subtract_polygon(
        land,
        grid_lon,
        grid_lat,
        [
            (135.82, 35.00),
            (136.03, 35.00),
            (136.08, 35.07),
            (136.02, 35.13),
            (135.89, 35.13),
            (135.82, 35.07),
        ],
    )

    matrix = land.astype(np.uint8)

    if matrix.shape != (64, 253):
        raise RuntimeError("Unexpected matrix shape")
    if matrix.size != 16192:
        raise RuntimeError("Unexpected matrix size")
    if not np.all((matrix == 0) | (matrix == 1)):
        raise RuntimeError("Matrix contains values other than 0 and 1")

    csv_path = os.path.join(OUTPUT_DIR, "candidate.csv")
    with open(csv_path, "w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for row in matrix:
            writer.writerow(row.tolist())

    original_pixels = np.where(matrix == 1, 255, 0).astype(np.uint8)
    original_image = Image.fromarray(original_pixels, mode="L")
    original_path = os.path.join(OUTPUT_DIR, "candidate.png")
    original_image.save(original_path, format="PNG")

    enlarged_image = original_image.resize(
        (COLS * 8, ROWS * 8),
        resample=Image.Resampling.NEAREST,
    )
    enlarged_path = os.path.join(OUTPUT_DIR, "candidate_8x.png")
    enlarged_image.save(enlarged_path, format="PNG")

    if original_image.size != (253, 64):
        raise RuntimeError("Unexpected original image size")
    if enlarged_image.size != (2024, 512):
        raise RuntimeError("Unexpected enlarged image size")

    original_values = set(np.asarray(original_image).ravel().tolist())
    enlarged_values = set(np.asarray(enlarged_image).ravel().tolist())
    if not original_values.issubset({0, 255}):
        raise RuntimeError("Original image is not strictly black and white")
    if not enlarged_values.issubset({0, 255}):
        raise RuntimeError("Enlarged image is not strictly black and white")

    enlarged_array = np.asarray(enlarged_image)
    expected = np.repeat(np.repeat(original_pixels, 8, axis=0), 8, axis=1)
    if not np.array_equal(enlarged_array, expected):
        raise RuntimeError("Enlarged image is not an exact nearest-neighbor enlargement")


if __name__ == "__main__":
    main()
