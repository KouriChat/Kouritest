from pathlib import Path
import csv

import numpy as np
from PIL import Image


# Coordinate convention: (longitude, latitude), with west longitudes negative.
# These are deliberately hand-written approximate outlines based on geographic
# knowledge. They are not imported from any map or geographic data source.

def ellipse(cx, cy, rx, ry, n=32):
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return [
        (cx + rx * float(np.cos(a)), cy + ry * float(np.sin(a)))
        for a in angles
    ]


# Approximate mainland coast east of Hecate Strait.
MAINLAND = [
    (-130.18, 55.00),
    (-130.17, 54.92),
    (-130.28, 54.82),
    (-130.36, 54.70),
    (-130.30, 54.61),
    (-130.43, 54.50),
    (-130.42, 54.37),
    (-130.33, 54.27),
    (-130.22, 54.18),
    (-130.05, 54.10),
    (-129.89, 54.00),
    (-129.72, 53.91),
    (-129.57, 53.80),
    (-129.43, 53.68),
    (-129.28, 53.55),
    (-129.13, 53.41),
    (-129.02, 53.25),
    (-128.94, 53.00),
    (-129.00, 53.00),
    (-129.00, 55.00),
]


# Approximate islands in and around Hecate Strait.
DUNDAS_ISLAND = [
    (-131.00, 54.62),
    (-130.92, 54.73),
    (-130.76, 54.78),
    (-130.62, 54.74),
    (-130.55, 54.63),
    (-130.63, 54.52),
    (-130.78, 54.48),
    (-130.92, 54.52),
]

MELVILLE_ISLAND = [
    (-130.93, 54.80),
    (-130.83, 54.86),
    (-130.70, 54.84),
    (-130.64, 54.76),
    (-130.70, 54.69),
    (-130.84, 54.70),
]

BANKS_ISLAND = [
    (-130.32, 54.53),
    (-130.15, 54.58),
    (-129.85, 54.56),
    (-129.55, 54.43),
    (-129.36, 54.22),
    (-129.28, 53.98),
    (-129.40, 53.83),
    (-129.65, 53.78),
    (-129.95, 53.84),
    (-130.18, 54.00),
    (-130.31, 54.22),
]

PORCHER_ISLAND = [
    (-130.54, 54.10),
    (-130.30, 54.18),
    (-129.98, 54.13),
    (-129.78, 54.00),
    (-129.82, 53.83),
    (-130.05, 53.76),
    (-130.35, 53.81),
    (-130.55, 53.94),
]

PITT_ISLAND = [
    (-130.50, 53.72),
    (-130.25, 53.76),
    (-129.95, 53.68),
    (-129.75, 53.52),
    (-129.70, 53.33),
    (-129.85, 53.18),
    (-130.12, 53.14),
    (-130.38, 53.27),
    (-130.55, 53.48),
]

GIL_ISLAND = [
    (-129.65, 53.75),
    (-129.40, 53.80),
    (-129.15, 53.68),
    (-129.02, 53.48),
    (-129.18, 53.29),
    (-129.48, 53.35),
    (-129.68, 53.55),
]

PRINCESS_ROYAL_WESTERN_PART = [
    (-129.72, 53.85),
    (-129.48, 53.92),
    (-129.26, 53.84),
    (-129.10, 53.68),
    (-129.02, 53.46),
    (-129.10, 53.22),
    (-129.30, 53.08),
    (-129.55, 53.17),
    (-129.70, 53.38),
    (-129.78, 53.62),
]

CHATHAM_ISLAND = [
    (-130.62, 54.38),
    (-130.48, 54.45),
    (-130.33, 54.40),
    (-130.27, 54.29),
    (-130.34, 54.18),
    (-130.50, 54.17),
    (-130.62, 54.25),
]

STEPHENS_ISLAND = [
    (-130.72, 54.04),
    (-130.59, 54.11),
    (-130.45, 54.06),
    (-130.41, 53.95),
    (-130.50, 53.87),
    (-130.66, 53.90),
]

# Small western fragment of Graham Island near the western edge of the frame.
# Most of Haida Gwaii lies west of the requested longitude range.
GRAHAM_EASTERN_FRAGMENT = [
    (-131.20, 54.90),
    (-131.08, 54.98),
    (-131.00, 54.78),
    (-130.985, 54.60),
    (-130.98, 54.43),
    (-131.00, 54.27),
    (-130.99, 54.16),
    (-131.08, 54.06),
    (-131.20, 54.00),
]

LAND_POLYGONS = [
    MAINLAND,
    DUNDAS_ISLAND,
    MELVILLE_ISLAND,
    BANKS_ISLAND,
    PORCHER_ISLAND,
    PITT_ISLAND,
    GIL_ISLAND,
    PRINCESS_ROYAL_WESTERN_PART,
    CHATHAM_ISLAND,
    STEPHENS_ISLAND,
    GRAHAM_EASTERN_FRAGMENT,

    # Approximate small islands and island groups.
    ellipse(-130.86, 54.31, 0.060, 0.045),
    ellipse(-130.63, 54.57, 0.075, 0.055),
    ellipse(-130.77, 54.24, 0.050, 0.035),
    ellipse(-130.52, 53.76, 0.085, 0.050),
    ellipse(-130.72, 53.63, 0.055, 0.040),
    ellipse(-130.88, 53.95, 0.045, 0.032),
    ellipse(-129.82, 53.70, 0.050, 0.035),
]


# Approximate marine inlets cutting into the mainland polygon.
# These are applied after land polygons, so they remain water.
WATER_CUTS = [
    [
        (-130.08, 55.00),
        (-129.46, 55.00),
        (-129.46, 54.88),
        (-129.62, 54.82),
        (-129.78, 54.75),
        (-129.90, 54.67),
        (-130.02, 54.72),
        (-130.16, 54.86),
    ],
    [
        (-129.66, 54.19),
        (-129.39, 54.19),
        (-129.39, 54.03),
        (-129.54, 53.95),
        (-129.67, 53.87),
        (-129.77, 53.91),
        (-129.81, 54.04),
    ],
    [
        (-129.55, 53.91),
        (-129.30, 53.91),
        (-129.18, 53.81),
        (-129.29, 53.72),
        (-129.47, 53.72),
        (-129.61, 53.80),
    ],
]


def point_in_polygon(x, y, polygon):
    """Ray-casting point test; boundary points are treated as land."""
    inside = False
    j = len(polygon) - 1

    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        cross = (x - xi) * (yj - yi) - (y - yi) * (xj - xi)
        if abs(cross) <= 1e-12:
            if (
                min(xi, xj) - 1e-12 <= x <= max(xi, xj) + 1e-12
                and min(yi, yj) - 1e-12 <= y <= max(yi, yj) + 1e-12
            ):
                return True

        crosses_horizontal_ray = (yi > y) != (yj > y)
        if crosses_horizontal_ray:
            x_intersection = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersection:
                inside = not inside

        j = i

    return inside


def is_land(longitude, latitude):
    # The rule is a union of hand-written land outlines.
    land = any(
        point_in_polygon(longitude, latitude, polygon)
        for polygon in LAND_POLYGONS
    )

    # Marine cuts override the mainland outline.
    if land:
        for water_polygon in WATER_CUTS:
            if point_in_polygon(longitude, latitude, water_polygon):
                return 0

    return 1 if land else 0


def build_matrix():
    # North to south and west to east, exactly 127 samples each.
    latitudes = np.linspace(55.0, 53.0, 127)
    longitudes = np.linspace(-131.0, -129.0, 127)

    matrix = np.zeros((127, 127), dtype=np.uint8)

    for row, latitude in enumerate(latitudes):
        for col, longitude in enumerate(longitudes):
            matrix[row, col] = is_land(
                float(longitude),
                float(latitude),
            )

    return matrix


def write_csv(matrix, path):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for row in matrix:
            writer.writerow([int(value) for value in row])


def validate_csv(path):
    with path.open("r", newline="") as handle:
        rows = list(csv.reader(handle))

    assert len(rows) == 127
    assert all(len(row) == 127 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert all(value != "" for row in rows for value in row)


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = build_matrix()

    # Matrix validation.
    assert matrix.shape == (127, 127)
    assert matrix.size == 16129
    assert set(np.unique(matrix).tolist()).issubset({0, 1})

    csv_path = output_dir / "candidate.csv"
    raw_png_path = output_dir / "candidate.png"
    enlarged_png_path = output_dir / "candidate_8x.png"

    write_csv(matrix, csv_path)
    validate_csv(csv_path)

    # Pillow mode L: 0 is black water, 255 is white land.
    pixel_values = (matrix * 255).astype(np.uint8)
    raw_image = Image.fromarray(pixel_values, mode="L")
    raw_image.save(raw_png_path)

    assert raw_image.size == (127, 127)
    assert set(np.unique(np.asarray(raw_image)).tolist()).issubset({0, 255})

    # Explicit nearest-neighbor enlargement.
    enlarged_image = raw_image.resize(
        (127 * 8, 127 * 8),
        resample=Image.Resampling.NEAREST,
    )
    enlarged_image.save(enlarged_png_path)

    expected_enlarged = np.repeat(
        np.repeat(pixel_values, 8, axis=0),
        8,
        axis=1,
    )

    assert enlarged_image.size == (1016, 1016)
    assert set(np.unique(np.asarray(enlarged_image)).tolist()).issubset({0, 255})
    assert np.array_equal(
        np.asarray(enlarged_image),
        expected_enlarged,
    )

    print("Generated:")
    print(f"  {csv_path}       127 rows x 127 columns")
    print(f"  {raw_png_path}   127 x 127 pixels")
    print(f"  {enlarged_png_path} 1016 x 1016 pixels")
    print(f"  valid values: {matrix.size}")
    print(f"  land points: {int(matrix.sum())}")
    print(f"  water points: {int(matrix.size - matrix.sum())}")


if __name__ == "__main__":
    main()
