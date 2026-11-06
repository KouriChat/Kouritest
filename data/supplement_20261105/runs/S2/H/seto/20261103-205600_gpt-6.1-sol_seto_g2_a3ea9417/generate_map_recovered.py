from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image


ROWS = 64
COLS = 253
STEP = 2.0 / 63.0

# Polygons are written as (longitude, latitude), in degrees.
# These are hand-estimated coastline outlines, not sourced from map data.
LAND_POLYGONS = [
    # Western and southern Honshu, north of the Seto Inland Sea.
    [
        (130.85, 33.90), (130.95, 34.00), (131.15, 34.05),
        (131.35, 34.03), (131.55, 34.00), (131.75, 34.03),
        (131.95, 34.10), (132.15, 34.14), (132.35, 34.15),
        (132.55, 34.18), (132.75, 34.20), (132.95, 34.20),
        (133.15, 34.22), (133.35, 34.25), (133.55, 34.27),
        (133.75, 34.30), (133.95, 34.33), (134.15, 34.38),
        (134.35, 34.40), (134.55, 34.43), (134.75, 34.45),
        (134.95, 34.50), (135.10, 34.55), (135.25, 34.62),
        (135.35, 34.70), (135.45, 34.80), (135.50, 35.00),
        (129.0, 35.0), (129.0, 33.0), (130.0, 33.0),
        (130.40, 33.35), (130.65, 33.65), (130.85, 33.90),
    ],

    # Kii Peninsula, including its Pacific-facing coast.
    [
        (135.18, 34.68), (135.30, 34.64), (135.38, 34.55),
        (135.40, 34.40), (135.30, 34.20), (135.16, 34.00),
        (135.05, 33.80), (135.00, 33.62), (135.10, 33.48),
        (135.28, 33.40), (135.50, 33.38), (135.72, 33.42),
        (135.90, 33.50), (136.08, 33.62), (136.22, 33.78),
        (136.36, 33.95), (136.55, 34.15), (136.70, 34.35),
        (136.82, 34.55), (136.92, 34.75), (137.0, 34.82),
        (137.0, 35.0), (135.45, 35.0), (135.18, 34.68),
    ],

    # Chugoku coast around the western mouth of the Seto Inland Sea.
    [
        (130.70, 33.85), (130.85, 33.92), (131.00, 33.98),
        (131.18, 34.00), (131.35, 33.98), (131.48, 33.92),
        (131.50, 34.20), (131.40, 34.45), (131.20, 34.75),
        (130.80, 35.00), (129.0, 35.0), (129.0, 34.0),
        (130.70, 33.85),
    ],

    # Northern Kyushu.
    [
        (129.0, 33.0), (131.0, 33.0), (131.15, 33.25),
        (131.20, 33.55), (131.18, 33.85), (131.12, 34.05),
        (130.95, 34.12), (130.72, 34.08), (130.52, 34.00),
        (130.30, 33.92), (130.05, 33.88), (129.80, 33.85),
        (129.55, 33.80), (129.35, 33.70), (129.15, 33.55),
        (129.0, 33.40), (129.0, 33.0),
    ],

    # Southern and eastern Kyushu.
    [
        (130.15, 33.90), (130.40, 33.95), (130.65, 34.00),
        (130.85, 34.02), (131.02, 33.95), (131.15, 33.78),
        (131.18, 33.50), (131.12, 33.25), (130.95, 33.00),
        (129.80, 33.00), (129.85, 33.25), (130.00, 33.50),
        (130.15, 33.70), (130.15, 33.90),
    ],

    # Shikoku.
    [
        (132.00, 33.55), (132.12, 33.72), (132.28, 33.82),
        (132.50, 33.88), (132.72, 33.92), (132.95, 33.95),
        (133.18, 33.95), (133.42, 33.90), (133.66, 33.88),
        (133.90, 33.85), (134.12, 33.82), (134.32, 33.78),
        (134.50, 33.72), (134.62, 33.63), (134.60, 33.45),
        (134.45, 33.30), (134.18, 33.22), (133.90, 33.20),
        (133.60, 33.22), (133.30, 33.23), (133.00, 33.25),
        (132.70, 33.28), (132.42, 33.32), (132.18, 33.38),
        (132.00, 33.55),
    ],

    # Awaji Island.
    [
        (134.78, 34.18), (134.88, 34.24), (134.95, 34.32),
        (134.96, 34.42), (134.91, 34.52), (134.83, 34.55),
        (134.76, 34.49), (134.73, 34.38), (134.74, 34.27),
        (134.78, 34.18),
    ],

    # Tsushima.
    [
        (129.18, 34.15), (129.23, 34.23), (129.31, 34.33),
        (129.36, 34.46), (129.39, 34.60), (129.37, 34.72),
        (129.31, 34.78), (129.24, 34.72), (129.20, 34.58),
        (129.17, 34.42), (129.16, 34.28), (129.18, 34.15),
    ],

    # Iki Island.
    [
        (129.65, 33.72), (129.72, 33.74), (129.79, 33.80),
        (129.81, 33.88), (129.78, 33.94), (129.70, 33.95),
        (129.64, 33.90), (129.62, 33.82), (129.65, 33.72),
    ],
]

# Elliptical island approximations: (center_lon, center_lat, radius_lon, radius_lat).
# The list focuses on the larger islands and island groups in the Seto Inland Sea.
ISLAND_ELLIPSES = [
    # Suo-nada and western Seto.
    (131.95, 33.98, 0.075, 0.105),  # Suo-Oshima
    (132.10, 34.02, 0.035, 0.050),
    (132.32, 34.04, 0.045, 0.055),
    (132.42, 34.12, 0.035, 0.045),
    (132.62, 34.18, 0.040, 0.050),
    (132.76, 34.22, 0.030, 0.040),

    # Hiroshima Bay and Geiyo Islands.
    (132.42, 34.22, 0.025, 0.035),  # Etajima vicinity
    (132.55, 34.18, 0.025, 0.035),
    (132.78, 34.25, 0.025, 0.035),
    (132.88, 34.26, 0.030, 0.040),  # Osakikamijima vicinity
    (132.98, 34.24, 0.045, 0.065),  # Omishima vicinity
    (133.08, 34.28, 0.030, 0.045),
    (133.18, 34.27, 0.040, 0.055),  # Innoshima vicinity
    (133.27, 34.28, 0.035, 0.055),  # Ikuchijima vicinity
    (133.37, 34.30, 0.025, 0.040),  # Mukaishima vicinity

    # Bisan Seto and Shodoshima.
    (133.55, 34.42, 0.030, 0.040),
    (133.72, 34.43, 0.025, 0.035),
    (133.82, 34.45, 0.028, 0.040),
    (133.95, 34.48, 0.030, 0.045),
    (134.08, 34.50, 0.025, 0.035),
    (134.18, 34.55, 0.095, 0.085),  # Shodoshima
    (134.05, 34.40, 0.030, 0.040),
    (134.22, 34.42, 0.025, 0.035),
    (134.32, 34.45, 0.025, 0.035),
    (134.42, 34.48, 0.025, 0.035),

    # Naoshima and nearby islands.
    (133.98, 34.46, 0.025, 0.030),
    (134.00, 34.43, 0.020, 0.025),
    (134.07, 34.44, 0.020, 0.025),

    # Small island groups south of Hiroshima and in the western Inland Sea.
    (132.25, 33.98, 0.025, 0.035),
    (132.48, 33.96, 0.030, 0.040),
    (132.68, 34.02, 0.025, 0.035),
    (132.90, 34.08, 0.025, 0.035),
    (133.14, 34.10, 0.025, 0.035),
    (133.42, 34.12, 0.025, 0.035),
    (133.70, 34.16, 0.025, 0.035),
    (133.95, 34.18, 0.025, 0.035),
    (134.24, 34.20, 0.025, 0.035),
    (134.50, 34.23, 0.025, 0.035),

    # Ise Bay / Mikawa Bay islands and capes, approximated conservatively.
    (136.82, 34.68, 0.045, 0.055),  # Chita Peninsula tip vicinity
    (136.98, 34.65, 0.050, 0.060),  # Atsumi Peninsula vicinity
    (136.70, 34.42, 0.025, 0.030),
]


def point_in_polygon(lon, lat, polygon):
    """Ray-casting point-in-polygon test."""
    inside = False
    x, y = lon, lat
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            crossing_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < crossing_x:
                inside = not inside
        j = i
    return inside


def is_land(lon, lat):
    if any(point_in_polygon(lon, lat, polygon) for polygon in LAND_POLYGONS):
        return 1

    for center_lon, center_lat, radius_lon, radius_lat in ISLAND_ELLIPSES:
        dx = (lon - center_lon) / radius_lon
        dy = (lat - center_lat) / radius_lat
        if dx * dx + dy * dy <= 1.0:
            return 1
    return 0


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = np.zeros((ROWS, COLS), dtype=np.uint8)
    for row in range(ROWS):
        lat = 35.0 - row * STEP
        for col in range(COLS):
            lon = 129.0 + col * STEP
            matrix[row, col] = is_land(lon, lat)

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Matrix value 1 maps to white; value 0 maps to black.
    image = Image.fromarray(matrix * 255, mode="L")
    png_path = output_dir / "candidate.png"
    image.save(png_path)

    enlarged = image.resize(
        (COLS * 8, ROWS * 8),
        resample=Image.Resampling.NEAREST,
    )
    enlarged.save(output_dir / "candidate_8x.png")

    # Validate the requested dimensions, values, CSV shape, and image palette.
    assert matrix.shape == (64, 253)
    assert int(matrix.sum() + (matrix == 0).sum()) == 64 * 253
    assert set(np.unique(matrix).tolist()) <= {0, 1}
    assert image.size == (253, 64)
    assert enlarged.size == (2024, 512)

    with csv_path.open("r", newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 64
    assert all(len(row) == 253 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert set(image.getdata()) <= {0, 255}
    assert set(enlarged.getdata()) <= {0, 255}

    print(f"Wrote {csv_path}, {png_path}, and {output_dir / 'candidate_8x.png'}")
    print(f"Matrix: {ROWS} x {COLS} = {ROWS * COLS} values")
    print(f"Images: {image.size[0]} x {image.size[1]}, {enlarged.size[0]} x {enlarged.size[1]}")


if __name__ == "__main__":
    main()
