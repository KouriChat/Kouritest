from math import cos, pi, sin
from pathlib import Path
import csv

import numpy as np
from PIL import Image


# Hand-written approximate coastlines, in (longitude, latitude) degrees.
# The polygons are intentionally approximate and are clipped to the requested
# sampling window by the grid itself.
HONSHU = [
    (130.72, 33.91), (130.90, 33.96), (131.10, 33.94),
    (131.30, 34.00), (131.55, 34.03), (131.85, 34.05),
    (132.10, 34.06), (132.35, 34.10), (132.60, 34.15),
    (132.85, 34.16), (133.10, 34.18), (133.35, 34.22),
    (133.60, 34.25), (133.85, 34.30), (134.10, 34.32),
    (134.35, 34.35), (134.55, 34.38), (134.75, 34.43),
    (134.90, 34.49), (135.00, 34.58), (135.12, 34.65),
    (135.26, 34.66), (135.37, 34.61), (135.40, 34.52),
    (135.30, 34.43), (135.15, 34.37), (135.06, 34.28),
    (135.13, 34.16), (135.31, 34.00), (135.49, 33.82),
    (135.72, 33.63), (135.92, 33.48), (136.12, 33.43),
    (136.30, 33.55), (136.49, 33.75), (136.67, 33.99),
    (136.82, 34.23), (136.88, 34.42), (136.84, 34.57),
    (136.91, 34.70), (137.00, 34.77),
    (137.00, 35.00), (130.72, 35.00),
]

SHIKOKU = [
    (132.00, 34.25), (132.18, 34.28), (132.42, 34.30),
    (132.70, 34.31), (132.98, 34.31), (133.25, 34.30),
    (133.53, 34.29), (133.80, 34.27), (134.05, 34.25),
    (134.30, 34.24), (134.56, 34.23), (134.73, 34.21),
    (134.72, 34.03), (134.62, 33.87), (134.48, 33.70),
    (134.28, 33.55), (134.03, 33.45), (133.77, 33.39),
    (133.49, 33.38), (133.20, 33.40), (132.92, 33.43),
    (132.66, 33.48), (132.42, 33.55), (132.21, 33.65),
    (132.07, 33.78), (132.00, 33.96), (131.98, 34.10),
]

KYUSHU = [
    (129.72, 32.95), (129.78, 33.19), (129.83, 33.40),
    (129.88, 33.61), (129.98, 33.79), (130.13, 33.91),
    (130.31, 33.98), (130.50, 34.00), (130.69, 33.97),
    (130.86, 33.89), (131.00, 33.76), (131.10, 33.59),
    (131.20, 33.40), (131.30, 33.20), (131.38, 33.00),
    (130.82, 32.95), (130.25, 32.95),
]

# Awaji, Shodoshima and representative islands of the Seto Inland Sea.
ISLANDS = [
    (134.93, 34.47, 0.12, 0.33),  # Awaji
    (134.30, 34.52, 0.15, 0.09),  # Shodoshima
    (133.02, 34.24, 0.10, 0.07),  # Osakikamijima area
    (132.92, 34.25, 0.07, 0.06),  # Okunoshima area
    (132.88, 34.22, 0.08, 0.07),  # Takehara islands
    (132.75, 34.17, 0.10, 0.12),  # Etajima
    (132.32, 34.28, 0.07, 0.12),  # Miyajima
    (132.98, 34.20, 0.07, 0.06),  # Innoshima
    (133.08, 34.20, 0.08, 0.07),  # Ikuchijima
    (133.12, 34.25, 0.09, 0.08),  # Omishima
    (133.25, 34.20, 0.08, 0.06),  # Hakatajima
    (133.42, 34.17, 0.08, 0.06),  # Oshima
    (133.57, 34.15, 0.06, 0.05),  # Geiyo islands
    (133.80, 34.25, 0.05, 0.04),  # Bisan islands
    (134.10, 34.35, 0.07, 0.05),  # Naoshima area
    (134.00, 34.43, 0.06, 0.04),  # Teshima area
    (134.55, 34.45, 0.05, 0.04),  # eastern Seto islands
    (131.95, 34.03, 0.06, 0.05),  # Suo-Oshima area
    (131.75, 33.98, 0.06, 0.04),
    (131.50, 33.93, 0.05, 0.04),
    (130.05, 33.65, 0.06, 0.08),  # Iki-like near-window island
    (129.90, 33.15, 0.07, 0.12),
]

# Small representative island groups in Ise Bay and the Kii Channel.
ISLANDS += [
    (136.70, 34.58, 0.05, 0.04),
    (136.55, 34.55, 0.04, 0.03),
    (134.95, 33.95, 0.05, 0.04),
    (135.05, 34.02, 0.05, 0.04),
]


def point_in_polygon(lon, lat, polygon):
    """Ray-crossing point-in-polygon test."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > lat) != (yj > lat):
            crossing_lon = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing_lon:
                inside = not inside
        j = i
    return inside


def ellipse_polygon(lon, lat, rx, ry, points=24):
    return [
        (
            lon + rx * cos(2 * pi * i / points),
            lat + ry * sin(2 * pi * i / points),
        )
        for i in range(points)
    ]


def is_land(lon, lat):
    if point_in_polygon(lon, lat, HONSHU):
        return 1
    if point_in_polygon(lon, lat, SHIKOKU):
        return 1
    if point_in_polygon(lon, lat, KYUSHU):
        return 1

    for island_lon, island_lat, rx, ry in ISLANDS:
        if point_in_polygon(
            lon, lat, ellipse_polygon(island_lon, island_lat, rx, ry)
        ):
            return 1
    return 0


def main():
    rows, cols = 64, 253
    step = 2 / 63
    latitudes = [35.0 - i * step for i in range(rows)]
    longitudes = [129.0 + j * step for j in range(cols)]

    matrix = np.array(
        [[is_land(lon, lat) for lon in longitudes] for lat in latitudes],
        dtype=np.uint8,
    )
    assert matrix.shape == (64, 253)
    assert matrix.size == 16192
    assert np.isin(matrix, (0, 1)).all()

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Matrix value 1 is white (land); value 0 is black (water).
    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(output_dir / "candidate.png")

    enlarged = image.resize((253 * 8, 64 * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")

    # Validate the written artifacts.
    with (output_dir / "candidate.csv").open("r", newline="", encoding="ascii") as f:
        csv_rows = list(csv.reader(f))
    assert len(csv_rows) == 64
    assert all(len(row) == 253 for row in csv_rows)
    assert all(value in {"0", "1"} for row in csv_rows for value in row)

    check_original = Image.open(output_dir / "candidate.png").convert("RGB")
    check_enlarged = Image.open(output_dir / "candidate_8x.png").convert("RGB")
    assert check_original.size == (253, 64)
    assert check_enlarged.size == (2024, 512)
    assert set(check_original.getdata()) <= {(0, 0, 0), (255, 255, 255)}
    assert check_enlarged.resize((253, 64), Image.Resampling.NEAREST).tobytes() == (
        check_original.resize((253, 64), Image.Resampling.NEAREST).tobytes()
    )
    print("Created outputs/candidate.csv, candidate.png, candidate_8x.png")


if __name__ == "__main__":
    main()
