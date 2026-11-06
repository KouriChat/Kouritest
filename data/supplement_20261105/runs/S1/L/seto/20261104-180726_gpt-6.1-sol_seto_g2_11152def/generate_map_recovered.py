from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image

# Approximate geographic shapes, written from prior geographic knowledge.
# Coordinates are (longitude, latitude). These intentionally simplify
# coastlines and small islands at the resolution of the requested grid.

# Japan's main islands and nearby larger islands within the requested frame.
LAND_POLYGONS = [
    # Kyushu
    [
        (129.65, 33.00), (130.10, 33.00), (130.45, 33.15),
        (130.72, 33.36), (130.80, 33.65), (131.05, 33.85),
        (131.32, 33.94), (131.55, 33.82), (131.72, 33.65),
        (131.76, 33.38), (131.58, 33.20), (131.35, 33.10),
        (131.15, 33.00), (130.80, 33.00),
    ],
    # Northern Kyushu
    [
        (129.75, 33.70), (129.98, 33.88), (130.28, 33.90),
        (130.45, 33.78), (130.52, 33.62), (130.40, 33.48),
        (130.22, 33.43), (130.00, 33.52), (129.83, 33.58),
    ],
    # Shikoku
    [
        (132.00, 33.20), (132.35, 33.25), (132.72, 33.30),
        (133.05, 33.35), (133.35, 33.42), (133.65, 33.47),
        (133.92, 33.55), (134.10, 33.62), (134.30, 33.66),
        (134.45, 33.78), (134.62, 33.90), (134.65, 34.08),
        (134.50, 34.18), (134.27, 34.18), (134.08, 34.12),
        (133.80, 34.10), (133.55, 34.05), (133.25, 34.05),
        (132.95, 34.08), (132.65, 34.10), (132.35, 34.08),
        (132.10, 33.97), (131.95, 33.78), (131.90, 33.55),
    ],
    # Honshu, western Chugoku coast and Seto Inland Sea shore
    [
        (130.75, 34.10), (130.90, 34.18), (131.15, 34.20),
        (131.40, 34.24), (131.65, 34.30), (131.95, 34.32),
        (132.18, 34.34), (132.40, 34.36), (132.65, 34.38),
        (132.88, 34.40), (133.12, 34.42), (133.35, 34.45),
        (133.60, 34.48), (133.85, 34.50), (134.08, 34.50),
        (134.30, 34.52), (134.55, 34.55), (134.78, 34.58),
        (135.00, 34.62), (135.22, 34.66), (135.45, 34.72),
        (135.62, 34.82), (135.78, 34.94), (135.90, 35.03),
        (136.08, 35.10), (136.30, 35.13), (136.55, 35.13),
        (136.80, 35.12), (137.00, 35.10), (137.00, 35.00),
        (136.70, 34.95), (136.42, 34.92), (136.18, 34.88),
        (135.95, 34.82), (135.72, 34.73), (135.55, 34.62),
        (135.40, 34.52), (135.22, 34.42), (135.00, 34.33),
        (134.78, 34.25), (134.55, 34.18), (134.30, 34.12),
        (134.08, 34.10), (133.85, 34.08), (133.60, 34.06),
        (133.35, 34.04), (133.10, 34.03), (132.85, 34.02),
        (132.60, 34.00), (132.35, 33.98), (132.10, 33.96),
        (131.85, 33.94), (131.60, 33.92), (131.35, 33.90),
        (131.10, 33.88), (130.90, 33.92), (130.75, 34.00),
    ],
    # Kii Peninsula and southern Kansai shore
    [
        (135.00, 33.45), (135.20, 33.50), (135.42, 33.58),
        (135.62, 33.70), (135.78, 33.88), (135.90, 34.08),
        (136.05, 34.22), (136.20, 34.38), (136.32, 34.55),
        (136.45, 34.72), (136.52, 34.90), (136.62, 35.00),
        (136.44, 35.02), (136.22, 34.95), (136.05, 34.88),
        (135.88, 34.80), (135.72, 34.68), (135.58, 34.54),
        (135.42, 34.40), (135.28, 34.23), (135.15, 34.05),
        (135.05, 33.85), (134.98, 33.65),
    ],
    # Awaji Island
    [
        (134.78, 34.20), (134.92, 34.22), (135.02, 34.35),
        (135.02, 34.55), (134.94, 34.68), (134.82, 34.62),
        (134.75, 34.48), (134.74, 34.32),
    ],
    # Ise Bay's eastern shore / Chita peninsula
    [
        (136.70, 34.62), (136.82, 34.66), (137.00, 34.70),
        (137.00, 34.95), (136.84, 34.90), (136.72, 34.80),
    ],
]

# Polygon islands across the Seto Inland Sea. Larger islands are represented
# explicitly; compact ellipses below add island groups and reefs.
ISLAND_POLYGONS = [
    # Suo-Oshima
    [(132.12, 33.84), (132.28, 33.82), (132.45, 33.88),
     (132.55, 33.98), (132.48, 34.04), (132.28, 34.03),
     (132.14, 33.96)],
    # Its neighboring islands
    [(132.00, 33.88), (132.08, 33.88), (132.10, 33.94), (132.03, 33.96)],
    # Kurahashi / Etajima group
    [(132.30, 34.10), (132.40, 34.10), (132.48, 34.20),
     (132.44, 34.31), (132.35, 34.30), (132.29, 34.20)],
    [(132.45, 34.18), (132.54, 34.18), (132.57, 34.26),
     (132.50, 34.29)],
    # Itsukushima
    [(132.29, 34.25), (132.34, 34.25), (132.36, 34.31), (132.31, 34.32)],
    # Geiyo islands: Omishima, Hakatajima, Oshima, Innoshima
    [(132.96, 34.18), (133.10, 34.18), (133.14, 34.30),
     (133.08, 34.37), (132.98, 34.34), (132.94, 34.25)],
    [(133.10, 34.14), (133.18, 34.15), (133.20, 34.23), (133.12, 34.24)],
    [(133.20, 34.05), (133.30, 34.05), (133.34, 34.16),
     (133.28, 34.23), (133.20, 34.18)],
    [(133.12, 34.32), (133.24, 34.32), (133.30, 34.40), (133.18, 34.42)],
    # Innoshima and nearby islands
    [(133.16, 34.35), (133.28, 34.35), (133.31, 34.43),
     (133.22, 34.46), (133.15, 34.41)],
    # Shodoshima
    [(134.10, 34.35), (134.25, 34.34), (134.42, 34.42),
     (134.45, 34.53), (134.32, 34.60), (134.15, 34.57),
     (134.08, 34.48)],
    # Naoshima
    [(133.95, 34.43), (134.01, 34.43), (134.03, 34.49), (133.98, 34.50)],
    # Teshima
    [(134.02, 34.43), (134.08, 34.43), (134.10, 34.48), (134.04, 34.49)],
    # Ibuki and small islands west of Shodoshima
    [(133.55, 34.10), (133.62, 34.10), (133.63, 34.16), (133.57, 34.17)],
    # Bisan islands
    [(133.65, 34.30), (133.72, 34.30), (133.74, 34.37), (133.68, 34.38)],
    [(133.78, 34.31), (133.84, 34.31), (133.86, 34.37), (133.80, 34.38)],
    [(133.88, 34.34), (133.94, 34.34), (133.96, 34.39), (133.90, 34.40)],
    # Innoshima / Geiyo smaller islands
    [(132.80, 34.16), (132.87, 34.16), (132.89, 34.22), (132.83, 34.23)],
    [(132.70, 34.12), (132.76, 34.12), (132.77, 34.17), (132.72, 34.18)],
    # Harima Nada islands
    [(134.45, 34.55), (134.52, 34.55), (134.53, 34.61), (134.47, 34.62)],
    [(134.60, 34.52), (134.66, 34.52), (134.68, 34.58), (134.62, 34.59)],
]

# Elliptical island and reef approximations: (lon, lat, radius_lon, radius_lat).
# Small features are included where a grid sample can plausibly intersect them.
ISLAND_ELLIPSES = [
    (129.65, 33.90, 0.035, 0.055),  # Iki
    (129.95, 33.15, 0.045, 0.025),  # Tsushima's southern end
    (130.15, 33.55, 0.045, 0.040),  # Hirado
    (130.55, 33.85, 0.030, 0.030),
    (131.15, 33.95, 0.025, 0.035),
    (131.80, 33.80, 0.025, 0.035),
    (132.02, 33.85, 0.040, 0.050),
    (132.18, 33.90, 0.060, 0.045),
    (132.38, 34.05, 0.045, 0.060),
    (132.48, 34.12, 0.035, 0.045),
    (132.55, 34.18, 0.035, 0.045),
    (132.66, 34.22, 0.035, 0.050),
    (132.78, 34.20, 0.035, 0.050),
    (132.91, 34.23, 0.035, 0.055),
    (133.02, 34.13, 0.040, 0.055),
    (133.15, 34.11, 0.045, 0.055),
    (133.26, 34.15, 0.035, 0.060),
    (133.35, 34.20, 0.035, 0.050),
    (133.45, 34.27, 0.035, 0.045),
    (133.57, 34.29, 0.035, 0.045),
    (133.70, 34.24, 0.030, 0.050),
    (133.84, 34.27, 0.030, 0.045),
    (133.98, 34.28, 0.035, 0.045),
    (134.12, 34.21, 0.035, 0.045),
    (134.28, 34.24, 0.035, 0.045),
    (134.42, 34.24, 0.030, 0.045),
    (134.55, 34.30, 0.035, 0.050),
    (134.66, 34.40, 0.035, 0.050),
    (134.76, 34.39, 0.030, 0.045),
    (134.88, 34.31, 0.035, 0.050),
    (135.12, 34.43, 0.035, 0.045),
    (135.30, 34.51, 0.030, 0.045),
    (135.55, 34.63, 0.030, 0.040),
    (135.72, 34.77, 0.030, 0.040),
]

def point_on_segment(x, y, x1, y1, x2, y2):
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > 1e-10:
        return False
    return (
        min(x1, x2) - 1e-10 <= x <= max(x1, x2) + 1e-10
        and min(y1, y2) - 1e-10 <= y <= max(y1, y2) + 1e-10
    )

def inside_polygon(lon, lat, polygon):
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if point_on_segment(lon, lat, x1, y1, x2, y2):
            return True
        if (y1 > lat) != (y2 > lat):
            crossing_lon = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < crossing_lon:
                inside = not inside
    return inside

def is_land(lon, lat):
    if any(inside_polygon(lon, lat, p) for p in LAND_POLYGONS):
        return True
    if any(inside_polygon(lon, lat, p) for p in ISLAND_POLYGONS):
        return True
    for cx, cy, rx, ry in ISLAND_ELLIPSES:
        if ((lon - cx) / rx) ** 2 + ((lat - cy) / ry) ** 2 <= 1.0:
            return True
    return False

def main():
    # Integer-index coordinates avoid cumulative floating-point drift.
    lats = 35.0 - np.arange(64, dtype=np.float64) * (2.0 / 63.0)
    lons = 129.0 + np.arange(253, dtype=np.float64) * (8.0 / 252.0)

    matrix = np.zeros((64, 253), dtype=np.uint8)
    for row, lat in enumerate(lats):
        for col, lon in enumerate(lons):
            matrix[row, col] = int(is_land(float(lon), float(lat)))

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)

    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Pillow's L mode gives 255 for land (white) and 0 for water (black).
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(out / "candidate.png")
    enlarged = image.resize((253 * 8, 64 * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(out / "candidate_8x.png")

    # Validate dimensions, values, encoding, and exact nearest-neighbor expansion.
    assert matrix.shape == (64, 253)
    assert int(matrix.size) == 16192
    assert set(np.unique(matrix).tolist()) <= {0, 1}

    with (out / "candidate.csv").open("r", newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 64
    assert all(len(row) == 253 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    check = Image.open(out / "candidate.png").convert("L")
    assert check.size == (253, 64)
    assert set(check.getdata()) <= {0, 255}

    check_8x = Image.open(out / "candidate_8x.png").convert("L")
    assert check_8x.size == (2024, 512)
    assert np.array_equal(
        np.asarray(check_8x),
        np.repeat(np.repeat(np.asarray(check), 8, axis=0), 8, axis=1),
    )

if __name__ == "__main__":
    main()
