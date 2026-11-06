#!/usr/bin/env python3
"""Knowledge-based approximate land/water map for the Seto Inland Sea region."""

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image

# Grid: north to south, west to east.
ROWS = 64
COLS = 253
LAT_TOP = 35.0
LON_LEFT = 129.0
STEP = 2.0 / 63.0

# Hand-drawn approximate land polygons, in (longitude, latitude) order.
# These outline the major Japanese islands and the irregular coast around
# the Seto Inland Sea and Ise Bay. Island groups are added separately below.
LAND_POLYGONS = [
    # Kyushu
    [(129.35,34.02),(129.55,33.82),(129.78,33.58),(130.05,33.45),
     (130.30,33.37),(130.48,33.28),(130.62,33.19),(130.84,33.15),
     (131.02,33.20),(131.15,33.35),(131.35,33.50),(131.62,33.62),
     (131.82,33.76),(131.92,33.94),(131.86,34.13),(131.69,34.30),
     (131.47,34.42),(131.25,34.48),(131.03,34.45),(130.84,34.35),
     (130.61,34.27),(130.37,34.22),(130.10,34.15),(129.80,34.08),
     (129.55,34.02)],
    # Shikoku
    [(132.02,34.28),(132.22,34.20),(132.45,34.16),(132.70,34.18),
     (132.96,34.20),(133.20,34.18),(133.47,34.16),(133.70,34.17),
     (133.92,34.18),(134.10,34.25),(134.20,34.34),(134.08,34.43),
     (133.85,34.46),(133.60,34.44),(133.36,34.43),(133.10,34.45),
     (132.85,34.45),(132.58,34.43),(132.35,34.40),(132.14,34.36)],
    # Honshu, Chugoku and Kinki coast facing the inland sea
    [(130.85,34.75),(131.03,34.82),(131.20,34.84),(131.38,34.82),
     (131.55,34.79),(131.72,34.77),(131.90,34.78),(132.08,34.78),
     (132.24,34.79),(132.38,34.78),(132.54,34.78),(132.68,34.77),
     (132.82,34.76),(132.98,34.76),(133.13,34.77),(133.28,34.77),
     (133.43,34.77),(133.56,34.78),(133.72,34.79),(133.86,34.80),
     (134.00,34.82),(134.15,34.84),(134.30,34.84),(134.45,34.82),
     (134.58,34.80),(134.72,34.79),(134.85,34.80),(134.98,34.82),
     (135.10,34.83),(135.24,34.84),(135.38,34.85),(135.52,34.87),
     (135.66,34.88),(135.79,34.91),(135.92,34.95),(136.02,35.00),
     (137.01,35.00),(137.01,33.00),(135.70,33.00),(135.62,33.18),
     (135.57,33.38),(135.53,33.58),(135.52,33.80),(135.53,34.00),
     (135.47,34.16),(135.36,34.26),(135.23,34.34),(135.08,34.39),
     (134.92,34.40),(134.77,34.38),(134.62,34.36),(134.47,34.37),
     (134.32,34.38),(134.19,34.38),(134.05,34.38),(133.91,34.37),
     (133.77,34.37),(133.62,34.36),(133.47,34.36),(133.32,34.36),
     (133.17,34.36),(133.02,34.36),(132.87,34.37),(132.72,34.38),
     (132.57,34.38),(132.42,34.39),(132.27,34.40),(132.12,34.42),
     (131.98,34.44),(131.83,34.47),(131.69,34.51),(131.55,34.55),
     (131.42,34.60),(131.30,34.66),(131.18,34.70),(131.04,34.72)],
    # Eastern Honshu / Ise Bay surroundings
    [(135.30,34.34),(135.42,34.30),(135.55,34.29),(135.68,34.30),
     (135.80,34.33),(135.91,34.38),(136.00,34.45),(136.06,34.55),
     (136.10,34.68),(136.12,34.83),(136.15,35.00),(137.01,35.00),
     (137.01,33.00),(136.15,33.00),(136.12,33.20),(136.10,33.40),
     (136.07,33.60),(136.04,33.82),(136.00,34.02),(135.94,34.18),
     (135.85,34.28),(135.73,34.34),(135.60,34.37),(135.46,34.37)],
    # Western Honshu shore and western bays
    [(129.00,34.58),(129.32,34.58),(129.53,34.60),(129.73,34.63),
     (129.94,34.66),(130.14,34.69),(130.34,34.71),(130.53,34.73),
     (130.72,34.75),(130.90,34.77),(131.00,34.82),(131.04,35.00),
     (129.00,35.00)],
]

# Small islands and island-group centers. Each entry is
# (longitude, latitude, approximate radius in km).
ISLANDS = [
    # Geiyo islands between Hiroshima/Ehime
    (132.45,34.23,5.0),(132.54,34.18,4.0),(132.65,34.20,3.5),
    (132.72,34.24,4.0),(132.82,34.20,3.5),(132.94,34.18,3.5),
    (133.05,34.20,3.0),(133.16,34.23,3.5),(133.26,34.25,3.0),
    (133.35,34.22,3.0),(133.48,34.20,3.0),(133.60,34.19,3.0),
    (133.72,34.21,3.5),(133.84,34.24,3.0),(133.94,34.27,3.0),
    # Hiroshima Bay and western inland sea
    (132.30,34.18,5.0),(132.18,34.12,3.5),(132.06,34.09,3.0),
    (131.94,34.05,3.0),(131.80,34.03,2.5),(131.68,34.02,2.5),
    (131.52,34.04,3.0),(131.36,34.08,2.5),(131.18,34.10,2.5),
    # Suo-Nada and north Kyushu approaches
    (131.05,33.98,4.0),(131.20,33.92,3.0),(131.38,33.90,2.5),
    (131.55,33.88,2.5),(131.74,33.89,3.0),(131.93,33.91,2.5),
    (132.10,33.92,2.0),(132.24,33.95,2.0),
    # Islands south of Shikoku / Kii Channel
    (133.45,33.98,2.5),(133.70,34.02,2.5),(133.92,34.05,2.0),
    (134.12,34.08,2.0),(134.30,34.10,2.5),(134.50,34.12,2.0),
    (134.70,34.15,2.0),
    # Awaji and nearby islands
    (134.86,34.45,12.0),(134.93,34.60,8.0),(134.95,34.34,4.0),
    # Shodoshima and nearby islands
    (134.28,34.52,9.0),(134.12,34.48,3.0),(134.43,34.48,3.0),
    (133.95,34.50,3.0),(133.80,34.48,2.5),
    # Ise Bay islands and nearby coast islands
    (136.72,34.55,4.0),(136.83,34.62,3.0),(136.55,34.66,2.0),
    # Inland Sea scattered islands
    (132.90,34.34,2.5),(133.12,34.31,2.0),(133.32,34.33,2.0),
    (133.56,34.30,2.0),(133.78,34.31,2.0),(134.02,34.32,2.0),
]

def point_in_polygon(lon, lat, polygon):
    """Ray-casting point-in-polygon test."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > lat) != (yj > lat):
            crossing_lon = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing_lon:
                inside = not inside
        j = i
    return inside

def near_island(lon, lat, island):
    ilon, ilat, radius_km = island
    dy = (lat - ilat) * 111.0
    dx = (lon - ilon) * 111.0 * math.cos(math.radians(ilat))
    return dx * dx + dy * dy <= radius_km * radius_km

def is_land(lon, lat):
    if any(point_in_polygon(lon, lat, poly) for poly in LAND_POLYGONS):
        return True
    return any(near_island(lon, lat, island) for island in ISLANDS)

def main():
    matrix = np.zeros((ROWS, COLS), dtype=np.uint8)
    for r in range(ROWS):
        lat = LAT_TOP - r * STEP
        for c in range(COLS):
            lon = LON_LEFT + c * STEP
            matrix[r, c] = 1 if is_land(lon, lat) else 0

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)

    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Pillow mode "1": 1 is white (land), 0 is black (water).
    image = Image.fromarray(matrix * 255, mode="L").convert("1")
    image.save(out / "candidate.png")
    image.resize((COLS * 8, ROWS * 8), Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )

    # Validate exported dimensions, values, and binary palette.
    assert matrix.shape == (64, 253)
    assert int(matrix.sum() + (matrix == 0).sum()) == 64 * 253
    with (out / "candidate.csv").open(newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 64 and all(len(row) == 253 for row in rows)
    assert all(value in ("0", "1") for row in rows for value in row)
    check = Image.open(out / "candidate.png")
    enlarged = Image.open(out / "candidate_8x.png")
    assert check.size == (253, 64)
    assert enlarged.size == (2024, 512)
    assert set(check.convert("L").getdata()) <= {0, 255}
    assert enlarged.tobytes() == check.resize(
        (2024, 512), Image.Resampling.NEAREST
    ).tobytes()

if __name__ == "__main__":
    main()
