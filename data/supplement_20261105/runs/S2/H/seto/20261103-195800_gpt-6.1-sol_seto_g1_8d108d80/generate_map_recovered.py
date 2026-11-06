#!/usr/bin/env python3
"""Knowledge-based approximate land/water raster for the Seto Inland Sea region."""

from pathlib import Path

import numpy as np
from PIL import Image


ROWS = 64
COLS = 253
LAT_TOP = 35.0
LAT_BOTTOM = 33.0
LON_LEFT = 129.0
LON_RIGHT = 137.0

# Hand-authored approximate coastline outlines in (longitude, latitude).
# These polygons are deliberately coarse and are not sourced from map data.
LAND_POLYGONS = [
    # Kyushu and adjacent western land clipped by the frame.
    [
        (129.0, 33.02), (129.0, 34.10), (129.20, 34.15),
        (129.40, 34.00), (129.58, 33.88), (129.75, 33.78),
        (129.84, 33.63), (130.00, 33.55), (130.10, 33.38),
        (130.28, 33.25), (130.44, 33.10), (130.63, 33.02),
        (130.85, 32.98), (131.10, 32.80), (131.27, 32.55),
        (131.42, 32.25), (131.58, 32.00), (131.72, 31.70),
        (131.86, 31.30), (132.00, 31.00), (132.05, 30.80),
        (129.0, 30.80),
    ],
    # Honshu: southern coast faces the Seto Inland Sea; north and east
    # edges are simplified where they intersect the requested frame.
    [
        (130.84, 33.95), (131.02, 34.02), (131.20, 34.04),
        (131.42, 34.02), (131.64, 34.06), (131.86, 34.10),
        (132.08, 34.14), (132.30, 34.20), (132.52, 34.23),
        (132.74, 34.26), (132.96, 34.28), (133.18, 34.31),
        (133.40, 34.34), (133.62, 34.38), (133.84, 34.43),
        (134.06, 34.48), (134.28, 34.53), (134.50, 34.58),
        (134.70, 34.63), (134.90, 34.67), (135.10, 34.69),
        (135.28, 34.66), (135.38, 34.52), (135.33, 34.36),
        (135.22, 34.20), (135.32, 34.02), (135.48, 33.82),
        (135.62, 33.60), (135.78, 33.42), (135.98, 33.40),
        (136.20, 33.50), (136.42, 33.69), (136.62, 33.86),
        (136.82, 34.00), (137.00, 34.13), (137.00, 35.20),
        (130.65, 35.20), (130.58, 34.82), (130.68, 34.55),
        (130.80, 34.28),
    ],
    # Shikoku.
    [
        (132.00, 34.18), (132.20, 34.16), (132.42, 34.15),
        (132.65, 34.16), (132.90, 34.18), (133.15, 34.20),
        (133.42, 34.22), (133.68, 34.23), (133.92, 34.24),
        (134.16, 34.22), (134.38, 34.18), (134.58, 34.10),
        (134.70, 33.94), (134.62, 33.76), (134.52, 33.58),
        (134.39, 33.42), (134.22, 33.30), (134.02, 33.25),
        (133.78, 33.25), (133.54, 33.27), (133.30, 33.27),
        (133.08, 33.24), (132.88, 33.25), (132.67, 33.32),
        (132.47, 33.43), (132.28, 33.56), (132.12, 33.70),
        (132.00, 33.87),
    ],
    # Awaji Island.
    [
        (134.72, 34.68), (134.88, 34.67), (135.00, 34.58),
        (135.02, 34.43), (134.94, 34.30), (134.82, 34.24),
        (134.70, 34.31), (134.64, 34.45), (134.66, 34.58),
    ],
    # Kii Peninsula and the nearby mainland east of Osaka Bay.
    [
        (135.10, 34.72), (135.42, 34.78), (135.65, 34.72),
        (135.80, 34.58), (135.82, 34.38), (135.76, 34.18),
        (135.71, 33.98), (135.65, 33.78), (135.56, 33.58),
        (135.43, 33.40), (135.24, 33.25), (135.02, 33.15),
        (134.92, 33.34), (135.02, 33.56), (135.10, 33.78),
        (135.14, 34.02), (135.18, 34.23), (135.12, 34.43),
    ],
    # Chita/Atsumi side of Ise Bay, simplified.
    [
        (136.55, 34.82), (136.72, 34.88), (136.90, 34.82),
        (137.00, 34.72), (137.00, 35.10), (136.55, 35.10),
    ],
]

# Approximate water cutouts over the simplified mainland polygons:
# Seto Inland Sea, Osaka Bay, and Ise Bay. Islands are added afterward.
WATER_POLYGONS = [
    [
        (130.94, 33.91), (131.14, 33.98), (131.38, 33.96),
        (131.63, 33.98), (131.88, 34.02), (132.12, 34.04),
        (132.38, 34.08), (132.62, 34.10), (132.88, 34.12),
        (133.12, 34.15), (133.38, 34.17), (133.64, 34.20),
        (133.90, 34.24), (134.16, 34.28), (134.40, 34.34),
        (134.62, 34.39), (134.82, 34.42), (134.97, 34.44),
        (135.02, 34.39), (134.91, 34.31), (134.70, 34.29),
        (134.46, 34.28), (134.20, 34.25), (133.96, 34.20),
        (133.70, 34.16), (133.44, 34.12), (133.18, 34.08),
        (132.92, 34.04), (132.66, 34.00), (132.40, 33.96),
        (132.14, 33.91), (131.90, 33.86), (131.66, 33.82),
        (131.42, 33.81), (131.18, 33.82),
    ],
    [
        (134.72, 34.66), (134.90, 34.63), (135.10, 34.63),
        (135.28, 34.58), (135.35, 34.48), (135.31, 34.35),
        (135.22, 34.27), (135.10, 34.29), (135.02, 34.39),
        (134.96, 34.48), (134.83, 34.52), (134.72, 34.56),
    ],
    [
        (136.55, 34.83), (136.72, 34.78), (136.90, 34.69),
        (137.05, 34.58), (137.05, 34.36), (136.92, 34.34),
        (136.80, 34.43), (136.68, 34.54), (136.58, 34.66),
    ],
]

# Ellipses are hand-estimated islands, represented by centers and radii
# in degrees. Radii are intentionally approximate.
ISLANDS = [
    # Western Seto islands.
    (131.15, 33.92, 0.055, 0.045),  # Suo-Oshima area
    (131.42, 33.91, 0.035, 0.028),
    (131.72, 33.91, 0.040, 0.026),
    (132.00, 33.94, 0.045, 0.030),
    # Hiroshima Bay and Geiyo island chain.
    (132.35, 34.08, 0.060, 0.035),
    (132.50, 34.13, 0.055, 0.040),
    (132.62, 34.18, 0.045, 0.040),
    (132.72, 34.13, 0.060, 0.050),
    (132.82, 34.21, 0.055, 0.050),
    (132.91, 34.16, 0.060, 0.045),
    (133.00, 34.20, 0.070, 0.050),
    (133.10, 34.17, 0.060, 0.045),
    (133.18, 34.22, 0.045, 0.040),
    (133.28, 34.18, 0.050, 0.040),
    (133.36, 34.20, 0.045, 0.035),
    (133.46, 34.22, 0.040, 0.032),
    # Shikoku offshore islands and Kagawa islands.
    (133.72, 34.31, 0.045, 0.035),
    (133.84, 34.36, 0.040, 0.030),
    (134.00, 34.39, 0.050, 0.035),
    (134.18, 34.40, 0.045, 0.030),
    (134.35, 34.42, 0.040, 0.030),
    (134.48, 34.44, 0.035, 0.028),
    (134.56, 34.49, 0.025, 0.022),
    (134.66, 34.55, 0.028, 0.022),
    # Small islands in the Kii Channel and Ise Bay.
    (135.05, 34.20, 0.035, 0.028),
    (136.70, 34.46, 0.025, 0.020),
]


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_intersection = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersection:
                inside = not inside
        j = i
    return inside


def ellipse_contains(x: float, y: float, cx: float, cy: float,
                     rx: float, ry: float) -> bool:
    if rx <= 0 or ry <= 0:
        return False
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


def is_land(lon: float, lat: float) -> bool:
    land = any(point_in_polygon(lon, lat, p) for p in LAND_POLYGONS)
    if land and any(point_in_polygon(lon, lat, p) for p in WATER_POLYGONS):
        land = False
    if any(ellipse_contains(lon, lat, *island) for island in ISLANDS):
        land = True
    return land


def main() -> None:
    lats = np.linspace(LAT_TOP, LAT_BOTTOM, ROWS)
    lons = np.linspace(LON_LEFT, LON_RIGHT, COLS)

    matrix = np.zeros((ROWS, COLS), dtype=np.uint8)
    for row, lat in enumerate(lats):
        for col, lon in enumerate(lons):
            matrix[row, col] = 1 if is_land(float(lon), float(lat)) else 0

    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", encoding="ascii", newline="") as f:
        for row in matrix:
            f.write(",".join(str(int(value)) for value in row) + "\n")

    # Pillow mode "1": 0 is black water, 255 is white land.
    image = Image.fromarray(matrix * 255, mode="L").convert("1")
    image.save(output_dir / "candidate.png")

    enlarged = image.resize((COLS * 8, ROWS * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")

    # Validate dimensions, binary values, CSV shape, and image palette.
    assert matrix.shape == (64, 253)
    assert int(matrix.size) == 16192
    assert set(np.unique(matrix).tolist()) <= {0, 1}

    with csv_path.open("r", encoding="ascii") as f:
        csv_rows = [line.rstrip("\n").split(",") for line in f]
    assert len(csv_rows) == 64
    assert all(len(row) == 253 for row in csv_rows)
    assert all(value in {"0", "1"} for row in csv_rows for value in row)

    original = Image.open(output_dir / "candidate.png")
    enlarged_check = Image.open(output_dir / "candidate_8x.png")
    assert original.size == (253, 64)
    assert enlarged_check.size == (2024, 512)
    assert set(original.getdata()) <= {0, 255}
    assert set(enlarged_check.getdata()) <= {0, 255}

    print(f"Wrote {csv_path}")
    print(f"Wrote {output_dir / 'candidate.png'} (253x64)")
    print(f"Wrote {output_dir / 'candidate_8x.png'} (2024x512)")
    print("Validated: 64x253 binary matrix, 16192 values, black/white nearest-neighbor images.")


if __name__ == "__main__":
    main()
