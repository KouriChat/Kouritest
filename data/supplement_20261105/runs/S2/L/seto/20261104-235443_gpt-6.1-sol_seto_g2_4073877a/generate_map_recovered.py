from pathlib import Path

import numpy as np
from PIL import Image


# Approximate land polygons, written from geographic knowledge.
# Coordinates are (longitude, latitude), WGS84 decimal degrees.
LAND_POLYGONS = [
    # Honshu, west to east: Seto Inland Sea coast first, then Sea of Japan coast.
    [
        (130.86, 33.93), (131.05, 34.03), (131.28, 33.96), (131.55, 34.00),
        (131.82, 34.06), (132.10, 34.12), (132.36, 34.16), (132.62, 34.20),
        (132.86, 34.22), (133.08, 34.22), (133.32, 34.22), (133.55, 34.20),
        (133.78, 34.22), (134.02, 34.24), (134.23, 34.28), (134.42, 34.30),
        (134.61, 34.33), (134.80, 34.39), (134.98, 34.43), (135.13, 34.50),
        (135.28, 34.55), (135.46, 34.62), (135.63, 34.68), (135.82, 34.72),
        (136.02, 34.75), (136.20, 34.79), (136.38, 34.83), (136.56, 34.88),
        (136.76, 34.94), (136.96, 35.00), (137.20, 35.08), (137.50, 35.17),
        (137.82, 35.27), (138.10, 35.38), (138.38, 35.50), (138.65, 35.63),
        (138.90, 35.75), (139.18, 35.88), (139.43, 36.02), (139.66, 36.17),
        (139.89, 36.33), (140.12, 36.48), (140.34, 36.65), (140.55, 36.83),
        (140.76, 37.02), (140.97, 37.22), (141.16, 37.43), (141.35, 37.66),
        (141.50, 37.88), (141.63, 38.12), (141.73, 38.36), (141.78, 38.62),
        (141.77, 38.87), (141.71, 39.12), (141.63, 39.37), (141.53, 39.62),
        (141.42, 39.87), (141.29, 40.12), (141.15, 40.38), (141.00, 40.63),
        (140.83, 40.88), (140.64, 41.10), (140.43, 41.31), (140.20, 41.50),
        (139.96, 41.65), (139.71, 41.78), (139.44, 41.88), (139.15, 41.96),
        (138.84, 42.01), (138.50, 42.05), (138.15, 42.08), (137.80, 42.10),
        (137.45, 42.10), (137.10, 42.08), (136.75, 42.05), (136.40, 42.00),
        (136.05, 41.93), (135.70, 41.83), (135.35, 41.70), (135.02, 41.54),
        (134.70, 41.35), (134.40, 41.13), (134.13, 40.88), (133.90, 40.62),
        (133.70, 40.34), (133.53, 40.05), (133.39, 39.75), (133.27, 39.44),
        (133.17, 39.13), (133.08, 38.82), (132.98, 38.51), (132.85, 38.20),
        (132.67, 37.90), (132.43, 37.62), (132.14, 37.37), (131.81, 37.16),
        (131.44, 36.98), (131.08, 36.81), (130.76, 36.62), (130.49, 36.40),
        (130.26, 36.16), (130.08, 35.91), (129.93, 35.65), (129.82, 35.39),
        (129.75, 35.14), (129.75, 34.91), (129.83, 34.69), (129.99, 34.49),
        (130.21, 34.31), (130.44, 34.16), (130.65, 34.04),
    ],
    # Shikoku.
    [
        (132.02, 33.78), (132.18, 33.89), (132.40, 33.94), (132.62, 33.95),
        (132.83, 33.93), (133.05, 33.91), (133.28, 33.90), (133.51, 33.90),
        (133.74, 33.91), (133.98, 33.92), (134.20, 33.94), (134.37, 33.99),
        (134.49, 34.07), (134.50, 34.16), (134.42, 34.23), (134.25, 34.27),
        (134.05, 34.27), (133.85, 34.25), (133.64, 34.23), (133.42, 34.22),
        (133.20, 34.23), (132.98, 34.24), (132.76, 34.23), (132.54, 34.20),
        (132.33, 34.15), (132.15, 34.08), (132.03, 33.98), (131.97, 33.88),
    ],
    # Kyushu, including northern and eastern coasts in this frame.
    [
        (130.86, 33.95), (131.04, 33.89), (131.25, 33.87), (131.45, 33.83),
        (131.64, 33.77), (131.81, 33.67), (131.93, 33.54), (132.02, 33.39),
        (132.08, 33.22), (132.11, 33.05), (132.10, 32.88), (132.06, 32.70),
        (131.99, 32.51), (131.88, 32.33), (131.74, 32.16), (131.57, 32.02),
        (131.37, 31.91), (131.15, 31.84), (130.91, 31.82), (130.68, 31.86),
        (130.45, 31.95), (130.24, 32.08), (130.05, 32.24), (129.88, 32.42),
        (129.73, 32.61), (129.61, 32.82), (129.52, 33.03), (129.48, 33.23),
        (129.50, 33.42), (129.56, 33.60), (129.67, 33.75), (129.82, 33.87),
        (130.00, 33.96), (130.20, 34.01), (130.41, 34.02), (130.60, 34.00),
        (130.75, 33.97),
    ],
    # Awaji Island.
    [
        (134.78, 34.15), (134.88, 34.20), (134.94, 34.34), (134.96, 34.51),
        (134.91, 34.68), (134.83, 34.79), (134.75, 34.77), (134.69, 34.65),
        (134.68, 34.48), (134.70, 34.30),
    ],
    # Oki Islands.
    [
        (133.00, 36.08), (133.17, 36.14), (133.25, 36.27), (133.20, 36.38),
        (133.06, 36.40), (132.96, 36.31), (132.95, 36.19),
    ],
    # Sado Island.
    [
        (138.20, 37.75), (138.43, 37.82), (138.56, 38.00), (138.54, 38.19),
        (138.43, 38.30), (138.25, 38.28), (138.14, 38.13), (138.12, 37.95),
    ],
    # Tsushima.
    [
        (129.18, 34.05), (129.36, 34.10), (129.48, 34.27), (129.47, 34.48),
        (129.40, 34.66), (129.25, 34.70), (129.13, 34.58), (129.10, 34.37),
    ],
    # Iki Island.
    [
        (129.60, 33.72), (129.78, 33.73), (129.87, 33.83), (129.82, 33.95),
        (129.67, 33.96), (129.57, 33.86),
    ],
]

# Goto, Geiyo, and other island groups. Ellipses are deliberate coarse outlines.
ISLAND_ELLIPSES = [
    # Goto Islands
    (128.75, 32.75, 0.28, 0.85, 25),
    # Geiyo Islands, between Hiroshima and Shikoku
    (132.80, 34.20, 0.10, 0.20, 10),
    (133.02, 34.18, 0.09, 0.17, -10),
    (133.23, 34.17, 0.09, 0.16, 5),
    (133.43, 34.17, 0.08, 0.15, -15),
    (133.63, 34.17, 0.08, 0.14, 15),
    (133.83, 34.18, 0.07, 0.12, -5),
    # Small islands in the western Seto Inland Sea
    (132.20, 33.98, 0.07, 0.13, 0),
    (132.43, 34.08, 0.06, 0.12, 15),
    (132.65, 34.10, 0.06, 0.12, -10),
    # Ise Bay islands and the Chita peninsula tip
    (136.82, 34.70, 0.08, 0.22, 5),
    # Kii Channel islands
    (134.95, 34.08, 0.05, 0.12, 0),
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


def point_in_ellipse(lon, lat, cx, cy, rx, ry, angle_degrees=0):
    """Approximate a small island as a rotated ellipse in lon/lat coordinates."""
    import math

    angle = math.radians(angle_degrees)
    dx = lon - cx
    dy = lat - cy
    x = dx * math.cos(angle) + dy * math.sin(angle)
    y = -dx * math.sin(angle) + dy * math.cos(angle)
    return (x / rx) ** 2 + (y / ry) ** 2 <= 1.0


def is_land(lon, lat):
    # Ise Bay: carve the bay water out of the broad Honshu approximation.
    in_ise_bay = (
        136.55 <= lon <= 137.20
        and 34.55 <= lat <= 35.05
        and lon <= 137.02 - 0.55 * (lat - 34.55)
    )
    if in_ise_bay:
        return False

    if any(point_in_polygon(lon, lat, poly) for poly in LAND_POLYGONS):
        return True

    return any(
        point_in_ellipse(lon, lat, *ellipse)
        for ellipse in ISLAND_ELLIPSES
    )


def main():
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Integer-index construction avoids cumulative floating-point drift.
    latitudes = 35.0 - np.arange(64, dtype=np.float64) * (2.0 / 63.0)
    longitudes = 129.0 + np.arange(253, dtype=np.float64) * (2.0 / 63.0)

    matrix = np.zeros((64, 253), dtype=np.uint8)
    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            matrix[row, col] = int(is_land(float(lon), float(lat)))

    csv_path = out_dir / "candidate.csv"
    with csv_path.open("w", encoding="ascii", newline="") as f:
        for row in matrix:
            f.write(",".join(str(int(value)) for value in row) + "\n")

    # Mode 1 image values map directly to black (water) and white (land).
    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(out_dir / "candidate.png")
    image.resize((253 * 8, 64 * 8), resample=Image.Resampling.NEAREST).save(
        out_dir / "candidate_8x.png"
    )

    # Verify dimensions, CSV contents, palette, and exact nearest-neighbor scale.
    assert matrix.shape == (64, 253)
    assert int(matrix.size) == 16192
    assert set(np.unique(matrix)).issubset({0, 1})

    lines = csv_path.read_text(encoding="ascii").splitlines()
    assert len(lines) == 64
    assert all(len(line.split(",")) == 253 for line in lines)
    assert all(set(line.split(",")) <= {"0", "1"} for line in lines)

    original = Image.open(out_dir / "candidate.png").convert("RGB")
    enlarged = Image.open(out_dir / "candidate_8x.png").convert("RGB")
    assert original.size == (253, 64)
    assert enlarged.size == (2024, 512)
    assert set(original.getdata()) <= {(0, 0, 0), (255, 255, 255)}
    expected = original.resize((2024, 512), resample=Image.Resampling.NEAREST)
    assert list(enlarged.getdata()) == list(expected.getdata())

    print(f"Wrote {csv_path}, {out_dir / 'candidate.png'}, "
          f"{out_dir / 'candidate_8x.png'}")
    print("Verified 64 x 253 = 16192 binary samples; PNGs are 253 x 64 "
          "and 2024 x 512.")


if __name__ == "__main__":
    main()
