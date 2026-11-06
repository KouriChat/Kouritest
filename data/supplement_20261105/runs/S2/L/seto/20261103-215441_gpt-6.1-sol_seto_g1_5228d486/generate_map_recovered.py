#!/usr/bin/env python3
"""Knowledge-based approximate land/water map for the Seto Inland Sea region.

No external geographic data is read or downloaded. Polygon coordinates below
are manually specified approximations. Each sample point is classified by
whether it lies in one of the land polygons.
"""

from pathlib import Path

import numpy as np
from PIL import Image


# Grid: 35 N to 33 N, 129 E to 137 E, with a 2/63 degree interval.
N_ROWS = 64
N_COLS = 253
STEP = 2.0 / 63.0

# Hand-written approximate land polygons in (longitude, latitude) order.
# The broad polygons trace the main islands and peninsulas; smaller polygons
# represent major islands in the Seto Inland Sea and Ise Bay.
LAND_POLYGONS = [
    # Kyushu, including its northeastern coast
    [
        (129.0, 33.0), (131.0, 33.0), (131.1, 33.3), (131.6, 33.4),
        (131.8, 33.7), (131.8, 34.0), (131.4, 34.2), (131.4, 34.5),
        (131.0, 34.6), (130.5, 34.5), (130.0, 34.4), (129.0, 34.3),
    ],
    # Western Honshu, from Yamaguchi through the Chugoku coast to the east
    [
        (130.9, 34.8), (131.0, 34.9), (131.4, 35.0), (132.0, 35.0),
        (132.7, 35.0), (133.5, 35.0), (134.2, 35.0), (135.0, 35.0),
        (135.8, 35.0), (136.5, 35.0), (137.0, 35.0), (137.0, 34.8),
        (136.6, 34.75), (136.1, 34.72), (135.7, 34.75), (135.2, 34.8),
        (134.8, 34.85), (134.4, 34.8), (134.1, 34.82), (133.8, 34.78),
        (133.5, 34.8), (133.2, 34.78), (132.9, 34.8), (132.5, 34.8),
        (132.0, 34.78), (131.6, 34.78), (131.2, 34.75),
    ],
    # Shikoku
    [
        (132.0, 34.25), (132.5, 34.25), (133.0, 34.2), (133.5, 34.2),
        (134.0, 34.25), (134.5, 34.25), (134.8, 34.2), (134.7, 33.9),
        (134.4, 33.7), (134.0, 33.6), (133.5, 33.55), (133.0, 33.55),
        (132.5, 33.65), (132.2, 33.8), (132.0, 34.0),
    ],
    # Eastern Honshu / Kii Peninsula and Ise Bay western shore
    [
        (135.0, 34.45), (135.2, 34.5), (135.4, 34.55), (135.6, 34.7),
        (135.8, 34.9), (136.0, 35.0), (137.0, 35.0), (137.0, 34.5),
        (136.8, 34.4), (136.6, 34.35), (136.4, 34.3), (136.2, 34.25),
        (136.0, 34.2), (135.8, 34.15), (135.6, 34.1), (135.4, 34.1),
        (135.2, 34.15), (135.0, 34.2),
    ],
    # Awaji Island
    [
        (134.75, 34.25), (134.95, 34.25), (135.02, 34.05),
        (134.95, 33.85), (134.82, 33.85), (134.75, 34.05),
    ],
    # Shodoshima
    [
        (134.15, 34.55), (134.35, 34.55), (134.4, 34.45),
        (134.25, 34.4), (134.12, 34.45),
    ],
    # Geiyo Islands: simplified island chain between Hiroshima and Imabari
    [
        (132.55, 34.35), (132.75, 34.35), (132.82, 34.25),
        (132.72, 34.18), (132.58, 34.22),
    ],
    [
        (132.85, 34.32), (133.05, 34.32), (133.1, 34.22),
        (132.98, 34.16), (132.86, 34.2),
    ],
    [
        (133.15, 34.30), (133.35, 34.30), (133.4, 34.2),
        (133.28, 34.13), (133.16, 34.18),
    ],
    [
        (133.45, 34.28), (133.65, 34.28), (133.7, 34.18),
        (133.58, 34.12), (133.46, 34.17),
    ],
    [
        (133.75, 34.3), (133.95, 34.3), (134.0, 34.2),
        (133.88, 34.14), (133.76, 34.18),
    ],
    [
        (134.0, 34.31), (134.18, 34.31), (134.22, 34.22),
        (134.1, 34.16), (134.0, 34.2),
    ],
    # Additional major islands around Geiyo and Hiroshima Bay
    [(132.2, 34.25), (132.42, 34.25), (132.45, 34.12), (132.28, 34.08)],
    [(132.4, 34.18), (132.55, 34.18), (132.58, 34.08), (132.45, 34.04)],
    [(132.7, 34.08), (132.85, 34.08), (132.9, 33.98), (132.76, 33.96)],
    [(133.1, 34.08), (133.25, 34.08), (133.28, 33.98), (133.14, 33.96)],
    [(133.5, 34.08), (133.65, 34.08), (133.68, 33.98), (133.54, 33.96)],
    # Inland Sea islands toward eastern Shikoku and Kii Channel
    [(134.3, 34.15), (134.45, 34.15), (134.48, 34.05), (134.34, 34.02)],
    [(134.55, 34.15), (134.68, 34.15), (134.7, 34.05), (134.58, 34.02)],
    # Ise Bay islands and eastern coastline detail
    [(136.55, 34.65), (136.72, 34.65), (136.76, 34.55), (136.62, 34.52)],
]


def point_on_segment(x, y, x1, y1, x2, y2):
    """Return true when (x, y) lies on a polygon edge."""
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > 1e-10:
        return False
    return (
        min(x1, x2) - 1e-10 <= x <= max(x1, x2) + 1e-10
        and min(y1, y2) - 1e-10 <= y <= max(y1, y2) + 1e-10
    )


def point_in_polygon(x, y, polygon):
    """Ray-casting point test; polygon boundary counts as land."""
    inside = False
    count = len(polygon)
    for i in range(count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]
        if point_on_segment(x, y, x1, y1, x2, y2):
            return True
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing_x:
                inside = not inside
    return inside


def build_matrix():
    latitudes = 35.0 - np.arange(N_ROWS, dtype=np.float64) * STEP
    longitudes = 129.0 + np.arange(N_COLS, dtype=np.float64) * STEP
    result = np.zeros((N_ROWS, N_COLS), dtype=np.uint8)

    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            if any(point_in_polygon(lon, lat, poly) for poly in LAND_POLYGONS):
                result[row, col] = 1

    return result


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = build_matrix()

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", encoding="ascii", newline="") as file:
        for row in matrix:
            file.write(",".join(str(int(value)) for value in row) + "\n")

    # Pillow mode "1" produces a strictly black-and-white source image.
    image = Image.fromarray(matrix * 255, mode="L").convert("1")
    image.save(output_dir / "candidate.png")

    enlarged = image.resize(
        (N_COLS * 8, N_ROWS * 8),
        resample=Image.Resampling.NEAREST,
    )
    enlarged.save(output_dir / "candidate_8x.png")

    # Verify dimensions, values, and the exact nearest-neighbor relationship.
    assert matrix.shape == (64, 253)
    assert matrix.size == 16192
    assert np.isin(matrix, (0, 1)).all()
    assert image.size == (253, 64)
    assert set(image.getdata()).issubset({0, 255})
    assert enlarged.size == (2024, 512)

    pixels = np.asarray(image.convert("L"))
    expected = np.repeat(np.repeat(pixels, 8, axis=0), 8, axis=1)
    assert np.array_equal(np.asarray(enlarged.convert("L")), expected)

    rows = csv_path.read_text(encoding="ascii").splitlines()
    assert len(rows) == 64
    assert all(
        len(row.split(",")) == 253
        and all(value in {"0", "1"} for value in row.split(","))
        for row in rows
    )

    print(f"Wrote {csv_path}, {output_dir / 'candidate.png'}, "
          f"and {output_dir / 'candidate_8x.png'}")
    print("Verified 64x253 values, binary CSV, black/white PNGs, "
          "and exact 8x nearest-neighbor enlargement.")


if __name__ == "__main__":
    main()
