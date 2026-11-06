#!/usr/bin/env python3
"""Reconstruct an approximate Seto Inland Sea region using hand-written polygons."""

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image


# Coordinates are (longitude, latitude). These are approximate, hand-written
# coastlines and island outlines, not data loaded from a map.
LAND_POLYGONS = [
    # Southern Honshu, from the Kanmon Strait eastward around the Seto coast,
    # then northward to the top of the sampling frame.
    [
        (130.92, 34.00), (131.02, 34.01), (131.12, 34.04),
        (131.23, 34.10), (131.36, 34.12), (131.50, 34.15),
        (131.66, 34.18), (131.82, 34.20), (131.98, 34.21),
        (132.13, 34.23), (132.27, 34.25), (132.42, 34.27),
        (132.57, 34.29), (132.72, 34.30), (132.87, 34.31),
        (133.02, 34.32), (133.17, 34.33), (133.32, 34.35),
        (133.47, 34.37), (133.62, 34.39), (133.77, 34.41),
        (133.92, 34.43), (134.07, 34.45), (134.22, 34.47),
        (134.37, 34.49), (134.52, 34.51), (134.67, 34.53),
        (134.82, 34.55), (134.97, 34.57), (135.12, 34.59),
        (135.27, 34.61), (135.42, 34.63), (135.57, 34.65),
        (135.72, 34.67), (135.87, 34.69), (136.02, 34.71),
        (136.17, 34.72), (136.32, 34.73), (136.47, 34.74),
        (136.62, 34.75), (136.76, 34.77), (136.88, 34.80),
        (137.00, 35.00), (130.80, 35.00),
    ],

    # Shikoku. The northern outline borders the Inland Sea; the southern
    # outline is a simplified Pacific coast.
    [
        (132.00, 33.88), (132.10, 33.94), (132.24, 34.00),
        (132.39, 34.02), (132.55, 34.03), (132.72, 34.04),
        (132.88, 34.05), (133.04, 34.06), (133.20, 34.08),
        (133.36, 34.10), (133.52, 34.12), (133.68, 34.14),
        (133.84, 34.16), (134.00, 34.18), (134.16, 34.20),
        (134.32, 34.22), (134.48, 34.24), (134.62, 34.25),
        (134.74, 34.24), (134.82, 34.20), (134.78, 34.10),
        (134.70, 34.00), (134.66, 33.87), (134.60, 33.74),
        (134.50, 33.62), (134.35, 33.53), (134.18, 33.46),
        (134.00, 33.40), (133.80, 33.35), (133.58, 33.31),
        (133.36, 33.29), (133.15, 33.28), (132.95, 33.29),
        (132.76, 33.31), (132.58, 33.35), (132.41, 33.40),
        (132.25, 33.47), (132.12, 33.57), (132.04, 33.70),
    ],

    # Northern Kyushu and the part of Kyushu inside the frame.
    [
        (129.58, 33.23), (129.73, 33.12), (129.95, 33.05),
        (130.18, 33.02), (130.40, 33.03), (130.61, 33.08),
        (130.78, 33.18), (130.91, 33.32), (131.00, 33.48),
        (131.07, 33.65), (131.12, 33.81), (131.12, 33.94),
        (131.04, 34.00), (130.91, 34.00), (130.76, 33.96),
        (130.58, 33.91), (130.39, 33.88), (130.19, 33.86),
        (130.00, 33.84), (129.82, 33.79), (129.68, 33.69),
        (129.60, 33.54),
    ],

    # Awaji Island.
    [
        (134.78, 34.27), (134.82, 34.32), (134.86, 34.39),
        (134.91, 34.47), (134.96, 34.55), (134.99, 34.61),
        (134.97, 34.65), (134.92, 34.64), (134.88, 34.59),
        (134.84, 34.52), (134.80, 34.44), (134.77, 34.36),
    ],

    # Shodoshima.
    [
        (134.08, 34.42), (134.14, 34.43), (134.20, 34.44),
        (134.26, 34.45), (134.31, 34.48), (134.32, 34.52),
        (134.29, 34.55), (134.23, 34.56), (134.17, 34.54),
        (134.11, 34.51), (134.08, 34.47),
    ],

    # Chita Peninsula, east side of Ise Bay.
    [
        (136.78, 34.88), (136.91, 34.88), (136.99, 34.83),
        (136.98, 34.75), (136.94, 34.68), (136.88, 34.61),
        (136.84, 34.56), (136.80, 34.61), (136.79, 34.69),
        (136.81, 34.77),
    ],

    # Small peninsulas and coastal projections around the western end of
    # the Inland Sea and northern Kyushu.
    [
        (130.70, 33.92), (130.79, 33.97), (130.91, 33.99),
        (130.99, 33.96), (130.98, 33.91), (130.89, 33.89),
        (130.79, 33.89),
    ],
]


# Approximate island centers and half-widths in degrees.
# Ellipses are generated as polygon outlines to keep the point test uniform.
ISLAND_ELLIPSES = [
    # Geiyo island chain, between Honshu and Shikoku.
    (132.75, 34.20, 0.045, 0.030),
    (132.88, 34.22, 0.035, 0.025),
    (133.02, 34.24, 0.050, 0.030),
    (133.16, 34.25, 0.040, 0.026),
    (133.30, 34.27, 0.035, 0.022),
    (133.43, 34.28, 0.055, 0.030),
    (133.57, 34.30, 0.040, 0.025),
    (133.70, 34.31, 0.035, 0.024),
    (133.83, 34.32, 0.035, 0.022),
    (133.95, 34.33, 0.030, 0.020),
    (132.95, 34.15, 0.035, 0.024),
    (133.20, 34.16, 0.040, 0.025),
    (133.48, 34.18, 0.035, 0.024),
    (133.72, 34.19, 0.030, 0.022),
    (133.94, 34.20, 0.035, 0.022),

    # Islands in Hiroshima Bay and westward.
    (132.30, 34.15, 0.045, 0.028),
    (132.48, 34.18, 0.035, 0.022),
    (132.67, 34.18, 0.030, 0.020),
    (132.22, 34.08, 0.030, 0.022),
    (132.48, 34.08, 0.035, 0.022),

    # Islands in the eastern Inland Sea.
    (133.72, 34.45, 0.035, 0.022),
    (133.88, 34.47, 0.030, 0.020),
    (134.00, 34.47, 0.025, 0.018),
    (134.48, 34.40, 0.035, 0.022),
    (134.58, 34.36, 0.030, 0.020),

    # Small islands and coastal rocks in the western Inland Sea.
    (131.35, 33.98, 0.035, 0.022),
    (131.55, 34.02, 0.030, 0.020),
    (131.78, 34.08, 0.030, 0.020),
    (132.00, 34.09, 0.035, 0.022),
    (132.12, 34.12, 0.030, 0.020),

    # Ise Bay islands and nearby coastal islets.
    (136.72, 34.65, 0.025, 0.020),
    (136.57, 34.58, 0.025, 0.018),
]


def ellipse_polygon(cx, cy, rx, ry, vertices=12):
    return [
        (
            cx + rx * math.cos(2.0 * math.pi * i / vertices),
            cy + ry * math.sin(2.0 * math.pi * i / vertices),
        )
        for i in range(vertices)
    ]


def point_on_segment(x, y, x1, y1, x2, y2, eps=1e-10):
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > eps:
        return False
    return (
        min(x1, x2) - eps <= x <= max(x1, x2) + eps
        and min(y1, y2) - eps <= y <= max(y1, y2) + eps
    )


def point_in_polygon(lon, lat, polygon):
    """Ray-casting point-in-polygon test; boundary points count as land."""
    inside = False
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        if point_on_segment(lon, lat, x1, y1, x2, y2):
            return True

        crosses_latitude = (y1 > lat) != (y2 > lat)
        if crosses_latitude:
            crossing_lon = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < crossing_lon:
                inside = not inside

    return inside


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    polygons = LAND_POLYGONS + [
        ellipse_polygon(*island) for island in ISLAND_ELLIPSES
    ]

    # Both axes use the exact requested 2/63-degree step.
    step = 2.0 / 63.0
    latitudes = [35.0 - row * step for row in range(64)]
    longitudes = [129.0 + col * step for col in range(253)]

    matrix = np.zeros((64, 253), dtype=np.uint8)
    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            matrix[row, col] = int(
                any(point_in_polygon(lon, lat, polygon) for polygon in polygons)
            )

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # White (255) is land; black (0) is water. No filtering or smoothing.
    pixels = matrix * 255
    image = Image.fromarray(pixels, mode="L")
    png_path = output_dir / "candidate.png"
    image.save(png_path)

    enlarged = image.resize((253 * 8, 64 * 8), resample=Image.Resampling.NEAREST)
    enlarged_path = output_dir / "candidate_8x.png"
    enlarged.save(enlarged_path)

    # Validate the requested dimensions, values, and pixel colors.
    assert matrix.shape == (64, 253)
    assert int(matrix.size) == 16192
    assert set(np.unique(matrix).tolist()) <= {0, 1}

    with csv_path.open("r", newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 64
    assert all(len(row) == 253 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    assert image.size == (253, 64)
    assert set(image.getdata()) <= {0, 255}
    assert enlarged.size == (2024, 512)
    assert set(enlarged.getdata()) <= {0, 255}

    print(f"Validated {matrix.shape[0]}x{matrix.shape[1]} matrix ({matrix.size} values)")
    print(f"Wrote {csv_path}, {png_path}, and {enlarged_path}")


if __name__ == "__main__":
    main()
