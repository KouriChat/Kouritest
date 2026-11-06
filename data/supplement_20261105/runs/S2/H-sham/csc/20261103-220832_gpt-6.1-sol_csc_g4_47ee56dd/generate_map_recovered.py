#!/usr/bin/env python3
"""Build a hand-estimated South China Sea land/water raster."""

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image


# Hand-drawn approximate outlines, as (longitude, latitude) vertices.
# The outlines are intentionally approximate and are not sourced from map data.
LAND_POLYGONS = [
    # Mainland China, Indochina, Thailand, and the Malay Peninsula.
    [
        (99.0, 25.0), (130.0, 25.0), (130.0, 24.0), (124.0, 24.0),
        (122.0, 24.0), (121.0, 23.5), (120.0, 23.0), (119.0, 24.0),
        (118.0, 24.0), (117.0, 23.5), (116.3, 22.5), (115.0, 22.7),
        (114.5, 22.4), (113.5, 22.7), (112.5, 21.8), (111.5, 21.4),
        (110.5, 21.2), (109.5, 21.5), (108.2, 21.6), (107.5, 20.8),
        (106.5, 20.6), (106.7, 19.6), (105.9, 19.1), (105.8, 18.2),
        (106.0, 17.4), (107.0, 16.2), (108.0, 15.8), (108.5, 14.5),
        (109.0, 13.5), (109.5, 12.5), (109.4, 11.5), (108.8, 10.5),
        (107.5, 9.5), (106.5, 9.0), (105.5, 8.8), (104.5, 9.5),
        (103.7, 10.4), (103.0, 10.6), (102.5, 10.0), (102.0, 9.5),
        (101.5, 8.5), (101.2, 7.5), (101.5, 6.5), (102.0, 5.5),
        (102.5, 4.5), (103.0, 3.5), (103.5, 2.5), (104.0, 1.0),
        (103.2, 1.0), (102.5, 2.0), (101.5, 3.0), (100.5, 4.0),
        (100.0, 5.0), (99.7, 6.0), (99.5, 7.0), (99.0, 8.0),
        (98.5, 9.0), (98.3, 10.0), (98.5, 11.0), (98.7, 12.0),
        (98.8, 13.0), (98.5, 14.0), (98.0, 15.0), (98.0, 17.0),
        (98.5, 19.0), (99.0, 21.0), (99.0, 25.0),
    ],

    # Hainan.
    [
        (108.6, 20.2), (109.1, 20.3), (109.7, 19.9), (110.4, 19.3),
        (110.5, 18.6), (110.0, 18.2), (109.4, 18.2), (108.8, 18.5),
        (108.6, 19.2),
    ],

    # Taiwan.
    [
        (120.0, 25.3), (121.0, 25.3), (121.8, 24.7), (121.5, 23.8),
        (121.0, 23.0), (120.7, 22.2), (120.3, 21.9), (120.0, 22.7),
        (120.0, 24.0),
    ],

    # Northern Philippines: Luzon and nearby islands.
    [
        (119.7, 18.6), (120.5, 18.8), (121.0, 18.4), (121.8, 18.5),
        (122.2, 17.8), (122.0, 16.8), (121.5, 16.0), (121.5, 15.0),
        (121.0, 14.0), (120.5, 13.5), (120.0, 14.0), (120.0, 15.0),
        (119.5, 16.0), (119.5, 17.0),
    ],
    # Mindoro.
    [
        (120.4, 13.5), (121.0, 13.5), (121.2, 13.0), (120.9, 12.3),
        (120.5, 12.2), (120.2, 12.8),
    ],
    # Samar and Leyte.
    [
        (124.0, 12.8), (124.8, 12.7), (125.2, 11.8), (125.0, 10.8),
        (124.5, 10.2), (124.0, 10.7), (124.1, 11.8),
    ],
    # Panay and Negros.
    [
        (121.8, 11.8), (122.5, 11.8), (123.0, 11.2), (123.0, 10.2),
        (122.5, 9.8), (122.1, 10.4),
    ],
    # Cebu and Bohol.
    [
        (123.3, 10.7), (123.8, 10.8), (124.0, 10.1), (123.7, 9.6),
        (123.4, 10.0),
    ],
    [
        (123.8, 10.0), (124.5, 10.0), (124.8, 9.6), (124.2, 9.5),
    ],
    # Mindanao.
    [
        (123.5, 9.8), (124.5, 9.5), (125.5, 8.8), (126.2, 7.8),
        (126.0, 6.5), (125.2, 6.0), (124.5, 6.2), (123.8, 7.0),
        (123.2, 8.0),
    ],
    # Palawan.
    [
        (117.0, 11.8), (117.6, 11.5), (118.0, 10.5), (118.2, 9.5),
        (118.5, 8.5), (118.3, 7.5), (117.8, 7.0), (117.4, 8.0),
        (117.2, 9.0), (116.8, 10.0),
    ],

    # Northern Borneo (Sabah, Sarawak, Brunei, and adjacent Borneo).
    [
        (109.0, 7.3), (109.8, 7.0), (110.5, 6.5), (111.5, 6.8),
        (112.5, 6.5), (113.5, 5.8), (114.5, 5.5), (115.5, 5.0),
        (116.5, 4.8), (117.5, 4.3), (118.0, 3.5), (117.5, 2.5),
        (116.5, 1.8), (115.5, 1.5), (114.5, 1.2), (113.5, 1.0),
        (112.0, 1.0), (111.0, 1.5), (110.0, 2.0), (109.5, 3.0),
        (109.0, 4.0), (108.8, 5.0), (109.0, 6.0),
    ],

    # Sumatra's eastern edge, which enters the western edge of this frame.
    [
        (99.0, 5.5), (99.5, 5.2), (100.0, 4.5), (100.5, 3.5),
        (101.0, 2.5), (101.5, 1.5), (102.0, 1.0), (99.0, 1.0),
    ],
]


# Small islands and exposed reef/atoll centers. Each entry is
# (longitude, latitude, approximate radius in degrees).
# The radius is deliberately modest to avoid filling surrounding sea.
ISLANDS_AND_REEFS = [
    # Gulf of Tonkin and southern China Sea.
    (111.7, 16.5, 0.10), (112.0, 16.5, 0.08), (112.3, 16.7, 0.08),
    (114.3, 10.4, 0.08), (115.8, 10.0, 0.08), (116.5, 9.8, 0.07),
    (117.2, 8.8, 0.08), (115.0, 8.0, 0.08),

    # Paracel Islands.
    (111.5, 16.5, 0.10), (112.0, 16.7, 0.09), (112.5, 16.5, 0.08),

    # Spratly region: selected reefs and low islands.
    (114.2, 11.9, 0.08), (114.3, 11.6, 0.08), (114.4, 11.0, 0.08),
    (114.7, 10.8, 0.08), (115.0, 10.8, 0.08), (115.2, 10.3, 0.08),
    (116.0, 10.0, 0.08), (116.5, 10.0, 0.08), (116.8, 9.9, 0.08),
    (117.0, 9.8, 0.08), (117.2, 9.7, 0.08), (117.5, 9.5, 0.08),
    (118.0, 9.0, 0.08), (118.5, 8.7, 0.08), (119.0, 8.7, 0.08),
    (119.2, 8.1, 0.08), (119.5, 7.8, 0.08), (116.0, 7.5, 0.08),
]


def point_in_polygon(lon, lat, polygon):
    """Return whether a point lies inside a polygon using ray crossing."""
    inside = False
    count = len(polygon)
    j = count - 1

    for i in range(count):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # Treat points on polygon edges as land.
        cross = (lon - xi) * (yj - yi) - (lat - yi) * (xj - xi)
        if (
            abs(cross) < 1e-10
            and min(xi, xj) - 1e-10 <= lon <= max(xi, xj) + 1e-10
            and min(yi, yj) - 1e-10 <= lat <= max(yi, yj) + 1e-10
        ):
            return True

        if (yi > lat) != (yj > lat):
            intersection_lon = xi + (lat - yi) * (xj - xi) / (yj - yi)
            if lon < intersection_lon:
                inside = not inside
        j = i

    return inside


def is_land(lon, lat):
    if any(point_in_polygon(lon, lat, polygon) for polygon in LAND_POLYGONS):
        return True

    for island_lon, island_lat, radius in ISLANDS_AND_REEFS:
        # Longitude distance is scaled by latitude for approximate distance.
        dx = (lon - island_lon) * math.cos(math.radians(lat))
        dy = lat - island_lat
        if dx * dx + dy * dy <= radius * radius:
            return True

    return False


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = 111
    cols = 141
    matrix = np.zeros((rows, cols), dtype=np.uint8)

    for row in range(rows):
        lat = 23.0 - 0.2 * row
        for col in range(cols):
            lon = 99.0 + 0.2 * col
            matrix[row, col] = 1 if is_land(lon, lat) else 0

    with (output_dir / "candidate.csv").open(
        "w", newline="", encoding="ascii"
    ) as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Mode "1" stores exactly two pixel values; convert to RGB for black/white PNG.
    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(output_dir / "candidate.png")

    enlarged = image.resize(
        (cols * 8, rows * 8),
        resample=Image.Resampling.NEAREST,
    )
    enlarged.save(output_dir / "candidate_8x.png")

    # Verify the matrix, CSV shape/content, image dimensions, colors, and scaling.
    assert matrix.shape == (111, 141)
    assert int(matrix.size) == 15651
    assert set(np.unique(matrix).tolist()) <= {0, 1}

    with (output_dir / "candidate.csv").open(
        "r", newline="", encoding="ascii"
    ) as csv_file:
        csv_rows = list(csv.reader(csv_file))
    assert len(csv_rows) == 111
    assert all(len(row) == 141 for row in csv_rows)
    assert all(value in {"0", "1"} for row in csv_rows for value in row)

    original = Image.open(output_dir / "candidate.png")
    scaled = Image.open(output_dir / "candidate_8x.png")
    assert original.size == (141, 111)
    assert scaled.size == (1128, 888)
    assert set(original.getdata()) <= {(0, 0, 0), (255, 255, 255)}
    assert scaled.resize((141, 111), Image.Resampling.NEAREST).tobytes() == (
        original.tobytes()
    )

    print("Generated and verified outputs/candidate.csv")
    print("Generated and verified outputs/candidate.png (141x111)")
    print("Generated and verified outputs/candidate_8x.png (1128x888)")


if __name__ == "__main__":
    main()
