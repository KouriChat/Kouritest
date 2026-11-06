#!/usr/bin/env python3
"""Generate a hand-drawn, approximate land/water mask for the South China Sea region."""

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image


# Rings are hand-estimated WGS84 longitude/latitude vertices. They are not
# sourced from or fitted to external map data. Coordinates are (lon, lat).
LAND_RINGS = [
    # Mainland Southeast Asia and southern China. The ring follows an
    # approximate coastline, with the frame boundary closing its north/west.
    [
        (99.0, 23.0), (117.4, 23.0), (117.0, 22.6), (116.2, 22.3),
        (115.7, 21.9), (114.9, 21.7), (114.4, 21.9), (113.8, 22.0),
        (113.4, 22.4), (112.8, 21.9), (111.8, 21.6), (110.7, 21.5),
        (109.7, 21.5), (108.7, 21.5), (108.2, 21.7), (107.7, 21.5),
        (107.0, 20.9), (106.6, 20.3), (106.5, 19.8), (106.0, 19.3),
        (105.9, 18.7), (106.0, 18.1), (106.4, 17.5), (107.0, 17.0),
        (107.4, 16.5), (107.8, 15.9), (108.2, 15.3), (108.5, 14.7),
        (109.0, 14.0), (109.3, 13.4), (109.4, 12.8), (109.1, 12.1),
        (108.8, 11.5), (108.1, 10.9), (107.5, 10.3), (106.8, 9.8),
        (106.2, 9.3), (105.6, 9.0), (105.0, 8.8), (104.7, 8.5),
        (104.2, 8.5), (103.7, 8.6), (103.1, 9.0), (102.7, 9.5),
        (102.4, 10.0), (101.9, 10.5), (101.4, 11.2), (100.9, 12.0),
        (100.5, 12.8), (100.0, 13.5), (99.5, 14.3), (99.0, 15.0),
        (99.0, 23.0),
    ],
    # Malay Peninsula, including the narrow southern end.
    [
        (99.6, 7.1), (100.0, 6.5), (100.3, 5.8), (100.4, 5.0),
        (100.5, 4.2), (100.7, 3.4), (101.0, 2.8), (101.4, 2.2),
        (102.0, 1.7), (102.7, 1.4), (103.3, 1.2), (103.7, 1.0),
        (104.0, 1.0), (104.1, 1.6), (103.9, 2.3), (103.7, 3.0),
        (103.5, 3.8), (103.4, 4.6), (103.4, 5.4), (103.3, 6.1),
        (103.2, 6.7), (102.8, 7.2), (102.2, 7.7), (101.7, 8.1),
        (101.2, 8.4), (100.7, 8.3), (100.2, 8.0), (99.8, 7.6),
    ],
    # Taiwan.
    [
        (120.0, 21.8), (120.3, 22.2), (120.7, 22.6), (121.0, 23.0),
        (121.5, 23.0), (121.8, 22.7), (121.7, 22.1), (121.4, 21.7),
        (121.0, 21.4), (120.6, 21.8),
    ],
    # Hainan.
    [
        (108.6, 19.2), (109.0, 19.6), (109.6, 19.8), (110.2, 19.8),
        (110.7, 19.5), (111.0, 19.0), (110.7, 18.5), (110.2, 18.2),
        (109.5, 18.2), (109.0, 18.5),
    ],
    # Leizhou Peninsula.
    [
        (109.5, 20.8), (110.0, 21.0), (110.5, 20.8), (110.6, 20.3),
        (110.4, 19.9), (109.9, 20.0), (109.6, 20.3),
    ],
    # Borneo.
    [
        (109.5, 7.2), (110.2, 7.3), (111.0, 7.1), (112.0, 7.0),
        (113.0, 7.1), (114.0, 7.0), (115.0, 6.9), (116.0, 6.8),
        (117.0, 6.5), (117.8, 5.8), (118.2, 5.0), (117.7, 4.2),
        (117.0, 3.4), (116.0, 2.8), (115.0, 2.2), (114.0, 1.7),
        (113.0, 1.2), (111.5, 1.0), (110.5, 1.2), (109.7, 2.0),
        (109.2, 3.0), (109.0, 4.0), (109.2, 5.0), (109.3, 6.0),
    ],
    # Sumatra, clipped naturally by the southern frame.
    [
        (99.7, 5.8), (100.2, 5.4), (101.0, 4.8), (101.8, 4.3),
        (102.5, 3.7), (103.0, 3.0), (103.3, 2.2), (103.5, 1.0),
        (102.5, 1.0), (101.7, 1.3), (101.0, 2.0), (100.4, 3.0),
        (99.9, 4.0), (99.6, 5.0),
    ],
    # Palawan.
    [
        (117.0, 11.8), (117.7, 11.9), (118.3, 11.4), (118.5, 10.7),
        (118.2, 9.8), (117.8, 8.9), (117.3, 8.0), (116.9, 7.4),
        (116.5, 7.7), (116.7, 8.7), (116.9, 9.7), (116.9, 10.8),
    ],
    # Luzon.
    [
        (119.7, 18.5), (120.2, 18.3), (120.7, 17.8), (121.2, 17.5),
        (121.5, 16.8), (121.4, 16.0), (121.2, 15.5), (121.4, 15.0),
        (121.1, 14.4), (120.7, 14.0), (120.2, 14.2), (120.0, 14.8),
        (119.7, 15.3), (119.5, 16.0), (119.4, 16.8), (119.5, 17.5),
    ],
    # Mindoro.
    [
        (120.4, 13.5), (120.9, 13.4), (121.1, 13.0), (120.9, 12.5),
        (120.5, 12.3), (120.1, 12.7),
    ],
    # Samar and Leyte.
    [
        (124.0, 12.5), (124.7, 12.6), (125.2, 12.2), (125.0, 11.6),
        (124.5, 11.3), (124.1, 11.7),
    ],
    [
        (124.7, 11.5), (125.2, 11.4), (125.3, 10.7), (124.9, 10.1),
        (124.5, 10.4), (124.4, 11.0),
    ],
    # Negros, Panay, Cebu, Bohol and nearby Visayan islands.
    [
        (122.0, 11.9), (122.7, 11.9), (123.0, 11.5), (122.7, 10.8),
        (122.2, 10.4), (121.9, 10.8),
    ],
    [
        (123.0, 11.7), (123.4, 11.8), (123.5, 10.9), (123.2, 10.1),
        (122.9, 10.5),
    ],
    [
        (123.5, 10.8), (124.0, 10.8), (124.1, 10.0), (123.7, 9.7),
        (123.4, 10.1),
    ],
    # Mindanao.
    [
        (122.0, 9.8), (122.5, 9.7), (123.0, 9.4), (123.6, 9.3),
        (124.2, 8.8), (125.0, 8.4), (125.5, 7.8), (125.4, 7.0),
        (125.0, 6.5), (124.4, 6.1), (123.7, 6.0), (123.0, 6.3),
        (122.5, 6.8), (122.0, 7.3), (121.7, 8.0), (121.8, 8.8),
    ],
    # Andaman and Nicobar island groups, in the western edge of the box.
    [
        (92.0, 14.0), (94.0, 14.0), (94.0, 10.0), (92.0, 10.0),
    ],
    [
        (92.0, 10.0), (94.0, 10.0), (94.0, 6.0), (92.0, 6.0),
    ],
]


# Small islands and reef groups are represented as small circles in degrees.
# The chosen radii deliberately keep most reef groups much smaller than a
# mainland coastline feature.
SMALL_LAND_CIRCLES = [
    # Gulf of Tonkin islands and coastal islands.
    (107.8, 20.8, 0.10), (108.5, 20.5, 0.08), (109.0, 20.4, 0.07),
    # Paracel Islands.
    (111.7, 16.5, 0.09), (112.0, 16.5, 0.06), (111.5, 16.7, 0.05),
    # Spratly Islands and reef groups.
    (114.3, 11.9, 0.07), (114.4, 10.7, 0.06), (115.7, 10.4, 0.06),
    (116.0, 9.8, 0.06), (115.0, 9.7, 0.05), (113.9, 9.7, 0.05),
    (114.3, 8.7, 0.05), (116.7, 8.8, 0.05), (117.5, 8.4, 0.04),
    # Other South China Sea islands.
    (109.5, 7.5, 0.06), (108.7, 7.5, 0.04),
    # Ryukyu / southern Japanese island fragments near the northeast edge.
    (124.2, 23.0, 0.08), (124.6, 24.0, 0.08),
    # Sulu and southern Philippine islands.
    (120.0, 6.0, 0.05), (120.9, 6.0, 0.05), (119.8, 5.5, 0.04),
    (126.0, 7.0, 0.05), (126.5, 7.5, 0.04),
]


def point_in_ring(lon, lat, ring):
    """Ray-casting point-in-polygon test."""
    inside = False
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat):
            x_cross = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_cross:
                inside = not inside
        j = i
    return inside


def is_land(lon, lat):
    if any(point_in_ring(lon, lat, ring) for ring in LAND_RINGS):
        return True

    for center_lon, center_lat, radius in SMALL_LAND_CIRCLES:
        # Scale longitude distance by latitude for approximate circular disks.
        dx = (lon - center_lon) * math.cos(math.radians(center_lat))
        dy = lat - center_lat
        if dx * dx + dy * dy <= radius * radius:
            return True
    return False


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in range(111):
        lat = round(23.0 - 0.2 * row, 10)
        rows.append([
            1 if is_land(round(99.0 + 0.2 * col, 10), lat) else 0
            for col in range(141)
        ])

    matrix = np.asarray(rows, dtype=np.uint8)
    assert matrix.shape == (111, 141)
    assert int(matrix.size) == 15651
    assert set(np.unique(matrix).tolist()) <= {0, 1}

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Pixel value 255 is white land; 0 is black water.
    pixels = (matrix * 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    image.save(output_dir / "candidate.png")

    enlarged = image.resize((141 * 8, 111 * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")

    # Validate the exported artifacts.
    with csv_path.open("r", newline="", encoding="ascii") as file:
        csv_rows = list(csv.reader(file))
    assert len(csv_rows) == 111
    assert all(len(row) == 141 for row in csv_rows)
    assert all(value in {"0", "1"} for row in csv_rows for value in row)

    check_image = Image.open(output_dir / "candidate.png")
    assert check_image.size == (141, 111)
    assert set(check_image.getdata()) <= {0, 255}

    check_enlarged = Image.open(output_dir / "candidate_8x.png")
    assert check_enlarged.size == (1128, 888)
    assert set(check_enlarged.getdata()) <= {0, 255}
    assert np.array_equal(
        np.asarray(check_enlarged),
        np.repeat(np.repeat(pixels, 8, axis=0), 8, axis=1),
    )

    print("Generated outputs/candidate.csv (111 x 141)")
    print("Generated outputs/candidate.png (141 x 111)")
    print("Generated outputs/candidate_8x.png (1128 x 888)")


if __name__ == "__main__":
    main()
