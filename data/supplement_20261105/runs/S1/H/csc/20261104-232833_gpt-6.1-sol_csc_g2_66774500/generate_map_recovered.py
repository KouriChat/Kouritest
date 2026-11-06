#!/usr/bin/env python3
"""Generate a hand-drawn, approximate South China Sea land/water raster.

No external geographic data is read or downloaded. Coordinates are WGS84
longitude/latitude; each grid sample is classified by point-in-polygon.
"""

from pathlib import Path
import csv

import numpy as np
from PIL import Image


LON_MIN, LON_MAX, LON_STEP = 99.0, 127.0, 0.2
LAT_MIN, LAT_MAX, LAT_STEP = 1.0, 23.0, 0.2
SCALE = 8

# Hand-estimated outlines, ordered around each land polygon. The outlines are
# intentionally approximate and are not intended for navigational use.
POLYGONS = [
    # Mainland Southeast Asia and southern China: broad outer coastline.
    [
        (99.0, 23.0), (102.0, 23.0), (105.0, 22.9), (106.0, 22.5),
        (107.0, 21.7), (108.0, 20.8), (108.5, 19.2), (107.8, 17.5),
        (108.2, 16.0), (109.0, 14.0), (109.5, 12.0), (109.3, 10.5),
        (108.4, 9.5), (107.0, 10.0), (105.5, 10.5), (104.0, 10.5),
        (103.0, 10.8), (102.0, 11.3), (101.0, 12.0), (100.0, 13.0),
        (99.0, 14.5), (99.0, 18.0),
    ],
    # Malay Peninsula.
    [
        (99.7, 7.0), (100.2, 7.0), (100.8, 6.5), (101.2, 5.5),
        (101.8, 4.5), (102.2, 3.5), (103.0, 2.5), (103.5, 1.2),
        (102.0, 1.2), (101.3, 2.0), (100.8, 3.0), (100.3, 4.0),
        (99.8, 5.0), (99.5, 6.0),
    ],
    # Hainan.
    [
        (108.55, 19.55), (109.15, 19.65), (109.75, 19.35),
        (110.55, 19.05), (110.75, 18.45), (110.25, 18.15),
        (109.55, 18.15), (109.0, 18.35), (108.65, 18.8),
    ],
    # Taiwan.
    [
        (120.0, 25.0), (121.0, 25.0), (121.6, 24.0), (121.5, 23.0),
        (121.2, 22.0), (120.8, 21.8), (120.3, 22.6), (120.0, 23.5),
    ],
    # Borneo.
    [
        (109.5, 7.0), (110.0, 4.5), (111.5, 1.0), (114.0, 0.8),
        (116.5, 1.5), (117.8, 3.0), (118.2, 5.0), (117.5, 7.0),
        (116.0, 7.5), (114.5, 6.8), (113.0, 4.5), (111.5, 3.5),
    ],
    # Sumatra, clipped naturally by the southern edge of the requested grid.
    [
        (99.0, 5.7), (100.0, 5.5), (101.0, 4.5), (102.0, 3.5),
        (103.0, 2.0), (104.0, 0.5), (102.0, -1.0), (100.0, -1.0),
        (99.0, 0.0),
    ],
    # Java, only its northern edge enters the sampling box.
    [(105.0, -5.5), (114.5, -5.5), (114.0, -7.0), (105.5, -7.5)],
    # Luzon.
    [
        (119.7, 18.5), (120.5, 18.6), (121.1, 17.8), (121.8, 16.5),
        (121.5, 15.0), (121.0, 14.2), (120.5, 13.0), (120.0, 13.0),
        (119.7, 14.5), (119.5, 16.0),
    ],
    # Mindoro, Palawan, and the central and southern Philippine islands.
    [(120.3, 13.5), (121.0, 13.5), (121.2, 12.5), (120.7, 12.0), (120.2, 12.5)],
    [(117.0, 11.5), (118.0, 11.5), (118.5, 10.0), (118.2, 8.0), (117.5, 7.5), (117.0, 9.0)],
    [(122.0, 11.8), (123.0, 11.8), (123.5, 10.5), (123.0, 9.5), (122.5, 9.5)],
    [(123.5, 12.0), (124.7, 12.0), (125.2, 11.0), (124.7, 10.5), (123.8, 11.0)],
    [(124.0, 10.0), (125.3, 9.8), (126.0, 8.5), (125.5, 7.0), (124.5, 6.0), (123.5, 6.5), (123.5, 8.0)],
    [(124.0, 7.0), (125.5, 7.0), (126.0, 6.0), (125.5, 5.0), (124.5, 5.0)],
    [(121.0, 10.0), (121.8, 10.0), (122.0, 9.0), (121.5, 8.5), (121.0, 9.0)],
    # Taiwan-adjacent and other larger islands.
    [(119.3, 22.0), (119.8, 22.0), (120.0, 21.5), (119.5, 21.2)],
    # Natuna and Anambas groups, represented as small land patches.
    [(108.0, 4.0), (108.5, 4.0), (108.6, 3.6), (108.1, 3.5)],
    [(106.0, 3.2), (106.5, 3.2), (106.6, 2.8), (106.1, 2.8)],
    # Spratly Islands: coarse, small land patches, not a blanket reef region.
    [(111.5, 12.0), (111.8, 12.0), (111.9, 11.8), (111.6, 11.7)],
    [(114.3, 11.5), (114.6, 11.5), (114.7, 11.2), (114.4, 11.2)],
    [(115.8, 10.5), (116.1, 10.5), (116.2, 10.2), (115.9, 10.2)],
    [(113.8, 9.8), (114.1, 9.8), (114.2, 9.5), (113.9, 9.5)],
    [(112.8, 8.7), (113.1, 8.7), (113.2, 8.4), (112.9, 8.4)],
    [(116.5, 9.0), (116.8, 9.0), (116.9, 8.7), (116.6, 8.7)],
    # Paracel Islands.
    [(111.5, 16.5), (111.9, 16.5), (112.0, 16.2), (111.6, 16.1)],
    [(112.5, 16.0), (112.8, 16.0), (112.9, 15.7), (112.6, 15.7)],
]


def points_in_polygon(lon_grid, lat_grid, polygon):
    """Vectorized ray-crossing point-in-polygon test."""
    inside = np.zeros(lon_grid.shape, dtype=bool)
    vertices = polygon
    n = len(vertices)

    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        crosses_y = (y1 > lat_grid) != (y2 > lat_grid)
        denominator = y2 - y1
        if denominator == 0:
            continue
        crossing_lon = x1 + (lat_grid - y1) * (x2 - x1) / denominator
        inside ^= crosses_y & (lon_grid < crossing_lon)

    return inside


def main():
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Construct endpoints by integer steps to avoid floating-point arange drift.
    lons = LON_MIN + LON_STEP * np.arange(141)
    lats = LAT_MAX - LAT_STEP * np.arange(111)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    land = np.zeros((111, 141), dtype=bool)
    for polygon in POLYGONS:
        land |= points_in_polygon(lon_grid, lat_grid, polygon)

    matrix = land.astype(np.uint8)

    csv_path = out_dir / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Pillow uses RGB values here: land is white and water is black.
    pixels = np.where(land[:, :, None], 255, 0).astype(np.uint8)
    rgb = np.repeat(pixels, 3, axis=2)
    original = Image.fromarray(rgb, mode="RGB")
    original.save(out_dir / "candidate.png")

    enlarged = original.resize(
        (141 * SCALE, 111 * SCALE),
        resample=Image.Resampling.NEAREST,
    )
    enlarged.save(out_dir / "candidate_8x.png")

    # Validate matrix, CSV, palette, dimensions, and exact nearest-neighbor scale.
    assert matrix.shape == (111, 141)
    assert matrix.size == 15651
    assert np.isin(matrix, (0, 1)).all()

    with csv_path.open("r", newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 111
    assert all(len(row) == 141 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    check_original = Image.open(out_dir / "candidate.png").convert("RGB")
    check_enlarged = Image.open(out_dir / "candidate_8x.png").convert("RGB")
    assert check_original.size == (141, 111)
    assert check_enlarged.size == (1128, 888)
    assert set(check_original.getdata()) <= {(0, 0, 0), (255, 255, 255)}
    assert check_enlarged == check_original.resize(
        (1128, 888), resample=Image.Resampling.NEAREST
    )

    print(f"Created {csv_path}, candidate.png, and candidate_8x.png")
    print("Validated 111 x 141 = 15651 binary samples and image dimensions.")


if __name__ == "__main__":
    main()
