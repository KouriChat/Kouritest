from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image


# Approximate hand-drawn coastlines in WGS84 decimal degrees.
# Each polygon represents land; points outside all polygons are water.
LAND_POLYGONS = [
    # Mainland Asia and the Malay Peninsula. The polygon is clipped to the
    # requested frame; its top and left edges intentionally run off-frame.
    [
        (99.0, 23.0), (127.0, 23.0), (117.0, 23.0),
        (116.5, 22.8), (115.8, 22.7), (115.2, 22.5),
        (114.8, 22.5), (114.5, 22.3), (114.1, 22.5),
        (113.5, 22.2), (112.9, 21.9), (112.2, 21.7),
        (111.5, 21.5), (110.9, 21.2), (110.5, 20.9),
        (110.3, 20.6), (109.8, 20.4), (109.4, 20.3),
        (109.0, 20.2), (108.6, 20.1), (108.3, 20.4),
        (108.1, 20.7), (107.8, 21.0), (107.5, 21.2),
        (107.2, 21.3), (106.8, 21.0), (106.8, 20.7),
        (106.7, 20.4), (106.5, 20.1), (106.4, 19.8),
        (106.3, 19.5), (106.2, 19.1), (106.0, 18.7),
        (105.9, 18.3), (106.0, 17.9), (106.5, 17.5),
        (107.0, 17.0), (107.5, 16.5), (108.0, 16.1),
        (108.2, 15.8), (108.0, 15.3), (107.7, 14.8),
        (107.4, 14.2), (107.2, 13.7), (109.0, 13.0),
        (109.4, 12.5), (109.4, 12.0), (109.3, 11.5),
        (109.0, 11.0), (108.7, 10.5), (108.3, 10.0),
        (107.8, 9.5), (107.3, 9.0), (106.8, 8.7),
        (106.3, 8.6), (105.8, 8.5), (105.2, 8.6),
        (104.8, 9.0), (104.4, 9.5), (104.0, 10.0),
        (103.5, 10.5), (103.0, 10.7), (102.6, 10.5),
        (102.4, 10.0), (102.3, 9.5), (102.3, 9.0),
        (102.5, 8.5), (102.6, 8.0), (102.5, 7.5),
        (102.2, 7.0), (101.8, 6.5), (101.5, 6.0),
        (101.3, 5.5), (101.2, 5.0), (101.2, 4.5),
        (101.3, 4.0), (101.5, 3.5), (101.7, 3.0),
        (102.0, 2.5), (102.4, 2.0), (103.0, 1.5),
        (103.5, 1.2), (104.2, 1.0),
        # East coast of the peninsula, then its west coast northward.
        (103.3, 1.0), (102.8, 1.3), (102.4, 1.8),
        (101.9, 2.3), (101.4, 2.8), (101.0, 3.4),
        (100.6, 4.0), (100.2, 4.7), (99.8, 5.4),
        (99.5, 6.0), (99.2, 6.6), (99.0, 7.0),
        (99.0, 23.0),
    ],

    # Hainan
    [
        (108.6, 19.2), (108.7, 19.7), (109.0, 20.0),
        (109.5, 20.1), (110.0, 20.0), (110.5, 19.8),
        (110.8, 19.5), (111.0, 19.0), (110.8, 18.5),
        (110.5, 18.2), (110.0, 18.2), (109.5, 18.2),
        (109.0, 18.4), (108.7, 18.8),
    ],

    # Taiwan
    [
        (120.0, 21.9), (120.2, 22.3), (120.6, 22.6),
        (121.0, 23.0), (121.3, 23.5), (121.5, 24.0),
        (121.6, 24.5), (121.4, 25.0), (121.1, 25.2),
        (120.8, 25.1), (120.5, 24.8), (120.3, 24.3),
        (120.2, 23.8), (120.1, 23.2), (120.0, 22.6),
    ],

    # Borneo
    [
        (109.5, 7.0), (109.4, 5.8), (109.8, 4.7),
        (110.5, 3.8), (111.0, 2.8), (111.5, 1.8),
        (112.0, 1.0), (114.0, 1.0), (115.0, 1.4),
        (116.0, 2.0), (117.0, 3.0), (117.8, 4.0),
        (118.0, 5.0), (117.8, 6.0), (117.5, 7.0),
        (117.0, 7.5), (116.0, 7.8), (115.0, 7.9),
        (114.0, 7.5), (113.0, 7.2), (112.0, 7.0),
        (111.0, 7.1), (110.0, 7.2),
    ],

    # Sumatra, clipped at the southern frame edge
    [
        (99.0, 1.0), (104.0, 1.0), (103.5, 1.6),
        (103.2, 2.2), (102.8, 3.0), (102.3, 4.0),
        (101.8, 5.0), (101.2, 6.0), (100.8, 6.5),
        (100.2, 6.3), (99.8, 5.8), (99.5, 5.0),
        (99.2, 4.0), (99.0, 3.0),
    ],

    # Luzon
    [
        (119.7, 17.0), (120.0, 17.5), (120.4, 18.2),
        (120.8, 18.8), (121.2, 19.4), (121.7, 19.8),
        (122.0, 19.5), (122.0, 18.8), (121.8, 18.2),
        (121.5, 17.6), (121.3, 17.0), (121.2, 16.5),
        (121.5, 16.0), (121.5, 15.5), (121.2, 15.0),
        (120.8, 14.8), (120.5, 15.2), (120.3, 15.8),
        (120.0, 16.2), (119.6, 16.5),
    ],

    # Mindoro
    [
        (120.4, 13.2), (120.5, 13.8), (121.0, 13.5),
        (121.2, 13.0), (121.1, 12.5), (120.8, 12.3),
        (120.5, 12.6),
    ],

    # Palawan
    [
        (117.0, 11.8), (117.2, 11.0), (117.5, 10.0),
        (118.0, 9.0), (118.3, 8.0), (118.5, 7.0),
        (118.3, 6.0), (118.0, 6.5), (117.7, 7.5),
        (117.4, 8.5), (117.1, 9.5), (116.8, 10.5),
        (116.7, 11.2),
    ],

    # Panay
    [
        (121.8, 11.8), (122.2, 11.7), (122.7, 11.5),
        (122.8, 11.1), (122.5, 10.7), (122.0, 10.7),
        (121.7, 11.0),
    ],

    # Negros
    [
        (122.4, 10.9), (122.8, 10.5), (123.0, 9.8),
        (123.0, 9.2), (122.7, 9.0), (122.4, 9.5),
        (122.2, 10.2),
    ],

    # Cebu
    [
        (123.3, 11.3), (123.6, 10.8), (123.7, 10.0),
        (123.5, 9.5), (123.3, 10.0), (123.2, 10.7),
    ],

    # Samar
    [
        (124.0, 12.0), (124.5, 12.3), (125.0, 12.1),
        (125.2, 11.5), (125.0, 11.0), (124.6, 10.8),
        (124.2, 11.2),
    ],

    # Leyte
    [
        (124.2, 11.5), (124.6, 11.4), (124.8, 10.8),
        (124.6, 10.0), (124.3, 9.8), (124.1, 10.4),
    ],

    # Mindanao
    [
        (123.0, 9.8), (123.3, 8.8), (123.8, 8.2),
        (124.2, 7.7), (124.8, 7.4), (125.3, 7.0),
        (125.8, 6.5), (126.2, 6.0), (126.3, 7.0),
        (126.0, 8.0), (125.5, 8.7), (125.2, 9.5),
        (124.8, 9.8), (124.5, 9.5), (124.1, 9.8),
        (123.8, 10.0), (123.5, 9.5),
    ],

    # Bohol and nearby larger islands
    [
        (123.7, 10.1), (124.2, 10.2), (124.5, 9.9),
        (124.1, 9.7), (123.8, 9.8),
    ],

    # Sulawesi northern arm and the land at the southern edge
    [
        (119.0, 1.0), (119.3, 1.4), (119.7, 1.8),
        (120.2, 1.8), (120.5, 1.5), (120.8, 1.0),
    ],
]

# Small islands represented as compact hand-drawn ellipses:
# (center longitude, center latitude, radius longitude, radius latitude).
ISLANDS = [
    # Taiwan Strait, Ryukyu chain
    (119.5, 23.6, 0.18, 0.12),  # Penghu
    (121.5, 24.0, 0.06, 0.05),  # northeast Taiwan is already covered
    (122.9, 24.8, 0.08, 0.06), (123.0, 24.4, 0.07, 0.05),
    (123.8, 24.4, 0.09, 0.06), (124.2, 24.0, 0.10, 0.06),
    (124.8, 24.5, 0.10, 0.07), (125.3, 24.7, 0.12, 0.08),
    (125.5, 25.0, 0.10, 0.07), (126.0, 26.0, 0.16, 0.10),

    # Gulf of Tonkin and South China coast islands
    (107.0, 20.8, 0.10, 0.07), (108.6, 21.0, 0.08, 0.06),
    (109.5, 21.5, 0.08, 0.06), (113.9, 22.3, 0.08, 0.05),

    # Spratly: only actual emergent features are treated as land.
    (114.36, 8.65, 0.035, 0.025),
    (114.22, 10.38, 0.025, 0.02),
    (115.82, 10.38, 0.025, 0.02),
    (116.72, 9.78, 0.025, 0.02),
    (117.25, 8.83, 0.025, 0.02),
    (112.34, 8.65, 0.025, 0.02),

    # Philippine smaller islands
    (123.9, 13.0, 0.12, 0.08), (124.0, 11.8, 0.08, 0.06),
    (122.0, 13.0, 0.08, 0.06), (122.7, 13.5, 0.08, 0.06),
    (123.5, 7.5, 0.10, 0.07), (125.0, 5.8, 0.08, 0.06),
    (126.0, 7.5, 0.07, 0.05),

    # Indonesia and Malaysia
    (108.0, 3.0, 0.08, 0.05), (108.5, 2.0, 0.06, 0.04),
    (104.0, 1.3, 0.10, 0.06),  # Singapore area
]


def point_on_segment(x, y, x1, y1, x2, y2, eps=1e-10):
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > eps:
        return False
    return (
        min(x1, x2) - eps <= x <= max(x1, x2) + eps
        and min(y1, y2) - eps <= y <= max(y1, y2) + eps
    )


def inside_polygon(x, y, polygon):
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if point_on_segment(x, y, x1, y1, x2, y2):
            return True
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing_x:
                inside = not inside
    return inside


def on_island(x, y, island):
    cx, cy, rx, ry = island
    if rx <= 0 or ry <= 0:
        return False
    # Approximate the island's local shape as an ellipse. Longitude is scaled
    # by cos(latitude) so the ellipses remain geographically plausible.
    dx = (x - cx) * math.cos(math.radians(cy))
    rx_scaled = rx * math.cos(math.radians(cy))
    return (dx / rx_scaled) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


def is_land(lon, lat):
    if any(inside_polygon(lon, lat, polygon) for polygon in LAND_POLYGONS):
        return 1
    if any(on_island(lon, lat, island) for island in ISLANDS):
        return 1
    return 0


def main():
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    lats = [23.0 - 0.2 * i for i in range(111)]
    lons = [99.0 + 0.2 * j for j in range(141)]

    matrix = np.fromfunction(
        np.vectorize(lambda i, j: is_land(lons[int(j)], lats[int(i)])),
        (111, 141),
        dtype=int,
    ).astype(np.uint8)

    # Headerless CSV: north to south, west to east.
    with (out_dir / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # 1 = white land, 0 = black water.
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(out_dir / "candidate.png")

    enlarged = image.resize((141 * 8, 111 * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(out_dir / "candidate_8x.png")

    # Validate dimensions, values, CSV shape, and nearest-neighbor output.
    assert matrix.shape == (111, 141)
    assert matrix.size == 15651
    assert set(np.unique(matrix).tolist()) <= {0, 1}

    with (out_dir / "candidate.csv").open("r", newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 111
    assert all(len(row) == 141 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    check = Image.open(out_dir / "candidate.png").convert("L")
    check8 = Image.open(out_dir / "candidate_8x.png").convert("L")
    assert check.size == (141, 111)
    assert set(check.getdata()) <= {0, 255}
    assert check8.size == (1128, 888)
    assert np.array_equal(
        np.asarray(check8),
        np.repeat(np.repeat(np.asarray(check), 8, axis=0), 8, axis=1),
    )

    print("Created and verified:")
    print("  outputs/candidate.csv       111 rows x 141 values")
    print("  outputs/candidate.png       141 x 111, black and white")
    print("  outputs/candidate_8x.png    1128 x 888, nearest-neighbor")


if __name__ == "__main__":
    main()
