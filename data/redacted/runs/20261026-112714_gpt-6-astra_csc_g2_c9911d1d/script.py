from pathlib import Path
import csv

import numpy as np
from PIL import Image

# Polygon vertices are (longitude, latitude), traced approximately from memory.
# Only features intersecting the requested sampling window are included.
LAND_POLYGONS = [
    # Southern China, including the mainland coastline and Pearl River region.
    [
        (99.0, 23.0), (117.4, 23.0), (117.1, 22.5), (116.6, 22.1),
        (116.0, 22.5), (115.3, 22.4), (114.8, 22.2), (114.2, 22.4),
        (113.7, 22.0), (113.2, 21.8), (112.5, 21.6), (111.7, 21.5),
        (110.8, 21.1), (110.2, 20.8), (109.6, 21.0), (109.0, 21.4),
        (108.2, 21.5), (107.8, 22.0), (106.5, 22.0), (105.5, 22.5),
        (104.0, 22.4), (102.0, 22.0), (99.0, 22.0),
    ],
    # Myanmar, Thailand, Laos and Cambodia, with a simplified Gulf of Thailand.
    [
        (99.0, 23.0), (108.2, 23.0), (108.0, 21.5), (106.5, 21.0),
        (105.5, 20.0), (104.5, 18.5), (103.5, 17.0), (102.5, 15.0),
        (101.8, 13.5), (102.0, 12.5), (103.0, 11.8), (104.0, 10.5),
        (103.0, 9.5), (101.5, 9.0), (100.2, 10.0), (99.5, 11.0),
        (99.0, 12.0),
    ],
    # Malay Peninsula and southern Thailand.
    [
        (99.0, 10.0), (100.0, 10.0), (100.7, 9.0), (101.0, 7.5),
        (101.5, 6.5), (102.0, 5.5), (102.2, 4.5), (103.0, 3.5),
        (103.5, 2.5), (104.0, 1.0), (99.0, 1.0),
    ],
    # Northern and western Sumatra.
    [
        (99.0, 5.8), (99.8, 5.5), (100.5, 4.8), (101.0, 3.8),
        (101.5, 2.8), (102.0, 1.8), (102.5, 1.0), (99.0, 1.0),
    ],
    # Java, only its northern portion falls in the frame.
    [
        (105.0, 1.0), (114.0, 1.0), (113.0, 1.4), (111.0, 1.7),
        (109.0, 1.8), (107.0, 1.6), (105.0, 1.4),
    ],
    # Borneo (Kalimantan, Brunei, Sabah and Sarawak).
    [
        (108.0, 7.0), (109.0, 7.3), (110.5, 7.0), (112.0, 7.1),
        (114.0, 7.0), (116.0, 6.5), (117.0, 5.0), (117.5, 3.5),
        (117.0, 1.0), (109.0, 1.0), (108.0, 2.5), (108.0, 4.5),
    ],
    # Taiwan.
    [
        (120.0, 21.8), (120.3, 22.1), (120.7, 22.5), (121.0, 23.0),
        (121.5, 23.0), (121.8, 22.5), (121.7, 21.8), (121.3, 21.0),
        (120.8, 21.8), (120.4, 22.0),
    ],
    # Hainan.
    [
        (108.5, 18.2), (109.0, 18.0), (109.7, 18.3), (110.0, 19.0),
        (109.7, 19.8), (109.0, 20.0), (108.6, 19.5),
    ],
    # Palawan, northern and central portions.
    [
        (117.0, 11.0), (117.4, 11.5), (118.0, 12.0), (118.5, 12.8),
        (119.0, 13.5), (119.0, 14.5), (118.5, 14.8), (118.0, 14.0),
        (117.5, 13.0), (117.0, 12.0),
    ],
    # Mindoro.
    [
        (120.0, 12.0), (120.5, 12.0), (121.0, 12.5), (121.0, 13.5),
        (120.5, 13.5), (120.0, 13.0),
    ],
    # Panay and Negros.
    [
        (121.5, 10.0), (122.0, 10.0), (122.8, 10.7), (122.8, 11.2),
        (122.2, 11.5), (121.8, 11.0),
    ],
    [
        (122.5, 9.0), (123.0, 9.0), (123.2, 10.0), (122.8, 10.5),
        (122.3, 10.0),
    ],
    # Cebu and Bohol.
    [
        (123.3, 9.5), (123.7, 9.7), (124.0, 10.2), (123.7, 10.5),
        (123.4, 10.2),
    ],
    [
        (123.5, 9.5), (124.5, 9.5), (124.7, 9.8), (124.0, 10.0),
    ],
    # Leyte and Samar.
    [
        (124.5, 10.0), (125.0, 10.2), (125.5, 11.0), (125.4, 11.8),
        (124.8, 11.5), (124.5, 10.8),
    ],
    [
        (125.0, 11.0), (125.5, 11.2), (126.3, 12.0), (126.3, 12.7),
        (125.8, 12.5), (125.3, 11.8),
    ],
    # Mindanao.
    [
        (122.0, 7.0), (122.5, 6.0), (123.5, 5.5), (124.5, 6.0),
        (125.5, 6.5), (126.5, 7.5), (126.5, 8.5), (125.8, 9.0),
        (125.0, 9.5), (124.0, 9.0), (123.0, 8.5), (122.0, 8.0),
    ],
    # Luzon and nearby large islands.
    [
        (119.5, 15.0), (120.0, 15.5), (120.0, 16.5), (120.5, 17.5),
        (121.0, 18.5), (121.5, 19.5), (122.0, 18.8), (122.2, 17.5),
        (121.8, 16.5), (121.5, 15.5), (121.0, 14.8), (120.5, 14.5),
    ],
    [
        (122.0, 17.0), (122.5, 17.2), (123.0, 18.0), (122.8, 18.5),
        (122.2, 18.0),
    ],
    # Ryukyu island chain within the eastern edge of the frame.
    [
        (123.0, 24.0), (124.0, 24.0), (124.5, 23.0), (123.8, 23.0),
    ],
]

# Small islands, cays and reef/atoll sites represented by compact ellipses.
# Each tuple is (longitude, latitude, east-west radius, north-south radius).
SMALL_LAND = [
    (120.3, 20.7, 0.10, 0.10),  # Lanyu
    (121.5, 22.0, 0.08, 0.08),
    (122.0, 24.0, 0.13, 0.12),
    (117.2, 15.8, 0.08, 0.08),  # Paracel area
    (111.7, 16.5, 0.10, 0.10),
    (112.3, 16.5, 0.08, 0.08),
    (114.3, 11.0, 0.08, 0.08),  # Spratly-area reefs and islands
    (114.5, 10.4, 0.10, 0.08),
    (115.0, 10.0, 0.08, 0.08),
    (115.8, 9.8, 0.10, 0.08),
    (116.0, 9.0, 0.08, 0.08),
    (116.5, 10.0, 0.08, 0.08),
    (117.0, 8.5, 0.08, 0.08),
    (117.8, 8.8, 0.08, 0.08),
    (118.0, 8.0, 0.08, 0.08),
    (118.5, 7.5, 0.08, 0.08),
    (119.5, 7.0, 0.10, 0.08),
    (120.0, 6.0, 0.08, 0.08),
    (120.5, 5.0, 0.08, 0.08),
    (121.0, 4.0, 0.08, 0.08),
    (121.5, 3.0, 0.08, 0.08),
    (122.0, 2.0, 0.08, 0.08),
    (123.0, 1.5, 0.08, 0.08),
    (123.5, 1.8, 0.08, 0.08),
    (124.0, 2.0, 0.08, 0.08),
    (124.5, 2.5, 0.08, 0.08),
    (125.0, 3.0, 0.08, 0.08),
    (126.0, 4.0, 0.08, 0.08),
]


def points_in_polygon(x, y, polygon):
    """Vectorized ray-casting test; polygon edges count as land."""
    inside = np.zeros(x.shape, dtype=bool)
    on_edge = np.zeros(x.shape, dtype=bool)
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        dx, dy = x2 - x1, y2 - y1
        cross = (x - x1) * dy - (y - y1) * dx
        on_edge |= (
            (np.abs(cross) < 1e-10)
            & (x >= min(x1, x2) - 1e-10)
            & (x <= max(x1, x2) + 1e-10)
            & (y >= min(y1, y2) - 1e-10)
            & (y <= max(y1, y2) + 1e-10)
        )
        if dy != 0:
            crosses = ((y1 > y) != (y2 > y)) & (
                x < (dx * (y - y1) / dy + x1)
            )
            inside ^= crosses
    return inside | on_edge


def build_matrix():
    latitudes = 23.0 - 0.2 * np.arange(111)
    longitudes = 99.0 + 0.2 * np.arange(141)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    land = np.zeros((111, 141), dtype=bool)

    for polygon in LAND_POLYGONS:
        land |= points_in_polygon(lon_grid, lat_grid, polygon)

    for lon, lat, rx, ry in SMALL_LAND:
        land |= ((lon_grid - lon) / rx) ** 2 + ((lat_grid - lat) / ry) ** 2 <= 1.0

    return land.astype(np.uint8)


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()

    assert matrix.shape == (111, 141)
    assert matrix.size == 15651
    assert np.isin(matrix, (0, 1)).all()

    with (output_dir / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # 1 maps to white land; 0 maps to black water.
    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(output_dir / "candidate.png")
    enlarged = image.resize((1128, 888), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")

    assert image.size == (141, 111)
    assert enlarged.size == (1128, 888)
    assert set(image.getdata()) <= {(0, 0, 0), (255, 255, 255)}


if __name__ == "__main__":
    main()
