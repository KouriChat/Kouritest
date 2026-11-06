from pathlib import Path
import math

import numpy as np
from PIL import Image


# WGS84 sample points: north to south, west to east.
LATITUDES = np.round(np.arange(23.0, 1.0 - 1e-9, -0.2), 10)
LONGITUDES = np.round(np.arange(99.0, 127.0 + 1e-9, 0.2), 10)
LON_GRID, LAT_GRID = np.meshgrid(LONGITUDES, LATITUDES)

# Hand-estimated coastline outlines, ordered around each land polygon.
# Coordinates are (longitude, latitude); only points inside the requested
# sampling box are evaluated.
POLYGONS = [
    # Mainland China, Indochina, and the northern Gulf of Thailand.
    [
        (98.0, 25.0), (130.0, 25.0),
        (122.4, 24.8), (121.5, 24.5), (120.8, 24.0),
        (120.2, 23.5), (119.7, 23.0), (119.3, 22.5),
        (118.7, 22.0), (117.8, 21.7), (116.8, 21.5),
        (115.5, 21.4), (114.7, 21.5), (113.8, 21.6),
        (112.8, 21.5), (111.8, 21.4), (110.8, 21.2),
        (109.8, 20.8), (109.2, 20.4), (108.8, 20.0),
        (108.4, 19.6), (107.9, 19.2), (107.2, 18.8),
        (106.8, 18.3), (106.5, 17.7), (106.6, 17.0),
        (107.0, 16.3), (107.8, 15.7), (108.5, 15.2),
        (109.0, 14.6), (109.3, 13.8), (109.2, 13.0),
        (108.9, 12.2), (108.5, 11.5), (108.0, 10.9),
        (107.4, 10.4), (106.8, 9.8), (106.2, 9.3),
        (105.7, 8.9), (105.2, 8.7), (104.7, 8.6),
        (104.2, 9.0), (103.8, 9.6), (103.5, 10.3),
        (103.2, 11.1), (102.8, 11.9), (102.3, 12.7),
        (101.7, 13.2), (101.0, 13.5), (100.4, 13.6),
        (99.8, 13.4), (99.2, 13.1), (98.7, 12.7),
        (98.0, 12.2), (98.0, 25.0),
    ],
    # Malay Peninsula, south of the mainland.
    [
        (99.5, 6.5), (100.2, 6.4), (101.0, 6.2), (101.8, 5.8),
        (102.5, 5.2), (103.0, 4.5), (103.4, 3.8), (103.7, 3.0),
        (104.0, 2.2), (104.3, 1.4), (103.8, 1.2), (103.3, 1.3),
        (102.9, 1.7), (102.5, 2.3), (102.0, 3.0), (101.5, 3.8),
        (100.9, 4.5), (100.3, 5.2), (99.8, 5.8), (99.5, 6.5),
    ],
    # Sumatra, including the portion clipped by the southern frame edge.
    [
        (105.9, 5.8), (105.2, 5.5), (104.6, 5.0), (104.1, 4.4),
        (103.6, 3.7), (103.1, 2.9), (102.7, 2.1), (102.2, 1.3),
        (101.7, 0.7), (101.0, 0.0), (99.0, 0.0), (99.0, 1.0),
        (99.4, 1.5), (99.7, 2.2), (100.0, 3.0), (100.3, 3.8),
        (100.5, 4.6), (100.7, 5.3), (101.0, 5.8), (101.5, 5.9),
        (102.2, 5.7), (103.0, 5.4), (103.8, 5.2), (104.6, 5.2),
        (105.3, 5.4), (105.9, 5.8),
    ],
    # Borneo.
    [
        (108.0, 7.0), (109.1, 7.0), (110.1, 6.8), (111.1, 6.4),
        (112.0, 5.9), (113.0, 5.6), (114.1, 5.1), (115.0, 4.5),
        (116.0, 3.8), (117.0, 3.2), (117.5, 2.5), (117.2, 1.6),
        (116.5, 0.8), (115.6, 0.4), (114.8, 0.0), (109.0, 0.0),
        (108.6, 0.8), (108.2, 1.8), (108.0, 3.0), (108.0, 4.2),
        (108.0, 5.5), (108.0, 7.0),
    ],
    # Taiwan.
    [
        (120.0, 25.4), (120.8, 25.2), (121.5, 24.5), (121.7, 23.6),
        (121.3, 22.7), (120.8, 22.0), (120.2, 21.9), (120.0, 22.7),
        (120.0, 24.0), (120.0, 25.4),
    ],
    # Hainan.
    [
        (108.6, 20.2), (109.3, 20.1), (110.0, 19.8), (110.5, 19.4),
        (110.6, 18.8), (110.2, 18.3), (109.6, 18.2), (109.0, 18.4),
        (108.6, 19.0), (108.5, 19.6), (108.6, 20.2),
    ],
    # Luzon.
    [
        (119.8, 18.6), (120.5, 18.5), (121.1, 18.2), (121.6, 17.6),
        (121.8, 16.8), (121.3, 16.1), (120.8, 15.6), (120.5, 15.0),
        (120.2, 14.5), (120.0, 13.7), (120.4, 13.0), (121.0, 13.1),
        (121.5, 13.8), (122.0, 14.7), (122.2, 15.7), (122.0, 16.8),
        (121.6, 17.8), (121.0, 18.3), (120.4, 18.6), (119.8, 18.6),
    ],
    # Mindoro.
    [
        (120.4, 13.5), (121.0, 13.4), (121.3, 13.0), (121.2, 12.4),
        (120.8, 12.0), (120.4, 12.2), (120.2, 12.8), (120.4, 13.5),
    ],
    # Palawan.
    [
        (117.0, 11.8), (117.7, 11.5), (118.0, 10.8), (118.0, 10.0),
        (117.7, 9.2), (117.5, 8.4), (117.2, 7.6), (117.0, 6.8),
        (116.8, 7.6), (116.8, 8.5), (116.9, 9.5), (116.8, 10.5),
        (116.7, 11.2), (117.0, 11.8),
    ],
    # Panay.
    [
        (121.7, 11.9), (122.5, 11.8), (122.9, 11.4), (122.7, 10.8),
        (122.1, 10.5), (121.7, 10.8), (121.5, 11.3), (121.7, 11.9),
    ],
    # Negros.
    [
        (122.7, 10.9), (123.2, 10.8), (123.4, 10.2), (123.2, 9.5),
        (122.9, 9.2), (122.6, 9.7), (122.7, 10.9),
    ],
    # Cebu.
    [
        (123.3, 11.3), (123.7, 11.1), (123.8, 10.3),
        (123.6, 9.7), (123.3, 10.2), (123.3, 11.3),
    ],
    # Bohol.
    [
        (123.7, 10.1), (124.5, 10.0), (124.6, 9.7),
        (123.8, 9.6), (123.7, 10.1),
    ],
    # Leyte.
    [
        (124.2, 11.1), (124.8, 10.9), (125.0, 10.3),
        (124.8, 9.7), (124.3, 9.8), (124.1, 10.4), (124.2, 11.1),
    ],
    # Samar.
    [
        (124.4, 12.0), (125.2, 11.8), (125.5, 11.2),
        (125.3, 10.6), (124.8, 10.7), (124.4, 11.2), (124.4, 12.0),
    ],
    # Mindanao.
    [
        (123.5, 9.8), (124.2, 9.4), (125.0, 9.0), (125.8, 8.7),
        (126.5, 8.4), (126.7, 7.7), (126.4, 7.0), (125.8, 6.5),
        (125.2, 6.0), (124.5, 5.5), (123.8, 5.4), (123.3, 5.9),
        (123.1, 6.7), (123.2, 7.5), (123.1, 8.3), (123.2, 9.0),
        (123.5, 9.8),
    ],
    # Bangka and Belitung, off eastern Sumatra.
    [
        (105.0, 3.6), (105.7, 3.5), (106.0, 2.8), (105.8, 2.0),
        (105.3, 1.8), (104.9, 2.4), (105.0, 3.6),
    ],
    [
        (107.4, 3.4), (108.0, 3.2), (108.3, 2.7),
        (108.0, 2.1), (107.5, 2.2), (107.2, 2.8), (107.4, 3.4),
    ],
    # Natuna and Anambas island groups (coarse island-sized outlines).
    [
        (108.0, 4.2), (108.5, 4.1), (108.7, 3.8),
        (108.4, 3.5), (108.0, 3.7), (108.0, 4.2),
    ],
    [
        (105.5, 3.0), (106.2, 2.9), (106.4, 2.5),
        (106.0, 2.2), (105.6, 2.4), (105.5, 3.0),
    ],
    # Sulu Archipelago and nearby small Philippine islands.
    [
        (120.0, 6.3), (121.0, 6.2), (121.2, 5.9),
        (120.3, 5.7), (120.0, 6.0), (120.0, 6.3),
    ],
]

# Elliptical island outlines: (center_lon, center_lat, radius_lon,
# radius_lat, rotation_degrees). These are deliberately approximate.
ISLAND_ELLIPSES = [
    # Spratly reef/island locations and western Philippine Sea islands.
    (112.3, 10.3, 0.11, 0.08, 0),
    (114.3, 10.0, 0.12, 0.08, 15),
    (114.6, 9.9, 0.10, 0.07, 0),
    (115.8, 10.4, 0.10, 0.07, 0),
    (116.7, 10.0, 0.10, 0.06, 0),
    (117.0, 8.9, 0.09, 0.06, 0),
    (109.5, 7.0, 0.12, 0.08, 0),
    (119.8, 10.2, 0.08, 0.07, 0),
    (120.0, 9.0, 0.06, 0.05, 0),
    (122.0, 7.0, 0.08, 0.06, 0),
    (124.0, 6.0, 0.08, 0.06, 0),
    # Small islands near the Gulf of Thailand and Singapore.
    (102.5, 8.0, 0.08, 0.05, 0),
    (103.8, 1.3, 0.08, 0.05, 0),
]


def point_in_polygon(x, y, vertices):
    """Return a Boolean grid for points inside a polygon."""
    inside = np.zeros(x.shape, dtype=bool)
    j = len(vertices) - 1

    for i in range(len(vertices)):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        crosses = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-30) + xi
        )
        inside ^= crosses
        j = i

    return inside


def ellipse_polygon(lon, lat, rx, ry, angle_degrees, steps=48):
    angle = math.radians(angle_degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    vertices = []

    for i in range(steps):
        t = 2.0 * math.pi * i / steps
        ex = rx * math.cos(t)
        ey = ry * math.sin(t)
        vertices.append((
            lon + ex * cos_a - ey * sin_a,
            lat + ex * sin_a + ey * cos_a,
        ))

    return vertices


def build_matrix():
    land = np.zeros(LON_GRID.shape, dtype=bool)

    for polygon in POLYGONS:
        land |= point_in_polygon(LON_GRID, LAT_GRID, polygon)

    for lon, lat, rx, ry, angle in ISLAND_ELLIPSES:
        land |= point_in_polygon(
            LON_GRID,
            LAT_GRID,
            ellipse_polygon(lon, lat, rx, ry, angle),
        )

    return land.astype(np.uint8)


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = build_matrix()
    if matrix.shape != (111, 141):
        raise RuntimeError(f"Unexpected matrix shape: {matrix.shape}")

    np.savetxt(
        output_dir / "candidate.csv",
        matrix,
        fmt="%d",
        delimiter=",",
    )

    # Grayscale PNG with exactly two values: 0 (black water), 255 (white land).
    pixels = matrix * 255
    image = Image.fromarray(pixels, mode="L")
    image.save(output_dir / "candidate.png")

    enlarged = image.resize(
        (141 * 8, 111 * 8),
        resample=Image.Resampling.NEAREST,
    )
    enlarged.save(output_dir / "candidate_8x.png")

    # Validate the saved deliverables.
    rows = (output_dir / "candidate.csv").read_text(encoding="ascii").splitlines()
    if len(rows) != 111:
        raise RuntimeError("CSV does not contain 111 rows")

    for row in rows:
        values = row.split(",")
        if len(values) != 141 or any(value not in {"0", "1"} for value in values):
            raise RuntimeError("CSV row is malformed")

    if matrix.size != 15651 or not np.isin(matrix, (0, 1)).all():
        raise RuntimeError("Matrix contains an invalid value")

    if image.size != (141, 111) or set(image.getdata()) - {0, 255}:
        raise RuntimeError("Original PNG validation failed")

    if enlarged.size != (1128, 888) or set(enlarged.getdata()) - {0, 255}:
        raise RuntimeError("Enlarged PNG validation failed")

    print("Generated and validated:")
    print("  outputs/candidate.csv       111 x 141")
    print("  outputs/candidate.png       141 x 111")
    print("  outputs/candidate_8x.png    1128 x 888")


if __name__ == "__main__":
    main()
