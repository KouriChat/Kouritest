from pathlib import Path

import numpy as np
from PIL import Image

LATITUDES = np.round(np.arange(23.0, 0.999999, -0.2), 10)
LONGITUDES = np.round(np.arange(99.0, 127.000001, 0.2), 10)
LON, LAT = np.meshgrid(LONGITUDES, LATITUDES)

# Approximate hand-drawn coastlines, represented as (longitude, latitude).
LAND_POLYGONS = [
    # Southern China and mainland Southeast Asia.
    [
        (97.0, 24.0), (111.0, 24.0), (110.5, 22.5), (110.0, 21.5),
        (109.5, 20.8), (108.4, 20.0), (107.5, 19.0), (106.8, 18.0),
        (106.4, 17.0), (106.0, 16.0), (105.8, 15.0), (106.1, 14.0),
        (105.8, 13.0), (105.2, 12.0), (104.8, 11.2), (104.0, 10.5),
        (103.5, 9.5), (103.7, 8.5), (103.2, 7.5), (102.7, 6.5),
        (101.8, 5.5), (101.0, 4.5), (100.0, 3.2), (99.0, 1.5),
        (98.0, 1.0), (97.0, 1.0),
    ],
    # Hainan
    [
        (108.6, 20.1), (109.2, 20.0), (110.0, 19.5), (110.5, 19.0),
        (110.2, 18.4), (109.6, 18.2), (109.0, 18.3), (108.6, 18.8),
    ],
    # Taiwan
    [
        (120.0, 25.3), (121.0, 25.2), (121.7, 24.7), (121.7, 23.8),
        (121.2, 22.8), (120.8, 22.0), (120.1, 22.0), (119.8, 23.0),
        (120.0, 24.0),
    ],
    # Sumatra
    [
        (96.5, 5.8), (98.0, 5.3), (99.5, 4.3), (101.0, 2.5),
        (102.5, 0.5), (103.5, -1.5), (103.0, -3.5), (101.5, -5.5),
        (100.0, -5.8), (98.8, -4.0), (98.0, -2.0), (97.2, 0.0),
    ],
    # Malay Peninsula
    [
        (100.0, 7.0), (100.5, 6.0), (101.0, 5.0), (101.5, 4.0),
        (102.0, 3.0), (103.0, 2.0), (104.0, 1.0), (104.0, 0.5),
        (103.0, 0.7), (102.0, 1.5), (101.0, 3.0), (100.5, 5.0),
    ],
    # Borneo
    [
        (109.5, 7.0), (110.0, 6.0), (110.0, 4.5), (109.7, 3.0),
        (110.0, 1.5), (111.0, 1.0), (112.0, 1.0), (113.0, 1.5),
        (114.0, 1.8), (115.0, 2.0), (116.0, 2.5), (117.0, 3.5),
        (118.0, 4.5), (117.5, 6.0), (116.5, 7.5), (115.5, 7.8),
        (114.0, 7.0), (112.5, 6.7), (111.0, 6.2), (110.0, 6.5),
    ],
    # Java
    [
        (105.0, -5.8), (106.5, -6.0), (108.0, -6.4), (109.5, -7.0),
        (111.0, -7.5), (113.0, -8.0), (114.5, -8.3), (114.0, -8.8),
        (112.0, -8.6), (110.0, -8.2), (108.0, -7.8), (106.0, -7.1),
    ],
    # Palawan
    [
        (117.0, 11.8), (117.5, 11.0), (118.0, 10.0), (118.5, 9.0),
        (119.0, 8.4), (118.5, 8.1), (117.8, 9.0), (117.2, 10.5),
    ],
    # Luzon
    [
        (119.8, 18.6), (120.5, 18.4), (121.2, 17.8), (121.8, 17.0),
        (122.0, 16.0), (121.7, 15.0), (121.2, 14.2), (120.5, 13.6),
        (119.8, 14.0), (119.8, 15.0), (120.2, 16.0), (119.7, 17.2),
    ],
    # Mindoro and Panay
    [
        (120.2, 13.5), (121.0, 13.3), (121.3, 12.8), (121.0, 12.2),
        (120.4, 12.2), (120.0, 12.8),
    ],
    [
        (121.8, 11.8), (122.5, 11.6), (122.8, 11.0), (122.2, 10.8),
        (121.8, 11.1),
    ],
    # Central Philippine islands.
    [
        (122.3, 10.8), (123.0, 10.5), (123.1, 9.5), (122.6, 9.0),
        (122.2, 9.5),
    ],
    [
        (123.0, 10.8), (123.5, 10.5), (123.7, 9.7), (123.3, 9.3),
        (123.0, 9.8),
    ],
    [
        (124.5, 12.5), (125.2, 12.3), (125.5, 11.8), (125.0, 11.4),
        (124.6, 11.5),
    ],
    # Mindanao
    [
        (124.0, 9.8), (124.8, 9.5), (125.5, 8.5), (126.0, 7.5),
        (125.6, 6.7), (125.0, 6.0), (124.2, 6.0), (123.5, 6.8),
        (122.8, 7.5), (123.0, 8.4), (123.6, 9.0),
    ],
]


def polygon_mask(vertices):
    """Classify grid points by the even-odd polygon rule."""
    inside = np.zeros(LON.shape, dtype=bool)
    j = len(vertices) - 1
    for i, (xi, yi) in enumerate(vertices):
        xj, yj = vertices[j]
        crosses = ((yi > LAT) != (yj > LAT)) & (
            LON < (xj - xi) * (LAT - yi) / ((yj - yi) or 1e-30) + xi
        )
        inside ^= crosses
        j = i
    return inside


def build_matrix():
    matrix = np.zeros(LON.shape, dtype=np.uint8)
    for polygon in LAND_POLYGONS:
        matrix[polygon_mask(polygon)] = 1

    # Small islands and reef/islet groups represented as approximate ellipses.
    islands = [
        (105.7, 3.0, 0.25, 0.18),
        (106.0, 2.5, 0.18, 0.14),
        (106.5, 3.2, 0.16, 0.12),
        (105.8, 1.5, 0.15, 0.12),
        (104.5, 2.5, 0.13, 0.10),
        (103.0, 1.3, 0.12, 0.10),
        (100.0, 7.8, 0.12, 0.10),
        (100.5, 7.5, 0.10, 0.09),
        (102.5, 6.0, 0.09, 0.08),
        (104.0, 10.0, 0.12, 0.08),
        (108.0, 10.0, 0.15, 0.10),
        (112.0, 10.0, 0.10, 0.08),
    ]
    for cx, cy, rx, ry in islands:
        matrix[((LON - cx) / rx) ** 2 + ((LAT - cy) / ry) ** 2 <= 1] = 1
    return matrix


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()
    if matrix.shape != (111, 141):
        raise RuntimeError(f"Unexpected matrix shape: {matrix.shape}")

    with (output_dir / "candidate.csv").open(
        "w", encoding="ascii", newline=""
    ) as stream:
        for row in matrix:
            stream.write(",".join(str(int(value)) for value in row) + "\n")

    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(output_dir / "candidate.png")
    enlarged = image.resize(
        (1128, 888), resample=Image.Resampling.NEAREST
    )
    enlarged.save(output_dir / "candidate_8x.png")

    rows = (output_dir / "candidate.csv").read_text(
        encoding="ascii"
    ).splitlines()
    if len(rows) != 111 or any(len(row.split(",")) != 141 for row in rows):
        raise RuntimeError("CSV dimensions are invalid")
    if any(v not in {"0", "1"} for row in rows for v in row.split(",")):
        raise RuntimeError("CSV must contain only binary values")
    if image.size != (141, 111) or enlarged.size != (1128, 888):
        raise RuntimeError("PNG dimensions are invalid")
    if set(image.getdata()) != {(0, 0, 0), (255, 255, 255)}:
        raise RuntimeError("Original image is not black and white")
    print(f"Validated {matrix.size} values in {output_dir}")


if __name__ == "__main__":
    main()
