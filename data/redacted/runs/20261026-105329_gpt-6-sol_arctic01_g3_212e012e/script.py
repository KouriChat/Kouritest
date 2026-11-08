from pathlib import Path

import numpy as np
from PIL import Image


# Approximate island outlines as (west longitude, north latitude) vertices.
LAND = [
    # Ellesmere Island, with the eastern edge clipped by the frame.
    [
        (81.0, 75.6), (81.5, 75.8), (82.2, 76.0), (83.0, 76.2),
        (84.1, 76.5), (84.8, 77.0), (85.0, 77.8), (84.7, 78.5),
        (84.3, 79.3), (84.5, 80.0), (83.8, 80.7), (83.3, 81.5),
        (82.7, 82.2), (81.8, 82.6), (81.0, 82.5), (81.0, 75.6),
    ],
    # Axel Heiberg Island.
    [
        (88.8, 78.0), (88.0, 78.3), (87.0, 78.7), (86.5, 79.4),
        (86.8, 80.3), (87.4, 81.0), (88.3, 81.5), (89.2, 81.3),
        (90.0, 80.8), (90.4, 80.0), (90.2, 79.1), (89.8, 78.4),
        (88.8, 78.0),
    ],
    # Devon Island.
    [
        (81.0, 74.5), (82.0, 74.6), (83.0, 74.8), (84.0, 75.0),
        (85.0, 75.2), (86.0, 75.4), (87.0, 75.6), (88.2, 75.9),
        (89.0, 76.3), (89.6, 76.8), (89.8, 77.4), (89.3, 77.9),
        (88.5, 78.2), (87.5, 78.1), (86.5, 77.9), (85.5, 77.8),
        (84.6, 77.6), (83.5, 77.3), (82.5, 77.0), (81.6, 76.6),
        (81.0, 76.0), (81.0, 74.5),
    ],
    # Bathurst Island.
    [
        (96.0, 75.0), (97.0, 75.1), (98.0, 75.3), (99.0, 75.7),
        (99.7, 76.2), (99.5, 76.8), (98.8, 77.1), (97.8, 77.0),
        (97.0, 76.7), (96.3, 76.2), (96.0, 75.7), (96.0, 75.0),
    ],
    # Cornwallis Island.
    [
        (94.4, 74.9), (95.0, 75.0), (95.8, 75.2), (96.2, 75.6),
        (96.1, 76.1), (95.7, 76.4), (95.0, 76.4), (94.5, 76.1),
        (94.3, 75.5), (94.4, 74.9),
    ],
]
# Prince of Wales Island: northern portion in the requested latitude range.
LAND.append([
    (103.0, 72.8), (102.0, 73.0), (101.0, 73.3), (100.0, 73.6),
    (99.0, 73.9), (98.0, 74.0), (97.0, 73.8), (96.2, 73.4),
    (96.0, 73.0), (97.0, 72.8), (99.0, 72.7), (101.0, 72.6),
    (103.0, 72.8),
])
# Somerset Island.
LAND.append([
    (95.7, 73.0), (95.4, 73.2), (95.0, 73.8), (94.5, 74.4),
    (93.8, 74.7), (93.0, 74.6), (92.4, 74.2), (92.0, 73.6),
    (92.2, 73.0), (92.6, 72.5), (93.2, 72.1), (94.2, 71.9),
    (95.0, 71.9), (95.7, 72.0), (95.7, 73.0),
])
# Boothia Peninsula.
LAND.append([
    (94.6, 73.0), (94.5, 72.5), (94.2, 71.9), (93.8, 72.5),
    (93.4, 73.0), (92.8, 73.4), (92.0, 73.6), (91.2, 73.4),
    (90.7, 73.0), (90.4, 72.4), (90.5, 71.8), (90.8, 71.2),
    (91.2, 70.7), (91.8, 70.2), (92.7, 69.9), (93.7, 69.9),
    (94.6, 70.0), (94.6, 73.0),
])
# Northern Victoria Island, clipped at the western frame boundary.
LAND.append([
    (105.0, 73.0), (104.0, 72.8), (103.0, 72.9), (102.0, 73.0),
    (101.0, 73.1), (100.0, 73.0), (99.0, 72.8), (98.0, 72.6),
    (97.0, 72.4), (96.0, 72.1), (95.0, 71.8), (94.0, 71.5),
    (93.0, 71.2), (92.0, 70.8), (91.0, 70.3), (90.0, 69.8),
    (89.0, 69.3), (88.0, 68.8), (87.0, 68.2), (86.0, 67.5),
    (105.0, 67.5), (105.0, 73.0),
])
# King William Island's northern tip reaches the southern edge of the frame.
LAND.append([
    (99.7, 73.0), (99.5, 73.2), (99.0, 73.25), (98.5, 73.2),
    (97.8, 73.1), (97.3, 73.0), (97.5, 73.0), (98.2, 73.0),
    (99.0, 73.0), (99.7, 73.0),
])


def points_in_polygon(lon_w, lat, polygon):
    """Vectorized ray-crossing test in west longitude and north latitude."""
    inside = np.zeros(lon_w.shape, dtype=bool)
    x0, y0 = polygon[-1]
    for x1, y1 in polygon:
        crosses = (y0 > lat) != (y1 > lat)
        if y1 != y0:
            edge_x = (x1 - x0) * (lat - y0) / (y1 - y0) + x0
            inside ^= crosses & (lon_w < edge_x)
        x0, y0 = x1, y1
    return inside


def generate():
    latitudes = 79.0 - np.arange(61) * 0.1
    longitudes_w = 105.0 - np.arange(241) * 0.1
    lon_grid, lat_grid = np.meshgrid(longitudes_w, latitudes)
    land = np.zeros((61, 241), dtype=bool)
    for polygon in LAND:
        land |= points_in_polygon(lon_grid, lat_grid, polygon)

    matrix = land.astype(np.uint8)
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "candidate.csv").open(
        "w", encoding="ascii", newline=""
    ) as csv_file:
        for row in matrix:
            csv_file.write(",".join(map(str, row.tolist())) + "\n")

    image = Image.fromarray(matrix * 255, mode="L")
    image.save(out_dir / "candidate.png")
    image.resize((1928, 488), Image.Resampling.NEAREST).save(
        out_dir / "candidate_8x.png"
    )
    assert matrix.shape == (61, 241)
    assert int(matrix.size) == 14701
    assert set(np.unique(matrix)).issubset({0, 1})
    assert image.size == (241, 61)
    assert set(image.getdata()).issubset({0, 255})
    enlarged = Image.open(out_dir / "candidate_8x.png")
    assert enlarged.size == (1928, 488)
    assert enlarged.resize((241, 61), Image.Resampling.NEAREST).tobytes() == (
        image.tobytes()
    )


if __name__ == "__main__":
    generate()
