from pathlib import Path

import numpy as np
from PIL import Image

# Approximate hand-drawn island outlines in (west longitude, north latitude).
# The outline set captures the major islands in the requested Northwest Passage
# region; all unclassified points are treated as water.
ISLANDS = [
    # Ellesmere Island
    [(-91.5, 77.0), (-90.5, 77.5), (-89.5, 77.7), (-88.8, 78.2),
     (-87.5, 78.4), (-86.0, 78.1), (-84.8, 78.4), (-83.0, 78.5),
     (-81.0, 78.3), (-78.0, 78.5), (-75.0, 79.0), (-72.0, 79.8),
     (-70.0, 81.0), (-69.0, 83.0), (-95.0, 83.0), (-95.0, 80.0),
     (-93.5, 78.0)],
    # Axel Heiberg Island
    [(-93.4, 78.0), (-92.8, 78.6), (-91.8, 79.4), (-90.6, 80.0),
     (-89.2, 79.8), (-88.0, 79.1), (-87.5, 78.2), (-88.3, 77.4),
     (-89.5, 77.2), (-91.0, 77.4), (-92.3, 77.3)],
    # Devon Island
    [(-90.2, 74.7), (-89.5, 75.1), (-88.8, 75.5), (-87.6, 75.7),
     (-86.8, 76.3), (-85.4, 76.6), (-83.8, 76.4), (-82.0, 76.0),
     (-81.0, 75.6), (-81.0, 74.8), (-82.0, 74.5), (-83.0, 74.6),
     (-84.2, 74.3), (-85.0, 74.5), (-86.5, 74.2), (-88.0, 74.3),
     (-89.2, 74.5)],
    # Bathurst Island
    [(-99.4, 75.0), (-98.8, 75.7), (-97.8, 76.2), (-96.8, 76.8),
     (-95.7, 77.3), (-94.8, 77.1), (-94.3, 76.4), (-94.6, 75.7),
     (-95.3, 75.1), (-96.5, 74.9), (-97.5, 75.0), (-98.5, 74.7)],
    # Eastern Melville Island
    [(-105.0, 75.0), (-104.0, 75.4), (-103.0, 75.5), (-102.5, 75.9),
     (-101.5, 76.0), (-100.8, 76.4), (-100.0, 76.3), (-99.5, 75.8),
     (-100.2, 75.4), (-101.0, 75.0), (-102.0, 74.8), (-103.0, 74.9),
     (-104.0, 74.7)],
    # Cornwallis Island
    [(-96.3, 74.9), (-95.8, 75.2), (-95.0, 75.4), (-94.3, 75.1),
     (-94.0, 74.7), (-94.5, 74.5), (-95.4, 74.6), (-96.0, 74.5)],
    # Prince of Wales Island
    [(-99.5, 72.8), (-99.0, 73.4), (-98.4, 74.1), (-97.7, 74.6),
     (-96.8, 74.8), (-96.0, 74.3), (-96.2, 73.6), (-96.7, 73.0),
     (-97.5, 72.6), (-98.5, 72.5)],
    # Somerset Island
    [(-95.2, 72.0), (-94.5, 72.5), (-94.1, 73.2), (-93.2, 73.8),
     (-92.0, 74.1), (-90.5, 74.5), (-88.8, 74.8), (-87.2, 74.7),
     (-85.8, 74.5), (-84.3, 74.2), (-83.0, 73.8), (-82.0, 73.1),
     (-81.0, 72.8), (-81.0, 72.0), (-83.0, 72.0), (-85.0, 72.2),
     (-87.0, 72.0), (-89.0, 72.1), (-91.0, 71.9), (-93.0, 71.8)],

    # Eastern Victoria Island
    [(-105.0, 72.9), (-104.0, 73.3), (-103.0, 73.7), (-102.2, 74.2),
     (-101.5, 74.7), (-100.8, 75.0), (-100.0, 74.8), (-99.8, 74.2),
     (-100.3, 73.7), (-101.0, 73.3), (-102.0, 73.0), (-103.0, 72.8),
     (-104.0, 72.7)],
    # Small islands in the central archipelago
    [(-101.4, 76.8), (-100.8, 77.1), (-100.1, 77.0), (-99.8, 76.6),
     (-100.3, 76.3), (-101.0, 76.4)],
    [(-97.4, 77.1), (-96.8, 77.4), (-96.2, 77.2), (-96.0, 76.8),
     (-96.5, 76.6), (-97.1, 76.7)],
    [(-93.6, 76.2), (-93.1, 76.5), (-92.7, 76.3), (-92.8, 75.9),
     (-93.3, 75.8)],
]


def polygon_contains(lon_grid, lat_grid, polygon):
    """Classify grid points with the ray-crossing point-in-polygon rule."""
    inside = np.zeros(lon_grid.shape, dtype=bool)
    vertices = polygon + [polygon[0]]
    for (x1, y1), (x2, y2) in zip(vertices, vertices[1:]):
        crosses = (y1 > lat_grid) != (y2 > lat_grid)
        if y1 != y2:
            x_cross = (x2 - x1) * (lat_grid - y1) / (y2 - y1) + x1
            inside ^= crosses & (lon_grid < x_cross)
    return inside


def main():
    latitudes = np.round(np.arange(79.0, 72.9, -0.1), 1)
    longitudes = np.round(np.arange(-105.0, -80.9, 0.1), 1)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    land = np.zeros((len(latitudes), len(longitudes)), dtype=bool)
    for polygon in ISLANDS:
        land |= polygon_contains(lon_grid, lat_grid, polygon)
    matrix = land.astype(np.uint8)
    if matrix.shape != (61, 241):
        raise RuntimeError(f"Unexpected matrix shape: {matrix.shape}")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", encoding="ascii", newline="") as stream:
        for row in matrix:
            stream.write(",".join(str(int(value)) for value in row) + "\n")

    image = Image.fromarray(np.where(land, 255, 0).astype(np.uint8), mode="L")
    image.save(output_dir / "candidate.png")
    enlarged = image.resize((1928, 488), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")

    rows = csv_path.read_text(encoding="ascii").splitlines()
    assert len(rows) == 61 and all(len(row.split(",")) == 241 for row in rows)
    assert set(",".join(rows).split(",")) <= {"0", "1"}
    assert image.size == (241, 61) and enlarged.size == (1928, 488)
    assert set(image.getdata()) <= {0, 255}
    assert set(enlarged.getdata()) <= {0, 255}


if __name__ == "__main__":
    main()
