from pathlib import Path
import csv

import numpy as np
from PIL import Image

# Approximate hand-authored outlines, coordinates are (longitude, latitude).
ISLANDS = [
    [(-90.8, 79.0), (-90.4, 78.0), (-89.4, 77.2), (-88.7, 76.4),
     (-87.0, 75.7), (-85.7, 75.5), (-84.9, 76.0), (-83.8, 76.0),
     (-82.5, 76.4), (-81.0, 77.0), (-81.0, 79.0)],
    [(-94.8, 78.9), (-94.2, 78.0), (-93.0, 77.5), (-91.7, 77.2),
     (-90.6, 77.5), (-89.7, 78.2), (-90.0, 79.0)],
    [(-90.3, 76.8), (-89.7, 75.9), (-88.9, 75.2), (-87.2, 74.8),
     (-85.4, 74.7), (-83.7, 75.1), (-82.2, 75.8), (-82.7, 76.5),
     (-84.5, 76.7), (-86.1, 76.4), (-87.8, 76.8), (-89.1, 76.9)],
    [(-105.0, 76.3), (-104.0, 76.1), (-103.3, 75.8), (-102.4, 75.7),
     (-101.5, 76.0), (-101.0, 76.6), (-101.7, 77.2), (-103.0, 77.3),
     (-104.2, 76.9), (-105.0, 76.9)],
    [(-96.0, 75.3), (-95.5, 74.9), (-94.6, 74.8), (-93.8, 75.0),
     (-93.4, 75.4), (-93.8, 75.8), (-94.7, 75.9), (-95.6, 75.7)],
    [(-99.1, 73.2), (-98.5, 72.9), (-97.9, 72.8), (-97.0, 73.0),
     (-96.0, 73.3), (-95.0, 73.7), (-94.0, 74.0), (-93.6, 74.6),
     (-94.4, 75.0), (-95.7, 74.9), (-96.5, 74.6), (-97.7, 74.3),
     (-98.6, 73.8)],
    [(-94.7, 74.0), (-94.1, 73.5), (-93.3, 73.4), (-92.4, 73.7),
     (-91.6, 74.2), (-90.6, 74.6), (-90.0, 75.1), (-90.7, 75.7),
     (-91.8, 75.8), (-92.6, 75.3), (-93.3, 74.8), (-94.2, 74.5)],
    [(-81.0, 73.0), (-79.8, 73.0), (-79.5, 73.5), (-79.7, 74.2),
     (-80.0, 74.9), (-80.6, 75.3), (-81.0, 75.1)],
    [(-105.0, 77.8), (-104.4, 77.5), (-103.7, 77.1), (-103.6, 76.8),
     (-104.4, 76.7), (-105.0, 76.9)],
    [(-101.6, 75.3), (-101.2, 75.1), (-100.8, 75.3), (-101.0, 75.6)],
    [(-99.8, 76.0), (-99.4, 75.8), (-99.0, 76.0), (-99.3, 76.3)],
    [(-98.8, 75.0), (-98.3, 74.8), (-97.9, 75.0), (-98.2, 75.3)],
    [(-97.5, 76.2), (-97.0, 75.9), (-96.7, 76.2), (-97.0, 76.5)],
    [(-92.1, 76.2), (-91.7, 76.0), (-91.3, 76.2), (-91.6, 76.5)],
    [(-88.0, 77.2), (-87.5, 77.0), (-87.1, 77.3), (-87.5, 77.6)],
]
# Additional island outlines in the Canadian Arctic Archipelago.
ISLANDS += [
    # Prince of Wales Island.
    [(-99.1, 73.2), (-98.5, 72.9), (-97.9, 72.8), (-97.0, 73.0),
     (-96.0, 73.3), (-95.0, 73.7), (-94.0, 74.0), (-93.6, 74.6),
     (-94.4, 75.0), (-95.7, 74.9), (-96.5, 74.6), (-97.7, 74.3),
     (-98.6, 73.8)],
    # Somerset Island.
    [(-94.7, 74.0), (-94.1, 73.5), (-93.3, 73.4), (-92.4, 73.7),
     (-91.6, 74.2), (-90.6, 74.6), (-90.0, 75.1), (-90.7, 75.7),
     (-91.8, 75.8), (-92.6, 75.3), (-93.3, 74.8), (-94.2, 74.5)],
    # Northwestern Baffin Island at the eastern frame edge.
    [(-81.0, 73.0), (-79.8, 73.0), (-79.5, 73.5), (-79.7, 74.2),
     (-80.0, 74.9), (-80.6, 75.3), (-81.0, 75.1)],
    # Eastern tip of Melville Island.
    [(-105.0, 77.8), (-104.4, 77.5), (-103.7, 77.1), (-103.6, 76.8),
     (-104.4, 76.7), (-105.0, 76.9)],
    [(-101.6, 75.3), (-101.2, 75.1), (-100.8, 75.3), (-101.0, 75.6)],
    [(-99.8, 76.0), (-99.4, 75.8), (-99.0, 76.0), (-99.3, 76.3)],
    [(-98.8, 75.0), (-98.3, 74.8), (-97.9, 75.0), (-98.2, 75.3)],
    [(-97.5, 76.2), (-97.0, 75.9), (-96.7, 76.2), (-97.0, 76.5)],
    [(-92.1, 76.2), (-91.7, 76.0), (-91.3, 76.2), (-91.6, 76.5)],
    [(-88.0, 77.2), (-87.5, 77.0), (-87.1, 77.3), (-87.5, 77.6)],
]
def points_in_polygon(longitudes, latitudes, polygon):
    """Return grid points inside a polygon using ray crossing."""
    inside = np.zeros(longitudes.shape, dtype=bool)
    x1, y1 = polygon[-1]
    for x2, y2 in polygon:
        crosses = (y1 > latitudes) != (y2 > latitudes)
        denominator = y2 - y1
        if denominator == 0:
            denominator = 1e-15
        x_cross = (x2 - x1) * (latitudes - y1) / denominator + x1
        inside ^= crosses & (longitudes < x_cross)
        x1, y1 = x2, y2
    return inside


def build_matrix():
    latitudes_1d = np.round(np.arange(79.0, 72.9, -0.1), 1)
    longitudes_1d = np.round(np.arange(-105.0, -80.9, 0.1), 1)
    longitudes, latitudes = np.meshgrid(longitudes_1d, latitudes_1d)
    land = np.zeros((len(latitudes_1d), len(longitudes_1d)), dtype=np.uint8)
    for polygon in ISLANDS:
        land[points_in_polygon(longitudes, latitudes, polygon)] = 1
    if land.shape != (61, 241):
        raise RuntimeError(f"Unexpected matrix shape: {land.shape}")
    return land
def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()

    with (output_dir / "candidate.csv").open(
        "w", newline="", encoding="ascii"
    ) as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    image = Image.fromarray(matrix * 255, mode="L")
    image.save(output_dir / "candidate.png")
    enlarged = image.resize(
        (241 * 8, 61 * 8), resample=Image.Resampling.NEAREST
    )
    enlarged.save(output_dir / "candidate_8x.png")

    with (output_dir / "candidate.csv").open(
        "r", newline="", encoding="ascii"
    ) as f:
        rows = list(csv.reader(f))
    assert len(rows) == 61 and all(len(row) == 241 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert sum(map(len, rows)) == 14701
    assert image.size == (241, 61)
    assert enlarged.size == (1928, 488)
    assert set(image.getdata()) <= {0, 255}
    assert set(enlarged.getdata()) <= {0, 255}


if __name__ == "__main__":
    main()
