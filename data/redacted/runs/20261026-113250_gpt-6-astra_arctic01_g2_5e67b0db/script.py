from pathlib import Path

import numpy as np
from PIL import Image


# Hand-drawn approximate island outlines, as (longitude, latitude).
LAND_POLYGONS = [
    # Ellesmere Island
    [
        (-82.8, 79.0), (-83.3, 78.5), (-83.5, 78.0), (-84.1, 77.5),
        (-83.7, 77.0), (-84.0, 76.5), (-83.2, 76.0), (-82.8, 75.5),
        (-82.7, 75.0), (-82.0, 74.7), (-81.0, 74.8), (-80.0, 75.3),
        (-78.0, 76.0), (-76.0, 77.0), (-75.0, 78.0), (-74.0, 79.0),
        (-74.0, 81.0), (-90.0, 81.0), (-90.0, 79.8), (-86.0, 79.5),
    ],
    # Axel Heiberg Island
    [
        (-92.5, 78.6), (-92.0, 79.0), (-90.6, 79.2), (-89.5, 79.0),
        (-88.5, 78.7), (-87.4, 78.2), (-86.8, 77.7), (-87.0, 77.2),
        (-87.8, 76.8), (-89.0, 76.6), (-90.1, 76.9), (-91.3, 77.3),
        (-92.1, 77.8),
    ],
    # Amund Ringnes Island
    [
        (-97.8, 78.1), (-97.0, 78.5), (-95.8, 78.6), (-94.8, 78.3),
        (-94.3, 77.8), (-94.8, 77.3), (-95.8, 77.1), (-96.9, 77.3),
        (-97.7, 77.7),
    ],
    # Ellef Ringnes and Lougheed Islands
    [
        (-103.2, 78.1), (-102.5, 78.7), (-101.3, 78.9), (-100.2, 78.6),
        (-99.5, 78.1), (-99.4, 77.5), (-100.0, 77.1), (-101.2, 77.0),
        (-102.3, 77.3), (-103.0, 77.7),
    ],
    [
        (-99.8, 77.4), (-99.1, 77.6), (-98.6, 77.4),
        (-98.8, 77.0), (-99.5, 77.0),
    ],
    # Bathurst Island
    [
        (-101.3, 76.6), (-100.7, 77.0), (-99.8, 77.2), (-98.9, 77.0),
        (-98.3, 76.5), (-98.4, 75.9), (-99.0, 75.5), (-100.0, 75.4),
        (-100.9, 75.8), (-101.4, 76.2),
    ],
    # Melville Island
    [
        (-106.0, 76.7), (-105.0, 76.9), (-104.0, 76.8), (-103.0, 76.5),
        (-102.0, 76.1), (-101.5, 75.7), (-102.0, 75.1), (-103.0, 74.8),
        (-104.2, 74.9), (-105.1, 75.4), (-106.0, 75.8),
    ],
    # Cornwallis Island
    [
        (-96.0, 75.4), (-95.4, 75.6), (-94.8, 75.4),
        (-94.7, 74.9), (-95.2, 74.6), (-95.8, 74.8),
    ],
    # Somerset Island
    [
        (-95.1, 74.5), (-94.3, 74.2), (-93.5, 73.7), (-92.8, 73.0),
        (-92.1, 72.6), (-92.4, 72.0), (-93.3, 71.9), (-94.3, 72.3),
        (-95.2, 73.0), (-95.6, 73.8),
    ],
    # Prince of Wales Island
    [
        (-102.4, 74.7), (-101.6, 75.0), (-100.6, 74.9), (-99.7, 74.4),
        (-99.0, 73.7), (-99.0, 72.8), (-99.7, 72.3), (-100.8, 72.4),
        (-101.8, 73.0), (-102.5, 73.8),
    ],
    # Approximate northwestern Baffin Island edge
    [
        (-82.0, 73.0), (-81.0, 73.0), (-81.0, 75.0), (-82.0, 75.0),
        (-83.0, 74.7), (-83.4, 74.1), (-83.0, 73.5),
    ],
]
def points_in_polygon(x_grid, y_grid, polygon):
    """Vectorized even-odd point-in-polygon test."""
    inside = np.zeros(x_grid.shape, dtype=bool)
    x1, y1 = polygon[-1]
    for x2, y2 in polygon:
        crosses = (y1 > y_grid) != (y2 > y_grid)
        x_intersection = (x2 - x1) * (y_grid - y1) / (
            (y2 - y1) if y2 != y1 else 1e-30
        ) + x1
        inside ^= crosses & (x_grid < x_intersection)
        x1, y1 = x2, y2
    return inside


def main():
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    latitudes = np.round(np.arange(79.0, 72.9, -0.1), 1)
    longitudes = np.round(np.arange(-105.0, -80.9, 0.1), 1)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    land = np.zeros((61, 241), dtype=np.uint8)
    for polygon in LAND_POLYGONS:
        land[points_in_polygon(lon_grid, lat_grid, polygon)] = 1

    np.savetxt(out_dir / "candidate.csv", land, fmt="%d", delimiter=",")
    image = Image.fromarray(land * 255, mode="L").convert("RGB")
    image.save(out_dir / "candidate.png")
    enlarged = image.resize(
        (241 * 8, 61 * 8), resample=Image.Resampling.NEAREST
    )
    enlarged.save(out_dir / "candidate_8x.png")

    loaded = np.loadtxt(out_dir / "candidate.csv", delimiter=",", dtype=np.uint8)
    if loaded.shape != (61, 241) or not np.isin(loaded, [0, 1]).all():
        raise RuntimeError("CSV validation failed")
    if image.size != (241, 61) or enlarged.size != (1928, 488):
        raise RuntimeError("Image dimension validation failed")
    if not set(enlarged.getdata()).issubset({(0, 0, 0), (255, 255, 255)}):
        raise RuntimeError("Image color validation failed")


if __name__ == "__main__":
    main()
