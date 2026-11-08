from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 127
STEP = 2.0 / 126.0

# Hand-authored approximate land polygons in (longitude, latitude) order.
LAND_POLYGONS = [
    # Mainland coast and the eastern landmass within the frame.
    [
        (-130.62, 55.00), (-130.54, 54.86), (-130.42, 54.72),
        (-130.30, 54.60), (-130.23, 54.43), (-130.28, 54.27),
        (-130.42, 54.12), (-130.50, 53.98), (-130.38, 53.84),
        (-130.24, 53.70), (-130.08, 53.55), (-129.90, 53.40),
        (-129.75, 53.22), (-129.62, 53.00), (-129.00, 53.00),
        (-129.00, 55.00),
    ],
    # Banks Island.
    [
        (-130.42, 54.16), (-130.31, 54.13), (-130.21, 54.08),
        (-130.13, 53.98), (-130.05, 53.86), (-130.00, 53.72),
        (-130.00, 53.56), (-130.05, 53.43), (-130.14, 53.36),
        (-130.25, 53.39), (-130.33, 53.48), (-130.38, 53.63),
        (-130.42, 53.82), (-130.46, 54.00),
    ],
    # Porcher Island.
    [
        (-130.68, 54.02), (-130.57, 54.03), (-130.48, 53.98),
        (-130.42, 53.90), (-130.43, 53.80), (-130.49, 53.73),
        (-130.58, 53.74), (-130.66, 53.81), (-130.71, 53.92),
    ],
    # Dundas Islands.
    [
        (-131.00, 54.70), (-130.93, 54.72), (-130.86, 54.68),
        (-130.84, 54.62), (-130.89, 54.57), (-130.97, 54.58),
        (-131.02, 54.63),
    ],
    [
        (-130.91, 54.54), (-130.84, 54.56), (-130.78, 54.52),
        (-130.79, 54.46), (-130.85, 54.44), (-130.91, 54.48),
    ],
    # Small islands scattered through the strait.
    [
        (-130.79, 54.82), (-130.74, 54.84), (-130.70, 54.80),
        (-130.72, 54.76), (-130.77, 54.77),
    ],
    [
        (-130.72, 54.36), (-130.67, 54.38), (-130.62, 54.33),
        (-130.65, 54.28), (-130.70, 54.30),
    ],
    # Aristazabal Island, clipped at the southern boundary.
    [
        (-130.28, 53.42), (-130.18, 53.38), (-130.10, 53.30),
        (-130.06, 53.18), (-130.08, 53.00), (-130.36, 53.00),
        (-130.38, 53.18), (-130.35, 53.32),
    ],
]
def points_in_polygon(x, y, polygon):
    """Vectorized ray-casting point-in-polygon test."""
    inside = np.zeros(x.shape, dtype=bool)
    x1, y1 = polygon[-1]
    for x2, y2 in polygon:
        crosses = (y1 > y) != (y2 > y)
        x_cross = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1
        inside ^= crosses & (x < x_cross)
        x1, y1 = x2, y2
    return inside


def main():
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    latitudes = 55.0 - np.arange(SIZE) * STEP
    longitudes = -131.0 + np.arange(SIZE) * STEP
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    land = np.zeros((SIZE, SIZE), dtype=bool)
    for polygon in LAND_POLYGONS:
        land |= points_in_polygon(lon_grid, lat_grid, polygon)

    matrix = land.astype(np.uint8)
    np.savetxt(out_dir / "candidate.csv", matrix, delimiter=",", fmt="%d")

    image = Image.fromarray(matrix * 255, mode="L")
    image.save(out_dir / "candidate.png")
    image.resize((SIZE * 8, SIZE * 8), Image.Resampling.NEAREST).save(
        out_dir / "candidate_8x.png"
    )


if __name__ == "__main__":
    main()
