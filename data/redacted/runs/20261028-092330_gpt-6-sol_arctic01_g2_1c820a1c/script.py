from pathlib import Path
import csv

import numpy as np
from PIL import Image


def polygon(*points):
    return np.asarray(points, dtype=np.float64)


def inside_polygon(x, y, p):
    """Vectorized even-odd point-in-polygon test."""
    x0, y0 = p[:, 0, None], p[:, 1, None]
    x1, y1 = np.roll(p[:, 0], -1)[:, None], np.roll(p[:, 1], -1)[:, None]
    xx, yy = x[None, :], y[None, :]
    crosses = (y0 > yy) != (y1 > yy)
    denominator = np.where(y1 == y0, 1e-30, y1 - y0)
    intersections = x0 + (yy - y0) * (x1 - x0) / denominator
    return (np.sum(crosses & (xx < intersections), axis=0) % 2) == 1


# Approximate hand-authored island outlines; coordinates are longitude, latitude.
LAND_POLYGONS = [
    # Axel Heiberg Island
    polygon((-98.5, 79.2), (-98.2, 78.5), (-97.4, 77.8), (-96, 77.2),
            (-94, 76.7), (-92.2, 76.5), (-91.3, 76.9), (-91.1, 77.8),
            (-91.4, 79.2)),
    # Ellesmere Island within the frame
    polygon((-90, 79.2), (-90, 78.5), (-89.3, 77.9), (-88.2, 77.4),
            (-86.2, 76.9), (-83.8, 77), (-81, 77.5), (-80.5, 78.3),
            (-80.5, 79.2)),
    # Devon Island
    polygon((-97.2, 76.3), (-94.5, 76.6), (-91, 76.7), (-87, 76.8),
            (-84, 76.9), (-82, 76.3), (-82.2, 75.5), (-83, 74.9),
            (-85.5, 74.6), (-89, 74.4), (-92, 74.5), (-94.3, 74.7),
            (-96.6, 75.2)),
    # Bathurst Island
    polygon((-101.3, 76.8), (-98.8, 77), (-96.5, 76.8), (-94.5, 76.4),
            (-93.6, 75.9), (-94.8, 75.5), (-97, 75.3), (-99.4, 75.5),
            (-101.2, 76)),
    # Cornwallis Island
    polygon((-97.8, 76.2), (-96, 76.5), (-93.6, 76.4), (-91.8, 75.8),
            (-91.7, 75.1), (-93.8, 74.7), (-96.2, 74.9), (-97.5, 75.5)),
    # Melville Island
    polygon((-108, 76.2), (-105, 76.3), (-102.2, 76.1), (-100, 75.6),
            (-98, 75.1), (-96, 74.8), (-93, 74.6), (-90.7, 74.1),
            (-90.3, 73.7), (-92.3, 73.4), (-96, 73.4), (-100, 73.6),
            (-104, 74), (-107, 74.9)),
    # Eglinton Island
    polygon((-105.5, 76.2), (-104, 76.5), (-102.8, 76.1), (-102.4, 75.5),
            (-103.4, 75), (-105.2, 75.2), (-105.8, 75.7)),
    # Byam Martin Island
    polygon((-95.4, 76.6), (-93.4, 76.7), (-91.8, 76.3), (-91.5, 75.8),
            (-92.5, 75.4), (-94.5, 75.5), (-95.5, 76)),
    # Little Cornwallis Island
    polygon((-96, 75.4), (-94.8, 75.5), (-94.2, 75.1), (-94.9, 74.8),
            (-96, 74.9)),
    # North Kent Island
    polygon((-91, 77.3), (-89.8, 77.5), (-88.8, 77.1), (-89.2, 76.5),
            (-90.5, 76.4), (-91.2, 76.8)),
    # Prince of Wales Island
    polygon((-102.5, 74), (-100.5, 74.5), (-98.2, 74.4), (-96.5, 73.9),
            (-96.6, 73.2), (-98.5, 72.9), (-101, 73), (-102.7, 73.4)),
    # Somerset Island
    polygon((-96.5, 74.3), (-94, 74.5), (-91.8, 74.2), (-90, 73.7),
            (-89.4, 73.1), (-90, 72.7), (-92.2, 72.6), (-94.5, 72.8),
            (-96.2, 73.3)),
    # Boothia Peninsula
    polygon((-95.6, 73), (-95, 73.4), (-93.5, 73.7), (-91.6, 73.6),
            (-90, 73.2), (-89, 72.4), (-88, 71), (-87.8, 69.5),
            (-97, 69.5), (-97.2, 71.4), (-96.5, 72.4)),
]
def build_matrix():
    lats = np.round(np.arange(79.0, 72.95, -0.1), 10)
    lons = np.round(np.arange(-105.0, -80.95, 0.1), 10)
    if len(lats) != 61 or len(lons) != 241:
        raise RuntimeError("Unexpected grid dimensions")
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    flat_lon, flat_lat = lon_grid.ravel(), lat_grid.ravel()
    land = np.zeros(flat_lon.shape, dtype=bool)
    for p in LAND_POLYGONS:
        land |= inside_polygon(flat_lon, flat_lat, p)
    matrix = land.astype(np.uint8).reshape(61, 241)
    if matrix.shape != (61, 241) or matrix.size != 14701:
        raise RuntimeError("Unexpected matrix shape")
    if not np.all((matrix == 0) | (matrix == 1)):
        raise RuntimeError("Invalid matrix value")
    return matrix


def write_outputs(matrix):
    out = Path.cwd() / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "candidate.csv"
    png_path = out / "candidate.png"
    big_path = out / "candidate_8x.png"

    with csv_path.open("w", newline="", encoding="ascii") as stream:
        csv.writer(stream, lineterminator="\n").writerows(matrix.tolist())

    image = Image.fromarray((matrix * 255).astype(np.uint8), mode="L")
    if image.size != (241, 61):
        raise RuntimeError("Unexpected image dimensions")
    image.save(png_path, format="PNG")

    nearest = getattr(Image, "Resampling", Image).NEAREST
    enlarged = image.resize((1928, 488), resample=nearest)
    enlarged.save(big_path, format="PNG")

    with csv_path.open("r", newline="", encoding="ascii") as stream:
        rows = list(csv.reader(stream))
    if len(rows) != 61 or any(len(row) != 241 for row in rows):
        raise RuntimeError("CSV dimensions are invalid")
    if any(v not in {"0", "1"} for row in rows for v in row):
        raise RuntimeError("CSV contains an invalid value")
    for result in (image, enlarged):
        if not set(np.unique(np.asarray(result))).issubset({0, 255}):
            raise RuntimeError("PNG is not strictly black and white")
    return csv_path, png_path, big_path


if __name__ == "__main__":
    matrix = build_matrix()
    paths = write_outputs(matrix)
    print("matrix:", matrix.shape, "values:", matrix.size)
    for path in paths:
        print(path)
