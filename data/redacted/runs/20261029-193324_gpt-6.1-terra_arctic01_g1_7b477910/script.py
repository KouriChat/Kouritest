from pathlib import Path
import csv

import numpy as np
from PIL import Image


# Hand-estimated (longitude, latitude) polygons; no external map data is used.
LAND_POLYGONS = [
    [(-105.0, 75.55), (-104.2, 75.78), (-104.8, 76.15),
     (-105.0, 76.55), (-105.0, 75.55)],
    [(-100.35, 72.72), (-99.65, 72.15), (-98.60, 72.05),
     (-97.35, 72.48), (-96.72, 73.35), (-96.78, 74.45),
     (-97.40, 75.10), (-98.25, 75.72), (-99.35, 75.55),
     (-100.20, 74.70), (-100.48, 73.75), (-100.35, 72.72)],
    [(-95.72, 72.03), (-94.52, 71.97), (-93.05, 72.30),
     (-91.82, 73.00), (-91.58, 73.78), (-92.18, 74.42),
     (-93.18, 74.72), (-94.35, 74.50), (-95.27, 73.92),
     (-95.72, 73.12), (-95.72, 72.03)],
    [(-96.05, 73.00), (-95.52, 72.82), (-94.72, 72.70),
     (-93.65, 72.76), (-92.88, 73.00), (-93.05, 73.38),
     (-93.62, 73.65), (-94.65, 73.72), (-95.45, 73.48),
     (-96.05, 73.00)],
    [(-101.52, 75.47), (-100.90, 75.35), (-99.92, 75.52),
     (-98.95, 75.89), (-97.62, 76.08), (-96.60, 76.52),
     (-96.85, 77.00), (-97.85, 77.29), (-99.06, 77.17),
     (-100.15, 76.83), (-101.12, 76.35), (-101.52, 75.47)],
    [(-96.28, 74.83), (-95.62, 74.72), (-94.67, 74.91),
     (-93.78, 75.37), (-93.35, 75.92), (-93.87, 76.25),
     (-94.85, 76.30), (-95.75, 76.02), (-96.20, 75.51),
     (-96.28, 74.83)],
    [(-93.64, 74.48), (-92.54, 74.43), (-91.23, 74.62),
     (-89.93, 74.77), (-88.67, 75.00), (-87.34, 75.19),
     (-86.15, 75.42), (-84.92, 75.83), (-83.72, 76.35),
     (-82.77, 76.94), (-83.45, 77.42), (-84.68, 77.60),
     (-86.08, 77.50), (-87.35, 77.30), (-88.66, 77.11),
     (-90.10, 77.00), (-91.36, 76.76), (-92.32, 76.34),
     (-93.17, 75.75), (-93.64, 74.48)],
]
LAND_POLYGONS += [
    [(-93.95, 78.00), (-93.45, 78.45), (-92.75, 78.92),
     (-91.62, 79.31), (-90.25, 79.53), (-88.90, 79.48),
     (-87.65, 79.22), (-86.33, 78.86), (-85.36, 78.42),
     (-85.82, 78.06), (-87.04, 77.91), (-88.45, 77.98),
     (-89.80, 78.14), (-91.05, 78.01), (-92.22, 77.91),
     (-93.24, 77.89), (-93.95, 78.00)],
    [(-90.35, 79.00), (-89.72, 79.40), (-88.76, 79.75),
     (-87.62, 80.00), (-86.22, 80.00), (-84.78, 79.72),
     (-83.62, 79.36), (-82.52, 78.89), (-81.42, 78.38),
     (-81.00, 78.03), (-81.00, 79.00), (-90.35, 79.00)],
    [(-91.40, 78.05), (-90.30, 78.10), (-89.26, 77.89),
     (-88.18, 77.62), (-87.22, 77.28), (-86.12, 76.97),
     (-85.22, 76.60), (-84.35, 76.20), (-83.65, 75.74),
     (-82.85, 75.31), (-81.75, 75.03), (-81.00, 75.12),
     (-81.00, 78.05), (-91.40, 78.05)],
    [(-95.50, 77.45), (-94.72, 77.33), (-93.55, 77.49),
     (-92.48, 77.79), (-91.78, 78.22), (-92.40, 78.58),
     (-93.52, 78.66), (-94.57, 78.46), (-95.30, 78.03),
     (-95.50, 77.45)],
    [(-102.58, 77.55), (-101.92, 77.45), (-100.82, 77.62),
     (-99.68, 77.92), (-98.93, 78.34), (-99.41, 78.80),
     (-100.50, 79.05), (-101.55, 78.90), (-102.43, 78.49),
     (-102.58, 77.55)],
    [(-101.02, 77.47), (-100.23, 77.46), (-99.30, 77.70),
     (-98.43, 78.05), (-98.20, 78.48), (-98.83, 78.78),
     (-99.78, 78.83), (-100.62, 78.53), (-101.07, 78.05),
     (-101.02, 77.47)],
    [(-100.22, 77.70), (-99.65, 77.61), (-98.96, 77.79),
     (-98.57, 78.13), (-98.90, 78.47), (-99.56, 78.51),
     (-100.16, 78.22), (-100.22, 77.70)],
]
LAND_POLYGONS += [
    [(-110.0, 77.18), (-108.8, 77.20), (-107.7, 77.50),
     (-106.9, 77.92), (-107.5, 78.25), (-108.8, 78.25),
     (-109.8, 77.90), (-110.0, 77.18)],
    [(-104.0, 78.42), (-103.5, 78.38), (-103.0, 78.60),
     (-103.2, 78.84), (-103.8, 78.83), (-104.0, 78.42)],
    [(-97.4, 78.85), (-96.8, 78.79), (-96.3, 79.00),
     (-96.6, 79.18), (-97.2, 79.15), (-97.4, 78.85)],
]


def points_in_polygon(x, y, polygon):
    """Vectorized ray-casting point-in-polygon test."""
    inside = np.zeros(x.shape, dtype=bool)
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:]):
        crosses = (y1 > y) != (y2 > y)
        x_intersection = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1
        inside ^= crosses & (x < x_intersection)
    return inside


def build_matrix():
    latitudes = np.arange(790, 729, -1, dtype=np.int16) / 10.0
    longitudes = np.arange(-1050, -809, 1, dtype=np.int16) / 10.0
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    land = np.zeros((61, 241), dtype=bool)
    for polygon in LAND_POLYGONS:
        land |= points_in_polygon(lon_grid, lat_grid, polygon)
    return land.astype(np.uint8)
def main():
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()
    if matrix.shape != (61, 241) or matrix.size != 14701:
        raise RuntimeError("Unexpected matrix dimensions")

    csv_path = out_dir / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

    image = Image.fromarray(matrix * 255, mode="L")
    image.save(out_dir / "candidate.png")
    enlarged = image.resize((1928, 488), resample=Image.Resampling.NEAREST)
    enlarged.save(out_dir / "candidate_8x.png")

    with csv_path.open("r", newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    if len(rows) != 61 or any(len(row) != 241 for row in rows):
        raise RuntimeError("CSV dimensions are incorrect")
    if any(v not in {"0", "1"} for row in rows for v in row):
        raise RuntimeError("CSV contains invalid values")
    if image.size != (241, 61) or enlarged.size != (1928, 488):
        raise RuntimeError("PNG dimensions are incorrect")
    if not set(np.unique(np.asarray(image))).issubset({0, 255}):
        raise RuntimeError("PNG is not strictly black and white")

    print("Created outputs/candidate.csv, candidate.png, candidate_8x.png")
    print(f"Matrix: {matrix.shape[0]} x {matrix.shape[1]} = {matrix.size}")


if __name__ == "__main__":
    main()
