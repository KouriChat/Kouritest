from pathlib import Path
import csv

import numpy as np
from PIL import Image


# Approximate land outlines, in (west longitude, latitude) coordinates.
LAND_POLYGONS = [
    # Mainland north coast and a section south of the sampling frame.
    [
        (-105.0, 73.70), (-103.5, 73.48), (-102.0, 73.26),
        (-100.5, 73.05), (-99.0, 72.82), (-97.5, 72.64),
        (-96.0, 72.72), (-94.5, 72.98), (-93.0, 73.18),
        (-91.5, 73.05), (-90.0, 72.85), (-88.5, 72.72),
        (-87.0, 72.80), (-85.5, 72.62), (-84.0, 72.48),
        (-81.0, 72.45), (-81.0, 70.0), (-105.0, 70.0),
    ],
    # Prince of Wales Island.
    [
        (-99.5, 73.45), (-99.0, 74.05), (-98.5, 74.65),
        (-97.6, 75.10), (-96.6, 75.25), (-95.8, 74.95),
        (-96.0, 74.45), (-96.6, 73.95), (-97.3, 73.52),
        (-98.1, 73.25), (-99.0, 73.20),
    ],
    # Somerset Island.
    [
        (-95.8, 73.55), (-95.4, 74.15), (-95.0, 74.85),
        (-94.1, 75.45), (-93.0, 75.55), (-92.2, 75.25),
        (-91.8, 74.75), (-92.0, 74.15), (-92.7, 73.65),
        (-93.7, 73.30), (-94.8, 73.22),
    ],
    # Cornwallis Island.
    [
        (-96.2, 74.55), (-95.7, 75.00), (-94.9, 75.18),
        (-94.0, 75.02), (-93.4, 74.72), (-93.8, 74.45),
        (-94.8, 74.36), (-95.7, 74.40),
    ],
]
LAND_POLYGONS.extend([
    # Eastern Bathurst Island.
    [
        (-105.0, 75.05), (-104.3, 75.38), (-103.8, 75.78),
        (-103.1, 76.18), (-102.3, 76.20), (-101.6, 75.92),
        (-101.2, 75.50), (-101.8, 75.18), (-102.7, 74.98),
        (-103.8, 74.96),
    ],
    # Eastern Melville Island.
    [
        (-105.0, 76.00), (-104.4, 76.42), (-103.8, 76.88),
        (-103.1, 77.25), (-102.2, 77.36), (-101.4, 77.10),
        (-101.0, 76.75), (-101.6, 76.40), (-102.4, 76.10),
        (-103.7, 75.94),
    ],
    # Axel Heiberg Island.
    [
        (-93.0, 78.08), (-92.1, 78.55), (-91.2, 79.05),
        (-90.1, 79.42), (-88.9, 79.45), (-87.9, 79.10),
        (-87.0, 78.72), (-86.0, 78.42), (-85.2, 78.04),
        (-85.4, 77.72), (-86.5, 77.62), (-87.5, 77.82),
        (-88.6, 77.94), (-89.7, 77.74), (-90.8, 77.72),
        (-91.8, 77.88), (-92.7, 77.82),
    ],
    # Ellesmere Island.
    [
        (-91.8, 78.30), (-91.4, 78.75), (-91.0, 79.15),
        (-90.6, 79.55), (-89.8, 79.92), (-88.8, 80.18),
        (-87.6, 80.46), (-86.4, 80.80), (-85.2, 81.10),
        (-83.7, 81.30), (-81.0, 81.42), (-81.0, 79.15),
        (-82.0, 78.80), (-83.2, 78.52), (-84.6, 78.24),
        (-86.0, 77.98), (-87.2, 77.72), (-88.4, 77.46),
        (-89.4, 77.22), (-90.3, 77.16), (-91.0, 77.42),
        (-91.4, 77.82),
    ],
])
LAND_POLYGONS.extend([
    # Devon Island.
    [
        (-90.8, 75.00), (-90.4, 75.65), (-89.7, 76.12),
        (-88.8, 76.58), (-87.8, 76.82), (-86.7, 76.84),
        (-85.7, 76.56), (-84.7, 76.32), (-83.7, 76.04),
        (-82.8, 75.72), (-82.0, 75.38), (-81.0, 75.08),
        (-81.0, 74.82), (-82.2, 74.66), (-83.5, 74.64),
        (-84.8, 74.73), (-86.0, 74.62), (-87.2, 74.56),
        (-88.4, 74.63), (-89.5, 74.72), (-90.3, 74.78),
    ],
    # Western edge of Baffin Island at the southeast border.
    [
        (-81.0, 73.00), (-79.8, 73.00), (-79.8, 74.20),
        (-80.2, 74.50), (-80.7, 74.35), (-81.0, 74.00),
    ],
    # Smaller central archipelago islands.
    [
        (-105.0, 77.20), (-104.4, 77.48), (-103.8, 77.62),
        (-103.2, 77.50), (-103.0, 77.24), (-103.6, 77.10),
        (-104.4, 77.08),
    ],
    [
        (-99.0, 76.20), (-98.4, 76.42), (-97.8, 76.36),
        (-97.5, 76.10), (-98.0, 75.92), (-98.7, 75.98),
    ],
    [
        (-96.0, 77.10), (-95.4, 77.42), (-94.8, 77.38),
        (-94.3, 77.12), (-94.6, 76.88), (-95.3, 76.86),
    ],
    [
        (-91.0, 76.05), (-90.5, 76.30), (-89.9, 76.28),
        (-89.7, 75.98), (-90.2, 75.82), (-90.8, 75.84),
    ],
    [
        (-88.0, 77.25), (-87.4, 77.48), (-86.8, 77.42),
        (-86.5, 77.14), (-87.0, 76.98), (-87.7, 77.00),
    ],
])
def point_in_polygon(lon, lat, polygon):
    """Ray casting test; points on polygon edges count as land."""
    inside = False
    count = len(polygon)
    for i in range(count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]
        cross = (lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
        if abs(cross) < 1e-10:
            if (min(x1, x2) - 1e-10 <= lon <= max(x1, x2) + 1e-10
                    and min(y1, y2) - 1e-10 <= lat <= max(y1, y2) + 1e-10):
                return True
        if (y1 > lat) != (y2 > lat):
            crossing_lon = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < crossing_lon:
                inside = not inside
    return inside


def is_land(lon, lat):
    return int(any(point_in_polygon(lon, lat, polygon)
                   for polygon in LAND_POLYGONS))
def main():
    latitudes = [79.0 - 0.1 * i for i in range(61)]
    longitudes = [-105.0 + 0.1 * i for i in range(241)]
    matrix = np.array(
        [[is_land(lon, lat) for lon in longitudes] for lat in latitudes],
        dtype=np.uint8,
    )
    if matrix.shape != (61, 241) or matrix.size != 14701:
        raise RuntimeError("Unexpected matrix dimensions")
    if not np.isin(matrix, (0, 1)).all():
        raise RuntimeError("Matrix contains values other than 0 and 1")

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerows(matrix.tolist())

    image = Image.fromarray(matrix * np.uint8(255), mode="L")
    image.save(output_dir / "candidate.png")
    enlarged = image.resize(
        (241 * 8, 61 * 8), resample=Image.Resampling.NEAREST
    )
    enlarged.save(output_dir / "candidate_8x.png")

    with csv_path.open("r", newline="", encoding="ascii") as file:
        rows = list(csv.reader(file))
    if len(rows) != 61 or any(len(row) != 241 for row in rows):
        raise RuntimeError("CSV dimensions are invalid")
    if any(value not in {"0", "1"} for row in rows for value in row):
        raise RuntimeError("CSV contains invalid or empty values")
    check = np.asarray(Image.open(output_dir / "candidate.png"))
    check_8x = np.asarray(Image.open(output_dir / "candidate_8x.png"))
    if check.shape != (61, 241):
        raise RuntimeError("Original PNG dimensions are invalid")
    if not set(np.unique(check)).issubset({0, 255}):
        raise RuntimeError("Original PNG contains non-black/white pixels")
    if check_8x.shape != (488, 1928):
        raise RuntimeError("Upscaled PNG dimensions are invalid")
    expected = np.repeat(np.repeat(check, 8, axis=0), 8, axis=1)
    if not np.array_equal(check_8x, expected):
        raise RuntimeError("Upscaled PNG is not exact nearest-neighbor")


if __name__ == "__main__":
    main()
