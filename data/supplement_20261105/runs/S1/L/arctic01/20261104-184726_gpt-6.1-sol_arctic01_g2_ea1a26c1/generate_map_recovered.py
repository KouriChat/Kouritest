from pathlib import Path
import csv

import numpy as np
from PIL import Image

# Polygon vertices are (longitude, latitude), with west longitudes negative.
# These hand-written outlines are deliberately approximate.
LAND_POLYGONS = [
    # Banks Island
    [(-125.0, 73.0), (-124.7, 74.0), (-123.8, 75.2), (-122.5, 76.3),
     (-120.8, 77.2), (-118.6, 77.8), (-116.3, 77.7), (-114.8, 77.0),
     (-113.5, 76.0), (-113.0, 74.9), (-113.7, 73.8), (-115.3, 73.0),
     (-119.0, 72.8), (-122.5, 72.9)],

    # Victoria Island, including its northern and western coasts
    [(-119.0, 73.0), (-117.5, 73.7), (-116.0, 74.6), (-115.2, 75.5),
     (-116.0, 76.2), (-117.7, 76.7), (-118.6, 77.4), (-117.7, 78.0),
     (-115.2, 78.5), (-112.5, 78.7), (-109.0, 78.5), (-106.0, 78.0),
     (-104.5, 77.2), (-103.5, 76.0), (-103.3, 74.8), (-104.4, 73.8),
     (-106.0, 73.0), (-110.0, 72.8), (-114.0, 72.8)],

    # Mainland coast south of Victoria Island
    [(-105.0, 73.35), (-102.8, 73.6), (-100.0, 73.9), (-97.5, 74.2),
     (-95.0, 74.4), (-92.0, 74.4), (-89.0, 74.3), (-86.0, 74.2),
     (-83.0, 74.0), (-81.0, 73.8), (-81.0, 73.0), (-105.0, 73.0)],

    # Prince of Wales Island
    [(-101.8, 72.9), (-101.5, 73.5), (-100.7, 74.2), (-99.8, 74.9),
     (-98.3, 75.5), (-96.5, 75.8), (-95.1, 75.5), (-94.5, 74.8),
     (-94.7, 74.0), (-95.6, 73.3), (-97.0, 72.8), (-99.4, 72.7)],

    # Somerset Island
    [(-95.0, 73.0), (-94.8, 73.8), (-94.2, 74.7), (-93.2, 75.5),
     (-91.7, 76.1), (-90.2, 76.2), (-89.2, 75.7), (-89.3, 74.8),
     (-90.0, 73.9), (-91.2, 73.2), (-93.0, 72.9)],

    # Boothia Peninsula
    [(-95.0, 73.0), (-93.0, 73.7), (-91.5, 74.8), (-90.5, 75.8),
     (-89.6, 76.7), (-88.5, 77.4), (-86.8, 77.7), (-85.5, 77.2),
     (-85.3, 76.4), (-86.2, 75.6), (-87.2, 74.8), (-88.5, 73.8),
     (-90.0, 73.0)],

    # King William Island
    [(-99.0, 68.0), (-98.0, 69.0), (-96.0, 70.0), (-94.0, 70.5),
     (-92.0, 70.2), (-91.0, 69.0), (-92.0, 68.0)],

    # Baffin Island, western portion in the requested frame
    [(-81.0, 73.0), (-82.0, 73.6), (-83.0, 74.4), (-84.0, 75.2),
     (-85.0, 76.0), (-86.0, 76.7), (-87.0, 77.4), (-88.0, 78.2),
     (-89.0, 79.0), (-81.0, 79.0)],

    # Somerset and nearby high-Arctic islands
    [(-94.0, 76.4), (-93.0, 77.0), (-91.0, 77.2), (-89.0, 76.9),
     (-88.2, 76.3), (-89.0, 75.9), (-91.0, 76.0), (-93.0, 76.1)],
    [(-91.0, 77.2), (-90.0, 78.0), (-88.5, 78.5), (-87.0, 78.3),
     (-86.7, 77.7), (-88.0, 77.1), (-89.5, 77.0)],
    [(-96.0, 77.2), (-95.0, 77.8), (-93.5, 78.0), (-92.0, 77.7),
     (-92.5, 77.1), (-94.0, 77.0)],

    # Smaller islands in the western archipelago
    [(-112.0, 78.9), (-110.8, 79.0), (-110.0, 78.7), (-111.0, 78.4),
     (-112.0, 78.5)],
    [(-108.0, 78.9), (-106.5, 79.0), (-105.8, 78.7), (-107.0, 78.5)],
    [(-104.0, 78.7), (-102.8, 78.9), (-101.8, 78.6), (-102.7, 78.3)],
    [(-100.0, 78.8), (-98.8, 79.0), (-98.0, 78.7), (-99.0, 78.4)],
    [(-97.0, 78.5), (-96.0, 78.8), (-95.0, 78.6), (-95.7, 78.2)],
    [(-94.0, 78.7), (-93.0, 79.0), (-92.0, 78.8), (-92.8, 78.4)],
    [(-90.0, 78.7), (-89.0, 79.0), (-88.0, 78.8), (-88.8, 78.4)],
]


def point_on_segment(x, y, x1, y1, x2, y2, eps=1e-10):
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > eps:
        return False
    return (min(x1, x2) - eps <= x <= max(x1, x2) + eps and
            min(y1, y2) - eps <= y <= max(y1, y2) + eps)


def point_in_polygon(lon, lat, polygon):
    inside = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        if point_on_segment(lon, lat, x1, y1, x2, y2):
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
    assert matrix.shape == (61, 241)
    assert matrix.size == 14701
    assert np.isin(matrix, (0, 1)).all()

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "candidate.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Matrix 1 (land) maps to white; 0 (water) maps to black.
    pixels = matrix * 255
    image = Image.fromarray(pixels, mode="L").convert("RGB")
    assert image.size == (241, 61)
    assert set(image.getdata()).issubset({(0, 0, 0), (255, 255, 255)})
    image.save(output_dir / "candidate.png")

    enlarged = image.resize((241 * 8, 61 * 8), resample=Image.Resampling.NEAREST)
    assert enlarged.size == (1928, 488)
    assert set(enlarged.getdata()).issubset({(0, 0, 0), (255, 255, 255)})
    enlarged.save(output_dir / "candidate_8x.png")


if __name__ == "__main__":
    main()
