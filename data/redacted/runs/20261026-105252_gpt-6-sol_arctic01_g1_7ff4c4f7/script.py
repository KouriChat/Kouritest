from pathlib import Path
import csv
import numpy as np
from PIL import Image

# Hand-estimated WGS84 coastline vertices (longitude west, latitude north).
# These intentionally approximate the major islands in the requested window.
ISLANDS = [
    # Prince of Wales Island
    [(-102.0, 76.5), (-99.8, 76.4), (-98.5, 75.9), (-97.0, 75.6),
     (-96.2, 75.0), (-96.1, 74.6), (-97.0, 74.2), (-98.2, 74.1),
     (-99.0, 74.4), (-100.0, 74.7), (-101.1, 75.0), (-102.0, 75.6)],
    # Somerset Island
    [(-95.7, 74.9), (-94.6, 74.9), (-93.4, 74.4), (-92.8, 73.9),
     (-93.2, 73.4), (-94.2, 72.9), (-95.2, 72.8), (-96.0, 73.3),
     (-96.1, 74.0), (-96.0, 74.6)],
    # Devon Island
    [(-90.0, 76.4), (-87.5, 76.6), (-85.5, 76.4), (-84.0, 76.0),
     (-82.3, 75.6), (-81.0, 75.2), (-81.0, 74.0), (-82.1, 74.0),
     (-83.0, 74.4), (-84.5, 74.5), (-86.0, 74.7), (-87.5, 75.0),
     (-89.2, 75.2), (-90.5, 75.8)],
    # Axel Heiberg Island
    [(-96.0, 79.0), (-95.1, 78.8), (-94.0, 78.4), (-93.0, 78.0),
     (-92.5, 77.7), (-91.4, 77.6), (-90.4, 77.9), (-89.8, 78.4),
     (-90.2, 78.8), (-91.5, 79.0), (-93.0, 79.2), (-94.5, 79.3)],
    # Ellesmere Island: western/southwestern margin in the map window
    [(-85.5, 79.0), (-84.8, 78.7), (-84.5, 78.2), (-84.0, 77.8),
     (-83.2, 77.5), (-82.2, 77.3), (-81.0, 77.1), (-81.0, 79.0)],
    # Ellef Ringnes Island
    [(-103.5, 78.0), (-102.5, 78.2), (-101.0, 78.2), (-99.8, 78.0),
     (-99.4, 77.7), (-100.0, 77.4), (-101.5, 77.3), (-103.0, 77.5)],
    # Amund Ringnes Island
    [(-97.0, 78.0), (-96.0, 78.2), (-94.5, 78.1), (-93.2, 77.8),
     (-93.5, 77.5), (-95.0, 77.4), (-96.4, 77.6)],
    # Bathurst Island
    [(-98.7, 76.9), (-97.8, 76.8), (-96.8, 76.4), (-96.4, 75.8),
     (-97.0, 75.4), (-98.2, 75.5), (-99.0, 76.0)],
    # Melville Island (eastern portion)
    [(-111.0, 76.9), (-108.5, 77.0), (-106.0, 76.8), (-104.0, 76.4),
     (-103.4, 76.0), (-104.0, 75.6), (-106.0, 75.5), (-108.0, 75.8),
     (-110.0, 76.0)],
    # Banks Island eastern tip, only a small part enters the western edge
    [(-105.0, 74.0), (-104.5, 74.0), (-104.5, 73.0), (-105.0, 73.0)],
    # Victoria Island northeastern tip at the southern frame edge
    [(-105.0, 73.0), (-104.0, 73.0), (-104.3, 73.3), (-105.0, 73.4)],
]


def on_land(lon_west, lat):
    for polygon in ISLANDS:
        inside = False
        j = len(polygon) - 1
        for i, (xi, yi) in enumerate(polygon):
            xj, yj = polygon[j]
            if (yi > lat) != (yj > lat):
                crossing = (xj - xi) * (lat - yi) / (yj - yi) + xi
                if lon_west < crossing:
                    inside = not inside
            j = i
        if inside:
            return 1
    return 0


def main():
    latitudes = [79.0 - 0.1 * row for row in range(61)]
    longitudes = [-105.0 + 0.1 * col for col in range(241)]
    matrix = np.array(
        [[on_land(lon, lat) for lon in longitudes] for lat in latitudes],
        dtype=np.uint8,
    )

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Matrix 1 (land) becomes white; 0 (water) becomes black.
    pixels = matrix * 255
    image = Image.fromarray(pixels, mode="L").convert("RGB")
    image.save(out / "candidate.png")
    image.resize((241 * 8, 61 * 8), resample=Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )


if __name__ == "__main__":
    main()
