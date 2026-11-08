import csv
import os
import numpy as np
from PIL import Image

# Hand-sketched WGS84 land polygons. Coordinates are (longitude, latitude).
# These are approximate geographic knowledge rules, not a digitized coastline.
LAND = [
    # Banks Island
    [(-124.8, 72.7), (-123.8, 73.1), (-123.8, 74.5), (-121.8, 75.1),
     (-119.3, 75.0), (-117.1, 74.5), (-116.8, 73.6), (-118.4, 72.9),
     (-121.0, 72.5), (-123.0, 72.4)],
    # Victoria Island
    [(-118.5, 69.8), (-116.5, 70.3), (-114.8, 71.2), (-112.0, 71.6),
     (-109.0, 72.4), (-106.0, 73.1), (-103.4, 73.5), (-101.0, 73.2),
     (-100.2, 72.7), (-102.2, 72.0), (-104.8, 71.7), (-106.8, 71.0),
     (-109.5, 70.4), (-112.0, 69.7), (-115.0, 69.4), (-117.0, 69.4)],
    # Mainland coast, including Boothia and Melville Peninsula
    [(-110.0, 69.0), (-108.0, 70.1), (-106.0, 71.0), (-104.0, 71.5),
     (-102.0, 71.4), (-100.0, 71.0), (-98.5, 71.3), (-97.5, 72.0),
     (-96.5, 72.8), (-94.0, 73.2), (-91.0, 73.4), (-88.0, 73.0),
     (-85.5, 72.5), (-83.5, 72.7), (-81.0, 73.0), (-80.0, 72.8),
     (-80.0, 73.0), (-105.0, 73.0), (-110.0, 73.0)],
    # Prince of Wales Island
    [(-99.8, 72.0), (-98.4, 72.4), (-97.0, 72.2), (-96.0, 71.5),
     (-96.5, 70.7), (-98.0, 70.4), (-99.3, 70.9)],
    # Somerset Island
    [(-94.8, 72.0), (-93.0, 72.4), (-91.5, 72.1), (-90.8, 71.3),
     (-91.8, 70.6), (-93.5, 70.5), (-94.5, 71.1)],
    # Prince of Wales / Somerset neighboring small islands
    [(-96.5, 72.5), (-95.7, 72.7), (-95.2, 72.4), (-95.8, 72.1)],
    # Baffin Island western/northwestern edge in the frame
    [(-84.8, 73.0), (-83.0, 73.2), (-81.0, 73.5), (-80.0, 74.2),
     (-80.0, 73.0)],
    # Ellesmere Island
    [(-91.5, 76.0), (-90.0, 77.0), (-88.5, 78.0), (-87.0, 79.0),
     (-80.0, 79.0), (-80.0, 77.0), (-82.0, 76.2), (-84.0, 75.8),
     (-85.0, 75.0), (-87.0, 74.6), (-88.5, 75.0), (-90.0, 75.1)],
    # Axel Heiberg Island
    [(-95.0, 78.0), (-93.8, 78.8), (-91.5, 79.0), (-90.0, 78.5),
     (-90.2, 77.5), (-91.5, 76.8), (-93.5, 77.0), (-94.8, 77.3)],
    # Devon Island
    [(-91.0, 74.6), (-89.0, 75.0), (-87.0, 75.5), (-85.0, 75.8),
     (-83.0, 75.4), (-82.0, 74.8), (-83.0, 74.3), (-85.5, 74.0),
     (-88.0, 74.1), (-90.0, 74.2)],
    # Melville Island
    [(-114.5, 75.0), (-112.0, 75.4), (-110.0, 76.0), (-108.0, 76.2),
     (-105.0, 76.0), (-103.0, 75.4), (-104.0, 74.8), (-107.0, 74.6),
     (-110.0, 74.8), (-112.5, 74.6)],
    # Bathurst Island
    [(-101.0, 75.1), (-99.5, 75.8), (-98.0, 76.2), (-96.5, 76.0),
     (-96.0, 75.2), (-97.5, 74.9), (-99.5, 74.8)],
    # Cornwallis Island
    [(-96.5, 75.0), (-95.5, 75.3), (-94.5, 75.0), (-94.7, 74.5),
     (-95.8, 74.4), (-96.5, 74.7)],
    # Ellesmere-area larger islands
    [(-83.8, 78.5), (-82.5, 78.8), (-81.0, 78.5), (-80.0, 78.0),
     (-80.0, 79.0), (-84.0, 79.0)],
    # Smaller island groups represented by compact hand-sketched polygons
    [(-106.0, 77.0), (-104.7, 77.4), (-103.7, 77.1), (-104.5, 76.7)],
    [(-100.5, 77.6), (-99.4, 77.9), (-98.7, 77.5), (-99.5, 77.2)],
    [(-88.0, 76.5), (-86.7, 76.8), (-85.7, 76.4), (-86.5, 76.1)],
    [(-82.2, 76.0), (-81.3, 76.3), (-80.5, 76.0), (-81.0, 75.7)],
]


def inside(lon, lat, polygon):
    hit = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > lat) != (yj > lat):
            cross = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < cross:
                hit = not hit
        j = i
    return hit


def main():
    lats = [79.0 - 0.1 * i for i in range(61)]
    lons = [-105.0 + 0.1 * i for i in range(241)]
    matrix = [[int(any(inside(lon, lat, p) for p in LAND)) for lon in lons]
              for lat in lats]
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/candidate.csv", "w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(matrix)
    pixels = np.where(np.asarray(matrix, dtype=np.uint8) == 1, 255, 0).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    image.save("outputs/candidate.png")
    image.resize((1928, 488), Image.Resampling.NEAREST).save("outputs/candidate_8x.png")


if __name__ == "__main__":
    main()
