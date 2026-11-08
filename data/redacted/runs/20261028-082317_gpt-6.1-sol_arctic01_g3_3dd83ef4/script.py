#!/usr/bin/env python3
"""Knowledge-based schematic land mask for the Canadian Arctic Archipelago."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

OUT = Path("outputs")
LATS = [79.0 - 0.1 * i for i in range(61)]
LONS = [-105.0 + 0.1 * i for i in range(241)]

# Hand-estimated coastline vertices as (longitude, latitude), WGS84 degrees.
# These polygons are intentionally approximate and use no external geographic data.
LAND = [
    # Banks Island
    [(-123, 72.8), (-121, 73.0), (-120, 73.5), (-120, 74.2),
     (-120.4, 75.0), (-121, 75.8), (-121.5, 76.5), (-122.5, 77.0),
     (-124, 77.6), (-125.5, 77.8), (-127, 77.4), (-128.5, 76.8),
     (-129.5, 76.1), (-130.2, 75.2), (-130.5, 74.5), (-129.8, 73.8),
     (-128, 73.5), (-126, 73.1), (-124, 72.9)],
    # Victoria Island
    [(-114.5, 71.8), (-112.8, 72.0), (-111.0, 72.4), (-109.5, 72.8),
     (-108.4, 73.4), (-107.0, 74.0), (-106.2, 74.6), (-105.4, 75.3),
     (-104.4, 76.0), (-103.4, 76.7), (-102.5, 77.3), (-101.0, 77.7),
     (-99.3, 77.5), (-98.0, 77.0), (-97.1, 76.2), (-96.4, 75.4),
     (-96.4, 74.8), (-97.2, 74.3), (-98.8, 74.0), (-100.0, 73.6),
     (-101.2, 73.0), (-102.0, 72.5), (-103.5, 72.1), (-105.5, 71.8),
     (-108.0, 71.6), (-110.5, 71.6), (-112.5, 71.7)],
    # Melville Island
    [(-115.5, 75.0), (-114.5, 75.5), (-113.0, 75.7), (-111.8, 76.0),
     (-110.6, 76.2), (-109.7, 76.8), (-109.5, 77.6), (-110.2, 78.4),
     (-111.8, 78.8), (-113.5, 78.7), (-114.8, 78.3), (-116.0, 77.8),
     (-117.2, 77.3), (-118.0, 76.8), (-117.5, 76.2), (-116.5, 75.7)],
    # Prince of Wales Island
    [(-99.5, 72.0), (-98.5, 72.4), (-97.9, 73.0), (-97.7, 73.8),
     (-98.2, 74.5), (-99.0, 75.1), (-100.0, 75.4), (-101.0, 75.1),
     (-101.4, 74.4), (-101.2, 73.7), (-100.8, 73.0), (-100.3, 72.4)],
    # Somerset Island
    [(-94.8, 72.0), (-93.8, 72.3), (-92.7, 72.7), (-91.9, 73.3),
     (-91.5, 74.0), (-91.9, 74.6), (-92.7, 75.0), (-93.8, 74.9),
     (-94.7, 74.5), (-95.1, 73.8), (-95.0, 73.0)],
    # Devon Island
    [(-90.2, 74.5), (-89.4, 74.8), (-88.0, 75.0), (-86.5, 75.3),
     (-84.8, 75.5), (-83.2, 75.8), (-81.2, 76.2), (-79.0, 76.5),
     (-77.2, 76.3), (-75.8, 75.8), (-76.2, 75.2), (-77.8, 74.9),
     (-79.8, 74.7), (-82.0, 74.3), (-84.0, 73.9), (-86.0, 73.6),
     (-88.0, 73.7), (-89.4, 74.0)],
    # Axel Heiberg Island
    [(-93.2, 78.0), (-92.0, 78.4), (-90.8, 78.8), (-89.3, 79.0),
     (-88.2, 78.7), (-87.3, 78.2), (-87.0, 77.7), (-87.8, 77.3),
     (-89.0, 77.1), (-90.3, 77.3), (-91.4, 77.5), (-92.5, 77.6)],
    # Ellesmere Island, including the north coast clipped by the frame.
    [(-93.0, 79.0), (-90.0, 79.0), (-87.0, 79.0), (-84.0, 79.0),
     (-81.0, 79.0), (-78.0, 78.7), (-76.5, 78.2), (-75.5, 77.5),
     (-74.8, 76.7), (-73.8, 75.9), (-72.7, 75.1), (-71.5, 74.6),
     (-70.0, 74.4), (-68.5, 74.7), (-67.0, 75.0), (-65.5, 75.5),
     (-64.0, 76.1), (-62.5, 76.8), (-61.0, 77.5), (-60.0, 78.2),
     (-59.5, 79.0)],
    # Baffin Island's western shore enters the eastern edge of the frame.
    [(-81.0, 73.0), (-80.0, 73.2), (-78.8, 73.8), (-77.8, 74.6),
     (-76.8, 75.4), (-75.5, 76.2), (-74.0, 77.0), (-72.0, 77.8),
     (-70.0, 77.7), (-67.0, 77.4), (-64.0, 76.8), (-61.0, 76.0),
     (-60.0, 75.5), (-60.0, 73.0)],
    # Mainland north coast and Boothia Peninsula, clipped at 73N.
    [(-105.0, 72.2), (-103.0, 72.0), (-101.0, 71.8), (-99.0, 71.7),
     (-97.0, 71.8), (-95.0, 72.0), (-93.5, 72.2), (-92.0, 72.3),
     (-90.0, 72.2), (-88.0, 72.0), (-86.0, 71.8), (-84.0, 71.7),
     (-82.0, 71.8), (-80.0, 71.9), (-78.0, 71.5), (-76.0, 71.2),
     (-74.0, 70.8), (-72.0, 70.5), (-70.0, 70.0), (-68.0, 69.5),
     (-66.0, 69.0), (-64.0, 68.0), (-62.0, 67.0), (-60.0, 66.0),
     (-60.0, 60.0), (-105.0, 60.0)],
]
def on_land(lon, lat):
    inside = False
    for poly in LAND:
        hit = False
        j = len(poly) - 1
        for i, (xi, yi) in enumerate(poly):
            xj, yj = poly[j]
            if (yi > lat) != (yj > lat):
                cross = (xj - xi) * (lat - yi) / (yj - yi) + xi
                if lon < cross:
                    hit = not hit
            j = i
        if hit:
            inside = not inside if False else True
            return True
    return False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    matrix = np.array([[int(on_land(lon, lat)) for lon in LONS]
                       for lat in LATS], dtype=np.uint8)
    with (OUT / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())
    # 1 -> white land; 0 -> black water.
    pixels = matrix * 255
    image = Image.fromarray(pixels, mode="L")
    image.save(OUT / "candidate.png")
    image.resize((241 * 8, 61 * 8), resample=Image.Resampling.NEAREST).save(
        OUT / "candidate_8x.png")

    with (OUT / "candidate.csv").open("r", newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 61 and all(len(row) == 241 for row in rows)
    assert all(v in ("0", "1") for row in rows for v in row)
    assert matrix.shape == (61, 241) and matrix.size == 14701
    assert set(np.unique(np.asarray(image))) == {0, 255}
    enlarged = Image.open(OUT / "candidate_8x.png")
    assert enlarged.size == (1928, 488)
    assert np.array_equal(np.asarray(enlarged),
                          np.repeat(np.repeat(pixels, 8, axis=0), 8, axis=1))
    print("validated: 61x241 matrix, 14,701 binary values, two-color PNGs")


if __name__ == "__main__":
    main()
