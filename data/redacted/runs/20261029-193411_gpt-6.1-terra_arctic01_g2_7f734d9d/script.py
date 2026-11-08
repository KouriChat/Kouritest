from pathlib import Path
import csv
import numpy as np
from PIL import Image

# Hand-drawn coastlines, in (west longitude, north latitude).  These are
# deliberately local approximations rather than imported GIS data.
def inside(x, y, poly):
    hit = False
    n = len(poly)
    for k in range(n):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n]
        if ((y1 > y) != (y2 > y)):
            xx = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xx:
                hit = not hit
    return hit

def P(*v):
    return list(zip(v[::2], v[1::2]))

# Major islands and the small land masses which materially affect this box.
ISLANDS = [
    # Victoria Island: only its eastern edge enters the western side here.
    P(-105,73.0, -104.7,74.0, -103.2,75.0, -101.0,75.8,
      -99.5,75.5, -99.0,74.5, -100.2,73.6, -101.8,73.0),
    # Melville Island, with its irregular southern and eastern coasts.
    P(-105,75.0, -104.2,76.3, -101.8,77.3, -98.5,78.1, -95.0,78.6,
      -91.2,78.4, -89.3,77.7, -89.8,76.7, -92.0,76.0, -96.0,75.5,
      -100.0,74.8, -103.0,74.7),
    # Bathurst Island.
    P(-101.8,75.6, -101.2,76.6, -99.5,77.1, -96.8,76.9,
      -95.0,76.2, -95.5,75.5, -98.0,75.1, -100.3,75.2),
    # Byam Martin Island.
    P(-94.5,77.2, -93.3,78.0, -91.0,78.3, -89.6,77.7,
      -90.4,77.0, -92.4,76.7),
    # Cornwallis Island.
    P(-96.0,74.7, -95.0,75.7, -93.4,76.4, -91.0,76.3,
      -89.8,75.7, -90.3,74.9, -92.1,74.5, -94.4,74.4),
    # Prince of Wales Island.
    P(-103.1,73.0, -101.0,73.1, -98.8,73.6, -96.0,74.0,
      -93.0,74.6, -90.8,75.4, -90.2,76.1, -92.1,76.5,
      -95.6,76.4, -99.0,76.0, -102.0,75.4, -103.4,74.5),
    # Somerset Island.
    P(-96.0,72.95, -94.2,72.9, -92.0,73.3, -90.3,74.0,
      -89.2,74.9, -90.0,75.5, -92.3,75.5, -94.5,75.0,
      -96.0,74.2),
    # Boothia Peninsula's northern tip at the southern frame.
    P(-97.0,73.0, -96.0,73.5, -94.0,73.8, -91.8,73.5,
      -90.0,73.0),
]
ISLANDS += [
    # Devon Island: broad high island east of Cornwallis and south of Ellesmere.
    P(-95.2,74.7, -92.4,74.4, -89.0,74.6, -85.5,75.0,
      -82.0,76.0, -80.8,77.0, -81.0,78.3, -84.3,78.8,
      -87.7,78.2, -90.2,77.3, -93.0,76.4, -95.0,75.6),
    # Axel Heiberg Island, separated from Ellesmere by a narrow sound.
    P(-93.0,77.0, -92.0,78.0, -90.2,79.0, -87.2,79.1,
      -85.5,78.3, -86.2,77.4, -88.2,76.9, -90.7,76.7),
    # Ellesmere's southwest corner and the western part visible in the frame.
    P(-88.0,77.8, -87.0,78.7, -85.0,79.1, -81.0,79.2,
      -80.7,76.8, -82.0,76.0, -84.0,76.3, -86.0,77.0),
    # Beechey Island and nearby small islands.
    P(-91.9,74.9, -91.0,75.2, -90.3,75.0, -90.6,74.6, -91.6,74.6),
    # Little Cornwallis Island.
    P(-95.7,76.4, -94.7,76.9, -93.5,76.8, -93.1,76.2,
      -94.0,75.9, -95.2,76.0),
    # Bylot Island, just touching the eastern edge of the sampling frame.
    P(-81.2,73.8, -80.4,74.5, -79.0,74.6, -78.1,74.0,
      -79.0,73.3, -80.5,73.2),
    # Several compact islands in the channels around the central group.
    P(-99.1,76.0, -98.4,76.4, -97.6,76.2, -97.4,75.8, -98.3,75.7),
    P(-96.2,77.0, -95.6,77.4, -94.8,77.3, -94.6,76.9, -95.3,76.7),
    P(-89.0,76.4, -88.4,76.8, -87.7,76.6, -87.6,76.1, -88.4,76.0),
]
def make_matrix():
    rows = []
    for j in range(61):
        lat = 79.0 - 0.1 * j
        row = []
        for i in range(241):
            lon = -105.0 + 0.1 * i
            row.append(1 if any(inside(lon, lat, q) for q in ISLANDS) else 0)
        rows.append(row)
    return rows

def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    matrix = make_matrix()
    assert len(matrix) == 61 and all(len(r) == 241 for r in matrix)
    assert all(v in (0, 1) for r in matrix for v in r)
    with (out / "candidate.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerows(matrix)
    # Pillow receives a direct binary conversion: land is white, water black.
    a = np.asarray(matrix, dtype=np.uint8) * 255
    im = Image.fromarray(a, mode="L")
    im.save(out / "candidate.png", format="PNG")
    big = im.resize((241 * 8, 61 * 8), resample=Image.Resampling.NEAREST)
    big.save(out / "candidate_8x.png", format="PNG")
    assert im.size == (241, 61) and big.size == (1928, 488)
    assert set(im.getdata()).issubset({0, 255})
    print("rows=61 cols=241 values=14701")
    print("candidate.png", im.size, "candidate_8x.png", big.size)

if __name__ == "__main__":
    main()
