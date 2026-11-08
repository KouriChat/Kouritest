import os
import csv
import numpy as np
from PIL import Image

# Hand-drawn coastline approximation. Coordinates are (longitude, latitude).
# The polygons are deliberately clipped only by the requested sampling window
# later; they continue a little beyond it where that makes a cleaner coastline.
LAND = []

def poly(*points):
    LAND.append([(float(x), float(y)) for x, y in points])

# Southern Ellesmere Island (the remainder lies north/east of the window).
poly(
    (-92.2, 76.05), (-89.8, 75.95), (-87.1, 76.05),
    (-84.8, 76.25), (-82.3, 76.65), (-80.0, 77.35),
    (-80.0, 79.5), (-92.0, 79.5), (-93.0, 78.5),
    (-93.0, 77.35)
)

# Axel Heiberg Island, separated from Ellesmere by a narrow sound.
poly(
    (-98.4, 77.05), (-97.4, 77.55), (-96.3, 78.25),
    (-94.8, 78.75), (-92.6, 79.25), (-90.0, 79.5),
    (-89.8, 80.2), (-93.5, 80.3), (-96.7, 79.75),
    (-99.0, 78.9), (-99.4, 77.8)
)

# Amund Ringnes and Ellef Ringnes Islands.
poly(
    (-98.8, 77.25), (-97.6, 77.1), (-96.3, 77.25),
    (-95.0, 77.75), (-95.0, 78.55), (-96.2, 78.85),
    (-98.0, 78.55), (-99.0, 77.9)
)
poly(
    (-96.0, 76.9), (-94.2, 76.8), (-92.7, 77.25),
    (-92.4, 78.05), (-93.4, 78.55), (-95.1, 78.35),
    (-96.2, 77.75)
)

# Melville Island, including its broad western part in the window.
poly(
    (-105.2, 74.55), (-103.8, 75.05), (-103.0, 76.0),
    (-103.7, 77.0), (-102.0, 77.75), (-99.4, 78.15),
    (-96.8, 78.0), (-95.7, 77.25), (-96.2, 76.35),
    (-95.2, 75.65), (-93.4, 75.1), (-91.0, 74.75),
    (-91.0, 74.15), (-95.0, 74.0), (-99.0, 74.05),
    (-102.2, 74.05)
)

# Smaller islands along the Melville/Bathurst group.
poly(
    (-101.8, 75.55), (-100.7, 75.75), (-99.6, 75.45),
    (-99.5, 76.25), (-100.4, 76.75), (-101.7, 76.55)
)
poly(
    (-101.2, 74.15), (-100.2, 74.1), (-99.7, 74.55),
    (-100.1, 75.0), (-101.0, 74.85)
)

# Bathurst Island.
poly(
    (-100.6, 75.45), (-99.6, 75.25), (-98.0, 75.35),
    (-96.8, 75.95), (-96.7, 76.75), (-97.8, 77.35),
    (-99.3, 77.45), (-100.4, 76.8), (-101.0, 76.1)
)

# Cornwallis Island.
poly(
    (-96.2, 75.25), (-94.8, 75.0), (-92.9, 75.2),
    (-91.2, 75.75), (-91.3, 76.45), (-92.5, 76.75),
    (-94.4, 76.65), (-95.8, 76.15)
)

# Devon Island, with its southern shore on the Jones/Lancaster sound region.
poly(
    (-96.0, 74.75), (-94.3, 74.15), (-92.0, 74.25),
    (-89.6, 74.45), (-87.2, 74.3), (-84.8, 74.55),
    (-82.8, 75.25), (-81.9, 76.0), (-82.2, 76.95),
    (-84.3, 77.55), (-87.0, 77.8), (-90.0, 77.65),
    (-92.3, 77.15), (-94.2, 76.55), (-95.4, 75.7)
)
# Prince of Wales Island.
poly(
    (-101.7, 72.15), (-101.5, 73.3), (-101.0, 74.35),
    (-100.1, 75.15), (-99.6, 76.0), (-98.0, 76.35),
    (-96.0, 76.2), (-94.5, 75.55), (-94.0, 74.55),
    (-94.5, 73.55), (-95.7, 72.85), (-97.2, 72.2),
    (-99.2, 71.85)
)

# Russell, Young and nearby small islands in the channels.
poly(
    (-102.3, 74.5), (-101.6, 74.25), (-100.9, 74.55),
    (-101.0, 75.15), (-101.8, 75.25), (-102.4, 74.95)
)
poly(
    (-97.0, 74.1), (-96.3, 73.95), (-95.8, 74.25),
    (-96.0, 74.75), (-96.7, 74.7), (-97.2, 74.4)
)

# Somerset Island.
poly(
    (-96.0, 72.25), (-95.0, 72.65), (-94.5, 73.45),
    (-94.0, 74.15), (-92.9, 74.7), (-91.0, 74.85),
    (-88.8, 74.7), (-86.9, 74.25), (-85.0, 74.15),
    (-83.0, 74.55), (-81.9, 74.0), (-82.1, 73.15),
    (-83.0, 72.75), (-84.9, 72.55), (-87.2, 72.1),
    (-89.5, 71.9), (-92.0, 71.95), (-94.2, 71.8)
)

# Boothia Peninsula, entering the frame at its northern end and continuing
# south beyond the lower edge.
poly(
    (-96.8, 73.0), (-96.0, 73.6), (-95.0, 74.0),
    (-93.5, 74.05), (-92.0, 73.65), (-90.6, 73.0),
    (-90.2, 72.0), (-90.0, 70.0), (-97.5, 70.0),
    (-97.5, 72.1)
)

# Northern mainland coast (the land continues south of the frame).
poly(
    (-90.8, 73.0), (-89.3, 73.45), (-87.2, 73.7),
    (-85.5, 73.45), (-84.2, 72.9), (-83.0, 72.75),
    (-82.4, 72.1), (-82.0, 70.0), (-91.0, 70.0)
)

# Western Baffin Island coast and Bylot Island touch the eastern edge.
poly(
    (-82.0, 73.0), (-81.4, 73.35), (-80.0, 73.55),
    (-80.0, 75.0), (-81.0, 75.15), (-82.0, 74.7),
    (-82.6, 74.0)
)
poly(
    (-81.8, 73.95), (-80.8, 74.0), (-79.8, 74.35),
    (-79.8, 75.35), (-80.8, 75.5), (-81.7, 75.0)
)

# Islands near the western mouth of Lancaster Sound.
poly(
    (-84.1, 74.75), (-83.4, 74.65), (-82.9, 74.95),
    (-83.2, 75.35), (-84.0, 75.3)
)
poly(
    (-88.1, 73.8), (-87.3, 73.55), (-86.6, 73.75),
    (-86.8, 74.15), (-87.7, 74.25)
)
def inside(x, y, vertices):
    """Odd-even point-in-polygon test, with boundary counted as land."""
    hit = False
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        # A small explicit boundary test avoids arbitrary holes along edges.
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) < 1e-10 and min(x1, x2) - 1e-10 <= x <= max(x1, x2) + 1e-10 and min(y1, y2) - 1e-10 <= y <= max(y1, y2) + 1e-10:
            return True
        if (y1 > y) != (y2 > y):
            at_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < at_x:
                hit = not hit
    return hit


def is_land(lon, lat):
    return int(any(inside(lon, lat, shape) for shape in LAND))


def make_matrix():
    # The requested order is north-to-south and west-to-east.
    lats = [79.0 - 0.1 * r for r in range(61)]
    lons = [-105.0 + 0.1 * c for c in range(241)]
    return np.array(
        [[is_land(lon, lat) for lon in lons] for lat in lats],
        dtype=np.uint8,
    )


def main():
    matrix = make_matrix()
    assert matrix.shape == (61, 241)
    assert matrix.size == 14701
    assert set(np.unique(matrix)).issubset({0, 1})

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/candidate.csv", "w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Pillow's L mode is exact: 0 is black water and 255 is white land.
    original = Image.fromarray(matrix * 255, mode="L")
    assert original.size == (241, 61)
    original.save("outputs/candidate.png")

    enlarged = original.resize((241 * 8, 61 * 8), resample=Image.Resampling.NEAREST)
    assert enlarged.size == (1928, 488)
    enlarged.save("outputs/candidate_8x.png")

    # Re-open the files so the checks also cover the serialized artifacts.
    with open("outputs/candidate.csv", encoding="ascii", newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 61 and all(len(row) == 241 for row in rows)
    assert all(v in {"0", "1"} for row in rows for v in row)
    assert set(original.getdata()).issubset({0, 255})
    assert enlarged.size == (1928, 488)


if __name__ == "__main__":
    main()
