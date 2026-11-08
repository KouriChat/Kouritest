from pathlib import Path
import csv
import numpy as np
from PIL import Image

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

LAT_COUNT = 61
LON_COUNT = 241


def inside(x, y, poly):
    """Inclusive-enough ray crossing test for a hand-drawn coastline."""
    hit = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        crosses = ((yi > y) != (yj > y))
        if crosses:
            at_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < at_x:
                hit = not hit
        j = i
    return hit


def ellipse(x, y, cx, cy, rx, ry):
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


# Coordinates are (west/east longitude, latitude), in degrees.
# These are deliberately hand-estimated outlines, not imported geography.
LAND_POLYGONS = [
    # Western and central Melville Island.
    [
        (-105.1, 76.00), (-104.5, 76.55), (-103.2, 77.05),
        (-102.8, 77.75), (-101.2, 78.15), (-98.5, 78.45),
        (-95.7, 78.65), (-92.7, 78.56), (-90.9, 78.10),
        (-90.1, 77.50), (-91.2, 77.04), (-93.2, 76.82),
        (-95.1, 76.32), (-97.5, 75.86), (-100.1, 75.55),
        (-102.6, 75.57), (-104.3, 75.72),
    ],
    # Bathurst Island.
    [
        (-102.7, 75.02), (-102.3, 75.64), (-101.0, 76.22),
        (-99.1, 76.66), (-97.0, 76.67), (-95.5, 76.34),
        (-95.0, 75.78), (-95.5, 75.18), (-96.8, 74.77),
        (-99.0, 74.57), (-101.0, 74.60), (-102.35, 74.77),
    ],
    # Cornwallis Island.
    [
        (-96.25, 74.78), (-96.05, 75.45), (-95.55, 76.08),
        (-94.55, 76.40), (-93.18, 76.46), (-92.45, 76.15),
        (-92.15, 75.55), (-92.55, 75.05), (-93.45, 74.68),
        (-94.85, 74.57), (-95.75, 74.60),
    ],
    # Devon Island, southern and western mass.
    [
        (-93.10, 74.28), (-92.15, 74.72), (-91.30, 75.16),
        (-90.15, 75.52), (-88.65, 75.87), (-87.00, 76.18),
        (-84.85, 76.46), (-82.75, 76.48), (-81.72, 76.15),
        (-81.10, 75.62), (-81.25, 75.03), (-82.30, 74.65),
        (-84.10, 74.30), (-86.45, 74.02), (-88.80, 73.96),
        (-90.70, 74.04), (-92.20, 74.08),
    ],
]
LAND_POLYGONS.extend([
    # Axel Heiberg Island.
    [
        (-95.10, 76.55), (-94.65, 77.30), (-94.25, 78.05),
        (-93.55, 78.82), (-92.10, 79.35), (-90.55, 79.46),
        (-89.15, 79.13), (-88.50, 78.42), (-88.62, 77.65),
        (-89.35, 77.05), (-90.35, 76.66), (-91.90, 76.34),
        (-93.55, 76.25),
    ],
    # Ellesmere Island, including its western shelf in this frame.
    [
        (-89.05, 76.10), (-88.62, 76.82), (-87.72, 77.40),
        (-86.35, 77.82), (-85.42, 78.38), (-84.65, 79.02),
        (-83.50, 79.44), (-81.85, 79.55), (-80.85, 79.30),
        (-80.50, 78.52), (-80.62, 77.70), (-81.08, 76.92),
        (-81.70, 76.36), (-82.45, 75.95), (-83.80, 75.62),
        (-85.25, 75.48), (-86.55, 75.54), (-87.68, 75.70),
    ],
    # Prince of Wales Island.
    [
        (-101.90, 72.96), (-101.55, 73.75), (-100.75, 74.45),
        (-99.55, 75.02), (-98.00, 75.56), (-96.25, 76.03),
        (-94.50, 76.22), (-92.72, 76.03), (-91.25, 75.50),
        (-90.62, 74.80), (-90.80, 74.05), (-91.70, 73.46),
        (-93.05, 73.10), (-94.75, 72.82), (-96.70, 72.68),
        (-98.65, 72.61), (-100.35, 72.70),
    ],
    # Somerset Island.
    [
        (-95.15, 72.78), (-94.62, 73.48), (-93.70, 74.05),
        (-92.45, 74.38), (-90.90, 74.42), (-89.30, 74.22),
        (-87.90, 73.78), (-86.45, 73.45), (-85.20, 73.08),
        (-84.30, 72.64), (-84.58, 72.20), (-86.15, 72.02),
        (-88.05, 72.12), (-90.02, 72.34), (-91.80, 72.38),
        (-93.25, 72.40), (-94.45, 72.48),
    ],
    # Eastern Victoria Island fragment at the western edge of the frame.
    [
        (-105.10, 73.00), (-105.10, 75.04), (-104.35, 75.55),
        (-103.20, 75.92), (-101.85, 76.02), (-100.70, 75.72),
        (-99.85, 75.18), (-99.35, 74.58), (-99.05, 73.95),
        (-99.35, 73.42), (-100.25, 73.05), (-102.20, 72.82),
        (-104.05, 72.83),
    ],
    # Boothia's northern tip, barely entering the southern edge.
    [
        (-95.20, 73.00), (-94.90, 73.38), (-94.20, 73.58),
        (-93.45, 73.48), (-93.05, 73.15), (-93.20, 72.96),
    ],
    # Representative small islands and skerries in the channels.
    [
        (-103.20, 76.18), (-102.92, 76.38), (-102.58, 76.34),
        (-102.48, 76.08), (-102.72, 75.94), (-103.05, 75.98),
    ],
    [
        (-100.55, 76.62), (-100.20, 76.83), (-99.85, 76.78),
        (-99.72, 76.52), (-99.95, 76.35), (-100.34, 76.39),
    ],
    [
        (-98.15, 77.18), (-97.78, 77.40), (-97.40, 77.33),
        (-97.32, 77.08), (-97.60, 76.94), (-97.98, 76.98),
    ],
    [
        (-88.55, 74.72), (-88.15, 74.92), (-87.76, 74.82),
        (-87.68, 74.54), (-88.00, 74.39), (-88.38, 74.47),
    ],
])

def is_land(lon, lat):
    for polygon in LAND_POLYGONS:
        if inside(lon, lat, polygon):
            return 1
    # A few tiny rocks are represented as ellipses so their point coverage
    # remains visible at the requested 0.1 degree sampling interval.
    tiny = (
        (-103.72, 75.88, 0.10, 0.07),
        (-101.62, 76.96, 0.11, 0.08),
        (-96.82, 74.34, 0.09, 0.06),
        (-90.72, 75.93, 0.10, 0.07),
        (-86.82, 74.85, 0.08, 0.06),
        (-84.08, 75.85, 0.09, 0.07),
    )
    return int(any(ellipse(lon, lat, *item) for item in tiny))


def build_matrix():
    lats = [79.0 - 0.1 * i for i in range(LAT_COUNT)]
    lons = [-105.0 + 0.1 * j for j in range(LON_COUNT)]
    matrix = [[is_land(lon, lat) for lon in lons] for lat in lats]
    assert len(matrix) == 61
    assert all(len(row) == 241 for row in matrix)
    assert all(value in (0, 1) for row in matrix for value in row)
    return matrix


def write_outputs(matrix):
    with (OUT / "candidate.csv").open("w", newline="", encoding="ascii") as handle:
        csv.writer(handle, lineterminator="\n").writerows(matrix)

    # Pillow receives 0 for water and 255 for land, with no interpolation.
    pixels = np.asarray(matrix, dtype=np.uint8) * 255
    image = Image.fromarray(pixels, mode="L")
    image.save(OUT / "candidate.png")
    enlarged = image.resize((LON_COUNT * 8, LAT_COUNT * 8), Image.Resampling.NEAREST)
    enlarged.save(OUT / "candidate_8x.png")

    assert pixels.shape == (61, 241)
    assert set(np.unique(pixels).tolist()) <= {0, 255}
    assert image.size == (241, 61)
    assert enlarged.size == (1928, 488)
    assert np.array_equal(
        np.asarray(enlarged),
        np.repeat(np.repeat(pixels, 8, axis=0), 8, axis=1),
    )


if __name__ == "__main__":
    result = build_matrix()
    write_outputs(result)
    print("generated 61x241 matrix and two PNGs")
