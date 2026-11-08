from pathlib import Path
import csv
from PIL import Image

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# Coordinates are (longitude, latitude).  These are deliberately hand-drawn
# envelopes of the principal islands, not imported GIS data.
LAND = [
    # Victoria Island
    [(-109,73.0),(-108,75.0),(-106,75.7),(-103,75.9),(-101,75.7),
     (-99,76.0),(-97,75.8),(-96.5,74.5),(-97.2,73.5),(-99.0,72.7),
     (-102,72.55),(-105,72.75)],
    # Prince of Wales Island
    [(-103.8,74.0),(-103.0,75.35),(-101.0,76.0),(-98.0,76.25),
     (-96.3,75.55),(-96.0,74.1),(-97.0,73.25),(-99.2,72.75),
     (-101.8,72.75),(-103.5,73.15)],
    # western and central Melville Island
    [(-107.0,75.25),(-106.2,76.35),(-103.5,77.15),(-100.0,77.65),
     (-96.5,77.55),(-94.0,77.0),(-92.0,76.55),(-91.3,75.45),
     (-93.0,74.95),(-96.5,74.7),(-100.0,74.85),(-103.2,74.8)],
    # eastern Melville Island
    [(-94.0,75.1),(-92.4,76.0),(-90.0,76.65),(-88.8,76.15),
     (-89.2,75.25),(-90.8,74.7),(-92.8,74.65)],
    # Bathurst Island
    [(-98.0,76.1),(-97.0,77.35),(-94.8,78.15),(-91.8,78.55),
     (-89.7,78.15),(-89.0,77.15),(-90.0,76.4),(-92.0,76.0),
     (-94.5,75.7),(-96.8,75.75)],
    # Cornwallis Island
    [(-96.0,74.9),(-95.2,76.0),(-93.3,76.45),(-90.7,76.25),
     (-89.7,75.55),(-90.5,74.8),(-92.8,74.55),(-94.7,74.55)],
    # Somerset Island
    [(-95.0,72.9),(-94.8,74.2),(-93.7,74.85),(-91.7,74.85),
     (-90.0,74.2),(-89.0,73.35),(-89.6,72.7),(-92.0,72.35),
     (-94.0,72.45)],
    # Boothia Peninsula (only its northern part is in the frame)
    [(-94.8,73.0),(-94.0,73.75),(-92.3,74.0),(-89.7,73.75),
     (-87.8,73.05),(-87.4,72.35),(-88.4,71.6),(-91.5,71.9),
     (-94.0,72.25)],
    # Devon Island
    [(-91.5,74.55),(-90.7,76.0),(-90.2,77.5),(-88.8,78.75),
     (-85.8,79.45),(-82.8,79.35),(-81.0,78.6),(-80.3,77.0),
     (-81.2,75.55),(-83.8,74.8),(-87.0,74.45),(-89.5,74.35)],
    # western edge of Ellesmere Island
    [(-82.0,76.35),(-81.1,77.0),(-80.2,77.9),(-79.8,79.0),
     (-79.0,79.5),(-78.0,79.7),(-78.0,79.0),(-80.0,76.5)],
]
# Narrow channels and large bays are subtracted after the island envelopes.
WATER = [
    # M'Clintock Channel between Victoria and Prince of Wales
    [(-101.6,76.2),(-99.8,76.45),(-97.3,76.15),(-96.4,75.35),
     (-97.0,74.3),(-98.0,73.35),(-99.7,72.55),(-101.2,72.65),
     (-100.0,73.2),(-99.1,74.2),(-99.2,75.1)],
    # Prince of Wales Strait and its southern opening
    [(-97.0,75.8),(-95.8,75.75),(-95.4,74.6),(-96.2,73.7),
     (-97.0,73.2),(-97.6,73.55),(-96.9,74.5)],
    # channel north of Melville / Bathurst
    [(-98.5,77.7),(-97.0,78.0),(-95.0,78.25),(-93.2,78.4),
     (-92.4,77.8),(-93.4,77.35),(-95.0,77.05),(-97.1,77.15)],
    # Byam Channel / western side of Bathurst
    [(-96.4,76.0),(-95.0,76.5),(-93.8,76.65),(-92.4,76.25),
     (-92.8,75.75),(-94.2,75.55),(-95.8,75.55)],
    # Barrow Strait
    [(-95.0,74.4),(-94.0,74.7),(-92.0,74.45),(-90.2,74.1),
     (-89.5,73.6),(-90.0,73.25),(-92.2,73.65),(-94.0,73.7)],
    # Lancaster Sound
    [(-90.5,74.7),(-88.0,74.6),(-85.0,74.8),(-82.0,75.3),
     (-80.8,76.0),(-81.0,76.8),(-83.0,76.3),(-85.8,75.9),
     (-88.7,75.7),(-90.5,75.5)],
    # Wellington Channel
    [(-91.0,76.4),(-90.0,76.6),(-89.2,77.0),(-89.7,77.5),
     (-90.8,77.5),(-91.5,77.0)],
    # Viscount Melville Sound indentation
    [(-106.0,75.0),(-104.0,75.0),(-102.0,75.15),(-100.0,74.9),
     (-99.0,74.35),(-100.5,74.05),(-103.0,74.2),(-105.5,74.45)],
]

# Distinct small islands and exposed rocks which are large enough to survive
# a tenth-degree sampling grid.
LAND += [
    # Banks Island's southeast corner / shore islands
    [(-105.0,75.5),(-104.5,75.8),(-103.8,75.7),(-103.6,75.35),
     (-104.2,75.15),(-104.8,75.2)],
    # Prince Leopold Island
    [(-90.25,74.0),(-90.0,74.18),(-89.7,74.12),(-89.62,73.9),
     (-89.9,73.8)],
    # Russell Island
    [(-96.1,76.95),(-95.7,77.25),(-95.2,77.2),(-95.0,76.85),
     (-95.4,76.65)],
    # Lowther and Young islands
    [(-95.0,75.0),(-94.7,75.28),(-94.35,75.25),(-94.25,74.95),
     (-94.6,74.8)],
    [(-92.1,75.3),(-91.8,75.52),(-91.45,75.45),(-91.38,75.18),
     (-91.7,75.05)],
    # small islands in the western sounds
    [(-104.2,76.45),(-103.8,76.7),(-103.4,76.6),(-103.3,76.3),
     (-103.7,76.15)],
    [(-101.0,77.0),(-100.75,77.25),(-100.35,77.2),(-100.2,76.92),
     (-100.55,76.82)],
    [(-92.5,77.9),(-92.2,78.15),(-91.8,78.1),(-91.65,77.82),
     (-92.0,77.7)],
]
def inside(x, y, poly):
    """Ray crossing test; boundary points count as land."""
    hit = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) < 1e-10 and min(x1, x2) <= x <= max(x1, x2) and \
                min(y1, y2) <= y <= max(y1, y2):
            return True
        if (y1 > y) != (y2 > y):
            xx = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xx:
                hit = not hit
    return hit


def is_land(lon, lat):
    # The union of the hand-written island outlines, followed by explicit
    # water cuts for sounds, straits, and the largest embayments.
    land = any(inside(lon, lat, p) for p in LAND)
    if land and any(inside(lon, lat, p) for p in WATER):
        return 0
    return int(land)


def make_matrix():
    rows = []
    for r in range(61):
        lat = 79.0 - 0.1 * r
        row = []
        for c in range(241):
            lon = -105.0 + 0.1 * c
            row.append(is_land(lon, lat))
        rows.append(row)
    return rows


matrix = make_matrix()
assert len(matrix) == 61 and all(len(row) == 241 for row in matrix)
assert sum(len(row) for row in matrix) == 14701
assert all(v in (0, 1) for row in matrix for v in row)

with (OUT / "candidate.csv").open("w", newline="") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(matrix)

raw = Image.new("1", (241, 61))
pix = raw.load()
for y, row in enumerate(matrix):
    for x, value in enumerate(row):
        pix[x, y] = 1 if value else 0
raw.save(OUT / "candidate.png")

large = raw.resize((241 * 8, 61 * 8), Image.Resampling.NEAREST)
large.save(OUT / "candidate_8x.png")

print("generated", len(matrix), "x", len(matrix[0]),
      "values:", sum(sum(r) for r in matrix))
