"""Memory-based approximation of the land around Hecate Strait (WGS84)."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image, ImageDraw

N = 127
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# Vertices are (longitude, latitude). These are hand-estimated outlines,
# not derived from an external map or geographic data set.
shapes = [
    # Mainland, with the approaches to Portland Inlet and the Skeena estuary.
    """-129 55 -130.07 55 -130.12 54.95 -130.08 54.89
    -130.17 54.85 -130.19 54.79 -130.13 54.73 -130.20 54.66
    -130.15 54.59 -130.20 54.54 -130.31 54.49 -130.30 54.44
    -130.24 54.39 -130.26 54.34 -130.15 54.29 -130.11 54.23
    -130.03 54.20 -130.00 54.15 -129.85 54.09 -129.85 54.05
    -129.78 54.01 -129.74 53.94 -129.67 53.89 -129.61 53.83
    -129.54 53.76 -129.43 53.72 -129.40 53.64 -129.38 53.57
    -129.30 53.51 -129.30 53.45 -129.36 53.41 -129.30 53.35
    -129.28 53.29 -129.18 53.22 -129.16 53.15 -129.09 53.12
    -129.02 53.04 -129 53.03""",
    # Dundas group, separated from the mainland by Chatham Sound.
    """-130.96 54.59 -130.93 54.66 -130.87 54.69 -130.80 54.70
    -130.75 54.67 -130.72 54.62 -130.66 54.59 -130.63 54.53
    -130.67 54.48 -130.72 54.45 -130.81 54.47 -130.89 54.52""",
    """-130.72 54.74 -130.68 54.78 -130.58 54.79 -130.53 54.74
    -130.56 54.68 -130.62 54.66""",
    """-130.89 54.40 -130.84 54.45 -130.79 54.44 -130.74 54.39
    -130.77 54.33 -130.84 54.32""",
    # Kaien Island and the smaller islands off Prince Rupert.
    """-130.40 54.38 -130.36 54.43 -130.29 54.44 -130.25 54.38
    -130.28 54.31 -130.36 54.29 -130.41 54.33""",
    """-130.52 54.28 -130.46 54.32 -130.43 54.29 -130.46 54.24
    -130.53 54.22""",
    # Stephens Island, Porcher Island, Goschen and their neighbours.
    """-130.88 54.12 -130.81 54.18 -130.70 54.20 -130.62 54.15
    -130.58 54.09 -130.63 54.03 -130.72 54.01 -130.81 54.05""",
    """-130.58 54.08 -130.50 54.15 -130.40 54.15 -130.34 54.10
    -130.32 54.04 -130.37 54.00 -130.42 53.99 -130.43 53.95
    -130.38 53.91 -130.43 53.85 -130.49 53.82 -130.58 53.85
    -130.66 53.90 -130.65 53.98 -130.61 54.02""",
    """-130.80 53.97 -130.73 54.00 -130.67 53.98 -130.65 53.92
    -130.70 53.88 -130.76 53.89""",
    """-130.33 53.94 -130.27 53.98 -130.22 53.94 -130.20 53.90
    -130.25 53.85 -130.32 53.87""",
    # Banks Island: long, rough-edged western shore facing Hecate Strait.
    """-130.87 53.83 -130.80 53.86 -130.70 53.85 -130.64 53.80
    -130.60 53.73 -130.58 53.66 -130.58 53.59 -130.55 53.53
    -130.55 53.47 -130.49 53.42 -130.49 53.34 -130.54 53.29
    -130.53 53.23 -130.58 53.20 -130.67 53.22 -130.70 53.27
    -130.74 53.31 -130.74 53.38 -130.81 53.43 -130.82 53.50
    -130.87 53.55 -130.84 53.62 -130.89 53.69 -130.87 53.75""",
    # McCauley Island and Pitt Island, either side of Principe Channel.
    """-130.45 53.83 -130.37 53.86 -130.27 53.83 -130.23 53.76
    -130.29 53.71 -130.36 53.70 -130.44 53.74""",
    """-130.25 53.83 -130.16 53.86 -130.09 53.81 -130.06 53.74
    -130.04 53.67 -130.03 53.60 -129.99 53.55 -129.95 53.48
    -129.91 53.42 -129.89 53.35 -129.91 53.28 -129.96 53.24
    -130.04 53.27 -130.09 53.34 -130.11 53.42 -130.18 53.48
    -130.19 53.56 -130.25 53.62 -130.26 53.70 -130.30 53.76""",
    # Islands near Grenville Channel and the south-east corner of the box.
    """-129.89 53.85 -129.83 53.87 -129.76 53.83 -129.74 53.75
    -129.78 53.67 -129.85 53.67 -129.89 53.74""",
    """-129.72 53.48 -129.66 53.52 -129.59 53.50 -129.56 53.42
    -129.58 53.35 -129.64 53.29 -129.71 53.31 -129.75 53.39""",
    """-129.83 53.20 -129.76 53.24 -129.68 53.21 -129.63 53.13
    -129.65 53.04 -129.72 53.00 -129.80 53.04""",
    """-129.48 53.08 -129.43 53.13 -129.37 53.12 -129.34 53.06
    -129.36 53.00 -129.45 53.00""",
]

# Rasterize on a finer canvas, then inspect ONLY each exact WGS84 sample point.
# Subpixel offsets here simply ensure the polygon predicate tests sample centers.
SCALE = 1000
mask = Image.new("1", (2001, 2001), 0)
draw = ImageDraw.Draw(mask)
def xy(lon, lat):
    return (round((lon + 131) * SCALE), round((55 - lat) * SCALE))
for text in shapes:
    vals = [float(v) for v in text.split()]
    draw.polygon([xy(vals[i], vals[i + 1])
                  for i in range(0, len(vals), 2)], fill=1)

# Hand-estimated narrow sea channels / inlet mouths cutting through mainland.
water_cuts = [
    """-130.31 54.50 -130.25 54.52 -130.16 54.58 -130.05 54.69
    -129.96 54.75 -130.00 54.68 -130.10 54.53 -130.19 54.46
    -130.27 54.43""",
    """-130.25 54.30 -130.17 54.29 -130.07 54.26 -129.97 54.25
    -129.87 54.31 -129.82 54.28 -129.91 54.19 -130.04 54.19
    -130.14 54.21""",
]
for text in water_cuts:
    vals = [float(v) for v in text.split()]
    draw.polygon([xy(vals[i], vals[i + 1])
                  for i in range(0, len(vals), 2)], fill=0)

# A few small reefs and islets in the otherwise open sound.
for lon, lat, rx, ry in [
    (-130.95, 54.45, .018, .015), (-130.98, 54.26, .017, .028),
    (-130.70, 54.27, .020, .023), (-130.54, 54.44, .021, .026),
    (-130.93, 54.02, .018, .021), (-130.95, 53.71, .023, .023),
    (-130.94, 53.46, .015, .021), (-130.64, 53.16, .023, .016),
    (-130.42, 53.57, .021, .035), (-130.31, 53.52, .017, .025),
    (-129.85, 53.31, .021, .022), (-129.51, 53.24, .022, .024),
]:
    draw.ellipse([xy(lon-rx, lat+ry), xy(lon+rx, lat-ry)], fill=1)

pixels = mask.load()
grid = np.array([[int(pixels[round(j * 2000 / 126),
                               round(i * 2000 / 126)])
                  for j in range(N)] for i in range(N)], dtype=np.uint8)
with (OUT / "candidate.csv").open("w", newline="", encoding="ascii") as file:
    csv.writer(file).writerows(grid.tolist())
image = Image.fromarray(grid * 255, mode="L")
image.save(OUT / "candidate.png")
large = image.resize((1016, 1016), Image.Resampling.NEAREST)
large.save(OUT / "candidate_8x.png")
assert grid.shape == (127, 127) and set(np.unique(grid)).issubset({0, 1})
assert set(image.getdata()).issubset({0, 255})
assert np.array_equal(np.asarray(large),
                      np.repeat(np.repeat(np.asarray(image), 8, axis=0), 8, axis=1))
