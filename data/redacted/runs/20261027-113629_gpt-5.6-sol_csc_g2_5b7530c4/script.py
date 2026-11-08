from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image

LAT_MAX, LAT_MIN, STEP = 23.0, 1.0, 0.2
LON_MIN, LON_MAX = 99.0, 127.0
lats = LAT_MAX - np.arange(111) * STEP
lons = LON_MIN + np.arange(141) * STEP
land = np.zeros((111, 141), dtype=np.uint8)

# Approximate hand-authored outlines; vertices are longitude, latitude.
POLYGONS = [
    # Southern China and the mainland Southeast Asian peninsula.
    [
        (99, 23), (117.2, 23), (116.6, 22.6), (115.8, 22.4),
        (115.1, 22.1), (114.7, 22.0), (114.2, 22.4), (113.8, 22.8),
        (113.3, 22.6), (113.0, 22.1), (112.4, 21.8), (111.7, 21.5),
        (111.0, 21.2), (110.5, 20.8), (109.9, 21.2), (109.4, 21.5),
        (108.3, 21.5), (107.5, 21.5), (107.4, 20.5), (106.8, 20.0),
        (106.5, 19.4), (106.0, 18.8), (105.8, 17.5), (105.5, 16.0),
        (105.1, 14.5), (104.8, 13.0), (104.4, 11.5), (104.0, 10.5),
        (103.2, 9.5), (102.2, 9.7), (101.4, 11.0), (100.8, 12.5),
        (100.0, 13.5), (99.5, 15.0), (99.0, 16.0), (99.0, 23.0),
    ],
    # Myanmar and western Thailand.
    [
        (99.0, 23.0), (101.0, 23.0), (100.5, 21.0), (100.2, 19.0),
        (99.8, 17.0), (99.5, 15.0), (99.2, 13.0), (98.9, 11.0),
        (99.4, 10.0), (100.0, 10.2), (100.2, 12.0), (100.8, 13.5),
        (101.5, 14.5), (102.0, 16.0), (102.2, 18.0), (102.5, 20.0),
        (103.0, 22.0), (103.0, 23.0),
    ],
    # Malay peninsula.
    [
        (100.0, 7.0), (100.8, 6.5), (101.5, 5.8), (102.0, 4.8),
        (103.0, 4.0), (103.5, 2.8), (104.0, 1.5), (103.2, 1.0),
        (101.8, 1.0), (101.2, 2.0), (100.8, 3.5), (100.2, 5.0),
    ],
]

# Island outlines within the sampling frame.
POLYGONS.extend([
    # Sumatra and Java (only their northern edge reaches this frame).
    [(95.0, 5.8), (97.0, 5.5), (99.0, 3.8), (101.0, 2.5),
     (103.0, 0.5), (102.0, -1.0), (100.0, -3.0), (98.5, -5.0),
     (97.0, -5.8), (95.5, -4.0), (95.0, -1.0)],
    [(105.0, -5.8), (107.0, -6.0), (109.0, -6.5), (111.0, -7.5),
     (114.0, -8.5), (114.5, -7.5), (112.0, -7.0), (109.0, -6.0),
     (106.0, -5.5)],
    # Borneo.
    [(108.0, 7.0), (110.0, 7.0), (112.5, 5.0), (115.0, 4.5),
     (117.0, 2.0), (117.5, 0.5), (115.0, -2.0), (113.0, -3.0),
     (110.0, -2.0), (109.0, 0.0), (108.0, 2.0)],
    # Taiwan and major Philippine islands.
    [(120.0, 25.5), (121.8, 25.2), (122.0, 24.0), (121.5, 22.0),
     (120.8, 21.8), (120.0, 23.0), (119.8, 24.5)],
    [(119.7, 18.5), (121.0, 18.6), (122.0, 17.5), (121.8, 16.0),
     (121.2, 15.0), (121.5, 13.5), (120.5, 12.0), (119.5, 13.0),
     (119.0, 15.0), (119.2, 17.0)],
    [(120.3, 13.5), (121.2, 13.5), (121.0, 12.3), (120.2, 12.2)],
    [(117.0, 11.5), (118.5, 11.2), (119.5, 10.0), (118.0, 8.5),
     (117.0, 8.8), (116.8, 10.2)],
    [(124.0, 12.5), (125.2, 12.0), (125.5, 11.0), (124.5, 10.5)],
    [(124.8, 11.5), (125.5, 11.2), (125.3, 10.0), (124.7, 10.0)],
    [(123.2, 11.5), (124.0, 11.3), (124.0, 10.0), (123.3, 10.0)],
    [(122.0, 10.8), (123.0, 10.7), (123.2, 9.5), (122.2, 9.5)],
    [(121.0, 11.8), (122.0, 11.5), (122.5, 10.5), (121.5, 10.0)],
    [(123.0, 9.8), (125.0, 9.5), (126.0, 8.0), (125.5, 6.5),
     (124.0, 5.5), (123.0, 6.2), (122.5, 7.8)],
    # Hainan, Natuna, Anambas, Paracel and Spratly island groups.
    [(108.6, 20.2), (110.8, 20.2), (111.0, 19.2), (110.2, 18.2),
     (109.2, 18.2), (108.5, 19.2)],
    [(108.0, 4.0), (109.0, 4.0), (109.0, 3.0), (108.0, 3.0)],
    [(105.5, 3.5), (106.5, 3.5), (106.5, 2.5), (105.5, 2.5)],
    [(111.0, 12.0), (112.0, 12.0), (112.0, 11.0), (111.0, 11.0)],
    [(113.5, 16.8), (114.5, 16.8), (114.5, 16.0), (113.5, 16.0)],
])

def point_in_polygon(lon, lat, polygon):
    """Ray casting with boundary points assigned to land."""
    inside = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
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

# Small island/reef approximations in longitude, latitude degrees.
ELLIPSES = [
    (114.3, 10.5, 0.18, 0.12), (115.0, 10.0, 0.16, 0.11),
    (116.0, 10.8, 0.15, 0.10), (112.9, 8.7, 0.14, 0.10),
    (114.0, 9.0, 0.12, 0.09), (115.8, 9.2, 0.12, 0.09),
    (116.7, 10.4, 0.12, 0.09), (113.8, 16.5, 0.22, 0.13),
    (100.0, 9.5, 0.12, 0.25), (99.3, 7.5, 0.10, 0.30),
    (121.5, 18.0, 0.25, 0.15), (126.0, 7.0, 0.15, 0.20),
    (119.5, 5.5, 0.20, 0.14), (118.0, 1.5, 0.18, 0.14),
]

for row, lat in enumerate(lats):
    for col, lon in enumerate(lons):
        x, y = float(lon), float(lat)
        if any(point_in_polygon(x, y, polygon) for polygon in POLYGONS):
            land[row, col] = 1
            continue
        for cx, cy, rx, ry in ELLIPSES:
            dx = (x - cx) * math.cos(math.radians(cy))
            if (dx / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1:
                land[row, col] = 1
                break

out_dir = Path("outputs")
out_dir.mkdir(parents=True, exist_ok=True)
with (out_dir / "candidate.csv").open("w", newline="", encoding="ascii") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(land.tolist())

image = Image.fromarray(land * 255, mode="L")
image.save(out_dir / "candidate.png")
image.resize((1128, 888), resample=Image.Resampling.NEAREST).save(
    out_dir / "candidate_8x.png"
)
