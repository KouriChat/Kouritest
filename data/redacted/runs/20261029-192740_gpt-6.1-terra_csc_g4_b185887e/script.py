"""Knowledge-based reconstruction of the South China Sea at 0.2 degree points.
All shapes below are hand-entered approximations; no geographic data are read.
"""
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 141, 111
LONS = [99.0 + 0.2 * i for i in range(W)]
LATS = [23.0 - 0.2 * i for i in range(H)]


def inside(x, y, poly):
    """Even-odd point-in-polygon test, including a conservative edge rule."""
    hit = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        cross = (x - xi) * (yj - yi) - (y - yi) * (xj - xi)
        if abs(cross) < 1e-10 and min(xi, xj) - 1e-10 <= x <= max(xi, xj) + 1e-10 and min(yi, yj) - 1e-10 <= y <= max(yi, yj) + 1e-10:
            return True
        if (yi > y) != (yj > y):
            at_x = xi + (xj - xi) * (y - yi) / (yj - yi)
            if x < at_x:
                hit = not hit
        j = i
    return hit


def ellipse(x, y, cx, cy, rx, ry, angle=0.0):
    a = math.radians(angle)
    dx, dy = x - cx, y - cy
    u = dx * math.cos(a) + dy * math.sin(a)
    v = -dx * math.sin(a) + dy * math.cos(a)
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1.0


# Mainland Asia, followed southward along the Gulf of Tonkin, Vietnam,
# Cambodia, Thailand and the Malay peninsula.  The western edge is outside
# the requested frame, so it is deliberately represented by the frame edge.
MAINLAND = [
    (99.0, 23.0), (115.0, 23.0), (114.8, 22.7), (114.3, 22.4),
    (113.6, 22.1), (112.8, 21.6), (111.7, 21.2), (110.8, 20.8),
    (110.4, 20.1), (109.6, 19.0), (108.7, 17.7), (107.8, 16.3),
    (107.0, 15.0), (106.1, 13.8), (105.0, 12.5), (104.0, 11.5),
    (103.7, 10.3), (103.8, 9.0), (104.0, 7.6), (104.0, 6.3),
    (103.7, 5.0), (103.8, 3.8), (104.0, 2.6), (104.0, 1.35),
    (103.75, 1.15), (99.0, 1.0),
]

# Hainan and a few larger coastal islands.
HAINAN = [(108.6, 19.35), (109.0, 20.05), (110.0, 20.15),
          (110.9, 19.75), (111.0, 19.0), (110.6, 18.25),
          (109.6, 18.05), (108.8, 18.45)]
LEIZHOU = [(109.4, 20.25), (110.9, 20.65), (111.5, 20.4),
           (111.0, 19.9), (109.9, 19.9)]


def add_polyland(x, y):
    return inside(x, y, MAINLAND) or inside(x, y, HAINAN) or inside(x, y, LEIZHOU)
# Major islands and archipelagos in the frame.
ISLANDS = [
    # Taiwan
    [(120.0, 21.9), (120.1, 24.8), (120.8, 25.4), (121.6, 25.1),
     (122.0, 24.2), (121.8, 23.1), (121.3, 22.0), (120.6, 21.7)],
    # Borneo (the southern and western portions are within the frame)
    [(109.0, 1.0), (114.0, 1.0), (117.8, 1.1), (118.8, 2.2),
     (119.1, 3.6), (118.8, 4.8), (118.2, 5.8), (117.5, 6.7),
     (116.8, 7.4), (115.4, 7.5), (113.8, 7.0), (112.3, 6.4),
     (111.0, 5.5), (110.0, 4.1), (109.3, 2.5)],
    # Sumatra, clipped naturally by the frame at its southern end
    [(99.0, 5.9), (100.0, 5.8), (101.4, 5.5), (103.0, 5.0),
     (104.4, 4.3), (105.3, 3.1), (105.7, 1.0), (100.5, 1.0),
     (99.8, 2.0), (99.3, 3.5)],
    # Luzon and its long eastern/western lobes
    [(119.3, 18.0), (119.7, 19.1), (120.5, 19.7), (121.5, 19.5),
     (122.3, 18.5), (122.1, 17.1), (121.5, 16.0), (121.0, 14.7),
     (120.3, 13.3), (119.6, 13.0), (119.1, 14.2), (119.4, 16.0)],
    # Mindoro, Palawan and the Calamian chain
    [(120.0, 13.1), (120.2, 13.7), (121.3, 13.6), (121.7, 12.8),
     (121.3, 12.2), (120.5, 12.1), (120.0, 12.5)],
    [(117.0, 11.8), (117.2, 10.7), (117.5, 9.5), (117.8, 8.3),
     (118.2, 7.2), (118.6, 6.8), (119.0, 7.3), (119.3, 8.8),
     (119.7, 10.2), (119.8, 11.3), (119.3, 12.0), (118.0, 12.2)],
    # Visayas: Panay, Negros, Cebu, Bohol, Leyte and Samar
    [(121.5, 11.8), (122.5, 11.9), (123.1, 11.1), (122.8, 10.3),
     (121.8, 10.2), (121.3, 10.8)],
    [(122.3, 10.5), (123.0, 10.8), (123.4, 10.0), (123.3, 9.2),
     (122.7, 9.1), (122.3, 9.7)],
    [(123.6, 11.2), (124.3, 11.7), (125.0, 11.2), (125.1, 10.0),
     (124.7, 9.1), (124.0, 9.4), (123.6, 10.3)],
    [(124.5, 12.4), (125.1, 12.0), (125.5, 11.0), (125.2, 10.0),
     (124.7, 10.2), (124.5, 11.2)],
    # Mindanao
    [(122.8, 9.8), (123.5, 9.5), (124.5, 9.3), (125.5, 8.6),
     (126.5, 7.5), (126.4, 6.2), (125.5, 5.3), (124.2, 5.0),
     (123.0, 5.5), (122.3, 6.6), (122.1, 8.0)],
    # Basilan and the Sulu archipelago, represented by island-sized polygons
    [(121.7, 6.9), (122.4, 7.1), (122.8, 6.6), (122.3, 6.2), (121.8, 6.4)],
    [(120.8, 6.2), (121.2, 6.4), (121.4, 6.0), (121.1, 5.7), (120.8, 5.9)],
    [(120.0, 5.1), (120.5, 5.3), (120.6, 4.9), (120.2, 4.7)],
    # Bangka and Belitung east of Sumatra
    [(105.4, 2.8), (106.0, 3.4), (106.8, 3.0), (106.9, 2.2),
     (106.3, 1.7), (105.6, 2.0)],
    [(107.0, 2.9), (107.8, 3.1), (108.2, 2.5), (107.9, 2.0),
     (107.1, 2.1)],
]


def add_major_islands(x, y):
    return any(inside(x, y, p) for p in ISLANDS)
# Small islands, reef cays and exposed reef tops.  Their radii are intentionally
# approximate and are only used to decide the sampled point, not to draw a map.
SMALL = [
    # Gulf of Tonkin and China's southern coast
    (107.9, 20.7, .10, .07, 0), (108.3, 20.5, .07, .05, 0),
    (109.3, 20.1, .08, .06, 0), (110.2, 20.5, .10, .06, 20),
    (113.9, 22.3, .08, .06, 0), (114.3, 22.2, .06, .05, 0),
    # Vietnam's nearshore islands and Phu Quoc
    (106.7, 20.7, .09, .06, 0), (105.9, 19.8, .07, .05, 0),
    (105.0, 17.0, .07, .05, 0), (104.0, 10.3, .20, .13, 20),
    (103.9, 10.0, .06, .05, 0),
    # Gulf of Thailand and Cambodia
    (102.6, 10.3, .08, .06, 0), (103.2, 10.0, .09, .06, 0),
    (102.5, 9.7, .07, .05, 0), (99.8, 7.2, .08, .05, 0),
    # Malay east coast islands
    (103.3, 5.8, .12, .07, 15), (103.4, 5.0, .08, .05, 0),
    (103.9, 4.0, .08, .05, 0), (104.3, 2.8, .10, .06, 0),
    # Riau, Natuna and Anambas groups
    (104.1, 1.1, .16, .10, 0), (104.5, 1.3, .18, .11, 0),
    (105.8, 2.9, .10, .08, 0), (106.2, 3.9, .22, .16, 20),
    (108.0, 3.9, .35, .25, 10), (108.4, 4.0, .10, .07, 0),
    (105.7, 3.0, .09, .06, 0), (106.0, 2.5, .08, .06, 0),
    # Paracel Islands
    (111.2, 15.8, .10, .07, 0), (111.7, 16.45, .12, .09, 0),
    (111.8, 16.53, .10, .07, 0), (112.3, 16.83, .14, .10, 0),
    (112.7, 16.55, .09, .06, 0), (112.1, 16.2, .08, .06, 0),
    # Pratas and the Macclesfield shoal's small exposed portions
    (116.72, 20.70, .13, .08, 0), (114.0, 16.0, .06, .04, 0),
    # Spratly islands and cays
    (112.3, 11.1, .10, .07, 0), (113.0, 10.7, .12, .08, 0),
    (114.3, 11.05, .13, .09, 0), (114.1, 10.0, .15, .10, 0),
    (114.4, 9.9, .13, .09, 0), (114.25, 9.7, .11, .08, 0),
    (114.2, 10.2, .10, .07, 0), (112.9, 9.55, .12, .08, 0),
    (113.8, 7.4, .13, .09, 0), (112.6, 8.65, .08, .06, 0),
    (115.0, 8.0, .08, .05, 0), (115.8, 9.0, .07, .05, 0),
    # Philippines outer islands
    (118.3, 20.4, .10, .07, 0), (121.8, 20.7, .10, .06, 0),
    (122.0, 10.8, .08, .06, 0), (120.0, 11.1, .09, .06, 0),
    (121.5, 9.0, .08, .05, 0), (125.3, 7.0, .12, .08, 0),
    (126.0, 9.0, .08, .06, 0), (126.5, 10.2, .08, .06, 0),
]

# The largest inland lake intersecting this reconstruction.  A point in it is
# water even though it lies inside the deliberately generalized mainland.
LAKES = [(104.05, 12.72, .28, .16, 0)]  # Tonle Sap


def add_small_land(x, y):
    return any(ellipse(x, y, *s) for s in SMALL)


def is_land(x, y):
    land = add_polyland(x, y) or add_major_islands(x, y) or add_small_land(x, y)
    if land and any(ellipse(x, y, *lake) for lake in LAKES):
        return False
    return land


def build_matrix():
    return np.array([[1 if is_land(lon, lat) else 0 for lon in LONS]
                     for lat in LATS], dtype=np.uint8)


def write_outputs(a):
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerows(a.tolist())
    # Pillow's L mode is an exact two-colour, one-byte-per-pixel image.
    raw = (a * 255).astype(np.uint8)
    Image.fromarray(raw, mode="L").save(out / "candidate.png")
    Image.fromarray(raw, mode="L").resize((W * 8, H * 8), Image.Resampling.NEAREST).save(
        out / "candidate_8x.png")


if __name__ == "__main__":
    matrix = build_matrix()
    assert matrix.shape == (111, 141)
    assert set(np.unique(matrix)) <= {0, 1}
    write_outputs(matrix)
