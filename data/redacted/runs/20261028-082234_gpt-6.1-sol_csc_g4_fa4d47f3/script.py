from pathlib import Path
import math
import csv
import numpy as np
from PIL import Image

NROWS, NCOLS = 111, 141
LAT0, LON0, STEP = 23.0, 99.0, 0.2

def inside(x, y, poly):
    hit = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi) + xi
        ):
            hit = not hit
        j = i
    return hit

def ellipse(x, y, cx, cy, rx, ry):
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0

def land(x, y):
    # Mainland Asia, with a hand-written approximation of the coast.
    mainland = [
        (99.0, 23.0), (117.5, 23.0), (117.25, 22.6),
        (116.6, 22.1), (115.5, 21.6), (114.5, 21.1),
        (113.5, 20.8), (112.5, 20.5), (111.5, 20.0),
        (110.5, 20.1), (109.5, 20.8), (108.5, 21.5),
        (107.8, 21.6), (107.2, 21.2), (106.9, 20.7),
        (106.7, 20.0), (106.5, 19.4), (106.2, 18.7),
        (106.0, 17.8), (106.2, 16.8), (107.0, 15.6),
        (108.0, 14.2), (109.0, 12.5), (109.2, 11.2),
        (108.6, 10.3), (107.8, 9.4), (106.7, 8.7),
        (105.7, 8.3), (104.8, 8.5), (104.0, 9.2),
        (103.0, 10.0), (102.3, 10.7), (101.8, 9.0),
        (101.5, 7.5), (101.8, 6.0), (102.2, 5.2),
        (102.5, 4.0), (102.9, 3.0), (103.3, 2.0),
        (103.7, 1.0), (99.0, 1.0), (99.0, 8.0),
        (99.4, 10.0), (99.5, 12.0), (99.2, 14.5),
        (99.0, 17.0)
    ]
    if inside(x, y, mainland):
        return True

    # Hainan and Taiwan.
    hainan = [
        (108.6, 19.2), (109.2, 19.9), (110.3, 20.1),
        (111.5, 20.0), (112.3, 19.5), (112.8, 18.7),
        (112.1, 18.1), (110.7, 18.1), (109.5, 18.3)
    ]
    taiwan = [
        (120.0, 22.6), (120.0, 23.0), (121.0, 23.0),
        (121.6, 22.7), (121.9, 22.1), (121.5, 21.8),
        (120.8, 21.9), (120.3, 22.2)
    ]
    return inside(x, y, hainan) or inside(x, y, taiwan)
def extra_land(x, y):
    # Sumatra, Borneo, and the small islands around the Malay Archipelago.
    sumatra = [
        (99.1, 5.5), (100.0, 5.8), (101.0, 5.3), (102.0, 4.8),
        (103.0, 4.3), (104.0, 3.5), (104.2, 2.5), (103.7, 1.1),
        (102.5, 0.9), (101.0, 1.1), (99.8, 1.8), (99.2, 3.3)
    ]
    borneo = [
        (109.5, 1.0), (109.5, 2.2), (110.0, 3.8), (111.0, 5.0),
        (112.5, 6.0), (114.0, 6.8), (115.5, 7.3), (117.0, 7.0),
        (118.5, 6.2), (119.3, 5.0), (119.0, 3.5), (118.2, 2.0),
        (117.0, 1.0)
    ]
    sulawesi = [
        (119.3, 1.0), (119.5, 2.0), (120.0, 2.7), (121.0, 2.0),
        (121.8, 1.2), (122.8, 1.0), (123.5, 1.7), (123.3, 2.8),
        (122.5, 3.4), (121.5, 3.0), (120.7, 2.8), (120.0, 1.9)
    ]
    if inside(x, y, sumatra) or inside(x, y, borneo) or inside(x, y, sulawesi):
        return True

    # Palawan and the principal Philippine islands.
    palawan = [
        (117.0, 11.8), (117.6, 11.5), (118.2, 10.8), (118.5, 9.8),
        (118.3, 9.0), (117.7, 8.4), (117.3, 9.0), (116.9, 10.0),
        (116.8, 11.0)
    ]
    luzon = [
        (119.5, 18.5), (120.2, 19.0), (121.2, 18.5), (122.0, 17.7),
        (122.2, 16.6), (121.7, 15.4), (121.0, 14.8), (120.5, 15.5),
        (120.0, 16.2), (119.5, 17.2)
    ]
    mindanao = [
        (122.0, 9.8), (123.0, 9.5), (124.0, 8.7), (125.2, 8.2),
        (126.0, 7.3), (125.5, 6.4), (124.5, 6.0), (123.5, 6.3),
        (122.8, 7.0), (121.8, 7.3), (121.4, 8.2)
    ]
    visayas = [
        (121.0, 12.5), (121.8, 12.8), (122.5, 12.2), (123.0, 11.5),
        (123.8, 11.0), (124.2, 10.2), (123.7, 9.7), (122.8, 10.0),
        (122.0, 10.5), (121.2, 11.2)
    ]
    if (inside(x, y, palawan) or inside(x, y, luzon) or
            inside(x, y, mindanao) or inside(x, y, visayas)):
        return True
    return False

# Attach the second set of hand-written land rules to the mainland rule.
_land_base = land
def land(x, y):
    return _land_base(x, y) or extra_land(x, y)
def archipelago_land(x, y):
    islands = [
        # Gulf of Thailand, Vietnam, and the Natuna group.
        [(103.95, 10.0), (104.2, 10.3), (104.5, 10.1),
         (104.4, 9.8), (104.0, 9.7)],
        [(103.8, 9.8), (104.2, 9.7), (104.3, 9.3),
         (103.9, 9.2), (103.6, 9.5)],
        [(104.0, 1.0), (104.4, 1.2), (104.5, 1.6),
         (104.0, 1.7), (103.8, 1.3)],
        [(105.4, 2.4), (106.0, 2.7), (106.4, 2.4),
         (106.2, 2.0), (105.6, 2.0)],
        [(106.8, 3.5), (107.4, 3.8), (107.7, 3.4),
         (107.4, 3.0), (106.9, 3.1)],
        # Philippine islands too small to be represented by the four main masses.
        [(123.5, 13.2), (124.1, 13.7), (124.8, 13.2),
         (124.6, 12.4), (123.8, 12.4)],
        [(124.0, 11.8), (124.6, 12.0), (125.0, 11.5),
         (124.7, 10.9), (124.2, 11.0)],
        [(124.2, 10.0), (124.8, 10.4), (125.2, 10.0),
         (125.0, 9.4), (124.5, 9.4)],
        [(120.4, 11.3), (121.0, 11.7), (121.5, 11.3),
         (121.3, 10.8), (120.7, 10.8)],
        [(120.7, 10.0), (121.2, 10.3), (121.7, 10.0),
         (121.5, 9.5), (120.9, 9.5)],
        [(119.6, 14.0), (120.0, 14.5), (120.3, 14.1),
         (120.1, 13.6), (119.7, 13.6)],
    ]
    if any(inside(x, y, p) for p in islands):
        return True

    # Major islands and visible reef rocks.  The radii are deliberately small
    # so that the sampled point test remains a point test.
    spots = [
        (111.65, 16.83, .12, .08), (112.00, 16.55, .09, .06),
        (112.25, 16.45, .08, .05), (112.65, 16.55, .07, .05),
        (117.72, 15.13, .10, .07),  # Scarborough Shoal
        (114.36, 11.45, .10, .07), (115.00, 11.08, .08, .06),
        (114.30, 10.40, .08, .06), (114.22, 9.72, .08, .06),
        (112.90, 9.55, .10, .07), (115.82, 9.88, .08, .06),
        (114.08, 9.20, .08, .05), (111.88, 8.65, .08, .05),
        (112.33, 8.65, .07, .05), (115.25, 8.80, .08, .05),
        (108.25, 3.95, .13, .08), (108.65, 4.05, .09, .06),
        (105.75, 3.15, .10, .07), (106.15, 3.00, .08, .05),
        (109.20, 6.10, .07, .05), (108.95, 6.35, .07, .05),
        (109.30, 11.60, .08, .06), (109.35, 10.35, .08, .06),
        (109.25, 9.20, .07, .05), (116.75, 20.70, .08, .05),
        (114.30, 22.25, .08, .05), (113.95, 22.30, .08, .05),
    ]
    return any(ellipse(x, y, *s) for s in spots)

_land_with_archipelago = land
def land(x, y):
    return _land_with_archipelago(x, y) or archipelago_land(x, y)
def minor_islands(x, y):
    small = [
        # Taiwan Strait and Guangdong coast
        [(119.55, 22.65), (119.85, 22.85), (120.0, 22.6),
         (119.75, 22.4), (119.5, 22.45)],
        [(114.0, 22.2), (114.3, 22.35), (114.5, 22.1),
         (114.25, 21.95), (114.0, 22.0)],
        # Anambas, Natuna, Bangka, and Singapore
        [(105.4, 3.0), (105.9, 3.3), (106.3, 3.0),
         (106.0, 2.6), (105.5, 2.7)],
        [(106.8, 3.0), (107.3, 3.2), (107.6, 2.9),
         (107.3, 2.5), (106.9, 2.6)],
        [(104.0, 1.35), (104.3, 1.45), (104.45, 1.25),
         (104.2, 1.1), (104.0, 1.15)],
        [(103.8, 1.15), (104.1, 1.25), (104.2, 1.0),
         (103.9, 1.0)],
        # Samar, Leyte, Panay, Negros, Cebu, and Bohol
        [(124.0, 12.5), (124.8, 12.8), (125.4, 12.3),
         (125.3, 11.5), (124.6, 11.4), (124.0, 11.8)],
        [(124.0, 11.3), (124.5, 11.2), (124.7, 10.4),
         (124.3, 9.9), (123.8, 10.2)],
        [(122.0, 11.2), (122.7, 11.8), (123.3, 11.5),
         (123.1, 10.8), (122.5, 10.5)],
        [(122.4, 10.4), (122.9, 10.6), (123.2, 10.1),
         (122.8, 9.7), (122.4, 9.9)],
        [(123.6, 10.3), (124.0, 10.5), (124.2, 10.1),
         (123.8, 9.9)],
        [(123.8, 13.1), (124.4, 13.5), (124.7, 13.1),
         (124.4, 12.7), (123.9, 12.7)],
    ]
    return any(inside(x, y, p) for p in small)

_land_with_minor = land
def land(x, y):
    if ellipse(x, y, 104.0, 12.7, .35, .18):  # Tonle Sap
        return False
    return _land_with_minor(x, y) or minor_islands(x, y)

def build_matrix():
    result = np.zeros((NROWS, NCOLS), dtype=np.uint8)
    for r in range(NROWS):
        lat = LAT0 - r * STEP
        for c in range(NCOLS):
            lon = LON0 + c * STEP
            result[r, c] = 1 if land(lon, lat) else 0
    return result

def write_outputs(matrix):
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())
    original = Image.fromarray((matrix * 255).astype(np.uint8), mode="L")
    original.save(out / "candidate.png")
    enlarged = original.resize((NCOLS * 8, NROWS * 8), Image.Resampling.NEAREST)
    enlarged.save(out / "candidate_8x.png")

if __name__ == "__main__":
    matrix = build_matrix()
    assert matrix.shape == (111, 141)
    assert set(np.unique(matrix)).issubset({0, 1})
    write_outputs(matrix)
