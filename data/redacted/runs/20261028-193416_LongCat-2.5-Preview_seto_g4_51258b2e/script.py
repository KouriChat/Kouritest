





"""Hand-written approximate land/water rules for the Seto Inland Sea region."""
import csv
import math
import os
import numpy as np
from PIL import Image

W, H = 253, 64
LON0, LAT0, STEP = 129.0, 35.0, 2.0 / 63.0


def ellipse(x, y, c, rx, ry, rot=0.0):
    ca, sa = math.cos(rot), math.sin(rot)
    dx, dy = x - c[0], y - c[1]
    u, v = ca * dx + sa * dy, -sa * dx + ca * dy
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1.0


def point_in_poly(x, y, p):
    inside = False
    n = len(p)
    for k in range(n):
        x1, y1 = p[k]
        x2, y2 = p[(k + 1) % n]
        if (y1 > y) != (y2 > y):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xc:
                inside = not inside
    return inside


def dist_to_seg(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy + 1e-30)))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))

def in_kyushu(lon, lat):
    main = [
        (129.0, 35.0), (130.0, 34.15), (130.55, 34.05), (130.95, 33.95),
        (131.08, 33.86), (131.20, 33.75), (131.25, 33.60), (131.20, 33.40),
        (131.00, 33.00), (129.00, 33.00),
    ]
    if not point_in_poly(lon, lat, main):
        return False
    nagasaki = [(129.55, 34.65), (129.75, 34.75), (130.05, 34.45),
                (130.15, 34.20), (129.95, 34.05), (129.65, 34.20)]
    if point_in_poly(lon, lat, nagasaki):
        return True
    return point_in_poly(lon, lat, main) and lat < 34.15


def in_shikoku(lon, lat):
    p = [
        (133.05, 34.05), (133.35, 34.25), (133.80, 34.20),
        (134.25, 33.95), (134.55, 33.75), (134.70, 33.50),
        (134.55, 33.30), (134.10, 33.25), (133.55, 33.25),
        (133.10, 33.45), (132.90, 33.75),
    ]
    return point_in_poly(lon, lat, p)


def in_awaji(lon, lat):
    return ellipse(lon, lat, (134.90, 34.35), 0.22, 0.16, -0.5)

def in_honshu(lon, lat):
    coast = [
        (131.20, 33.75), (131.45, 34.05), (131.75, 34.10),
        (132.05, 34.25), (132.35, 34.30), (132.65, 34.35),
        (133.00, 34.45), (133.40, 34.55), (133.80, 34.60),
        (134.20, 34.65), (134.50, 34.70), (134.80, 34.65),
        (135.00, 34.75), (135.40, 34.80), (135.80, 34.90),
        (136.30, 34.80), (136.60, 34.60), (136.80, 34.70),
        (137.00, 35.00),
    ]
    if 131.20 <= lon <= 137.0:
        for k in range(len(coast) - 1):
            x1, y1 = coast[k]
            x2, y2 = coast[k + 1]
            if x1 <= lon <= x2 and lat >= y1 + (lon - x1) * (y2 - y1) / (x2 - x1):
                return True
    kii = [
        (134.80, 34.70), (135.15, 34.75), (135.50, 34.55),
        (135.80, 34.30), (136.00, 34.00), (136.10, 33.60),
        (135.90, 33.40), (135.60, 33.45), (135.30, 33.65),
        (135.00, 34.00), (134.80, 34.40),
    ]
    if point_in_poly(lon, lat, kii):
        return True
    ise = [(136.35, 35.00), (136.55, 34.85), (136.70, 34.65),
           (136.85, 34.70), (136.95, 35.00)]
    if point_in_poly(lon, lat, ise):
        return False
    if ellipse(lon, lat, (136.00, 35.15), 0.22, 0.18, 0.2):
        return True
    return False

def in_small_islands_west(lon, lat):
    specs = [
        ((129.10, 33.05), 0.20, 0.18, 0.3),
        ((129.00, 33.30), 0.12, 0.10, 0.0),
        ((129.55, 33.35), 0.12, 0.08, 0.4),
        ((129.75, 33.40), 0.10, 0.07, 0.0),
        ((129.90, 33.55), 0.08, 0.06, 0.0),
        ((130.10, 33.70), 0.07, 0.06, 0.0),
        ((130.30, 33.80), 0.06, 0.05, 0.0),
        ((131.80, 33.90), 0.12, 0.08, 0.5),
        ((132.20, 33.90), 0.15, 0.09, 0.0),
        ((132.30, 34.10), 0.10, 0.07, 0.0),
        ((132.40, 34.25), 0.08, 0.06, 0.0),
        ((132.55, 34.30), 0.07, 0.05, 0.0),
        ((132.70, 34.20), 0.08, 0.06, 0.0),
        ((132.90, 34.35), 0.06, 0.05, 0.0),
        ((133.00, 34.40), 0.06, 0.05, 0.0),
        ((133.10, 34.45), 0.06, 0.05, 0.0),
        ((133.20, 34.50), 0.06, 0.05, 0.0),
        ((133.30, 34.55), 0.05, 0.05, 0.0),
        ((133.40, 34.60), 0.05, 0.05, 0.0),
        ((133.50, 34.65), 0.05, 0.05, 0.0),
    ]
    return any(ellipse(lon, lat, c, rx, ry, r) for c, rx, ry, r in specs)

def in_small_islands_central(lon, lat):
    specs = [
        ((132.35, 34.30), 0.10, 0.08, 0.0),
        ((132.65, 34.25), 0.10, 0.07, 0.0),
        ((133.15, 34.35), 0.12, 0.08, 0.0),
        ((133.35, 34.30), 0.12, 0.08, 0.0),
        ((133.50, 34.25), 0.10, 0.08, 0.0),
        ((133.35, 34.40), 0.08, 0.07, 0.0),
        ((133.60, 34.35), 0.08, 0.07, 0.0),
        ((133.75, 34.45), 0.07, 0.06, 0.0),
        ((133.90, 34.55), 0.07, 0.06, 0.0),
        ((133.55, 34.55), 0.06, 0.06, 0.0),
        ((133.70, 34.65), 0.06, 0.06, 0.0),
        ((133.85, 34.70), 0.06, 0.06, 0.0),
        ((134.18, 34.35), 0.22, 0.16, 0.2),
        ((134.00, 34.45), 0.10, 0.08, 0.0),
        ((134.10, 34.50), 0.08, 0.07, 0.0),
        ((134.25, 34.55), 0.08, 0.07, 0.0),
        ((134.35, 34.60), 0.07, 0.06, 0.0),
        ((134.45, 34.65), 0.07, 0.06, 0.0),
        ((134.55, 34.60), 0.06, 0.06, 0.0),
        ((134.65, 34.55), 0.06, 0.06, 0.0),
        ((134.75, 34.50), 0.06, 0.06, 0.0),
        ((134.85, 34.45), 0.06, 0.06, 0.0),
        ((134.95, 34.40), 0.06, 0.06, 0.0),
    ]
    return any(ellipse(lon, lat, c, rx, ry, r) for c, rx, ry, r in specs)

def in_small_islands_east(lon, lat):
    specs = [
        ((133.35, 34.65), 0.06, 0.05, 0.0),
        ((133.45, 34.70), 0.06, 0.05, 0.0),
        ((133.55, 34.75), 0.06, 0.05, 0.0),
        ((133.65, 34.80), 0.06, 0.05, 0.0),
        ((133.75, 34.85), 0.06, 0.05, 0.0),
        ((133.85, 34.90), 0.06, 0.05, 0.0),
        ((133.95, 34.95), 0.06, 0.05, 0.0),
        ((134.05, 34.95), 0.06, 0.05, 0.0),
        ((134.15, 34.90), 0.06, 0.05, 0.0),
        ((134.25, 34.85), 0.06, 0.05, 0.0),
        ((134.35, 34.80), 0.06, 0.05, 0.0),
        ((134.45, 34.75), 0.06, 0.05, 0.0),
        ((134.55, 34.70), 0.06, 0.05, 0.0),
        ((134.65, 34.65), 0.06, 0.05, 0.0),
        ((134.75, 34.60), 0.06, 0.05, 0.0),
        ((134.85, 34.55), 0.06, 0.05, 0.0),
        ((134.95, 34.50), 0.06, 0.05, 0.0),
        ((134.00, 34.40), 0.10, 0.08, 0.0),
        ((134.05, 34.60), 0.07, 0.06, 0.0),
        ((134.10, 34.45), 0.09, 0.07, 0.0),
        ((134.20, 34.45), 0.07, 0.06, 0.0),
        ((134.50, 34.50), 0.08, 0.07, 0.0),
        ((134.60, 34.45), 0.07, 0.06, 0.0),
        ((134.70, 34.40), 0.07, 0.06, 0.0),
    ]
    return any(ellipse(lon, lat, c, rx, ry, r) for c, rx, ry, r in specs)

def is_land(lon, lat):
    if in_kyushu(lon, lat) or in_shikoku(lon, lat) or in_honshu(lon, lat):
        return 1
    if in_awaji(lon, lat) or in_small_islands_west(lon, lat):
        return 1
    if in_small_islands_central(lon, lat) or in_small_islands_east(lon, lat):
        return 1
    return 0


def main():
    os.makedirs("outputs", exist_ok=True)
    grid = np.zeros((H, W), dtype=np.uint8)
    for i in range(H):
        lat = LAT0 - i * STEP
        for j in range(W):
            lon = LON0 + j * STEP
            grid[i, j] = is_land(lon, lat)
    with open("outputs/candidate.csv", "w", newline="", encoding="ascii") as f:
        out = csv.writer(f)
        for row in grid:
            out.writerow(row.tolist())
    img = Image.fromarray(np.where(grid == 1, 255, 0).astype(np.uint8), mode="L")
    img.save("outputs/candidate.png")
    img.resize((W * 8, H * 8), Image.Resampling.NEAREST).save("outputs/candidate_8x.png")
    assert grid.shape == (H, W)
    assert set(np.unique(grid)) <= {0, 1}


if __name__ == "__main__":
    main()
