import os
import csv
import numpy as np
from PIL import Image

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

NROW, NCOL = 64, 253
lats = 35.0 - np.arange(NROW) * (2.0 / 63.0)
lons = 129.0 + np.arange(NCOL) * (8.0 / 252.0)

def inside(x, y, p):
    """Even-odd point test for a hand-drawn polygon."""
    hit = False
    j = len(p) - 1
    for i in range(len(p)):
        xi, yi = p[i]
        xj, yj = p[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi) + xi
        ):
            hit = not hit
        j = i
    return hit

def add_poly(mask, poly):
    xmin = max(0, int((min(x for x, y in poly) - 129.0) / 8.0 * 252) - 2)
    xmax = min(NCOL - 1, int((max(x for x, y in poly) - 129.0) / 8.0 * 252) + 2)
    ymin = max(0, int((35.0 - max(y for x, y in poly)) / 2.0 * 63) - 2)
    ymax = min(NROW - 1, int((35.0 - min(y for x, y in poly)) / 2.0 * 63) + 2)
    for r in range(ymin, ymax + 1):
        for c in range(xmin, xmax + 1):
            if inside(lons[c], lats[r], poly):
                mask[r, c] = 1

def add_ellipse(mask, cx, cy, rx, ry, angle=0.0):
    # Ellipses are deliberately small, for islands and exposed rock groups.
    co, si = np.cos(angle), np.sin(angle)
    for r, y in enumerate(lats):
        if abs(y - cy) > ry * 1.3:
            continue
        for c, x in enumerate(lons):
            dx, dy = x - cx, y - cy
            u, v = co * dx + si * dy, -si * dx + co * dy
            if (u / rx) ** 2 + (v / ry) ** 2 <= 1.0:
                mask[r, c] = 1

def land_mask():
    m = np.zeros((NROW, NCOL), dtype=np.uint8)

    # Northern Kyushu.  The western edge of the experiment cuts through
    # Nagasaki and the southern edge cuts through the Kyushu interior.
    add_poly(m, [
        (129.0, 33.00), (131.70, 33.00), (131.78, 33.45),
        (131.65, 33.85), (131.48, 34.15), (131.15, 34.32),
        (130.70, 34.43), (130.20, 34.48), (129.72, 34.42),
        (129.25, 34.28), (129.0, 34.10)
    ])

    # Honshu: the southern Chugoku coast and the inland side of Kansai.
    add_poly(m, [
        (130.82, 35.00), (137.00, 35.00), (137.00, 34.76),
        (136.75, 34.70), (136.42, 34.73), (136.12, 34.78),
        (135.80, 34.75), (135.50, 34.68), (135.22, 34.68),
        (134.98, 34.62), (134.75, 34.55), (134.45, 34.49),
        (134.12, 34.43), (133.78, 34.39), (133.42, 34.36),
        (133.05, 34.33), (132.70, 34.28), (132.35, 34.24),
        (131.98, 34.20), (131.62, 34.16), (131.25, 34.13),
        (130.95, 34.16), (130.82, 34.32)
    ])

    # Shikoku, including its irregular south coast.
    add_poly(m, [
        (131.82, 34.18), (132.08, 34.39), (132.45, 34.47),
        (132.88, 34.50), (133.28, 34.49), (133.70, 34.48),
        (134.10, 34.43), (134.47, 34.34), (134.76, 34.16),
        (134.65, 33.91), (134.48, 33.70), (134.30, 33.46),
        (133.88, 33.28), (133.43, 33.24), (132.98, 33.27),
        (132.61, 33.37), (132.30, 33.55), (132.03, 33.82),
        (131.88, 34.02)
    ])

    # Kii Peninsula, south of Osaka and Ise bays.
    add_poly(m, [
        (134.80, 34.53), (135.08, 34.55), (135.32, 34.43),
        (135.45, 34.22), (135.52, 34.02), (135.54, 33.78),
        (135.42, 33.54), (135.25, 33.32), (135.55, 33.42),
        (135.82, 33.48), (136.05, 33.67), (136.25, 33.91),
        (136.38, 34.12), (136.15, 34.28), (135.92, 34.42),
        (135.67, 34.53), (135.40, 34.56), (135.10, 34.67)
    ])

    return m
def add_islands(m):
    # Offshore islands at the western mouth: Tsushima, Iki, Hirado and Goto.
    add_poly(m, [
        (129.18, 34.17), (129.27, 34.38), (129.35, 34.62),
        (129.43, 34.82), (129.58, 34.76), (129.61, 34.51),
        (129.52, 34.28), (129.40, 34.12), (129.25, 34.09)
    ])
    add_ellipse(m, 129.72, 33.75, .16, .12, -.25)
    add_ellipse(m, 129.56, 33.38, .13, .18, .35)
    add_ellipse(m, 129.25, 33.70, .11, .22, -.25)
    add_ellipse(m, 129.18, 33.27, .11, .24, .25)
    add_ellipse(m, 129.12, 33.02, .09, .18, .20)

    # Islands in the Suo-nada and around the western Seto strait.
    add_ellipse(m, 131.52, 33.96, .17, .20, .30)   # Himeshima
    add_ellipse(m, 132.20, 33.93, .28, .13, -.25)   # Suo-Oshima
    add_ellipse(m, 132.02, 34.02, .10, .12, -.30)
    add_ellipse(m, 132.43, 34.08, .12, .10, -.10)
    add_ellipse(m, 132.28, 34.24, .13, .18, .15)    # Miyajima
    add_ellipse(m, 132.40, 34.23, .22, .15, -.25)   # Etajima/Nomi
    add_ellipse(m, 132.53, 34.08, .18, .14, .20)
    add_ellipse(m, 132.69, 34.21, .11, .10, -.15)
    add_ellipse(m, 132.79, 34.16, .10, .08, .15)

    # The dense Geiyo archipelago, placed as separate land rather than
    # connecting islands with a smoothing operation.
    add_ellipse(m, 132.88, 34.25, .20, .12, -.20)   # Osakikamijima
    add_ellipse(m, 133.00, 34.30, .16, .14, .25)    # Omishima
    add_ellipse(m, 133.10, 34.19, .12, .18, -.15)   # Hakata
    add_ellipse(m, 133.18, 34.08, .18, .12, .20)    # Oshima
    add_ellipse(m, 133.18, 34.34, .14, .17, -.25)   # Innoshima
    add_ellipse(m, 133.30, 34.38, .13, .10, .10)
    add_ellipse(m, 133.36, 34.26, .13, .12, -.30)
    add_ellipse(m, 133.08, 34.40, .09, .08, .15)
    add_ellipse(m, 133.47, 34.42, .10, .08, -.15)

    # Islands off the Bisan and Harima seas.
    add_ellipse(m, 133.72, 34.42, .12, .08, -.20)
    add_ellipse(m, 133.92, 34.43, .18, .10, .10)   # Naoshima group
    add_ellipse(m, 134.06, 34.48, .12, .10, -.25)
    add_ellipse(m, 134.12, 34.54, .15, .11, .20)   # Teshima
    add_ellipse(m, 134.27, 34.50, .33, .19, -.25)  # Shodoshima
    add_ellipse(m, 134.47, 34.38, .10, .08, .10)
    add_ellipse(m, 134.54, 34.50, .10, .08, -.20)

    # Awaji is a large, continuous island between Osaka and Harima bays.
    add_poly(m, [
        (134.62, 34.38), (134.66, 34.60), (134.77, 34.78),
        (134.95, 34.80), (135.10, 34.67), (135.13, 34.46),
        (135.05, 34.26), (134.90, 34.12), (134.73, 34.17)
    ])
    add_ellipse(m, 135.28, 34.31, .07, .08, .2)

    # Small islands at the eastern mouth of the Inland Sea and Ise bay.
    add_ellipse(m, 135.73, 34.35, .08, .09, -.2)
    add_ellipse(m, 136.72, 34.48, .07, .09, .1)
    add_ellipse(m, 136.88, 34.55, .05, .07, -.2)
    return m
def write_outputs():
    matrix = land_mask()
    matrix = add_islands(matrix)

    assert matrix.shape == (64, 253)
    assert set(np.unique(matrix)).issubset({0, 1})
    assert int(matrix.size) == 16192

    with open(os.path.join(OUT, "candidate.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        for row in matrix:
            w.writerow([int(v) for v in row])

    # L mode is used with exactly the two values 0 and 255.
    small = Image.fromarray((matrix * 255).astype(np.uint8), mode="L")
    assert small.size == (253, 64)
    small.save(os.path.join(OUT, "candidate.png"))
    large = small.resize((2024, 512), resample=Image.Resampling.NEAREST)
    assert large.size == (2024, 512)
    large.save(os.path.join(OUT, "candidate_8x.png"))

if __name__ == "__main__":
    write_outputs()
