"""Knowledge-only reconstruction of the Seto Inland Sea region.
All coordinates below are hand-written approximations; no map data is read.
"""
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 253, 64
LON0, LAT0, STEP = 129.0, 33.0, 2.0 / 63.0


def polygon(x, y, p):
    """Ray crossing test for one hand-drawn coastline polygon."""
    inside = False
    j = len(p) - 1
    for i, (xi, yi) in enumerate(p):
        xj, yj = p[j]
        hit = ((yi > y) != (yj > y)) and (x < (xj-xi)*(y-yi)/(yj-yi)+xi)
        if hit:
            inside = not inside
        j = i
    return inside


def ellipse(x, y, cx, cy, rx, ry):
    return ((x-cx)/rx)**2 + ((y-cy)/ry)**2 <= 1.0


LAND_POLYS = []
ELLIPSES = []


def poly(*points):
    LAND_POLYS.append(points)


def ell(cx, cy, rx, ry):
    ELLIPSES.append((cx, cy, rx, ry))


# Southern edge of Honshu (the polygon continues beyond the north frame).
poly((130.78, 34.00), (131.10, 33.98), (131.45, 34.06),
     (131.78, 34.14), (132.15, 34.23), (132.55, 34.29),
     (132.95, 34.35), (133.35, 34.40), (133.75, 34.47),
     (134.15, 34.52), (134.55, 34.60), (134.92, 34.69),
     (135.22, 34.72), (135.52, 34.68), (135.82, 34.73),
     (136.10, 34.82), (136.42, 34.86), (136.72, 34.77),
     (137.00, 34.86), (137.00, 35.00), (130.78, 35.00))

# Kyushu, clipped naturally by the requested frame.
poly((129.42, 33.00), (129.20, 33.22), (129.17, 33.50),
     (129.40, 33.76), (129.82, 33.94), (130.28, 34.03),
     (130.72, 34.04), (131.16, 34.00), (131.55, 33.88),
     (131.78, 33.68), (131.84, 33.35), (131.72, 33.00))

# Shikoku's main mass: north coast, west cape, Pacific coast, east cape.
poly((132.00, 33.73), (132.10, 34.00), (132.48, 34.14),
     (132.88, 34.20), (133.30, 34.24), (133.72, 34.31),
     (134.10, 34.34), (134.46, 34.25), (134.70, 34.03),
     (134.72, 33.78), (134.58, 33.53), (134.36, 33.28),
     (134.02, 33.08), (133.62, 32.98), (133.16, 32.96),
     (132.72, 33.03), (132.34, 33.22), (132.10, 33.45))

# Kii peninsula and the south/east shore of Ise Bay.
poly((135.02, 34.43), (135.35, 34.48), (135.78, 34.50),
     (136.18, 34.46), (136.48, 34.36), (136.76, 34.18),
     (137.00, 34.02), (137.00, 33.00), (135.58, 33.00),
     (135.36, 33.28), (135.12, 33.60), (134.98, 33.96))

# Awaji and the larger islands west of it.
poly((134.62, 34.68), (134.86, 34.69), (135.00, 34.53),
     (134.97, 34.27), (134.83, 34.12), (134.62, 34.25),
     (134.55, 34.47))
poly((133.83, 34.55), (134.02, 34.58), (134.13, 34.48),
     (134.08, 34.37), (133.91, 34.35), (133.80, 34.44))
poly((134.12, 34.55), (134.27, 34.56), (134.34, 34.48),
     (134.29, 34.39), (134.15, 34.40), (134.08, 34.48))
# More coastal detail: peninsulas and the principal islands of the inland sea.

# Chita peninsula, on the western side of Ise Bay.
poly((136.68, 34.80), (137.00, 34.82), (137.00, 34.35),
     (136.88, 34.43), (136.73, 34.60))

# Western Seto islands and the islands at the Bungo-water entrance.
ell(131.65, 33.72, .115, .105)       # Himeshima
ell(132.28, 33.93, .22, .16)         # Suo-Oshima
ell(132.12, 33.77, .075, .055)
ell(131.98, 33.82, .055, .040)
ell(132.43, 34.06, .09, .07)
ell(132.53, 34.10, .08, .06)
ell(132.30, 34.29, .095, .10)         # Miyajima
ell(132.46, 34.29, .13, .095)         # Etajima
ell(132.55, 34.18, .10, .075)
ell(132.47, 34.03, .075, .055)
ell(132.67, 34.22, .055, .040)
ell(132.72, 34.08, .075, .050)
ell(132.82, 34.11, .055, .040)

# Geiyo / Shimanami Kaido island chain.
ell(132.95, 34.24, .13, .10)          # Osakikamijima
ell(133.03, 34.19, .11, .105)         # Omishima
ell(133.12, 34.13, .095, .13)         # Oshima
ell(133.19, 34.23, .055, .075)        # Hakatajima
ell(133.10, 34.31, .115, .10)         # Ikuchijima
ell(133.18, 34.36, .13, .09)          # Innoshima
ell(133.25, 34.43, .10, .07)          # Mukaishima
ell(133.28, 34.28, .05, .04)
ell(133.35, 34.22, .055, .035)
ell(133.38, 34.12, .08, .045)
ell(133.48, 34.18, .06, .045)
ell(133.52, 34.28, .05, .035)
ell(133.61, 34.25, .055, .04)
ell(133.70, 34.30, .05, .035)

# Islands between Matsuyama and Kagawa.
ell(132.72, 33.92, .09, .06)
ell(132.86, 33.88, .07, .05)
ell(133.00, 33.93, .08, .05)
ell(133.13, 33.96, .065, .045)
ell(133.25, 33.98, .07, .05)
ell(133.39, 34.00, .055, .04)
ell(133.55, 34.08, .06, .04)
ell(133.67, 34.12, .06, .045)
ell(133.80, 34.18, .07, .05)
ell(133.88, 34.27, .06, .04)

# Northern Kagawa waters: Shodoshima and the smaller islands around Takamatsu.
poly((134.02, 34.61), (134.30, 34.66), (134.48, 34.56),
     (134.49, 34.42), (134.33, 34.32), (134.12, 34.36),
     (134.00, 34.47))
ell(133.95, 34.46, .09, .075)          # Naoshima
ell(133.82, 34.46, .07, .045)
ell(133.75, 34.50, .055, .035)
ell(133.98, 34.40, .065, .045)         # Megijima/Ogijima area
ell(134.05, 34.42, .055, .04)
ell(134.20, 34.43, .06, .04)
ell(134.42, 34.35, .07, .045)
ell(134.55, 34.44, .055, .04)
ell(134.60, 34.52, .055, .04)

# Small islands in the Harima-nada and Osaka-nada approaches.
ell(134.48, 34.62, .05, .035)
ell(134.70, 34.57, .055, .035)
ell(134.82, 34.58, .04, .03)
ell(134.95, 34.58, .05, .035)
ell(135.05, 34.55, .045, .03)
# Islands outside the main Seto chain, especially around Kyushu and Bungo.

# Tsushima is a long, indented island in the north-west corner.
poly((129.22, 34.20), (129.30, 34.43), (129.28, 34.68),
     (129.40, 34.82), (129.57, 34.72), (129.58, 34.48),
     (129.50, 34.25), (129.42, 34.08), (129.27, 34.10))
poly((129.28, 34.05), (129.42, 34.10), (129.50, 34.25),
     (129.43, 34.34), (129.27, 34.25))

# Iki and the Hirado / northern Nagasaki island groups.
poly((129.63, 33.72), (129.78, 33.68), (129.93, 33.78),
     (129.91, 33.94), (129.75, 34.00), (129.61, 33.91))
ell(129.54, 33.38, .10, .14)           # Hirado
ell(129.39, 33.52, .055, .08)
ell(129.30, 33.33, .065, .045)
ell(129.20, 33.18, .07, .05)
ell(129.08, 33.05, .06, .04)
ell(129.78, 33.31, .065, .045)
ell(129.94, 33.22, .055, .035)
ell(130.05, 33.10, .06, .04)

# A few visible islands along the western Kyushu coast.
ell(130.24, 33.44, .055, .04)
ell(130.40, 33.56, .05, .035)
ell(130.58, 33.65, .06, .04)
ell(130.70, 33.78, .05, .035)
ell(130.87, 33.84, .045, .03)

# Suwada, Aoshima and other small Bungo-channel islands.
ell(131.78, 33.28, .045, .03)
ell(131.88, 33.18, .055, .035)
ell(132.00, 33.12, .04, .028)
ell(132.10, 33.05, .045, .03)
ell(132.20, 33.00, .04, .026)
ell(132.42, 33.38, .06, .04)
ell(132.58, 33.43, .055, .035)
ell(132.70, 33.48, .06, .035)

# Southern Shikoku outliers and the Kochi-side islands.
ell(132.35, 33.00, .04, .027)
ell(132.60, 33.03, .045, .028)
ell(133.04, 32.98, .04, .025)
ell(133.45, 32.92, .045, .025)
ell(133.85, 32.92, .04, .025)
ell(134.35, 33.00, .05, .03)
ell(134.72, 33.42, .055, .035)

# Islands around the east entrance to the Inland Sea and Osaka Bay.
ell(135.02, 34.12, .045, .03)
ell(135.14, 34.20, .06, .04)
ell(135.31, 34.27, .05, .035)
ell(135.47, 34.36, .055, .035)
ell(135.62, 34.42, .045, .03)
ell(135.78, 34.47, .04, .028)

# Sparse islands in Ise Bay (the mainland shores remain polygonal).
ell(136.66, 34.48, .035, .025)
ell(136.78, 34.39, .045, .03)
ell(136.56, 34.27, .035, .025)
ell(136.42, 34.20, .04, .027)

def is_land(lon, lat):
    # The point is classified directly; there is no cell averaging or cleanup.
    if any(polygon(lon, lat, p) for p in LAND_POLYS):
        return True
    return any(ellipse(lon, lat, *e) for e in ELLIPSES)


def build_matrix():
    # The north-to-south and west-to-east ordering is part of the output format.
    rows = []
    for r in range(H):
        lat = 35.0 - r * STEP
        row = [1 if is_land(129.0 + c * STEP, lat) else 0
               for c in range(W)]
        rows.append(row)
    return np.asarray(rows, dtype=np.uint8)


def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()
    assert matrix.shape == (64, 253)
    assert set(np.unique(matrix)).issubset({0, 1})

    with (out / "candidate.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # L mode is an exact two-colour image: 0 black water, 255 white land.
    raw = Image.fromarray(matrix * 255, mode="L")
    assert raw.size == (253, 64)
    raw.save(out / "candidate.png")
    enlarged = raw.resize((2024, 512), resample=Image.Resampling.NEAREST)
    enlarged.save(out / "candidate_8x.png")

    # Lightweight self-checks for the requested artefacts.
    with (out / "candidate.csv").open("r", encoding="utf-8") as f:
        check = list(csv.reader(f))
    assert len(check) == 64 and all(len(row) == 253 for row in check)
    assert all(v in {"0", "1"} for row in check for v in row)
    assert set(raw.getdata()).issubset({0, 255})
    assert enlarged.size == (2024, 512)
    print("generated", matrix.shape, "land points", int(matrix.sum()))


if __name__ == "__main__":
    main()
