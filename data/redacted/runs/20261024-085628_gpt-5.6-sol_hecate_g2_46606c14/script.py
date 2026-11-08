"""Knowledge-only reconstruction of the Hecate Strait sample grid.

No files are read and no geographic service is used.  The coast is represented
by hand-written, approximate WGS84 polygons and ellipses, then every requested
sample coordinate is tested against their union.
"""
from pathlib import Path
import csv
import math
import numpy as np
from PIL import Image

N = 127
STEP = 2.0 / 126.0


def inside_polygon(x, y, p):
    """Even/odd point-in-polygon test (boundary is treated as land)."""
    hit = False
    j = len(p) - 1
    for i in range(len(p)):
        xi, yi = p[i]
        xj, yj = p[j]
        dx, dy = xj - xi, yj - yi
        cross = (x-xi)*dy - (y-yi)*dx
        if abs(cross) < 1e-10 and min(xi,xj)-1e-10 <= x <= max(xi,xj)+1e-10 \
                and min(yi,yj)-1e-10 <= y <= max(yi,yj)+1e-10:
            return True
        if ((yi > y) != (yj > y)):
            xx = xi + (y-yi) * dx / dy
            if x < xx:
                hit = not hit
        j = i
    return hit


def in_ellipse(x, y, e):
    cx, cy, rx, ry, ang = e
    a = math.radians(ang)
    u = (x-cx)*math.cos(a) + (y-cy)*math.sin(a)
    v = -(x-cx)*math.sin(a) + (y-cy)*math.cos(a)
    return (u/rx)**2 + (v/ry)**2 <= 1.0


# Principal islands, proceeding roughly north to south.  Vertices are solely
# a hand reconstruction from remembered regional layout.
POLYS = [
 # Wales / Pearse group at the head of Chatham Sound
 [(-130.35,55),(-130.31,54.91),(-130.34,54.80),(-130.25,54.72),
  (-130.11,54.75),(-130.08,54.86),(-130.12,55)],
 [(-130.55,55),(-130.50,54.91),(-130.51,54.82),(-130.42,54.78),
  (-130.35,54.84),(-130.37,54.94),(-130.41,55)],
 # Dundas
 [(-130.91,54.69),(-130.80,54.72),(-130.69,54.66),(-130.64,54.55),
  (-130.68,54.43),(-130.79,54.38),(-130.89,54.42),(-130.96,54.52),
  (-130.96,54.62)],
 # Melville and Dunira
 [(-130.66,54.62),(-130.58,54.64),(-130.48,54.57),(-130.48,54.47),
  (-130.55,54.40),(-130.65,54.43),(-130.70,54.52)],
 [(-130.45,54.54),(-130.37,54.53),(-130.32,54.45),(-130.36,54.38),
  (-130.44,54.40),(-130.49,54.47)],
 # Stephens
 [(-130.81,54.34),(-130.70,54.36),(-130.59,54.31),(-130.52,54.23),
  (-130.54,54.13),(-130.63,54.08),(-130.74,54.12),(-130.82,54.21)],
 # Digby and Kaien (Prince Rupert)
 [(-130.52,54.36),(-130.43,54.37),(-130.39,54.30),(-130.43,54.22),
  (-130.51,54.23),(-130.56,54.29)],
 [(-130.38,54.34),(-130.26,54.35),(-130.17,54.29),(-130.15,54.20),
  (-130.23,54.15),(-130.34,54.18),(-130.40,54.25)],
 # Porcher
 [(-130.69,54.13),(-130.55,54.15),(-130.42,54.10),(-130.33,54.01),
  (-130.35,53.91),(-130.45,53.82),(-130.58,53.84),(-130.65,53.93),
  (-130.73,54.02)],
 # McCauley
 [(-130.42,53.98),(-130.31,53.98),(-130.23,53.91),(-130.20,53.79),
  (-130.27,53.70),(-130.37,53.72),(-130.44,53.81)],
]
# Long islands bordering central Hecate Strait.
POLYS += [
 # Banks Island
 [(-130.91,53.89),(-130.82,53.91),(-130.72,53.85),(-130.60,53.74),
  (-130.49,53.61),(-130.36,53.48),(-130.27,53.35),(-130.29,53.25),
  (-130.39,53.23),(-130.50,53.31),(-130.60,53.42),(-130.72,53.54),
  (-130.82,53.68),(-130.91,53.80)],
 # Pitt Island
 [(-130.29,54.02),(-130.18,54.01),(-130.05,53.92),(-129.94,53.82),
  (-129.82,53.70),(-129.70,53.57),(-129.61,53.43),(-129.59,53.32),
  (-129.66,53.26),(-129.77,53.30),(-129.88,53.39),(-129.98,53.51),
  (-130.09,53.63),(-130.19,53.75),(-130.28,53.88)],
 # Trutch Island
 [(-130.22,53.30),(-130.13,53.32),(-130.02,53.26),(-129.91,53.17),
  (-129.86,53.08),(-129.91,53.02),(-130.01,53.05),(-130.11,53.12),
  (-130.20,53.21)],
 # Campania Island, clipped by the south edge
 [(-129.62,53.19),(-129.52,53.22),(-129.40,53.17),(-129.31,53.09),
  (-129.28,53.00),(-129.55,53.00),(-129.63,53.09)],
 # northwestern Princess Royal Island
 [(-129.35,53.50),(-129.22,53.53),(-129.08,53.48),(-129.00,53.39),
  (-129.00,53.00),(-129.20,53.00),(-129.23,53.11),(-129.30,53.22),
  (-129.39,53.33),(-129.43,53.43)],
]

# Mainland and large coastal peninsulas.  Deep fjords are left as water by
# using separate polygons rather than pretending the eastern edge is solid.
POLYS += [
 # northern mainland west of Portland/Work channels
 [(-130.08,55),(-129.83,55),(-129.81,54.93),(-129.87,54.85),
  (-129.82,54.76),(-129.91,54.67),(-130.03,54.66),(-130.10,54.73)],
 # peninsula between Portland and Work channels
 [(-129.74,55),(-129.43,55),(-129.43,54.87),(-129.50,54.77),
  (-129.58,54.69),(-129.69,54.67),(-129.78,54.74),(-129.76,54.86)],
 # eastern mainland around Work Channel
 [(-129.30,55),(-129.00,55),(-129.00,54.31),(-129.13,54.31),
  (-129.19,54.41),(-129.17,54.52),(-129.25,54.63),(-129.31,54.76)],
 # Tsimshian peninsula
 [(-130.18,54.66),(-130.04,54.65),(-129.94,54.57),(-129.91,54.46),
  (-129.95,54.35),(-130.05,54.29),(-130.14,54.34),(-130.20,54.44)],
 # land east of lower Work Channel and north of Skeena
 [(-129.56,54.72),(-129.43,54.73),(-129.31,54.66),(-129.22,54.56),
  (-129.17,54.44),(-129.20,54.33),(-129.32,54.30),(-129.43,54.36),
  (-129.51,54.47),(-129.61,54.57)],
 # islands/peninsulas south of the Skeena mouth
 [(-130.10,54.20),(-129.98,54.22),(-129.86,54.17),(-129.80,54.08),
  (-129.84,53.99),(-129.94,53.96),(-130.05,54.01),(-130.13,54.10)],
 [(-129.71,54.25),(-129.58,54.26),(-129.47,54.20),(-129.41,54.10),
  (-129.46,54.02),(-129.58,54.01),(-129.69,54.08),(-129.75,54.17)],
 # mainland shore east of Grenville Channel
 [(-129.32,54.31),(-129.00,54.31),(-129.00,53.54),(-129.10,53.51),
  (-129.18,53.57),(-129.23,53.69),(-129.27,53.82),(-129.34,53.95),
  (-129.38,54.09),(-129.37,54.21)],
]

# Remembered small offshore islands and rocky groups (elliptical approximations).
# (centre longitude, centre latitude, east-west radius, north-south radius, angle)
ELLIPSES = [
 (-130.99,54.64,.035,.065,-10), (-130.58,54.72,.035,.055,15),
 (-130.31,54.69,.026,.045,-20), (-130.74,54.48,.025,.035,0),
 (-130.97,54.37,.020,.032,15), (-130.89,54.29,.031,.050,-12),
 (-130.69,54.00,.025,.036,0), (-130.75,53.91,.020,.030,0),
 (-130.31,54.08,.018,.034,-20), (-130.15,53.89,.022,.040,-25),
 (-130.06,53.82,.020,.031,0), (-130.96,53.82,.025,.050,-18),
 (-130.87,53.69,.016,.035,-25), (-130.74,53.47,.019,.030,-25),
 (-130.62,53.29,.022,.036,-30), (-130.48,53.16,.021,.033,-25),
 (-130.36,53.07,.022,.045,-25), (-130.72,53.13,.016,.025,5),
 (-129.78,53.21,.028,.052,-30), (-129.67,53.04,.020,.035,-25),
 (-129.50,53.35,.018,.040,-15), (-129.47,53.57,.020,.045,-10),
 (-129.40,53.76,.019,.042,-12), (-129.30,53.91,.020,.035,-10),
]
# Tiny reefs/islets are represented only where their remembered groups are
# large enough to have a realistic chance of hitting this 1/63 degree lattice.
ELLIPSES += [
 (-130.89,54.76,.012,.018,0), (-130.47,54.69,.012,.018,0),
 (-130.25,54.58,.015,.022,0), (-130.08,54.47,.014,.025,-10),
 (-130.91,54.10,.014,.023,0), (-130.82,53.99,.012,.020,0),
 (-130.56,53.71,.012,.024,-25), (-130.46,53.53,.010,.019,-20),
 (-130.32,53.39,.011,.021,-25), (-130.18,53.53,.012,.022,-20),
 (-129.90,53.30,.013,.024,-30), (-129.72,53.13,.011,.020,-25),
]


def is_land(lon, lat):
    return any(inside_polygon(lon, lat, p) for p in POLYS) or \
           any(in_ellipse(lon, lat, e) for e in ELLIPSES)


def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    a = np.zeros((N, N), dtype=np.uint8)
    for r in range(N):
        lat = 55.0 - r * STEP
        for c in range(N):
            lon = -131.0 + c * STEP
            a[r, c] = 1 if is_land(lon, lat) else 0

    assert a.shape == (127, 127)
    assert set(np.unique(a)).issubset({0, 1})
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(a.tolist())

    raw = Image.fromarray(a * 255, mode="L")
    raw.save(out / "candidate.png")
    large = raw.resize((1016, 1016), resample=Image.Resampling.NEAREST)
    large.save(out / "candidate_8x.png")

    # Validate the serialized deliverables, including exact nearest-neighbour
    # replication rather than merely trusting dimensions.
    with (out / "candidate.csv").open(newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 127 and all(len(row) == 127 for row in rows)
    assert all(v in ("0", "1") for row in rows for v in row)
    with Image.open(out / "candidate.png") as im:
        check = np.asarray(im)
        assert im.size == (127, 127) and set(np.unique(check)) <= {0, 255}
    with Image.open(out / "candidate_8x.png") as im:
        zoom = np.asarray(im)
        assert im.size == (1016, 1016) and set(np.unique(zoom)) <= {0, 255}
        assert np.array_equal(zoom, np.repeat(np.repeat(check, 8, 0), 8, 1))
    print("validated: 127x127 CSV, 127x127 binary PNG, 1016x1016 nearest-neighbor PNG")


if __name__ == "__main__":
    main()
