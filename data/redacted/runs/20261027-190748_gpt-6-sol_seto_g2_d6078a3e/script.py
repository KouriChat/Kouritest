from math import cos, sin, pi
import csv, os
from PIL import Image

W, H = 253, 64
LON0, LAT0 = 129.0, 35.0
DLON = 8.0 / 252.0
DLAT = 2.0 / 63.0

def poly(*p):
    return [(float(x), float(y)) for x, y in p]

def ellipse(cx, cy, rx, ry, n=20, phase=0.0):
    return [(cx + rx*cos(phase+2*pi*i/n),
             cy + ry*sin(phase+2*pi*i/n)) for i in range(n)]

def inside(x, y, p):
    hit = False
    j = len(p)-1
    for i in range(len(p)):
        xi, yi = p[i]; xj, yj = p[j]
        if ((yi > y) != (yj > y)):
            xx = (xj-xi)*(y-yi)/(yj-yi) + xi
            if x < xx:
                hit = not hit
        j = i
    return hit

land = []
cuts = []
islands = []

# Honshu south-western shore, from western Yamaguchi to the Osaka area.
land.append(poly(
 (129.00,34.70),(129.48,34.72),(129.82,34.78),
 (130.16,34.84),(130.42,35.00),(137.00,35.00),(137.00,34.12),
 (136.78,34.05),(136.58,33.88),(136.38,33.65),
 (136.15,33.49),(135.92,33.43),(135.73,33.55),
 (135.60,33.82),(135.54,34.08),(135.42,34.35),
 (135.28,34.58),(135.05,34.70),(134.72,34.73),
 (134.42,34.69),(134.10,34.62),(133.82,34.54),
 (133.52,34.45),(133.23,34.38),(132.91,34.36),
 (132.60,34.34),(132.28,34.29),(131.98,34.25),
 (131.70,34.20),(131.40,34.20),(131.10,34.18),
 (130.82,34.20),(130.56,34.12),(130.32,34.03),
 (130.08,33.96),(129.86,34.02),(129.63,34.18),
 (129.42,34.38),(129.25,34.60),(129.00,34.72)
))

# Northern Kyushu (the lower edge is deliberately clipped by the frame).
land.append(poly(
 (129.00,34.02),(129.25,34.02),(129.48,33.94),
 (129.73,33.84),(129.98,33.76),(130.24,33.73),
 (130.48,33.77),(130.70,33.69),(130.91,33.56),
 (131.10,33.43),(131.29,33.25),(131.48,33.02),
 (131.60,33.00),(129.00,33.00)
))

# Shikoku, including the broad central island.
land.append(poly(
 (131.80,33.53),(131.92,33.78),(132.10,34.02),
 (132.34,34.12),(132.62,34.16),(132.92,34.19),
 (133.22,34.22),(133.52,34.29),(133.82,34.35),
 (134.10,34.31),(134.38,34.22),(134.62,34.08),
 (134.72,33.90),(134.55,33.76),(134.34,33.64),
 (134.12,33.49),(133.88,33.31),(133.62,33.18),
 (133.34,33.10),(133.04,33.08),(132.73,33.16),
 (132.43,33.26),(132.18,33.37),(131.98,33.45)
))

# Large bays which indent otherwise continuous coastal land.
cuts.append(poly(
 (135.10,34.72),(135.35,34.66),(135.54,34.52),
 (135.64,34.32),(135.70,34.10),(135.72,33.90),
 (135.60,33.82),(135.54,34.08),(135.42,34.35),
 (135.28,34.58)
))
cuts.append(poly(
 (136.25,35.00),(137.00,35.00),(137.00,34.12),
 (136.78,34.05),(136.90,34.30),(136.88,34.58),
 (136.72,34.80),(136.46,34.92)
))

# Kyoto-Osaka inland water is left as water; small land polygons below
# restore the principal islands and peninsular tips.
# Principal islands and island groups, represented by hand-drawn polygons
# or compact ellipses.  These are not data-derived coastlines.
islands += [
 poly((129.18,34.42),(129.22,34.62),(129.32,34.77),
      (129.47,34.73),(129.55,34.55),(129.52,34.32),
      (129.43,34.12),(129.29,34.18)),
 poly((129.62,33.67),(129.68,33.82),(129.82,33.86),
      (129.94,33.77),(129.91,33.62),(129.77,33.56)),
 poly((129.38,33.27),(129.47,33.43),(129.61,33.45),
      (129.70,33.34),(129.64,33.20),(129.49,33.15)),
 poly((129.02,33.04),(129.12,33.26),(129.28,33.36),
      (129.38,33.23),(129.32,33.00)),
 # Suo-Oshima and the islands off Hiroshima
 ellipse(132.22,33.90,.16,.32,24,.2),
 ellipse(132.30,34.27,.10,.18,18,.3),
 ellipse(132.47,34.22,.18,.16,20,.1),
 ellipse(132.70,34.18,.11,.12,18),
 ellipse(132.92,34.19,.10,.12,18),
 ellipse(133.04,34.27,.14,.15,20,.2),
 ellipse(133.20,34.30,.15,.12,20),
 ellipse(133.42,34.32,.10,.10,18),
 ellipse(133.57,34.37,.12,.08,18),
 # western and central Inland Sea islands
 ellipse(132.05,34.04,.10,.10,18),
 ellipse(132.38,33.98,.09,.10,18),
 ellipse(132.63,33.96,.11,.08,18),
 ellipse(132.87,33.99,.09,.07,18),
 ellipse(133.23,34.05,.12,.08,18),
 ellipse(133.51,34.05,.10,.07,18),
 # islands near the eastern Seto Sea
 poly((133.78,34.28),(133.85,34.39),(133.98,34.43),
      (134.08,34.37),(134.08,34.25),(133.94,34.20)),
 ellipse(134.08,34.43,.09,.07,18),
 ellipse(134.30,34.43,.07,.06,18),
 # Shodo, Awaji and the Naruto-side islands
 poly((134.02,34.42),(134.10,34.57),(134.28,34.62),
      (134.41,34.55),(134.45,34.40),(134.34,34.31),
      (134.14,34.32)),
 poly((134.65,34.20),(134.72,34.48),(134.91,34.68),
      (135.08,34.65),(135.17,34.43),(135.05,34.20),
      (134.84,34.08)),
 ellipse(134.72,34.43,.07,.08,18),
 ellipse(134.89,34.37,.06,.06,18),
 ellipse(134.56,34.38,.05,.06,16),
 # small Osaka/Kii and Ise bay islands
 ellipse(135.25,34.40,.05,.04,16),
 ellipse(135.33,34.34,.05,.04,16),
 ellipse(136.72,34.54,.08,.10,18),
 ellipse(136.86,34.67,.06,.07,16),
]
# Kii Peninsula and the two sides of Ise Bay.
land.append(poly(
 (134.98,34.62),(135.18,34.58),(135.36,34.40),
 (135.48,34.18),(135.53,33.98),(135.48,33.75),
 (135.38,33.55),(135.35,33.30),(135.50,33.20),
 (135.73,33.28),(135.95,33.43),(136.05,33.66),
 (136.03,33.90),(135.90,34.12),(135.70,34.27),
 (135.48,34.42),(135.27,34.59)
))

# Chita and Atsumi peninsulas, with the bay between them.
land.append(poly(
 (136.62,35.00),(137.00,35.00),(137.00,34.40),
 (136.90,34.33),(136.78,34.35),(136.66,34.49),
 (136.56,34.66)
))
land.append(poly(
 (136.48,34.72),(136.67,34.65),(136.76,34.48),
 (136.68,34.35),(136.52,34.27),(136.38,34.20),
 (136.27,34.12),(136.20,34.18),(136.25,34.35),
 (136.34,34.54)
))

# A few recognizable small islands in Ise Bay and Kii Channel.
islands += [
 ellipse(135.32,34.22,.045,.035,16),
 ellipse(135.42,34.14,.035,.030,16),
 ellipse(135.55,34.28,.040,.030,16),
 ellipse(135.72,34.13,.045,.035,16),
 ellipse(135.82,34.05,.035,.030,16),
 ellipse(136.17,34.23,.050,.035,16),
 ellipse(136.39,34.28,.035,.030,16),
 ellipse(134.73,34.02,.050,.035,16),
 ellipse(134.82,33.98,.035,.030,16),
]

# The broad Osaka and Harima embayments are water.  The cuts are intentionally
# coarse: their purpose is to preserve the major water/land topology.
cuts += [
 poly((134.40,34.65),(134.75,34.72),(135.05,34.72),
      (135.20,34.60),(135.08,34.42),(134.86,34.32),
      (134.61,34.34),(134.44,34.45)),
 poly((132.80,34.55),(133.18,34.58),(133.55,34.55),
      (133.78,34.46),(133.55,34.36),(133.18,34.34),
      (132.88,34.40)),
]

def is_land(x, y):
    # A point is first accepted by a manually outlined mainland/peninsula
    # region, then removed by manually outlined bays, then islands restore
    # the land classification.  No external geographic lookup is used.
    value = any(inside(x, y, p) for p in land)
    if any(inside(x, y, p) for p in cuts):
        value = False
    if any(inside(x, y, p) for p in islands):
        value = True
    return 1 if value else 0
def main():
    rows = []
    for r in range(H):
        lat = LAT0 - r*DLAT
        row = []
        for c in range(W):
            lon = LON0 + c*DLON
            row.append(is_land(lon, lat))
        rows.append(row)

    assert len(rows) == 64 and all(len(row) == 253 for row in rows)
    assert all(v in (0, 1) for row in rows for v in row)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/candidate.csv", "w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)

    # Pillow's one-byte pixels make the source image exactly two-valued.
    img = Image.new("1", (W, H))
    pix = img.load()
    for y, row in enumerate(rows):
        for x, v in enumerate(row):
            pix[x, y] = 1 if v else 0
    img.save("outputs/candidate.png")
    big = img.resize((W*8, H*8), Image.Resampling.NEAREST)
    big.save("outputs/candidate_8x.png")

    with open("outputs/candidate.csv", encoding="ascii") as f:
        check = list(csv.reader(f))
    assert len(check) == 64
    assert all(len(row) == 253 and set(row) <= {"0", "1"} for row in check)
    assert Image.open("outputs/candidate.png").size == (253, 64)
    assert Image.open("outputs/candidate_8x.png").size == (2024, 512)

if __name__ == "__main__":
    main()
