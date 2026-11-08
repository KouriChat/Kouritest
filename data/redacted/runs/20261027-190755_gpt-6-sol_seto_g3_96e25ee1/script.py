import os, csv, math
import numpy as np
from PIL import Image

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
W, H = 253, 64
LON0, LAT0 = 129.0, 35.0
DX, DY = 8.0 / 252.0, 2.0 / 63.0

def inside(x, y, p):
    c = False
    j = len(p) - 1
    for i in range(len(p)):
        xi, yi = p[i]; xj, yj = p[j]
        if ((yi > y) != (yj > y)):
            xx = (xj-xi)*(y-yi)/(yj-yi) + xi
            if x < xx:
                c = not c
        j = i
    return c

def ellipse(x, y, cx, cy, rx, ry, angle=0.0):
    ca, sa = math.cos(angle), math.sin(angle)
    xx, yy = x-cx, y-cy
    u, v = xx*ca+yy*sa, -xx*sa+yy*ca
    return (u/rx)**2 + (v/ry)**2 <= 1.0

def add_poly(poly):
    POLYS.append(poly)

def add_ell(cx, cy, rx, ry, angle=0.0):
    ELLS.append((cx, cy, rx, ry, angle))

POLYS = []
ELLS = []
# Broad mainland outlines.  Each polygon is deliberately a hand-drawn
# coastline; its far edge is continued beyond the requested frame.
add_poly([
    (130.72,34.08),(130.95,34.12),(131.25,34.18),(131.55,34.22),
    (131.9,34.20),(132.25,34.18),(132.55,34.24),(132.82,34.32),
    (133.08,34.34),(133.35,34.38),(133.62,34.43),(133.82,34.54),
    (134.08,34.55),(134.32,34.68),(134.58,34.74),(134.84,34.75),
    (135.08,34.70),(135.32,34.58),(135.58,34.64),(135.85,34.67),
    (136.12,34.66),(136.37,34.72),(136.62,34.79),(137.1,34.82),
    (137.1,36.0),(130.7,36.0)
])

# Kyushu, including the irregular northern and western shore.
add_poly([
    (129.72,33.0),(131.78,33.0),(131.82,33.28),(131.72,33.55),
    (131.65,33.78),(131.52,33.98),(131.28,34.07),(130.95,34.08),
    (130.62,34.03),(130.35,33.93),(130.15,33.76),(129.98,33.55),
    (129.85,33.30)
])

# The south coast of Shikoku.
add_poly([
    (131.22,33.0),(134.86,33.0),(134.90,33.28),(134.82,33.55),
    (134.67,33.78),(134.47,33.89),(134.17,33.83),(133.85,33.70),
    (133.53,33.57),(133.20,33.48),(132.87,33.46),(132.53,33.43),
    (132.22,33.50),(131.93,33.65),(131.62,33.79),(131.38,33.90),
    (131.18,33.80)
])

# Shikoku's northern shore is separated from Honshu by the Inland Sea.
add_poly([
    (131.22,33.80),(131.43,33.93),(131.72,34.00),(132.02,34.05),
    (132.28,34.10),(132.55,34.12),(132.82,34.18),(133.08,34.22),
    (133.35,34.22),(133.63,34.17),(133.92,34.18),(134.20,34.22),
    (134.47,34.28),(134.70,34.30),(134.82,34.18),(134.80,33.88),
    (134.57,33.83),(134.25,33.84),(133.9,33.7),(133.5,33.57),
    (133.15,33.48),(132.8,33.46),(132.48,33.44),(132.17,33.51),
    (131.83,33.66),(131.52,33.80)
])

# Kii Peninsula, south of Osaka Bay and east of the Kii Channel.
add_poly([
    (134.98,34.42),(135.18,34.34),(135.32,34.18),(135.42,33.98),
    (135.62,33.82),(135.88,33.69),(136.18,33.62),(136.48,33.66),
    (136.72,33.82),(136.90,34.08),(137.05,34.35),(137.05,33.0),
    (134.98,33.0)
])

# Mie / Ise Bay's eastern shore and the Chubu mainland.
add_poly([
    (136.43,34.63),(136.58,34.73),(136.66,34.92),(136.82,35.08),
    (137.1,35.18),(137.1,36.0),(136.3,36.0),(136.3,35.45),
    (136.34,35.0)
])

# A small clipped piece of the Korean coast at the north-west corner.
add_poly([
    (128.85,35.12),(129.03,35.13),(129.20,35.20),(129.28,35.38),
    (128.85,35.45)
])
# Larger named islands and island groups around the Inland Sea.
add_poly([
    (129.16,34.05),(129.25,34.00),(129.43,34.10),(129.55,34.34),
    (129.55,34.60),(129.46,34.78),(129.28,34.77),(129.16,34.55)
])                                      # Tsushima
add_ell(129.78,33.78,.13,.10,.15)       # Iki
add_ell(129.58,33.38,.12,.13,.3)        # Hirado
add_ell(129.28,33.20,.10,.13,-.25)      # northern Goto island
add_ell(129.55,33.08,.10,.12,.2)

add_poly([
    (134.62,34.18),(134.70,34.02),(134.82,33.92),(134.98,33.93),
    (135.11,34.08),(135.16,34.30),(135.06,34.55),(134.91,34.62),
    (134.77,34.50),(134.69,34.34)
])                                      # Awaji
add_ell(134.17,34.52,.22,.11,.1)        # Shodoshima
add_ell(134.05,34.47,.09,.07,-.15)      # Teshima
add_ell(133.96,34.46,.08,.055,.1)       # Naoshima
add_ell(134.12,34.55,.045,.035,0)
add_ell(134.28,34.50,.045,.035,.2)
add_ell(134.08,34.39,.07,.045,.3)       # Megijima
add_ell(134.02,34.40,.06,.04,-.4)       # Ogijima
add_ell(134.10,34.57,.045,.03,0)        # Inujima

# Hiroshima Bay and western Suo islands
add_ell(132.33,34.30,.12,.13,.15)       # Miyajima
add_ell(132.44,34.27,.16,.09,.25)       # Etajima
add_ell(132.58,34.20,.10,.07,-.2)
add_ell(132.26,34.02,.16,.08,.1)        # Yashiro
add_ell(132.33,33.92,.15,.07,.1)        # Suo-Oshima
add_ell(132.05,34.12,.08,.05,.3)
add_ell(131.88,34.20,.07,.04,-.2)

# Geiyo archipelago: the long, crowded chain between Hiroshima and Imabari.
add_ell(132.78,34.28,.13,.10,.25)       # Osakikamijima
add_ell(132.90,34.24,.16,.12,.1)        # Omishima
add_ell(133.04,34.28,.11,.13,.15)       # Ikuchijima
add_ell(133.16,34.32,.11,.16,-.2)       # Innoshima
add_ell(133.16,34.16,.10,.16,.2)        # Hakata/Oshima
add_ell(133.31,34.13,.14,.10,-.2)
add_ell(133.39,34.19,.07,.05,0)
add_ell(132.67,34.15,.08,.05,.1)
add_ell(132.52,34.08,.09,.05,-.2)
add_ell(133.02,34.08,.07,.05,.2)
# Central and eastern Inland Sea islands.
add_ell(133.50,34.40,.10,.055,.2)
add_ell(133.63,34.38,.09,.045,-.2)
add_ell(133.76,34.35,.09,.05,.1)
add_ell(133.86,34.31,.06,.04,0)
add_ell(133.48,34.25,.07,.045,.2)
add_ell(133.67,34.20,.08,.05,-.3)
add_ell(133.80,34.18,.07,.04,.1)
add_ell(133.92,34.23,.06,.04,.2)
add_ell(134.28,34.30,.08,.045,0)
add_ell(134.42,34.34,.07,.04,.2)
add_ell(134.55,34.37,.05,.035,0)

# Islands off the Okayama and Kagawa shores.
add_ell(133.70,34.55,.06,.04,.2)
add_ell(133.86,34.52,.06,.04,-.1)
add_ell(134.20,34.62,.05,.03,.2)
add_ell(134.38,34.57,.05,.03,0)
add_ell(134.48,34.48,.06,.04,.2)
add_ell(134.62,34.39,.05,.035,.1)
add_ell(134.72,34.38,.05,.03,-.2)
add_ell(134.86,34.40,.06,.04,.2)
add_ell(134.96,34.43,.045,.03,0)
add_ell(135.02,34.50,.04,.03,.1)

# Small islands in the Kii Channel, Osaka Bay, and Ise Bay.
add_ell(134.97,34.67,.06,.04,.2)
add_ell(135.22,34.58,.07,.035,.1)
add_ell(135.34,34.53,.05,.03,0)
add_ell(135.47,34.54,.06,.035,-.1)
add_ell(135.55,34.62,.04,.03,.2)
add_ell(135.82,34.55,.05,.035,.1)
add_ell(136.00,34.48,.05,.03,.2)
add_ell(136.22,34.43,.06,.04,-.1)
add_ell(136.43,34.36,.05,.03,.1)

# Southern Kyushu / Bungo Strait islands visible in this frame.
add_ell(131.73,33.22,.06,.04,.2)
add_ell(131.93,33.28,.05,.03,0)
add_ell(132.15,33.28,.05,.03,.1)
add_ell(132.42,33.30,.05,.03,-.1)

def is_land(x, y):
    for p in POLYS:
        if inside(x, y, p):
            return 1
    for e in ELLS:
        if ellipse(x, y, *e):
            return 1
    return 0

mat = []
for r in range(H):
    lat = LAT0 - r * DY
    mat.append([is_land(LON0 + c * DX, lat) for c in range(W)])

assert len(mat) == 64 and all(len(row) == 253 for row in mat)
assert all(v in (0, 1) for row in mat for v in row)

with open(os.path.join(OUT, "candidate.csv"), "w", newline="", encoding="ascii") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerows(mat)

a = np.asarray(mat, dtype=np.uint8)
im = Image.fromarray(a * 255, mode="L")
im.save(os.path.join(OUT, "candidate.png"))
big = im.resize((W * 8, H * 8), resample=Image.Resampling.NEAREST)
big.save(os.path.join(OUT, "candidate_8x.png"))

assert im.size == (253, 64) and big.size == (2024, 512)
assert set(im.getdata()).issubset({0, 255})
assert set(big.getdata()).issubset({0, 255})
