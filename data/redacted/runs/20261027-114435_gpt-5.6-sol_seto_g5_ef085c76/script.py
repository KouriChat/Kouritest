from pathlib import Path
import csv
import numpy as np
from PIL import Image

# Memory-only reconstruction: polygons and ellipses below are hand-authored.
ROWS, COLS = 64, 253
lat = np.linspace(35.0, 33.0, ROWS)
lon = np.linspace(129.0, 137.0, COLS)
X, Y = np.meshgrid(lon, lat)
land = np.zeros((ROWS, COLS), dtype=bool)

def inside_poly(points):
    """Vectorized even/odd point-in-polygon test at the sample points."""
    hit = np.zeros_like(land)
    xj, yj = points[-1]
    for xi, yi in points:
        cross = ((yi > Y) != (yj > Y))
        xcut = (xj - xi) * (Y - yi) / ((yj - yi) or 1e-12) + xi
        hit ^= cross & (X < xcut)
        xj, yj = xi, yi
    return hit

def add_poly(points):
    global land
    land |= inside_poly(points)

def cut_poly(points):
    global land
    land &= ~inside_poly(points)

def ellipse(cx, cy, rx, ry, angle=0.0):
    a = np.deg2rad(angle)
    dx, dy = X - cx, Y - cy
    u = dx * np.cos(a) + dy * np.sin(a)
    v = -dx * np.sin(a) + dy * np.cos(a)
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1

def add_ellipse(cx, cy, rx, ry, angle=0.0):
    global land
    land |= ellipse(cx, cy, rx, ry, angle)

def cut_ellipse(cx, cy, rx, ry, angle=0.0):
    global land
    land &= ~ellipse(cx, cy, rx, ry, angle)

def add_chain(items):
    for cx, cy, rx, ry, angle in items:
        add_ellipse(cx, cy, rx, ry, angle)

# Large landmasses.  Vertices follow remembered capes, bays, and coast trends.
add_poly([  # northern Kyushu
    (129.32,33.00),(129.26,33.18),(129.45,33.39),(129.48,33.58),
    (129.61,33.66),(129.70,33.88),(129.92,33.96),(130.08,33.89),
    (130.20,33.98),(130.35,33.96),(130.47,34.10),(130.66,34.18),
    (130.88,34.05),(130.98,33.93),(131.10,33.93),(131.19,33.74),
    (131.05,33.56),(131.00,33.32),(131.15,33.12),(131.17,33.00)
])

add_poly([  # Honshu: Chugoku, Kansai, and the Kii peninsula
    (130.86,34.08),(131.03,34.17),(131.15,34.31),(131.35,34.42),
    (131.66,34.46),(131.86,34.40),(132.06,34.33),(132.18,34.25),
    (132.31,34.31),(132.43,34.23),(132.52,34.31),(132.69,34.27),
    (132.83,34.36),(133.04,34.42),(133.24,34.43),(133.41,34.48),
    (133.61,34.46),(133.83,34.50),(134.08,34.55),(134.30,34.61),
    (134.52,34.65),(134.75,34.68),(134.94,34.66),(135.13,34.71),
    (135.28,34.68),(135.42,34.59),(135.38,34.47),(135.24,34.39),
    (135.13,34.28),(135.08,34.08),(135.12,33.88),(135.25,33.68),
    (135.43,33.54),(135.64,33.45),(135.79,33.45),(135.89,33.56),
    (135.96,33.76),(136.08,33.92),(136.28,34.08),(136.48,34.22),
    (136.58,34.37),(136.63,34.55),(136.75,34.70),(136.89,34.78),
    (137.00,34.80),(137.00,35.00),(132.85,35.00),(132.62,34.96),
    (132.32,34.92),(132.06,34.86),(131.83,34.72),(131.57,34.61),
    (131.30,34.54),(131.14,34.40),(131.08,34.25),(130.92,34.17)
])

# Small Korean edge visible at the extreme northwest corner.
add_poly([(129.00,35.00),(129.18,35.00),(129.12,34.96),(129.00,34.94)])

# Shikoku, with its narrow western arm and deeply curved southern coast.
add_poly([
    (131.96,33.34),(132.18,33.35),(132.38,33.45),(132.52,33.58),
    (132.62,33.75),(132.76,33.86),(132.92,33.91),(133.03,34.05),
    (133.18,34.16),(133.39,34.21),(133.58,34.19),(133.76,34.27),
    (133.95,34.35),(134.15,34.40),(134.31,34.34),(134.43,34.25),
    (134.63,34.22),(134.76,34.14),(134.69,34.01),(134.52,33.84),
    (134.39,33.66),(134.31,33.43),(134.21,33.24),(134.07,33.04),
    (133.99,33.00),(132.58,33.00),(132.57,33.12),(132.45,33.24),
    (132.27,33.30),(132.10,33.28)
])

# Awaji-shima, forming the east side of the Inland Sea.
add_poly([
    (134.67,34.19),(134.78,34.22),(134.87,34.34),(134.92,34.46),
    (135.01,34.56),(134.97,34.62),(134.87,34.58),(134.77,34.48),
    (134.69,34.35),(134.64,34.25)
])

# Tsushima's two principal lobes and Iki.
add_poly([(129.16,34.09),(129.25,34.14),(129.32,34.28),(129.31,34.43),
          (129.38,34.51),(129.34,34.58),(129.25,34.51),(129.19,34.36)])
add_poly([(129.30,34.47),(129.42,34.54),(129.48,34.67),(129.45,34.72),
          (129.36,34.68),(129.29,34.59)])
add_ellipse(129.70,33.78,0.095,0.075,-12)

# Hirado and the outer islands off northwestern Kyushu.
add_poly([(129.31,33.16),(129.40,33.20),(129.48,33.36),(129.52,33.49),
          (129.46,33.57),(129.37,33.52),(129.34,33.35)])
add_chain([
    (129.05,33.18,.055,.10,-12),(129.13,33.08,.045,.07,-20),
    (129.23,33.03,.035,.045,0),(129.54,33.61,.035,.025,0),
])

# Western Inland Sea: Suo-nada and Hiroshima Bay islands.
add_chain([
    (131.40,33.90,.07,.035,-18),   # Himeshima
    (131.70,33.92,.045,.025,5),(131.87,34.00,.035,.055,-20),
    (132.02,33.92,.055,.035,20),(132.12,34.02,.035,.060,-15),
    (132.22,34.12,.050,.10,-25),(132.29,34.18,.035,.065,-25),
    (132.30,34.28,.045,.11,-38),   # Itsukushima
    (132.44,34.20,.070,.14,12),(132.52,34.11,.060,.12,18), # Etajima
    (132.60,34.02,.055,.080,20),(132.70,34.08,.035,.055,10),
])

# Geiyo archipelago between Hiroshima and Imabari.
add_chain([
    (132.75,34.18,.060,.080,5),(132.83,34.27,.075,.090,-20),
    (132.91,34.17,.075,.105,-8),(132.96,34.04,.045,.060,18),
    (133.02,34.23,.105,.105,12),   # Omi-shima
    (133.12,34.16,.050,.080,-10),(133.14,34.27,.065,.080,20),
    (133.21,34.31,.050,.075,-18),(133.20,34.39,.055,.070,28),
    (133.28,34.45,.035,.050,15),(133.09,34.36,.055,.055,-10),
    (132.88,34.36,.030,.050,0),(132.69,34.34,.035,.060,-20),
])

# Central/eastern Inland Sea, including Shodo-shima.
add_chain([
    (133.47,34.10,.040,.055,-10),(133.54,34.25,.035,.060,15),
    (133.66,34.32,.040,.050,-10),(133.77,34.37,.030,.045,10),
    (133.85,34.43,.035,.050,0),(133.95,34.48,.040,.060,-10),
    (134.08,34.49,.070,.050,-5),(134.18,34.48,.035,.045,0),
    (134.29,34.51,.145,.095,-8),   # Shodo-shima
    (134.39,34.55,.065,.055,20),(134.52,34.49,.025,.040,0),
    (134.63,34.38,.030,.050,-5),(134.51,34.31,.030,.040,0),
])

# A scattering of remembered small but sample-scale islets.
add_chain([
    (131.55,34.04,.025,.025,0),(131.95,34.12,.020,.028,0),
    (132.38,34.05,.022,.032,0),(132.57,34.30,.022,.035,0),
    (132.73,34.43,.020,.025,0),(133.34,34.18,.022,.030,0),
    (133.43,34.30,.020,.025,0),(133.60,34.41,.018,.025,0),
    (134.00,34.34,.020,.025,0),(134.18,34.36,.020,.028,0),
])

# Carve the two large bays whose concave shores are not captured by the
# single outer Honshu polygon.
cut_poly([  # Ise Bay
    (136.66,35.00),(136.89,35.00),(136.86,34.88),(136.82,34.73),
    (136.88,34.61),(136.78,34.54),(136.67,34.59),(136.57,34.72),
    (136.55,34.86)
])
cut_poly([  # Hakata Bay
    (130.18,33.70),(130.28,33.73),(130.42,33.70),(130.48,33.64),
    (130.39,33.58),(130.25,33.59)
])

# Restore headlands and peninsulas around the hand-cut bays.
add_poly([(136.82,35.00),(137.00,35.00),(137.00,34.60),(136.94,34.61),
          (136.89,34.72),(136.88,34.86)])  # Chita/Atsumi edge
add_poly([(130.05,33.62),(130.16,33.68),(130.20,33.64),(130.15,33.53),
          (130.05,33.50),(129.98,33.55)])  # Itoshima

# Islands at Ise Bay's mouth and along the ragged Shima/Kumano coast.
add_chain([
    (136.98,34.55,.035,.025,15),(136.91,34.52,.020,.030,-10),
    (136.88,34.42,.025,.035,0),(136.79,34.33,.020,.030,-10),
    (136.67,34.25,.018,.025,0),(136.48,34.14,.020,.025,0),
    (136.17,33.91,.018,.025,0),(135.78,33.47,.030,.025,0),
])

# Islands of Genkai-nada and off Yamaguchi's Japan-Sea coast.
add_chain([
    (129.92,33.98,.040,.032,-10),(130.04,34.24,.028,.040,0),
    (130.33,34.24,.025,.022,0),(130.49,34.20,.020,.025,0),
    (130.68,34.22,.025,.035,-10),(130.82,34.27,.020,.028,0),
    (131.13,34.77,.035,.055,-10),(131.39,34.50,.025,.035,0),
    (131.83,34.54,.020,.025,0),(132.08,34.62,.018,.025,0),
])

# Additional near-shore islands of western Kyushu.
add_chain([
    (129.64,33.48,.025,.040,-10),(129.68,33.37,.020,.030,5),
    (129.73,33.22,.025,.040,-10),(129.79,33.10,.018,.030,0),
    (129.90,33.06,.020,.025,0),(130.01,33.12,.018,.025,0),
])

# Export and perform strict structural checks.
out = Path("outputs")
out.mkdir(parents=True, exist_ok=True)
matrix = land.astype(np.uint8)

csv_path = out / "candidate.csv"
with csv_path.open("w", newline="", encoding="ascii") as f:
    csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

raw = Image.fromarray(matrix * 255, mode="L")
raw_path = out / "candidate.png"
big_path = out / "candidate_8x.png"
raw.save(raw_path)
big = raw.resize((COLS * 8, ROWS * 8), resample=Image.Resampling.NEAREST)
big.save(big_path)

assert matrix.shape == (64, 253)
assert set(np.unique(matrix)).issubset({0, 1})
assert raw.size == (253, 64) and big.size == (2024, 512)
assert set(raw.getdata()).issubset({0, 255})
assert set(big.getdata()).issubset({0, 255})
assert np.array_equal(np.asarray(big)[::8, ::8], np.asarray(raw))

with csv_path.open(encoding="ascii") as f:
    check = [line.rstrip("\n").split(",") for line in f]
assert len(check) == 64 and all(len(row) == 253 for row in check)
assert all(value in {"0", "1"} for row in check for value in row)

