"""Knowledge-only reconstruction of land/water in 129--137 E, 33--35 N.

The vertices below were written from geographic knowledge, not obtained from
map data.  Classification is the union of hand-drawn polygons and ellipses,
evaluated at each requested WGS84 sample point.
"""
from pathlib import Path
import csv
import math
import numpy as np
from PIL import Image

W, H = 253, 64
lons = 129.0 + np.arange(W) * (8.0 / 252.0)
lats = 35.0 - np.arange(H) * (2.0 / 63.0)
X, Y = np.meshgrid(lons, lats)
land = np.zeros((H, W), dtype=bool)


def polygon(points):
    """Add a polygon using an even/odd point test at the sample coordinates."""
    global land
    q = np.zeros_like(land)
    xj, yj = points[-1]
    for xi, yi in points:
        crossing = ((yi > Y) != (yj > Y))
        xcut = (xj - xi) * (Y - yi) / ((yj - yi) + 1e-30) + xi
        q ^= crossing & (X < xcut)
        xj, yj = xi, yi
    land |= q


def ellipse(cx, cy, rx, ry, angle=0.0):
    """Add an elliptical small island (angle in degrees)."""
    global land
    a = math.radians(angle)
    dx, dy = X - cx, Y - cy
    u = dx * math.cos(a) + dy * math.sin(a)
    v = -dx * math.sin(a) + dy * math.cos(a)
    land |= (u / rx) ** 2 + (v / ry) ** 2 <= 1.0


# Southern Korean coast at the extreme north-west edge.
polygon([
    (128.80,35.35),(129.55,35.35),(129.52,35.16),(129.42,35.10),
    (129.34,35.08),(129.29,35.02),(129.19,35.02),(129.13,35.07),
    (129.08,35.12),(129.02,35.10),(128.95,35.05),(128.80,35.03)
])

# Northern Kyushu: a deliberately articulated coast, continuing south of frame.
polygon([
    (128.85,32.70),(132.05,32.70),
    (132.03,33.05),(131.92,33.18),(131.82,33.28),(131.70,33.36),
    (131.58,33.39),(131.47,33.48),(131.34,33.55),(131.16,33.60),
    (131.03,33.72),(130.94,33.82),(130.88,33.92),(130.78,34.02),
    (130.67,34.05),(130.58,34.01),(130.49,33.91),(130.42,33.84),
    (130.31,33.79),(130.20,33.70),(130.13,33.62),(130.07,33.55),
    (129.98,33.49),(129.90,33.40),(129.84,33.32),(129.77,33.25),
    (129.70,33.18),(129.62,33.13),(129.56,33.07),(129.49,33.00),
    (129.39,32.91),(129.28,32.84),(129.15,32.79)
])

# Fukuoka peninsula and bays: land additions around Hakata and Karatsu.
polygon([(129.83,33.30),(129.91,33.42),(130.03,33.53),(130.13,33.61),
         (130.19,33.68),(130.12,33.72),(130.03,33.69),(129.95,33.61),
         (129.86,33.55),(129.78,33.45)])
polygon([(130.18,33.65),(130.28,33.73),(130.35,33.82),(130.41,33.91),
         (130.34,33.96),(130.25,33.92),(130.20,33.84),(130.13,33.75)])
# Honshu mainland.  Its northern side is closed outside the requested frame.
# The long lower edge follows Yamaguchi, Hiroshima, Okayama, Osaka, Kii and Ise.
polygon([
    (131.05,35.35),(137.30,35.35),(137.30,34.18),
    # Chita peninsula and east shore of Ise Bay
    (136.98,34.68),(136.94,34.62),(136.91,34.55),(136.88,34.47),
    (136.85,34.39),(136.82,34.34),(136.79,34.43),(136.77,34.56),
    (136.78,34.69),(136.83,34.79),(136.91,34.87),
    # Nagoya / inner Ise Bay, then western shore
    (136.90,34.96),(136.82,35.00),(136.73,35.00),(136.66,34.97),
    (136.63,34.90),(136.61,34.82),(136.57,34.75),(136.53,34.67),
    (136.50,34.58),(136.47,34.50),(136.43,34.42),(136.39,34.35),
    (136.34,34.28),(136.29,34.23),(136.25,34.18),(136.20,34.14),
    # Shima peninsula's deeply indented coast
    (136.14,34.12),(136.09,34.14),(136.05,34.10),(136.00,34.08),
    (135.96,34.03),(135.91,34.02),(135.86,33.98),(135.78,33.93),
    # Kii peninsula Pacific coast, south tip and west side
    (135.73,33.86),(135.69,33.77),(135.66,33.67),(135.66,33.57),
    (135.70,33.49),(135.76,33.43),(135.82,33.39),(135.86,33.34),
    (135.81,33.28),(135.72,33.25),(135.62,33.27),(135.52,33.32),
    (135.43,33.39),(135.34,33.47),(135.27,33.56),(135.20,33.65),
    (135.13,33.75),(135.08,33.85),(135.03,33.96),(134.98,34.08),
    (134.94,34.18),(134.91,34.27),(134.91,34.33),
    # Osaka Bay north/east shore and narrow Akashi side
    (134.96,34.38),(135.04,34.42),(135.14,34.48),(135.25,34.55),
    (135.36,34.59),(135.45,34.64),(135.43,34.70),(135.34,34.72),
    (135.25,34.69),(135.14,34.66),(135.05,34.64),(134.96,34.63),
    (134.88,34.65),(134.81,34.69),(134.77,34.73),
    # Harima coast westward
    (134.70,34.72),(134.62,34.72),(134.53,34.70),(134.44,34.67),
    (134.34,34.66),(134.24,34.64),(134.15,34.62),(134.06,34.59),
    (133.96,34.56),(133.86,34.53),(133.77,34.51),
    # Kojima peninsula / Okayama and islands-facing coast
    (133.68,34.49),(133.62,34.45),(133.58,34.40),(133.51,34.39),
    (133.46,34.43),(133.42,34.48),(133.35,34.47),(133.29,34.43),
    (133.23,34.40),(133.17,34.43),(133.10,34.47),(133.02,34.46),
    (132.94,34.42),(132.87,34.39),(132.79,34.36),
    # Hiroshima Bay: east shore in, then peninsulas and west shore
    (132.72,34.32),(132.65,34.29),(132.61,34.25),(132.58,34.20),
    (132.54,34.17),(132.51,34.22),(132.49,34.29),(132.45,34.35),
    (132.40,34.38),(132.34,34.35),(132.31,34.29),(132.27,34.23),
    (132.22,34.18),(132.17,34.15),(132.11,34.12),(132.06,34.07),
    (132.00,34.03),(131.93,34.01),(131.87,34.03),(131.80,34.06),
    # Suo-nada / Yamaguchi coast
    (131.73,34.04),(131.67,34.00),(131.60,33.96),(131.52,33.93),
    (131.44,33.91),(131.37,33.93),(131.31,33.98),(131.24,34.02),
    (131.17,34.05),(131.10,34.07),(131.04,34.11),(130.99,34.17),
    # Kanmon north bank and Sea of Japan side
    (130.94,34.22),(130.90,34.27),(130.88,34.34),(130.90,34.40),
    (130.95,34.45),(131.02,34.50),(131.09,34.57),(131.15,34.66),
    (131.20,34.75),(131.23,34.86),(131.20,34.97),(131.12,35.08)
])
# Shikoku, clockwise from Cape Sada along the Inland Sea shore.
polygon([
    (132.00,33.33),(132.06,33.41),(132.12,33.48),(132.18,33.55),
    (132.23,33.63),(132.28,33.72),(132.34,33.80),(132.41,33.86),
    (132.48,33.91),(132.55,33.95),(132.62,34.00),(132.69,34.06),
    (132.77,34.11),(132.86,34.15),(132.95,34.18),(133.03,34.20),
    (133.12,34.22),(133.20,34.25),(133.28,34.29),(133.36,34.31),
    (133.45,34.34),(133.54,34.37),(133.63,34.39),(133.72,34.39),
    (133.81,34.37),(133.90,34.34),(133.99,34.31),(134.08,34.27),
    (134.17,34.22),(134.25,34.17),(134.32,34.12),(134.39,34.08),
    (134.46,34.05),(134.53,34.04),(134.59,34.07),
    # Naruto corner and east coast
    (134.64,34.12),(134.68,34.18),(134.73,34.22),(134.78,34.21),
    (134.82,34.16),(134.82,34.08),(134.80,34.00),(134.77,33.91),
    (134.73,33.82),(134.69,33.73),(134.65,33.64),(134.60,33.55),
    (134.55,33.47),(134.49,33.40),(134.43,33.35),(134.36,33.30),
    (134.28,33.25),(134.20,33.21),(134.13,33.17),(134.05,33.13),
    (133.97,33.08),(133.90,33.03),(133.82,32.96),
    # south side is mostly below the frame; return along Kochi/Ehime
    (132.65,32.86),(132.52,32.92),(132.42,33.00),(132.33,33.08),
    (132.25,33.16),(132.16,33.23),(132.08,33.28)
])

# Awaji: narrow north end, broad middle, pointed south-west.
polygon([
    (134.99,34.61),(134.94,34.57),(134.91,34.51),(134.88,34.43),
    (134.84,34.35),(134.80,34.27),(134.76,34.20),(134.71,34.15),
    (134.67,34.20),(134.68,34.28),(134.71,34.36),(134.75,34.45),
    (134.80,34.52),(134.86,34.57),(134.92,34.61)
])

# Major islands west to east. Ellipses are intentionally no smaller than a
# sample can meaningfully resolve unless their centre nearly hits a sample.
# Tsushima pieces and Iki.
ellipse(129.28,34.60,0.10,0.25,-18)
ellipse(129.35,34.36,0.09,0.18,-15)
ellipse(129.69,33.79,0.12,0.09,-10)

# Hirado, Takashima and offshore north-Kyushu islands.
ellipse(129.55,33.28,0.055,0.16,-12)
ellipse(129.43,33.18,0.035,0.07,-10)
ellipse(129.76,33.58,0.035,0.055,0)
ellipse(130.10,33.90,0.035,0.035,0)
ellipse(130.35,34.00,0.025,0.025,0)

# Himeshima in the western Suo-nada.
ellipse(131.65,33.73,0.07,0.025,0)

# Yashiro/Suo-Oshima and nearby.
ellipse(132.28,33.91,0.19,0.055,-30)
ellipse(132.13,33.84,0.045,0.025,-15)

# Hiroshima Bay: Itsukushima, Etajima/Nomi, Kurahashi.
ellipse(132.30,34.27,0.035,0.12,-25)
ellipse(132.45,34.20,0.07,0.13,-18)
ellipse(132.50,34.08,0.07,0.11,-28)
ellipse(132.39,34.05,0.045,0.065,-20)
# Aki-nada and Geiyo archipelago (larger islands plus resolvable stepping stones).
ellipse(132.62,34.18,0.055,0.090,-20)  # Shimokamagari/Kamikamagari
ellipse(132.72,34.16,0.060,0.105,-18)
ellipse(132.82,34.12,0.080,0.120,-25)  # Osaki-shimojima
ellipse(132.88,34.20,0.050,0.075,-15)
ellipse(132.94,34.10,0.055,0.090,-12)
ellipse(133.03,34.16,0.055,0.105,-15)  # Omishima
ellipse(133.10,34.23,0.065,0.085,-10)
ellipse(133.16,34.19,0.045,0.080,-20)  # Hakata
ellipse(133.19,34.28,0.042,0.070,-12)
ellipse(133.05,34.27,0.030,0.055,-5)

# Kutsuna islands west of Matsuyama.
ellipse(132.63,33.97,0.040,0.075,-15)
ellipse(132.67,33.90,0.030,0.055,-5)
ellipse(132.55,33.94,0.025,0.045,-10)

# Bingo / central Seto islands.
ellipse(133.13,34.34,0.030,0.050,0)
ellipse(133.20,34.32,0.035,0.060,-10)
ellipse(133.27,34.36,0.030,0.050,-15)
ellipse(133.37,34.35,0.030,0.060,-8)
ellipse(133.47,34.31,0.035,0.055,0)
ellipse(133.53,34.35,0.025,0.045,0)

# Kasaoka islands.
ellipse(133.50,34.45,0.028,0.045,0)
ellipse(133.54,34.40,0.025,0.050,0)
ellipse(133.58,34.34,0.025,0.055,-8)

# Shiwaku group and islands around Bisan Seto.
ellipse(133.65,34.34,0.025,0.040,0)
ellipse(133.75,34.32,0.035,0.055,-10)
ellipse(133.82,34.34,0.025,0.045,0)
ellipse(133.89,34.32,0.028,0.045,5)
ellipse(133.97,34.34,0.030,0.050,10)
ellipse(134.06,34.38,0.035,0.060,5)
ellipse(134.13,34.41,0.027,0.045,0)

# Shodoshima and immediate satellites.
ellipse(134.27,34.50,0.19,0.105,-12)
ellipse(134.36,34.47,0.08,0.10,-10)
ellipse(134.18,34.43,0.032,0.045,0)

# Naoshima/Teshima/Inujima between Takamatsu and Okayama.
ellipse(133.98,34.47,0.055,0.055,0)
ellipse(134.08,34.49,0.050,0.040,10)
ellipse(134.18,34.48,0.025,0.035,0)
ellipse(134.10,34.56,0.025,0.030,0)

# Harima-nada islands: Ie, Boze, Awashima.
ellipse(134.51,34.66,0.050,0.055,-10)
ellipse(134.57,34.54,0.040,0.030,0)
ellipse(134.70,34.50,0.025,0.035,0)

# Osaka Bay small islands only where likely to hit this coarse lattice.
ellipse(135.05,34.43,0.025,0.045,-15)  # Tomogashima

# Kii Channel islands.
ellipse(134.93,34.28,0.025,0.040,-15)
ellipse(134.99,34.18,0.020,0.035,-10)

# Ise Bay mouth: Kamishima/Toshijima/Sugashima.
ellipse(136.98,34.55,0.022,0.045,-12)
ellipse(136.90,34.51,0.045,0.050,-15)
ellipse(136.87,34.48,0.026,0.040,5)
ellipse(136.84,34.42,0.024,0.035,0)

# A few Pacific islands near Kii/Shima.
ellipse(135.88,33.47,0.025,0.045,0)
ellipse(136.02,33.57,0.020,0.035,0)
ellipse(136.08,33.92,0.018,0.030,0)

# Write and rigorously validate all requested artifacts.
out = Path("outputs")
out.mkdir(parents=True, exist_ok=True)
matrix = land.astype(np.uint8)
assert matrix.shape == (64, 253)
assert set(np.unique(matrix)).issubset({0, 1})

with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
    csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

raw = Image.fromarray(matrix * 255, mode="L")
raw.save(out / "candidate.png")
large = raw.resize((W * 8, H * 8), resample=Image.Resampling.NEAREST)
large.save(out / "candidate_8x.png")

with (out / "candidate.csv").open(newline="", encoding="ascii") as f:
    rows = list(csv.reader(f))
assert len(rows) == 64 and all(len(r) == 253 for r in rows)
assert all(v in ("0", "1") for r in rows for v in r)
assert raw.size == (253, 64) and large.size == (2024, 512)
assert set(raw.getdata()).issubset({0, 255})
assert set(large.getdata()).issubset({0, 255})
print("generated 64x253 matrix; land points:", int(matrix.sum()))
