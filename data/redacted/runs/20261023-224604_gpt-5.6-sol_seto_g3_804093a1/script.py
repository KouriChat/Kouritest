"""Knowledge-only reconstruction of the Seto Inland Sea region.

All vertices and island parameters below were written from geographic memory.
No external files, web services, map libraries, or geographic databases are
read.  The rules are polygon/ellipse inclusion rules, evaluated automatically
at the requested sample points.
"""
from pathlib import Path
import csv
import math
import numpy as np
from PIL import Image

W, H = 253, 64
LONS = 129.0 + np.arange(W) * (2.0 / 63.0)
LATS = 35.0 - np.arange(H) * (2.0 / 63.0)
X, Y = np.meshgrid(LONS, LATS)
land = np.zeros((H, W), dtype=bool)

def polygon(vertices):
    """Union a simple lon/lat polygon into the land mask (ray casting)."""
    global land
    inside = np.zeros_like(land)
    xj, yj = vertices[-1]
    for xi, yi in vertices:
        cross = ((yi > Y) != (yj > Y))
        xhit = (xj-xi) * (Y-yi) / ((yj-yi) if yj != yi else 1e-30) + xi
        inside ^= cross & (X < xhit)
        xj, yj = xi, yi
    land |= inside

def ellipse(cx, cy, rx, ry, angle=0.0):
    """Union a rotated geographic ellipse (used only for small islands)."""
    global land
    a = math.radians(angle)
    dx, dy = X-cx, Y-cy
    u = dx*math.cos(a) + dy*math.sin(a)
    v = -dx*math.sin(a) + dy*math.cos(a)
    land |= (u/rx)**2 + (v/ry)**2 <= 1.0

# Western Honshu, Kii Peninsula, and the land surrounding Ise Bay.  The long
# lower edge follows the Seto/Osaka/Kii/Ise shore; the return follows Japan Sea.
polygon([
 (130.84,34.02),(130.94,34.10),(131.04,34.07),(131.10,34.00),
 (131.22,33.96),(131.36,33.94),(131.48,34.00),(131.60,33.98),
 (131.72,34.05),(131.88,34.00),(132.03,34.08),(132.17,34.10),
 (132.23,34.17),(132.28,34.27),(132.31,34.36),(132.42,34.38),
 (132.48,34.31),(132.45,34.24),(132.52,34.20),(132.55,34.11),
 (132.64,34.15),(132.69,34.22),(132.80,34.20),(132.88,34.27),
 (133.00,34.31),(133.08,34.37),(133.20,34.36),(133.30,34.42),
 (133.42,34.43),(133.50,34.38),(133.58,34.43),(133.64,34.49),
 (133.75,34.51),(133.83,34.46),(133.80,34.36),(133.89,34.32),
 (134.00,34.38),(134.10,34.37),(134.21,34.43),(134.32,34.48),
 (134.45,34.54),(134.58,34.62),(134.75,34.68),(134.89,34.65),
 (135.02,34.62),(135.16,34.67),(135.31,34.68),(135.43,34.62),
 (135.46,34.53),(135.38,34.43),(135.29,34.35),(135.20,34.26),
 (135.12,34.14),(135.08,33.99),(135.15,33.84),(135.25,33.70),
 (135.39,33.58),(135.57,33.48),(135.76,33.42),(135.88,33.48),
 (135.95,33.59),(136.07,33.69),(136.19,33.83),(136.28,33.96),
 (136.40,34.06),(136.53,34.13),(136.62,34.20),(136.76,34.25),
 (136.86,34.32),(136.77,34.40),(136.69,34.49),(136.66,34.62),
 (136.65,34.75),(136.70,34.88),(136.78,35.08),
 (136.91,35.12),(136.93,34.96),(136.90,34.82),(136.88,34.69),
 (136.96,34.59),(137.08,34.55),(137.25,34.61),(137.30,35.20),
 (132.12,35.20),(132.08,35.03),(131.99,34.90),(131.87,34.78),
 (131.72,34.66),(131.57,34.54),(131.43,34.43),(131.29,34.34),
 (131.15,34.26),(131.04,34.17),(130.94,34.10)
])

# Kyushu: northern shore, Kanmon Strait, Suo Sea, Kunisaki and Oita coast.
polygon([
 (129.65,32.85),(129.69,33.05),(129.76,33.18),(129.82,33.33),
 (129.78,33.47),(129.84,33.57),(129.96,33.61),(130.06,33.58),
 (130.17,33.59),(130.25,33.55),(130.32,33.57),(130.39,33.64),
 (130.50,33.68),(130.60,33.76),(130.70,33.84),(130.79,33.90),
 (130.88,33.96),(130.94,33.93),(130.98,33.86),(131.08,33.80),
 (131.15,33.72),(131.27,33.68),(131.38,33.60),(131.49,33.59),
 (131.58,33.65),(131.69,33.70),(131.77,33.65),(131.78,33.55),
 (131.70,33.46),(131.60,33.39),(131.59,33.31),(131.69,33.25),
 (131.80,33.24),(131.91,33.15),(132.02,32.85)
])
# Shikoku, including its very narrow western Sada Peninsula.
polygon([
 (131.95,33.34),(132.08,33.39),(132.22,33.45),(132.37,33.50),
 (132.49,33.59),(132.58,33.70),(132.67,33.80),(132.75,33.88),
 (132.86,33.94),(132.96,34.04),(133.04,34.09),(133.14,34.10),
 (133.24,34.05),(133.35,33.99),(133.48,34.00),(133.59,34.06),
 (133.65,34.14),(133.76,34.20),(133.87,34.27),(134.00,34.34),
 (134.12,34.39),(134.25,34.38),(134.35,34.32),(134.44,34.26),
 (134.57,34.23),(134.65,34.16),(134.61,34.07),(134.54,34.00),
 (134.58,33.89),(134.52,33.78),(134.43,33.65),(134.33,33.48),
 (134.23,33.31),(134.15,33.23),(134.02,33.25),(133.88,33.32),
 (133.73,33.39),(133.62,33.46),(133.52,33.49),(133.40,33.44),
 (133.28,33.36),(133.16,33.27),(133.05,33.14),(132.98,33.02),
 (132.96,32.82),(132.65,32.80),(132.57,32.98),(132.50,33.13),
 (132.44,33.25),(132.35,33.32),(132.23,33.33),(132.10,33.30)
])

# Awaji-shima.
polygon([
 (134.62,34.60),(134.70,34.64),(134.82,34.61),(134.94,34.57),
 (135.00,34.49),(134.99,34.39),(134.94,34.30),(134.86,34.22),
 (134.74,34.18),(134.65,34.23),(134.62,34.34),(134.59,34.45)
])

# Tsushima is shown as its two main masses, separated by a narrow drowned cut.
polygon([
 (129.17,34.08),(129.25,34.11),(129.30,34.20),(129.34,34.31),
 (129.39,34.38),(129.35,34.46),(129.28,34.43),(129.22,34.34),
 (129.18,34.23)
])
polygon([
 (129.35,34.43),(129.42,34.47),(129.49,34.57),(129.49,34.67),
 (129.43,34.72),(129.36,34.68),(129.31,34.58),(129.30,34.50)
])

# Iki, Hirado, the northern Goto islands, and a few near-Kyushu islands.
ellipse(129.70,33.78,0.115,0.095,-15)
polygon([(129.45,33.16),(129.52,33.20),(129.56,33.30),(129.53,33.39),
         (129.46,33.43),(129.41,33.35),(129.42,33.25)])
ellipse(129.29,33.18,0.055,0.085,-18)
ellipse(129.12,33.10,0.055,0.105,-22)
ellipse(129.04,33.00,0.045,0.075,-20)
ellipse(129.69,33.57,0.045,0.035,0)
ellipse(130.04,33.69,0.035,0.055,-25)

# Western Inland Sea islands: Yashiro, Miyajima, Etajima/Kuraha-shima,
# and the larger stepping stones toward the Geiyo archipelago.
ellipse(132.25,33.91,0.175,0.060,-12)  # Suo-Oshima
ellipse(132.29,34.27,0.105,0.035,-30)  # Itsukushima
ellipse(132.44,34.22,0.070,0.095,-15)  # Etajima
ellipse(132.51,34.10,0.080,0.095,18)   # Kurahashi
ellipse(132.67,34.12,0.050,0.060,-10)
ellipse(132.75,34.20,0.060,0.050,0)
ellipse(132.88,34.20,0.090,0.055,-8)   # Osaki-kamijima

# Geiyo islands.  Separate ellipses retain the narrow tidal channels.
ellipse(132.99,34.26,0.090,0.075,-12)  # Omishima
ellipse(133.09,34.25,0.080,0.050,5)    # Ikuchijima
ellipse(133.16,34.30,0.065,0.075,-10)  # Innoshima
ellipse(133.18,34.37,0.065,0.040,-10)  # Mukaishima
ellipse(133.08,34.18,0.052,0.055,0)    # Hakata
ellipse(133.00,34.13,0.070,0.075,12)   # Oshima
ellipse(132.93,34.09,0.035,0.045,0)

# Central/eastern Inland Sea islands.
ellipse(133.53,34.30,0.050,0.035,15)
ellipse(133.66,34.28,0.055,0.030,-10)
ellipse(133.78,34.31,0.060,0.035,15)
ellipse(133.87,34.35,0.040,0.050,0)
ellipse(133.97,34.46,0.040,0.052,-8)   # Naoshima
ellipse(134.08,34.48,0.055,0.065,5)    # Teshima
ellipse(134.26,34.50,0.195,0.115,8)    # Shodoshima
ellipse(134.46,34.44,0.040,0.060,0)
# Further islands large enough to affect this sampling lattice.
ellipse(131.68,33.72,0.040,0.025,0)    # Hime-shima
ellipse(131.91,33.49,0.025,0.045,-15)
ellipse(132.04,33.37,0.022,0.040,0)
ellipse(132.35,33.76,0.040,0.060,-15)
ellipse(132.49,33.87,0.030,0.050,-10)
ellipse(132.59,33.92,0.035,0.060,-10)
ellipse(132.68,33.99,0.025,0.040,-10)
ellipse(131.99,33.94,0.035,0.025,0)
ellipse(131.73,33.97,0.035,0.022,0)
ellipse(132.09,34.04,0.025,0.025,0)
ellipse(132.35,34.05,0.035,0.050,0)
ellipse(132.56,34.01,0.035,0.055,0)
ellipse(132.75,34.03,0.022,0.035,0)
ellipse(133.30,34.10,0.025,0.030,0)
ellipse(133.39,34.20,0.026,0.030,0)
ellipse(133.48,34.35,0.035,0.025,0)
ellipse(133.58,34.37,0.030,0.035,0)
ellipse(133.72,34.39,0.035,0.025,0)
ellipse(134.63,34.30,0.025,0.035,0)
ellipse(134.72,34.11,0.030,0.025,0)   # Naruto-channel islets
ellipse(135.00,34.29,0.035,0.025,0)   # Tomogashima group
ellipse(134.92,34.16,0.025,0.040,0)
ellipse(136.88,34.23,0.022,0.030,0)

def water_polygon(vertices):
    """Cut an explicitly remembered bay out of an existing land polygon."""
    global land
    inside = np.zeros_like(land)
    xj, yj = vertices[-1]
    for xi, yi in vertices:
        cross = ((yi > Y) != (yj > Y))
        xhit = (xj-xi) * (Y-yi) / ((yj-yi) if yj != yi else 1e-30) + xi
        inside ^= cross & (X < xhit)
        xj, yj = xi, yi
    land &= ~inside

# Omura and Ariake bays are appreciable water bodies within the coarse Kyushu
# outline.  Both cuts extend below the requested frame, as their mouths do.
water_polygon([
 (129.77,32.85),(130.03,32.85),(130.04,33.00),(130.01,33.12),
 (129.95,33.19),(129.86,33.20),(129.80,33.14),(129.78,33.03)
])
water_polygon([
 (130.08,32.82),(130.76,32.82),(130.66,32.96),(130.56,33.08),
 (130.43,33.16),(130.29,33.18),(130.19,33.12),(130.15,33.02)
])

# A few islands in those bays are restored after cutting the water.
ellipse(129.88,33.13,0.018,0.025,0)
ellipse(130.20,33.10,0.018,0.030,0)

# Export and perform all requested structural checks.
out = Path("outputs")
out.mkdir(parents=True, exist_ok=True)
matrix = land.astype(np.uint8)
assert matrix.shape == (64, 253)
assert matrix.size == 16192
assert set(np.unique(matrix)).issubset({0, 1})

with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
    csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

raw_pixels = matrix * 255
raw = Image.fromarray(raw_pixels, mode="L")
raw.save(out / "candidate.png")
large = raw.resize((W*8, H*8), resample=Image.Resampling.NEAREST)
large.save(out / "candidate_8x.png")

# Validate the serialized CSV and both monochrome images.
with (out / "candidate.csv").open(newline="", encoding="ascii") as f:
    rows = list(csv.reader(f))
assert len(rows) == 64 and all(len(r) == 253 for r in rows)
assert all(v in ("0", "1") for r in rows for v in r)
with Image.open(out / "candidate.png") as im:
    check_raw = np.asarray(im)
    assert im.size == (253, 64) and set(np.unique(check_raw)) <= {0, 255}
with Image.open(out / "candidate_8x.png") as im:
    check_big = np.asarray(im)
    assert im.size == (2024, 512) and set(np.unique(check_big)) <= {0, 255}
assert np.array_equal(check_big, np.repeat(np.repeat(check_raw, 8, 0), 8, 1))

print("Generated 64x253 matrix and 253x64 / 2024x512 monochrome PNG files.")
