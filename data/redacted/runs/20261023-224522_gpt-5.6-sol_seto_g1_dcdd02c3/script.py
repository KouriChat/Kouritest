"""Knowledge-only reconstruction of the Seto Inland Sea region.

All vertices below were written from memory as deliberately approximate
coastlines.  No map, geographic file, web service, or point classifier is
read by this program.
"""
from pathlib import Path
import csv
import math
import numpy as np
from PIL import Image

W, H = 253, 64
xs = np.array([129.0 + j * (2.0 / 63.0) for j in range(W)])
ys = np.array([35.0 - i * (2.0 / 63.0) for i in range(H)])
X, Y = np.meshgrid(xs, ys)


def inside_polygon(vertices):
    """Vectorized even/odd point-in-polygon test."""
    ans = np.zeros((H, W), dtype=bool)
    x0, y0 = vertices[-1]
    for x1, y1 in vertices:
        cross = ((y1 > Y) != (y0 > Y))
        xhit = (x0 - x1) * (Y - y1) / ((y0 - y1) + 1e-300) + x1
        ans ^= cross & (X < xhit)
        x0, y0 = x1, y1
    return ans


def ellipse(cx, cy, rx, ry, angle=0.0):
    """Rotated ellipse, used only as a hand-written small-island outline."""
    a = math.radians(angle)
    dx, dy = X - cx, Y - cy
    u = dx * math.cos(a) + dy * math.sin(a)
    v = -dx * math.sin(a) + dy * math.cos(a)
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1.0


land = np.zeros((H, W), dtype=bool)

# Western Honshu, Kansai, and the Kii peninsula.  The top edge is merely the
# clipping boundary; the long lower chain follows the remembered south coast.
honshu = [
    (130.77, 35.01), (136.62, 35.01),
    (136.61, 34.94), (136.58, 34.84), (136.53, 34.73),
    (136.53, 34.62), (136.62, 34.52), (136.76, 34.47),
    (136.85, 34.50), (136.91, 34.43), (136.88, 34.34),
    (136.79, 34.25), (136.66, 34.29), (136.55, 34.20),
    (136.42, 34.10), (136.31, 33.97), (136.21, 33.82),
    (136.04, 33.63), (135.88, 33.48), (135.76, 33.42),
    (135.60, 33.45), (135.45, 33.53), (135.34, 33.67),
    (135.26, 33.83), (135.20, 34.00), (135.15, 34.13),
    (135.06, 34.27), (135.08, 34.35), (135.18, 34.43),
    (135.31, 34.52), (135.43, 34.56), (135.45, 34.63),
    (135.39, 34.67), (135.27, 34.68), (135.16, 34.68),
    (135.03, 34.64), (134.91, 34.66), (134.78, 34.70),
    (134.67, 34.73), (134.54, 34.73), (134.43, 34.75),
    (134.34, 34.68), (134.22, 34.63), (134.11, 34.61),
    (134.02, 34.58), (133.99, 34.50), (133.88, 34.43),
    (133.76, 34.44), (133.79, 34.51), (133.69, 34.54),
    (133.55, 34.49), (133.40, 34.47), (133.28, 34.41),
    (133.11, 34.39), (132.98, 34.34), (132.84, 34.29),
    (132.69, 34.25), (132.59, 34.19), (132.52, 34.23),
    (132.48, 34.31), (132.39, 34.35), (132.30, 34.29),
    (132.24, 34.20), (132.12, 34.13), (131.99, 34.08),
    (131.84, 34.05), (131.70, 34.05), (131.58, 34.01),
    (131.43, 33.96), (131.27, 33.95), (131.13, 33.96),
    (131.02, 33.93), (130.92, 33.96), (130.86, 34.06),
    (130.87, 34.19), (130.94, 34.36), (130.90, 34.57),
]
land |= inside_polygon(honshu)

# Chita peninsula: its connection to Honshu lies north of this crop.
chita = [
    (136.82, 35.01), (137.01, 35.01), (137.01, 34.93),
    (136.98, 34.87), (136.99, 34.79), (137.01, 34.71),
    (136.98, 34.67), (136.93, 34.70), (136.89, 34.78),
    (136.88, 34.88), (136.84, 34.95),
]
land |= inside_polygon(chita)
# Northern Kyushu.  Bays which materially affect this crop are cut out below.
kyushu = [
    (129.58, 32.99), (129.56, 33.10), (129.61, 33.23),
    (129.68, 33.34), (129.79, 33.43), (129.91, 33.48),
    (130.01, 33.57), (130.14, 33.62), (130.28, 33.63),
    (130.39, 33.68), (130.48, 33.77), (130.61, 33.83),
    (130.75, 33.90), (130.88, 33.96), (130.98, 33.92),
    (131.01, 33.84), (130.98, 33.75), (131.07, 33.68),
    (131.20, 33.66), (131.34, 33.63), (131.47, 33.67),
    (131.60, 33.71), (131.74, 33.67), (131.76, 33.59),
    (131.71, 33.50), (131.62, 33.42), (131.57, 33.34),
    (131.62, 33.26), (131.76, 33.22), (131.87, 33.14),
    (131.97, 33.03), (132.00, 32.99),
]
land |= inside_polygon(kyushu)

# Omura and Ariake bays, plus the bite of Beppu Bay.
omura_bay = [
    (129.76, 32.99), (129.75, 33.06), (129.79, 33.14),
    (129.87, 33.18), (129.97, 33.14), (130.01, 33.06),
    (130.00, 32.99),
]
ariake_bay = [
    (130.10, 32.99), (130.14, 33.07), (130.22, 33.16),
    (130.34, 33.22), (130.47, 33.21), (130.58, 33.14),
    (130.67, 32.99),
]
beppu_bay = [
    (131.43, 33.29), (131.51, 33.38), (131.62, 33.42),
    (131.71, 33.49), (131.72, 33.40), (131.66, 33.30),
    (131.57, 33.25),
]
land &= ~inside_polygon(omura_bay)
land &= ~inside_polygon(ariake_bay)
land &= ~inside_polygon(beppu_bay)

# Shikoku, including the long, narrow Sada peninsula in its northwest.
shikoku = [
    (131.96, 33.33), (132.07, 33.34), (132.20, 33.41),
    (132.31, 33.50), (132.39, 33.62), (132.52, 33.70),
    (132.62, 33.80), (132.69, 33.90), (132.78, 33.99),
    (132.90, 34.05), (132.99, 34.12), (133.08, 34.08),
    (133.17, 34.02), (133.29, 34.02), (133.40, 34.07),
    (133.51, 34.15), (133.58, 34.23), (133.70, 34.28),
    (133.84, 34.30), (133.96, 34.36), (134.09, 34.36),
    (134.23, 34.34), (134.35, 34.29), (134.46, 34.25),
    (134.58, 34.25), (134.64, 34.19), (134.62, 34.10),
    (134.66, 34.01), (134.65, 33.90), (134.61, 33.78),
    (134.55, 33.67), (134.47, 33.55), (134.36, 33.43),
    (134.25, 33.31), (134.18, 33.23), (134.09, 33.27),
    (134.02, 33.36), (133.93, 33.43), (133.83, 33.46),
    (133.70, 33.48), (133.57, 33.49), (133.43, 33.45),
    (133.30, 33.36), (133.19, 33.22), (133.12, 33.08),
    (133.08, 32.99), (132.46, 32.99), (132.40, 33.08),
    (132.34, 33.18), (132.25, 33.25), (132.13, 33.29),
]
land |= inside_polygon(shikoku)

# Awaji-shima.
awaji = [
    (134.67, 34.18), (134.72, 34.25), (134.78, 34.31),
    (134.82, 34.40), (134.88, 34.50), (134.93, 34.58),
    (135.00, 34.61), (135.05, 34.57), (135.03, 34.48),
    (135.00, 34.39), (134.96, 34.30), (134.91, 34.22),
    (134.84, 34.17), (134.76, 34.16),
]
land |= inside_polygon(awaji)
# Open embayments along the north Kyushu coast (their north sides are open).
karatsu_bay = [
    (129.86, 33.49), (129.94, 33.55), (130.03, 33.58),
    (130.11, 33.61), (129.88, 33.62),
]
hakata_bay = [
    (130.18, 33.62), (130.27, 33.56), (130.38, 33.56),
    (130.47, 33.62),
]
land &= ~inside_polygon(karatsu_bay)
land &= ~inside_polygon(hakata_bay)

# Tsushima (two principal masses separated by the very narrow central sound).
south_tsushima = [
    (129.13, 34.09), (129.17, 34.18), (129.19, 34.28),
    (129.24, 34.37), (129.31, 34.43), (129.36, 34.38),
    (129.34, 34.29), (129.29, 34.20), (129.27, 34.11),
    (129.21, 34.05),
]
north_tsushima = [
    (129.28, 34.34), (129.32, 34.42), (129.34, 34.53),
    (129.39, 34.64), (129.46, 34.70), (129.51, 34.65),
    (129.49, 34.55), (129.44, 34.46), (129.41, 34.37),
    (129.35, 34.31),
]
land |= inside_polygon(south_tsushima)
land |= inside_polygon(north_tsushima)

# Iki and the Hirado archipelago.
land |= ellipse(129.69, 33.79, 0.105, 0.085, -12)
hirado = [
    (129.34, 33.17), (129.37, 33.28), (129.40, 33.39),
    (129.46, 33.48), (129.52, 33.45), (129.55, 33.36),
    (129.53, 33.27), (129.48, 33.18), (129.41, 33.12),
]
land |= inside_polygon(hirado)
land |= ellipse(129.42, 33.43, 0.035, 0.095, -8)   # Ikitsuki
land |= ellipse(129.55, 33.10, 0.045, 0.060, 10)   # Takashima
land |= ellipse(129.61, 33.17, 0.040, 0.045, -15)  # Kuroshima
land |= ellipse(129.73, 33.28, 0.035, 0.045, 10)
land |= ellipse(129.86, 33.47, 0.040, 0.025, 15)
land |= ellipse(129.77, 33.57, 0.035, 0.020, -20)

# Bits of the Goto chain and Amakusa which enter the crop.
land |= ellipse(129.05, 33.01, 0.055, 0.100, 20)
land |= ellipse(129.17, 33.05, 0.025, 0.055, -5)
land |= ellipse(130.10, 33.02, 0.045, 0.090, 20)

# Islands of Suo-nada and the west end of the Inland Sea.
land |= ellipse(131.69, 33.72, 0.075, 0.022, 3)    # Hime-shima
land |= ellipse(131.31, 33.82, 0.030, 0.020, -5)
land |= ellipse(131.64, 33.88, 0.038, 0.020, 10)
land |= ellipse(131.87, 33.91, 0.055, 0.025, -10)
land |= ellipse(132.00, 33.89, 0.050, 0.025, -5)

# Yashiro (Suo-Oshima) has a recognizably long east-west outline.
yashiro = [
    (132.08, 33.91), (132.17, 33.93), (132.27, 33.98),
    (132.37, 34.01), (132.46, 34.00), (132.47, 33.96),
    (132.39, 33.92), (132.28, 33.88), (132.17, 33.86),
    (132.10, 33.87),
]
land |= inside_polygon(yashiro)
land |= ellipse(132.03, 34.02, 0.045, 0.025, -10)
land |= ellipse(132.18, 34.08, 0.035, 0.025, 0)
land |= ellipse(132.10, 33.79, 0.025, 0.018, 0)
# Hiroshima Bay.
land |= ellipse(132.28, 34.27, 0.105, 0.038, 48)  # Itsukushima
land |= ellipse(132.43, 34.28, 0.040, 0.052, -12) # Eta-jima north
land |= ellipse(132.45, 34.20, 0.085, 0.075, 35)  # Nomi-jima
land |= ellipse(132.51, 34.11, 0.100, 0.045, 55)  # Kurahashi-jima
land |= ellipse(132.39, 34.08, 0.030, 0.024, 0)
land |= ellipse(132.34, 34.12, 0.022, 0.035, 15)
land |= ellipse(132.55, 34.03, 0.036, 0.020, 0)
land |= ellipse(132.61, 34.17, 0.045, 0.025, 5)

# Iyo-nada / Matsuyama islands.
land |= ellipse(132.52, 33.97, 0.030, 0.055, -8)
land |= ellipse(132.59, 33.98, 0.052, 0.072, -20)
land |= ellipse(132.67, 33.99, 0.045, 0.066, 25)
land |= ellipse(132.69, 33.90, 0.036, 0.063, -12)
land |= ellipse(132.63, 33.86, 0.020, 0.035, 0)
land |= ellipse(132.76, 34.00, 0.028, 0.043, 20)
land |= ellipse(132.80, 34.08, 0.035, 0.045, -5)

# Aki-nada: a dense west-to-east chain.
land |= ellipse(132.68, 34.19, 0.056, 0.044, 5)
land |= ellipse(132.74, 34.16, 0.075, 0.047, 25)
land |= ellipse(132.81, 34.20, 0.055, 0.040, -20)
land |= ellipse(132.87, 34.25, 0.087, 0.070, 8)   # Osaki-kami
land |= ellipse(132.91, 34.15, 0.040, 0.050, -5)
land |= ellipse(132.96, 34.08, 0.058, 0.050, 15)

# Geiyo islands between Imabari and Onomichi.
land |= ellipse(133.00, 34.24, 0.105, 0.090, -12) # Omi-shima
land |= ellipse(133.10, 34.21, 0.055, 0.068, 5)   # Hakata-jima
land |= ellipse(133.01, 34.12, 0.090, 0.070, 15)  # Oshima
land |= ellipse(133.08, 34.31, 0.085, 0.045, 15)  # Ikuchi-jima
land |= ellipse(133.17, 34.29, 0.055, 0.065, 5)   # Innoshima
land |= ellipse(133.21, 34.36, 0.055, 0.035, 10)  # Mukaishima
land |= ellipse(133.16, 34.18, 0.032, 0.045, -5)
land |= ellipse(133.24, 34.09, 0.023, 0.032, 0)

# Secondary islands of the same chain; although simple ellipses, these remain
# point tests against explicit remembered positions, not image postprocessing.
geiyo_minor = [
    (132.58,34.26,.022,.018,0), (132.64,34.26,.030,.020,0),
    (132.72,34.28,.028,.020,10), (132.77,34.31,.028,.018,-10),
    (132.84,34.34,.025,.018,0), (132.91,34.35,.020,.020,0),
    (132.98,34.36,.024,.017,0), (133.02,34.40,.018,.014,0),
    (133.25,34.28,.018,.025,0), (133.30,34.31,.025,.018,0),
    (133.30,34.22,.020,.016,0), (133.36,34.15,.024,.016,0),
]
for item in geiyo_minor:
    land |= ellipse(*item)
# Shiwaku island group.
shiwaku = [
    (133.42,34.21,.022,.018,0), (133.51,34.13,.035,.022,-10),
    (133.55,34.29,.022,.030,0), (133.61,34.27,.060,.027,25),
    (133.68,34.33,.035,.025,0), (133.72,34.37,.070,.042,5),
    (133.79,34.37,.055,.042,-15), (133.86,34.36,.025,.020,0),
    (133.88,34.30,.022,.018,0), (133.91,34.40,.025,.018,0),
]
for item in shiwaku:
    land |= ellipse(*item)

# Islands off Okayama and Takamatsu.
land |= ellipse(133.97, 34.47, 0.042, 0.052, -12) # Naoshima
land |= ellipse(134.08, 34.49, 0.070, 0.046, 10)  # Teshima
land |= ellipse(134.10, 34.56, 0.030, 0.020, 5)   # Inujima
land |= ellipse(134.05, 34.39, 0.022, 0.050, 0)   # Megi
land |= ellipse(134.06, 34.44, 0.018, 0.035, 0)   # Ogi

# Shodo-shima, represented by its two broad lobes and narrow middle.
land |= ellipse(134.22, 34.49, 0.105, 0.070, -10)
land |= ellipse(134.34, 34.52, 0.100, 0.075, 14)
land |= ellipse(134.28, 34.50, 0.080, 0.040, 0)

# Harima-nada islands.
harima = [
    (134.43,34.44,.027,.020,0), (134.50,34.47,.020,.018,0),
    (134.52,34.61,.030,.018,0), (134.57,34.66,.072,.035,-8),
    (134.66,34.65,.045,.024,10), (134.72,34.55,.020,.018,0),
    (134.80,34.48,.024,.020,0), (134.84,34.39,.018,.022,0),
]
for item in harima:
    land |= ellipse(*item)

# Naruto/Kii Channel islets.
land |= ellipse(134.67, 34.25, 0.030, 0.020, 20)
land |= ellipse(134.94, 34.17, 0.026, 0.038, 5)   # Nushima
land |= ellipse(135.00, 34.28, 0.038, 0.025, 15)  # Tomogashima
land |= ellipse(135.06, 34.30, 0.022, 0.016, 0)

# Major man-made islands are genuine land for the requested point criterion.
land |= ellipse(135.24, 34.43, 0.042, 0.014, 45)  # Kansai airport
land |= ellipse(135.22, 34.63, 0.025, 0.012, 0)   # Kobe airport
land |= ellipse(135.18, 34.66, 0.025, 0.015, 0)   # Port Island
land |= ellipse(135.27, 34.68, 0.035, 0.014, 0)   # Rokko Island

# Ise Bay mouth, Toba islands, and the offshore airport.
ise_islands = [
    (136.79,34.85,.038,.014,12), (136.86,34.52,.021,.032,0),
    (136.90,34.50,.027,.040,10), (136.91,34.44,.022,.030,-5),
    (136.87,34.37,.022,.018,0), (136.94,34.32,.018,.028,10),
]
for item in ise_islands:
    land |= ellipse(*item)

# A few conspicuous rocks/islets along the Pacific Kii coast.
land |= ellipse(135.75, 33.43, 0.018, 0.012, 0)
land |= ellipse(135.89, 33.54, 0.016, 0.012, 0)
land |= ellipse(136.08, 33.66, 0.017, 0.012, 0)

# Serialize exactly in the orientation specified by the task.
matrix = land.astype(np.uint8)
outdir = Path("outputs")
outdir.mkdir(parents=True, exist_ok=True)
csv_path = outdir / "candidate.csv"
with csv_path.open("w", newline="", encoding="ascii") as f:
    csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

raw = Image.fromarray(matrix * 255, mode="L")
raw.save(outdir / "candidate.png")
enlarged = raw.resize((W * 8, H * 8), resample=Image.Resampling.NEAREST)
enlarged.save(outdir / "candidate_8x.png")

# Internal checks deliberately fail loudly rather than emit partial products.
assert matrix.shape == (64, 253)
assert set(np.unique(matrix).tolist()) <= {0, 1}
assert raw.size == (253, 64)
assert enlarged.size == (2024, 512)
assert set(raw.getdata()) <= {0, 255}
assert np.array_equal(np.asarray(enlarged), np.repeat(np.repeat(
    np.asarray(raw), 8, axis=0), 8, axis=1))
print("wrote 64x253 CSV and 253x64 / 2024x512 binary PNGs")
