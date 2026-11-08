"""Knowledge-only, hand-drawn land mask for 129--137 E, 33--35 N.

No map, web service, geodata file, or per-cell lookup is used.  The mask is
the union of hand-written polygons and rotated ellipses.  A few bay polygons
are then removed.  Sampling is at the requested WGS84 point coordinates.
"""
from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image


W, H = 253, 64
LONS = np.linspace(129.0, 137.0, W)
LATS = np.linspace(35.0, 33.0, H)
X, Y = np.meshgrid(LONS, LATS)


def in_polygon(vertices):
    """Vectorized even/odd point-in-polygon test (boundary choice arbitrary)."""
    inside = np.zeros((H, W), dtype=bool)
    xj, yj = vertices[-1]
    for xi, yi in vertices:
        crossing = ((yi > Y) != (yj > Y))
        x_cross = (xj - xi) * (Y - yi) / (yj - yi + 1e-300) + xi
        inside ^= crossing & (X < x_cross)
        xj, yj = xi, yi
    return inside


def ellipse(cx, cy, rx, ry, angle=0.0):
    a = math.radians(angle)
    dx, dy = X - cx, Y - cy
    u = dx * math.cos(a) + dy * math.sin(a)
    v = -dx * math.sin(a) + dy * math.cos(a)
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1.0


land = np.zeros((H, W), dtype=bool)


def add_poly(points):
    global land
    land |= in_polygon(points)


def add_ellipse(*args):
    global land
    land |= ellipse(*args)


def cut_poly(points):
    global land
    land &= ~in_polygon(points)


def cut_ellipse(*args):
    global land
    land &= ~ellipse(*args)


# Honshu: the southern shoreline is deliberately much more detailed than the
# artificial out-of-frame closure.  The final vertices trace the Japan-Sea
# shore back to Shimonoseki.
add_poly([
    (130.83, 33.92), (130.91, 33.93), (130.97, 33.98),
    (131.08, 33.97), (131.18, 33.94), (131.31, 33.98),
    (131.43, 34.02), (131.57, 34.03), (131.67, 34.00),
    (131.78, 34.03), (131.88, 34.00), (132.00, 34.04),
    (132.10, 34.08), (132.20, 34.10),
    # Hiroshima Bay
    (132.25, 34.19), (132.23, 34.27), (132.31, 34.34),
    (132.42, 34.38), (132.49, 34.35), (132.50, 34.28),
    (132.54, 34.23), (132.58, 34.18), (132.66, 34.20),
    (132.73, 34.25), (132.82, 34.29), (132.91, 34.35),
    # Bingo and Okayama
    (133.02, 34.38), (133.10, 34.37), (133.18, 34.40),
    (133.27, 34.43), (133.39, 34.46), (133.49, 34.48),
    (133.60, 34.50), (133.70, 34.52), (133.77, 34.58),
    (133.88, 34.62), (133.98, 34.66), (134.08, 34.62),
    (134.08, 34.53), (134.14, 34.48), (134.26, 34.46),
    (134.36, 34.53), (134.45, 34.58), (134.54, 34.65),
    (134.64, 34.71), (134.74, 34.74), (134.85, 34.70),
    # Akashi, Kobe, and the eastern side of Osaka Bay
    (134.96, 34.65), (135.08, 34.65), (135.19, 34.68),
    (135.31, 34.68), (135.42, 34.70), (135.49, 34.66),
    (135.47, 34.58), (135.43, 34.49), (135.37, 34.39),
    (135.29, 34.32), (135.20, 34.25), (135.13, 34.22),
    # Kii Peninsula west, south, and east coasts
    (135.10, 34.13), (135.16, 34.03), (135.17, 33.93),
    (135.24, 33.83), (135.34, 33.72), (135.43, 33.61),
    (135.55, 33.51), (135.67, 33.44), (135.76, 33.43),
    (135.85, 33.47), (135.94, 33.55), (136.03, 33.65),
    (136.10, 33.76), (136.17, 33.88), (136.21, 34.00),
    (136.29, 34.12), (136.39, 34.21), (136.50, 34.29),
    (136.62, 34.35), (136.73, 34.40), (136.82, 34.45),
    # Shima Peninsula and western shore of Ise Bay
    (136.91, 34.48), (136.88, 34.55), (136.83, 34.61),
    (136.82, 34.70), (136.77, 34.78), (136.75, 34.88),
    (136.76, 35.04), (136.72, 35.25), (131.95, 35.25),
    # Japan-Sea shoreline, westward
    (132.05, 34.98), (131.98, 34.89), (131.87, 34.78),
    (131.74, 34.69), (131.62, 34.61), (131.49, 34.55),
    (131.37, 34.49), (131.25, 34.46), (131.14, 34.39),
    (131.04, 34.32), (130.98, 34.23), (130.90, 34.14),
    (130.87, 34.04),
])

# Chita Peninsula / mainland east of Ise Bay.  It joins the rest of Honshu
# north of the sampled box.  Atsumi reaches the extreme eastern edge.
add_poly([
    (136.89, 35.20), (137.25, 35.20), (137.25, 34.62),
    (137.09, 34.64), (137.00, 34.67), (136.94, 34.64),
    (136.89, 34.68), (136.91, 34.77), (136.89, 34.88),
    (136.91, 34.98),
])
add_poly([(136.96, 34.54), (137.25, 34.59), (137.25, 34.68),
          (137.06, 34.65), (136.99, 34.61)])

# Kyushu: northwest shore, Hakata/Kitakyushu, Kunisaki and Oita.
add_poly([
    (129.36, 32.85), (132.12, 32.85), (132.03, 33.05),
    (131.94, 33.15), (131.85, 33.23), (131.73, 33.25),
    (131.63, 33.31), (131.49, 33.34), (131.53, 33.42),
    (131.66, 33.48), (131.76, 33.57), (131.76, 33.64),
    (131.67, 33.69), (131.56, 33.68), (131.47, 33.63),
    (131.36, 33.66), (131.27, 33.72), (131.15, 33.75),
    (131.05, 33.82), (130.98, 33.90), (130.94, 33.97),
    (130.85, 33.96), (130.75, 33.91), (130.68, 33.84),
    (130.59, 33.76), (130.49, 33.71), (130.43, 33.67),
    # Hakata Bay and Itoshima
    (130.37, 33.60), (130.29, 33.59), (130.23, 33.63),
    (130.16, 33.59), (130.12, 33.53), (130.03, 33.49),
    (129.96, 33.47), (129.90, 33.38), (129.84, 33.34),
    (129.80, 33.42), (129.72, 33.43), (129.66, 33.37),
    (129.66, 33.28), (129.59, 33.19), (129.54, 33.08),
    (129.44, 33.00),
])

# Omura Bay's northern end and the deep cuts of Imari/Sasebo bays.
cut_ellipse(129.86, 32.99, 0.135, 0.125, -8)
cut_poly([(129.84, 33.43), (129.88, 33.39), (129.94, 33.31),
          (129.98, 33.28), (129.96, 33.41), (129.91, 33.47)])

# Shikoku main body.
add_poly([
    (132.61, 32.88), (132.68, 33.04), (132.61, 33.14),
    (132.50, 33.23), (132.44, 33.34), (132.45, 33.44),
    (132.52, 33.54), (132.61, 33.65), (132.67, 33.76),
    (132.73, 33.85), (132.72, 33.94), (132.69, 34.02),
    (132.77, 34.12), (132.88, 34.18), (132.97, 34.20),
    (133.03, 34.15), (133.05, 34.09), (133.14, 34.03),
    (133.25, 33.98), (133.38, 33.97), (133.51, 33.99),
    (133.61, 34.05), (133.67, 34.14), (133.73, 34.23),
    (133.84, 34.27), (133.94, 34.32), (134.04, 34.35),
    (134.14, 34.33), (134.24, 34.38), (134.34, 34.35),
    (134.43, 34.29), (134.53, 34.25), (134.63, 34.25),
    (134.70, 34.20), (134.72, 34.13), (134.67, 34.05),
    (134.59, 33.94), (134.54, 33.83), (134.49, 33.70),
    (134.42, 33.55), (134.34, 33.40), (134.25, 33.28),
    (134.17, 33.24), (134.06, 33.29), (133.95, 33.36),
    (133.83, 33.43), (133.73, 33.48), (133.63, 33.52),
    # Tosa Bay
    (133.54, 33.56), (133.45, 33.54), (133.36, 33.48),
    (133.29, 33.38), (133.23, 33.25), (133.16, 33.12),
    (133.10, 32.96),
])

# The very narrow Sada Peninsula.
add_poly([(132.50, 33.52), (132.39, 33.48), (132.28, 33.43),
          (132.17, 33.39), (132.04, 33.36), (131.98, 33.32),
          (132.08, 33.31), (132.22, 33.34), (132.35, 33.38),
          (132.48, 33.41), (132.57, 33.45)])

# Tsushima, Iki, Hirado and the northern Goto islands.
add_poly([(129.18, 34.70), (129.26, 34.72), (129.34, 34.64),
          (129.39, 34.53), (129.42, 34.43), (129.38, 34.34),
          (129.31, 34.27), (129.25, 34.30), (129.22, 34.40),
          (129.17, 34.52)])
add_poly([(129.27, 34.33), (129.35, 34.34), (129.43, 34.27),
          (129.47, 34.18), (129.43, 34.09), (129.35, 34.07),
          (129.29, 34.13), (129.25, 34.23)])
add_poly([(129.54, 33.87), (129.66, 33.90), (129.77, 33.85),
          (129.82, 33.77), (129.74, 33.70), (129.61, 33.69),
          (129.53, 33.76)])
add_poly([(129.43, 33.43), (129.50, 33.46), (129.57, 33.40),
          (129.61, 33.31), (129.59, 33.21), (129.54, 33.12),
          (129.48, 33.13), (129.45, 33.24)])
add_ellipse(129.40, 33.47, .025, .070, -6)   # Ikitsuki
add_ellipse(129.12, 33.27, .055, .060, 12)   # Uku
add_ellipse(129.06, 33.19, .055, .075, -8)   # Ojika group
add_ellipse(129.12, 33.07, .070, .105, -20)  # top of Nakadori

# Awaji-shima.
add_poly([(134.82, 34.62), (134.92, 34.60), (135.00, 34.52),
          (135.03, 34.42), (134.99, 34.32), (134.91, 34.21),
          (134.84, 34.17), (134.76, 34.23), (134.72, 34.34),
          (134.73, 34.45)])

# Western and central Seto Inland Sea islands.
islands = [
    # cx, cy, rx, ry, angle
    (131.68, 33.72, .060, .018, 0),   # Himeshima
    (131.85, 33.95, .065, .025, -18), # Kasado
    (132.06, 33.78, .125, .025, -3),  # Heigun
    (132.31, 33.91, .245, .070, -7),  # Suo-Oshima
    (132.14, 34.18, .030, .022, 0),
    (132.30, 34.27, .140, .043, -58), # Miyajima
    (132.44, 34.19, .073, .115, -16), # Etajima
    (132.51, 34.11, .070, .095, -24), # Nomi/Kurahashi
    (132.63, 33.98, .040, .050, 12),
    (132.61, 34.00, .110, .065, -8),  # Nakajima group
    (132.69, 33.90, .080, .045, 15),  # Gogoshima
    (132.71, 34.17, .075, .040, -12), # Kamagari
    (132.82, 34.13, .105, .045, -18),
    (132.89, 34.25, .105, .065, -2),  # Osakikamijima
    (132.97, 34.25, .095, .085, 10),  # Omishima
    (133.05, 34.13, .075, .075, 10),  # Oshima
    (133.09, 34.20, .055, .080, -8),  # Hakatajima
    (133.09, 34.31, .100, .050, 0),   # Ikuchijima
    (133.18, 34.30, .058, .075, 8),   # Innoshima
    (133.20, 34.38, .045, .055, -5),  # Mukaishima
    (133.23, 34.24, .050, .035, 0),
    (133.54, 34.18, .035, .025, 0),
    (133.61, 34.27, .045, .025, 0),
    (133.67, 34.35, .045, .025, -8),
    (133.76, 34.37, .065, .040, 2),   # Honjima
    (133.85, 34.34, .035, .022, 0),
    (133.92, 34.41, .030, .040, 0),
    (133.99, 34.46, .040, .052, -8),  # Naoshima
    (134.06, 34.49, .070, .050, -8),  # Teshima
    (134.05, 34.40, .025, .060, 5),   # Megijima
    (134.12, 34.35, .020, .040, 0),
    (134.25, 34.49, .180, .085, 3),   # Shodoshima
    (134.53, 34.67, .095, .035, 3),   # Ieshima group
]
for args in islands:
    add_ellipse(*args)

# Small but resolvable islands/reclaimed islands in major bays and straits.
for args in [
    (130.30, 33.68, .040, .055, 0),   # Nokonoshima
    (130.36, 33.77, .035, .025, 0),
    (130.49, 33.90, .045, .025, 0),
    (130.92, 33.94, .028, .018, 0),
    (134.59, 34.23, .032, .020, 0),   # Naruto islands
    (134.68, 34.30, .025, .030, 0),
    (134.99, 34.28, .035, .020, 0),   # Tomogashima
    (135.23, 34.43, .060, .016, -12), # Kansai airport
    (135.22, 34.63, .030, .014, 0),   # Kobe airport
    (135.27, 34.68, .035, .018, 0),   # Rokko island
    (135.19, 34.67, .027, .020, 0),   # Port island
    (136.81, 34.86, .045, .014, -8),  # Centrair
    (136.88, 34.52, .045, .025, 0),   # Toba islands
    (136.96, 34.55, .035, .020, 0),
]:
    add_ellipse(*args)

# Export and self-check.
matrix = land.astype(np.uint8)
assert matrix.shape == (64, 253)
assert set(np.unique(matrix)).issubset({0, 1})

out = Path("outputs")
out.mkdir(parents=True, exist_ok=True)
with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
    csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

raw = Image.fromarray(matrix * 255, mode="L")
assert raw.size == (253, 64) and set(raw.getdata()).issubset({0, 255})
raw.save(out / "candidate.png")
large = raw.resize((2024, 512), resample=Image.Resampling.NEAREST)
assert large.size == (2024, 512)
assert np.array_equal(np.asarray(large), np.repeat(np.repeat(matrix * 255, 8, 0), 8, 1))
large.save(out / "candidate_8x.png")

with (out / "candidate.csv").open("r", encoding="ascii") as f:
    check = list(csv.reader(f))
assert len(check) == 64 and all(len(r) == 253 for r in check)
assert all(v in ("0", "1") for r in check for v in r)

print("OK: 64x253 = 16192 points; binary PNGs 253x64 and 2024x512")
