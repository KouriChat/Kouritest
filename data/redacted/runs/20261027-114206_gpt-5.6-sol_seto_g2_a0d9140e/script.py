from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------

ROWS = 64
COLS = 253
SCALE = 8

LONS = np.linspace(129.0, 137.0, COLS)
LATS = np.linspace(35.0, 33.0, ROWS)
X, Y = np.meshgrid(LONS, LATS)


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------

def polygon_mask(points):
    """Vectorized even-odd point-in-polygon test."""
    pts = np.asarray(points, dtype=float)
    inside = np.zeros(X.shape, dtype=bool)

    xj, yj = pts[-1]
    for xi, yi in pts:
        crosses = ((yi > Y) != (yj > Y))
        x_intersection = (
            (xj - xi) * (Y - yi) / ((yj - yi) + 1e-300) + xi
        )
        inside ^= crosses & (X < x_intersection)
        xj, yj = xi, yi

    return inside


def ellipse_mask(cx, cy, rx, ry, angle_deg=0.0):
    """Rotated ellipse in longitude/latitude coordinates."""
    a = math.radians(angle_deg)
    dx = X - cx
    dy = Y - cy
    u = dx * math.cos(a) + dy * math.sin(a)
    v = -dx * math.sin(a) + dy * math.cos(a)
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1.0


def add_polygon(mask, points):
    mask |= polygon_mask(points)


def remove_polygon(mask, points):
    mask &= ~polygon_mask(points)


def add_ellipse(mask, cx, cy, rx, ry, angle=0.0):
    mask |= ellipse_mask(cx, cy, rx, ry, angle)


def remove_ellipse(mask, cx, cy, rx, ry, angle=0.0):
    mask &= ~ellipse_mask(cx, cy, rx, ry, angle)


# ---------------------------------------------------------------------
# Hand-reconstructed major land masses
# ---------------------------------------------------------------------

land = np.zeros((ROWS, COLS), dtype=bool)

# Western and southern Honshu, including the Kii Peninsula and the
# western side of Ise Bay. The lower edge follows the remembered coast.
honshu = [
    (131.65, 35.18),
    (130.95, 34.74),
    (130.82, 34.55),
    (130.79, 34.40),
    (130.87, 34.27),
    (130.79, 34.10),
    (130.86, 33.96),  # Shimonoseki
    (131.02, 33.94),
    (131.18, 33.97),
    (131.35, 34.02),
    (131.48, 34.08),
    (131.67, 34.09),
    (131.83, 34.14),
    (132.02, 34.14),
    (132.16, 34.20),
    (132.23, 34.30),
    (132.30, 34.38),
    (132.40, 34.42),  # Hiroshima
    (132.49, 34.36),
    (132.44, 34.25),
    (132.48, 34.18),  # Kure peninsula
    (132.60, 34.19),
    (132.68, 34.25),
    (132.82, 34.28),
    (132.96, 34.34),
    (133.12, 34.39),
    (133.30, 34.42),
    (133.46, 34.43),
    (133.61, 34.46),
    (133.72, 34.52),
    (133.79, 34.47),
    (133.75, 34.35),
    (133.82, 34.29),  # Kojima Peninsula
    (133.92, 34.31),
    (133.94, 34.42),
    (134.04, 34.49),
    (134.19, 34.54),
    (134.37, 34.58),
    (134.55, 34.63),
    (134.72, 34.66),
    (134.88, 34.65),
    (135.04, 34.64),
    (135.18, 34.69),  # Kobe
    (135.34, 34.69),
    (135.46, 34.63),  # Osaka
    (135.48, 34.54),
    (135.45, 34.43),
    (135.39, 34.32),
    (135.31, 34.24),
    (135.20, 34.20),
    (135.13, 34.13),  # Wakayama
    (135.08, 33.99),
    (135.06, 33.84),
    (135.12, 33.70),
    (135.27, 33.57),
    (135.45, 33.47),
    (135.65, 33.43),
    (135.82, 33.47),
    (136.00, 33.58),
    (136.15, 33.68),
    (136.29, 33.73),
    (136.43, 33.82),
    (136.52, 33.95),
    (136.58, 34.08),
    (136.71, 34.18),
    (136.84, 34.27),
    (136.89, 34.36),  # Toba/Shima
    (136.83, 34.46),
    (136.75, 34.55),
    (136.69, 34.66),
    (136.68, 34.78),
    (136.72, 34.91),
    (136.82, 35.03),  # western Ise Bay head
    (136.90, 35.08),
    (136.96, 35.04),
    (136.94, 34.94),
    (136.89, 34.83),
    (136.85, 34.73),  # Chita west coast
    (136.88, 34.67),
    (136.94, 34.73),
    (136.98, 34.84),
    (136.97, 34.94),
    (137.07, 35.02),  # Mikawa Bay head, beyond east edge
    (137.25, 35.08),
    (137.25, 35.30),
    (131.65, 35.30),
]
add_polygon(land, honshu)

# Atsumi Peninsula, entering through the east edge.
atsumi = [
    (136.88, 34.63),
    (136.95, 34.58),
    (137.08, 34.57),
    (137.25, 34.55),
    (137.25, 34.83),
    (137.10, 34.78),
    (136.99, 34.72),
    (136.90, 34.68),
]
add_polygon(land, atsumi)

# Kyushu core.
kyushu = [
    (129.56, 32.78),
    (132.10, 32.78),
    (132.02, 33.02),
    (131.91, 33.14),
    (131.82, 33.25),
    (131.71, 33.36),
    (131.62, 33.48),
    (131.48, 33.56),
    (131.34, 33.62),
    (131.21, 33.70),
    (131.11, 33.79),
    (131.02, 33.89),
    (130.91, 33.96),
    (130.82, 33.96),
    (130.73, 33.91),
    (130.62, 33.86),
    (130.55, 33.76),
    (130.47, 33.69),
    (130.39, 33.68),
    (130.34, 33.62),  # Hakata Bay
    (130.25, 33.60),
    (130.17, 33.64),
    (130.08, 33.59),
    (130.03, 33.51),
    (129.94, 33.46),
    (129.88, 33.39),
    (129.89, 33.30),
    (129.81, 33.22),
    (129.77, 33.10),
    (129.68, 33.00),
]
add_polygon(land, kyushu)

# Nagasaki/Saga western peninsulas.
nagasaki_north = [
    (129.66, 32.92),
    (129.65, 33.06),
    (129.58, 33.15),
    (129.55, 33.27),
    (129.61, 33.38),
    (129.72, 33.45),
    (129.84, 33.43),
    (129.93, 33.35),
    (129.94, 33.23),
    (129.88, 33.13),
    (129.82, 33.02),
]
add_polygon(land, nagasaki_north)

nagasaki_peninsula = [
    (129.68, 32.86),
    (129.77, 32.91),
    (129.88, 32.99),
    (129.92, 33.10),
    (129.88, 33.20),
    (129.78, 33.18),
    (129.70, 33.09),
    (129.64, 32.97),
]
add_polygon(land, nagasaki_peninsula)

# Shimabara Peninsula.
shimabara = [
    (130.18, 32.96),
    (130.24, 33.04),
    (130.27, 33.15),
    (130.20, 33.25),
    (130.29, 33.33),
    (130.39, 33.30),
    (130.45, 33.19),
    (130.42, 33.06),
    (130.34, 32.96),
]
add_polygon(land, shimabara)

# Shikoku.
shikoku = [
    (132.00, 33.30),  # western end of Sadamisaki
    (132.16, 33.34),
    (132.33, 33.39),
    (132.46, 33.48),
    (132.53, 33.62),
    (132.61, 33.76),
    (132.66, 33.91),
    (132.73, 34.00),
    (132.88, 34.06),  # Matsuyama/Imabari side
    (132.99, 34.13),
    (133.12, 34.17),
    (133.25, 34.18),
    (133.37, 34.22),
    (133.50, 34.27),
    (133.63, 34.30),
    (133.78, 34.34),
    (133.94, 34.36),
    (134.08, 34.35),
    (134.22, 34.31),
    (134.37, 34.28),
    (134.51, 34.22),
    (134.61, 34.16),  # Naruto
    (134.62, 34.04),
    (134.66, 33.91),
    (134.65, 33.75),
    (134.57, 33.60),
    (134.45, 33.45),
    (134.32, 33.31),
    (134.19, 33.22),  # Muroto
    (134.05, 33.20),
    (133.89, 33.18),
    (133.74, 33.13),
    (133.59, 33.07),
    (133.43, 33.02),
    (133.27, 32.97),
    (133.10, 32.91),
    (132.92, 32.85),
    (132.73, 32.83),
    (132.57, 32.90),
    (132.47, 33.02),
    (132.42, 33.16),
    (132.31, 33.27),
    (132.17, 33.30),
]
add_polygon(land, shikoku)


# ---------------------------------------------------------------------
# Larger islands
# ---------------------------------------------------------------------

# Tsushima: two elongated masses separated by a narrow central channel.
add_ellipse(land, 129.36, 34.51, 0.105, 0.245, -18)
add_ellipse(land, 129.25, 34.23, 0.105, 0.190, -20)
add_ellipse(land, 129.31, 34.36, 0.075, 0.120, -18)
remove_ellipse(land, 129.30, 34.36, 0.030, 0.055, -18)

# Iki.
add_ellipse(land, 129.70, 33.79, 0.100, 0.125, 8)
add_ellipse(land, 129.77, 33.72, 0.045, 0.050, 5)

# Hirado and Ikitsuki.
add_ellipse(land, 129.49, 33.31, 0.080, 0.205, -10)
add_ellipse(land, 129.43, 33.41, 0.045, 0.110, -8)
add_ellipse(land, 129.42, 33.24, 0.055, 0.105, -5)
add_ellipse(land, 129.44, 33.37, 0.027, 0.085, -7)

# Goto/Uku/Ojika fragments visible near the western edge.
add_ellipse(land, 129.11, 33.27, 0.065, 0.075, 18)
add_ellipse(land, 129.06, 33.19, 0.055, 0.070, -10)
add_ellipse(land, 129.02, 33.10, 0.060, 0.085, 8)
add_ellipse(land, 129.18, 33.18, 0.030, 0.040, 0)

# Awaji.
awaji = [
    (134.66, 34.57),
    (134.76, 34.62),
    (134.87, 34.60),
    (134.96, 34.52),
    (135.00, 34.42),
    (134.96, 34.31),
    (134.88, 34.21),
    (134.79, 34.18),
    (134.72, 34.25),
    (134.70, 34.36),
    (134.65, 34.47),
]
add_polygon(land, awaji)

# Shodoshima.
shodoshima = [
    (134.10, 34.47),
    (134.16, 34.53),
    (134.25, 34.56),
    (134.36, 34.55),
    (134.45, 34.51),
    (134.43, 34.45),
    (134.34, 34.42),
    (134.23, 34.43),
    (134.15, 34.41),
    (134.09, 34.44),
]
add_polygon(land, shodoshima)

# Yashiro-oshima.
add_ellipse(land, 132.31, 33.91, 0.145, 0.095, -10)
add_ellipse(land, 132.43, 33.95, 0.085, 0.055, -8)

# Etajima, Nomi, Kurahashi and Miyajima.
add_ellipse(land, 132.42, 34.18, 0.090, 0.145, -12)
add_ellipse(land, 132.47, 34.09, 0.085, 0.100, -20)
add_ellipse(land, 132.54, 34.10, 0.105, 0.070, -16)
add_ellipse(land, 132.28, 34.27, 0.060, 0.125, -25)

# Geiyo Islands.
geiyo_ellipses = [
    (132.67, 34.17, 0.050, 0.065, -10),  # Kamagari vicinity
    (132.74, 34.20, 0.055, 0.055, 5),
    (132.82, 34.24, 0.075, 0.070, -10),  # Osakikamijima
    (132.82, 34.14, 0.075, 0.060, 5),
    (132.91, 34.20, 0.045, 0.060, -10),
    (132.96, 34.25, 0.085, 0.100, -10),  # Omishima
    (133.04, 34.22, 0.070, 0.095, 15),
    (133.10, 34.18, 0.050, 0.065, 5),    # Hakata
    (133.04, 34.09, 0.075, 0.080, -5),   # Oshima
    (133.15, 34.30, 0.065, 0.075, -8),   # Innoshima
    (133.08, 34.30, 0.060, 0.075, -20),  # Ikuchijima
    (133.19, 34.38, 0.050, 0.070, -15),  # Mukaishima
    (133.25, 34.34, 0.040, 0.045, 0),
]
for args in geiyo_ellipses:
    add_ellipse(land, *args)

# Bisan and Kagawa islands.
bisan_ellipses = [
    (133.52, 34.36, 0.035, 0.045, 0),
    (133.62, 34.38, 0.042, 0.050, 0),
    (133.79, 34.39, 0.045, 0.050, 0),
    (133.97, 34.46, 0.045, 0.055, -10),  # Naoshima
    (134.07, 34.48, 0.055, 0.060, 0),    # Teshima
    (134.01, 34.41, 0.032, 0.040, 0),
    (134.48, 34.41, 0.035, 0.045, 0),
    (134.55, 34.38, 0.030, 0.038, 0),
]
for args in bisan_ellipses:
    add_ellipse(land, *args)


# ---------------------------------------------------------------------
# Smaller remembered islands and artificial islands
# ---------------------------------------------------------------------

small_islands = [
    # Genkai Sea and northern Kyushu
    (129.93, 33.60, 0.030, 0.032, 0),
    (130.03, 33.68, 0.028, 0.032, 0),
    (130.19, 33.68, 0.026, 0.030, 0),
    (130.28, 33.72, 0.028, 0.030, 0),
    (130.10, 33.88, 0.035, 0.038, 0),

    # Kanmon and Suo Sea
    (130.89, 33.94, 0.020, 0.025, 0),
    (131.16, 33.85, 0.025, 0.025, 0),
    (131.39, 33.80, 0.030, 0.035, 0),
    (131.51, 33.76, 0.026, 0.030, 0),
    (131.67, 33.72, 0.055, 0.032, 5),    # Himeshima
    (131.84, 33.82, 0.025, 0.030, 0),
    (132.02, 33.86, 0.030, 0.032, 0),
    (132.13, 33.88, 0.028, 0.035, 0),

    # Western Inland Sea
    (132.18, 34.04, 0.030, 0.035, 0),
    (132.25, 34.02, 0.025, 0.030, 0),
    (132.34, 34.04, 0.025, 0.030, 0),
    (132.55, 33.98, 0.025, 0.030, 0),
    (132.61, 34.04, 0.032, 0.035, 0),
    (132.71, 34.08, 0.026, 0.030, 0),
    (132.89, 34.05, 0.025, 0.030, 0),
    (133.18, 34.12, 0.022, 0.028, 0),
    (133.31, 34.19, 0.025, 0.030, 0),

    # Osaka Bay and Kii Channel
    (134.81, 34.17, 0.045, 0.030, -10),  # Nushima
    (135.00, 34.27, 0.026, 0.042, -10),  # Tomogashima group
    (135.05, 34.28, 0.020, 0.030, 0),
    (135.24, 34.43, 0.043, 0.018, -15),  # Kansai Airport
    (135.22, 34.64, 0.025, 0.014, 0),    # Kobe Airport
    (135.44, 34.62, 0.018, 0.014, 0),

    # Ise Bay
    (136.82, 34.86, 0.030, 0.017, -8),   # Chubu Airport
    (136.89, 34.50, 0.025, 0.032, 0),
    (136.87, 34.42, 0.022, 0.032, 0),
]
for args in small_islands:
    add_ellipse(land, *args)


# ---------------------------------------------------------------------
# Selected bays/channels cut back out of the broad land polygons
# ---------------------------------------------------------------------

# Omura Bay, approximated as internal seawater west of central Kyushu.
remove_ellipse(land, 129.87, 33.02, 0.075, 0.115, -8)

# Imari Bay indentation.
remove_polygon(
    land,
    [
        (129.76, 33.44),
        (129.86, 33.44),
        (129.91, 33.36),
        (129.84, 33.30),
        (129.77, 33.34),
    ],
)

# Hakata Bay opening.
remove_polygon(
    land,
    [
        (130.24, 33.67),
        (130.42, 33.72),
        (130.42, 33.64),
        (130.35, 33.60),
        (130.26, 33.61),
    ],
)

# Hiroshima Bay water fingers.
remove_polygon(
    land,
    [
        (132.32, 34.36),
        (132.39, 34.39),
        (132.45, 34.35),
        (132.44, 34.27),
        (132.38, 34.24),
        (132.33, 34.29),
    ],
)

# Okayama Bay.
remove_polygon(
    land,
    [
        (133.80, 34.53),
        (133.94, 34.55),
        (134.00, 34.48),
        (133.95, 34.39),
        (133.87, 34.37),
        (133.81, 34.42),
    ],
)

# Re-add Kojima Peninsula after cutting Okayama Bay.
add_polygon(
    land,
    [
        (133.74, 34.34),
        (133.82, 34.30),
        (133.91, 34.32),
        (133.94, 34.39),
        (133.88, 34.43),
        (133.78, 34.42),
    ],
)

# Ensure the narrow Kanmon Strait remains water.
remove_ellipse(land, 130.89, 33.94, 0.030, 0.017, -20)

# Preserve major channels between the Kii Peninsula and Awaji.
remove_polygon(
    land,
    [
        (134.95, 34.32),
        (135.05, 34.33),
        (135.08, 34.27),
        (135.02, 34.23),
        (134.96, 34.25),
    ],
)

# Restore remembered Tomogashima islands after the channel cut.
add_ellipse(land, 135.00, 34.27, 0.026, 0.042, -10)
add_ellipse(land, 135.05, 34.28, 0.020, 0.030, 0)


# ---------------------------------------------------------------------
# Export and validation
# ---------------------------------------------------------------------

matrix = land.astype(np.uint8)

assert matrix.shape == (64, 253)
assert matrix.size == 16192
assert np.all((matrix == 0) | (matrix == 1))

output_dir = Path.cwd() / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

csv_path = output_dir / "candidate.csv"
png_path = output_dir / "candidate.png"
png_8x_path = output_dir / "candidate_8x.png"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(matrix.tolist())

pixels = matrix * np.uint8(255)
original = Image.fromarray(pixels, mode="L")
original.save(png_path)

try:
    nearest = Image.Resampling.NEAREST
except AttributeError:
    nearest = Image.NEAREST

enlarged = original.resize((COLS * SCALE, ROWS * SCALE), resample=nearest)
enlarged.save(png_8x_path)

# Read files back and validate their actual serialized contents.
with csv_path.open("r", newline="", encoding="utf-8") as f:
    rows = list(csv.reader(f))

assert len(rows) == ROWS
assert all(len(row) == COLS for row in rows)
assert all(value in {"0", "1"} for row in rows for value in row)

check_original = np.asarray(Image.open(png_path).convert("L"))
check_enlarged = np.asarray(Image.open(png_8x_path).convert("L"))

assert check_original.shape == (64, 253)
assert check_enlarged.shape == (512, 2024)
assert set(np.unique(check_original).tolist()) <= {0, 255}
assert set(np.unique(check_enlarged).tolist()) <= {0, 255}

expected_8x = np.repeat(
    np.repeat(check_original, SCALE, axis=0),
    SCALE,
    axis=1,
)
assert np.array_equal(check_enlarged, expected_8x)

print(f"Generated {matrix.shape[0]}x{matrix.shape[1]} matrix")
print(f"Valid values: {matrix.size}")
print(csv_path)
print(png_path)
print(png_8x_path)
