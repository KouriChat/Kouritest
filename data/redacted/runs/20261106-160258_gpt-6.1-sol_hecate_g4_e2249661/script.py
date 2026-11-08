"""Knowledge-only sketch; no maps, databases, downloads, or geographic libraries.
Coordinates below are hand-estimated, not individually inferred sample labels.
Run: python outputs/generate_map.py   Dependencies: numpy, Pillow.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

N = 127
lon, lat = np.meshgrid(np.linspace(-131, -129, N),
                       np.linspace(55, 53, N))
land = np.zeros((N, N), dtype=bool)

def polygon(points):
    inside = np.zeros((N, N), dtype=bool)
    x0, y0 = points[-1]
    for x1, y1 in points:
        if y1 != y0:
            cross = ((y0 > lat) != (y1 > lat))
            edge = (x1-x0)*(lat-y0)/(y1-y0)+x0
            inside ^= cross & (lon < edge)
        x0, y0 = x1, y1
    return inside

def add(points):
    global land
    land |= polygon(points)

def cut(points):
    global land
    land &= ~polygon(points)

def channel(points, widths):
    """Widths are approximate full water widths in latitude degrees."""
    global land
    for i, (a, b) in enumerate(zip(points[:-1], points[1:])):
        ax, ay = a[0]*0.585, a[1]
        bx, by = b[0]*0.585, b[1]
        dx, dy = bx-ax, by-ay
        t = np.clip(((lon*0.585-ax)*dx+(lat-ay)*dy)/(dx*dx+dy*dy), 0, 1)
        radius = (widths[i]*(1-t)+widths[i+1]*t)/2
        land &= ((lon*0.585-ax-t*dx)**2+(lat-ay-t*dy)**2 > radius**2)

# Mainland, including the Alaska coast at the northwest corner.
add([(-131.17,55.20),(-129.0,55.20),(-128.7,53.60),
     (-129.02,53.60),(-129.16,53.53),(-129.29,53.47),
     (-129.39,53.42),(-129.51,53.35),(-129.66,53.52),
     (-129.73,53.62),(-129.81,53.73),(-129.91,53.86),
     (-130.02,53.99),(-130.12,54.12),(-130.15,54.19),
     (-130.16,54.20),(-130.23,54.27),(-130.30,54.34),
     (-130.34,54.41),(-130.40,54.47),(-130.43,54.57),
     (-130.46,54.63),(-130.57,54.70),(-130.72,54.71),
     (-130.83,54.76),(-130.96,54.77),(-131.00,54.84),
     (-130.98,54.91),(-131.04,54.97)])
# Northern fjords and passages; these carve water, not inland lakes.
channel([(-130.69,54.66),(-130.43,54.73),(-130.19,54.84),
         (-129.99,54.90),(-129.76,54.98)], [.09,.075,.068,.055,.025])
channel([(-130.40,54.75),(-130.29,54.89),(-130.22,55.02),
         (-130.12,55.18)], [.035,.027,.022,.017])
channel([(-130.63,54.71),(-130.60,54.83),(-130.48,54.94),
         (-130.25,55.00)], [.025,.018,.018,.022])
channel([(-130.16,54.86),(-130.10,54.98),(-130.04,55.12)],
        [.023,.018,.018])
channel([(-129.99,54.90),(-129.92,55.00),(-129.91,55.13)],
        [.035,.023,.016])
channel([(-131.10,55.01),(-130.91,55.035),(-130.72,55.00),
         (-130.57,55.10)], [.025,.023,.018,.015])
channel([(-130.43,54.65),(-130.26,54.57),(-130.14,54.44),
         (-130.04,54.32)], [.027,.018,.014,.009])
channel([(-130.32,54.34),(-130.29,54.26),(-130.22,54.22)],
        [.014,.018,.025])
channel([(-130.19,54.13),(-130.02,54.19),(-129.84,54.23),
         (-129.67,54.30)], [.07,.043,.028,.018])
channel([(-129.61,53.57),(-129.49,53.57),(-129.40,53.63)],
        [.017,.015,.008])
channel([(-129.99,54.00),(-129.83,54.02),(-129.73,54.09)],
        [.016,.013,.008])

# Dundas, Zayas, Baron, and the southern Dundas-group islands.
add([(-130.99,54.64),(-130.87,54.65),(-130.78,54.61),
     (-130.75,54.56),(-130.77,54.50),(-130.85,54.47),
     (-130.91,54.49),(-130.95,54.54),(-131.00,54.58)])
add([(-131.00,54.73),(-130.92,54.72),(-130.89,54.68),
     (-130.94,54.65),(-131.00,54.67)])
add([(-130.75,54.48),(-130.66,54.48),(-130.62,54.43),
     (-130.66,54.38),(-130.73,54.38),(-130.78,54.42)])
add([(-130.88,54.43),(-130.80,54.43),(-130.75,54.38),
     (-130.79,54.34),(-130.86,54.36)])
add([(-130.78,54.35),(-130.70,54.36),(-130.65,54.32),
     (-130.69,54.29),(-130.75,54.30)])
# Digby Island, Kaien Island, and islands off the Skeena mouth.
add([(-130.49,54.36),(-130.41,54.37),(-130.36,54.32),
     (-130.38,54.27),(-130.46,54.23),(-130.51,54.27)])
add([(-130.35,54.33),(-130.30,54.34),(-130.26,54.30),
     (-130.24,54.24),(-130.28,54.21),(-130.34,54.25)])
add([(-130.25,54.18),(-130.18,54.20),(-130.12,54.15),
     (-130.15,54.10),(-130.23,54.10)])
add([(-130.15,54.08),(-130.08,54.13),(-130.02,54.12),
     (-129.99,54.05),(-130.05,54.00),(-130.11,54.02)])
# Stephens Island and Porcher Island.
add([(-130.85,54.21),(-130.76,54.23),(-130.68,54.18),
     (-130.64,54.10),(-130.68,54.06),(-130.76,54.11),
     (-130.82,54.15)])
add([(-130.66,54.08),(-130.54,54.11),(-130.45,54.08),
     (-130.37,54.08),(-130.30,54.03),(-130.31,53.96),
     (-130.25,53.91),(-130.30,53.85),(-130.30,53.77),
     (-130.39,53.75),(-130.48,53.81),(-130.54,53.87),
     (-130.63,53.91),(-130.67,53.98)])
channel([(-130.46,54.10),(-130.47,54.01),(-130.40,53.96)],
        [.025,.015,.009])
channel([(-130.32,53.81),(-130.40,53.86),(-130.42,53.93)],
        [.02,.014,.009])
# McCauley, Dolphin and smaller islands north of Banks.
add([(-130.23,53.99),(-130.15,53.99),(-130.10,53.92),
     (-130.11,53.82),(-130.18,53.73),(-130.24,53.74),
     (-130.27,53.84),(-130.24,53.91)])
add([(-130.39,53.83),(-130.32,53.85),(-130.28,53.80),
     (-130.32,53.72),(-130.39,53.75)])
add([(-130.48,53.72),(-130.41,53.73),(-130.36,53.67),
     (-130.40,53.63),(-130.46,53.66)])
# Banks Island.
add([(-130.62,53.73),(-130.51,53.72),(-130.39,53.65),
     (-130.30,53.56),(-130.21,53.45),(-130.12,53.33),
     (-130.03,53.21),(-129.94,53.07),(-129.92,52.96),
     (-130.03,53.00),(-130.14,53.10),(-130.25,53.23),
     (-130.37,53.35),(-130.47,53.49),(-130.57,53.58),
     (-130.62,53.66)])
# Pitt Island, west bank of Grenville Channel.
add([(-130.12,54.02),(-130.05,54.00),(-129.98,53.92),
     (-129.89,53.81),(-129.80,53.69),(-129.71,53.56),
     (-129.62,53.43),(-129.53,53.31),(-129.44,53.18),
     (-129.44,53.10),(-129.54,53.14),(-129.64,53.24),
     (-129.72,53.36),(-129.85,53.47),(-129.94,53.59),
     (-130.04,53.69),(-130.08,53.80),(-130.10,53.91)])
channel([(-129.91,53.55),(-129.81,53.58),(-129.80,53.64)],
        [.018,.012,.007])
# Gil, Gribbell, and the northern end of Princess Royal Island.
add([(-129.25,53.40),(-129.13,53.34),(-129.07,53.24),
     (-129.09,53.13),(-129.19,53.02),(-129.31,52.97),
     (-129.39,53.05),(-129.42,53.17),(-129.35,53.30)])
add([(-129.07,53.58),(-128.91,53.56),(-128.85,53.42),
     (-128.97,53.34),(-129.09,53.39),(-129.13,53.48)])
add([(-128.94,53.25),(-128.80,53.23),(-128.70,52.80),
     (-129.36,52.80),(-129.26,53.03),(-129.14,53.09),
     (-129.06,53.15),(-129.01,53.22)])
# Douglas Channel and its branches, plus narrow Grenville waterway.
channel([(-129.29,53.41),(-129.22,53.53),(-129.17,53.66),
         (-129.03,53.80),(-128.85,53.94)], [.045,.035,.035,.03,.03])
channel([(-129.16,53.44),(-129.02,53.51),(-128.83,53.54)],
        [.027,.025,.025])
channel([(-130.15,54.10),(-130.045,53.99),(-129.91,53.83),
         (-129.76,53.63),(-129.60,53.41),(-129.47,53.25)],
        [.014,.012,.010,.010,.013,.022])
# A few hand-estimated minor islands; unknown tiny reefs are not invented.
add([(-130.64,54.26),(-130.59,54.28),(-130.55,54.24),
     (-130.59,54.21),(-130.64,54.23)])
add([(-130.54,54.19),(-130.49,54.20),(-130.46,54.17),
     (-130.50,54.14)])
add([(-130.13,53.64),(-130.08,53.64),(-130.04,53.59),
     (-130.08,53.56),(-130.13,53.59)])

out = Path.cwd() / "outputs"
out.mkdir(parents=True, exist_ok=True)
matrix = land.astype(np.uint8)
assert matrix.shape == (127, 127) and matrix.size == 16129
assert set(np.unique(matrix)).issubset({0, 1})
with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
    csv.writer(f, lineterminator="\n").writerows(matrix.tolist())
pixels = matrix * np.uint8(255)
image = Image.fromarray(pixels)
image.save(out / "candidate.png")
nearest = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
enlarged = image.resize((1016, 1016), resample=nearest)
enlarged.save(out / "candidate_8x.png")

# Validate actual exported files, not just the arrays used to create them.
with (out / "candidate.csv").open(newline="", encoding="ascii") as f:
    rows = list(csv.reader(f))
assert len(rows) == 127
assert all(len(row) == 127 for row in rows)
assert all(v in ("0", "1") for row in rows for v in row)
assert np.array_equal(np.array(rows, dtype=np.uint8), matrix)
with Image.open(out / "candidate.png") as f:
    assert f.size == (127, 127)
    original = np.array(f)
with Image.open(out / "candidate_8x.png") as f:
    assert f.size == (1016, 1016)
    expanded = np.array(f)
assert set(np.unique(original)) == {0, 255}
assert set(np.unique(expanded)) == {0, 255}
assert np.array_equal(original, pixels)
assert np.array_equal(expanded, np.repeat(np.repeat(original, 8, axis=0), 8, axis=1))
print("Validated: 127x127 = 16,129 binary points; PNG 127x127 and nearest 1016x1016.")
