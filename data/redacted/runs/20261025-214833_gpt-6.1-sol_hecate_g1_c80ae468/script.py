"""Knowledge-only reconstruction; no map/data input or network access.

Coordinates below are hand-estimated WGS84 (longitude, latitude) vertices.
They are NOT surveyed coastlines. Every grid point is classified by polygon
membership, with a few hand-drawn marine inlets subtracted afterward. This
is rule-based sampling, not independent language-model reasoning per point.
Dependencies: Python 3, numpy, Pillow. Run from the desired working directory.
"""
from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image

N = 127
LON, LAT = np.meshgrid(-131.0 + np.arange(N) * (2.0 / 126),
                      55.0 - np.arange(N) * (2.0 / 126))
land = np.zeros((N, N), dtype=bool)

def polygon(vertices):
    inside = np.zeros((N, N), dtype=bool)
    ax, ay = vertices[-1]
    for bx, by in vertices:
        if ay != by:
            cross_x = ax + (LAT - ay) * (bx - ax) / (by - ay)
            inside ^= ((ay > LAT) != (by > LAT)) & (LON < cross_x)
        ax, ay = bx, by
    return inside

def add(vertices):
    global land
    land |= polygon(vertices)

def inlet(vertices, widths):
    """Subtract tapered marine channels; widths are full widths in km."""
    global land
    x, y = LON * 64.5, LAT * 111.2
    wet = np.zeros_like(land)
    for i in range(len(vertices) - 1):
        ax, ay = vertices[i][0] * 64.5, vertices[i][1] * 111.2
        bx, by = vertices[i+1][0] * 64.5, vertices[i+1][1] * 111.2
        dx, dy = bx - ax, by - ay
        t = np.clip(((x-ax)*dx + (y-ay)*dy)/(dx*dx+dy*dy), 0, 1)
        radius = (widths[i] + t*(widths[i+1]-widths[i])) / 2
        wet |= (x-ax-t*dx)**2 + (y-ay-t*dy)**2 <= radius**2
    land &= ~wet

# Mainland: Tsimpsean peninsula, Skeena coast and coast beside Grenville.
add([(-130.18,55.04),(-130.15,54.96),(-130.02,54.90),
     (-130.15,54.81),(-130.34,54.72),(-130.44,54.69),
     (-130.46,54.61),(-130.43,54.55),(-130.47,54.49),
     (-130.40,54.45),(-130.40,54.39),(-130.34,54.36),
     (-130.27,54.35),(-130.24,54.29),(-130.18,54.26),
     (-130.15,54.20),(-130.04,54.17),(-130.09,54.12),
     (-130.08,54.06),(-129.98,53.97),(-129.92,53.91),
     (-129.84,53.84),(-129.78,53.77),(-129.68,53.69),
     (-129.61,53.62),(-129.52,53.54),(-129.44,53.47),
     (-129.31,53.44),(-129.25,53.38),(-129.13,53.36),
     (-129.05,53.41),(-128.90,53.42),(-128.90,55.04)])

# Southeastern Alaska coast around Cape Fox / Nakat.
add([(-130.98,55.04),(-130.94,54.99),(-130.86,54.95),
     (-130.79,54.95),(-130.75,54.89),(-130.71,54.84),
     (-130.69,54.80),(-130.60,54.77),(-130.57,54.83),
     (-130.54,54.88),(-130.46,54.93),(-130.37,54.95),
     (-130.32,55.04)])
# Wales and Pearse islands, separating the northern marine passages.
add([(-130.62,54.81),(-130.58,54.86),(-130.53,54.90),
     (-130.44,54.88),(-130.43,54.83),(-130.47,54.78),
     (-130.52,54.73),(-130.58,54.75)])
add([(-130.40,54.85),(-130.36,54.94),(-130.29,55.01),
     (-130.20,55.04),(-130.19,54.99),(-130.19,54.95),
     (-130.24,54.91),(-130.29,54.87),(-130.34,54.83)])

# Dundas group, in Chatham Sound.
add([(-131.04,54.57),(-131.01,54.63),(-130.97,54.67),
     (-130.91,54.70),(-130.84,54.68),(-130.80,54.65),
     (-130.82,54.60),(-130.78,54.57),(-130.83,54.53),
     (-130.82,54.49),(-130.88,54.45),(-130.95,54.47),
     (-130.98,54.52),(-131.02,54.53)])
add([(-130.89,54.44),(-130.82,54.48),(-130.76,54.46),
     (-130.72,54.42),(-130.76,54.37),(-130.84,54.37),
     (-130.88,54.40)])
add([(-130.74,54.53),(-130.67,54.55),(-130.62,54.52),
     (-130.65,54.48),(-130.72,54.48)])
add([(-130.69,54.46),(-130.63,54.47),(-130.59,54.44),
     (-130.60,54.40),(-130.66,54.39),(-130.70,54.42)])
# Stephens Island and nearby islands.
add([(-130.91,54.33),(-130.86,54.36),(-130.80,54.33),
     (-130.76,54.29),(-130.75,54.24),(-130.79,54.21),
     (-130.83,54.24),(-130.87,54.28)])
add([(-130.79,54.19),(-130.74,54.20),(-130.71,54.17),
     (-130.74,54.15)])
# Prince Rupert: Digby, Kaien, Ridley and Smith islands.
add([(-130.49,54.34),(-130.43,54.37),(-130.38,54.34),
     (-130.38,54.29),(-130.42,54.25),(-130.48,54.26)])
add([(-130.34,54.34),(-130.28,54.35),(-130.24,54.32),
     (-130.25,54.27),(-130.30,54.25),(-130.34,54.28)])
add([(-130.35,54.25),(-130.31,54.25),(-130.29,54.21),
     (-130.33,54.20),(-130.36,54.22)])
add([(-130.28,54.21),(-130.23,54.24),(-130.16,54.21),
     (-130.13,54.17),(-130.18,54.13),(-130.24,54.14)])

# Porcher Island. Large features only; inlets are subtracted below.
add([(-130.76,54.09),(-130.72,54.15),(-130.66,54.18),
     (-130.58,54.20),(-130.52,54.17),(-130.48,54.16),
     (-130.42,54.12),(-130.39,54.07),(-130.37,54.02),
     (-130.40,53.98),(-130.40,53.94),(-130.37,53.89),
     (-130.42,53.83),(-130.49,53.80),(-130.54,53.84),
     (-130.60,53.86),(-130.64,53.91),(-130.70,53.94),
     (-130.69,53.99),(-130.74,54.02)])
# Gurd and Dolphin (Kitkatla) islands.
add([(-130.37,54.01),(-130.32,54.03),(-130.29,53.99),
     (-130.30,53.94),(-130.34,53.91),(-130.38,53.93)])
add([(-130.47,53.82),(-130.42,53.83),(-130.37,53.79),
     (-130.40,53.75),(-130.46,53.76)])

# McCauley and adjacent islands, between Ogden and Petrel channels.
add([(-130.33,54.07),(-130.25,54.08),(-130.18,54.04),
     (-130.12,54.02),(-130.10,53.96),(-130.13,53.89),
     (-130.13,53.84),(-130.18,53.78),(-130.23,53.73),
     (-130.29,53.75),(-130.32,53.80),(-130.30,53.86),
     (-130.34,53.90),(-130.33,53.96),(-130.36,54.01)])
add([(-130.38,53.79),(-130.33,53.80),(-130.30,53.77),
     (-130.31,53.72),(-130.36,53.71),(-130.39,53.74)])
# Kennedy Island at the Skeena estuary.
add([(-130.10,54.16),(-130.04,54.19),(-129.99,54.17),
     (-129.98,54.12),(-130.02,54.09),(-130.08,54.10)])

# Pitt Island: north end to the south end opposite Campania.
add([(-130.14,54.03),(-130.12,54.04),(-130.11,54.01),
     (-130.04,53.98),(-129.98,53.93),(-129.93,53.88),
     (-129.85,53.81),(-129.79,53.74),(-129.69,53.66),
     (-129.63,53.59),(-129.55,53.51),(-129.47,53.44),
     (-129.38,53.37),(-129.34,53.30),(-129.34,53.24),
     (-129.39,53.19),(-129.48,53.17),(-129.55,53.21),
     (-129.59,53.29),(-129.66,53.34),(-129.68,53.41),
     (-129.76,53.46),(-129.82,53.53),(-129.87,53.57),
     (-129.92,53.64),(-129.99,53.69),(-130.01,53.76),
     (-130.07,53.81),(-130.06,53.87),(-130.09,53.94)])

# Banks Island, west of Principe Channel.
add([(-130.55,53.64),(-130.50,53.70),(-130.42,53.74),
     (-130.36,53.71),(-130.31,53.65),(-130.27,53.60),
     (-130.20,53.55),(-130.15,53.48),(-130.09,53.43),
     (-130.05,53.36),(-129.98,53.30),(-129.96,53.23),
     (-129.89,53.16),(-129.88,53.08),(-129.93,53.00),
     (-129.98,52.96),(-130.06,53.02),(-130.11,53.10),
     (-130.19,53.17),(-130.23,53.24),(-130.31,53.31),
     (-130.35,53.39),(-130.43,53.46),(-130.45,53.52),
     (-130.52,53.57)])
add([(-130.66,53.51),(-130.62,53.53),(-130.59,53.50),
     (-130.61,53.47),(-130.65,53.47)])

# Campania, northern Estevan group and Gil Island.
add([(-129.62,53.23),(-129.57,53.25),(-129.53,53.21),
     (-129.51,53.14),(-129.47,53.07),(-129.43,53.00),
     (-129.40,52.95),(-129.46,52.94),(-129.52,53.00),
     (-129.56,53.08),(-129.60,53.15)])
add([(-129.79,53.10),(-129.73,53.15),(-129.66,53.13),
     (-129.62,53.09),(-129.65,53.04),(-129.71,53.00),
     (-129.77,53.02)])
add([(-129.79,53.01),(-129.74,53.04),(-129.69,53.01),
     (-129.64,52.97),(-129.75,52.93)])
add([(-129.38,53.29),(-129.32,53.35),(-129.25,53.33),
     (-129.20,53.29),(-129.17,53.22),(-129.19,53.15),
     (-129.24,53.10),(-129.29,53.12),(-129.33,53.18)])
# Western edge of Gribbell, and northern Princess Royal / Aristazabal.
add([(-129.14,53.46),(-129.10,53.53),(-129.03,53.55),
     (-128.94,53.51),(-128.94,53.31),(-129.03,53.31),
     (-129.09,53.36)])
add([(-129.13,53.11),(-129.07,53.15),(-128.94,53.14),
     (-128.94,52.96),(-129.08,52.96),(-129.10,53.03)])
add([(-129.27,53.00),(-129.21,53.04),(-129.16,53.01),
     (-129.14,52.94),(-129.25,52.94)])
# Small named islands, represented by explicitly estimated polygons.
add([(-130.63,54.31),(-130.60,54.32),(-130.58,54.30),
     (-130.61,54.28)])  # Lucy
add([(-130.37,54.13),(-130.34,54.14),(-130.32,54.12),
     (-130.35,54.10)])  # Lawyer group
add([(-130.76,54.20),(-130.72,54.21),(-130.70,54.19),
     (-130.73,54.17)])  # Tree Nob vicinity
add([(-130.78,54.77),(-130.75,54.79),(-130.72,54.77),
     (-130.74,54.75)])  # Tongass vicinity

# Marine inlets are water even when enclosed by the mainland outline.
inlet([(-130.73,54.82),(-130.69,54.91),(-130.60,55.02)],
      [2.5,1.5,0.7])
inlet([(-130.43,54.70),(-130.23,54.80),(-130.03,54.90),
       (-129.88,54.96),(-129.75,55.03)], [4.0,3.8,3.5,3.0,2.5])
inlet([(-130.03,54.90),(-130.17,54.97),(-130.16,55.04)],
      [1.8,1.2,1.1])
inlet([(-130.39,54.85),(-130.35,54.93),(-130.30,55.03)],
      [1.2,1.1,1.0])
# Khutzeymateen, Work Channel, and a northern branch of Portland Inlet.
inlet([(-130.23,54.75),(-130.09,54.73),(-129.93,54.77),
       (-129.78,54.79)], [2.4,1.8,1.1,0.3])
inlet([(-130.40,54.57),(-130.22,54.53),(-130.08,54.45),
       (-129.97,54.40)], [1.8,1.3,1.0,0.3])
inlet([(-129.86,54.97),(-129.72,54.93),(-129.61,54.98),
       (-129.55,55.03)], [2.5,2.0,1.4,1.0])
# Prince Rupert harbour and channels around Kaien.
inlet([(-130.36,54.37),(-130.34,54.33),(-130.36,54.27),
       (-130.39,54.23)], [1.4,1.4,1.8,2.2])
inlet([(-130.27,54.36),(-130.245,54.32),(-130.245,54.27)],
      [0.7,0.5,0.6])
# Skeena estuary and its two approximate branches.
inlet([(-130.17,54.13),(-130.06,54.20),(-129.89,54.23),
       (-129.76,54.31),(-129.68,54.40)], [3.0,2.8,2.0,1.6,1.0])
inlet([(-130.01,54.08),(-129.91,54.13),(-129.87,54.22)],
      [2.0,1.5,1.3])
# Narrow Grenville Channel, using the same hand-estimated shore framework.
inlet([(-130.11,54.06),(-129.99,53.95),(-129.85,53.82),
       (-129.66,53.64),(-129.50,53.49),(-129.36,53.40)],
      [1.4,1.0,0.9,1.0,1.2,2.0])
inlet([(-129.64,53.63),(-129.56,53.66),(-129.51,53.70)],
      [1.0,0.8,0.2])
inlet([(-129.57,53.55),(-129.51,53.55),(-129.48,53.59)],
      [1.0,0.8,0.2])
# Porcher's two main indentations and a Banks west-coast indentation.
inlet([(-130.56,54.19),(-130.57,54.12),(-130.60,54.08)],
      [1.3,1.1,0.3])
inlet([(-130.37,53.94),(-130.48,53.96),(-130.55,54.01)],
      [2.1,1.6,0.3])
inlet([(-130.43,53.46),(-130.34,53.47),(-130.29,53.51)],
      [1.6,1.0,0.3])
# Douglas/Verney passages at the eastern edge.
inlet([(-129.13,53.37),(-129.19,53.49),(-129.18,53.62),
       (-129.10,53.73),(-128.98,53.79)], [3.4,2.5,2.0,1.8,2.0])
inlet([(-129.08,53.40),(-129.02,53.51),(-128.96,53.65)],
      [2.5,2.5,2.0])

out = Path.cwd() / "outputs"
out.mkdir(parents=True, exist_ok=True)
matrix = land.astype(np.uint8)
assert matrix.shape == (127,127) and matrix.size == 16129
assert np.isin(matrix, [0,1]).all()
np.savetxt(out / "candidate.csv", matrix, fmt="%d", delimiter=",")
pixels = matrix * np.uint8(255)
image = Image.fromarray(pixels)
image.save(out / "candidate.png")
nearest = getattr(Image, "Resampling", Image).NEAREST
image.resize((1016,1016), resample=nearest).save(out / "candidate_8x.png")

# Independently reread the exported products, including every enlarged pixel.
with (out / "candidate.csv").open(newline="", encoding="utf-8") as f:
    rows = list(csv.reader(f))
assert len(rows) == 127
assert all(len(row) == 127 for row in rows)
assert all(value in ("0","1") for row in rows for value in row)
assert np.array_equal(np.asarray(rows, dtype=np.uint8), matrix)
with Image.open(out / "candidate.png") as im:
    assert im.size == (127,127)
    raw = np.array(im)
with Image.open(out / "candidate_8x.png") as im:
    assert im.size == (1016,1016)
    enlarged = np.array(im)
assert np.isin(raw, [0,255]).all()
assert np.isin(enlarged, [0,255]).all()
assert np.array_equal(raw, pixels)
assert np.array_equal(enlarged, np.repeat(np.repeat(raw,8,axis=0),8,axis=1))
print(json.dumps({"shape":list(matrix.shape), "valid_values":int(matrix.size),
                  "land_points":int(matrix.sum()), "water_points":int((1-matrix).sum()),
                  "original_png":[127,127], "enlarged_png":[1016,1016],
                  "binary_only":True, "nearest_neighbor_verified":True}))
