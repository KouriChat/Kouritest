"""Knowledge-only approximation: no map, network, GIS, or reference inputs.

Run from the current directory with Python 3, numpy and Pillow installed.
Coordinates below are hand-written estimates from general geographic memory.
Classification is by polygon containment and narrow-inlet rules, not by
independent language-model reasoning at each sample. Inland lakes stay land.
"""
from pathlib import Path
import csv
import json
import math
import numpy as np
from PIL import Image

N = 127
LONGITUDES = -131.0 + np.arange(N, dtype=float) * (2.0 / 126.0)
LATITUDES = 55.0 - np.arange(N, dtype=float) * (2.0 / 126.0)
X, Y = np.meshgrid(LONGITUDES, LATITUDES)
land = np.zeros((N, N), dtype=bool)


def polygon(vertices):
    """Even-odd point-in-polygon test at the actual grid points."""
    inside = np.zeros(X.shape, dtype=bool)
    x0, y0 = vertices[-1]
    for x1, y1 in vertices:
        if y0 != y1:
            crossing = (y0 > Y) != (y1 > Y)
            edge_x = x0 + (Y - y0) * (x1 - x0) / (y1 - y0)
            inside ^= crossing & (X < edge_x)
        x0, y0 = x1, y1
    return inside


def add(vertices):
    land[:] |= polygon(vertices)


def remove(vertices):
    land[:] &= ~polygon(vertices)


def inlet(points, radii_km):
    """Water along a hand-estimated centerline, with interpolated half-width."""
    sx, sy = 111.32 * math.cos(math.radians(54)), 111.32
    px, py = X * sx, Y * sy
    for a, b, r0, r1 in zip(points, points[1:],
                             radii_km, radii_km[1:]):
        ax, ay, bx, by = a[0]*sx, a[1]*sy, b[0]*sx, b[1]*sy
        dx, dy = bx-ax, by-ay
        t = np.clip(((px-ax)*dx + (py-ay)*dy)/(dx*dx+dy*dy), 0, 1)
        radius = r0 + t*(r1-r0)
        mask = (px-ax-t*dx)**2 + (py-ay-t*dy)**2 <= radius**2
        land[mask] = False


# Mainland east of Portland Inlet, Chatham Sound and Grenville Channel.
add([
    (-129.96,55.12),(-128.7,55.12),(-128.7,53.30),
    (-129.12,53.38),(-129.25,53.43),(-129.30,53.49),
    (-129.29,53.57),(-129.23,53.67),(-129.18,53.76),
    (-129.12,53.88),(-129.07,54.00),(-129.03,54.07),
    (-129.13,54.11),(-129.23,53.99),(-129.29,53.89),
    (-129.31,53.79),(-129.37,53.66),(-129.40,53.55),
    (-129.40,53.48),(-129.51,53.52),(-129.59,53.61),
    (-129.69,53.73),(-129.81,53.84),(-129.93,53.96),
    (-130.01,54.07),(-130.03,54.13),(-130.03,54.18),
    (-130.12,54.21),(-130.20,54.23),(-130.23,54.27),
    (-130.22,54.32),(-130.26,54.36),(-130.31,54.37),
    (-130.35,54.35),(-130.41,54.36),(-130.45,54.39),
    (-130.44,54.44),(-130.46,54.49),(-130.44,54.54),
    (-130.42,54.57),(-130.44,54.60),(-130.38,54.64),
    (-130.30,54.68),(-130.23,54.73),(-130.18,54.79),
    (-130.10,54.84),(-130.04,54.90),(-129.98,54.94),
    (-129.96,54.99),(-129.98,55.05),
])

# Southeast Alaska mainland: Cape Fox and the coast west of Pearse Canal.
add([
    (-131.25,55.15),(-130.37,55.15),(-130.31,55.08),
    (-130.35,55.02),(-130.39,54.96),(-130.45,54.90),
    (-130.53,54.85),(-130.62,54.81),(-130.68,54.78),
    (-130.76,54.79),(-130.83,54.77),(-130.92,54.78),
    (-130.98,54.77),(-131.01,54.81),(-131.02,54.86),
    (-131.06,54.89),(-131.04,54.93),(-131.10,54.96),
    (-131.12,55.02),(-131.20,55.07),
])

# Pearse Island and Wales Island, keeping the boundary passages open.
add([
    (-130.29,55.09),(-130.22,55.08),(-130.14,55.03),
    (-130.12,54.98),(-130.16,54.94),(-130.18,54.90),
    (-130.24,54.85),(-130.28,54.81),(-130.35,54.79),
    (-130.41,54.83),(-130.43,54.89),(-130.38,54.94),
    (-130.34,55.01),
])
add([
    (-130.60,54.82),(-130.51,54.84),(-130.45,54.81),
    (-130.43,54.76),(-130.40,54.72),(-130.43,54.67),
    (-130.49,54.65),(-130.55,54.67),(-130.58,54.72),
    (-130.63,54.75),
])
# Outer Chatham Sound islands: Zayas, Dundas, and the smaller southern group.
add([
    (-131.02,54.82),(-130.96,54.83),(-130.91,54.80),
    (-130.89,54.75),(-130.92,54.72),(-130.98,54.73),
    (-131.02,54.77),
])
add([
    (-131.01,54.63),(-130.96,54.67),(-130.89,54.68),
    (-130.84,54.65),(-130.79,54.65),(-130.76,54.61),
    (-130.72,54.60),(-130.74,54.56),(-130.78,54.54),
    (-130.78,54.50),(-130.84,54.48),(-130.88,54.51),
    (-130.95,54.50),(-130.98,54.54),(-131.01,54.57),
])
add([
    (-130.98,54.49),(-130.92,54.50),(-130.89,54.47),
    (-130.91,54.43),(-130.96,54.42),(-130.99,54.45),
])
add([
    (-130.85,54.48),(-130.79,54.49),(-130.74,54.46),
    (-130.75,54.42),(-130.81,54.40),(-130.85,54.43),
])
add([
    (-130.80,54.40),(-130.74,54.42),(-130.69,54.39),
    (-130.70,54.35),(-130.76,54.34),(-130.81,54.37),
])

# Stephens Island and Prescott Island.
add([
    (-130.89,54.37),(-130.85,54.40),(-130.80,54.37),
    (-130.77,54.33),(-130.72,54.32),(-130.69,54.28),
    (-130.64,54.26),(-130.64,54.22),(-130.69,54.20),
    (-130.73,54.22),(-130.79,54.23),(-130.82,54.27),
    (-130.87,54.30),
])
add([
    (-130.83,54.20),(-130.77,54.21),(-130.72,54.18),
    (-130.74,54.15),(-130.80,54.14),(-130.84,54.17),
])

# Porcher Island: irregular headlands, Kitkatla Inlet cut later.
add([
    (-130.76,54.16),(-130.70,54.20),(-130.64,54.19),
    (-130.58,54.16),(-130.52,54.17),(-130.47,54.14),
    (-130.44,54.11),(-130.37,54.10),(-130.33,54.06),
    (-130.29,54.06),(-130.27,54.02),(-130.29,53.99),
    (-130.25,53.95),(-130.22,53.90),(-130.25,53.86),
    (-130.31,53.83),(-130.34,53.79),(-130.40,53.78),
    (-130.45,53.82),(-130.50,53.83),(-130.53,53.88),
    (-130.59,53.88),(-130.64,53.91),(-130.67,53.96),
    (-130.65,54.01),(-130.71,54.04),(-130.73,54.10),
    (-130.78,54.13),
])

# Digby, Kaien and Ridley Islands at Prince Rupert.
add([
    (-130.49,54.34),(-130.44,54.36),(-130.39,54.33),
    (-130.37,54.29),(-130.38,54.26),(-130.35,54.23),
    (-130.40,54.21),(-130.44,54.23),(-130.48,54.25),
    (-130.51,54.29),
])
add([
    (-130.32,54.35),(-130.27,54.34),(-130.25,54.31),
    (-130.24,54.27),(-130.26,54.23),(-130.29,54.22),
    (-130.31,54.25),(-130.34,54.28),(-130.35,54.31),
])
add([
    (-130.34,54.25),(-130.31,54.24),(-130.30,54.21),
    (-130.32,54.19),(-130.35,54.21),
])

# Skeena estuary islands: Smith, De Horsey, Kennedy and a few delta islands.
add([
    (-130.31,54.21),(-130.24,54.20),(-130.20,54.17),
    (-130.20,54.13),(-130.25,54.10),(-130.30,54.11),
    (-130.35,54.15),
])
add([
    (-130.17,54.19),(-130.12,54.19),(-130.09,54.16),
    (-130.10,54.12),(-130.14,54.11),(-130.18,54.14),
])
add([
    (-130.06,54.18),(-130.01,54.17),(-129.99,54.14),
    (-130.02,54.10),(-130.07,54.11),(-130.09,54.14),
])

# Dolphin Island and nearby islands at the entrance to Kitkatla Inlet.
add([
    (-130.48,53.83),(-130.42,53.82),(-130.39,53.78),
    (-130.42,53.75),(-130.48,53.76),(-130.51,53.79),
])
add([
    (-130.38,53.82),(-130.33,53.82),(-130.29,53.78),
    (-130.30,53.74),(-130.35,53.75),(-130.38,53.78),
])
# Banks Island: the outer island south of Browning Entrance.
add([
    (-130.57,53.69),(-130.49,53.73),(-130.41,53.71),
    (-130.37,53.67),(-130.36,53.62),(-130.29,53.57),
    (-130.25,53.51),(-130.19,53.47),(-130.15,53.40),
    (-130.09,53.35),(-130.06,53.28),(-130.00,53.24),
    (-129.96,53.17),(-129.92,53.12),(-129.95,53.07),
    (-130.02,53.06),(-130.08,53.11),(-130.15,53.15),
    (-130.18,53.21),(-130.26,53.27),(-130.31,53.34),
    (-130.39,53.39),(-130.44,53.47),(-130.49,53.51),
    (-130.52,53.59),(-130.57,53.63),
])

# Pitt Island between Grenville, Principe and Otter Channels.
add([
    (-130.13,54.02),(-130.06,54.04),(-130.01,54.01),
    (-129.98,53.97),(-129.92,53.92),(-129.87,53.87),
    (-129.82,53.82),(-129.77,53.78),(-129.70,53.71),
    (-129.66,53.65),(-129.60,53.59),(-129.56,53.53),
    (-129.49,53.47),(-129.44,53.42),(-129.43,53.35),
    (-129.47,53.30),(-129.46,53.25),(-129.51,53.19),
    (-129.57,53.14),(-129.63,53.15),(-129.66,53.21),
    (-129.70,53.27),(-129.71,53.34),(-129.77,53.40),
    (-129.79,53.47),(-129.87,53.54),(-129.91,53.62),
    (-129.98,53.68),(-130.02,53.75),(-130.08,53.81),
    (-130.12,53.89),(-130.11,53.95),(-130.15,53.99),
])

# McCauley Island, southwest of Pitt and east of Principe Channel.
add([
    (-130.21,53.66),(-130.15,53.69),(-130.09,53.65),
    (-130.05,53.59),(-129.99,53.55),(-129.96,53.49),
    (-129.90,53.45),(-129.86,53.38),(-129.82,53.34),
    (-129.85,53.29),(-129.91,53.30),(-129.96,53.34),
    (-130.03,53.37),(-130.07,53.43),(-130.12,53.48),
    (-130.15,53.55),(-130.20,53.59),
])

# Campania, Gil, and the northwestern edge of Princess Royal Island.
add([
    (-129.55,53.17),(-129.49,53.18),(-129.43,53.12),
    (-129.41,53.05),(-129.39,52.97),(-129.40,52.86),
    (-129.51,52.83),(-129.60,52.91),(-129.62,53.00),
    (-129.59,53.08),
])
add([
    (-129.31,53.35),(-129.24,53.34),(-129.19,53.29),
    (-129.14,53.24),(-129.13,53.18),(-129.18,53.13),
    (-129.23,53.08),(-129.29,53.08),(-129.33,53.13),
    (-129.38,53.17),(-129.41,53.23),(-129.38,53.29),
])
add([
    (-128.89,53.22),(-128.7,53.22),(-128.7,52.8),
    (-129.24,52.8),(-129.21,52.94),(-129.17,53.02),
    (-129.11,53.09),(-129.05,53.13),(-128.99,53.17),
])


def islet(lon, lat, radius_east_km, radius_north_km):
    # Simple ellipses are also handwritten approximations, not retrieved data.
    dx = (X-lon)*111.32*math.cos(math.radians(lat))/radius_east_km
    dy = (Y-lat)*111.32/radius_north_km
    land[:] |= dx*dx + dy*dy <= 1


# Smaller islands remembered in the sounds and passages.
for spec in [
    (-130.74,54.76,1.4,2.0),  # Tongass
    (-130.82,54.79,1.8,2.3),  # island by Tongass Passage
    (-130.70,54.56,0.9,1.0),  # Green Island
    (-130.59,54.30,1.1,0.7),  # Lucy Islands
    (-130.88,54.295,0.35,0.35),  # Triple Islands
    (-130.64,53.49,1.4,2.0),  # Bonilla Island
    (-130.29,53.74,1.4,2.0),
    (-130.18,53.92,1.8,3.0),
    (-130.20,53.98,1.3,1.5),
    (-129.76,53.22,2.2,3.0),
    (-129.72,53.14,1.1,1.6),
]:
    islet(*spec)
# Marine fjords only: inland lakes are deliberately NOT carved out.
# Boca de Quadra and small southeastern Alaska embayments.
inlet([(-131.12,54.975),(-131.00,54.982),(-130.91,55.015),
       (-130.84,55.065),(-130.78,55.12)], [1.5,1.2,1.0,1.0,0.6])
inlet([(-130.96,54.995),(-130.89,54.965),(-130.80,54.96)],
      [0.85,0.7,0.35])
inlet([(-131.025,54.88),(-130.96,54.89),(-130.92,54.92)],
      [0.8,0.65,0.3])

# Observatory Inlet and the tidal part of Nass Bay.
inlet([(-130.035,54.925),(-129.985,54.975),(-129.94,55.04),
       (-129.91,55.12)], [1.6,1.8,1.4,1.2])
inlet([(-129.985,54.973),(-129.89,54.976),(-129.80,54.998),
       (-129.73,55.015)], [1.7,1.8,1.35,0.45])

# Khutzeymateen Inlet and Work Channel cut the mountainous mainland.
inlet([(-130.215,54.765),(-130.16,54.72),(-130.05,54.68),
       (-129.93,54.675),(-129.80,54.70),(-129.72,54.69)],
      [1.5,1.4,1.2,1.05,0.75,0.3])
inlet([(-130.325,54.675),(-130.26,54.615),(-130.19,54.55),
       (-130.115,54.48),(-130.04,54.42),(-130.00,54.36),
       (-129.985,54.325)], [1.15,1.05,0.85,0.75,0.65,0.55,0.3])
inlet([(-130.19,54.55),(-130.10,54.57),(-130.045,54.56)],
      [0.6,0.45,0.2])

# Prince Rupert Harbour, Tuck Inlet and the passages beside Kaien Island.
inlet([(-130.35,54.345),(-130.30,54.39),(-130.25,54.43),
       (-130.215,54.465)], [0.85,0.75,0.6,0.3])
inlet([(-130.25,54.345),(-130.235,54.30),(-130.24,54.265),
       (-130.26,54.235)], [0.35,0.35,0.4,0.5])

# Skeena tidal estuary and Inverness Passage.
inlet([(-130.22,54.205),(-130.12,54.205),(-130.00,54.202),
       (-129.90,54.215),(-129.78,54.23),(-129.66,54.25)],
      [1.1,1.35,1.5,1.1,0.75,0.3])
inlet([(-130.35,54.205),(-130.28,54.22),(-130.20,54.205)],
      [0.65,0.55,0.6])
inlet([(-129.99,54.19),(-129.94,54.155),(-129.88,54.13)],
      [0.65,0.5,0.25])

# Porcher Island's saltwater indentations.
inlet([(-130.44,53.79),(-130.44,53.86),(-130.49,53.92),
       (-130.53,53.96),(-130.51,54.005)], [1.5,1.4,1.1,0.8,0.3])
inlet([(-130.72,54.10),(-130.65,54.085),(-130.60,54.10)],
      [0.9,0.7,0.25])
inlet([(-130.29,54.025),(-130.36,54.015),(-130.39,54.04)],
      [0.65,0.5,0.2])

# Side fjords of Grenville Channel, with a very narrow coastal entrance.
inlet([(-129.90,53.915),(-129.97,53.925),(-130.035,53.95)],
      [0.4,0.55,0.25])
inlet([(-129.87,53.91),(-129.79,53.93),(-129.72,53.96)],
      [0.5,0.7,0.3])
inlet([(-129.655,53.695),(-129.57,53.715),(-129.50,53.75)],
      [0.55,0.85,0.3])
inlet([(-129.565,53.56),(-129.50,53.57),(-129.48,53.60)],
      [0.5,0.75,0.3])
inlet([(-129.64,53.62),(-129.70,53.59),(-129.75,53.60)],
      [0.4,0.6,0.25])

# Banks and McCauley have several long narrow marine embayments.
inlet([(-130.21,53.49),(-130.29,53.50),(-130.36,53.54)],
      [0.7,0.9,0.3])
inlet([(-130.07,53.32),(-130.16,53.34),(-130.23,53.38)],
      [0.65,0.85,0.3])
inlet([(-129.91,53.43),(-130.00,53.46),(-130.06,53.50)],
      [0.55,0.75,0.3])

# Douglas Channel, its southern junction and the eastern boundary islands.
inlet([(-129.34,53.43),(-129.32,53.55),(-129.25,53.68),
       (-129.19,53.81),(-129.13,53.93),(-129.04,54.06)],
      [2.0,2.0,2.4,2.1,1.9,1.7])
# The passage north of Gribbell and south of Hawkesbury.
inlet([(-129.28,53.59),(-129.16,53.605),(-129.06,53.61),
       (-128.92,53.62)], [1.4,1.3,1.1,1.0])
# Verney Passage, mostly outside the eastern edge of this window.
inlet([(-129.02,53.605),(-128.99,53.71),(-128.96,53.82)],
      [0.8,0.8,0.7])
# Output and exhaustive format / pixel validation.
# No coastal smoothing, resampling before classification, or majority voting.
matrix = land.astype(np.uint8)
out = Path.cwd() / "outputs"
out.mkdir(parents=True, exist_ok=True)
csv_path = out / "candidate.csv"
small_path = out / "candidate.png"
large_path = out / "candidate_8x.png"

with csv_path.open("w", encoding="ascii", newline="") as f:
    csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

pixels = matrix * np.uint8(255)
small = Image.fromarray(pixels)
small.save(small_path, format="PNG")
resampling = getattr(Image, "Resampling", Image)
large = small.resize((N*8, N*8), resample=resampling.NEAREST)
large.save(large_path, format="PNG")

assert matrix.shape == (127, 127)
assert matrix.size == 16129
assert np.isin(matrix, [0, 1]).all()
assert LONGITUDES[0] == -131 and LONGITUDES[-1] == -129
assert LATITUDES[0] == 55 and LATITUDES[-1] == 53
with csv_path.open("r", encoding="ascii", newline="") as f:
    rows = list(csv.reader(f))
assert len(rows) == 127
assert all(len(row) == 127 for row in rows)
assert all(value in ("0", "1") for row in rows for value in row)
decoded_csv = np.array(rows, dtype=np.uint8)
assert np.array_equal(decoded_csv, matrix)

with Image.open(small_path) as image:
    assert image.size == (127, 127) and image.mode == "L"
    original = np.array(image)
with Image.open(large_path) as image:
    assert image.size == (1016, 1016) and image.mode == "L"
    enlarged = np.array(image)
assert set(np.unique(original).tolist()) == {0, 255}
assert set(np.unique(enlarged).tolist()) == {0, 255}
assert np.array_equal(original, pixels)
expected = np.repeat(np.repeat(original, 8, axis=0), 8, axis=1)
assert np.array_equal(enlarged, expected)
assert np.array_equal(enlarged[::8, ::8], original)

print(json.dumps({
    "method": "handwritten knowledge-only polygons and marine inlet rules",
    "shape": list(matrix.shape),
    "valid_values": int(matrix.size),
    "land_points": int(matrix.sum()),
    "water_points": int(matrix.size - matrix.sum()),
    "csv_valid": True,
    "png_original": [127, 127],
    "png_enlarged": [1016, 1016],
    "colors": [0, 255],
    "nearest_neighbor_verified": True,
    "files": [str(csv_path), str(small_path), str(large_path)],
}, ensure_ascii=False, indent=2))
