"""Memory-only reconstruction; no map, geographic dataset, or network input.

All vertices below are hand-estimated (longitude, latitude), not surveyed.
Classification is point-in-polygon on the exact requested sampling lattice,
not independent model reasoning for each point and not cell-area voting.
Dependencies: numpy, Pillow. Run from the desired output parent directory.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

LAND = []
WATER = []

def land(points):
    LAND.append(points)

def water(points):
    WATER.append(points)

# Mainland south/east of Portland Inlet, including Tsimpsean Peninsula.
land([
    (-129.00,55.10),(-129.57,55.10),(-129.68,54.98),
    (-129.69,54.88),(-129.82,54.84),(-129.96,54.78),
    (-130.09,54.73),(-130.22,54.67),(-130.37,54.64),
    (-130.43,54.58),(-130.44,54.53),(-130.40,54.48),
    (-130.44,54.42),(-130.44,54.36),(-130.36,54.34),
    (-130.29,54.36),(-130.23,54.31),(-130.20,54.24),
    (-130.15,54.21),(-130.12,54.17),(-130.04,54.18),
    (-129.98,54.15),(-129.97,54.10),(-129.89,54.04),
    (-129.85,53.98),(-129.80,53.93),(-129.78,53.86),
    (-129.72,53.80),(-129.68,53.75),(-129.64,53.69),
    (-129.58,53.63),(-129.54,53.57),(-129.49,53.52),
    (-129.46,53.47),(-129.38,53.44),(-129.29,53.44),
    (-129.25,53.40),(-129.20,53.38),(-129.16,53.31),
    (-129.10,53.27),(-129.04,53.20),(-129.02,53.11),
    (-128.94,53.06),(-128.90,52.90),(-128.80,52.90),
    (-128.80,55.10)
])

# Northern mainland west of Portland Canal.
land([
    (-130.80,55.10),(-130.76,55.00),(-130.68,54.96),
    (-130.65,54.88),(-130.61,54.84),(-130.56,54.80),
    (-130.55,54.75),(-130.46,54.73),(-130.40,54.77),
    (-130.33,54.81),(-130.24,54.85),(-130.19,54.91),
    (-130.16,54.96),(-130.16,55.10)
])

# Wales and Pearse islands: Portland Inlet/Canal remain water.
land([
    (-130.62,54.72),(-130.64,54.79),(-130.59,54.88),
    (-130.51,54.94),(-130.43,54.95),(-130.39,54.90),
    (-130.42,54.84),(-130.48,54.77),(-130.53,54.72)
])
land([
    (-130.35,54.89),(-130.34,54.97),(-130.27,55.07),
    (-130.18,55.10),(-130.21,54.98),(-130.24,54.92),
    (-130.29,54.86)
])

# Dundas, Zayas, Stephens, and Melville islands.
land([
    (-131.04,54.56),(-131.01,54.62),(-130.96,54.66),
    (-130.88,54.65),(-130.80,54.61),(-130.76,54.55),
    (-130.77,54.49),(-130.82,54.46),(-130.87,54.49),
    (-130.92,54.47),(-130.95,54.52),(-131.01,54.52)
])
land([
    (-131.03,54.74),(-131.00,54.79),(-130.93,54.80),
    (-130.88,54.76),(-130.89,54.70),(-130.94,54.68),
    (-131.00,54.70)
])
land([
    (-130.84,54.39),(-130.79,54.45),(-130.73,54.47),
    (-130.66,54.46),(-130.62,54.41),(-130.64,54.35),
    (-130.69,54.31),(-130.75,54.32),(-130.78,54.36)
])
land([
    (-130.94,54.42),(-130.91,54.47),(-130.87,54.48),
    (-130.83,54.44),(-130.85,54.39),(-130.89,54.38)
])

# Digby and Kaien islands at Prince Rupert.
land([
    (-130.51,54.27),(-130.49,54.33),(-130.44,54.35),
    (-130.39,54.32),(-130.38,54.28),(-130.42,54.24),
    (-130.47,54.23)
])
land([
    (-130.36,54.30),(-130.35,54.33),(-130.30,54.34),
    (-130.26,54.31),(-130.25,54.27),(-130.28,54.23),
    (-130.32,54.25)
])

# Porcher Island: Chatham Sound north, Hecate Strait west.
land([
    (-130.76,54.04),(-130.70,54.10),(-130.62,54.12),
    (-130.57,54.16),(-130.48,54.14),(-130.43,54.10),
    (-130.35,54.10),(-130.31,54.05),(-130.32,54.00),
    (-130.27,53.96),(-130.29,53.90),(-130.36,53.85),
    (-130.43,53.86),(-130.49,53.90),(-130.56,53.88),
    (-130.63,53.92),(-130.64,53.97),(-130.71,53.98)
])
# Inlets open to the sea, not inland lakes.
water([
    (-130.77,54.06),(-130.60,54.05),(-130.51,54.02),
    (-130.46,53.97),(-130.52,53.99),(-130.62,54.02),
    (-130.77,54.03)
])
water([
    (-130.44,53.83),(-130.42,53.96),(-130.39,54.01),
    (-130.39,53.93),(-130.40,53.83)
])
# McCauley, Dolphin, and smaller approaches to Ogden Channel.
land([
    (-130.24,54.03),(-130.16,54.04),(-130.10,53.99),
    (-130.09,53.94),(-130.13,53.90),(-130.22,53.91),
    (-130.25,53.96)
])
land([
    (-130.34,53.88),(-130.30,53.91),(-130.27,53.88),
    (-130.28,53.81),(-130.33,53.82)
])
land([
    (-130.13,54.14),(-130.09,54.16),(-130.04,54.13),
    (-130.03,54.08),(-130.08,54.06),(-130.13,54.10)
])
# Goschen Island.
land([
    (-130.71,53.84),(-130.68,53.91),(-130.61,53.93),
    (-130.54,53.88),(-130.53,53.82),(-130.59,53.77),
    (-130.66,53.78)
])

# Banks Island, west of Principe Channel.
land([
    (-130.64,53.97),(-130.55,53.98),(-130.47,53.94),
    (-130.41,53.90),(-130.33,53.86),(-130.27,53.79),
    (-130.22,53.71),(-130.13,53.65),(-130.08,53.57),
    (-130.01,53.49),(-129.96,53.43),(-129.89,53.35),
    (-129.84,53.27),(-129.78,53.19),(-129.73,53.11),
    (-129.78,53.08),(-129.86,53.10),(-129.95,53.17),
    (-130.04,53.23),(-130.10,53.32),(-130.20,53.40),
    (-130.28,53.47),(-130.37,53.54),(-130.42,53.63),
    (-130.49,53.68),(-130.52,53.75),(-130.57,53.79),
    (-130.59,53.88),(-130.64,53.92)
])
# Pitt Island, east of Principe and west of Grenville Channel.
land([
    (-130.13,53.99),(-130.05,54.00),(-129.99,53.94),
    (-129.92,53.88),(-129.87,53.81),(-129.79,53.74),
    (-129.73,53.66),(-129.67,53.59),(-129.61,53.52),
    (-129.56,53.46),(-129.50,53.40),(-129.45,53.31),
    (-129.45,53.25),(-129.51,53.20),(-129.58,53.22),
    (-129.65,53.28),(-129.70,53.36),(-129.76,53.42),
    (-129.82,53.51),(-129.87,53.57),(-129.93,53.64),
    (-130.02,53.71),(-130.07,53.80),(-130.12,53.86),
    (-130.16,53.91)
])
# Campania, Gil, and the northern Estevan islands.
land([
    (-129.57,53.18),(-129.49,53.20),(-129.42,53.15),
    (-129.38,53.09),(-129.33,53.01),(-129.32,52.91),
    (-129.40,52.89),(-129.46,52.96),(-129.50,53.04),
    (-129.56,53.09)
])
land([
    (-129.37,53.26),(-129.31,53.28),(-129.25,53.24),
    (-129.21,53.18),(-129.17,53.10),(-129.18,53.03),
    (-129.24,52.98),(-129.30,53.03),(-129.33,53.12),
    (-129.38,53.19)
])
land([
    (-129.82,53.08),(-129.74,53.09),(-129.68,53.04),
    (-129.65,52.94),(-129.73,52.92),(-129.78,52.98)
])
land([
    (-129.62,53.03),(-129.58,53.08),(-129.54,53.04),
    (-129.54,52.95),(-129.60,52.94)
])
# Bonilla Island, and a few remembered coastal islets.
land([(-130.68,53.50),(-130.64,53.52),(-130.61,53.49),
      (-130.63,53.46),(-130.67,53.47)])
land([(-130.60,54.23),(-130.55,54.25),(-130.51,54.22),
      (-130.54,54.18),(-130.58,54.19)])
land([(-130.66,54.30),(-130.61,54.32),(-130.59,54.29),
      (-130.62,54.26)])

# Narrow the hand-estimated mainland/Pitt separation along Grenville.
# This is an explicit correction to the hand-drawn outline, not map input.
LAND[0] = [(x - 0.075 if 53.47 <= y <= 54.10 else x, y)
           for x, y in LAND[0]]

# Work Channel, Skeena estuary, Observatory Inlet and Douglas Channel.
water([
    (-130.26,54.73),(-130.19,54.69),(-130.13,54.63),
    (-130.06,54.57),(-129.98,54.52),(-129.91,54.46),
    (-129.88,54.44),(-129.92,54.50),(-130.00,54.56),
    (-130.08,54.62),(-130.15,54.70),(-130.21,54.76)
])
water([
    (-130.18,54.16),(-130.10,54.20),(-130.00,54.20),
    (-129.90,54.23),(-129.79,54.25),(-129.67,54.29),
    (-129.55,54.31),(-129.53,54.30),(-129.66,54.27),
    (-129.78,54.23),(-129.88,54.20),(-130.00,54.17),
    (-130.13,54.13)
])
water([
    (-129.97,54.81),(-129.94,54.88),(-129.92,54.94),
    (-129.91,55.03),(-129.85,55.10),(-129.81,55.10),
    (-129.87,54.99),(-129.87,54.92),(-129.90,54.85),
    (-129.91,54.81)
])
water([
    (-129.29,53.42),(-129.22,53.47),(-129.16,53.52),
    (-129.10,53.58),(-129.02,53.63),(-128.90,53.68),
    (-128.90,53.62),(-129.02,53.59),(-129.09,53.54),
    (-129.15,53.48),(-129.20,53.42),(-129.24,53.39)
])
water([
    (-129.57,53.57),(-129.47,53.59),(-129.40,53.62),
    (-129.37,53.67),(-129.35,53.66),(-129.38,53.60),
    (-129.45,53.56),(-129.54,53.54)
])

def inside_polygon(x, y, vertices):
    """Vectorized even/odd point rule; deterministic on boundary points."""
    inside = np.zeros(x.shape, dtype=bool)
    ax, ay = vertices[-1]
    for bx, by in vertices:
        if ay != by:
            crosses = (ay > y) != (by > y)
            intersection = ax + (y - ay) * (bx - ax) / (by - ay)
            inside ^= crosses & (x < intersection)
        ax, ay = bx, by
    return inside

def main():
    out = Path.cwd() / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    # Endpoints and step are exact by construction; do not use rounded labels.
    lon = -131.0 + np.arange(127, dtype=np.float64) * (2.0 / 126.0)
    lat = 55.0 - np.arange(127, dtype=np.float64) * (2.0 / 126.0)
    x, y = np.meshgrid(lon, lat)
    mask = np.zeros((127, 127), dtype=bool)
    for polygon in LAND:
        mask |= inside_polygon(x, y, polygon)
    for polygon in WATER:
        mask &= ~inside_polygon(x, y, polygon)
    matrix = mask.astype(np.uint8)
    assert matrix.shape == (127, 127) and matrix.size == 16129
    assert np.isin(matrix, [0, 1]).all()
    csv_path = out / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(matrix.tolist())
    pixels = matrix * np.uint8(255)
    image = Image.fromarray(pixels)
    image.save(out / "candidate.png")
    enlarged = image.resize((1016, 1016), Image.Resampling.NEAREST)
    enlarged.save(out / "candidate_8x.png")

    # Validate saved artifacts, including every pixel of the 8x expansion.
    with csv_path.open(newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 127
    assert all(len(row) == 127 for row in rows)
    assert all(value in ("0", "1") for row in rows for value in row)
    assert np.array_equal(np.array(rows, dtype=np.uint8), matrix)
    with Image.open(out / "candidate.png") as small:
        assert small.size == (127, 127)
        saved_small = np.array(small)
    with Image.open(out / "candidate_8x.png") as big:
        assert big.size == (1016, 1016)
        saved_big = np.array(big)
    assert set(np.unique(saved_small).tolist()) <= {0, 255}
    assert set(np.unique(saved_big).tolist()) <= {0, 255}
    assert np.array_equal(saved_small, pixels)
    assert np.array_equal(saved_big, pixels.repeat(8, axis=0).repeat(8, axis=1))
    print("Validated: 127x127 = 16,129 binary point classifications;")
    print("CSV complete; PNGs 127x127 and 1016x1016; exact nearest-neighbor 8x.")
    print("Land:", int(matrix.sum()), "Water:", int(matrix.size-matrix.sum()))

if __name__ == "__main__":
    main()
