"""Knowledge-only reconstruction of the Hecate Strait coast (53-55 N, 131-129 W).

No geographic files or services are read.  Every outline below was sketched from
memory as a polygon in (longitude, latitude); the matrix is then point-sampled.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

# Mainland shore.  The jagged western edge is intentionally schematic: the
# large notches represent Portland Inlet, Work Channel and the Skeena estuary.
MAINLAND = [
 [(-129.00,55.00),(-129.00,53.00),(-129.03,53.00),(-129.08,53.07),
  (-129.02,53.14),(-129.07,53.22),(-129.02,53.30),(-129.09,53.38),
  (-129.03,53.47),(-129.10,53.55),(-129.03,53.64),(-129.08,53.72),
  (-129.02,53.82),(-129.10,53.91),(-129.04,54.00),(-129.11,54.07),
  (-129.22,54.10),(-129.34,54.12),(-129.48,54.16),(-129.59,54.21),
  (-129.66,54.28),(-129.64,54.34),(-129.52,54.39),(-129.40,54.44),
  (-129.31,54.50),(-129.28,54.57),(-129.36,54.63),(-129.49,54.66),
  (-129.61,54.64),(-129.72,54.59),(-129.82,54.55),(-129.91,54.56),
  (-129.96,54.62),(-129.92,54.70),(-129.81,54.77),(-129.68,54.82),
  (-129.57,54.88),(-129.49,54.95),(-129.47,55.00)],
 # Tsimpsean peninsula, separated schematically from the mainland by inlets.
 [(-130.42,54.42),(-130.48,54.49),(-130.44,54.58),(-130.35,54.67),
  (-130.23,54.73),(-130.10,54.76),(-130.01,54.72),(-130.00,54.64),
  (-130.06,54.57),(-130.08,54.49),(-130.02,54.42),(-129.93,54.37),
  (-129.90,54.30),(-129.97,54.25),(-130.08,54.27),(-130.19,54.34)],
]

# Named and unnamed coastal islands.  Fine crenulations are hand-drawn rather
# than generated noise, so all geometry remains an explicit knowledge rule.
ISLANDS = [
 # Rose Spit, just reaching the western edge of the frame
 [(-131.00,54.205),(-130.965,54.190),(-130.940,54.173),
  (-130.968,54.166),(-131.00,54.172)],
 # Dundas group
 [(-130.98,54.61),(-130.94,54.68),(-130.84,54.73),(-130.72,54.72),
  (-130.64,54.67),(-130.65,54.59),(-130.72,54.52),(-130.83,54.49),
  (-130.93,54.53)],
 [(-130.66,54.44),(-130.60,54.49),(-130.49,54.49),(-130.43,54.44),
  (-130.47,54.38),(-130.57,54.37),(-130.64,54.40)],
 [(-130.91,54.42),(-130.87,54.46),(-130.79,54.45),(-130.76,54.40),
  (-130.81,54.36),(-130.89,54.37)],
 # Stephens Island
 [(-130.88,54.16),(-130.82,54.25),(-130.73,54.30),(-130.62,54.28),
  (-130.56,54.22),(-130.58,54.14),(-130.66,54.08),(-130.78,54.07)],
 # Digby and Kaien Islands at Prince Rupert
 [(-130.53,54.26),(-130.48,54.34),(-130.39,54.35),(-130.35,54.29),
  (-130.38,54.22),(-130.46,54.20)],
 [(-130.36,54.20),(-130.33,54.31),(-130.24,54.37),(-130.16,54.33),
  (-130.15,54.23),(-130.21,54.16),(-130.30,54.14)],
 # Porcher Island
 [(-130.72,53.91),(-130.67,54.04),(-130.58,54.13),(-130.45,54.16),
  (-130.33,54.12),(-130.28,54.04),(-130.31,53.94),(-130.40,53.85),
  (-130.54,53.83),(-130.65,53.86)],
 # small outer islands west and south of Porcher
 [(-130.90,53.99),(-130.86,54.04),(-130.78,54.03),(-130.76,53.97),
  (-130.82,53.93),(-130.89,53.94)],
 [(-130.61,53.72),(-130.57,53.78),(-130.49,53.79),(-130.44,53.74),
  (-130.48,53.68),(-130.56,53.67)],
 # Banks Island
 [(-130.52,53.32),(-130.50,53.45),(-130.46,53.58),(-130.43,53.72),
  (-130.37,53.84),(-130.28,53.91),(-130.18,53.88),(-130.13,53.78),
  (-130.14,53.65),(-130.18,53.52),(-130.23,53.39),(-130.31,53.27),
  (-130.41,53.23),(-130.48,53.26)],
 # McCauley Island
 [(-130.13,53.48),(-130.10,53.62),(-130.05,53.75),(-129.97,53.83),
  (-129.89,53.80),(-129.87,53.68),(-129.91,53.55),(-129.99,53.45),
  (-130.07,53.42)],
]
# Continuation of the explicit island list.
ISLANDS += [
 # Pitt Island, long and narrow between Principe and Grenville Channels
 [(-129.99,53.32),(-129.94,53.44),(-129.91,53.58),(-129.89,53.72),
  (-129.85,53.87),(-129.78,54.01),(-129.69,54.10),(-129.59,54.12),
  (-129.52,54.07),(-129.53,53.98),(-129.59,53.87),(-129.62,53.73),
  (-129.66,53.58),(-129.72,53.45),(-129.81,53.34),(-129.91,53.28)],
 # Fin, Farrant and other islands north/east of Pitt
 [(-129.61,54.13),(-129.56,54.20),(-129.47,54.23),(-129.40,54.20),
  (-129.42,54.14),(-129.50,54.10),(-129.57,54.09)],
 [(-129.48,53.91),(-129.44,54.00),(-129.36,54.05),(-129.29,54.02),
  (-129.30,53.93),(-129.37,53.87),(-129.44,53.86)],
 [(-129.58,53.60),(-129.53,53.69),(-129.45,53.72),(-129.39,53.67),
  (-129.42,53.58),(-129.50,53.53),(-129.56,53.55)],
 # Gil and Gribbell groups
 [(-129.48,53.20),(-129.44,53.32),(-129.37,53.42),(-129.28,53.45),
  (-129.21,53.40),(-129.22,53.29),(-129.27,53.18),(-129.35,53.12),
  (-129.43,53.14)],
 [(-129.21,53.32),(-129.16,53.44),(-129.09,53.51),(-129.02,53.49),
  (-129.00,53.39),(-129.04,53.29),(-129.12,53.25)],
 # Campania Island (southern edge of the requested frame)
 [(-129.64,53.00),(-129.62,53.08),(-129.56,53.17),(-129.47,53.20),
  (-129.39,53.16),(-129.36,53.08),(-129.39,53.00)],
 # northern tip of Princess Royal Island
 [(-129.26,53.00),(-129.23,53.09),(-129.17,53.16),(-129.09,53.18),
  (-129.02,53.13),(-129.00,53.00)],
 # Goschen and Dolphin islands in the Skeena approaches
 [(-130.10,53.98),(-130.06,54.08),(-129.98,54.13),(-129.91,54.10),
  (-129.90,54.02),(-129.95,53.94),(-130.03,53.92)],
 [(-130.21,54.08),(-130.18,54.15),(-130.10,54.18),(-130.05,54.14),
  (-130.08,54.07),(-130.15,54.04)],
 # small islets retained when they are large enough to hit the sample lattice
 [(-130.77,54.76),(-130.73,54.80),(-130.67,54.79),(-130.65,54.74),
  (-130.71,54.71),(-130.76,54.72)],
 [(-130.55,54.57),(-130.52,54.62),(-130.45,54.63),(-130.42,54.58),
  (-130.46,54.53),(-130.52,54.53)],
 [(-130.22,54.48),(-130.18,54.54),(-130.10,54.55),(-130.06,54.49),
  (-130.11,54.44),(-130.18,54.44)],
 [(-130.95,53.76),(-130.91,53.81),(-130.84,53.80),(-130.82,53.74),
  (-130.88,53.70),(-130.94,53.71)],
 [(-130.73,53.48),(-130.69,53.54),(-130.62,53.53),(-130.60,53.47),
  (-130.65,53.43),(-130.71,53.44)],
 [(-130.15,53.18),(-130.12,53.24),(-130.05,53.25),(-130.02,53.19),
  (-130.07,53.14),(-130.13,53.14)],
 [(-129.84,53.16),(-129.80,53.23),(-129.73,53.25),(-129.68,53.20),
  (-129.71,53.13),(-129.78,53.10)],
]

# Narrow marine inlets cut into the broad mainland polygon.  These are applied
# after its polygon test and before islands are added.
INLETS = [
 [(-129.00,54.91),(-129.18,54.88),(-129.37,54.82),(-129.55,54.75),
  (-129.74,54.69),(-129.88,54.66),(-129.77,54.73),(-129.61,54.81),
  (-129.43,54.91),(-129.31,55.00),(-129.00,55.00)],
 [(-129.00,54.47),(-129.18,54.45),(-129.35,54.40),(-129.54,54.34),
  (-129.64,54.29),(-129.55,54.38),(-129.38,54.48),(-129.20,54.54),
  (-129.00,54.57)],
 [(-129.00,54.04),(-129.18,54.06),(-129.34,54.10),(-129.48,54.15),
  (-129.34,54.14),(-129.16,54.12),(-129.00,54.14)],
 [(-129.00,53.66),(-129.08,53.69),(-129.16,53.74),(-129.10,53.65),
  (-129.00,53.57)],
]
def points_in_polygon(x, y, poly):
    """Even/odd point-in-polygon test, vectorized over x and y arrays."""
    inside = np.zeros(x.shape, dtype=bool)
    xj, yj = poly[-1]
    for xi, yi in poly:
        crosses = ((yi > y) != (yj > y))
        x_at_y = (xj - xi) * (y - yi) / ((yj - yi) or 1e-300) + xi
        inside ^= crosses & (x < x_at_y)
        xj, yj = xi, yi
    return inside


def make_matrix():
    # Exact requested sampling sequence, avoiding accumulated floating error.
    lons = -131.0 + np.arange(127, dtype=float) * (2.0 / 126.0)
    lats = 55.0 - np.arange(127, dtype=float) * (2.0 / 126.0)
    x, y = np.meshgrid(lons, lats)

    mainland = np.zeros((127, 127), dtype=bool)
    for polygon in MAINLAND:
        mainland |= points_in_polygon(x, y, polygon)
    for inlet in INLETS:
        mainland &= ~points_in_polygon(x, y, inlet)

    land = mainland
    for island in ISLANDS:
        land |= points_in_polygon(x, y, island)
    return land.astype(np.uint8)


def write_outputs(matrix):
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)

    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

    raw = Image.fromarray(matrix * 255, mode="L")
    raw.save(out / "candidate.png")
    nearest = getattr(Image, "Resampling", Image).NEAREST
    large = raw.resize((1016, 1016), resample=nearest)
    large.save(out / "candidate_8x.png")

    # Self-check the requested invariants without reading any geographic data.
    assert matrix.shape == (127, 127)
    assert matrix.size == 16129
    assert set(np.unique(matrix).tolist()) <= {0, 1}
    assert raw.size == (127, 127) and large.size == (1016, 1016)
    assert set(raw.getdata()) <= {0, 255}
    assert set(large.getdata()) <= {0, 255}
    # Every 8x8 block must exactly equal its source pixel (nearest neighbour).
    check = np.asarray(large).reshape(127, 8, 127, 8)
    assert np.array_equal(check[:, 0, :, 0], np.asarray(raw))
    assert np.all(check == check[:, :1, :, :1])


if __name__ == "__main__":
    write_outputs(make_matrix())
