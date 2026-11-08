"""Hand-drawn approximate land mask for northern Hecate Strait.

Coordinates below are invented geographic approximations, not external data.
Polygon coordinates are (degrees west, degrees north); smaller west is east.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
yy, xx = np.meshgrid(55 - np.arange(127) * 2 / 126,
                     131 - np.arange(127) * 2 / 126, indexing="ij")


def polygon(points):
    """Even/odd point-in-polygon classification at sample coordinates."""
    inside = np.zeros(xx.shape, dtype=bool)
    for (a, b), (c, d) in zip(points, points[1:] + points[:1]):
        if b == d:
            continue
        crossing = (b > yy) != (d > yy)
        x_at_y = a + (yy - b) * (c - a) / (d - b)
        inside ^= crossing & (xx < x_at_y)
    return inside


land = np.zeros((127, 127), dtype=bool)
# Coast of the mainland, broadly following the outer edge of Chatham Sound
# and then the mainland side of the island channels farther south.
coast = [
 (130.12,55),(130.13,54.96),(130.06,54.92),(130.11,54.87),
 (130.07,54.84),(130.14,54.79),(130.08,54.75),(130.19,54.71),
 (130.20,54.67),(130.12,54.63),(130.25,54.60),(130.23,54.55),
 (130.32,54.51),(130.25,54.47),(130.39,54.43),(130.38,54.39),
 (130.31,54.36),(130.37,54.33),(130.31,54.29),(130.35,54.26),
 (130.29,54.23),(130.30,54.18),(130.22,54.14),(130.20,54.10),
 (130.13,54.06),(130.14,54.02),(130.10,53.97),(130.04,53.93),
 (130.07,53.88),(129.99,53.83),(129.97,53.78),(129.90,53.75),
 (129.93,53.69),(129.85,53.65),(129.79,53.59),(129.77,53.54),
 (129.70,53.49),(129.70,53.44),(129.64,53.39),(129.59,53.33),
 (129.55,53.27),(129.49,53.22),(129.45,53.16),(129.40,53.09),
 (129.34,53.04),(129.31,53),(129,53),(129,55)]
land |= polygon(coast)

# Long, narrow saltwater inlets interrupting the mainland shoreline.
water_cuts = [
 [(130.13,54.78),(130.04,54.80),(129.98,54.84),(129.95,54.90),
  (129.88,54.93),(129.76,54.98),(129.82,55),(129.95,55),
  (130.06,54.94),(130.16,54.86)],
 [(130.40,54.40),(130.25,54.39),(130.13,54.41),(130.05,54.44),
  (129.99,54.48),(129.92,54.49),(129.95,54.45),(130.02,54.38),
  (130.20,54.32),(130.37,54.34)],
 [(130.34,54.20),(130.20,54.20),(130.09,54.22),(129.99,54.26),
  (129.89,54.30),(129.79,54.32),(129.81,54.26),(129.95,54.18),
  (130.11,54.14),(130.26,54.15)],
 [(130.13,54.06),(129.99,54.05),(129.93,54.07),(129.85,54.10),
  (129.77,54.10),(129.85,54.04),(129.94,53.99),(130.11,53.99)]
]
for shape in water_cuts:
    land &= ~polygon(shape)

# Major islands of Chatham Sound, and Banks / Pitt / Porcher archipelago.
islands = [
 # Dundas and neighboring northern islands
 [(130.91,54.62),(130.87,54.69),(130.77,54.73),(130.71,54.68),
  (130.70,54.61),(130.76,54.54),(130.75,54.48),(130.82,54.46),
  (130.89,54.50),(130.94,54.56)],
 [(130.73,54.58),(130.69,54.61),(130.62,54.59),(130.59,54.55),
  (130.62,54.50),(130.69,54.49),(130.75,54.52)],
 [(130.61,54.49),(130.56,54.53),(130.49,54.50),(130.48,54.45),
  (130.54,54.43),(130.61,54.45)],
 # Digby and Kaien islands, around Prince Rupert
 [(130.59,54.38),(130.54,54.40),(130.47,54.39),(130.43,54.35),
  (130.45,54.30),(130.51,54.28),(130.57,54.33)],
 [(130.43,54.36),(130.39,54.38),(130.33,54.36),(130.30,54.30),
  (130.35,54.26),(130.42,54.29)],
 # Stephens Island and the more southerly islands of Chatham Sound
 [(130.72,54.25),(130.67,54.27),(130.61,54.24),(130.58,54.18),
  (130.56,54.12),(130.60,54.07),(130.66,54.10),(130.71,54.16)],
 [(130.53,54.19),(130.48,54.22),(130.45,54.19),(130.46,54.14),
  (130.51,54.12)],
 # Porcher Island, with deeply indented north and east shores
 [(130.64,54.04),(130.60,54.10),(130.51,54.10),(130.46,54.08),
  (130.41,54.10),(130.34,54.07),(130.27,54.06),(130.24,54.00),
  (130.20,53.96),(130.22,53.90),(130.27,53.85),(130.35,53.82),
  (130.40,53.77),(130.48,53.76),(130.54,53.81),(130.58,53.88),
  (130.65,53.93)],
 # Banks Island: a long, rocky island west of Principe Channel
 [(130.42,53.79),(130.37,53.79),(130.32,53.75),(130.30,53.68),
  (130.28,53.62),(130.31,53.55),(130.28,53.50),(130.25,53.45),
  (130.28,53.37),(130.23,53.30),(130.22,53.22),(130.26,53.16),
  (130.33,53.20),(130.37,53.29),(130.41,53.39),(130.45,53.48),
  (130.46,53.58),(130.49,53.66),(130.47,53.73)],
 # Pitt Island, separated from the mainland by Grenville Channel
 [(129.96,53.90),(129.91,53.88),(129.88,53.81),(129.82,53.76),
  (129.79,53.69),(129.77,53.62),(129.72,53.55),(129.67,53.49),
  (129.65,53.41),(129.62,53.33),(129.60,53.27),(129.58,53.19),
  (129.60,53.13),(129.66,53.16),(129.71,53.24),(129.75,53.30),
  (129.79,53.37),(129.84,53.44),(129.88,53.53),(129.91,53.63),
  (129.95,53.73)],
 # Small islands near the southern edge of the frame
 [(130.22,53.13),(130.18,53.17),(130.12,53.15),(130.09,53.09),
  (130.11,53.02),(130.18,53.00),(130.23,53.05)],
 [(129.95,53.10),(129.91,53.16),(129.86,53.15),(129.83,53.08),
  (129.87,53.02),(129.93,53.00)]
]
for shape in islands:
    land |= polygon(shape)

# A handful of islets and reefs, encoded as tiny elliptical land patches.
islets = [
 (130.97,54.72,.025,.025),(130.95,54.64,.024,.035),
 (130.67,54.73,.025,.030),(130.59,54.69,.023,.025),
 (130.53,54.64,.032,.029),(130.45,54.59,.026,.039),
 (130.79,54.38,.038,.044),(130.72,54.35,.025,.029),
 (130.64,54.36,.025,.030),(130.83,54.23,.028,.034),
 (130.81,54.12,.023,.027),(130.44,54.22,.023,.027),
 (130.75,53.99,.028,.031),(130.72,53.86,.023,.032),
 (130.65,53.75,.024,.035),(130.55,53.69,.025,.033),
 (130.57,53.56,.026,.034),(130.50,53.38,.023,.030),
 (130.38,53.09,.025,.038),(130.08,53.72,.037,.053),
 (130.04,53.60,.026,.042),(130.03,53.46,.024,.030),
 (129.98,53.32,.023,.034),(129.96,53.20,.020,.027)
]
for cx, cy, rx, ry in islets:
    land |= ((xx - cx) / rx)**2 + ((yy - cy) / ry)**2 <= 1

mask = land.astype(np.uint8)
assert mask.shape == (127, 127) and np.isin(mask, (0, 1)).all()
with (OUT / "candidate.csv").open("w", newline="", encoding="ascii") as f:
    csv.writer(f).writerows(mask.tolist())
original = Image.fromarray(mask * 255, mode="L")
original.save(OUT / "candidate.png")
enlarged = original.resize((1016, 1016), Image.Resampling.NEAREST)
enlarged.save(OUT / "candidate_8x.png")
with (OUT / "candidate.csv").open(newline="", encoding="ascii") as f:
    rows = list(csv.reader(f))
assert len(rows) == 127 and all(len(row) == 127 for row in rows)
assert all(v in ("0", "1") for row in rows for v in row)
assert set(original.getdata()) <= {0, 255}
assert set(enlarged.getdata()) <= {0, 255}
assert np.array_equal(np.asarray(enlarged)[::8, ::8], np.asarray(original))
assert np.array_equal(np.asarray(enlarged),
                      np.repeat(np.repeat(np.asarray(original), 8, 0), 8, 1))
print("Validated: 127x127, 16129 values, black/white PNG, 1016x1016 nearest neighbor.")
