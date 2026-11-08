"""Memory-only approximation of land in 53–55 N, 131–129 W.

No files or geographic services are read.  The hand-entered polygons and
ellipses below are deliberately approximate recollections of the mainland
coast and the islands bordering Hecate Strait.
"""
from pathlib import Path
import csv, math
import numpy as np
from PIL import Image

N = 127
lons = np.linspace(-131.0, -129.0, N)
lats = np.linspace(55.0, 53.0, N)
X, Y = np.meshgrid(lons, lats)
land = np.zeros((N, N), dtype=bool)

def polygon(points):
    """Even/odd point-in-polygon test on all sample-point centres."""
    inside = np.zeros_like(land)
    xj, yj = points[-1]
    for xi, yi in points:
        cross = ((yi > Y) != (yj > Y)) & (X < (xj-xi)*(Y-yi)/(yj-yi+1e-30)+xi)
        inside ^= cross
        xj, yj = xi, yi
    return inside

def ellipse(cx, cy, rx, ry, angle=0):
    a = math.radians(angle); dx=X-cx; dy=Y-cy
    u=dx*math.cos(a)+dy*math.sin(a); v=-dx*math.sin(a)+dy*math.cos(a)
    return (u/rx)**2 + (v/ry)**2 <= 1

# Eastern mainland / large coastal land, with a memory-sketched outer shore.
mainland = [(-129.58,55.00),(-129.55,54.91),(-129.64,54.83),(-129.58,54.73),
 (-129.70,54.64),(-129.68,54.53),(-129.75,54.43),(-129.66,54.33),
 (-129.72,54.23),(-129.61,54.13),(-129.67,54.03),(-129.57,53.92),
 (-129.66,53.82),(-129.58,53.70),(-129.48,53.61),(-129.43,53.48),
 (-129.30,53.38),(-129.25,53.26),(-129.13,53.16),(-129.06,53.05),
 (-129.00,53.02),(-129.00,55.00)]
land |= polygon(mainland)

# Major outer islands: Dundas group, Stephens, Porcher, Pitt, Banks,
# McCauley/Gil/Campania, and the edge of Aristazabal.
polys = [
 [(-130.94,54.70),(-130.75,54.77),(-130.51,54.70),(-130.38,54.57),
  (-130.47,54.43),(-130.69,54.38),(-130.90,54.48)],
 [(-130.82,54.37),(-130.65,54.42),(-130.48,54.32),(-130.51,54.17),
  (-130.69,54.10),(-130.84,54.20)],
 [(-130.75,54.16),(-130.54,54.20),(-130.35,54.10),(-130.30,53.98),
  (-130.43,53.86),(-130.64,53.87),(-130.78,53.99)],
 [(-130.07,54.50),(-129.91,54.46),(-129.81,54.27),(-129.75,54.04),
  (-129.77,53.79),(-129.84,53.57),(-130.02,53.43),(-130.15,53.52),
  (-130.23,53.75),(-130.26,54.00),(-130.19,54.26)],
 [(-130.45,54.10),(-130.28,54.10),(-130.17,53.94),(-130.12,53.69),
  (-130.18,53.42),(-130.30,53.23),(-130.45,53.30),(-130.54,53.54),
  (-130.59,53.80),(-130.55,53.99)],
 [(-129.70,53.73),(-129.55,53.69),(-129.47,53.55),(-129.50,53.38),
  (-129.62,53.29),(-129.75,53.37),(-129.80,53.55)],
 [(-129.63,53.39),(-129.49,53.36),(-129.39,53.20),(-129.43,53.06),
  (-129.57,53.01),(-129.66,53.15)],
 [(-129.43,53.25),(-129.22,53.28),(-129.06,53.17),(-129.00,53.03),
  (-129.00,53.00),(-129.34,53.00),(-129.49,53.10)],
]
for p in polys: land |= polygon(p)

# Smaller named/unnamed coastal islands and offshore rocks.
islets = [
 (-130.99,54.18,.055,.025,-20),(-130.88,54.33,.065,.035,25),
 (-130.32,54.48,.075,.040,-20),(-130.16,54.61,.055,.030,35),
 (-130.43,54.29,.055,.030,10),(-130.89,54.07,.050,.025,-25),
 (-130.93,53.91,.045,.022,15),(-130.74,53.76,.040,.020,-15),
 (-130.65,53.58,.030,.018,5),(-130.68,53.39,.045,.020,-30),
 (-130.58,53.20,.040,.020,15),(-130.14,53.19,.035,.020,-20),
 (-130.06,53.05,.045,.024,15),(-129.88,53.28,.045,.020,-20),
 (-129.82,53.12,.030,.018,10),(-129.78,54.63,.050,.025,35),
 (-129.92,54.72,.045,.022,-10),(-130.21,54.75,.040,.020,20),
 (-130.58,54.81,.045,.020,-15),(-130.78,54.83,.040,.018,10),
]
for e in islets: land |= ellipse(*e)

# Open salt-water cuts remembered as principal passages/inlets.
cuts = [
 [(-129.63,54.88),(-129.45,55.00),(-129.34,55.00),(-129.56,54.79)],
 [(-129.71,54.47),(-129.48,54.58),(-129.28,54.54),(-129.66,54.38)],
 [(-129.68,54.20),(-129.42,54.27),(-129.21,54.20),(-129.60,54.10)],
 [(-129.61,53.86),(-129.37,53.91),(-129.20,53.84),(-129.58,53.76)],
 [(-129.45,53.53),(-129.25,53.60),(-129.10,53.54),(-129.40,53.43)],
]
for p in cuts: land &= ~polygon(p)

# Rose Spit / eastern Haida Gwaii and a few marginal rocks at the west edge.
land |= polygon([(-131.00,54.25),(-130.91,54.20),(-130.93,54.16),
                 (-131.00,54.17)])
land |= ellipse(-130.995,53.05,.025,.065,-8)
land |= ellipse(-130.975,53.00,.018,.025,0)

# Write and independently validate all requested products.
arr = land.astype(np.uint8)
out = Path("outputs"); out.mkdir(exist_ok=True)
with (out/"candidate.csv").open("w", newline="", encoding="ascii") as f:
    csv.writer(f, lineterminator="\n").writerows(arr.tolist())
img = Image.fromarray(arr*255, mode="L")
img.save(out/"candidate.png")
img.resize((N*8,N*8), Image.Resampling.NEAREST).save(out/"candidate_8x.png")
assert arr.shape == (127,127) and set(np.unique(arr)) <= {0,1}
with (out/"candidate.csv").open(encoding="ascii") as f:
    rows=list(csv.reader(f))
assert len(rows)==127 and all(len(r)==127 and set(r)<={"0","1"} for r in rows)
for fn, size in [("candidate.png",(127,127)),("candidate_8x.png",(1016,1016))]:
    with Image.open(out/fn) as im:
        assert im.size==size and set(im.convert("L").getdata()) <= {0,255}
with Image.open(out/"candidate.png") as a, Image.open(out/"candidate_8x.png") as b:
    assert np.array_equal(np.array(b), np.repeat(np.repeat(np.array(a),8,0),8,1))
print(f"wrote 127x127 products; land={int(arr.sum())}, water={int((1-arr).sum())}")
