"""Knowledge-only reconstruction of the Canadian Arctic Archipelago.

Coordinates below are hand-written approximations in (degrees west, latitude).
No map, GIS package, coastline file, web service, or external input is used.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

ROWS, COLS = 61, 241
LATS = np.round(np.linspace(79.0, 73.0, ROWS), 1)
WEST = np.round(np.linspace(105.0, 81.0, COLS), 1)

def inside_poly(x, y, p):
    """Boundary-inclusive ray casting for one hand-written polygon."""
    hit = False
    n = len(p)
    for i in range(n):
        x1, y1 = p[i]
        x2, y2 = p[(i + 1) % n]
        cross = (x-x1)*(y2-y1) - (y-y1)*(x2-x1)
        if abs(cross) < 1e-10 and min(x1,x2)-1e-10 <= x <= max(x1,x2)+1e-10 \
                and min(y1,y2)-1e-10 <= y <= max(y1,y2)+1e-10:
            return True
        if (y1 > y) != (y2 > y):
            xi = x1 + (y-y1)*(x2-x1)/(y2-y1)
            if x <= xi:
                hit = not hit
    return hit

def poly(points):
    return ("poly", points)

def ellipse(cx, cy, rx, ry):
    return ("ellipse", (cx, cy, rx, ry))

def in_shape(x, y, shape):
    kind, a = shape
    if kind == "poly":
        return inside_poly(x, y, a)
    cx, cy, rx, ry = a
    return ((x-cx)/rx)**2 + ((y-cy)/ry)**2 <= 1.0 + 1e-12

# Operations are painted in order.  Large islands come first; hand-estimated
# bays/fjords can cut them back to water, and offshore islets are added last.
ops = []
def land(*shapes):
    ops.extend((1, s) for s in shapes)
def water(*shapes):
    ops.extend((0, s) for s in shapes)

# Ellef Ringnes Island (only the part east of 105 W is in the window).
land(poly([(105,77.35),(104.3,77.25),(103.3,77.28),(102.6,77.45),
 (101.4,77.55),(100.5,77.85),(100.4,78.2),(100.8,78.55),
 (101.4,78.85),(102.3,79),(104.5,79),(105,78.75)]))
water(poly([(104.8,78.62),(103.9,78.48),(103.25,78.5),(103.55,78.7),
 (104.7,78.82)]))

# Amund Ringnes Island.
land(poly([(99.55,78.05),(98.95,77.9),(97.95,77.86),(97.1,78.02),
 (96.55,78.35),(96.55,78.7),(97.05,78.98),(98.65,79),
 (99.25,78.75),(99.7,78.38)]))
water(poly([(99.5,78.28),(98.9,78.2),(98.4,78.28),(98.85,78.42),
 (99.55,78.45)]))

# Axel Heiberg Island, southwest and central portion.
land(poly([(95.8,79),(95.4,78.78),(94.65,78.58),(94.05,78.38),
 (93.35,78.25),(92.65,78.16),(91.65,78.15),(90.75,78.3),
 (89.7,78.45),(88.75,78.72),(88.25,79)]))
water(poly([(95.4,78.74),(94.4,78.65),(93.4,78.72),(92.8,78.88),
 (92.65,79),(94.95,79)]))
water(poly([(91.8,78.16),(91.0,78.34),(90.4,78.55),(91.25,78.48),
 (92.2,78.3)]))
# Southern Ellesmere Island.  Eureka and Norwegian Bay form its western edge;
# Jones Sound forms its southern edge.
land(poly([(88.15,79),(87.65,78.78),(87.0,78.63),(86.55,78.38),
 (86.0,78.25),(85.55,77.98),(84.75,77.75),(84.1,77.48),
 (83.1,77.25),(82.0,77.02),(81,76.95),(81,79)]))
# Several long Ellesmere fjords, represented as narrow water wedges.
water(poly([(81,77.32),(82.0,77.33),(83.0,77.47),(84.0,77.62),
 (83.25,77.34),(82.0,77.16),(81,77.12)]))
water(poly([(81,78.05),(82.0,77.98),(83.2,78.08),(84.15,78.3),
 (83.0,78.2),(82.0,78.25),(81,78.34)]))
water(poly([(85.85,78.3),(85.15,78.48),(84.2,78.57),(83.45,78.75),
 (84.45,78.66),(85.45,78.52),(86.25,78.48)]))

# Devon Island.  The stepped south coast is the north shore of Lancaster
# Sound; Wellington Channel is on the west.
land(poly([(96.15,75.78),(95.95,76.2),(95.45,76.52),(94.6,76.72),
 (93.4,76.82),(91.8,76.86),(90.2,76.85),(88.7,76.73),
 (87.5,76.58),(86.0,76.52),(84.7,76.55),(83.2,76.72),
 (81.7,76.82),(81,76.82),(81,74.62),(82.4,74.64),
 (83.7,74.68),(85.15,74.66),(86.55,74.62),(87.75,74.55),
 (88.9,74.5),(90.1,74.48),(91.15,74.53),(92.0,74.58),
 (92.8,74.72),(93.45,74.9),(94.15,75.05),(94.85,75.25),
 (95.45,75.48)]))
# Grinnell Peninsula notch and large south-opening Devon bays.
water(poly([(95.95,76.18),(95.25,76.2),(94.55,76.37),(93.85,76.58),
 (94.65,76.62),(95.55,76.47),(96.05,76.35)]))
water(poly([(93.25,74.72),(92.85,74.92),(92.62,75.22),(92.8,75.55),
 (93.15,75.8),(93.38,75.45),(93.25,75.08),(93.65,74.8)]))
water(poly([(90.15,74.48),(89.85,74.75),(89.65,75.08),(89.7,75.42),
 (90.05,75.7),(90.28,75.35),(90.2,75.02),(90.55,74.58)]))
water(poly([(86.2,74.62),(85.85,74.9),(85.72,75.22),(85.95,75.52),
 (86.25,75.72),(86.22,75.35),(86.45,75.02),(86.75,74.64)]))
water(poly([(83.8,74.68),(83.45,74.95),(83.35,75.25),(83.5,75.52),
 (83.75,75.3),(83.78,75.02),(84.25,74.7)]))

# Small islands in Norwegian Bay and Jones Sound.
land(ellipse(95.15,77.62,.62,.35),       # Cornwall Island
     ellipse(91.15,77.42,.48,.22),       # Graham Island
     ellipse(91.0,77.0,.38,.18),         # Buckingham Island
     ellipse(87.65,76.92,.34,.16),
     ellipse(84.45,76.86,.30,.13))
# Bathurst Island.  It is deliberately made from lobes because Erskine,
# May and Young inlets almost divide the real island.
land(poly([(101.25,75.55),(100.7,75.38),(99.85,75.38),(99.1,75.52),
 (98.55,75.78),(98.25,76.15),(97.75,76.42),(97.45,76.72),
 (97.7,76.98),(98.45,77.08),(99.35,77.02),(100.0,76.82),
 (100.55,76.62),(101.0,76.28),(101.35,75.92)]))
# Bathurst's west-opening May Inlet and east-opening Erskine Inlet.
water(poly([(101.3,76.02),(100.75,76.08),(100.15,76.25),(99.75,76.47),
 (100.35,76.4),(100.85,76.27),(101.28,76.25)]))
water(poly([(97.55,76.67),(98.15,76.62),(98.8,76.7),(99.25,76.88),
 (98.65,76.82),(98.05,76.78),(97.5,76.82)]))
water(poly([(99.0,75.55),(99.25,75.83),(99.65,76.05),(99.82,76.35),
 (99.45,76.15),(99.05,75.98),(98.7,75.7)]))

# Western Bathurst satellites, separated by the narrow channels retained
# between these polygons.
land(poly([(105,76.18),(104.55,76.05),(103.9,76.08),(103.45,76.3),
 (103.45,76.65),(103.75,76.92),(104.45,77.02),(105,76.95)]), # Cameron
 poly([(104.55,75.62),(103.85,75.55),(103.1,75.68),(102.7,75.98),
 (102.8,76.25),(103.45,76.43),(104.1,76.32),(104.55,76.05)]), # Vanier
 poly([(103.05,75.45),(102.45,75.38),(101.75,75.55),(101.48,75.82),
 (101.6,76.08),(102.2,76.22),(102.8,76.05),(103.15,75.78)]), # Massey
 poly([(102.35,75.2),(101.75,75.08),(101.15,75.18),(100.95,75.48),
 (101.25,75.7),(101.9,75.65),(102.4,75.48)]))                # Alexander

# Melville Island's Sabine Peninsula reaches into the far western edge.
land(poly([(105,74.25),(104.55,74.35),(104.25,74.62),(104.3,74.95),
 (104.65,75.15),(105,75.2)]),
 poly([(105,75.55),(104.78,75.7),(104.72,75.98),(105,76.18)]))

# Byam Martin Island in Byam Martin Channel.
land(poly([(104.9,74.82),(104.45,74.72),(103.95,74.78),(103.55,74.98),
 (103.55,75.25),(103.9,75.45),(104.45,75.48),(104.85,75.3)]))

# Cornwallis Island and its western neighbour Little Cornwallis.
land(poly([(96.15,74.72),(95.72,74.55),(95.0,74.5),(94.25,74.62),
 (93.72,74.88),(93.48,75.18),(93.55,75.5),(94.0,75.72),
 (94.72,75.86),(95.35,75.72),(95.82,75.5),(96.12,75.15)]),
 poly([(97.4,75.22),(97.05,75.08),(96.55,75.12),(96.25,75.35),
 (96.32,75.62),(96.72,75.78),(97.18,75.68),(97.48,75.45)]))
water(poly([(95.95,74.78),(95.55,74.82),(95.2,75.02),(95.35,75.23),
 (95.8,75.28),(96.15,75.15)]))

# Baillie-Hamilton and neighbouring Wellington Channel islands.
land(poly([(94.7,75.72),(94.32,75.72),(93.92,75.9),(93.82,76.18),
 (94.05,76.42),(94.48,76.38),(94.78,76.15),(94.9,75.92)]),
     ellipse(96.15,76.25,.28,.18),
     ellipse(96.75,76.65,.22,.14))
# Northern Prince of Wales Island.  Viscount Melville Sound is above it and
# Peel Sound is retained along its eastern side.
land(poly([(103.8,73),(103.65,73.28),(103.25,73.55),(102.7,73.75),
 (102.05,73.95),(101.25,74.16),(100.5,74.3),(99.75,74.25),
 (99.25,74.08),(98.65,73.98),(98.05,73.9),(97.55,73.7),
 (97.2,73.42),(96.95,73),(96.35,73)]))
# Ommanney Bay and several north-coast indentations.
water(poly([(102.65,73),(102.55,73.24),(102.25,73.48),(101.75,73.65),
 (101.92,73.38),(102.08,73.12),(102.05,73)]))
water(poly([(100.2,74.28),(99.88,74.05),(99.48,73.92),(99.65,74.18),
 (99.85,74.3)]))
water(poly([(97.65,73.78),(97.82,73.52),(98.25,73.3),(98.42,73),
 (97.72,73),(97.42,73.42)]))

# Somerset Island, divided from Prince of Wales by Peel Sound and from the
# Baffin mainland by Prince Regent Inlet.
land(poly([(96.0,73),(95.88,73.35),(95.55,73.62),(95.05,73.82),
 (94.5,73.92),(93.92,74.08),(93.35,74.18),(92.85,74.08),
 (92.45,73.85),(92.05,73.65),(91.5,73.45),(90.82,73.25),
 (90.35,73),(89.65,73)]))
water(poly([(95.05,73),(94.9,73.28),(94.55,73.55),(94.3,73.85),
 (94.7,73.72),(95.12,73.48),(95.48,73.12),(95.55,73)]))
water(poly([(91.05,73),(91.35,73.18),(91.72,73.38),(92.05,73.65),
 (91.5,73.45),(90.82,73.25),(90.35,73)]))

# Brodeur Peninsula of Baffin Island at the southern edge.  Admiralty Inlet
# creates the east-side notch.  Only points in the requested frame are drawn.
land(poly([(89.55,73),(89.1,73.18),(88.72,73.42),(88.25,73.58),
 (87.65,73.7),(86.95,73.72),(86.25,73.6),(85.7,73.42),
 (85.15,73.25),(84.45,73.12),(83.8,73),(81,73)]))
water(poly([(85.55,73),(85.7,73.22),(85.95,73.48),(86.25,73.6),
 (86.65,73.68),(86.35,73.42),(86.18,73.15),(86.2,73)]))
water(poly([(82.8,73),(83.05,73.14),(83.45,73.25),(83.8,73.38),
 (83.35,73.35),(82.9,73.22),(82.35,73)]))

# Barrow Strait, Peel Sound and Lancaster Sound islands large enough to have
# a fair chance of meeting a 0.1-degree sample.
land(ellipse(100.1,74.48,.42,.17),       # Young Island area
     ellipse(97.65,74.55,.55,.22),       # Lowther Island
     ellipse(95.55,74.58,.47,.22),       # Griffith Island
     ellipse(96.55,74.82,.20,.13),       # Browne Island
     ellipse(92.0,74.72,.24,.12),        # Beechey Island
     ellipse(90.05,74.05,.22,.14),       # Prince Leopold Island
     ellipse(96.75,73.08,.20,.34),       # Prescott Island chain
     ellipse(97.05,73.65,.16,.25),
     ellipse(88.8,73.78,.17,.11))

# Islets omitted by the broad outlines in Sverdrup/Norwegian Bay.
land(ellipse(101.25,77.88,.28,.17),      # King Christian Island
     ellipse(102.25,77.25,.18,.11),
     ellipse(96.05,77.2,.18,.12),
     ellipse(89.7,77.75,.16,.10),
     ellipse(86.8,77.25,.13,.09))
# Evaluate every requested point against the same ordered outline rules.
matrix = np.zeros((ROWS, COLS), dtype=np.uint8)
for r, lat in enumerate(LATS):
    for c, west_lon in enumerate(WEST):
        value = 0
        for paint, shape in ops:
            if in_shape(float(west_lon), float(lat), shape):
                value = paint
        matrix[r, c] = value

out = Path("outputs")
out.mkdir(parents=True, exist_ok=True)

with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
    csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

raw = Image.fromarray(matrix * 255, mode="L")
raw.save(out / "candidate.png")
large = raw.resize((COLS * 8, ROWS * 8), resample=Image.Resampling.NEAREST)
large.save(out / "candidate_8x.png")

# Fail loudly if any contractual property has accidentally been broken.
assert matrix.shape == (61, 241)
assert matrix.size == 14701
assert set(np.unique(matrix)).issubset({0, 1})
assert raw.size == (241, 61) and large.size == (1928, 488)
assert set(raw.getdata()).issubset({0, 255})
assert set(large.getdata()).issubset({0, 255})
assert np.array_equal(np.asarray(large), np.repeat(np.repeat(
    np.asarray(raw), 8, axis=0), 8, axis=1))

with (out / "candidate.csv").open("r", encoding="ascii") as f:
    check = list(csv.reader(f))
assert len(check) == 61 and all(len(row) == 241 for row in check)
assert all(v in ("0", "1") for row in check for v in row)
print(f"wrote {matrix.shape[0]}x{matrix.shape[1]} matrix; "
      f"land={int(matrix.sum())}, water={matrix.size-int(matrix.sum())}")
