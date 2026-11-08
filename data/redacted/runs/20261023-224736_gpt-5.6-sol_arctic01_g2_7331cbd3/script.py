"""Knowledge-only sketch of the Canadian Arctic Archipelago, 73--79 N.

Coordinates below are hand-entered approximate coastlines (degrees north and
west).  They are not read from, or compared with, any external geographic
dataset.  Point-in-polygon and analytic ellipse rules create the raster.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

LATS = np.round(np.arange(79.0, 72.99, -0.1), 1)
WEST = np.round(np.arange(105.0, 80.99, -0.1), 1)

def inside_poly(w, lat, p):
    """Ray crossing test; p is a hand-written sequence of (west, north)."""
    hit = False
    x0, y0 = p[-1]
    for x1, y1 in p:
        if ((y1 > lat) != (y0 > lat)):
            cross = (x0-x1) * (lat-y1) / (y0-y1) + x1
            if w < cross:
                hit = not hit
        x0, y0 = x1, y1
    return hit

def ellipse(w, lat, cw, cy, rw, ry, angle=0.0):
    a = np.deg2rad(angle)
    x, y = (w-cw)*np.cos(np.deg2rad(cy)), lat-cy
    ca, sa = np.cos(a), np.sin(a)
    u, v = x*ca+y*sa, -x*sa+y*ca
    return (u/(rw*np.cos(np.deg2rad(cy))))**2 + (v/ry)**2 <= 1

# Larger islands are intentionally angular: vertices are remembered capes,
# bays, and peninsula-scale bends rather than a fitted real coastline.
POLYS = []

# Ellef Ringnes Island (upper-left, cropped by the north and west limits).
POLYS.append([
 (105.0,78.18),(104.45,78.03),(103.75,77.91),(102.92,77.86),
 (102.20,77.93),(101.62,78.10),(100.82,78.20),(100.25,78.43),
 (99.72,78.57),(99.52,78.85),(99.72,79.00),(105.0,79.00)
])
# Southern and eastern peninsulas give this island its broken outline.
POLYS += [
 [(104.25,78.14),(104.55,77.90),(104.13,77.73),(103.45,77.68),
  (102.95,77.82),(103.20,78.05)],
 [(101.42,78.17),(100.92,78.00),(100.26,77.98),(99.92,78.17),
  (100.28,78.40)]
]

# Amund Ringnes Island.
POLYS.append([
 (99.10,78.36),(98.75,78.06),(98.05,77.86),(97.25,77.78),
 (96.45,77.82),(95.65,77.98),(94.85,78.06),(94.25,78.30),
 (94.38,78.55),(94.93,78.76),(95.72,78.87),(96.55,78.84),
 (97.30,78.72),(98.12,78.68),(98.78,78.54)
])
# Axel Heiberg, whose much larger northern body continues out of frame.
POLYS.append([
 (96.10,79.00),(95.85,78.83),(95.22,78.69),(94.62,78.46),
 (94.16,78.18),(93.66,78.02),(93.10,78.04),(92.74,78.20),
 (92.21,78.31),(91.60,78.34),(91.04,78.47),(90.60,78.71),
 (90.13,78.84),(89.82,79.00)
])
# A low southern spur of Axel Heiberg.
POLYS.append([
 (94.28,78.31),(94.05,78.03),(93.58,77.82),(92.94,77.73),
 (92.55,77.83),(92.79,78.08),(93.44,78.29)
])
# Ellesmere Island: only its fjord-cut southwest corner lies in the frame.
POLYS.append([
 (89.82,79.00),(90.05,78.80),(90.64,78.66),(91.30,78.53),
 (91.58,78.34),(91.18,78.15),(90.58,78.04),(90.26,77.78),
 (89.82,77.57),(89.13,77.43),(88.86,77.20),(88.31,77.03),
 (87.58,76.90),(86.88,76.75),(85.92,76.63),(84.86,76.55),
 (83.72,76.52),(82.55,76.43),(81.00,76.42),(81.00,79.00)
])
# Western fingers separated by broad remembered fjord mouths.
POLYS += [
 [(91.40,78.50),(92.06,78.41),(92.61,78.48),(92.93,78.68),
  (92.60,78.84),(91.94,78.88),(91.15,78.75)],
 [(89.93,77.83),(90.52,77.70),(90.83,77.51),(90.54,77.32),
  (89.87,77.30),(89.36,77.47)]
]

# Devon Island, the broad land mass south of Jones Sound.
POLYS.append([
 (96.10,76.66),(95.40,76.60),(94.78,76.43),(94.18,76.19),
 (93.80,75.93),(93.65,75.61),(93.86,75.32),(93.92,74.98),
 (93.63,74.72),(92.86,74.61),(91.97,74.67),(91.13,74.55),
 (90.18,74.49),(89.42,74.52),(88.74,74.43),(87.83,74.39),
 (86.88,74.43),(86.06,74.51),(85.22,74.55),(84.43,74.50),
 (83.62,74.61),(82.72,74.67),(81.82,74.72),(81.00,74.70),
 (81.00,76.24),(82.02,76.30),(82.86,76.42),(83.80,76.43),
 (84.72,76.38),(85.52,76.28),(86.22,76.31),(86.91,76.40),
 (87.62,76.51),(88.38,76.55),(89.08,76.47),(89.74,76.35),
 (90.42,76.29),(91.14,76.31),(91.83,76.45),(92.43,76.58),
 (93.12,76.72),(93.90,76.80),(94.73,76.82),(95.48,76.76)
])
# Grinnell Peninsula and several coastal lobes on western Devon.
POLYS += [
 [(95.82,76.64),(96.43,76.72),(96.88,76.64),(96.93,76.42),
  (96.47,76.22),(95.73,76.20),(95.30,76.40)],
 [(94.12,75.42),(94.65,75.35),(94.88,75.08),(94.54,74.83),
  (93.82,74.79),(93.51,75.04)],
 [(91.35,74.62),(91.78,74.48),(91.63,74.34),(91.05,74.34),
  (90.76,74.49)]
]

# Marine fjords cut from the two big eastern islands later.
CUTOUTS = []
# Baumann Fiord and Bay Fiord, Ellesmere's western embayments.
CUTOUTS += [
 [(90.94,78.43),(90.08,78.44),(89.24,78.52),(88.54,78.67),
  (87.77,78.74),(87.50,78.64),(88.12,78.47),(89.02,78.34),
  (89.92,78.23),(90.66,78.24)],
 [(90.20,77.60),(89.42,77.69),(88.63,77.84),(87.78,78.02),
  (87.36,78.00),(87.76,77.80),(88.42,77.64),(89.18,77.48),
  (89.85,77.42)]
]
# Devon's long north-coast fjords and shorter Lancaster Sound bays.
CUTOUTS += [
 [(91.10,76.39),(90.58,76.23),(90.02,76.05),(89.43,75.88),
  (89.08,75.90),(89.35,76.13),(89.87,76.35),(90.47,76.49)],
 [(86.57,76.39),(86.05,76.23),(85.72,75.99),(85.36,75.80),
  (85.10,75.87),(85.25,76.12),(85.70,76.35),(86.14,76.48)],
 [(83.87,74.53),(84.13,74.72),(84.42,74.91),(84.27,75.10),
  (83.98,75.05),(83.78,74.79),(83.51,74.58)]
]
# Bathurst Island.
POLYS.append([
 (102.90,76.55),(102.31,76.68),(101.57,76.76),(100.83,76.70),
 (100.17,76.76),(99.54,76.65),(98.86,76.66),(98.20,76.50),
 (97.63,76.35),(97.25,76.10),(97.10,75.84),(97.30,75.61),
 (97.78,75.43),(98.38,75.37),(99.03,75.43),(99.62,75.35),
 (100.38,75.39),(100.94,75.49),(101.55,75.51),(102.05,75.68),
 (102.52,75.87),(102.65,76.13),(103.05,76.36)
])
# Bathurst's long peninsulas across its major inlets.
POLYS += [
 [(102.65,76.31),(103.28,76.23),(103.61,76.06),(103.31,75.88),
  (102.69,75.86),(102.32,76.01)],
 [(101.57,76.74),(101.31,76.99),(100.78,77.07),(100.28,76.96),
  (100.35,76.69)],
 [(99.20,76.67),(98.89,76.87),(98.35,76.91),(97.93,76.71),
  (98.23,76.48)]
]
# Deep seawater inlets of Bathurst.
CUTOUTS += [
 [(102.91,76.29),(102.13,76.32),(101.34,76.27),(100.58,76.17),
  (99.86,76.06),(99.47,75.93),(99.81,75.87),(100.50,75.96),
  (101.23,76.05),(102.10,76.10),(102.78,76.08)],
 [(100.18,76.85),(100.02,76.60),(99.91,76.36),(99.69,76.11),
  (99.50,76.13),(99.55,76.42),(99.64,76.70),(99.79,76.88)]
]

# Cornwallis Island.
POLYS.append([
 (96.66,75.88),(96.05,75.96),(95.33,75.92),(94.68,75.82),
 (94.03,75.64),(93.54,75.42),(93.32,75.17),(93.38,74.91),
 (93.78,74.70),(94.42,74.60),(95.09,74.61),(95.66,74.72),
 (96.14,74.91),(96.53,75.15),(96.72,75.43)
])
# Resolute Bay and a northern indentation.
CUTOUTS += [
 [(95.18,74.64),(95.02,74.82),(94.92,75.00),(94.72,75.07),
  (94.62,74.91),(94.70,74.70)],
 [(95.88,75.94),(95.70,75.75),(95.43,75.58),(95.18,75.61),
  (95.30,75.82)]
]

# The chain west of Bathurst: Byam Martin, Alexander, Massey, Vanier,
# Cameron, and Helena.  The ellipses are deliberately separate water-gapped
# islands, and their dimensions are in geographic degrees.
ELLIPSES = [
 (104.45,75.18,0.72,0.40,-12), # Byam Martin
 (102.63,75.91,0.48,0.24, 10), # Alexander
 (102.31,76.05,0.52,0.22,-12), # Massey
 (103.42,76.21,0.52,0.27,  5), # Vanier
 (104.22,76.50,0.72,0.32,-10), # Cameron
 (101.52,76.72,0.59,0.23,-18), # Helena
]
# Little Cornwallis Island and neighbouring small lands.
ELLIPSES += [
 (96.78,75.52,0.48,0.28,-10),
 (97.28,75.74,0.24,0.15,  5),
 (96.45,76.02,0.25,0.13,-10),
]
# Prince of Wales Island, continuing south of the frame.
POLYS.append([
 (103.34,73.00),(103.29,73.37),(103.05,73.72),(102.72,74.00),
 (102.16,74.18),(101.48,74.29),(100.92,74.24),(100.58,74.08),
 (100.43,73.89),(99.99,73.80),(99.48,73.88),(99.05,74.08),
 (98.58,74.16),(98.19,74.04),(97.92,73.83),(97.63,73.57),
 (97.55,73.25),(97.65,73.00)
])
# The north shore is divided by Ommanney Bay.
CUTOUTS.append([
 (100.69,74.21),(100.47,73.99),(100.10,73.77),(99.73,73.56),
 (99.45,73.35),(99.22,73.37),(99.33,73.62),(99.58,73.88),
 (100.04,74.12),(100.42,74.28)
])

# Somerset Island, bounded by Peel Sound and Prince Regent Inlet.
POLYS.append([
 (96.12,73.00),(96.25,73.31),(96.22,73.61),(95.96,73.83),
 (95.49,74.00),(94.88,74.11),(94.25,74.08),(93.68,74.00),
 (93.16,74.08),(92.58,74.06),(92.06,73.92),(91.48,73.82),
 (90.95,73.64),(90.52,73.40),(90.20,73.09),(90.15,73.00)
])
# Cunningham Inlet in north Somerset.
CUTOUTS.append([
 (93.73,74.05),(93.68,73.82),(93.48,73.60),(93.16,73.43),
 (92.95,73.47),(93.04,73.69),(93.23,73.91),(93.43,74.08)
])

# Russell Island off the northeast coast of Prince of Wales.
POLYS.append([
 (99.42,74.28),(99.02,74.35),(98.55,74.31),(98.16,74.17),
 (97.93,73.99),(98.04,73.82),(98.40,73.76),(98.79,73.83),
 (99.14,74.01)
])

# Northwest Baffin Island is split by Admiralty Inlet into Brodeur and
# Borden peninsulas; both continue south/east beyond this small frame.
POLYS.append([
 (89.02,73.00),(89.01,73.30),(88.89,73.55),(88.54,73.72),
 (88.02,73.79),(87.40,73.82),(86.75,73.78),(86.15,73.72),
 (85.58,73.66),(85.08,73.49),(84.81,73.23),(84.73,73.00)
])
POLYS.append([
 (82.84,73.00),(82.75,73.27),(82.50,73.49),(82.12,73.68),
 (81.68,73.78),(81.00,73.82),(81.00,73.00)
])

# Medium and small islands in Peel Sound and the Barrow/Lancaster corridor.
ELLIPSES += [
 (96.72,73.18,0.47,0.38,-5),  # Prescott
 (97.55,74.54,0.52,0.19,-8),  # Lowther
 (95.55,74.55,0.48,0.17, 4),  # Griffith
 (96.55,74.76,0.24,0.13, 0),  # Browne
 (91.88,74.72,0.29,0.12,-8),  # Beechey
 (90.02,74.05,0.20,0.11, 8),  # Prince Leopold
 (96.74,73.63,0.18,0.16, 0),
 (96.72,72.99,0.22,0.12, 0),
]
# Islands scattered through Norwegian Bay and the northern channels.
ELLIPSES += [
 (104.83,77.43,0.72,0.35,-10), # cropped Lougheed Island
 (102.02,77.76,0.78,0.27, -4), # King Christian Island
 (94.82,77.62,1.62,0.31, -4),  # Cornwall Island
 (90.48,77.43,1.10,0.27, -7),  # Graham Island
 (91.10,77.15,0.42,0.20,  8),  # Buckingham Island
 (90.18,76.70,0.75,0.27,-12),  # North Kent Island
 (94.58,75.87,0.64,0.26, -5),  # Baillie-Hamilton Island
 (100.02,76.86,0.20,0.12, 0),
 (98.65,77.30,0.19,0.11,  0),
 (92.35,77.00,0.21,0.12,  0),
 (97.45,75.23,0.24,0.12,  0),
 (97.10,76.25,0.17,0.10,  0),
 (103.92,74.70,0.20,0.12, 0),
]

def classify(w, lat):
    # First union the hand-drawn island bodies.
    land = any(inside_poly(w, lat, p) for p in POLYS)
    # Named marine inlets override those coarse bodies.
    if land and any(inside_poly(w, lat, p) for p in CUTOUTS):
        land = False
    # Separate islands, including any lying at a bay mouth, are then restored.
    if any(ellipse(w, lat, *e) for e in ELLIPSES):
        land = True
    return 1 if land else 0

matrix = np.empty((len(LATS), len(WEST)), dtype=np.uint8)
for r, lat in enumerate(LATS):
    for c, w in enumerate(WEST):
        matrix[r, c] = classify(float(w), float(lat))

out = Path("outputs")
out.mkdir(parents=True, exist_ok=True)
with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
    csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

raw = Image.fromarray(matrix * 255, mode="L")
raw.save(out / "candidate.png")
large = raw.resize((241*8, 61*8), resample=Image.Resampling.NEAREST)
large.save(out / "candidate_8x.png")

# Fail loudly if an edit ever violates any deliverable invariant.
assert matrix.shape == (61, 241)
assert matrix.size == 14701
assert set(np.unique(matrix).tolist()) <= {0, 1}
assert raw.size == (241, 61) and large.size == (1928, 488)
assert set(raw.getdata()) <= {0, 255}
assert np.array_equal(np.asarray(large), np.repeat(np.repeat(
    np.asarray(raw), 8, axis=0), 8, axis=1))

print(f"wrote {matrix.shape[0]}x{matrix.shape[1]} matrix; "
      f"land={int(matrix.sum())}, water={matrix.size-int(matrix.sum())}")
