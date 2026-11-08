from pathlib import Path
import csv
import numpy as np
from PIL import Image

# Hand-drawn reconstruction only: longitude is represented as degrees west.
# No external files, maps, network services, or geographic packages are used.
W = np.round(np.arange(105.0, 80.9, -0.1), 1)
LAT = np.round(np.arange(79.0, 72.9, -0.1), 1)
X, Y = np.meshgrid(W, LAT)
land = np.zeros(X.shape, dtype=np.uint8)

def polygon(vertices, value=1):
    """Even/odd point-in-polygon test, applied to every requested sample point."""
    inside = np.zeros(X.shape, dtype=bool)
    xj, yj = vertices[-1]
    for xi, yi in vertices:
        cross = ((yi > Y) != (yj > Y)) & (X <= (xj-xi)*(Y-yi)/(yj-yi+1e-30)+xi)
        inside ^= cross
        xj, yj = xi, yi
    land[inside] = value

def ellipse(cx, cy, rx, ry, angle=0, value=1):
    a = np.deg2rad(angle)
    dx, dy = X-cx, Y-cy
    u, v = dx*np.cos(a)+dy*np.sin(a), -dx*np.sin(a)+dy*np.cos(a)
    land[(u/rx)**2 + (v/ry)**2 <= 1] = value

# Ellef Ringnes eastern margin and the islands of the western Sverdrup group.
polygon([(105,79),(102.7,79),(102.5,78.72),(103.05,78.38),(104.1,78.10),
         (105,78.08)])
polygon([(105,77.98),(103.8,77.95),(102.35,77.66),(101.85,77.35),
         (102.65,77.12),(104.2,77.15),(105,77.34)])              # Lougheed
polygon([(104.9,77.10),(103.55,77.05),(102.65,76.76),(102.65,76.30),
         (103.35,76.12),(104.25,76.25),(104.75,76.62)])           # Cameron
polygon([(102.45,77.12),(101.15,77.12),(100.55,76.85),(100.72,76.38),
         (101.55,76.17),(102.25,76.43)])                           # Vanier/Massey
ellipse(103.2,77.37,.62,.22,-12)                                   # Helena
ellipse(101.15,77.48,.62,.22,8)                                    # King Christian

# Amund Ringnes, Cornwall, Graham and small Norwegian Bay islands.
polygon([(99.15,78.88),(97.10,78.98),(95.72,78.72),(95.15,78.25),
         (95.45,77.92),(96.45,77.77),(97.55,77.88),(98.55,78.22)])
polygon([(96.30,78.23),(94.88,78.30),(94.35,78.05),(94.72,77.75),
         (95.72,77.67),(96.25,77.88)])                             # Cornwall
ellipse(91.65,77.48,.88,.34,-8)                                    # Graham
ellipse(94.15,77.35,.38,.17,10)
ellipse(98.15,77.20,.35,.18,-10)

# Axel Heiberg's southern part along the top, separated by Norwegian Bay.
polygon([(96.05,79),(87.65,79),(87.90,78.72),(88.65,78.35),(89.55,78.17),
         (90.55,78.15),(91.25,78.38),(92.05,78.35),(92.50,78.60),
         (93.45,78.62),(94.00,78.82),(95.25,78.82)])
# Fjords cutting the south coast of Axel Heiberg.
polygon([(89.00,78.25),(89.22,78.78),(89.48,79),(90.00,79),(89.70,78.55),
         (89.65,78.18)],0)
polygon([(92.05,78.35),(92.35,78.72),(92.55,79),(93.00,79),(92.72,78.55),
         (92.52,78.30)],0)

# Southern Ellesmere Island, with an irregular Jones Sound shore.
polygon([(87.55,79),(81,79),(81,76.92),(82.05,76.82),(82.85,76.95),
         (83.55,77.20),(84.40,77.18),(85.05,77.42),(86.05,77.47),
         (86.62,77.83),(87.35,78.10),(87.85,78.55)])
# Major southwest-facing fiords.
polygon([(84.40,77.17),(84.25,77.70),(84.55,78.30),(85.00,78.62),
         (84.80,77.80),(84.75,77.25)],0)
polygon([(86.02,77.47),(85.78,77.82),(85.75,78.35),(86.15,78.55),
         (86.12,77.82),(86.45,77.58)],0)
polygon([(82.05,76.82),(81.82,77.25),(81.88,77.80),(82.22,78.05),
         (82.18,77.20),(82.55,76.90)],0)

# Bathurst Island, deliberately assembled as its several deeply indented lobes.
polygon([(104.55,76.68),(103.60,76.82),(102.20,76.75),(101.45,76.50),
         (100.62,76.62),(99.75,76.45),(99.62,76.02),(98.85,75.78),
         (98.55,75.35),(99.25,75.12),(100.18,75.18),(100.72,75.45),
         (101.55,75.25),(102.55,75.28),(103.18,75.62),(104.05,75.70),
         (104.48,76.10)])
# Erskine and May inlets and the channel between major lobes.
polygon([(101.05,75.15),(100.85,75.55),(100.95,76.15),(101.28,76.52),
         (101.55,76.45),(101.38,75.85),(101.50,75.28)],0)
polygon([(103.35,75.57),(102.95,75.88),(102.72,76.38),(102.95,76.77),
         (103.30,76.70),(103.25,76.15),(103.72,75.70)],0)
polygon([(99.70,75.28),(99.88,75.60),(99.82,76.05),(99.55,76.32),
         (99.38,75.80),(99.20,75.35)],0)
ellipse(104.65,75.50,.28,.17,5)                                    # small western island
ellipse(97.88,76.18,.57,.28,-20)                                   # Alexander
ellipse(97.22,76.65,.52,.25,-8)                                    # Marc
# Cornwallis and Little Cornwallis Islands.
polygon([(96.75,75.62),(95.92,75.86),(94.68,75.83),(93.68,75.65),
         (92.82,75.28),(92.72,74.88),(93.38,74.60),(94.45,74.60),
         (95.28,74.78),(96.15,75.08)])
polygon([(97.30,75.88),(96.55,76.05),(95.75,75.88),(95.88,75.50),
         (96.62,75.38),(97.20,75.56)])
# McDougall Sound indentation and nearby islets.
polygon([(96.18,75.08),(95.78,75.22),(95.48,75.55),(95.58,75.82),
         (95.92,75.78),(95.90,75.38),(96.42,75.18)],0)
ellipse(93.05,75.92,.38,.18,-5)                                    # Baillie-Hamilton
ellipse(98.18,75.02,.48,.17,-12)                                   # Garrett
ellipse(97.25,74.60,.45,.18,5)                                     # Lowther
ellipse(95.25,74.57,.32,.16,8)                                     # Griffith

# Devon Island: a long mass north of Lancaster Sound.
polygon([(96.05,76.68),(94.72,76.78),(93.50,76.70),(92.15,76.82),
         (90.75,76.68),(89.48,76.80),(88.20,76.58),(86.78,76.55),
         (85.50,76.42),(84.10,76.22),(82.75,76.12),(81,76.18),
         (81,74.92),(82.20,74.83),(83.48,74.78),(84.75,74.55),
         (86.15,74.45),(87.52,74.48),(88.70,74.66),(89.70,74.82),
         (90.92,74.80),(92.02,74.98),(93.12,75.18),(94.05,75.52),
         (95.00,75.78),(95.72,76.18)])
# Devon's large north-coast fiords and eastern indentations.
polygon([(92.05,76.83),(91.85,76.52),(91.92,76.05),(92.28,75.72),
         (92.52,75.82),(92.35,76.30),(92.55,76.72)],0)
polygon([(89.48,76.80),(89.28,76.47),(89.40,76.12),(89.78,75.82),
         (90.00,75.93),(89.72,76.42),(89.85,76.73)],0)
polygon([(86.82,76.57),(86.58,76.22),(86.62,75.82),(87.00,75.48),
         (87.24,75.60),(87.02,76.12),(87.25,76.53)],0)
polygon([(84.10,76.23),(83.75,75.95),(83.62,75.57),(83.82,75.22),
         (84.18,75.30),(84.12,75.72),(84.52,76.12)],0)
polygon([(81,75.45),(81.78,75.35),(82.42,75.58),(82.72,76.12),(81.90,76.25),
         (81,76.20)],0)
# North Kent Island in Cardigan Strait, plus small Wellington Channel islands.
polygon([(91.15,76.80),(90.38,76.72),(89.93,76.45),(90.18,76.10),
         (90.82,76.08),(91.20,76.40)])
ellipse(96.80,75.03,.26,.15,-5)
ellipse(91.92,74.70,.25,.13,8)                                     # Beechey
ellipse(90.05,74.20,.23,.14,5)                                     # Prince Leopold

# Prince of Wales Island south of Barrow Strait.
polygon([(102.80,74.40),(101.92,74.55),(100.78,74.50),(99.78,74.35),
         (98.75,74.42),(97.85,74.28),(97.08,74.02),(96.78,73.62),
         (96.08,73.28),(95.92,73.00),(102.35,73.00),(102.62,73.35),
         (103.05,73.72)])
# Ommanney Bay and other north/east coast cuts.
polygon([(99.82,74.36),(99.48,74.08),(99.34,73.72),(99.58,73.42),
         (99.82,73.62),(99.78,74.02),(100.15,74.32)],0)
polygon([(97.55,74.18),(97.42,73.92),(97.55,73.55),(97.90,73.30),
         (98.05,73.50),(97.82,73.88),(97.92,74.22)],0)
polygon([(102.78,74.08),(102.45,73.88),(102.32,73.50),(102.42,73.10),
         (102.78,73.20),(102.72,73.65),(103.02,73.82)],0)

# Somerset Island, east of Peel Sound.
polygon([(95.82,74.02),(94.98,74.13),(94.18,74.00),(93.28,74.05),
         (92.38,73.92),(91.62,73.62),(91.10,73.22),(91.08,73.00),
         (95.72,73.00),(95.52,73.38),(95.72,73.72)])
polygon([(93.30,74.02),(93.05,73.72),(93.10,73.32),(93.45,73.02),
         (93.68,73.12),(93.48,73.58),(93.70,73.92)],0)
polygon([(91.55,73.62),(91.45,73.25),(91.62,73.00),(92.02,73.00),
         (91.90,73.45),(92.15,73.78)],0)

# Brodeur Peninsula (Baffin Island) and the edge of Bylot Island.
polygon([(88.95,73.00),(88.58,73.35),(87.98,73.58),(87.18,73.62),
         (86.40,73.45),(85.85,73.20),(85.62,73.00)])
polygon([(83.05,73.00),(82.65,73.25),(81.92,73.42),(81,73.38),(81,73.00)])

# A few genuine-scale small islands in the main channels.
ellipse(104.15,74.57,.26,.11,-5)
ellipse(101.55,74.72,.20,.10,0)
ellipse(89.42,74.22,.14,.09,0)
ellipse(88.70,74.02,.12,.08,0)

# Export and enforce all requested invariants.
out = Path("outputs")
out.mkdir(parents=True, exist_ok=True)
assert land.shape == (61,241) and land.size == 14701
assert set(np.unique(land)).issubset({0,1})
with (out/"candidate.csv").open("w", newline="", encoding="ascii") as f:
    csv.writer(f, lineterminator="\n").writerows(land.tolist())
im = Image.fromarray(land*255, mode="L")
im.save(out/"candidate.png")
nearest = getattr(Image, "Resampling", Image).NEAREST
big = im.resize((1928,488), resample=nearest)
big.save(out/"candidate_8x.png")
assert im.size == (241,61) and big.size == (1928,488)
assert set(im.getdata()) <= {0,255} and set(big.getdata()) <= {0,255}
print("wrote 61x241 CSV, 241x61 PNG, and 1928x488 nearest-neighbor PNG")
