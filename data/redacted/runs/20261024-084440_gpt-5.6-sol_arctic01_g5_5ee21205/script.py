"""Knowledge-only reconstruction of the Canadian Arctic Archipelago.

Coordinates below are deliberately hand-drawn geographic approximations.  The
program does not read any map, GIS file, coastline package, or network source.
"""
from pathlib import Path
import csv
import math
import numpy as np
from PIL import Image


def inside_poly(w, n, p):
    """Odd/even point in polygon test; coordinates are (degrees west, north)."""
    hit = False
    j = len(p) - 1
    for i in range(len(p)):
        xi, yi = p[i]
        xj, yj = p[j]
        if ((yi > n) != (yj > n)):
            cut = (xj-xi) * (n-yi) / (yj-yi) + xi
            if w < cut:
                hit = not hit
        j = i
    return hit


def ellipse(w, n, cw, cn, rw, rn, tilt=0):
    a = math.radians(tilt)
    x, y = w-cw, n-cn
    u = x*math.cos(a) + y*math.sin(a)
    v = -x*math.sin(a) + y*math.cos(a)
    return (u/rw)**2 + (v/rn)**2 <= 1


# Broad island shores, recalled and simplified to a few dozen vertices.
LAND = [
 # Ellef Ringnes (continues beyond north and west edges)
 [(105,77.85),(103.8,77.72),(102.4,77.86),(101.0,77.78),(99.8,78.02),
  (98.0,78.16),(96.7,78.52),(96.5,79.0),(105,79.0)],
 # Amund Ringnes
 [(97.0,77.80),(95.7,77.62),(94.1,77.63),(92.7,77.90),(92.0,78.22),
  (92.4,78.53),(93.6,78.73),(95.1,78.78),(96.4,78.55),(97.1,78.18)],
 # Axel Heiberg, its southern and central portion
 [(92.1,78.05),(91.0,77.90),(89.8,78.00),(89.1,78.25),(88.4,78.48),
  (87.1,78.73),(85.6,78.93),(84.7,79.0),(93.0,79.0),(93.0,78.55)],
 # Ellesmere's SW margin at Jones Sound
 [(83.5,77.80),(82.2,77.78),(81.0,77.92),(81.0,79.0),(84.8,79.0),
  (84.5,78.65)],
 # Lougheed Island
 [(105,77.12),(103.8,77.12),(102.3,77.28),(101.0,77.55),(101.2,77.88),
  (102.4,78.02),(104.0,77.90),(105,77.65)],
 # King Christian Island
 [(103.0,77.65),(101.9,77.58),(101.1,77.73),(101.25,77.98),
  (102.3,78.10),(103.2,77.95)],
 # Cornwall Island
 [(96.7,77.38),(95.1,77.38),(94.1,77.62),(94.2,77.96),
  (95.3,78.12),(96.4,77.93),(96.9,77.66)],
 # Graham Island
 [(91.9,77.20),(90.4,77.16),(89.4,77.40),(89.7,77.72),
  (90.8,77.82),(91.8,77.62)],
 # Devon Island
 [(96.5,74.72),(95.8,75.15),(95.0,75.70),(93.8,76.18),(92.0,76.55),
  (90.2,76.70),(88.3,76.64),(86.6,76.42),(85.0,76.25),(83.4,76.18),
  (82.0,76.30),(81.0,76.46),(81.0,74.55),(83.0,74.48),(85.0,74.45),
  (87.1,74.38),(89.2,74.34),(91.2,74.35),(93.4,74.42),(95.2,74.50)],
 # Bathurst Island, main body
 [(104.8,75.08),(104.3,75.58),(103.8,76.05),(102.7,76.48),
  (101.5,76.78),(100.1,76.84),(98.9,76.65),(98.0,76.20),
  (97.5,75.65),(97.8,75.15),(98.7,74.90),(100.0,75.03),
  (101.0,75.27),(102.0,75.18),(103.0,74.93),(104.2,74.88)],
 # Cameron Island
 [(105,76.12),(104.3,76.10),(103.3,76.32),(103.1,76.72),
  (103.8,77.02),(105,77.02)],
 # Vanier and Massey islands
 [(104.2,75.62),(102.8,75.55),(101.8,75.78),(101.7,76.10),
  (102.6,76.32),(103.7,76.22)],
 [(102.5,75.45),(101.3,75.42),(100.3,75.62),(100.3,75.92),
  (101.1,76.08),(102.3,75.93)],
 # Alexander Island
 [(102.1,76.15),(100.8,76.14),(99.8,76.38),(99.9,76.68),
  (101.0,76.83),(102.0,76.60)],
 # Byam Martin Island
 [(105,75.02),(104.1,74.98),(103.5,75.18),(103.6,75.52),
  (104.4,75.72),(105,75.66)],
 # Cornwallis Island
 [(97.0,74.70),(96.4,75.10),(96.1,75.50),(95.3,75.78),
  (94.1,75.70),(93.1,75.40),(93.0,75.02),(93.6,74.72),
  (94.8,74.55),(96.0,74.57)],
 # Little Cornwallis
 [(97.8,75.35),(96.8,75.32),(96.0,75.53),(96.2,75.82),
  (97.2,75.92),(97.9,75.70)],
]

# Southern islands bounding the actual Northwest Passage channels.
LAND += [
 # northeast Victoria Island, clipped by the west and south frame
 [(105,73.0),(105,73.82),(104.4,73.76),(103.8,73.55),(103.4,73.22),
  (103.5,73.0)],
 # Prince of Wales Island
 [(102.9,73.0),(102.8,73.45),(102.5,73.92),(101.8,74.25),
  (100.7,74.48),(99.4,74.46),(98.2,74.25),(97.0,73.97),
  (96.2,73.60),(96.0,73.20),(96.2,73.0)],
 # Somerset Island (south continues beyond frame)
 [(95.6,73.0),(95.8,73.45),(95.5,73.83),(94.8,74.08),(93.6,74.14),
  (92.6,74.00),(91.7,73.75),(91.0,73.38),(90.3,73.0)],
 # Brodeur Peninsula of Baffin Island, cut by Admiralty Inlet
 [(89.4,73.0),(89.1,73.24),(88.3,73.48),(87.2,73.62),(86.1,73.72),
  (85.2,73.82),(84.4,73.85),(84.2,73.55),(84.5,73.20),(84.7,73.0)],
 [(82.8,73.0),(82.9,73.32),(82.5,73.55),(81.8,73.72),(81.0,73.76),
  (81.0,73.0)],
]

# A few named, resolvable islands in the larger channels.
ISLETS = [
 (98.55, 74.67, .36, .13, -15),   # Griffith
 (100.15,74.65, .20, .11, 0),
 (91.95, 74.22, .32, .11, 5),     # Browne/nearby islets
 (93.35, 74.28, .22, .09, 0),
 (96.55, 74.27, .18, .09, 0),
 (89.00, 74.15, .34, .10, 10),
 (82.15, 74.15, .35, .10, -10),
 (104.55,74.30, .32, .10, 5),
]

# Inlets drawn after the broad land polygons.  These are water, not lakes.
WATER = [
 # Bathurst: May Inlet and the deeply divided southeast coast
 [(101.2,76.88),(100.7,76.65),(100.5,76.34),(100.7,76.05),
  (101.0,75.82),(101.25,76.12),(101.45,76.50)],
 [(99.0,74.88),(99.1,75.18),(99.5,75.45),(100.0,75.55),
  (99.8,75.27),(99.5,75.02)],
 [(103.5,74.88),(103.1,75.16),(102.7,75.38),(102.45,75.30),
  (102.7,75.05)],
 # Devon's three conspicuous south-coast bays
 [(92.0,74.34),(91.7,74.62),(91.8,74.94),(92.15,75.17),
  (92.45,74.91),(92.35,74.58)],
 [(89.2,74.34),(88.7,74.65),(88.5,75.02),(88.7,75.32),
  (89.15,75.18),(89.45,74.78)],
 [(85.7,74.43),(85.2,74.75),(84.9,75.18),(84.45,75.35),
  (84.15,75.16),(84.35,74.78),(84.85,74.47)],
 # smaller north-coast embayments of Devon
 [(91.1,76.69),(90.8,76.48),(90.5,76.25),(90.15,76.32),
  (90.25,76.60)],
 [(86.7,76.43),(86.3,76.20),(86.0,75.92),(85.7,76.20),(85.8,76.34)],
 # Cornwallis: McDougall/Resolute embayments
 [(95.2,74.55),(95.0,74.78),(95.15,75.03),(95.48,74.83)],
 [(93.8,74.68),(94.0,74.91),(94.2,74.72)],
 # Prince of Wales northern and eastern indentations
 [(100.6,74.48),(100.2,74.19),(100.3,73.88),(100.65,74.02)],
 [(96.05,73.18),(96.5,73.42),(96.85,73.78),(97.15,73.72),
  (96.9,73.28),(96.65,73.0)],
 # Somerset bays
 [(95.6,73.0),(95.3,73.30),(95.25,73.68),(94.95,73.78),
  (94.8,73.36),(94.95,73.0)],
 [(91.0,73.05),(91.45,73.35),(91.65,73.72),(91.95,73.61),
  (91.75,73.22),(91.55,73.0)],
]

def classify(w, n):
    land = any(inside_poly(w, n, p) for p in LAND)
    if land and any(inside_poly(w, n, p) for p in WATER):
        land = False
    if any(ellipse(w, n, *q) for q in ISLETS):
        land = True
    return 1 if land else 0


def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    # Integer construction avoids accumulated floating-point coordinate drift.
    northings = [79.0 - i/10 for i in range(61)]
    westings = [105.0 - j/10 for j in range(241)]
    matrix = np.array([[classify(w, n) for w in westings]
                       for n in northings], dtype=np.uint8)

    assert matrix.shape == (61, 241)
    assert matrix.size == 14701
    assert set(np.unique(matrix)).issubset({0, 1})

    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

    pixels = matrix * np.uint8(255)
    Image.fromarray(pixels, mode="L").save(out / "candidate.png")
    enlarged = np.repeat(np.repeat(pixels, 8, axis=0), 8, axis=1)
    Image.fromarray(enlarged, mode="L").save(out / "candidate_8x.png")

    # Re-open deliverables and independently check their serialized form.
    rows = list(csv.reader((out / "candidate.csv").open(encoding="ascii")))
    assert len(rows) == 61 and all(len(r) == 241 for r in rows)
    assert all(v in ("0", "1") for r in rows for v in r)
    with Image.open(out / "candidate.png") as im:
        assert im.size == (241, 61) and set(im.getdata()) <= {0, 255}
    with Image.open(out / "candidate_8x.png") as im:
        assert im.size == (1928, 488) and set(im.getdata()) <= {0, 255}
        z = np.asarray(im)
        assert np.array_equal(z, np.repeat(np.repeat(pixels, 8, 0), 8, 1))
    print("validated: 61x241 CSV; 241x61 PNG; 1928x488 nearest-neighbor PNG")


if __name__ == "__main__":
    main()
