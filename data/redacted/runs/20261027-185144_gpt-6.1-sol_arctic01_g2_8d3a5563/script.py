from pathlib import Path
import csv
import numpy as np
from PIL import Image

# Hand-drawn, deliberately approximate coastlines.  Coordinates are (west/east
# longitude, latitude), in decimal degrees; no external geographic data is used.
BOX_W, BOX_E = -105.0, -81.0
BOX_S, BOX_N = 73.0, 79.0
DX = DY = 0.1

def poly(*xy):
    return [(float(x), float(y)) for x, y in xy]

def inside(x, y, p):
    hit = False
    j = len(p) - 1
    for i, (xi, yi) in enumerate(p):
        xj, yj = p[j]
        cross = ((yi > y) != (yj > y))
        if cross and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            hit = not hit
        j = i
    return hit

land = []

# The large northern islands and their characteristic bays/inlets.
land += [
    poly((-91.2,79.0),(-90.7,78.5),(-90.4,77.8),(-89.7,77.3),
         (-88.3,77.0),(-86.2,76.8),(-84.3,76.7),(-82.2,76.9),
         (-80.5,77.5),(-79.1,78.2),(-78.0,79.0)),
    poly((-97.2,79.0),(-96.8,78.4),(-96.0,77.9),(-94.8,77.7),
         (-93.0,77.8),(-91.8,78.5),(-91.4,79.0)),
    poly((-105.0,78.7),(-103.8,78.0),(-102.2,77.8),(-100.3,77.9),
         (-99.0,78.4),(-98.5,79.0)),
    # Ellesmere south-western and central mass
    poly((-91.6,77.0),(-91.0,76.5),(-90.0,76.0),(-89.2,75.5),
         (-88.4,75.0),(-87.3,74.8),(-86.0,74.9),(-84.6,75.4),
         (-83.8,76.1),(-83.0,76.7),(-82.1,77.1),(-80.5,77.6),
         (-79.2,78.2),(-78.0,79.0),(-91.6,79.0)),
]

# Axel Heiberg and the Sverdrup islands.
land += [
    poly((-97.0,79.0),(-96.4,78.5),(-95.4,78.0),(-94.1,77.6),
         (-93.1,77.5),(-92.5,78.0),(-92.4,79.0)),
    poly((-100.8,79.0),(-100.0,78.6),(-98.8,78.2),(-97.7,78.0),
         (-96.8,78.2),(-96.5,78.8),(-96.4,79.0)),
    poly((-105.0,77.9),(-103.8,77.5),(-102.1,77.3),(-100.5,77.4),
         (-99.4,77.8),(-99.0,78.3),(-99.6,79.0),(-105.0,79.0)),
    # Amund Ringnes
    poly((-101.4,77.5),(-100.1,76.9),(-98.7,76.8),(-97.3,77.1),
         (-96.3,77.7),(-96.7,78.1),(-98.2,78.2),(-100.0,78.0)),
    # Ellef Ringnes
    poly((-105.0,77.0),(-103.8,76.4),(-102.1,76.3),(-100.6,76.7),
         (-99.7,77.3),(-100.0,77.8),(-101.5,78.0),(-103.5,77.8)),
]
# Islands south of the Sverdrup group.
land += [
    # Melville Island: broad, indented island at the north-west of the
    # central passage.
    poly((-105.0,76.8),(-103.4,76.4),(-101.8,76.0),(-100.8,75.5),
         (-100.0,75.0),(-98.7,74.7),(-97.0,74.8),(-95.8,75.2),
         (-94.7,75.9),(-94.0,76.5),(-94.5,77.0),(-96.0,77.0),
         (-97.2,76.6),(-98.5,76.3),(-99.8,76.4),(-101.2,76.8),
         (-102.8,77.1),(-104.3,77.2)),
    # Bathurst Island
    poly((-101.7,76.2),(-100.7,75.7),(-99.4,75.5),(-98.0,75.7),
         (-97.0,76.2),(-97.2,76.7),(-98.4,76.9),(-100.1,76.8),
         (-101.3,76.6)),
    # Cornwallis Island
    poly((-96.0,76.4),(-95.0,76.1),(-94.0,75.5),(-93.5,75.0),
         (-94.1,74.7),(-95.3,74.8),(-96.2,75.2),(-96.6,75.8)),
    # Devon Island, with a long south coast and a high north-west corner.
    poly((-105.0,75.5),(-103.4,75.0),(-101.7,74.7),(-100.2,74.6),
         (-98.7,74.9),(-97.8,75.4),(-97.5,76.0),(-98.2,76.7),
         (-99.5,77.0),(-101.2,76.9),(-102.7,76.5),(-104.0,76.1),
         (-105.0,76.2)),
    # small islands in Lancaster Sound / Wellington Channel
    poly((-94.0,76.8),(-93.3,76.5),(-92.7,76.1),(-92.4,75.7),
         (-92.8,75.4),(-93.5,75.7),(-94.0,76.1)),
    poly((-99.4,74.0),(-98.7,73.8),(-98.3,73.4),(-98.8,73.1),
         (-99.5,73.2),(-99.8,73.6)),
]

# Victoria Island (only its north-eastern part is inside this frame).
land += [
    poly((-105.0,75.4),(-104.0,75.0),(-102.6,74.8),(-101.6,74.2),
         (-100.8,73.7),(-99.6,73.3),(-98.7,73.0),(-105.0,73.0)),
    poly((-105.0,75.9),(-104.1,75.7),(-103.0,75.5),(-102.2,75.1),
         (-101.2,74.8),(-100.5,75.1),(-100.6,75.7),(-101.5,76.0),
         (-103.0,76.1),(-104.2,76.3)),
]

# Prince of Wales Island, Somerset Island, and northern Boothia.
land += [
    poly((-101.0,75.0),(-100.0,74.7),(-98.7,74.6),(-97.6,74.1),
         (-96.7,73.5),(-96.0,73.0),(-101.0,73.0),(-101.8,73.6),
         (-101.9,74.3)),
    poly((-99.4,74.1),(-98.5,73.8),(-97.3,73.5),(-96.2,73.2),
         (-95.4,73.0),(-100.0,73.0),(-100.1,73.5)),
    poly((-95.0,73.4),(-94.2,73.1),(-93.1,73.0),(-92.4,73.0),
         (-92.6,73.7),(-93.3,74.1),(-94.3,74.2)),
]
# Eastern edge of the frame: the west coast of Baffin Island and a few
# outlying rocks.  The very narrow pieces are retained because this is a
# point-sampled map rather than an area-averaged map.
land += [
    poly((-81.0,73.0),(-80.3,73.2),(-79.8,73.8),(-79.6,74.4),
         (-80.2,74.8),(-81.0,74.8)),
    poly((-84.0,73.0),(-83.6,73.2),(-83.4,73.5),(-83.7,73.7),
         (-84.1,73.5)),
    poly((-95.0,73.0),(-94.5,73.2),(-94.1,73.4),(-94.3,73.7),
         (-95.1,73.5),(-95.4,73.1)),
]

# Water polygons cut the deliberately overlapping coarse island outlines.
# They represent named sounds and channels, not a hidden map lookup.
water = [
    # Nansen Sound and its southern continuation
    poly((-95.0,79.0),(-94.1,78.5),(-93.1,77.8),(-92.4,77.1),
         (-92.9,76.8),(-93.8,77.3),(-94.8,78.0),(-96.0,78.6)),
    # Norwegian Bay
    poly((-91.8,76.8),(-90.8,76.3),(-89.4,75.8),(-88.2,75.5),
         (-87.0,75.7),(-86.1,76.3),(-86.6,76.8),(-88.0,76.7),
         (-89.5,77.0),(-90.8,77.4)),
    # Wellington Channel
    poly((-96.0,76.2),(-95.0,75.8),(-94.0,75.1),(-93.2,74.6),
         (-92.8,74.1),(-93.5,74.0),(-94.3,74.6),(-95.0,75.3),
         (-96.2,75.7)),
    # Barrow Strait / Viscount Melville Sound
    poly((-99.0,75.1),(-97.7,74.9),(-96.1,74.5),(-94.8,74.2),
         (-94.1,73.7),(-94.5,73.4),(-96.1,73.7),(-97.5,74.2),
         (-99.0,74.5),(-100.3,74.7)),
    # Peel Sound, between Somerset and Prince of Wales
    poly((-100.2,74.4),(-99.2,74.1),(-98.2,73.6),(-97.0,73.0),
         (-96.1,73.0),(-96.7,73.7),(-98.0,74.4),(-99.3,74.8)),
    # Franklin Strait and the southern approach
    poly((-97.2,74.0),(-96.1,73.6),(-95.2,73.0),(-94.5,73.0),
         (-95.0,73.7),(-96.0,74.4)),
    # M'Clintock Channel
    poly((-105.0,74.1),(-103.8,73.8),(-102.5,73.5),(-101.2,73.0),
         (-100.2,73.0),(-101.0,73.7),(-102.6,74.3),(-104.0,74.6)),
]
def is_land(lon, lat):
    # The order is intentional: channels are water even where coarse hand
    # polygons overlap at their edges.
    return any(inside(lon, lat, p) for p in land) and not any(
        inside(lon, lat, p) for p in water
    )

def make_matrix():
    a = np.zeros((61, 241), dtype=np.uint8)
    for r in range(61):
        lat = BOX_N - r * DY
        for c in range(241):
            lon = BOX_W + c * DX
            a[r, c] = 1 if is_land(lon, lat) else 0
    return a

def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    a = make_matrix()
    if a.shape != (61, 241) or int(a.size) != 14701:
        raise RuntimeError("unexpected matrix dimensions")
    if not np.all((a == 0) | (a == 1)):
        raise RuntimeError("matrix contains a value other than 0 or 1")

    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        w = csv.writer(f, lineterminator="\n")
        for row in a:
            w.writerow([int(v) for v in row])

    raw = Image.fromarray(a * 255, mode="L")
    if raw.size != (241, 61) or set(raw.getdata()) - {0, 255}:
        raise RuntimeError("raw image validation failed")
    raw.save(out / "candidate.png")

    enlarged = raw.resize((241 * 8, 61 * 8), Image.Resampling.NEAREST)
    if enlarged.size != (1928, 488) or set(enlarged.getdata()) - {0, 255}:
        raise RuntimeError("enlarged image validation failed")
    enlarged.save(out / "candidate_8x.png")

    # Validate the serialized CSV too, rather than only the in-memory array.
    lines = (out / "candidate.csv").read_text(encoding="ascii").splitlines()
    if len(lines) != 61 or any(len(x.split(",")) != 241 for x in lines):
        raise RuntimeError("CSV shape validation failed")
    if any(ch not in "01,\n" for ch in (out / "candidate.csv").read_text()):
        raise RuntimeError("CSV alphabet validation failed")

if __name__ == "__main__":
    main()
