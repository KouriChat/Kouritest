import os
import csv
import math
import numpy as np
from PIL import Image

NROW, NCOL = 64, 253
west, east = 129.0, 137.0
south, north = 33.0, 35.0
step = 2.0 / 63.0
lons = np.array([west + j * step for j in range(NCOL)])
lats = np.array([north - i * step for i in range(NROW)])
land_shapes = []
water_shapes = []

def poly(points):
    """Store a hand-drawn closed polygon, in (longitude, latitude) order."""
    land_shapes.append([(float(x), float(y)) for x, y in points])

def water_poly(points):
    water_shapes.append([(float(x), float(y)) for x, y in points])

def ellipse(cx, cy, rx, ry, n=20, angle=0.0):
    ca, sa = math.cos(angle), math.sin(angle)
    poly([(cx + rx * math.cos(2 * math.pi*k/n) * ca
           - ry * math.sin(2 * math.pi*k/n) * sa,
           cy + rx * math.cos(2 * math.pi*k/n) * sa
           + ry * math.sin(2 * math.pi*k/n) * ca)
          for k in range(n)])

def inside(x, y, p):
    hit = False
    x0, y0 = p[-1]
    for x1, y1 in p:
        if (y1 > y) != (y0 > y):
            at = (x0 - x1) * (y - y1) / (y0 - y1) + x1
            if x < at:
                hit = not hit
        x0, y0 = x1, y1
    return hit

def fill_shape(a, p):
    # Bounding-box rejection keeps the deliberately simple geometry fast.
    xs = [q[0] for q in p]
    ys = [q[1] for q in p]
    js = [j for j, x in enumerate(lons) if min(xs) <= x <= max(xs)]
    is_ = [i for i, y in enumerate(lats) if min(ys) <= y <= max(ys)]
    for i in is_:
        for j in js:
            if inside(lons[j], lats[i], p):
                a[i, j] = 1

def clear_shape(a, p):
    xs = [q[0] for q in p]
    ys = [q[1] for q in p]
    js = [j for j, x in enumerate(lons) if min(xs) <= x <= max(xs)]
    is_ = [i for i, y in enumerate(lats) if min(ys) <= y <= max(ys)]
    for i in is_:
        for j in js:
            if inside(lons[j], lats[i], p):
                a[i, j] = 0

# Large, continuous land masses.  These are intentionally coastal outlines,
# rather than a rectangular approximation of the islands.

# Western Honshu and the Chugoku coast (the north side of the Inland Sea).
poly([
 (130.55,35.00),(137.00,35.00),(137.00,34.82),(136.72,34.78),
 (136.48,34.70),(136.20,34.67),(135.92,34.67),(135.68,34.64),
 (135.50,34.62),(135.35,34.58),(135.23,34.55),(135.08,34.54),
 (134.92,34.55),(134.76,34.58),(134.56,34.60),(134.36,34.61),
 (134.18,34.58),(133.98,34.57),(133.80,34.52),(133.62,34.48),
 (133.45,34.43),(133.27,34.38),(133.08,34.34),(132.90,34.30),
 (132.72,34.28),(132.54,34.25),(132.36,34.22),(132.18,34.16),
 (132.00,34.11),(131.84,34.06),(131.70,34.01),(131.55,33.98),
 (131.38,33.95),(131.18,33.93),(130.96,33.94),(130.82,34.00),
 (130.72,34.14),(130.68,34.32),(130.66,34.52),(130.62,34.72),
 (130.58,34.90)
])

# Southern/eastern Honshu, including the Kii peninsula on the lower right.
poly([
 (135.10,35.00),(137.00,35.00),(137.00,33.00),(136.76,33.00),
 (136.60,33.15),(136.48,33.34),(136.31,33.52),(136.17,33.70),
 (136.06,33.91),(135.93,34.10),(135.77,34.28),(135.62,34.43),
 (135.46,34.52),(135.28,34.57),(135.12,34.62),(134.98,34.71)
])

# Kyushu, clipped naturally by the requested frame at its southern edge.
poly([
 (129.00,33.00),(129.34,33.00),(129.55,33.08),(129.74,33.18),
 (129.92,33.30),(130.08,33.43),(130.24,33.56),(130.42,33.68),
 (130.62,33.76),(130.82,33.82),(131.04,33.86),(131.22,33.82),
 (131.38,33.73),(131.52,33.61),(131.63,33.47),(131.69,33.32),
 (131.74,33.16),(131.82,33.00),(129.00,33.00)
])

# Shikoku, with its characteristic long south coast.
poly([
 (132.00,33.66),(132.16,33.72),(132.36,33.77),(132.58,33.82),
 (132.82,33.86),(133.08,33.89),(133.36,33.91),(133.64,33.93),
 (133.92,33.94),(134.18,33.93),(134.42,33.88),(134.66,33.80),
 (134.83,33.68),(134.91,33.52),(134.83,33.38),(134.70,33.27),
 (134.52,33.19),(134.30,33.13),(134.06,33.09),(133.80,33.07),
 (133.53,33.06),(133.28,33.08),(133.02,33.12),(132.78,33.17),
 (132.56,33.23),(132.37,33.31),(132.22,33.42),(132.10,33.55)
])

# Principal islands around Kyushu and the western entrance to the Inland Sea.
poly([
 (129.18,34.63),(129.25,34.78),(129.38,34.82),(129.49,34.74),
 (129.51,34.58),(129.46,34.43),(129.48,34.28),(129.41,34.13),
 (129.29,34.08),(129.19,34.18),(129.17,34.36),(129.21,34.52)
])                                      # Tsushima
ellipse(129.76, 33.80, .13, .11, 18, -.15)  # Iki
ellipse(129.46, 33.39, .10, .12, 16, -.4)   # Hirado
ellipse(129.18, 33.49, .07, .10, 14, -.2)   # western Goto island
ellipse(129.74, 33.20, .07, .10, 14, .2)
ellipse(129.99, 33.27, .09, .07, 14, -.2)

# Islands and peninsular rocks along the northern Kyushu coast.
ellipse(130.06, 33.78, .08, .055, 14, .2)
ellipse(130.38, 33.83, .07, .05, 14, -.2)
ellipse(130.78, 33.90, .06, .04, 12, 0)
ellipse(131.08, 33.93, .055, .035, 12, 0)

# Western and central Inland Sea islands.
ellipse(131.56, 33.98, .08, .045, 14, -.1)
ellipse(131.87, 34.05, .10, .055, 14, .1)
ellipse(132.31, 34.29, .09, .07, 18, .2)    # Miyajima
ellipse(132.56, 34.12, .07, .04, 14, -.1)
ellipse(132.72, 34.19, .09, .05, 14, .2)
ellipse(132.93, 34.16, .07, .04, 14, -.1)
ellipse(133.07, 34.29, .14, .07, 18, .2)    # Innoshima
ellipse(133.20, 34.30, .12, .065, 18, .2)   # Mukaishima
ellipse(133.30, 34.28, .13, .07, 18, -.2)   # Ikuchijima
ellipse(133.42, 34.25, .14, .075, 18, .15)  # Omishima
ellipse(133.59, 34.20, .12, .06, 16, .1)
ellipse(133.76, 34.23, .10, .06, 16, -.1)
ellipse(133.89, 34.21, .07, .045, 14, .1)

# Kagawa-side islands.
ellipse(132.86, 34.38, .07, .045, 14, 0)
ellipse(133.54, 34.40, .08, .05, 14, -.1)
ellipse(133.83, 34.43, .07, .045, 14, .1)
ellipse(134.03, 34.39, .11, .065, 18, -.15) # Shodoshima west/east
ellipse(134.20, 34.47, .12, .075, 18, .1)
ellipse(134.38, 34.42, .07, .045, 14, .1)
ellipse(134.50, 34.48, .06, .04, 12, -.2)
ellipse(134.69, 34.38, .08, .05, 14, 0)
ellipse(134.76, 34.45, .055, .035, 12, 0)

# Islands in the central and eastern Inland Sea.
ellipse(134.91, 34.33, .07, .045, 14, -.2)
ellipse(135.12, 34.34, .06, .04, 12, 0)
ellipse(135.28, 34.32, .08, .045, 14, .1)
ellipse(135.42, 34.30, .065, .04, 12, -.1)
ellipse(135.55, 34.35, .06, .04, 12, 0)

# Awaji is large enough to deserve an explicit hand-drawn outline.
poly([
 (134.72,34.62),(134.84,34.68),(135.00,34.64),(135.10,34.53),
 (135.12,34.38),(135.07,34.23),(134.98,34.12),(134.87,34.10),
 (134.76,34.20),(134.69,34.34),(134.67,34.49)
])

# Southern Shikoku islands and a few larger islands at the eastern end.
ellipse(132.08, 33.78, .06, .04, 12, 0)
ellipse(132.46, 33.77, .07, .04, 14, -.1)
ellipse(132.88, 33.83, .06, .035, 12, 0)
ellipse(133.32, 33.83, .07, .04, 14, .1)
ellipse(133.74, 33.84, .06, .035, 12, -.2)
ellipse(134.15, 33.82, .07, .04, 14, .1)
ellipse(134.50, 33.76, .07, .045, 14, 0)
ellipse(134.79, 33.65, .06, .04, 12, -.2)

# A number of small islands in the Kii/Osaka approaches.
ellipse(135.32, 34.39, .05, .035, 12, 0)
ellipse(135.53, 34.43, .055, .035, 12, 0)
ellipse(135.72, 34.45, .07, .04, 12, -.2)
ellipse(135.94, 34.47, .05, .03, 12, 0)
ellipse(136.20, 34.48, .06, .035, 12, .1)

# Water polygons remove the broad bays which are swallowed by the simple
# southern Honshu outline above.  They are applied after land polygons.
water_poly([
 (136.62,35.01),(136.91,35.01),(136.93,34.90),(136.87,34.78),
 (136.78,34.65),(136.66,34.65),(136.60,34.79)
])
water_poly([
 (136.88,34.84),(137.01,34.84),(137.01,34.42),(136.91,34.42),
 (136.83,34.55),(136.82,34.70)
])

# Chita and Atsumi are retained as land tongues on the east side of Ise Bay.
poly([
 (136.91,34.98),(137.01,34.98),(137.01,34.62),(136.96,34.55),
 (136.89,34.62),(136.86,34.76)
])
poly([
 (136.99,34.62),(137.01,34.62),(137.01,34.26),(136.91,34.31),
 (136.84,34.43),(136.88,34.54)
])

def make_outputs():
    a = np.zeros((NROW, NCOL), dtype=np.uint8)
    for p in land_shapes:
        fill_shape(a, p)
    for p in water_shapes:
        clear_shape(a, p)
    # Re-apply the two narrow peninsulas after the bay cut.
    for p in land_shapes[-2:]:
        fill_shape(a, p)

    assert a.shape == (64, 253)
    assert int(a.sum()) >= 0
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/candidate.csv", "w", newline="", encoding="ascii") as f:
        w = csv.writer(f, lineterminator="\n")
        for row in a:
            w.writerow([int(v) for v in row])

    # Pillow's L mode gives exactly two possible pixel values.
    img = Image.fromarray((a * 255).astype(np.uint8), mode="L")
    img.save("outputs/candidate.png")
    big = img.resize((NCOL * 8, NROW * 8), resample=Image.Resampling.NEAREST)
    big.save("outputs/candidate_8x.png")

if __name__ == "__main__":
    make_outputs()
