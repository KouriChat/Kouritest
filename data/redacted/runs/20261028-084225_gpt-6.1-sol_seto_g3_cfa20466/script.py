import csv
import os
import numpy as np
from PIL import Image

# Hand-drawn approximate coastline polygons in (longitude, latitude).
# Polygon interiors are land; the polygons describe only this task's region.
LAND = [
    # Kyushu, including its northern and eastern shores.
    [(129.0,33.0),(129.0,34.0),(129.25,34.05),(129.45,33.95),
     (129.72,33.88),(129.92,33.72),(130.05,33.55),(130.18,33.45),
     (130.38,33.43),(130.55,33.52),(130.72,33.55),(130.82,33.75),
     (130.95,33.82),(131.08,33.86),(131.28,33.88),(131.48,33.95),
     (131.68,33.92),(131.82,33.82),(131.95,33.65),(131.95,33.0),
     (130.8,33.0),(130.4,32.9),(129.8,32.95)],
    # Western Honshu: Sea of Japan coast through Chugoku to Shimonoseki.
    [(130.85,34.22),(130.95,34.45),(131.08,34.62),(131.25,34.78),
     (131.52,34.9),(131.78,35.02),(132.02,35.15),(132.3,35.28),
     (132.55,35.38),(132.85,35.5),(133.15,35.58),(133.4,35.67),
     (133.7,35.73),(134.02,35.78),(134.35,35.82),(134.7,35.92),
     (135.0,36.0),(135.0,35.45),(134.85,35.2),(134.65,35.02),
     (134.45,34.9),(134.25,34.83),(134.05,34.8),(133.85,34.75),
     (133.62,34.72),(133.4,34.7),(133.2,34.68),(133.0,34.65),
     (132.8,34.62),(132.6,34.58),(132.4,34.55),(132.2,34.52),
     (132.0,34.5),(131.8,34.48),(131.6,34.46),(131.4,34.43),
     (131.2,34.4),(131.02,34.36),(130.9,34.3)],
    # Shikoku.
    [(132.0,33.25),(132.18,33.28),(132.35,33.32),(132.55,33.36),
     (132.75,33.4),(132.95,33.45),(133.15,33.48),(133.35,33.52),
     (133.55,33.58),(133.75,33.62),(133.95,33.68),(134.15,33.72),
     (134.32,33.76),(134.48,33.82),(134.62,33.9),(134.72,34.02),
     (134.68,34.14),(134.55,34.22),(134.35,34.25),(134.15,34.27),
     (133.95,34.3),(133.75,34.31),(133.55,34.32),(133.35,34.31),
     (133.15,34.3),(132.95,34.28),(132.75,34.26),(132.55,34.24),
     (132.35,34.22),(132.15,34.18),(132.0,34.12),(131.95,33.9),
     (131.95,33.6)],
    # Kii Peninsula and central/southern Honshu, including Ise Bay rim.
    [(134.35,34.83),(134.55,34.92),(134.75,35.05),(134.95,35.2),
     (135.08,35.34),(135.2,35.48),(135.34,35.55),(135.52,35.55),
     (135.72,35.52),(135.95,35.48),(136.2,35.45),(136.45,35.42),
     (136.7,35.45),(136.95,35.5),(137.0,35.7),(137.0,36.3),
     (136.55,36.35),(136.1,36.32),(135.7,36.28),(135.3,36.22),
     (134.9,36.12),(134.7,35.98),(134.8,35.8),(134.95,35.65),
     (134.88,35.5),(134.72,35.4),(134.58,35.27),(134.45,35.1)],
]
# Add the southern Kii Peninsula, eastern Honshu, and the Korean-side edge
# as separate hand-drawn land masses where the coarse frame reaches them.
LAND += [
    [(134.45,34.82),(134.7,34.72),(134.92,34.55),(135.05,34.3),
     (135.12,34.05),(135.08,33.8),(135.18,33.55),(135.38,33.35),
     (135.68,33.2),(136.0,33.1),(136.35,33.02),(136.7,33.0),
     (136.82,33.16),(136.72,33.38),(136.62,33.6),(136.68,33.88),
     (136.8,34.15),(136.85,34.4),(136.78,34.62),(136.62,34.82),
     (136.45,35.0),(136.25,35.15),(136.02,35.28),(135.78,35.38),
     (135.55,35.48),(135.35,35.5),(135.15,35.45),(134.98,35.3),
     (134.82,35.12),(134.65,34.98)],
    # Chubu coast around Mikawa Bay and the Ise Bay eastern shore.
    [(136.25,34.72),(136.45,34.78),(136.62,34.88),(136.78,35.02),
     (136.95,35.18),(137.0,35.45),(137.0,36.2),(136.75,36.28),
     (136.5,36.3),(136.25,36.25),(136.05,36.15),(135.9,36.0),
     (135.82,35.85),(135.9,35.68),(136.05,35.55),(136.15,35.4),
     (136.18,35.2),(136.15,35.0)],
]


def ellipse(cx, cy, rx, ry, n=20):
    """Return a fixed polygon approximation to a small island."""
    import math
    return [(cx + rx * math.cos(2 * math.pi * k / n),
             cy + ry * math.sin(2 * math.pi * k / n)) for k in range(n)]


# Major islands and the denser island groups of the Seto Inland Sea.
ISLANDS = [
    ellipse(129.55, 34.25, .22, .30),       # Tsushima
    ellipse(130.30, 34.18, .10, .18),       # Iki
    ellipse(130.55, 33.95, .07, .10),
    ellipse(132.00, 34.18, .12, .16),       # western Suo islands
    ellipse(132.42, 34.18, .14, .18),       # Etajima / adjacent land
    ellipse(132.75, 34.18, .12, .17),
    ellipse(133.02, 34.18, .18, .22),       # Geiyo group
    ellipse(133.25, 34.14, .13, .18),
    ellipse(133.48, 34.1, .12, .17),
    ellipse(133.68, 34.12, .10, .14),
    ellipse(133.88, 34.16, .08, .12),
    ellipse(134.08, 34.18, .09, .14),
    ellipse(134.34, 34.18, .07, .11),
    ellipse(134.78, 34.35, .16, .26),       # Awaji
    ellipse(134.98, 34.47, .08, .12),
    ellipse(135.12, 34.52, .06, .09),
    ellipse(135.42, 34.58, .06, .08),
    ellipse(136.72, 34.48, .09, .12),       # Ise Bay islands
    ellipse(136.95, 34.62, .05, .08),
    ellipse(131.10, 33.98, .08, .11),
    ellipse(131.34, 34.02, .07, .10),
    ellipse(131.58, 34.08, .06, .09),
    ellipse(131.82, 34.10, .08, .11),
]
# Smaller named/representative islands; deliberately coarse at this grid size.
for cx, cy, rx, ry in [
    (132.05,34.12,.055,.075),(132.18,34.16,.045,.07),(132.3,34.13,.05,.07),
    (132.55,34.14,.06,.09),(132.68,34.12,.05,.08),(132.88,34.14,.06,.09),
    (133.12,34.12,.05,.07),(133.36,34.12,.055,.08),(133.58,34.10,.05,.07),
    (133.78,34.13,.05,.07),(133.98,34.15,.045,.07),(134.18,34.15,.05,.08),
    (134.45,34.17,.04,.065),(134.58,34.19,.04,.06),
    (134.85,34.22,.045,.07),(134.93,34.29,.035,.055),
    (131.05,34.08,.06,.08),(131.25,34.12,.055,.075),(131.48,34.14,.05,.07),
    (131.7,34.12,.05,.07),(131.9,34.16,.055,.08),
    (132.02,33.98,.05,.07),(132.2,34.00,.04,.06),
    (134.45,34.58,.045,.06),(134.62,34.56,.04,.055),
]:
    ISLANDS.append(ellipse(cx, cy, rx, ry, 16))


def inside(lon, lat, polygon):
    """Even-odd point-in-polygon test; boundary is treated as land."""
    hit = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        cross = (yi > lat) != (yj > lat)
        if cross:
            edge_x = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < edge_x:
                hit = not hit
        dx, dy = xj - xi, yj - yi
        denom = dx * dx + dy * dy
        if denom:
            t = max(0.0, min(1.0, ((lon-xi)*dx+(lat-yi)*dy)/denom))
            if (lon-(xi+t*dx))**2 + (lat-(yi+t*dy))**2 < 1e-12:
                return True
        j = i
    return hit


def is_land(lon, lat):
    return int(any(inside(lon, lat, p) for p in LAND) or
               any(inside(lon, lat, p) for p in ISLANDS))
def main():
    rows, cols = 64, 253
    lat0, lon0 = 35.0, 129.0
    step = 2.0 / 63.0
    matrix = [
        [is_land(lon0 + c * step, lat0 - r * step) for c in range(cols)]
        for r in range(rows)
    ]
    assert len(matrix) == 64 and all(len(row) == 253 for row in matrix)
    assert sum(len(row) for row in matrix) == 16192
    assert all(v in (0, 1) for row in matrix for v in row)

    out = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, "candidate.csv")
    with open(csv_path, "w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix)

    # Use uint8 values directly: black water and white land.
    pixels = np.asarray(matrix, dtype=np.uint8) * 255
    image = Image.fromarray(pixels, mode="L")
    png_path = os.path.join(out, "candidate.png")
    image.save(png_path)
    enlarged = image.resize((253 * 8, 64 * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(os.path.join(out, "candidate_8x.png"))

    with open(csv_path, encoding="ascii", newline="") as f:
        loaded = list(csv.reader(f))
    assert len(loaded) == 64
    assert all(len(row) == 253 for row in loaded)
    assert all(cell in ("0", "1") for row in loaded for cell in row)
    assert image.size == (253, 64)
    assert enlarged.size == (2024, 512)
    assert set(image.getdata()) <= {0, 255}
    # Every source pixel must expand to a constant 8x8 block.
    check = np.asarray(enlarged)[::8, ::8]
    assert np.array_equal(check, pixels)
    assert np.array_equal(np.asarray(enlarged)[1::8, 1::8], pixels)
    print("validated: 64x253, 16192 values, CSV 0/1, PNG 253x64 and 2024x512")


if __name__ == "__main__":
    main()
