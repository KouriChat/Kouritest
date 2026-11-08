from pathlib import Path
import csv
import numpy as np
from PIL import Image

# Hand-drawn WGS84 outlines, represented as (longitude, latitude) vertices.
# These are intentionally approximate and are not sourced from map data.
LAND = [
    # Western Honshu / Chugoku, including the Seto-facing shoreline.
    [(129,35.2),(134.0,35.2),(134.25,34.95),(134.45,34.82),(134.75,34.78),
     (135.0,34.82),(135.25,34.77),(135.45,34.72),(135.65,34.67),(135.78,34.62),
     (135.95,34.62),(136.08,34.68),(136.22,34.72),(136.42,34.76),(136.58,34.82),
     (136.8,34.9),(137,35.05),(137,36),(129,36)],
    # Shikoku, with its Inland Sea coast on the north and Pacific coast south.
    [(132.15,34.28),(132.45,34.42),(132.8,34.42),(133.1,34.38),(133.45,34.42),
     (133.75,34.38),(134.05,34.4),(134.35,34.38),(134.65,34.28),(134.82,34.12),
     (134.75,33.92),(134.55,33.75),(134.3,33.55),(134.05,33.38),(133.8,33.25),
     (133.5,33.2),(133.15,33.22),(132.85,33.28),(132.55,33.35),(132.32,33.52),
     (132.2,33.78),(132.1,34.02)],
    # Eastern Kyushu and the north-east coast of Kyushu.
    [(129.0,33.85),(129.4,34.0),(129.85,33.95),(130.15,33.83),(130.4,33.68),
     (130.63,33.55),(130.85,33.42),(131.05,33.38),(131.24,33.42),(131.48,33.5),
     (131.58,33.72),(131.5,34.0),(131.45,34.25),(131.38,34.45),(131.1,34.65),
     (130.75,34.78),(130.3,34.8),(129.9,34.7),(129.5,34.45),(129,34.2)],
    # Kii Peninsula and the southern side of Osaka Bay.
    [(135.0,34.55),(135.25,34.58),(135.48,34.5),(135.65,34.38),(135.72,34.2),
     (135.65,33.98),(135.5,33.75),(135.35,33.55),(135.18,33.42),(135.02,33.48),
     (134.98,33.72),(135.02,34.0),(134.95,34.25)],
    # Ise-Shima / eastern Kii margin and Chubu south coast.
    [(136.45,34.9),(136.65,34.86),(136.85,34.75),(137.1,34.72),(137.1,33.7),
     (136.9,33.68),(136.72,33.78),(136.55,33.95),(136.42,34.15),(136.35,34.4)],
    # Awaji Island.
    [(134.78,34.68),(134.92,34.67),(134.98,34.53),(134.97,34.32),
     (134.9,34.22),(134.82,34.3),(134.77,34.48)],
    # Yashiro and Shodo islands.
    [(134.15,34.62),(134.28,34.63),(134.35,34.53),(134.3,34.43),(134.18,34.45)],
    [(134.0,34.58),(134.12,34.59),(134.15,34.49),(134.08,34.43),(133.99,34.48)],
]

# Island groups in the Geiyo chain and other prominent Inland Sea islands.
ISLANDS = [
    (132.8,34.25,132.98,34.39),(132.98,34.18,133.14,34.31),
    (133.1,34.14,133.27,34.27),(133.28,34.12,133.46,34.27),
    (133.48,34.08,133.65,34.23),(133.68,34.1,133.84,34.25),
    (133.88,34.1,134.0,34.2),(132.45,34.12,132.63,34.23),
    (132.62,34.2,132.8,34.32),(131.05,33.9,131.25,34.0),
    (131.45,33.95,131.58,34.05),(134.38,34.22,134.48,34.33),
]

WATER = [
    [(134.65,34.75),(135.1,34.76),(135.35,34.7),(135.42,34.57),(135.25,34.48),
     (135.02,34.46),(134.9,34.55)],
    [(134.35,34.68),(134.7,34.72),(134.9,34.64),(134.8,34.5),(134.52,34.48)],
    [(132.2,34.45),(132.6,34.52),(132.9,34.49),(133.0,34.39),(132.8,34.34),
     (132.5,34.34),(132.3,34.38)],
    [(134.85,34.35),(135.05,34.32),(135.12,34.12),(135.08,33.95),
     (134.85,34.0),(134.78,34.16)],
    [(136.45,34.75),(136.72,34.78),(136.92,34.68),(137.02,34.48),(136.98,34.24),
     (136.82,34.12),(136.62,34.18),(136.48,34.35),(136.42,34.55)],
    [(136.58,34.85),(136.8,34.82),(136.95,34.72),(136.85,34.58),(136.63,34.55)],
]

def inside(lon, lat, polygon):
    hit = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > lat) != (yj > lat):
            if lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
                hit = not hit
        j = i
    return hit

def is_land(lon, lat):
    land = any(inside(lon, lat, p) for p in LAND)
    if any(inside(lon, lat, p) for p in WATER):
        land = False
    if any(x0 <= lon <= x1 and y0 <= lat <= y1 for x0, y0, x1, y1 in ISLANDS):
        land = True
    return int(land)

def main():
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    matrix = np.array([[is_land(129 + c * 2 / 63, 35 - r * 2 / 63)
                        for c in range(253)] for r in range(64)], dtype=np.uint8)
    with (out / "candidate.csv").open("w", newline="") as f:
        csv.writer(f).writerows(matrix.tolist())
    Image.fromarray(matrix * 255, mode="L").convert("RGB").save(out / "candidate.png")
    image = Image.open(out / "candidate.png")
    image.resize((2024, 512), Image.Resampling.NEAREST).save(out / "candidate_8x.png")
    assert matrix.shape == (64, 253) and matrix.size == 16192
    assert set(np.unique(matrix)) <= {0, 1}
    with (out / "candidate.csv").open(newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 64 and all(len(row) == 253 and set(row) <= {"0", "1"} for row in rows)
    assert image.size == (253, 64)
    enlarged = Image.open(out / "candidate_8x.png")
    assert enlarged.size == (2024, 512) and len(enlarged.getcolors(2024 * 512)) == 2

if __name__ == "__main__":
    main()
