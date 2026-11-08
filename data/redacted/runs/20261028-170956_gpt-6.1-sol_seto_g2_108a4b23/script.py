import csv
from pathlib import Path

import numpy as np
from PIL import Image

STEP = 2 / 63

# Coarse, hand-authored coastline vertices (longitude, latitude). These broad
# outlines prioritize the major islands and bays; small islands are explicit.
LAND = [
    # Kyushu
    [(129.3,33.1),(129.5,33.0),(130.3,33.0),(131.1,33.2),(131.8,32.8),
     (131.9,33.5),(131.3,34.0),(130.8,34.4),(130.2,34.5),(129.8,34.2),
     (129.5,33.8),(129.3,33.1)],
    # Shikoku
    [(132.0,33.3),(132.7,33.1),(133.5,33.2),(134.4,33.4),(134.8,33.8),
     (134.6,34.1),(133.7,34.2),(132.8,34.1),(132.1,33.9),(132.0,33.3)],
    # Honshu, west-to-east southern coastline, then northern return
    [(130.9,34.0),(131.4,33.9),(132.0,34.0),(132.3,34.2),(132.8,34.3),
     (133.3,34.3),(133.8,34.4),(134.2,34.4),(134.5,34.5),(134.8,34.5),
     (135.1,34.6),(135.4,34.6),(135.7,34.7),(136.0,34.8),(136.3,34.9),
     (136.6,35.0),(137.1,35.0),(137.1,36.0),(130.9,36.0),(130.9,34.0)],
    # Korean peninsula edge at far west
    [(129.0,34.8),(129.0,36.0),(129.5,36.0),(129.6,35.4),(129.3,35.0),
     (129.0,34.8)],
    # Awaji Island
    [(134.7,34.2),(135.0,34.2),(135.1,34.5),(134.8,34.6),(134.7,34.2)],
    # Large islands and peninsulas in the Seto Inland Sea
    [(132.2,34.1),(132.5,34.1),(132.6,34.3),(132.3,34.4),(132.2,34.1)],
    [(133.0,34.0),(133.4,34.0),(133.6,34.2),(133.2,34.3),(133.0,34.0)],
    [(133.7,34.0),(134.0,34.0),(134.1,34.2),(133.8,34.2),(133.7,34.0)],
    [(134.0,33.9),(134.4,33.9),(134.5,34.1),(134.1,34.2),(134.0,33.9)],
    # Ise peninsula / eastern Honshu coastal detail
    [(136.0,34.1),(136.4,34.0),(136.8,34.2),(137.0,34.6),(136.7,34.9),
     (136.2,34.8),(136.0,34.1)],
]

# Approximate island groups in the Geiyo archipelago and nearby straits.
ISLANDS = [
    (132.65,34.20,.10,.055),(132.77,34.22,.08,.05),(132.88,34.20,.09,.05),
    (133.00,34.19,.07,.045),(133.12,34.18,.08,.05),(133.24,34.19,.10,.05),
    (133.35,34.20,.08,.05),(133.48,34.21,.08,.05),(133.60,34.22,.07,.045),
    (133.72,34.20,.08,.05),(133.84,34.18,.09,.05),(133.96,34.16,.07,.045),
    (134.08,34.13,.08,.05),(134.20,34.11,.07,.045),(134.32,34.08,.06,.04),
    (132.45,34.27,.07,.045),(132.90,34.32,.08,.05),(133.60,34.31,.07,.045),
    (134.20,34.28,.06,.04),(131.95,33.95,.08,.05),(132.15,33.98,.07,.045),
]

def inside(lon, lat, polygon):
    hit = False
    j = len(polygon) - 1
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[j]
        if (y1 > lat) != (y2 > lat):
            cross = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
            if lon < cross:
                hit = not hit
        j = i
    return hit


def is_land(lon, lat):
    if any(inside(lon, lat, poly) for poly in LAND):
        return True
    # Elliptical hand-drawn islands; not a geographic data lookup.
    return any(((lon - x) / rx) ** 2 + ((lat - y) / ry) ** 2 <= 1
               for x, y, rx, ry in ISLANDS)


def main():
    lats = [35.0 - r * STEP for r in range(64)]
    lons = [129.0 + c * STEP for c in range(253)]
    matrix = np.array([[int(is_land(lon, lat)) for lon in lons] for lat in lats],
                      dtype=np.uint8)
    assert matrix.shape == (64, 253) and matrix.size == 16192
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(matrix.tolist())
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(out / "candidate.png")
    image.resize((2024, 512), resample=Image.Resampling.NEAREST).save(
        out / "candidate_8x.png")


if __name__ == "__main__":
    main()
