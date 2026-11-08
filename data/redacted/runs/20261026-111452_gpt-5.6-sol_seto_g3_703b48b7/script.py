from pathlib import Path
import csv
import numpy as np
from PIL import Image

# Hand-authored approximate land outlines in longitude/latitude.
LAND = [
    # Northern Kyushu and western Kyushu
    [(129.0,33.0),(129.0,34.0),(129.45,33.98),(129.75,33.86),
     (130.0,33.82),(130.23,33.88),(130.45,33.93),(130.75,33.91),
     (130.95,33.98),(131.08,33.93),(131.05,33.65),(130.85,33.45),
     (130.55,33.32),(130.25,33.28),(129.95,33.22),(129.7,33.08),
     (129.4,33.0)],
    # Honshu: southern shore of Chugoku, Kinki and Ise region, then north edge
    [(130.82,34.03),(131.05,34.10),(131.28,34.08),(131.45,34.18),
     (131.68,34.20),(131.83,34.25),(132.03,34.22),(132.22,34.25),
     (132.40,34.20),(132.60,34.22),(132.80,34.28),(133.02,34.30),
     (133.18,34.34),(133.38,34.32),(133.55,34.38),(133.78,34.35),
     (133.98,34.42),(134.18,34.38),(134.35,34.46),(134.55,34.48),
     (134.72,34.55),(134.90,34.58),(135.05,34.62),(135.22,34.68),
     (135.35,34.66),(135.48,34.70),(135.58,34.72),(135.68,34.68),
     (135.78,34.70),(135.90,34.72),(136.02,34.73),(136.10,34.80),
     (136.20,34.86),(136.30,34.88),(136.38,34.84),(136.48,34.82),
     (136.58,34.85),(136.68,34.83),(136.78,34.78),(136.88,34.72),
     (136.98,34.68),(137.0,35.0),(130.82,35.0)],
    # Shikoku
    [(132.00,33.95),(132.18,34.00),(132.38,33.99),(132.55,34.02),
     (132.75,34.03),(132.95,34.05),(133.15,34.04),(133.35,34.08),
     (133.55,34.08),(133.78,34.10),(133.98,34.12),(134.12,34.10),
     (134.25,34.05),(134.38,33.99),(134.48,33.91),(134.55,33.82),
     (134.52,33.72),(134.42,33.68),(134.30,33.70),(134.18,33.74),
     (134.04,33.73),(133.90,33.72),(133.75,33.70),(133.58,33.68),
     (133.42,33.66),(133.25,33.65),(133.08,33.67),(132.90,33.68),
     (132.72,33.67),(132.55,33.70),(132.38,33.72),(132.20,33.78),
     (132.05,33.84)],
]
# Small islands and island groups; each tuple is center lon/lat and radii.
ISLANDS = [
    (129.30,34.12,.10,.25),  # Tsushima
    (129.72,33.86,.08,.09),  # Iki
    (130.15,33.78,.08,.10),  # western Suo-nada islets
    (131.05,34.02,.05,.06),
    # Geiyo islands between Hiroshima and Shikoku
    (132.42,34.18,.035,.045),(132.52,34.17,.04,.04),
    (132.62,34.18,.035,.05),(132.72,34.19,.035,.04),
    (132.82,34.21,.04,.045),(132.92,34.23,.035,.04),
    (133.02,34.25,.035,.035),(133.12,34.27,.04,.035),
    (133.22,34.28,.035,.03),(133.32,34.29,.035,.03),
    (133.42,34.30,.03,.03),(133.52,34.31,.035,.03),
    (133.62,34.32,.03,.03),(133.72,34.33,.035,.03),
    (133.82,34.34,.035,.03),
    # Inland sea islands and peninsulas
    (134.05,34.22,.07,.05),(134.22,34.28,.035,.04),
    (134.35,34.32,.04,.035),(134.45,34.34,.03,.03),
    (134.68,34.48,.12,.07),  # Awaji
    (134.30,33.93,.04,.045),(134.42,33.87,.03,.035),
    (134.62,34.26,.035,.035),(134.78,34.31,.03,.03),
    (133.9,34.15,.025,.03),(133.7,34.18,.03,.025),
    (132.0,34.05,.03,.035),(131.7,34.02,.025,.03),
    # Ise Bay islands and offshore islands
    (136.72,34.62,.025,.03),(136.82,34.58,.025,.03),
    (136.55,34.34,.06,.06),(136.95,34.30,.05,.05),
    (135.0,34.18,.045,.05),(135.28,34.14,.03,.035),
    # Southern islands
    (130.55,33.15,.12,.07),(130.25,32.75,.10,.13),
    (130.65,32.55,.08,.12),(130.85,32.20,.10,.17),
    (130.55,31.55,.18,.30),(130.25,31.00,.13,.22),
]


def inside_polygon(lon, lat, polygon):
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > lat) != (yj > lat):
            crossing = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing:
                inside = not inside
        j = i
    return inside


def is_land(lon, lat):
    if inside_polygon(lon, lat, LAND[0]) or inside_polygon(lon, lat, LAND[1]) \
            or inside_polygon(lon, lat, LAND[2]):
        return True
    # Approximate southern Korea coast clipped at the western map edge.
    if 129.0 <= lon <= 129.20 and lat > 34.94 + (lon - 129.0) * .22:
        return True
    for cx, cy, rx, ry in ISLANDS:
        if ((lon - cx) / rx) ** 2 + ((lat - cy) / ry) ** 2 <= 1:
            return True
    return False
def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    rows = 64
    cols = 253
    matrix = np.zeros((rows, cols), dtype=np.uint8)
    for r in range(rows):
        lat = 35.0 - r * (2.0 / 63.0)
        for c in range(cols):
            lon = 129.0 + c * (8.0 / 252.0)
            matrix[r, c] = int(is_land(lon, lat))

    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    image = Image.fromarray(matrix * 255, mode="L")
    image.save(out / "candidate.png")
    image.resize((cols * 8, rows * 8), Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )

    with (out / "candidate.csv").open(newline="", encoding="ascii") as f:
        parsed = list(csv.reader(f))
    assert len(parsed) == rows and all(len(row) == cols for row in parsed)
    assert all(value in {"0", "1"} for row in parsed for value in row)
    assert sum(map(len, parsed)) == 16192
    assert image.size == (253, 64)
    assert set(image.getdata()) <= {0, 255}
    enlarged = Image.open(out / "candidate_8x.png")
    assert enlarged.size == (2024, 512)
    assert set(enlarged.getdata()) <= {0, 255}
    assert enlarged.resize(image.size, Image.Resampling.NEAREST).tobytes() == image.tobytes()
    print("validated: 64x253 CSV, 253x64 PNG, 2024x512 nearest-neighbor PNG")


if __name__ == "__main__":
    main()
