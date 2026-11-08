from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image

LAT_N = 64
LON_N = 253
LAT0, LON0 = 35.0, 129.0
STEP = 2.0 / 63.0


def inside(x, y, poly):
    hit = False
    j = len(poly) - 1
    for i, (px, py) in enumerate(poly):
        qx, qy = poly[j]
        crosses = ((py > y) != (qy > y))
        if crosses and x < (qx - px) * (y - py) / (qy - py) + px:
            hit = not hit
        j = i
    return hit


def ellipse(x, y, cx, cy, rx, ry):
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


# Hand-drawn coast approximations.  Coordinates are longitude, latitude.
POLYGONS = [
    # western and central Honshu, including the Chugoku and Kansai uplands
    [(130.88, 35.0), (137.0, 35.0), (137.0, 34.78),
     (136.55, 34.76), (136.15, 34.82), (135.85, 34.68),
     (135.55, 34.58), (135.28, 34.62), (134.9, 34.75),
     (134.35, 34.70), (133.85, 34.58), (133.3, 34.45),
     (132.8, 34.35), (132.4, 34.28), (131.95, 34.22),
     (131.5, 34.12), (131.15, 34.02), (130.88, 34.02)],

    # Mie and the Kii peninsula, on the Pacific side of the Ise/Osaka bays
    [(135.25, 34.35), (135.55, 34.32), (135.9, 34.25),
     (136.28, 34.18), (136.65, 34.02), (136.85, 33.75),
     (137.0, 33.58), (137.0, 33.0), (136.55, 33.0),
     (136.25, 33.18), (135.95, 33.43), (135.72, 33.62),
     (135.47, 33.82), (135.28, 34.08)],

    # Chita peninsula and the eastern shore around Nagoya
    [(136.45, 35.0), (137.0, 35.0), (137.0, 34.72),
     (136.88, 34.57), (136.7, 34.48), (136.55, 34.55),
     (136.45, 34.72)],

    # Kyushu
    [(129.88, 33.98), (130.35, 34.02), (130.92, 33.93),
     (131.35, 33.72), (131.66, 33.38), (132.0, 33.08),
     (132.0, 33.0), (129.65, 33.0), (129.55, 33.27),
     (129.62, 33.58), (129.72, 33.83)],

    # Shikoku
    [(132.0, 34.18), (132.38, 34.28), (132.85, 34.31),
     (133.35, 34.28), (133.88, 34.23), (134.38, 34.12),
     (134.72, 33.92), (134.7, 33.55), (134.48, 33.2),
     (134.12, 32.95), (133.68, 32.78), (133.18, 32.72),
     (132.72, 32.78), (132.3, 32.98), (132.08, 33.35)],
]


def base_land(lon, lat):
    return any(inside(lon, lat, poly) for poly in POLYGONS)


def island_land(lon, lat):
    # Tsushima, Iki, Hirado and the nearby northwestern Kyushu islands.
    if inside(lon, lat, [(129.16, 34.75), (129.43, 34.7),
                         (129.52, 34.4), (129.43, 34.12),
                         (129.2, 34.08), (129.1, 34.4)]):
        return True
    if ellipse(lon, lat, 129.72, 33.78, 0.17, 0.15):
        return True
    if ellipse(lon, lat, 129.47, 33.36, 0.12, 0.17):
        return True
    if ellipse(lon, lat, 129.6, 33.08, 0.08, 0.12):
        return True

    # Awaji and the larger islands in the eastern Inland Sea.
    if inside(lon, lat, [(134.65, 34.62), (134.95, 34.55),
                         (135.08, 34.22), (134.9, 34.02),
                         (134.62, 34.08), (134.48, 34.35)]):
        return True
    if ellipse(lon, lat, 134.3, 34.52, 0.16, 0.12):
        return True  # Shodoshima
    if ellipse(lon, lat, 134.0, 34.45, 0.08, 0.07):
        return True  # Teshima/Naoshima group
    return detailed_island_land(lon, lat)

def detailed_island_land(lon, lat):
    # Hiroshima Bay and the western Geiyo islands.
    if ellipse(lon, lat, 132.35, 34.30, 0.13, 0.12):
        return True
    if inside(lon, lat, [(132.35, 34.38), (132.62, 34.31),
                         (132.65, 34.08), (132.42, 34.03),
                         (132.28, 34.17)]):
        return True
    if ellipse(lon, lat, 132.54, 34.12, 0.13, 0.10):
        return True
    if ellipse(lon, lat, 132.82, 34.25, 0.12, 0.10):
        return True
    if ellipse(lon, lat, 132.94, 34.20, 0.08, 0.08):
        return True

    # The islands along the Shimanami Kaido and the central Geiyo group.
    if ellipse(lon, lat, 133.02, 34.27, 0.13, 0.12):
        return True
    if ellipse(lon, lat, 133.10, 34.31, 0.14, 0.10):
        return True
    if ellipse(lon, lat, 133.18, 34.32, 0.12, 0.11):
        return True
    if ellipse(lon, lat, 133.28, 34.25, 0.08, 0.08):
        return True
    if ellipse(lon, lat, 133.19, 34.10, 0.14, 0.11):
        return True
    if ellipse(lon, lat, 133.42, 34.16, 0.07, 0.06):
        return True

    # Kagawa-side islands and the central Inland Sea.
    if ellipse(lon, lat, 133.68, 34.34, 0.12, 0.08):
        return True
    if ellipse(lon, lat, 133.86, 34.35, 0.09, 0.07):
        return True
    if ellipse(lon, lat, 134.02, 34.39, 0.08, 0.07):
        return True
    if ellipse(lon, lat, 134.14, 34.39, 0.06, 0.055):
        return True
    if ellipse(lon, lat, 134.08, 34.48, 0.07, 0.065):
        return True
    if ellipse(lon, lat, 134.02, 34.53, 0.06, 0.06):
        return True
    if ellipse(lon, lat, 133.55, 34.27, 0.06, 0.06):
        return True

    # Islands at the western mouth and southern edge of the Inland Sea.
    if ellipse(lon, lat, 131.78, 33.98, 0.09, 0.06):
        return True
    if ellipse(lon, lat, 132.05, 33.92, 0.07, 0.06):
        return True
    if ellipse(lon, lat, 132.75, 33.83, 0.08, 0.07):
        return True
    if ellipse(lon, lat, 133.15, 33.78, 0.06, 0.05):
        return True
    if ellipse(lon, lat, 133.72, 33.92, 0.07, 0.05):
        return True
    if ellipse(lon, lat, 134.38, 34.00, 0.07, 0.05):
        return True

    # A few sizeable islands at the Kii and Ise bay approaches.
    if ellipse(lon, lat, 135.08, 34.25, 0.08, 0.06):
        return True
    if ellipse(lon, lat, 136.72, 34.27, 0.06, 0.05):
        return True
    return False
def main():
    matrix = np.zeros((LAT_N, LON_N), dtype=np.uint8)
    for row in range(LAT_N):
        lat = LAT0 - row * STEP
        for col in range(LON_N):
            lon = LON0 + col * STEP
            matrix[row, col] = int(
                base_land(lon, lat) or island_land(lon, lat)
            )

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    image = Image.fromarray(matrix * 255, mode="L")
    image.save(out / "candidate.png")
    enlarged = image.resize((LON_N * 8, LAT_N * 8), Image.Resampling.NEAREST)
    enlarged.save(out / "candidate_8x.png")

    assert matrix.shape == (64, 253)
    assert int(matrix.size) == 16192
    assert set(np.unique(matrix).tolist()) <= {0, 1}
    with (out / "candidate.csv").open(newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 64 and all(
        len(r) == 253 and all(v in ("0", "1") for v in r) for r in rows
    )
    assert image.size == (253, 64)
    assert set(image.getdata()) == {0, 255}
    assert enlarged.size == (2024, 512)
    assert set(enlarged.getdata()) == {0, 255}
    print("validated: 64x253 / 16192 values; binary CSV and images")


if __name__ == "__main__":
    main()
