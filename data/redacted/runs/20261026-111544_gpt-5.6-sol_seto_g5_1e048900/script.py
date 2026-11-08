from pathlib import Path
import csv

import numpy as np
from PIL import Image


# Hand-drawn WGS84 coastline approximations. Each ring is (lon, lat).
# Land polygons are deliberately coarse; no geographic data is accessed.
LAND = [
    # Kyushu, with its north and west coasts
    [(129.55, 33.0), (129.66, 33.15), (129.72, 33.35), (129.78, 33.55),
     (129.82, 33.75), (129.92, 33.91), (130.05, 34.02), (130.20, 34.10),
     (130.35, 34.18), (130.49, 34.23), (130.62, 34.32), (130.73, 34.43),
     (130.81, 34.53), (130.85, 34.68), (130.92, 34.80), (131.02, 34.89),
     (131.12, 35.0), (129.0, 35.0), (129.0, 33.0)],
    # Honshu southern coast, Chugoku through Ise and the Pacific-facing edge
    [(130.75, 34.22), (130.90, 34.28), (131.08, 34.27), (131.28, 34.24),
     (131.48, 34.20), (131.68, 34.19), (131.88, 34.20), (132.08, 34.19),
     (132.28, 34.17), (132.48, 34.15), (132.68, 34.14), (132.88, 34.13),
     (133.08, 34.12), (133.28, 34.12), (133.48, 34.13), (133.68, 34.14),
     (133.88, 34.17), (134.08, 34.19), (134.28, 34.20), (134.48, 34.23),
     (134.68, 34.27), (134.88, 34.29), (135.08, 34.34), (135.28, 34.39),
     (135.48, 34.43), (135.65, 34.48), (135.77, 34.55), (135.84, 34.62),
     (135.91, 34.69), (136.00, 34.76), (136.12, 34.81), (136.24, 34.85),
     # Ise Bay indentation and Chita peninsula
     (136.34, 34.84), (136.43, 34.79), (136.49, 34.72), (136.54, 34.64),
     (136.60, 34.58), (136.66, 34.57), (136.71, 34.62), (136.75, 34.70),
     (136.79, 34.79), (136.86, 34.84), (137.0, 34.88),
     (137.0, 35.0), (130.75, 35.0)],
    # Shikoku
    [(132.00, 33.35), (132.18, 33.48), (132.38, 33.56), (132.58, 33.60),
     (132.78, 33.63), (132.98, 33.66), (133.18, 33.68), (133.38, 33.68),
     (133.58, 33.68), (133.78, 33.69), (133.98, 33.71), (134.18, 33.73),
     (134.38, 33.77), (134.56, 33.82), (134.72, 33.87), (134.78, 34.02),
     (134.70, 34.13), (134.51, 34.16), (134.30, 34.14), (134.08, 34.11),
     (133.86, 34.08), (133.64, 34.07), (133.42, 34.07), (133.20, 34.08),
     (132.98, 34.09), (132.76, 34.09), (132.54, 34.08), (132.33, 34.03),
     (132.14, 33.95), (132.00, 33.83), (131.95, 33.65)],
    # Awaji Island
    [(134.82, 34.15), (134.90, 34.18), (134.94, 34.28), (134.93, 34.42),
     (134.90, 34.55), (134.85, 34.63), (134.80, 34.57), (134.78, 34.43),
     (134.78, 34.28)],
]
# Larger islands and island groups inside the Seto Inland Sea.
LAND += [
    # Geiyo island chain and nearby islands
    [(132.67, 34.17), (132.77, 34.19), (132.81, 34.24), (132.77, 34.29),
     (132.70, 34.28), (132.65, 34.23)],
    [(132.88, 34.18), (133.00, 34.20), (133.04, 34.27), (132.99, 34.33),
     (132.91, 34.30), (132.86, 34.24)],
    [(133.12, 34.17), (133.25, 34.19), (133.28, 34.27), (133.22, 34.32),
     (133.13, 34.28), (133.09, 34.22)],
    [(133.39, 34.17), (133.53, 34.19), (133.57, 34.26), (133.51, 34.33),
     (133.42, 34.31), (133.36, 34.23)],
    [(133.74, 34.18), (133.90, 34.20), (133.96, 34.27), (133.91, 34.34),
     (133.80, 34.31), (133.72, 34.24)],
    [(134.10, 34.20), (134.25, 34.22), (134.31, 34.29), (134.26, 34.35),
     (134.15, 34.33), (134.08, 34.26)],
    [(131.65, 34.12), (131.78, 34.14), (131.82, 34.20), (131.77, 34.26),
     (131.68, 34.23), (131.62, 34.17)],
    # Shodo, Naoshima, Teshima, Inujima, and islands near Hiroshima
    [(134.02, 34.40), (134.15, 34.40), (134.26, 34.45), (134.30, 34.51),
     (134.24, 34.56), (134.12, 34.55), (134.03, 34.49)],
    [(133.95, 34.44), (134.00, 34.44), (134.02, 34.48), (133.99, 34.51),
     (133.95, 34.49)],
    [(134.04, 34.48), (134.09, 34.48), (134.11, 34.52), (134.08, 34.55),
     (134.04, 34.53)],
    [(134.17, 34.55), (134.23, 34.55), (134.25, 34.59), (134.22, 34.62),
     (134.18, 34.60)],
    [(132.10, 34.18), (132.19, 34.18), (132.23, 34.23), (132.18, 34.28),
     (132.11, 34.25)],
    [(132.30, 34.17), (132.37, 34.17), (132.40, 34.22), (132.36, 34.26),
     (132.30, 34.23)],
    [(132.45, 34.18), (132.52, 34.19), (132.55, 34.23), (132.51, 34.27),
     (132.46, 34.24)],
    # Kii peninsula tip and small islands along its approaches
    [(135.05, 33.65), (135.28, 33.71), (135.50, 33.82), (135.69, 33.98),
     (135.82, 34.15), (135.83, 34.30), (135.73, 34.40), (135.56, 34.43),
     (135.38, 34.38), (135.22, 34.29), (135.10, 34.14), (135.02, 33.95)],
    # Tsushima
    [(129.18, 34.10), (129.30, 34.12), (129.42, 34.24), (129.48, 34.42),
     (129.46, 34.62), (129.38, 34.76), (129.26, 34.77), (129.17, 34.63),
     (129.14, 34.42)],
]


def inside(lon, lat, ring):
    """Even-odd point-in-polygon test; boundary samples are classed as land."""
    hit = False
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        cross = (yi > lat) != (yj > lat)
        if cross and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            hit = not hit
        j = i
    return hit
def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    rows, cols = 64, 253
    latitudes = [35.0 - i * (2.0 / 63.0) for i in range(rows)]
    longitudes = [129.0 + j * (8.0 / 252.0) for j in range(cols)]

    matrix = np.zeros((rows, cols), dtype=np.uint8)
    for r, lat in enumerate(latitudes):
        for c, lon in enumerate(longitudes):
            matrix[r, c] = any(inside(lon, lat, ring) for ring in LAND)

    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

    # 0 -> black water, 1 -> white land. No interpolation or smoothing.
    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(out / "candidate.png")
    enlarged = image.resize((cols * 8, rows * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(out / "candidate_8x.png")

    with (out / "candidate.csv").open(newline="", encoding="ascii") as f:
        loaded = list(csv.reader(f))
    assert len(loaded) == rows and all(len(row) == cols for row in loaded)
    assert all(value in {"0", "1"} for row in loaded for value in row)
    assert sum(map(len, loaded)) == 16192
    assert image.size == (253, 64)
    assert enlarged.size == (2024, 512)
    assert set(image.getdata()) <= {(0, 0, 0), (255, 255, 255)}
    print("Validated 64x253 matrix, 16192 binary values, and both PNG dimensions.")


if __name__ == "__main__":
    main()
