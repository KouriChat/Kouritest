import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image


# Hand-drawn WGS84 coastline traces, in (longitude, latitude) order.
# Polygon edges are intentionally approximate; no external geography is used.
LAND = [
    # Western Honshu: Sea of Japan coast around the top, Seto coast below.
    [(130.80, 34.02), (130.78, 34.45), (130.88, 34.82), (131.05, 35.15),
     (131.35, 35.45), (131.80, 35.62), (132.20, 35.67), (132.55, 35.64),
     (132.88, 35.72), (133.20, 35.80), (133.55, 35.85), (133.85, 35.73),
     (134.20, 35.72), (134.55, 35.62), (134.86, 35.65), (135.15, 35.55),
     (135.50, 35.52), (135.82, 35.38), (136.08, 35.28), (136.40, 35.24),
     (136.72, 35.20), (137.10, 35.15), (137.10, 36.20), (130.80, 36.20)],
    # Shikoku, with the north shore facing the Inland Sea.
    [(132.00, 33.20), (132.10, 33.48), (132.18, 33.67), (132.35, 33.78),
     (132.62, 33.83), (132.92, 33.86), (133.20, 33.88), (133.50, 33.85),
     (133.82, 33.88), (134.08, 33.90), (134.34, 33.88), (134.58, 33.82),
     (134.70, 33.62), (134.56, 33.40), (134.34, 33.30), (134.05, 33.25),
     (133.74, 33.22), (133.42, 33.20), (133.12, 33.18), (132.82, 33.15),
     (132.50, 33.13), (132.20, 33.14)],
    # Eastern Kyushu, including the Beppu-Oita side.
    [(130.72, 32.70), (130.90, 33.02), (131.20, 33.15), (131.45, 33.25),
     (131.68, 33.32), (131.82, 33.45), (131.87, 33.62), (131.82, 33.78),
     (131.65, 33.91), (131.37, 33.98), (131.10, 34.03), (130.88, 34.10),
     (130.67, 34.18), (130.40, 34.18), (130.15, 34.12), (129.95, 34.02),
     (129.75, 33.90), (129.55, 33.78), (129.30, 33.62), (129.10, 33.42),
     (128.90, 33.20), (128.90, 32.70)],
    # Awaji Island.
    [(134.67, 34.12), (134.78, 34.22), (134.86, 34.43), (134.91, 34.62),
     (134.86, 34.72), (134.76, 34.68), (134.69, 34.49), (134.64, 34.29)],
    # Kii Peninsula fills south of Osaka Bay and east toward Ise.
    [(134.80, 34.58), (135.03, 34.68), (135.25, 34.73), (135.48, 34.75),
     (135.68, 34.72), (135.84, 34.65), (135.93, 34.52), (135.92, 34.36),
     (135.83, 34.18), (135.76, 33.98), (135.67, 33.77), (135.55, 33.56),
     (135.39, 33.40), (135.20, 33.35), (135.01, 33.43), (134.88, 33.61),
     (134.82, 33.83), (134.79, 34.07), (134.76, 34.34)],
    # Chita Peninsula and the eastern shore of Ise Bay.
    [(136.58, 34.82), (136.77, 34.83), (136.90, 34.75), (136.88, 34.59),
     (136.81, 34.43), (136.72, 34.33), (136.62, 34.40), (136.58, 34.58)],
    # Atsumi Peninsula.
    [(136.60, 34.62), (136.82, 34.70), (137.10, 34.68), (137.10, 34.38),
     (136.91, 34.40), (136.72, 34.48)],
]

LAND.extend([
    # Hiroshima-Yamaguchi coastal land, continuing into western Honshu.
    [(130.80, 34.02), (131.05, 34.08), (131.30, 34.08), (131.56, 34.12),
     (131.82, 34.16), (132.05, 34.20), (132.28, 34.26), (132.52, 34.30),
     (132.78, 34.33), (133.02, 34.37), (133.25, 34.42), (133.48, 34.48),
     (133.72, 34.55), (133.98, 34.61), (134.22, 34.67), (134.45, 34.73),
     (134.66, 34.80), (134.88, 34.91), (135.10, 35.02), (135.35, 35.10),
     (135.60, 35.13), (135.84, 35.12), (136.08, 35.10), (136.34, 35.08),
     (136.62, 35.10), (136.92, 35.14), (137.10, 35.15), (137.10, 36.20),
     (130.80, 36.20)],
])

# Islands represented separately so that their pixels survive narrow channels.
ISLANDS = [
    # Suo-Oshima, Miyajima, Etajima, Nomi, Kurahashi and nearby Hiroshima islands.
    [(132.10, 33.95), (132.24, 34.00), (132.31, 34.08), (132.25, 34.18),
     (132.12, 34.21), (132.02, 34.13), (132.02, 34.02)],
    [(132.30, 34.16), (132.39, 34.20), (132.43, 34.30), (132.37, 34.36),
     (132.27, 34.31), (132.25, 34.22)],
    [(132.43, 34.08), (132.55, 34.12), (132.63, 34.22), (132.60, 34.34),
     (132.50, 34.40), (132.40, 34.32), (132.38, 34.20)],
    [(132.58, 34.02), (132.75, 34.02), (132.83, 34.10), (132.78, 34.19),
     (132.66, 34.21), (132.57, 34.13)],
    [(132.82, 34.10), (132.94, 34.13), (133.00, 34.22), (132.94, 34.30),
     (132.84, 34.27), (132.78, 34.18)],
    # Geiyo Islands around Imabari and the central Inland Sea.
    [(132.94, 34.03), (133.10, 34.02), (133.20, 34.10), (133.17, 34.22),
     (133.05, 34.27), (132.94, 34.20)],
    [(133.20, 34.07), (133.38, 34.07), (133.48, 34.16), (133.45, 34.29),
     (133.33, 34.35), (133.22, 34.28)],
    [(133.52, 34.10), (133.67, 34.10), (133.75, 34.19), (133.70, 34.29),
     (133.57, 34.29), (133.50, 34.19)],
    [(133.78, 34.17), (133.94, 34.17), (134.04, 34.25), (134.02, 34.36),
     (133.88, 34.38), (133.78, 34.30)],
    [(134.10, 34.22), (134.25, 34.21), (134.35, 34.28), (134.33, 34.39),
     (134.20, 34.42), (134.10, 34.34)],
    # Shodoshima and Naoshima area.
    [(134.00, 34.43), (134.23, 34.43), (134.38, 34.52), (134.38, 34.66),
     (134.24, 34.74), (134.05, 34.69), (133.96, 34.57)],
    [(133.72, 34.42), (133.84, 34.42), (133.89, 34.50), (133.82, 34.57),
     (133.72, 34.53)],
    [(134.48, 34.45), (134.60, 34.47), (134.66, 34.55), (134.60, 34.64),
     (134.49, 34.61)],
    # Islands north of Tokushima / Awaji and the Osaka Bay islands.
    [(134.48, 34.75), (134.60, 34.76), (134.66, 34.83), (134.61, 34.91),
     (134.50, 34.90), (134.44, 34.82)],
    [(135.02, 34.53), (135.10, 34.55), (135.14, 34.62), (135.08, 34.68),
     (135.00, 34.64)],
    # Ise Bay and Mikawa Bay islands.
    [(136.68, 34.55), (136.77, 34.57), (136.81, 34.65), (136.76, 34.72),
     (136.68, 34.68)],
    [(136.86, 34.65), (136.95, 34.67), (137.02, 34.73), (137.00, 34.81),
     (136.90, 34.80), (136.84, 34.73)],
]


def inside(point, polygon):
    x, y = point
    hit = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            crossing = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing:
                hit = not hit
        previous = current
    return hit


def is_land(lon, lat):
    point = (lon, lat)
    return any(inside(point, polygon) for polygon in LAND + ISLANDS)


def main():
    output = Path("outputs")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in range(64):
        lat = 35.0 - row * (2.0 / 63.0)
        rows.append([
            int(is_land(129.0 + col * (8.0 / 252.0), lat))
            for col in range(253)
        ])

    assert len(rows) == 64 and all(len(row) == 253 for row in rows)
    assert sum(map(len, rows)) == 16192
    with (output / "candidate.csv").open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(rows)

    pixels = np.where(np.asarray(rows, dtype=np.uint8) == 1, 255, 0).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L").convert("1")
    image.save(output / "candidate.png")
    enlarged = image.resize((2024, 512), resample=Image.Resampling.NEAREST)
    enlarged.save(output / "candidate_8x.png")

    with (output / "candidate.csv").open("r", newline="", encoding="ascii") as stream:
        check = list(csv.reader(stream))
    assert len(check) == 64
    assert all(len(row) == 253 and set(row) <= {"0", "1"} for row in check)
    assert image.size == (253, 64)
    assert set(image.getdata()) <= {0, 255}
    assert enlarged.size == (2024, 512)
    for y in range(64):
        for x in range(253):
            assert all(enlarged.getpixel((x * 8 + dx, y * 8 + dy)) ==
                       image.getpixel((x, y)) for dy in range(8) for dx in range(8))


if __name__ == "__main__":
    main()
