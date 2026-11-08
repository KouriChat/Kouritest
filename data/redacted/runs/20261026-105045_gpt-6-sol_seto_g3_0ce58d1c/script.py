from pathlib import Path

import numpy as np
from PIL import Image


STEP = 2.0 / 63.0
ROWS = 64
COLS = 253


def polygon(points):
    return tuple((float(x), float(y)) for x, y in points)


# Approximate outlines written from geographic knowledge only.
LAND = [
    polygon([
        (129.34, 33.00), (130.30, 33.00), (130.58, 33.25),
        (130.72, 33.55), (130.93, 33.80), (131.25, 34.00),
        (131.50, 34.22), (131.42, 34.48), (131.19, 34.70),
        (130.88, 34.86), (130.55, 34.84), (130.24, 34.68),
        (129.96, 34.43), (129.72, 34.15), (129.48, 33.86),
    ]),
    polygon([
        (130.78, 35.00), (137.00, 35.00), (137.00, 34.72),
        (136.70, 34.65), (136.35, 34.53), (136.05, 34.42),
        (135.72, 34.38), (135.45, 34.44), (135.26, 34.58),
        (135.05, 34.64), (134.90, 34.55), (134.98, 34.42),
        (135.10, 34.31), (134.91, 34.24), (134.65, 34.35),
        (134.38, 34.47), (134.10, 34.53), (133.75, 34.48),
        (133.36, 34.39), (132.96, 34.31), (132.55, 34.24),
        (132.14, 34.16), (131.78, 34.10), (131.45, 34.03),
        (131.10, 34.03), (130.88, 34.20), (130.82, 34.55),
        (130.77, 34.82),
    ]),
    polygon([
        (131.98, 33.00), (134.90, 33.00), (134.78, 33.33),
        (134.59, 33.62), (134.30, 33.84), (133.93, 33.96),
        (133.52, 34.02), (133.10, 33.97), (132.72, 33.82),
        (132.36, 33.61), (132.10, 33.31),
    ]),
]


def on_edge(x, y, a, b):
    ax, ay = a
    bx, by = b
    cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
    return abs(cross) <= 1e-12 and min(ax, bx) - 1e-12 <= x <= max(ax, bx) + 1e-12 and min(ay, by) - 1e-12 <= y <= max(ay, by) + 1e-12
# Island outlines: Awaji, Shodo, Tsushima, Iki, and representative
# islands of the Geiyo chain and Seto Inland Sea.
LAND.extend([
    polygon([(134.70, 34.10), (134.85, 34.12), (134.94, 34.25),
             (134.91, 34.43), (134.81, 34.56), (134.73, 34.48)]),
    polygon([(134.10, 34.43), (134.23, 34.42), (134.31, 34.49),
             (134.25, 34.55), (134.12, 34.52)]),
    polygon([(129.17, 34.28), (129.40, 34.29), (129.49, 34.45),
             (129.43, 34.64), (129.30, 34.72), (129.16, 34.58)]),
    polygon([(129.62, 33.72), (129.84, 33.73), (129.90, 33.82),
             (129.82, 33.89), (129.63, 33.86)]),
])

for x, y, rx, ry in [
    (132.98, 34.15, .055, .035), (133.20, 34.16, .060, .035),
    (133.45, 34.18, .060, .035), (133.68, 34.19, .065, .038),
    (133.92, 34.21, .060, .035), (134.18, 34.22, .070, .040),
    (134.43, 34.24, .060, .035), (132.67, 34.12, .055, .035),
    (132.40, 34.11, .055, .035), (132.14, 34.08, .050, .030),
    (131.84, 34.04, .050, .030), (131.53, 34.00, .045, .030),
    (135.22, 34.20, .055, .032), (135.52, 34.21, .050, .030),
    (135.84, 34.24, .055, .032), (136.16, 34.27, .050, .030),
]:
    LAND.append(polygon([
        (x-rx, y), (x-rx*.65, y+ry*.8), (x, y+ry),
        (x+rx*.7, y+ry*.7), (x+rx, y), (x+rx*.65, y-ry*.7),
        (x, y-ry), (x-rx*.7, y-ry*.7),
    ]))

WATER = [
    polygon([(134.86, 34.24), (135.04, 34.23), (135.22, 34.29),
             (135.34, 34.40), (135.29, 34.52), (135.14, 34.59),
             (135.00, 34.55), (134.93, 34.43)]),
    polygon([(136.54, 34.42), (136.72, 34.48), (136.98, 34.53),
             (137.00, 34.35), (136.92, 34.20), (136.78, 34.14),
             (136.63, 34.24)]),
]
def inside(x, y, points):
    hit = False
    for i, (ax, ay) in enumerate(points):
        bx, by = points[(i + 1) % len(points)]
        if on_edge(x, y, (ax, ay), (bx, by)):
            return True
        if (ay > y) != (by > y):
            edge_x = ax + (y - ay) * (bx - ax) / (by - ay)
            if x < edge_x:
                hit = not hit
    return hit


def classify(lon, lat):
    land = any(inside(lon, lat, outline) for outline in LAND)
    if land and any(inside(lon, lat, outline) for outline in WATER):
        return 0
    return int(land)


def main():
    output = Path("outputs")
    output.mkdir(parents=True, exist_ok=True)
    matrix = np.zeros((ROWS, COLS), dtype=np.uint8)
    for row in range(ROWS):
        lat = 35.0 - row * STEP
        for col in range(COLS):
            lon = 129.0 + col * STEP
            matrix[row, col] = classify(lon, lat)

    with (output / "candidate.csv").open(
        "w", encoding="ascii", newline=""
    ) as stream:
        for row in matrix:
            stream.write(",".join(str(int(value)) for value in row) + "\n")

    image = Image.fromarray(matrix * 255, mode="L")
    image.save(output / "candidate.png")
    image.resize((2024, 512), Image.Resampling.NEAREST).save(
        output / "candidate_8x.png"
    )

    assert matrix.shape == (64, 253) and matrix.size == 16192
    assert set(np.unique(matrix)).issubset({0, 1})
    assert image.size == (253, 64)


if __name__ == "__main__":
    main()
