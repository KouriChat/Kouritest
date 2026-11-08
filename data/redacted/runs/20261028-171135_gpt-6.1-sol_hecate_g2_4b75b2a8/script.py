from pathlib import Path

import numpy as np
from PIL import Image


ROWS = COLS = 127
STEP = 2 / 126
NORTH, WEST = 55.0, -131.0


def polygon(x, y, points):
    """Ray-cast a geographic point against a manually sketched polygon."""
    inside = False
    j = len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        if (yi > y) != (yj > y):
            crossing = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < crossing:
                inside = not inside
        j = i
    return inside


# Approximate mainland outline, traced from south to north. The polygon's
# remaining edge follows the eastern map boundary, which is inland here.
MAINLAND_COAST = [
    (-129.12, 53.00), (-129.16, 53.10), (-129.23, 53.18),
    (-129.30, 53.26), (-129.38, 53.34), (-129.43, 53.43),
    (-129.50, 53.50), (-129.59, 53.57), (-129.64, 53.64),
    (-129.70, 53.72), (-129.75, 53.80), (-129.81, 53.87),
    (-129.87, 53.94), (-129.91, 54.02), (-129.97, 54.09),
    (-130.02, 54.17), (-130.08, 54.23), (-130.14, 54.29),
    (-130.19, 54.34), (-130.26, 54.39), (-130.31, 54.44),
    (-130.37, 54.48), (-130.43, 54.51), (-130.47, 54.56),
    (-130.51, 54.60), (-130.55, 54.65), (-130.59, 54.70),
    (-130.63, 54.75), (-130.68, 54.81), (-130.73, 54.86),
    (-130.77, 54.91), (-130.81, 54.96), (-130.84, 55.00),
]
MAINLAND = MAINLAND_COAST + [(-129.0, 55.0), (-129.0, 53.0)]

# Large islands and coastal island groups within the sampling frame.
ISLANDS = [
    # Graham Island's eastern fringe, mostly beyond the western frame edge.
    [(-131.00, 54.12), (-130.98, 54.17), (-131.00, 54.22),
     (-131.00, 54.27), (-131.00, 54.31), (-131.00, 54.34),
     (-131.00, 54.38), (-131.00, 54.42), (-131.00, 54.45),
     (-131.00, 54.48), (-131.00, 54.52), (-131.00, 54.54),
     (-131.00, 54.50), (-130.99, 54.46), (-130.98, 54.42),
     (-130.98, 54.38), (-130.97, 54.34), (-130.97, 54.30),
     (-130.97, 54.26), (-130.97, 54.22), (-130.98, 54.18)],
    # Banks Island.
    [(-129.82, 54.78), (-129.72, 54.76), (-129.62, 54.78),
     (-129.55, 54.83), (-129.51, 54.91), (-129.52, 55.00),
     (-129.83, 55.00), (-129.86, 54.93), (-129.85, 54.86)],
    # Campania Island.
    [(-129.82, 53.28), (-129.74, 53.27), (-129.68, 53.30),
     (-129.65, 53.36), (-129.68, 53.43), (-129.73, 53.45),
     (-129.78, 53.41), (-129.82, 53.35)],
    # Aristazabal Island.
    [(-129.83, 53.48), (-129.75, 53.47), (-129.68, 53.52),
     (-129.65, 53.60), (-129.67, 53.69), (-129.72, 53.74),
     (-129.79, 53.70), (-129.84, 53.62), (-129.87, 53.55)],
    # Banks / Pitt / McCauley archipelago details.
    [(-129.91, 53.77), (-129.84, 53.75), (-129.79, 53.79),
     (-129.78, 53.84), (-129.82, 53.89), (-129.88, 53.87)],
    [(-129.96, 53.94), (-129.90, 53.92), (-129.85, 53.96),
     (-129.84, 54.02), (-129.88, 54.06), (-129.94, 54.04)],
    [(-130.02, 54.10), (-129.96, 54.08), (-129.90, 54.11),
     (-129.88, 54.16), (-129.92, 54.21), (-129.99, 54.20)],
    # Dundas Island and nearby islands at the north-western entrance.
    [(-130.35, 54.53), (-130.28, 54.50), (-130.23, 54.53),
     (-130.23, 54.59), (-130.27, 54.62), (-130.33, 54.59)],
    [(-130.16, 54.58), (-130.12, 54.56), (-130.08, 54.59),
     (-130.09, 54.63), (-130.14, 54.64)],
    # Porcher Island.
    [(-130.42, 53.84), (-130.34, 53.81), (-130.27, 53.84),
     (-130.24, 53.91), (-130.25, 53.99), (-130.30, 54.05),
     (-130.37, 54.03), (-130.42, 53.96), (-130.45, 53.90)],
    # Banks' northern neighbors and scattered outer-coast islands.
    [(-130.07, 54.49), (-130.02, 54.46), (-129.98, 54.49),
     (-129.99, 54.54), (-130.04, 54.56)],
    [(-129.96, 54.40), (-129.91, 54.38), (-129.87, 54.42),
     (-129.88, 54.47), (-129.93, 54.48)],
    [(-130.18, 53.63), (-130.12, 53.60), (-130.08, 53.64),
     (-130.09, 53.69), (-130.15, 53.70)],
    [(-130.31, 53.50), (-130.27, 53.48), (-130.23, 53.51),
     (-130.24, 53.55), (-130.29, 53.56)],
]

# Hand-sketched inlet polygons are water even where they cross the mainland
# polygon. Their coordinates represent fjord mouths and lower channels.
INLETS = [
    [(-129.72, 53.63), (-129.55, 53.67), (-129.42, 53.72),
     (-129.38, 53.77), (-129.48, 53.79), (-129.61, 53.74)],
    [(-129.88, 53.95), (-129.68, 54.00), (-129.52, 54.05),
     (-129.47, 54.10), (-129.59, 54.12), (-129.78, 54.07)],
    [(-130.03, 54.18), (-129.84, 54.21), (-129.67, 54.25),
     (-129.62, 54.30), (-129.74, 54.32), (-129.94, 54.27)],
    [(-130.47, 54.53), (-130.29, 54.55), (-130.13, 54.60),
     (-130.08, 54.65), (-130.21, 54.67), (-130.39, 54.62)],
]


def classify(lon, lat):
    land = polygon(lon, lat, MAINLAND)
    if any(polygon(lon, lat, island) for island in ISLANDS):
        land = True
    if any(polygon(lon, lat, inlet) for inlet in INLETS):
        land = False
    return int(land)


def main():
    matrix = np.zeros((ROWS, COLS), dtype=np.uint8)
    for row in range(ROWS):
        lat = NORTH - row * STEP
        for col in range(COLS):
            lon = WEST + col * STEP
            matrix[row, col] = classify(lon, lat)

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        for row in matrix:
            f.write(",".join(str(int(value)) for value in row) + "\n")

    image = Image.fromarray(matrix * 255, mode="L")
    image.save(out / "candidate.png")
    image.resize((COLS * 8, ROWS * 8), Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )

    assert matrix.shape == (127, 127) and matrix.size == 16129
    assert set(np.unique(matrix)).issubset({0, 1})
    assert Image.open(out / "candidate.png").size == (127, 127)
    enlarged = Image.open(out / "candidate_8x.png")
    assert enlarged.size == (1016, 1016)
    assert set(enlarged.getdata()).issubset({0, 255})
    print(f"validated {matrix.size} points; land={int(matrix.sum())}")


if __name__ == "__main__":
    main()
