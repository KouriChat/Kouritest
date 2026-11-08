from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 127
OUT = Path("outputs")

# Hand-drawn approximate polygons in (longitude, latitude).
MAINLAND = [
    (-129.58, 55.00), (-129.60, 54.88), (-129.68, 54.80),
    (-129.63, 54.70), (-129.72, 54.61), (-129.61, 54.48),
    (-129.68, 54.37), (-129.57, 54.25), (-129.66, 54.12),
    (-129.56, 54.00), (-129.61, 53.88), (-129.52, 53.78),
    (-129.58, 53.66), (-129.49, 53.55), (-129.55, 53.43),
    (-129.43, 53.32), (-129.52, 53.20), (-129.42, 53.08),
    (-129.45, 53.00), (-128.00, 53.00), (-128.00, 55.00),
]

ISLANDS = [
    [
        (-129.52, 54.30), (-129.42, 54.35), (-129.30, 54.32),
        (-129.19, 54.25), (-129.08, 54.13), (-129.03, 54.00),
        (-129.04, 53.88), (-129.10, 53.79), (-129.18, 53.76),
        (-129.23, 53.83), (-129.29, 53.86), (-129.36, 53.91),
        (-129.43, 53.96), (-129.48, 54.04), (-129.55, 54.13),
    ],
    [
        (-129.93, 54.40), (-129.82, 54.43), (-129.75, 54.38),
        (-129.78, 54.30), (-129.88, 54.28), (-129.97, 54.32),
    ],
    [
        (-129.84, 54.20), (-129.76, 54.23), (-129.70, 54.18),
        (-129.72, 54.10), (-129.80, 54.08), (-129.87, 54.12),
    ],
    [
        (-129.94, 53.98), (-129.87, 54.00), (-129.83, 53.94),
        (-129.86, 53.88), (-129.94, 53.88), (-129.99, 53.93),
    ],
    [
        (-129.70, 53.60), (-129.61, 53.64), (-129.53, 53.60),
        (-129.55, 53.52), (-129.64, 53.50), (-129.71, 53.54),
    ],
    [
        (-129.88, 53.44), (-129.79, 53.48), (-129.72, 53.43),
        (-129.75, 53.35), (-129.84, 53.34), (-129.91, 53.38),
    ],
]

INLETS = [
    [
        (-129.60, 55.05), (-129.57, 54.96), (-129.68, 54.88),
        (-129.80, 54.80), (-129.94, 54.74), (-130.08, 54.72),
        (-130.08, 54.68), (-129.91, 54.69), (-129.75, 54.75),
        (-129.62, 54.83), (-129.56, 54.92),
    ],
    [
        (-129.65, 54.63), (-129.76, 54.59), (-129.87, 54.53),
        (-129.97, 54.46), (-130.02, 54.39), (-129.98, 54.36),
        (-129.89, 54.41), (-129.79, 54.49), (-129.68, 54.56),
    ],
    [
        (-129.62, 54.24), (-129.75, 54.20), (-129.87, 54.15),
        (-130.00, 54.09), (-130.08, 54.02), (-130.05, 53.99),
        (-129.93, 54.02), (-129.81, 54.09), (-129.69, 54.15),
    ],
    [
        (-129.56, 53.92), (-129.68, 53.88), (-129.81, 53.83),
        (-129.94, 53.77), (-130.02, 53.70), (-129.99, 53.67),
        (-129.87, 53.70), (-129.74, 53.77), (-129.62, 53.83),
    ],
]


def on_segment(x, y, a, b):
    ax, ay = a
    bx, by = b
    cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
    if abs(cross) > 1e-12:
        return False
    return (
        min(ax, bx) - 1e-12 <= x <= max(ax, bx) + 1e-12
        and min(ay, by) - 1e-12 <= y <= max(ay, by) + 1e-12
    )


def inside_polygon(x, y, polygon):
    inside = False
    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]
        if on_segment(x, y, a, b):
            return True
        ax, ay = a
        bx, by = b
        if (ay > y) != (by > y):
            crossing_x = ax + (y - ay) * (bx - ax) / (by - ay)
            if x < crossing_x:
                inside = not inside
    return inside


def is_land(lon, lat):
    if inside_polygon(lon, lat, MAINLAND):
        if any(inside_polygon(lon, lat, inlet) for inlet in INLETS):
            return 0
        return 1
    return int(any(inside_polygon(lon, lat, island) for island in ISLANDS))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    matrix = np.zeros((SIZE, SIZE), dtype=np.uint8)
    for row in range(SIZE):
        lat = 55.0 - 2.0 * row / 126.0
        for col in range(SIZE):
            lon = -131.0 + 2.0 * col / 126.0
            matrix[row, col] = is_land(lon, lat)

    with (OUT / "candidate.csv").open(
        "w", encoding="ascii", newline=""
    ) as f:
        for row in matrix:
            f.write(",".join(str(int(value)) for value in row) + "\n")

    image = Image.fromarray(matrix * 255, mode="L")
    image.save(OUT / "candidate.png")
    image.resize((1016, 1016), Image.Resampling.NEAREST).save(
        OUT / "candidate_8x.png"
    )

    assert matrix.shape == (127, 127)
    assert matrix.size == 16129
    assert set(np.unique(matrix)) <= {0, 1}
    with (OUT / "candidate.csv").open("r", encoding="ascii") as f:
        rows = [line.rstrip("\n").split(",") for line in f]
    assert len(rows) == 127
    assert all(len(row) == 127 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert Image.open(OUT / "candidate.png").size == (127, 127)
    enlarged = Image.open(OUT / "candidate_8x.png")
    assert enlarged.size == (1016, 1016)
    assert set(enlarged.getdata()) <= {0, 255}


if __name__ == "__main__":
    main()
