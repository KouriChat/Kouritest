from pathlib import Path
import csv

import numpy as np
from PIL import Image

SIZE = 127
STEP = 1.0 / 63.0

LAND = [
    [
        (-129.0, 55.0), (-129.0, 53.0), (-129.85, 53.0),
        (-129.95, 53.18), (-130.10, 53.32), (-130.08, 53.48),
        (-130.25, 53.62), (-130.20, 53.78), (-130.38, 53.92),
        (-130.35, 54.10), (-130.48, 54.26), (-130.42, 54.42),
        (-130.55, 54.56), (-130.50, 54.72), (-130.68, 54.86),
        (-130.62, 55.0),
    ],
    [
        (-129.98, 54.95), (-129.78, 54.93), (-129.72, 54.80),
        (-129.78, 54.66), (-129.92, 54.62), (-130.02, 54.73),
    ],
    [
        (-130.55, 53.98), (-130.35, 54.00), (-130.27, 53.91),
        (-130.35, 53.82), (-130.52, 53.84), (-130.60, 53.92),
    ],
    [
        (-129.78, 53.55), (-129.62, 53.57), (-129.55, 53.44),
        (-129.60, 53.28), (-129.72, 53.22), (-129.83, 53.36),
    ],
    [
        (-130.22, 54.35), (-130.16, 54.37), (-130.12, 54.32),
        (-130.17, 54.27), (-130.23, 54.29),
    ],
    [
        (-130.05, 54.10), (-129.98, 54.12), (-129.94, 54.06),
        (-130.00, 54.01), (-130.06, 54.04),
    ],
    [
        (-130.31, 53.70), (-130.24, 53.71), (-130.20, 53.65),
        (-130.25, 53.60), (-130.32, 53.63),
    ],
]
# Additional islands and exposed eastern tips of Haida Gwaii.
LAND.extend([
    [
        (-129.94, 53.82), (-129.88, 53.84), (-129.84, 53.78),
        (-129.89, 53.73), (-129.95, 53.76),
    ],
    [
        (-129.66, 53.82), (-129.59, 53.83), (-129.55, 53.77),
        (-129.61, 53.72), (-129.67, 53.76),
    ],
    [
        (-131.0, 54.25), (-130.96, 54.24), (-130.96, 54.02),
        (-131.0, 53.94), (-131.0, 54.10),
    ],
    [
        (-131.0, 53.58), (-130.96, 53.56), (-130.96, 53.22),
        (-131.0, 53.18),
    ],
    [
        (-130.98, 53.88), (-130.95, 53.88), (-130.95, 53.85),
        (-130.98, 53.85),
    ],
    [
        (-130.98, 53.05), (-130.95, 53.05), (-130.95, 53.02),
        (-130.98, 53.02),
    ],
])

# Approximate major mainland inlets as water cuts within the land outline.
WATER_CUTS = [
    [
        (-130.28, 55.0), (-130.20, 55.0), (-130.25, 54.78),
        (-130.31, 54.70), (-130.34, 54.80),
    ],
    [
        (-130.08, 54.57), (-130.00, 54.58), (-130.04, 54.34),
        (-130.12, 54.28), (-130.13, 54.45),
    ],
    [
        (-129.89, 54.15), (-129.82, 54.17), (-129.85, 53.96),
        (-129.93, 53.91), (-129.94, 54.05),
    ],
    [
        (-130.00, 53.76), (-129.93, 53.77), (-129.97, 53.58),
        (-130.04, 53.55), (-130.05, 53.68),
    ],
]
def inside_polygon(lon, lat, polygon):
    """Ray-crossing point-in-polygon test."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > lat) != (yj > lat):
            crossing_lon = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing_lon:
                inside = not inside
        j = i
    return inside


def is_land(lon, lat):
    on_land = any(inside_polygon(lon, lat, polygon) for polygon in LAND)
    if on_land and any(inside_polygon(lon, lat, polygon)
                       for polygon in WATER_CUTS):
        return 0
    return int(on_land)


def main():
    matrix = []
    for row in range(SIZE):
        lat = 55.0 - row * STEP
        matrix.append([
            is_land(-131.0 + col * STEP, lat)
            for col in range(SIZE)
        ])

    array = np.asarray(matrix, dtype=np.uint8)
    assert array.shape == (127, 127)
    assert set(np.unique(array)).issubset({0, 1})

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)

    with (out / "candidate.csv").open(
        "w", newline="", encoding="ascii"
    ) as stream:
        csv.writer(stream, lineterminator="\n").writerows(matrix)

    image = Image.new("RGB", (SIZE, SIZE))
    image.putdata([
        (255, 255, 255) if value else (0, 0, 0)
        for row in matrix for value in row
    ])
    image.save(out / "candidate.png")

    enlarged = image.resize(
        (SIZE * 8, SIZE * 8), Image.Resampling.NEAREST
    )
    enlarged.save(out / "candidate_8x.png")
    # Validate the written CSV and image properties.
    with (out / "candidate.csv").open(
        newline="", encoding="ascii"
    ) as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 127
    assert all(len(row) == 127 for row in rows)
    assert all(value in ("0", "1") for row in rows for value in row)
    assert sum(len(row) for row in rows) == 16129

    original = Image.open(out / "candidate.png").convert("RGB")
    assert original.size == (127, 127)
    assert set(original.getdata()) <= {(0, 0, 0), (255, 255, 255)}

    enlarged = Image.open(out / "candidate_8x.png").convert("RGB")
    assert enlarged.size == (1016, 1016)
    for y in range(127):
        for x in range(127):
            color = original.getpixel((x, y))
            assert all(
                enlarged.getpixel((x * 8 + dx, y * 8 + dy)) == color
                for dy in range(8) for dx in range(8)
            )


if __name__ == "__main__":
    main()
