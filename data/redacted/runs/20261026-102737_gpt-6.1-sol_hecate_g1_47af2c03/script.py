from pathlib import Path

import numpy as np
from PIL import Image

ROWS = COLS = 127
STEP = 2 / 126
OUT = Path("outputs")

# Hand-drawn approximations in (longitude, latitude); mainland lies east.
MAINLAND_COAST = [
    (-130.15, 55.05),
    (-130.34, 54.88),
    (-130.30, 54.73),
    (-130.12, 54.59),
    (-130.02, 54.43),
    (-129.94, 54.25),
    (-129.98, 54.08),
    (-130.10, 53.94),
    (-130.28, 53.82),
    (-130.40, 53.66),
    (-130.34, 53.50),
    (-130.17, 53.35),
    (-129.98, 53.20),
    (-129.82, 53.00),
]
MAINLAND = MAINLAND_COAST + [(-128.5, 53.0), (-128.5, 55.05)]

ISLANDS = [
    # Banks Island
    [
        (-130.18, 53.60), (-130.02, 53.61), (-129.93, 53.52),
        (-129.93, 53.39), (-129.98, 53.26), (-130.09, 53.21),
        (-130.17, 53.31), (-130.21, 53.47),
    ],
    # Additional small nearshore islands
    [
        (-130.40, 54.08), (-130.33, 54.10), (-130.28, 54.05),
        (-130.31, 53.99), (-130.38, 54.00),
    ],
    [
        (-130.59, 54.18), (-130.53, 54.21), (-130.47, 54.16),
        (-130.50, 54.10), (-130.57, 54.11),
    ],
    [
        (-130.26, 53.84), (-130.21, 53.87), (-130.16, 53.83),
        (-130.18, 53.78), (-130.24, 53.78),
    ],
    [
        (-130.66, 54.74), (-130.60, 54.77), (-130.55, 54.73),
        (-130.57, 54.68), (-130.63, 54.68),
    ],
    [
        (-130.45, 54.93), (-130.40, 54.96), (-130.35, 54.92),
        (-130.37, 54.87), (-130.43, 54.88),
    ],
]


def point_in_polygon(lon, lat, polygon):
    """Ray casting; points exactly on polygon edges are treated as land."""
    inside = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        cross = (lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
        if (
            abs(cross) < 1e-12
            and min(x1, x2) - 1e-12 <= lon <= max(x1, x2) + 1e-12
            and min(y1, y2) - 1e-12 <= lat <= max(y1, y2) + 1e-12
        ):
            return True
        if (y1 > lat) != (y2 > lat):
            if lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
    return inside


def is_land(lon, lat):
    return point_in_polygon(lon, lat, MAINLAND) or any(
        point_in_polygon(lon, lat, island) for island in ISLANDS
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    latitudes = 55.0 - np.arange(ROWS) * STEP
    longitudes = -131.0 + np.arange(COLS) * STEP
    matrix = np.zeros((ROWS, COLS), dtype=np.uint8)

    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            matrix[row, col] = int(is_land(float(lon), float(lat)))

    with (OUT / "candidate.csv").open("w", encoding="ascii", newline="") as f:
        for row in matrix:
            f.write(",".join(str(int(value)) for value in row) + "\n")

    image = Image.fromarray(matrix * 255, mode="L")
    image.save(OUT / "candidate.png")
    image.resize((1016, 1016), Image.Resampling.NEAREST).save(
        OUT / "candidate_8x.png"
    )

    assert matrix.shape == (127, 127) and matrix.size == 16129
    assert set(np.unique(matrix)).issubset({0, 1})
    lines = (OUT / "candidate.csv").read_text(encoding="ascii").splitlines()
    assert len(lines) == 127
    assert all(len(line.split(",")) == 127 for line in lines)
    assert all(value in {"0", "1"} for line in lines for value in line.split(","))
    original = Image.open(OUT / "candidate.png")
    enlarged = Image.open(OUT / "candidate_8x.png")
    assert original.size == (127, 127)
    assert set(original.getdata()).issubset({0, 255})
    assert enlarged.size == (1016, 1016)
    assert enlarged == original.resize((1016, 1016), Image.Resampling.NEAREST)


if __name__ == "__main__":
    main()
