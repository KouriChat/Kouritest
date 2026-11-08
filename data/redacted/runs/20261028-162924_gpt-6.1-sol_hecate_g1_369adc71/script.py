from pathlib import Path
import csv

import numpy as np
from PIL import Image

SIZE = 127
LAT_NORTH, LAT_SOUTH = 55.0, 53.0
LON_WEST, LON_EAST = -131.0, -129.0
STEP = 2.0 / 126.0

# Hand-drawn approximate land polygons, in (longitude, latitude).
LAND_POLYGONS = [
    [
        (-129.00, 55.00), (-129.00, 53.00), (-129.55, 53.00),
        (-129.66, 53.08), (-129.60, 53.16), (-129.72, 53.23),
        (-129.67, 53.31), (-129.82, 53.39), (-129.75, 53.47),
        (-129.90, 53.55), (-129.83, 53.63), (-130.02, 53.70),
        (-129.93, 53.78), (-130.12, 53.87), (-130.06, 53.94),
        (-130.25, 54.02), (-130.15, 54.10), (-130.34, 54.17),
        (-130.28, 54.26), (-130.47, 54.33), (-130.39, 54.40),
        (-130.58, 54.47), (-130.52, 54.54), (-130.69, 54.61),
        (-130.61, 54.69), (-130.80, 54.76), (-130.72, 54.84),
        (-130.91, 54.92), (-130.84, 55.00),
    ],
    [
        (-130.33, 53.24), (-130.20, 53.27), (-130.10, 53.36),
        (-130.08, 53.48), (-130.14, 53.59), (-130.23, 53.68),
        (-130.36, 53.72), (-130.46, 53.67), (-130.50, 53.56),
        (-130.46, 53.44), (-130.41, 53.33),
    ],
    [
        (-130.52, 53.85), (-130.43, 53.91), (-130.35, 54.01),
        (-130.36, 54.12), (-130.44, 54.18), (-130.53, 54.13),
        (-130.58, 54.02), (-130.60, 53.92),
    ],
    [
        (-130.47, 53.72), (-130.37, 53.74), (-130.31, 53.80),
        (-130.34, 53.85), (-130.45, 53.84), (-130.51, 53.79),
    ],
    [
        (-130.18, 53.83), (-130.11, 53.86), (-130.06, 53.92),
        (-130.12, 53.97), (-130.20, 53.93),
    ],
    [
        (-130.05, 54.12), (-129.98, 54.16), (-129.93, 54.22),
        (-129.98, 54.27), (-130.07, 54.24), (-130.10, 54.18),
    ],
    [
        (-130.29, 54.31), (-130.22, 54.34), (-130.17, 54.40),
        (-130.23, 54.45), (-130.31, 54.41), (-130.34, 54.35),
    ],
    [
        (-130.09, 54.53), (-130.02, 54.56), (-129.99, 54.62),
        (-130.05, 54.67), (-130.13, 54.63), (-130.15, 54.57),
    ],
    [
        (-131.00, 54.55), (-130.94, 54.62), (-130.91, 54.73),
        (-130.94, 54.84), (-130.90, 54.95), (-130.87, 55.00),
        (-131.00, 55.00),
    ],
    [
        (-131.00, 53.00), (-130.94, 53.00), (-130.91, 53.10),
        (-130.94, 53.22), (-130.98, 53.34), (-131.00, 53.42),
    ],
]

def on_segment(x, y, x1, y1, x2, y2):
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > 1e-10:
        return False
    return (
        min(x1, x2) - 1e-10 <= x <= max(x1, x2) + 1e-10
        and min(y1, y2) - 1e-10 <= y <= max(y1, y2) + 1e-10
    )

def inside_polygon(lon, lat, polygon):
    inside = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        if on_segment(lon, lat, x1, y1, x2, y2):
            return True
        if (y1 > lat) != (y2 > lat):
            cross_lon = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < cross_lon:
                inside = not inside
    return inside

def is_land(lon, lat):
    return int(any(inside_polygon(lon, lat, p) for p in LAND_POLYGONS))

def main():
    lats = [LAT_NORTH - i * STEP for i in range(SIZE)]
    lons = [LON_WEST + j * STEP for j in range(SIZE)]
    matrix = np.array(
        [[is_land(lon, lat) for lon in lons] for lat in lats],
        dtype=np.uint8,
    )
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "candidate.csv").open(
        "w", newline="", encoding="ascii"
    ) as f:
        csv.writer(f, lineterminator="\n").writerows(matrix.tolist())
    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(output_dir / "candidate.png")
    image.resize((1016, 1016), Image.Resampling.NEAREST).save(
        output_dir / "candidate_8x.png"
    )
    assert matrix.shape == (127, 127)
    assert matrix.size == 16129
    assert set(np.unique(matrix).tolist()) <= {0, 1}
    with (output_dir / "candidate.csv").open(
        newline="", encoding="ascii"
    ) as f:
        rows = list(csv.reader(f))
    assert len(rows) == 127
    assert all(len(row) == 127 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    original = Image.open(output_dir / "candidate.png")
    enlarged = Image.open(output_dir / "candidate_8x.png")
    assert original.size == (127, 127)
    assert enlarged.size == (1016, 1016)
    assert set(original.getdata()) <= {(0, 0, 0), (255, 255, 255)}
    assert enlarged == original.resize((1016, 1016), Image.Resampling.NEAREST)

if __name__ == "__main__":
    main()
