from pathlib import Path
import csv
import numpy as np
from PIL import Image

N = 127
LAT_TOP = 55.0
LON_LEFT = -131.0
STEP = 2.0 / 126.0

# Hand-authored coarse land boundaries, in longitude/latitude.
MAINLAND = [
    (-131.0, 55.0), (-130.65, 55.0), (-130.46, 54.94),
    (-130.33, 54.85), (-130.25, 54.73), (-130.22, 54.60),
    (-130.14, 54.48), (-130.07, 54.35), (-129.98, 54.23),
    (-129.91, 54.10), (-129.83, 53.98), (-129.78, 53.84),
    (-129.70, 53.72), (-129.64, 53.58), (-129.57, 53.43),
    (-129.51, 53.28), (-129.43, 53.13), (-129.37, 53.0),
    (-131.0, 53.0),
]
GRAHAM = [
    (-132.0, 54.25), (-131.75, 54.32), (-131.48, 54.37),
    (-131.22, 54.42), (-130.97, 54.45), (-130.78, 54.41),
    (-130.66, 54.34), (-130.61, 54.24), (-130.66, 54.14),
    (-130.77, 54.06), (-130.95, 54.00), (-131.15, 53.95),
    (-131.38, 53.91), (-131.62, 53.94), (-131.83, 54.02),
    (-132.0, 54.12),
]
MORESBY = [
    (-131.65, 53.93), (-131.47, 53.90), (-131.29, 53.85),
    (-131.13, 53.78), (-130.98, 53.68), (-130.88, 53.57),
    (-130.78, 53.45), (-130.68, 53.32), (-130.60, 53.18),
    (-130.54, 53.0), (-131.0, 53.0), (-131.12, 53.15),
    (-131.22, 53.30), (-131.34, 53.44), (-131.47, 53.58),
    (-131.58, 53.73), (-131.72, 53.84),
]
ISLANDS = [
    [(-129.77, 54.86), (-129.53, 54.92), (-129.28, 54.90),
     (-129.08, 54.80), (-129.0, 54.65), (-129.0, 54.39),
     (-129.20, 54.38), (-129.42, 54.44), (-129.61, 54.57),
     (-129.76, 54.70)],
    [(-130.42, 53.96), (-130.23, 54.02), (-130.05, 53.98),
     (-129.92, 53.88), (-129.94, 53.75), (-130.11, 53.71),
     (-130.30, 53.76), (-130.43, 53.85)],
    [(-130.90, 54.30), (-130.78, 54.34), (-130.66, 54.31),
     (-130.61, 54.23), (-130.68, 54.17), (-130.82, 54.19)],
    [(-130.80, 53.43), (-130.64, 53.49), (-130.50, 53.44),
     (-130.43, 53.31), (-130.51, 53.18), (-130.67, 53.15),
     (-130.80, 53.25)],
    [(-130.12, 53.58), (-129.97, 53.63), (-129.83, 53.57),
     (-129.81, 53.44), (-129.91, 53.36), (-130.06, 53.40)],
    [(-129.95, 53.95), (-129.79, 54.02), (-129.67, 53.96),
     (-129.63, 53.84), (-129.73, 53.75), (-129.88, 53.80)],
    [(-129.53, 54.27), (-129.37, 54.33), (-129.22, 54.26),
     (-129.19, 54.13), (-129.29, 54.06), (-129.45, 54.11)],
    [(-129.88, 53.22), (-129.70, 53.29), (-129.56, 53.22),
     (-129.53, 53.08), (-129.64, 53.0), (-129.84, 53.04)],
    [(-130.55, 54.57), (-130.47, 54.61), (-130.40, 54.57),
     (-130.42, 54.50), (-130.51, 54.50)],
    [(-130.32, 54.75), (-130.24, 54.79), (-130.17, 54.74),
     (-130.19, 54.67), (-130.27, 54.68)],
    [(-130.63, 53.66), (-130.54, 53.70), (-130.47, 53.65),
     (-130.49, 53.58), (-130.58, 53.59)],
    [(-129.32, 53.70), (-129.22, 53.74), (-129.15, 53.68),
     (-129.19, 53.60), (-129.29, 53.62)],
]
REEFS = [
    [(-130.06, 54.45), (-130.02, 54.47), (-129.99, 54.44),
     (-130.02, 54.41), (-130.06, 54.42)],
    [(-130.30, 54.12), (-130.26, 54.14), (-130.22, 54.11),
     (-130.25, 54.08), (-130.29, 54.09)],
    [(-129.48, 53.53), (-129.43, 53.55), (-129.39, 53.52),
     (-129.42, 53.49), (-129.47, 53.50)],
]

def point_on_segment(x, y, a, b, eps=1e-12):
    ax, ay = a
    bx, by = b
    cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
    return (abs(cross) <= eps and min(ax, bx) - eps <= x <= max(ax, bx) + eps
            and min(ay, by) - eps <= y <= max(ay, by) + eps)

def inside_polygon(x, y, polygon):
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if point_on_segment(x, y, previous, current):
            return True
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside

def is_land(lon, lat):
    if any(inside_polygon(lon, lat, p)
           for p in (MAINLAND, GRAHAM, MORESBY)):
        return True
    return any(inside_polygon(lon, lat, p) for p in ISLANDS + REEFS)

def build_matrix():
    matrix = np.zeros((N, N), dtype=np.uint8)
    for row in range(N):
        lat = LAT_TOP - row * STEP
        for col in range(N):
            lon = LON_LEFT + col * STEP
            matrix[row, col] = int(is_land(lon, lat))
    return matrix

def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()
    assert matrix.shape == (127, 127) and matrix.size == 16129
    assert set(np.unique(matrix).tolist()) <= {0, 1}

    csv_path = out / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(matrix.tolist())
    with csv_path.open("r", newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 127 and all(len(row) == 127 for row in rows)
    assert all(v in {"0", "1"} for row in rows for v in row)

    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(out / "candidate.png")
    image.resize((1016, 1016), Image.Resampling.NEAREST).save(
        out / "candidate_8x.png")
    original = Image.open(out / "candidate.png")
    enlarged = Image.open(out / "candidate_8x.png")
    assert original.size == (127, 127)
    assert enlarged.size == (1016, 1016)
    assert set(original.getdata()) <= {(0, 0, 0), (255, 255, 255)}
    assert set(enlarged.getdata()) <= {(0, 0, 0), (255, 255, 255)}
    print("Validated matrix, CSV and both PNG outputs.")

if __name__ == "__main__":
    main()
