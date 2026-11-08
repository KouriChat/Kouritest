from pathlib import Path
import csv
import numpy as np
from PIL import Image


N = 127
STEP = 2.0 / 126.0


def inside_polygon(lon, lat, vertices):
    """Ray-casting point test; vertices are (longitude, latitude)."""
    inside = False
    j = len(vertices) - 1
    for i, (xi, yi) in enumerate(vertices):
        xj, yj = vertices[j]
        if (yi > lat) != (yj > lat):
            cross = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < cross:
                inside = not inside
        j = i
    return inside


# Approximate west-facing mainland coast, from the Portland Canal area south
# toward the Skeena coast. Land is east of this hand-drawn boundary.
MAINLAND = [
    (-130.62, 55.00), (-130.48, 54.91), (-130.34, 54.82),
    (-130.22, 54.72), (-130.14, 54.62), (-130.02, 54.54),
    (-129.98, 54.43), (-129.91, 54.33), (-129.96, 54.22),
    (-130.04, 54.11), (-130.02, 54.02), (-129.91, 53.91),
    (-129.83, 53.80), (-129.74, 53.69), (-129.64, 53.58),
    (-129.57, 53.47), (-129.47, 53.36), (-129.38, 53.25),
    (-129.27, 53.14), (-129.16, 53.03), (-129.11, 53.00),
    (-129.00, 53.00), (-129.00, 55.00),
]

# Hand-drawn island outlines. These are intentionally approximate and are
# clipped naturally by the requested sampling rectangle.
ISLANDS = [
    # Graham Island: north and east shore, including the Dixon Entrance coast.
    [(-131.13, 54.25), (-131.04, 54.20), (-130.98, 54.12),
     (-131.01, 54.03), (-131.08, 53.96), (-131.12, 53.88),
     (-131.10, 53.79), (-131.14, 53.70), (-131.19, 53.62),
     (-131.20, 53.53), (-131.17, 53.43), (-131.12, 53.36),
     (-131.10, 53.30), (-131.28, 53.30), (-131.32, 53.40),
     (-131.33, 53.55), (-131.31, 53.72), (-131.27, 53.89),
     (-131.24, 54.05), (-131.20, 54.17)],
    # Moresby Island, whose east coast is near the western edge of the box.
    [(-131.04, 53.20), (-130.98, 53.13), (-130.97, 53.04),
     (-131.02, 53.00), (-131.17, 53.00), (-131.20, 53.07),
     (-131.16, 53.15), (-131.12, 53.20)],
    # Banks Island in the central Hecate Strait.
    [(-130.38, 53.92), (-130.30, 53.95), (-130.21, 53.90),
     (-130.14, 53.81), (-130.11, 53.70), (-130.15, 53.59),
     (-130.22, 53.52), (-130.31, 53.54), (-130.38, 53.63),
     (-130.42, 53.75), (-130.42, 53.85)],
    # Pitt Island and adjacent larger coastal islands.
    [(-129.82, 53.43), (-129.73, 53.47), (-129.66, 53.41),
     (-129.62, 53.31), (-129.65, 53.20), (-129.72, 53.13),
     (-129.80, 53.17), (-129.84, 53.28)],
    [(-129.98, 53.54), (-129.91, 53.57), (-129.86, 53.51),
     (-129.88, 53.43), (-129.94, 53.40), (-130.00, 53.45)],
    [(-130.28, 54.12), (-130.22, 54.14), (-130.18, 54.09),
     (-130.20, 54.03), (-130.26, 54.01), (-130.30, 54.06)],
]


def is_land(lon, lat):
    if inside_polygon(lon, lat, MAINLAND):
        return True
    return any(inside_polygon(lon, lat, shape) for shape in ISLANDS)
def main():
    matrix = np.zeros((N, N), dtype=np.uint8)
    for row in range(N):
        lat = 55.0 - row * STEP
        for col in range(N):
            lon = -131.0 + col * STEP
            matrix[row, col] = int(is_land(lon, lat))

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)

    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # 1 (land) is white; 0 (water) is black.
    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(out / "candidate.png")
    image.resize((N * 8, N * 8), resample=Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )

    with (out / "candidate.csv").open("r", newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == N and all(len(row) == N for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert matrix.size == 16129
    assert set(np.unique(matrix)).issubset({0, 1})
    assert image.size == (127, 127)
    assert set(image.getdata()).issubset({(0, 0, 0), (255, 255, 255)})

    enlarged = Image.open(out / "candidate_8x.png")
    assert enlarged.size == (1016, 1016)
    for y in range(N):
        for x in range(N):
            pixel = image.getpixel((x, y))
            assert all(enlarged.getpixel((x * 8 + dx, y * 8 + dy)) == pixel
                       for dy in range(8) for dx in range(8))
    print(f"Validated {matrix.shape[0]}x{matrix.shape[1]} matrix; "
          f"land={int(matrix.sum())}, water={matrix.size - int(matrix.sum())}")


if __name__ == "__main__":
    main()
