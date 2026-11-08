from pathlib import Path
import csv
import numpy as np
from PIL import Image


SIZE = 127
STEP = 2.0 / 126.0

# Coordinates use degrees west as x and north latitude as y.
# These hand-drawn points approximate the mainland shore and major islands.
MAINLAND = [
    (128.70, 55.00), (129.16, 55.00), (129.22, 54.91),
    (129.30, 54.82), (129.34, 54.70), (129.43, 54.62),
    (129.48, 54.50), (129.53, 54.40), (129.52, 54.30),
    (129.62, 54.20), (129.63, 54.08), (129.71, 53.98),
    (129.72, 53.87), (129.77, 53.77), (129.74, 53.66),
    (129.82, 53.57), (129.80, 53.47), (129.75, 53.39),
    (129.70, 53.30), (129.57, 53.22), (129.47, 53.14),
    (129.34, 53.08), (129.26, 53.00), (128.70, 53.00),
]

ISLANDS = [
    # Banks Island / adjacent mainland-fringing islands
    [(129.20, 53.36), (129.29, 53.39), (129.37, 53.36),
     (129.41, 53.30), (129.36, 53.23), (129.28, 53.22),
     (129.21, 53.27)],
    # McCauley, Aristazabal and nearby island groups
    [(129.58, 53.79), (129.65, 53.83), (129.72, 53.80),
     (129.70, 53.73), (129.63, 53.71)],
    [(129.69, 53.93), (129.77, 53.96), (129.82, 53.91),
     (129.79, 53.85), (129.72, 53.86)],
    [(129.80, 54.14), (129.86, 54.20), (129.93, 54.18),
     (129.95, 54.11), (129.89, 54.07), (129.82, 54.09)],
    [(130.06, 54.05), (130.11, 54.10), (130.16, 54.07),
     (130.15, 54.01), (130.09, 54.00)],
    # Small western islands near the north end of the strait
    [(130.57, 54.67), (130.64, 54.72), (130.71, 54.69),
     (130.69, 54.62), (130.61, 54.61)],
    [(130.81, 54.48), (130.87, 54.51), (130.91, 54.46),
     (130.88, 54.41), (130.82, 54.42)],
    # Easternmost Graham Island coast, only its edge enters the frame
    [(130.94, 54.28), (131.08, 54.31), (131.08, 54.12),
     (131.03, 54.02), (130.98, 53.88), (131.00, 53.66),
     (131.07, 53.51), (131.05, 53.32), (130.99, 53.20),
     (130.96, 53.06), (131.20, 53.00), (131.20, 54.32)],
]


def inside_polygon(x, y, polygon):
    """Ray-crossing point test; coastline ties are resolved toward land."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            crossing = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < crossing:
                inside = not inside
        j = i
    return inside

def classify_point(west_lon, lat):
    if inside_polygon(west_lon, lat, MAINLAND):
        return 1
    return int(any(inside_polygon(west_lon, lat, island) for island in ISLANDS))


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = np.zeros((SIZE, SIZE), dtype=np.uint8)
    for row in range(SIZE):
        lat = 55.0 - row * STEP
        for col in range(SIZE):
            west_lon = 131.0 - col * STEP
            matrix[row, col] = classify_point(west_lon, lat)

    with (output_dir / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Pillow maps 0 to black and 1 to white; nearest-neighbor preserves pixels.
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(output_dir / "candidate.png")
    enlarged = image.resize((SIZE * 8, SIZE * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")

    assert matrix.shape == (127, 127)
    assert matrix.size == 16129 and set(np.unique(matrix)) <= {0, 1}
    with (output_dir / "candidate.csv").open(newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 127
    assert all(len(row) == 127 and all(v in {"0", "1"} for v in row) for row in rows)
    assert image.size == (127, 127) and set(image.getdata()) <= {0, 255}
    assert enlarged.size == (1016, 1016)
    for y in range(SIZE):
        for x in range(SIZE):
            assert enlarged.getpixel((x * 8, y * 8)) == image.getpixel((x, y))
            assert enlarged.getpixel((x * 8 + 7, y * 8 + 7)) == image.getpixel((x, y))
    print("Validated 127x127 matrix, CSV, two-color PNG, and 8x nearest-neighbor PNG.")


if __name__ == "__main__":
    main()
