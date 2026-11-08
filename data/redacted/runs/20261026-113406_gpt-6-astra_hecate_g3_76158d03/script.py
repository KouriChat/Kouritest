from pathlib import Path
import csv

import numpy as np
from PIL import Image


# Hand-estimated outlines; coordinates are (latitude, longitude).
MAINLAND = [
    (55.00, -129.95), (54.91, -130.10), (54.83, -130.32),
    (54.75, -130.52), (54.67, -130.55), (54.61, -130.43),
    (54.54, -130.34), (54.47, -130.20), (54.40, -130.06),
    (54.33, -129.95), (54.26, -129.82), (54.18, -129.73),
    (54.10, -129.66), (54.02, -129.59), (53.94, -129.54),
    (53.86, -129.47), (53.76, -129.40), (53.66, -129.32),
    (53.55, -129.24), (53.43, -129.17), (53.30, -129.10),
    (53.16, -129.04), (53.00, -128.99),
    (53.00, -127.00), (55.00, -127.00),
]

INLETS = [
    [
        (55.02, -130.02), (54.96, -130.00), (54.90, -129.99),
        (54.84, -129.98), (54.79, -130.00), (54.77, -130.09),
        (54.82, -130.12), (54.90, -130.10), (54.97, -130.11),
    ],
    [
        (54.67, -130.53), (54.63, -130.45), (54.60, -130.36),
        (54.57, -130.30), (54.54, -130.34), (54.56, -130.43),
        (54.60, -130.52),
    ],
    [
        (54.42, -130.14), (54.38, -130.08), (54.34, -130.01),
        (54.31, -129.97), (54.30, -130.04), (54.34, -130.12),
        (54.38, -130.18),
    ],
    [
        (54.13, -129.77), (54.08, -129.71), (54.03, -129.65),
        (53.99, -129.62), (53.98, -129.69), (54.03, -129.76),
        (54.08, -129.81),
    ],
]
ISLANDS = [
    # Eastern fringe of Graham Island, Haida Gwaii.
    [
        (54.56, -131.00), (54.53, -131.02), (54.49, -131.05),
        (54.43, -131.07), (54.35, -131.09), (54.27, -131.11),
        (54.19, -131.13), (54.11, -131.16), (54.03, -131.19),
        (53.95, -131.22), (53.87, -131.25), (53.79, -131.28),
        (53.70, -131.30), (53.61, -131.31), (53.52, -131.32),
        (53.43, -131.32), (53.34, -131.31), (53.25, -131.30),
        (53.16, -131.28), (53.08, -131.25), (53.00, -131.21),
        (53.00, -131.00), (53.12, -131.00), (53.24, -131.00),
        (53.36, -131.00), (53.48, -131.00), (53.60, -131.00),
        (53.72, -131.00), (53.84, -131.00), (53.96, -131.00),
        (54.08, -131.00), (54.20, -131.00), (54.32, -131.00),
        (54.44, -131.00), (54.52, -131.00),
    ],
    # Banks Island.
    [
        (54.50, -130.22), (54.47, -130.15), (54.43, -130.12),
        (54.38, -130.13), (54.34, -130.18), (54.33, -130.24),
        (54.36, -130.29), (54.41, -130.30), (54.46, -130.27),
    ],
    # Pitt Island.
    [
        (53.55, -129.73), (53.52, -129.68), (53.47, -129.67),
        (53.43, -129.70), (53.41, -129.76), (53.44, -129.80),
        (53.50, -129.79),
    ],
    # Porcher Island.
    [
        (53.95, -130.40), (53.91, -130.34), (53.86, -130.34),
        (53.82, -130.39), (53.83, -130.46), (53.88, -130.49),
        (53.93, -130.46),
    ],
    [(53.72, -130.13), (53.69, -130.08), (53.65, -130.10),
     (53.64, -130.15), (53.68, -130.18)],
    [(54.05, -130.70), (54.02, -130.66), (53.98, -130.68),
     (53.98, -130.73), (54.02, -130.75)],
    [(54.72, -130.82), (54.69, -130.77), (54.65, -130.79),
     (54.65, -130.84), (54.69, -130.86)],
]
def point_in_polygon(lat, lon, polygon):
    """Ray-casting point test; points on polygon edges count as land."""
    inside = False
    for i, (y1, x1) in enumerate(polygon):
        y2, x2 = polygon[(i + 1) % len(polygon)]
        cross = (lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
        if abs(cross) < 1e-10:
            if (min(x1, x2) - 1e-10 <= lon <= max(x1, x2) + 1e-10
                    and min(y1, y2) - 1e-10 <= lat <= max(y1, y2) + 1e-10):
                return True
        if (y1 > lat) != (y2 > lat):
            crossing_lon = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < crossing_lon:
                inside = not inside
    return inside


def is_land(lat, lon):
    if any(point_in_polygon(lat, lon, shape) for shape in INLETS):
        return False
    return (point_in_polygon(lat, lon, MAINLAND)
            or any(point_in_polygon(lat, lon, shape) for shape in ISLANDS))


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    latitudes = np.linspace(55.0, 53.0, 127)
    longitudes = np.linspace(-131.0, -129.0, 127)
    matrix = np.zeros((127, 127), dtype=np.uint8)
    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            matrix[row, col] = int(is_land(float(lat), float(lon)))

    with (output_dir / "candidate.csv").open(
            "w", newline="", encoding="ascii") as stream:
        csv.writer(stream, lineterminator="\n").writerows(matrix.tolist())

    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(output_dir / "candidate.png")
    image.resize((1016, 1016), Image.Resampling.NEAREST).save(
        output_dir / "candidate_8x.png")
    assert matrix.shape == (127, 127)
    assert matrix.size == 16129
    assert set(np.unique(matrix).tolist()) <= {0, 1}
    with (output_dir / "candidate.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 127
    assert all(len(row) == 127 and set(row) <= {"0", "1"} for row in rows)
    original = Image.open(output_dir / "candidate.png")
    enlarged = Image.open(output_dir / "candidate_8x.png")
    assert original.size == (127, 127)
    assert enlarged.size == (1016, 1016)
    assert set(original.getdata()) <= {(0, 0, 0), (255, 255, 255)}
    for y in range(127):
        for x in range(127):
            pixel = original.getpixel((x, y))
            assert all(enlarged.getpixel((x * 8 + dx, y * 8 + dy)) == pixel
                       for dx in range(8) for dy in range(8))


if __name__ == "__main__":
    main()
