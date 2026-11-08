from pathlib import Path
import csv

import numpy as np
from PIL import Image

# Hand-drawn approximate outlines. Coordinates are (positive degrees west,
# degrees north); no external geographic data is used.
MAINLAND = [
    (130.10, 55.12), (130.08, 55.00), (130.12, 54.94),
    (130.11, 54.88), (130.15, 54.83), (130.12, 54.76),
    (130.17, 54.72), (130.13, 54.66), (130.17, 54.61),
    (130.13, 54.56), (130.19, 54.53), (130.15, 54.47),
    (130.21, 54.44), (130.17, 54.39), (130.24, 54.36),
    (130.20, 54.32), (130.28, 54.29), (130.24, 54.25),
    (130.17, 54.26), (130.10, 54.22), (130.15, 54.18),
    (130.09, 54.15), (130.04, 54.17), (129.98, 54.12),
    (130.02, 54.08), (129.96, 54.04), (129.99, 54.00),
    (129.91, 53.96), (129.94, 53.92), (129.86, 53.88),
    (129.89, 53.83), (129.80, 53.80), (129.84, 53.75),
    (129.76, 53.71), (129.78, 53.66), (129.70, 53.62),
    (129.73, 53.57), (129.65, 53.53), (129.69, 53.48),
    (129.59, 53.44), (129.64, 53.39), (129.55, 53.34),
    (129.59, 53.29), (129.49, 53.24), (129.54, 53.18),
    (129.44, 53.13), (129.48, 53.08), (129.39, 53.03),
    (129.43, 52.88), (128.88, 52.88), (128.88, 55.12),
]

ISLANDS = [
    # Dundas Island and neighboring small islands
    [(130.94, 54.69), (130.87, 54.76), (130.76, 54.76),
     (130.67, 54.69), (130.64, 54.60), (130.69, 54.51),
     (130.77, 54.47), (130.86, 54.50), (130.93, 54.58)],
    [(130.99, 54.77), (130.96, 54.81), (130.91, 54.80),
     (130.91, 54.75), (130.95, 54.73)],
    [(130.61, 54.73), (130.56, 54.76), (130.51, 54.73),
     (130.53, 54.68), (130.58, 54.67)],
    [(130.72, 54.42), (130.66, 54.46), (130.59, 54.43),
     (130.58, 54.37), (130.64, 54.34), (130.71, 54.36)],
    # Prince Rupert vicinity
    [(130.62, 54.33), (130.56, 54.37), (130.49, 54.35),
     (130.44, 54.28), (130.47, 54.20), (130.55, 54.16),
     (130.64, 54.22), (130.67, 54.28)],
    [(130.49, 54.34), (130.43, 54.38), (130.38, 54.34),
     (130.37, 54.25), (130.42, 54.22), (130.47, 54.27)],
    [(130.37, 54.34), (130.31, 54.35), (130.27, 54.31),
     (130.28, 54.26), (130.34, 54.25)],
    [(130.61, 54.16), (130.56, 54.19), (130.51, 54.15),
     (130.53, 54.10), (130.59, 54.09)],
    # Porcher Island
    [(130.54, 54.17), (130.45, 54.18), (130.34, 54.12),
     (130.29, 54.03), (130.31, 53.96), (130.40, 53.88),
     (130.49, 53.89), (130.56, 53.97), (130.58, 54.07)],
    # Banks Island
    [(130.54, 53.93), (130.47, 53.92), (130.37, 53.84),
     (130.30, 53.75), (130.28, 53.65), (130.20, 53.56),
     (130.15, 53.46), (130.12, 53.36), (130.18, 53.31),
     (130.26, 53.36), (130.32, 53.45), (130.38, 53.55),
     (130.44, 53.65), (130.50, 53.76), (130.57, 53.85)],
    # Pitt Island
    [(130.24, 54.01), (130.17, 54.05), (130.09, 54.03),
     (130.03, 53.95), (129.99, 53.85), (129.95, 53.76),
     (129.94, 53.66), (129.89, 53.56), (129.91, 53.49),
     (129.98, 53.49), (130.05, 53.56), (130.11, 53.67),
     (130.16, 53.78), (130.22, 53.89)],
    # Northern Princess Royal Island and adjacent islets
    [(129.86, 53.54), (129.78, 53.53), (129.72, 53.47),
     (129.69, 53.39), (129.73, 53.31), (129.80, 53.24),
     (129.86, 53.14), (129.89, 53.03), (129.96, 52.89),
     (130.07, 52.89), (130.06, 53.00), (130.02, 53.11),
     (129.99, 53.21), (129.94, 53.33), (129.91, 53.45)],
    [(130.27, 53.33), (130.21, 53.34), (130.15, 53.28),
     (130.13, 53.21), (130.18, 53.17), (130.24, 53.23)],
    [(130.34, 53.16), (130.29, 53.19), (130.23, 53.15),
     (130.24, 53.09), (130.30, 53.06), (130.36, 53.10)],
]

# Water passages cut through the mainland coast.
INLETS = [
    [(130.34, 54.30), (130.20, 54.33), (130.08, 54.35),
     (129.96, 54.39), (129.85, 54.43), (129.82, 54.39),
     (129.94, 54.34), (130.06, 54.30), (130.19, 54.27),
     (130.34, 54.25)],
    [(130.18, 54.20), (130.05, 54.22), (129.94, 54.21),
     (129.81, 54.17), (129.72, 54.11), (129.68, 54.07),
     (129.73, 54.04), (129.84, 54.12), (129.95, 54.16),
     (130.10, 54.16), (130.18, 54.14)],
    [(130.01, 54.02), (129.89, 54.05), (129.76, 54.05),
     (129.67, 54.09), (129.59, 54.07), (129.65, 54.02),
     (129.75, 54.00), (129.86, 53.99), (130.00, 53.96)],
    [(129.85, 53.79), (129.74, 53.82), (129.62, 53.81),
     (129.52, 53.76), (129.48, 53.72), (129.54, 53.69),
     (129.64, 53.76), (129.76, 53.76), (129.86, 53.73)],
    [(129.73, 53.57), (129.58, 53.62), (129.48, 53.60),
     (129.35, 53.55), (129.25, 53.54), (129.19, 53.49),
     (129.25, 53.46), (129.36, 53.50), (129.49, 53.55),
     (129.59, 53.55), (129.70, 53.50)],
    [(129.57, 53.30), (129.43, 53.35), (129.31, 53.33),
     (129.19, 53.29), (129.10, 53.32), (129.07, 53.27),
     (129.18, 53.24), (129.31, 53.27), (129.43, 53.27),
     (129.56, 53.22)],
]


def inside_polygon(west, north, vertices):
    """Vectorized point-in-polygon classification by ray crossing."""
    inside = np.zeros(west.shape, dtype=bool)
    for i, (x1, y1) in enumerate(vertices):
        x2, y2 = vertices[(i + 1) % len(vertices)]
        if y1 == y2:
            continue
        crosses = (y1 > north) != (y2 > north)
        intersection = x1 + (north - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (west < intersection)
    return inside


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    west_values = 131.0 - np.arange(127) * (2.0 / 126.0)
    north_values = 55.0 - np.arange(127) * (2.0 / 126.0)
    west, north = np.meshgrid(west_values, north_values)

    mainland = inside_polygon(west, north, MAINLAND)
    for inlet in INLETS:
        mainland &= ~inside_polygon(west, north, inlet)

    land = mainland
    for island in ISLANDS:
        land |= inside_polygon(west, north, island)
    matrix = land.astype(np.uint8)
    assert matrix.shape == (127, 127)
    assert np.isin(matrix, (0, 1)).all()

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as file:
        csv.writer(file).writerows(matrix.tolist())

    image = Image.fromarray(matrix * 255, mode="L")
    image.save(output_dir / "candidate.png")
    image.resize((1016, 1016), Image.Resampling.NEAREST).save(
        output_dir / "candidate_8x.png"
    )

    with csv_path.open(newline="", encoding="ascii") as file:
        rows = list(csv.reader(file))
    assert len(rows) == 127
    assert all(len(row) == 127 for row in rows)
    assert all(value in ("0", "1") for row in rows for value in row)
    assert np.array_equal(np.array(rows, dtype=np.uint8), matrix)

    with Image.open(output_dir / "candidate.png") as saved:
        original = np.asarray(saved.convert("L"))
        assert saved.size == (127, 127)
        assert set(np.unique(original).tolist()) <= {0, 255}
        assert np.array_equal(original, matrix * 255)
    with Image.open(output_dir / "candidate_8x.png") as saved:
        enlarged = np.asarray(saved.convert("L"))
        assert saved.size == (1016, 1016)
        assert set(np.unique(enlarged).tolist()) <= {0, 255}
        assert np.array_equal(
            enlarged, np.repeat(np.repeat(original, 8, axis=0), 8, axis=1)
        )


if __name__ == "__main__":
    main()
