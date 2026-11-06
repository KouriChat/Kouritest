#!/usr/bin/env python3
"""Approximate Hecate Strait land/water raster from hand-written polygons."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 127
STEP = 2.0 / 126.0
OUT = Path("outputs")

# Coordinates are (degrees west, degrees north). Larger west values are farther
# west. The mainland polygon is bounded by an approximate Pacific-facing coast;
# its eastern side is clipped beyond the requested frame.
MAINLAND_COAST = [
    (130.49, 55.00),
    (130.44, 54.88),
    (130.37, 54.76),
    (130.40, 54.66),
    (130.33, 54.57),
    (130.30, 54.48),
    (130.25, 54.39),
    (130.19, 54.30),
    (130.20, 54.22),
    (130.12, 54.15),
    (130.04, 54.08),
    (130.02, 54.00),
    (129.96, 53.93),
    (129.91, 53.86),
    (129.86, 53.79),
    (129.80, 53.72),
    (129.75, 53.65),
    (129.69, 53.58),
    (129.65, 53.51),
    (129.61, 53.44),
    (129.59, 53.36),
    (129.55, 53.28),
    (129.52, 53.20),
    (129.50, 53.12),
    (129.48, 53.00),
]
MAINLAND = MAINLAND_COAST + [(128.0, 53.0), (128.0, 55.0)]

# Approximate eastern edge of Graham Island. The polygon extends westward
# beyond the frame; only its eastern coastal fringe can affect these samples.
GRAHAM_EAST_COAST = [
    (131.28, 55.00),
    (131.34, 54.87),
    (131.28, 54.75),
    (131.21, 54.63),
    (131.14, 54.51),
    (131.08, 54.39),
    (131.03, 54.27),
    (131.01, 54.15),
    (131.00, 54.03),
    (131.02, 53.92),
    (131.07, 53.81),
    (131.15, 53.70),
    (131.25, 53.60),
]
GRAHAM_ISLAND = GRAHAM_EAST_COAST + [
    (132.5, 53.60),
    (132.5, 55.0),
]

# Hand-drawn island and reef approximations in the central and eastern strait.
# Small polygons intentionally represent only broad, likely island locations.
ISLANDS = [
    # Banks Island
    [(129.36, 54.18), (129.24, 54.15), (129.15, 54.08),
     (129.11, 53.98), (129.13, 53.88), (129.20, 53.80),
     (129.31, 53.78), (129.41, 53.83), (129.46, 53.92),
     (129.44, 54.04), (129.40, 54.13)],
    # Pitt Island
    [(130.06, 53.78), (129.98, 53.72), (129.93, 53.62),
     (129.92, 53.51), (129.96, 53.40), (130.02, 53.30),
     (130.10, 53.25), (130.18, 53.29), (130.22, 53.39),
     (130.20, 53.51), (130.15, 53.63)],
    # Porcher Island
    [(130.51, 54.03), (130.44, 54.00), (130.38, 53.94),
     (130.36, 53.87), (130.40, 53.82), (130.48, 53.81),
     (130.55, 53.85), (130.58, 53.92), (130.56, 53.99)],
    # Smaller islands around the mainland-side channels
    [(130.73, 54.18), (130.68, 54.15), (130.66, 54.10),
     (130.70, 54.06), (130.76, 54.07), (130.79, 54.12)],
    [(130.88, 53.99), (130.83, 53.96), (130.82, 53.91),
     (130.86, 53.87), (130.91, 53.89), (130.93, 53.94)],
    [(130.63, 53.68), (130.58, 53.64), (130.57, 53.59),
     (130.61, 53.55), (130.67, 53.58), (130.68, 53.63)],
    [(129.72, 54.38), (129.68, 54.35), (129.67, 54.30),
     (129.71, 54.27), (129.76, 54.30), (129.76, 54.35)],
    [(129.88, 54.27), (129.83, 54.24), (129.82, 54.19),
     (129.86, 54.16), (129.91, 54.19), (129.92, 54.24)],
    [(129.57, 54.54), (129.53, 54.51), (129.52, 54.47),
     (129.56, 54.44), (129.61, 54.47), (129.61, 54.51)],
    [(130.20, 54.50), (130.16, 54.47), (130.15, 54.43),
     (130.19, 54.40), (130.24, 54.43), (130.24, 54.47)],
    [(130.34, 54.29), (130.30, 54.26), (130.29, 54.22),
     (130.33, 54.19), (130.38, 54.22), (130.38, 54.26)],
    [(130.45, 53.56), (130.41, 53.53), (130.40, 53.49),
     (130.44, 53.46), (130.49, 53.49), (130.49, 53.53)],
    [(129.34, 53.55), (129.29, 53.52), (129.28, 53.47),
     (129.32, 53.43), (129.37, 53.46), (129.38, 53.51)],
    # Small reefs and islets near Banks Island
    [(129.57, 54.08), (129.54, 54.06), (129.54, 54.03),
     (129.57, 54.01), (129.60, 54.04), (129.60, 54.06)],
    [(129.67, 53.94), (129.64, 53.92), (129.64, 53.89),
     (129.67, 53.87), (129.70, 53.90), (129.70, 53.92)],
    [(129.78, 54.04), (129.75, 54.02), (129.75, 53.99),
     (129.78, 53.97), (129.81, 54.00), (129.81, 54.02)],
    [(130.00, 54.12), (129.97, 54.10), (129.97, 54.07),
     (130.00, 54.05), (130.03, 54.08), (130.03, 54.10)],
]


def point_in_polygon(x, y, polygon):
    """Ray-crossing point-in-polygon test."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            crossing_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < crossing_x:
                inside = not inside
        j = i
    return inside


def is_land(west, north):
    if point_in_polygon(west, north, MAINLAND):
        return True
    if point_in_polygon(west, north, GRAHAM_ISLAND):
        return True
    return any(point_in_polygon(west, north, polygon) for polygon in ISLANDS)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Row 0 is 55N; column 0 is 131W. Values are point classifications.
    matrix = np.zeros((SIZE, SIZE), dtype=np.uint8)
    for row in range(SIZE):
        north = 55.0 - row * STEP
        for col in range(SIZE):
            west = 131.0 - col * STEP
            matrix[row, col] = int(is_land(west, north))

    csv_path = OUT / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # White is land (1); black is water (0).
    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(OUT / "candidate.png")
    image.resize((SIZE * 8, SIZE * 8), resample=Image.Resampling.NEAREST).save(
        OUT / "candidate_8x.png"
    )

    # Validate the exported matrix and image properties.
    with csv_path.open("r", newline="", encoding="ascii") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == SIZE
    assert all(len(row) == SIZE for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert sum(len(row) for row in rows) == 16129

    original = Image.open(OUT / "candidate.png")
    enlarged = Image.open(OUT / "candidate_8x.png")
    assert original.size == (127, 127)
    assert enlarged.size == (1016, 1016)
    assert set(original.getdata()) <= {(0, 0, 0), (255, 255, 255)}
    expected = original.resize((1016, 1016), resample=Image.Resampling.NEAREST)
    assert list(enlarged.getdata()) == list(expected.getdata())

    print(f"Wrote {csv_path}, outputs/candidate.png, outputs/candidate_8x.png")
    print("Validated 127x127 CSV and black/white nearest-neighbor PNGs.")


if __name__ == "__main__":
    main()
