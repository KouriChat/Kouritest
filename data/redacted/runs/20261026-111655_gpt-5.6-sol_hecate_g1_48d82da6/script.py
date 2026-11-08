from pathlib import Path
import csv

import numpy as np
from PIL import Image


ROWS = 127
COLS = 127
STEP = 2.0 / 126.0


# Hand-estimated coastline and island outlines. Coordinates are (longitude,
# latitude) in WGS84 degrees; only these simple polygons inform classification.
LAND_POLYGONS = [
    # Mainland coast, including a deliberately irregular nearshore boundary.
    [
        (-129.52, 55.00), (-129.48, 54.88), (-129.57, 54.80),
        (-129.50, 54.69), (-129.63, 54.61), (-129.54, 54.50),
        (-129.67, 54.43), (-129.59, 54.31), (-129.73, 54.22),
        (-129.61, 54.12), (-129.73, 54.01), (-129.66, 53.91),
        (-129.82, 53.83), (-129.70, 53.72), (-129.88, 53.62),
        (-129.77, 53.52), (-129.95, 53.42), (-129.83, 53.32),
        (-130.02, 53.22), (-129.88, 53.12), (-130.06, 53.02),
        (-129.78, 53.00), (-129.73, 53.20), (-129.67, 53.40),
        (-129.64, 53.60), (-129.59, 53.80), (-129.55, 54.00),
        (-129.54, 54.20), (-129.50, 54.40), (-129.48, 54.60),
        (-129.47, 54.80), (-129.45, 55.00),
    ],

    # Banks Island, with its outer fringe simplified.
    [
        (-130.18, 54.45), (-130.08, 54.42), (-129.98, 54.39),
        (-129.91, 54.32), (-129.90, 54.22), (-129.95, 54.14),
        (-130.02, 54.10), (-130.10, 54.13), (-130.17, 54.20),
        (-130.21, 54.30), (-130.18, 54.45),
    ],
    # Pitt Island.
    [
        (-130.12, 53.78), (-130.03, 53.77), (-129.97, 53.72),
        (-129.96, 53.64), (-130.01, 53.57), (-130.09, 53.58),
        (-130.15, 53.65), (-130.12, 53.78),
    ],
    # Aristazabal Island.
    [
        (-130.28, 53.22), (-130.17, 53.20), (-130.10, 53.14),
        (-130.11, 53.06), (-130.20, 53.03), (-130.29, 53.08),
        (-130.32, 53.15), (-130.28, 53.22),
    ],
    # Dundas Islands and nearby small island groups near the northern coast.
    [
        (-130.82, 54.70), (-130.74, 54.69), (-130.68, 54.63),
        (-130.70, 54.57), (-130.78, 54.55), (-130.84, 54.60),
        (-130.82, 54.70),
    ],
    [
        (-130.64, 54.61), (-130.60, 54.60), (-130.57, 54.56),
        (-130.60, 54.53), (-130.65, 54.55), (-130.64, 54.61),
    ],
    # Small offshore islets and reefs represented as compact land patches.
    [
        (-130.30, 54.53), (-130.27, 54.54), (-130.25, 54.51),
        (-130.28, 54.49), (-130.31, 54.50), (-130.30, 54.53),
    ],
    [
        (-130.33, 53.89), (-130.29, 53.90), (-130.27, 53.87),
        (-130.30, 53.84), (-130.34, 53.86), (-130.33, 53.89),
    ],
]


def point_in_polygon(x, y, polygon):
    """Return whether one coordinate lies inside a hand-drawn polygon."""
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        if (current_y > y) != (previous_y > y):
            crossing_x = (previous_x - current_x) * (y - current_y) / (
                previous_y - current_y
            ) + current_x
            if x < crossing_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def is_land(longitude, latitude):
    return any(
        point_in_polygon(longitude, latitude, polygon)
        for polygon in LAND_POLYGONS
    )

def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = np.zeros((ROWS, COLS), dtype=np.uint8)
    for row in range(ROWS):
        latitude = 55.0 - row * STEP
        for column in range(COLS):
            longitude = -131.0 + column * STEP
            matrix[row, column] = int(is_land(longitude, latitude))

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(matrix.tolist())

    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(output_dir / "candidate.png")
    image.resize((COLS * 8, ROWS * 8), Image.Resampling.NEAREST).save(
        output_dir / "candidate_8x.png"
    )

    with csv_path.open("r", newline="", encoding="ascii") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == ROWS and all(len(row) == COLS for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert sum(map(len, rows)) == 16129

    base = Image.open(output_dir / "candidate.png").convert("RGB")
    enlarged = Image.open(output_dir / "candidate_8x.png").convert("RGB")
    assert base.size == (127, 127)
    assert enlarged.size == (1016, 1016)
    assert set(base.getdata()) <= {(0, 0, 0), (255, 255, 255)}
    expected = base.resize((1016, 1016), Image.Resampling.NEAREST)
    assert list(expected.getdata()) == list(enlarged.getdata())

    print(f"Validated {ROWS}x{COLS} = {ROWS * COLS} samples.")
    print("CSV values: 0/1 only; images: black/white, nearest-neighbor 8x.")


if __name__ == "__main__":
    main()
