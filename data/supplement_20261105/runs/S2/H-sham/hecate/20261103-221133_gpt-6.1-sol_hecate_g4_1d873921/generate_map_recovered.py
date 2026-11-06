from pathlib import Path
import csv

import numpy as np
from PIL import Image


ROWS = 127
COLS = 127
STEP = 2.0 / 126.0
LAT_NORTH = 55.0
LON_WEST = -131.0

# Hand-approximated mainland coastline, ordered south to north.
# Mainland is east of this line.
MAINLAND_COAST = [
    (53.00, -129.52),
    (53.08, -129.61),
    (53.18, -129.69),
    (53.28, -129.78),
    (53.38, -129.84),
    (53.48, -129.91),
    (53.58, -129.97),
    (53.68, -130.03),
    (53.78, -130.08),
    (53.88, -130.14),
    (53.98, -130.20),
    (54.08, -130.27),
    (54.18, -130.31),
    (54.28, -130.34),
    (54.38, -130.30),
    (54.48, -130.24),
    (54.58, -130.20),
    (54.68, -130.24),
    (54.78, -130.32),
    (54.88, -130.40),
    (55.00, -130.46),
]

# Approximate island shorelines. Coordinates are (latitude, longitude).
ISLANDS = [
    # Banks Island
    [
        (53.68, -130.44), (53.72, -130.34), (53.78, -130.28),
        (53.84, -130.22), (53.91, -130.15), (53.98, -130.08),
        (54.04, -130.02), (54.07, -129.97), (54.02, -129.94),
        (53.94, -129.98), (53.86, -130.03), (53.78, -130.10),
        (53.72, -130.19), (53.68, -130.30),
    ],
    # Porcher Island
    [
        (53.82, -130.54), (53.87, -130.50), (53.91, -130.44),
        (53.96, -130.42), (54.01, -130.37), (54.06, -130.34),
        (54.11, -130.28), (54.15, -130.22), (54.13, -130.16),
        (54.08, -130.14), (54.02, -130.19), (53.97, -130.23),
        (53.91, -130.27), (53.86, -130.34), (53.82, -130.43),
    ],
    # Pitt Island
    [
        (53.24, -130.22), (53.29, -130.18), (53.34, -130.12),
        (53.40, -130.08), (53.46, -130.02), (53.53, -129.98),
        (53.60, -129.96), (53.65, -129.91), (53.61, -129.86),
        (53.55, -129.84), (53.48, -129.88), (53.42, -129.91),
        (53.36, -129.96), (53.30, -130.03), (53.26, -130.12),
    ],
    # Small islands around the northern entrance and Dixon Entrance.
    [
        (54.48, -130.89), (54.53, -130.83), (54.59, -130.79),
        (54.65, -130.80), (54.69, -130.86), (54.65, -130.92),
        (54.58, -130.94), (54.52, -130.93),
    ],
    [
        (54.34, -130.73), (54.38, -130.69), (54.43, -130.68),
        (54.46, -130.72), (54.43, -130.77), (54.38, -130.78),
    ],
    # Easternmost approximate Graham Island shoreline fragments.
    [
        (53.45, -131.00), (53.52, -130.99), (53.60, -130.98),
        (53.68, -130.97), (53.76, -130.96), (53.84, -130.95),
        (53.92, -130.95), (54.00, -130.97), (54.08, -130.99),
        (54.14, -131.00), (54.08, -131.00), (54.00, -131.00),
        (53.92, -131.00), (53.84, -131.00), (53.76, -131.00),
        (53.68, -131.00), (53.60, -131.00), (53.52, -131.00),
    ],
    # Approximate small islets near the eastern edge of Moresby Island.
    [
        (53.02, -131.00), (53.08, -130.99), (53.15, -130.98),
        (53.22, -130.98), (53.28, -130.99), (53.31, -131.00),
        (53.25, -131.00), (53.18, -131.00), (53.10, -131.00),
    ],
]

# Approximate mainland fjord and inlet water areas. These carve water from
# the broad mainland polygon where channels plausibly extend into the frame.
INLETS = [
    [
        (53.38, -129.84), (53.43, -129.87), (53.49, -129.92),
        (53.54, -129.98), (53.58, -130.04), (53.63, -130.07),
        (53.67, -130.05), (53.64, -129.98), (53.59, -129.92),
        (53.53, -129.87), (53.46, -129.83),
    ],
    [
        (53.76, -130.08), (53.81, -130.13), (53.85, -130.19),
        (53.90, -130.24), (53.94, -130.28), (53.98, -130.27),
        (53.96, -130.21), (53.91, -130.16), (53.85, -130.11),
    ],
    [
        (54.07, -130.27), (54.12, -130.31), (54.17, -130.36),
        (54.22, -130.40), (54.27, -130.41), (54.24, -130.35),
        (54.19, -130.31), (54.14, -130.27),
    ],
]


def point_in_polygon(lat, lon, polygon):
    """Ray-casting point test for a polygon given as (lat, lon) vertices."""
    inside = False
    j = len(polygon) - 1
    for i, (yi, xi) in enumerate(polygon):
        yj, xj = polygon[j]
        crosses = (yi > lat) != (yj > lat)
        if crosses:
            crossing_lon = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing_lon:
                inside = not inside
        j = i
    return inside


def mainland_at(lat, lon):
    coast_lon = None
    for i in range(len(MAINLAND_COAST) - 1):
        lat1, lon1 = MAINLAND_COAST[i]
        lat2, lon2 = MAINLAND_COAST[i + 1]
        if lat1 <= lat <= lat2:
            fraction = (lat - lat1) / (lat2 - lat1)
            coast_lon = lon1 + fraction * (lon2 - lon1)
            break

    # The eastern edge of the sample box lies inland in this approximation.
    if coast_lon is None:
        return lon > -129.0
    return lon > coast_lon


def classify(lat, lon):
    land = mainland_at(lat, lon)

    if any(point_in_polygon(lat, lon, island) for island in ISLANDS):
        land = True

    if any(point_in_polygon(lat, lon, inlet) for inlet in INLETS):
        land = False

    return 1 if land else 0


def main():
    latitudes = [LAT_NORTH - row * STEP for row in range(ROWS)]
    longitudes = [LON_WEST + col * STEP for col in range(COLS)]

    matrix = np.array(
        [[classify(lat, lon) for lon in longitudes] for lat in latitudes],
        dtype=np.uint8,
    )

    if matrix.shape != (127, 127) or matrix.size != 16129:
        raise AssertionError("Unexpected matrix dimensions")
    if not np.isin(matrix, (0, 1)).all():
        raise AssertionError("Matrix contains values other than 0 and 1")

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # White is land (1); black is water (0).
    pixels = np.where(matrix == 1, 255, 0).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    image.save(output_dir / "candidate.png")

    enlarged = image.resize((1016, 1016), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")

    with csv_path.open("r", newline="", encoding="ascii") as stream:
        rows = list(csv.reader(stream))
    if len(rows) != 127 or any(len(row) != 127 for row in rows):
        raise AssertionError("CSV dimensions are not 127 by 127")
    if any(value not in ("0", "1") for row in rows for value in row):
        raise AssertionError("CSV contains an invalid or missing value")

    with Image.open(output_dir / "candidate.png") as original:
        if original.size != (127, 127) or set(original.getdata()) != {0, 255}:
            raise AssertionError("Original PNG validation failed")
    with Image.open(output_dir / "candidate_8x.png") as enlarged_check:
        if enlarged_check.size != (1016, 1016):
            raise AssertionError("Enlarged PNG dimensions are incorrect")
        if set(enlarged_check.getdata()) != {0, 255}:
            raise AssertionError("Enlarged PNG contains unexpected colors")

    print("Wrote and validated outputs/candidate.csv")
    print("Wrote and validated outputs/candidate.png")
    print("Wrote and validated outputs/candidate_8x.png")


if __name__ == "__main__":
    main()
