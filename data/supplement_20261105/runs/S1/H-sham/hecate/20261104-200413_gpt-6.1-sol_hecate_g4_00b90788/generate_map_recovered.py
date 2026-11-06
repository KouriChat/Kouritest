from pathlib import Path

import numpy as np
from PIL import Image


SIZE = 127
STEP = 2.0 / 126.0

# Approximate mainland shoreline as western longitude versus latitude.
# Land is east of this line (smaller west-longitude values).
COAST = [
    (53.00, 129.48),
    (53.15, 129.55),
    (53.30, 129.66),
    (53.45, 129.77),
    (53.60, 129.87),
    (53.75, 129.96),
    (53.90, 130.06),
    (54.05, 130.16),
    (54.20, 130.25),
    (54.35, 130.25),
    (54.50, 130.18),
    (54.65, 130.08),
    (54.80, 129.98),
    (54.90, 129.92),
    (55.00, 129.86),
]

# Hand-drawn approximate land polygons for prominent islands in the box.
# Coordinates are (latitude, degrees west).
ISLANDS = [
    # Banks Island
    [
        (53.83, 129.38), (53.88, 129.30), (53.98, 129.24),
        (54.10, 129.20), (54.22, 129.23), (54.32, 129.31),
        (54.38, 129.42), (54.34, 129.52), (54.25, 129.57),
        (54.10, 129.55), (53.96, 129.49), (53.87, 129.44),
    ],
    # Pitt Island
    [
        (53.27, 129.55), (53.33, 129.48), (53.43, 129.47),
        (53.54, 129.52), (53.66, 129.59), (53.75, 129.69),
        (53.73, 129.78), (53.65, 129.83), (53.53, 129.79),
        (53.42, 129.73), (53.32, 129.66),
    ],
    # Porcher Island
    [
        (53.79, 130.00), (53.84, 129.89), (53.91, 129.85),
        (53.98, 129.89), (54.00, 129.98), (53.97, 130.08),
        (53.91, 130.14), (53.84, 130.11),
    ],
    # Dundas Island group
    [
        (54.58, 130.43), (54.62, 130.35), (54.68, 130.36),
        (54.73, 130.43), (54.72, 130.52), (54.67, 130.56),
        (54.61, 130.52),
    ],
    [
        (54.51, 130.57), (54.54, 130.53), (54.58, 130.55),
        (54.58, 130.60), (54.54, 130.62),
    ],
    # Small coastal islands near the mainland
    [
        (54.34, 130.47), (54.39, 130.41), (54.45, 130.44),
        (54.47, 130.51), (54.43, 130.56), (54.37, 130.54),
    ],
    [
        (54.04, 130.33), (54.08, 130.30), (54.12, 130.34),
        (54.11, 130.39), (54.06, 130.40),
    ],
]

# Approximate fjords and inlets cut into the mainland.
INLETS = [
    # Portland Inlet / Portland Canal area, clipped by the north edge.
    [
        (54.57, 129.35), (54.67, 129.47), (54.78, 129.58),
        (54.90, 129.68), (55.00, 129.72), (55.00, 129.48),
        (54.90, 129.45), (54.79, 129.40), (54.67, 129.34),
    ],
    # Inlet system north of Prince Rupert.
    [
        (54.08, 129.45), (54.15, 129.57), (54.24, 129.69),
        (54.33, 129.82), (54.43, 129.91), (54.48, 130.00),
        (54.40, 130.02), (54.31, 129.93), (54.22, 129.81),
        (54.14, 129.68), (54.08, 129.57),
    ],
    # Skeena estuary and adjacent channels.
    [
        (53.75, 129.43), (53.80, 129.55), (53.86, 129.67),
        (53.92, 129.80), (53.94, 129.90), (53.88, 129.92),
        (53.82, 129.81), (53.77, 129.68), (53.73, 129.55),
    ],
]


def point_in_polygon(lat, west_lon, polygon):
    """Even-odd point-in-polygon test for (latitude, west longitude)."""
    inside = False
    j = len(polygon) - 1
    for i, (yi, xi) in enumerate(polygon):
        yj, xj = polygon[j]
        crosses = (yi > lat) != (yj > lat)
        if crosses:
            x_cross = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if west_lon < x_cross:
                inside = not inside
        j = i
    return inside


def coast_west_longitude(lat):
    for (lat0, x0), (lat1, x1) in zip(COAST, COAST[1:]):
        if lat0 <= lat <= lat1:
            fraction = (lat - lat0) / (lat1 - lat0)
            return x0 + fraction * (x1 - x0)
    if lat < COAST[0][0]:
        return COAST[0][1]
    return COAST[-1][1]


def build_matrix():
    # Rows run north to south; columns run west to east.
    latitudes = 55.0 - np.arange(SIZE) * STEP
    west_longitudes = 131.0 - np.arange(SIZE) * STEP
    matrix = np.zeros((SIZE, SIZE), dtype=np.uint8)

    for row, lat in enumerate(latitudes):
        coast_x = coast_west_longitude(float(lat))
        for col, west_lon in enumerate(west_longitudes):
            x = float(west_lon)

            # Mainland rule, then carve approximate marine inlets.
            land = x <= coast_x
            if land and any(point_in_polygon(float(lat), x, p) for p in INLETS):
                land = False

            # Add island land independently of the mainland rule.
            if not land and any(
                point_in_polygon(float(lat), x, p) for p in ISLANDS
            ):
                land = True

            matrix[row, col] = 1 if land else 0

    return matrix


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = build_matrix()
    if matrix.shape != (127, 127) or matrix.size != 16129:
        raise RuntimeError("Unexpected matrix dimensions")
    if not np.isin(matrix, (0, 1)).all():
        raise RuntimeError("Matrix contains values other than 0 and 1")

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", encoding="ascii", newline="") as f:
        for row in matrix:
            f.write(",".join(str(int(value)) for value in row) + "\n")

    # Land is white; water is black.
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(output_dir / "candidate.png")

    enlarged = image.resize((1016, 1016), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")

    # Verify CSV dimensions/content and image sizes/colors.
    lines = csv_path.read_text(encoding="ascii").splitlines()
    if len(lines) != 127 or any(
        len(line.split(",")) != 127
        or any(value not in ("0", "1") for value in line.split(","))
        for line in lines
    ):
        raise RuntimeError("CSV validation failed")

    if image.size != (127, 127) or set(image.getdata()) != {0, 255}:
        raise RuntimeError("Original image validation failed")
    if enlarged.size != (1016, 1016) or set(enlarged.getdata()) != {0, 255}:
        raise RuntimeError("Enlarged image validation failed")

    print("Created outputs/candidate.csv")
    print("Created outputs/candidate.png (127x127)")
    print("Created outputs/candidate_8x.png (1016x1016, nearest neighbor)")
    print(f"Validated {matrix.size} binary samples")


if __name__ == "__main__":
    main()
