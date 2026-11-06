from pathlib import Path

import numpy as np
from PIL import Image


# Hand-drawn approximate land polygons in (longitude, latitude).
# Coordinates and shorelines are deliberately approximate.
ISLANDS = [
    # Banks Island
    [
        (-130.24, 54.49), (-130.12, 54.43), (-129.99, 54.35),
        (-129.91, 54.23), (-129.88, 54.07), (-129.91, 53.91),
        (-129.98, 53.80), (-130.08, 53.83), (-130.17, 53.97),
        (-130.22, 54.13), (-130.28, 54.30),
    ],
    # Pitt Island
    [
        (-130.03, 53.79), (-129.88, 53.76), (-129.76, 53.66),
        (-129.71, 53.52), (-129.76, 53.39), (-129.86, 53.29),
        (-129.98, 53.23), (-130.08, 53.31), (-130.12, 53.47),
        (-130.10, 53.63),
    ],
    # Porcher Island, clipped by the western edge of the sampling area
    [
        (-131.10, 53.96), (-130.80, 54.00), (-130.55, 53.93),
        (-130.43, 53.82), (-130.50, 53.70), (-130.78, 53.70),
        (-131.10, 53.76),
    ],
    # Dundas Island group, mostly west of the frame
    [
        (-131.20, 54.74), (-130.84, 54.79), (-130.58, 54.70),
        (-130.53, 54.58), (-130.74, 54.52), (-131.20, 54.55),
    ],
    # Small nearshore island groups
    [
        (-130.12, 54.67), (-130.00, 54.69), (-129.94, 54.63),
        (-129.98, 54.57), (-130.09, 54.58),
    ],
    [
        (-129.86, 53.17), (-129.76, 53.19), (-129.71, 53.12),
        (-129.76, 53.06), (-129.85, 53.08),
    ],
]

# Approximate mainland west coast, ordered north to south.
# Land is classified east of this line. The irregular vertices represent
# major coastal bends, not a detailed tracing of inlets and fjords.
MAINLAND_COAST = [
    (55.00, -130.00),
    (54.85, -130.00),
    (54.72, -129.94),
    (54.60, -129.88),
    (54.48, -129.92),
    (54.38, -130.00),
    (54.30, -129.96),
    (54.20, -129.85),
    (54.08, -129.77),
    (53.95, -129.70),
    (53.82, -129.60),
    (53.68, -129.53),
    (53.55, -129.47),
    (53.40, -129.42),
    (53.25, -129.36),
    (53.00, -129.28),
]


def point_in_polygon(lon, lat, polygon):
    """Return whether a point lies inside a polygon using ray crossing."""
    inside = False
    count = len(polygon)
    j = count - 1

    for i in range(count):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        crosses = (yi > lat) != (yj > lat)
        if crosses:
            crossing_lon = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing_lon:
                inside = not inside
        j = i

    return inside


def mainland_west_edge(lat):
    """Linearly interpolate the approximate mainland shoreline longitude."""
    for (lat0, lon0), (lat1, lon1) in zip(MAINLAND_COAST, MAINLAND_COAST[1:]):
        if lat0 >= lat >= lat1:
            fraction = (lat0 - lat) / (lat0 - lat1)
            return lon0 + fraction * (lon1 - lon0)

    if lat > MAINLAND_COAST[0][0]:
        return MAINLAND_COAST[0][1]
    return MAINLAND_COAST[-1][1]


def is_land(lon, lat):
    # Mainland lies east of the hand-drawn west-coast line.
    if lon >= mainland_west_edge(lat):
        return True

    return any(point_in_polygon(lon, lat, island) for island in ISLANDS)


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Endpoints included: 127 samples at a spacing of 2/126 degrees.
    latitudes = np.linspace(55.0, 53.0, 127)
    longitudes = np.linspace(-131.0, -129.0, 127)

    matrix = np.zeros((127, 127), dtype=np.uint8)
    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            matrix[row, col] = 1 if is_land(float(lon), float(lat)) else 0

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", encoding="ascii", newline="") as f:
        for row in matrix:
            f.write(",".join(str(int(value)) for value in row) + "\n")

    # Matrix 1 is white (land); matrix 0 is black (water).
    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(output_dir / "candidate.png")

    enlarged = image.resize((1016, 1016), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")

    # Validate dimensions, values, CSV shape, colors, and nearest-neighbor scaling.
    assert matrix.shape == (127, 127)
    assert int(matrix.size) == 16129
    assert set(np.unique(matrix).tolist()) <= {0, 1}

    lines = csv_path.read_text(encoding="ascii").splitlines()
    assert len(lines) == 127
    assert all(len(line.split(",")) == 127 for line in lines)
    assert all(value in {"0", "1"} for line in lines for value in line.split(","))

    assert image.size == (127, 127)
    assert set(image.getdata()) <= {(0, 0, 0), (255, 255, 255)}
    assert enlarged.size == (1016, 1016)
    expected = np.repeat(np.repeat(np.asarray(image), 8, axis=0), 8, axis=1)
    assert np.array_equal(np.asarray(enlarged), expected)

    print(f"Generated {matrix.shape[0]}x{matrix.shape[1]} matrix ({matrix.size} values).")
    print(f"CSV and PNG outputs written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
