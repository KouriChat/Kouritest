from pathlib import Path

import numpy as np
from PIL import Image


ROWS = 127
COLS = 127


def inside_polygon(lon, lat, polygon):
    """Return whether a point lies inside a longitude/latitude polygon."""
    inside = False
    previous_lon, previous_lat = polygon[-1]
    for current_lon, current_lat in polygon:
        crosses = (current_lat > lat) != (previous_lat > lat)
        if crosses:
            crossing_lon = (
                (previous_lon - current_lon) * (lat - current_lat)
                / (previous_lat - current_lat)
                + current_lon
            )
            if lon < crossing_lon:
                inside = not inside
        previous_lon, previous_lat = current_lon, current_lat
    return inside


# Approximate mainland coast, drawn as the western edge of a land polygon.
# Vertex spacing deliberately captures broad inlets rather than detailed charts.
MAINLAND = [
    (-130.30, 55.00), (-130.16, 54.91), (-130.24, 54.82),
    (-130.05, 54.75), (-130.18, 54.64), (-130.39, 54.56),
    (-130.35, 54.47), (-130.10, 54.40), (-130.16, 54.31),
    (-130.02, 54.22), (-130.08, 54.13), (-129.91, 54.04),
    (-129.96, 53.95), (-129.82, 53.87), (-129.91, 53.77),
    (-129.74, 53.68), (-129.82, 53.58), (-129.67, 53.48),
    (-129.77, 53.38), (-129.58, 53.27), (-129.70, 53.18),
    (-129.49, 53.08), (-129.58, 53.00),
    (-128.70, 53.00), (-128.70, 55.00),
]

# Eastern Graham Island and Moresby Island edges within the western frame.
# The main Haida Gwaii landmasses lie mostly west of the sampled box.
GRAHAM_EAST = [
    (-131.015, 54.12), (-130.985, 54.06), (-130.978, 53.98),
    (-130.992, 53.90), (-130.974, 53.82), (-131.005, 53.75),
    (-131.015, 53.67), (-131.030, 53.60), (-131.012, 53.52),
    (-131.025, 53.44), (-131.060, 53.37), (-131.100, 53.33),
    (-131.170, 53.32), (-131.210, 53.37), (-131.205, 53.47),
    (-131.190, 53.57), (-131.180, 53.67), (-131.160, 53.77),
    (-131.145, 53.87), (-131.130, 53.96), (-131.100, 54.05),
]

MORESBY_NORTHEAST = [
    (-131.000, 53.34), (-130.965, 53.29), (-130.958, 53.22),
    (-130.980, 53.16), (-131.010, 53.10), (-131.045, 53.06),
    (-131.100, 53.04), (-131.150, 53.07), (-131.175, 53.13),
    (-131.168, 53.20), (-131.145, 53.27), (-131.110, 53.32),
]
# Principal islands east of Haida Gwaii and near the mainland coast.
ISLANDS = [
    [(-130.50, 54.55), (-130.39, 54.58), (-130.30, 54.53),
     (-130.27, 54.43), (-130.35, 54.35), (-130.45, 54.37),
     (-130.53, 54.44)],
    [(-130.26, 54.31), (-130.17, 54.34), (-130.10, 54.28),
     (-130.13, 54.19), (-130.22, 54.16), (-130.29, 54.22)],
    [(-130.44, 53.96), (-130.32, 54.01), (-130.22, 53.96),
     (-130.20, 53.85), (-130.29, 53.78), (-130.40, 53.81),
     (-130.47, 53.88)],
    [(-130.22, 53.77), (-130.12, 53.80), (-130.04, 53.75),
     (-130.00, 53.67), (-130.07, 53.60), (-130.18, 53.63),
     (-130.24, 53.69)],
    [(-130.22, 53.50), (-130.10, 53.54), (-130.03, 53.48),
     (-130.01, 53.39), (-130.08, 53.32), (-130.18, 53.35),
     (-130.25, 53.42)],
    [(-130.02, 53.24), (-129.92, 53.29), (-129.82, 53.25),
     (-129.80, 53.16), (-129.87, 53.10), (-129.97, 53.12),
     (-130.04, 53.17)],
]


def is_land(lon, lat):
    if inside_polygon(lon, lat, MAINLAND):
        return True
    if inside_polygon(lon, lat, GRAHAM_EAST):
        return True
    if inside_polygon(lon, lat, MORESBY_NORTHEAST):
        return True
    return any(inside_polygon(lon, lat, polygon) for polygon in ISLANDS)


def build_matrix():
    latitudes = np.linspace(55.0, 53.0, ROWS)
    longitudes = np.linspace(-131.0, -129.0, COLS)
    return np.array(
        [[int(is_land(lon, lat)) for lon in longitudes] for lat in latitudes],
        dtype=np.uint8,
    )
def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()

    with (output_dir / "candidate.csv").open("w", encoding="ascii", newline="") as csv_file:
        for row in matrix:
            csv_file.write(",".join(str(int(value)) for value in row) + "\n")

    # Land is white (255), water is black (0); no interpolation is applied.
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(output_dir / "candidate.png")
    enlarged = image.resize((COLS * 8, ROWS * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")

    assert matrix.shape == (127, 127)
    assert matrix.size == 16129 and set(np.unique(matrix)) <= {0, 1}
    assert image.size == (127, 127)
    assert enlarged.size == (1016, 1016)
    assert set(image.getdata()) <= {0, 255}
    print(f"Validated {matrix.size} samples; land={int(matrix.sum())}, water={matrix.size - int(matrix.sum())}")


if __name__ == "__main__":
    main()
