from pathlib import Path

import numpy as np
from PIL import Image


WIDTH = 127
HEIGHT = 127
WEST = -131.0
EAST = -129.0
NORTH = 55.0
SOUTH = 53.0
STEP = 2.0 / 126.0

# Hand-estimated shoreline for the BC mainland. The polygon covers land east
# of this western shoreline, including the land continuing beyond the frame.
MAINLAND_WEST_COAST = [
    (-130.36, 55.00),
    (-130.44, 54.92),
    (-130.34, 54.86),
    (-130.50, 54.78),
    (-130.38, 54.70),
    (-130.58, 54.63),
    (-130.48, 54.55),
    (-130.65, 54.48),
    (-130.54, 54.40),
    (-130.73, 54.34),
    (-130.61, 54.27),
    (-130.82, 54.20),
    (-130.72, 54.13),
    (-130.86, 54.05),
    (-130.74, 53.98),
    (-130.92, 53.91),
    (-130.79, 53.84),
    (-130.94, 53.77),
    (-130.76, 53.70),
    (-130.90, 53.62),
    (-130.73, 53.55),
    (-130.85, 53.48),
    (-130.68, 53.41),
    (-130.78, 53.34),
    (-130.57, 53.27),
    (-130.49, 53.20),
    (-130.33, 53.14),
    (-130.19, 53.07),
    (-130.08, 53.00),
]

def land_polygon_from_west_coast(coast):
    """Close a mainland shoreline polygon along the map's north/east/south edges."""
    return coast + [
        (EAST, SOUTH),
        (EAST, NORTH),
    ]


POLYGONS = [
    land_polygon_from_west_coast(MAINLAND_WEST_COAST),

    # Approximate outer eastern tips of Haida Gwaii that may reach the
    # western edge of this frame. Most of the archipelago lies west of it.
    [
        (-131.000, 54.23),
        (-130.985, 54.20),
        (-130.988, 54.16),
        (-131.000, 54.13),
    ],
    [
        (-131.000, 53.58),
        (-130.982, 53.55),
        (-130.986, 53.51),
        (-131.000, 53.48),
    ],

    # Small coastal islands and island groups, generalized from regional
    # geographic knowledge. These deliberately remain rough approximations.
    [
        (-130.44, 54.82),
        (-130.38, 54.83),
        (-130.35, 54.78),
        (-130.39, 54.75),
        (-130.45, 54.77),
    ],
    [
        (-130.56, 54.63),
        (-130.49, 54.65),
        (-130.45, 54.60),
        (-130.50, 54.56),
        (-130.57, 54.58),
    ],
    [
        (-130.69, 54.46),
        (-130.63, 54.48),
        (-130.59, 54.43),
        (-130.63, 54.39),
        (-130.70, 54.41),
    ],
    [
        (-130.77, 54.27),
        (-130.71, 54.29),
        (-130.67, 54.24),
        (-130.71, 54.20),
        (-130.78, 54.22),
    ],
    [
        (-130.78, 53.94),
        (-130.72, 53.96),
        (-130.68, 53.91),
        (-130.73, 53.87),
        (-130.79, 53.89),
    ],
    [
        (-130.83, 53.72),
        (-130.76, 53.74),
        (-130.72, 53.69),
        (-130.77, 53.65),
        (-130.84, 53.67),
    ],
    [
        (-130.70, 53.47),
        (-130.64, 53.49),
        (-130.60, 53.44),
        (-130.64, 53.40),
        (-130.71, 53.42),
    ],
    [
        (-130.56, 53.28),
        (-130.50, 53.30),
        (-130.46, 53.25),
        (-130.50, 53.21),
        (-130.57, 53.23),
    ],
]


def point_on_segment(x, y, ax, ay, bx, by, epsilon=1e-12):
    cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
    if abs(cross) > epsilon:
        return False
    dot = (x - ax) * (bx - ax) + (y - ay) * (by - ay)
    if dot < -epsilon:
        return False
    length_squared = (bx - ax) ** 2 + (by - ay) ** 2
    return dot <= length_squared + epsilon


def point_in_polygon(x, y, polygon):
    """Ray-casting point test; boundary points are treated as land."""
    inside = False
    for i, (ax, ay) in enumerate(polygon):
        bx, by = polygon[(i + 1) % len(polygon)]

        if point_on_segment(x, y, ax, ay, bx, by):
            return True

        if (ay > y) != (by > y):
            crossing_x = ax + (y - ay) * (bx - ax) / (by - ay)
            if x < crossing_x:
                inside = not inside
    return inside


def is_land(lon, lat):
    return any(point_in_polygon(lon, lat, polygon) for polygon in POLYGONS)


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Rows run north to south; columns run west to east.
    matrix = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for row in range(HEIGHT):
        lat = NORTH - row * STEP
        for col in range(WIDTH):
            lon = WEST + col * STEP
            matrix[row, col] = 1 if is_land(lon, lat) else 0

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", encoding="ascii", newline="") as f:
        for row in matrix:
            f.write(",".join(str(int(value)) for value in row) + "\n")

    # Land is white and water is black.
    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(output_dir / "candidate.png")

    enlarged = image.resize(
        (WIDTH * 8, HEIGHT * 8),
        resample=Image.Resampling.NEAREST,
    )
    enlarged.save(output_dir / "candidate_8x.png")

    # Validate the generated artifacts.
    assert matrix.shape == (127, 127)
    assert matrix.size == 16129
    assert set(np.unique(matrix)).issubset({0, 1})

    with csv_path.open("r", encoding="ascii") as f:
        lines = f.read().splitlines()
    assert len(lines) == 127
    assert all(
        len(line.split(",")) == 127
        and all(value in {"0", "1"} for value in line.split(","))
        for line in lines
    )

    with Image.open(output_dir / "candidate.png") as original:
        assert original.size == (127, 127)
        assert set(original.getdata()).issubset({(0, 0, 0), (255, 255, 255)})

    with Image.open(output_dir / "candidate_8x.png") as enlarged_check:
        assert enlarged_check.size == (1016, 1016)
        assert set(enlarged_check.getdata()).issubset(
            {(0, 0, 0), (255, 255, 255)}
        )

    print(f"Wrote {csv_path}")
    print(f"Wrote {output_dir / 'candidate.png'}")
    print(f"Wrote {output_dir / 'candidate_8x.png'}")
    print("Validated 127x127 matrix, 16,129 binary values, and image dimensions.")


if __name__ == "__main__":
    main()
