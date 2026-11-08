from pathlib import Path

import numpy as np
from PIL import Image


# The coastline below is an intentionally hand-authored approximation based on
# geographic knowledge.  Coordinates are longitude, latitude in WGS84 degrees.
# Each polygon is treated as land; points outside all polygons are water.

def mainland_polygon():
    # West edge is the approximate mainland coast.  The other edge is the
    # eastern boundary of the requested frame.
    coast = [
        (-129.75, 55.00),
        (-129.85, 54.85),
        (-129.72, 54.70),
        (-129.95, 54.55),
        (-129.83, 54.40),
        (-130.05, 54.28),
        (-129.75, 54.12),
        (-129.60, 53.98),
        (-129.48, 53.78),
        (-129.30, 53.58),
        (-129.18, 53.38),
        (-129.08, 53.18),
        (-129.10, 53.00),
    ]
    return coast + [(-129.00, 53.00), (-129.00, 55.00)]


def land_polygons():
    polygons = [mainland_polygon()]

    # The visible eastern edge of Graham Island.  Most of Haida Gwaii lies
    # west of the frame; this narrow clipped polygon retains its edge.
    polygons.append([
        (-131.20, 53.15), (-131.02, 53.24), (-130.98, 53.42),
        (-131.00, 53.75), (-130.99, 54.00), (-130.96, 54.18),
        (-131.05, 54.28), (-131.20, 54.25),
    ])

    # Northern islands around Dixon Entrance and Chatham Sound.
    polygons.extend([
        [
            (-131.00, 54.57), (-130.72, 54.58), (-130.55, 54.76),
            (-130.68, 54.94), (-130.94, 54.93), (-131.03, 54.76),
        ],
        [
            (-130.63, 54.64), (-130.38, 54.67), (-130.27, 54.79),
            (-130.45, 54.87), (-130.63, 54.77),
        ],
        [
            (-130.52, 54.18), (-130.22, 54.17), (-130.13, 54.36),
            (-130.34, 54.46), (-130.54, 54.35),
        ],
        [
            (-130.44, 54.05), (-130.26, 54.10), (-130.18, 54.23),
            (-130.38, 54.27), (-130.54, 54.18),
        ],
        [
            (-130.73, 53.88), (-130.52, 54.08), (-130.30, 54.08),
            (-130.25, 53.91), (-130.43, 53.76), (-130.67, 53.77),
        ],
        [
            (-130.26, 53.77), (-130.10, 54.02), (-129.88, 53.98),
            (-129.80, 53.78), (-130.03, 53.62), (-130.25, 53.64),
        ],
    ])

    # The larger island group south of Prince Rupert.
    polygons.extend([
        [
            (-130.15, 53.40), (-129.86, 53.70), (-129.56, 53.67),
            (-129.46, 53.38), (-129.68, 53.10), (-129.99, 53.20),
        ],
        [
            (-129.98, 53.98), (-129.63, 54.00), (-129.35, 53.86),
            (-129.15, 53.64), (-129.18, 53.33), (-129.41, 53.14),
            (-129.72, 53.08), (-129.91, 53.28), (-130.06, 53.61),
        ],
        [
            (-129.70, 53.72), (-129.46, 53.85), (-129.18, 53.72),
            (-129.13, 53.45), (-129.37, 53.22), (-129.65, 53.31),
        ],
        [
            (-129.62, 53.18), (-129.36, 53.32), (-129.16, 53.20),
            (-129.15, 53.02), (-129.47, 53.00), (-129.72, 53.08),
        ],
    ])

    return polygons

def mark_polygon(x_grid, y_grid, polygon):
    """Vectorized odd-even point-in-polygon test for one hand-written outline."""
    px = np.asarray([p[0] for p in polygon], dtype=float)
    py = np.asarray([p[1] for p in polygon], dtype=float)
    inside = np.zeros(x_grid.shape, dtype=bool)

    j = len(polygon) - 1
    for i in range(len(polygon)):
        yi, yj = py[i], py[j]
        if yi != yj:
            crosses = ((yi > y_grid) != (yj > y_grid))
            x_at_y = (px[j] - px[i]) * (y_grid - yi) / (yj - yi) + px[i]
            inside ^= crosses & (x_grid < x_at_y)
        j = i
    return inside


def build_matrix():
    # Exact 2/126 degree spacing, north-to-south and west-to-east.
    step = 2.0 / 126.0
    latitudes = 55.0 - np.arange(127, dtype=float) * step
    longitudes = -131.0 + np.arange(127, dtype=float) * step
    x_grid, y_grid = np.meshgrid(longitudes, latitudes)

    land = np.zeros((127, 127), dtype=bool)
    for polygon in land_polygons():
        land |= mark_polygon(x_grid, y_grid, polygon)
    return land.astype(np.uint8)


def write_outputs(matrix):
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        output_dir / "candidate.csv",
        matrix,
        fmt="%d",
        delimiter=",",
    )

    # A uint8 mask makes the only image values exactly 0 and 255.
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(output_dir / "candidate.png")

    enlarged = image.resize((127 * 8, 127 * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")

def validate_outputs(matrix):
    assert matrix.shape == (127, 127)
    assert matrix.size == 16129
    assert set(np.unique(matrix).tolist()) <= {0, 1}

    output_dir = Path("outputs")
    rows = (output_dir / "candidate.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 127
    for row in rows:
        values = row.split(",")
        assert len(values) == 127
        assert all(value in {"0", "1"} for value in values)

    original = Image.open(output_dir / "candidate.png")
    enlarged = Image.open(output_dir / "candidate_8x.png")
    assert original.size == (127, 127)
    assert enlarged.size == (1016, 1016)
    assert set(original.convert("L").getdata()) <= {0, 255}
    assert set(enlarged.convert("L").getdata()) <= {0, 255}

    # Check each 8x8 block is identical to its source pixel, which verifies
    # the saved enlargement is nearest-neighbor at the requested scale.
    original_array = np.asarray(original.convert("L"))
    enlarged_array = np.asarray(enlarged.convert("L"))
    assert np.array_equal(
        enlarged_array.reshape(127, 8, 127, 8)[:, 0, :, 0],
        original_array,
    )
    assert np.all(
        enlarged_array.reshape(127, 8, 127, 8)
        == original_array[:, None, :, None]
    )


if __name__ == "__main__":
    matrix = build_matrix()
    write_outputs(matrix)
    validate_outputs(matrix)
    print("generated and validated 127x127 map and 8x enlargement")
