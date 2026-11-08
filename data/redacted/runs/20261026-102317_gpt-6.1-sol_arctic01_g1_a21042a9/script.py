from pathlib import Path

import numpy as np
from PIL import Image


OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

N_ROWS = 61
N_COLS = 241


def polygon(points):
    """Return a closed polygon as (longitude, latitude) pairs."""
    return tuple(points)


def inside_polygon(x, y, points):
    # Ray casting is sufficient for the hand-written, non-self-intersecting
    # coastline approximations used below.
    hit = False
    j = len(points) - 1
    for i in range(len(points)):
        xi, yi = points[i]
        xj, yj = points[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            edge_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < edge_x:
                hit = not hit
        j = i
    return hit


def any_polygon(lon, lat, shapes):
    return any(inside_polygon(lon, lat, shape) for shape in shapes)


def make_grid():
    longitudes = [-105.0 + 0.1 * col for col in range(N_COLS)]
    latitudes = [79.0 - 0.1 * row for row in range(N_ROWS)]
    grid = np.zeros((N_ROWS, N_COLS), dtype=np.uint8)
    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            grid[row, col] = int(is_land(lon, lat))
    return grid


def save_outputs(grid):
    if grid.shape != (N_ROWS, N_COLS):
        raise AssertionError(f"unexpected shape: {grid.shape}")
    if not np.isin(grid, (0, 1)).all():
        raise AssertionError("grid contains values other than 0 and 1")

    with (OUT / "candidate.csv").open("w", encoding="ascii", newline="") as fh:
        for row in grid:
            fh.write(",".join(str(int(value)) for value in row))
            fh.write("\n")

    # 0 is water (black), 1 is land (white).
    original = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
    original.save(OUT / "candidate.png")
    enlarged = original.resize((N_COLS * 8, N_ROWS * 8), Image.Resampling.NEAREST)
    enlarged.save(OUT / "candidate_8x.png")



# The following outlines are deliberately approximate, hand-written from
# geographic knowledge.  Coordinates are longitude, latitude.
LAND = []
WATER = []


def add_land(*points):
    LAND.append(polygon(points))


def add_water(*points):
    WATER.append(polygon(points))


# Melville Island, including its eastern shore and the broken southern coast.
add_land(
    (-105.0, 78.15), (-103.0, 78.35), (-100.5, 78.55),
    (-97.0, 78.45), (-94.5, 78.05), (-92.0, 77.45),
    (-91.4, 76.65), (-92.6, 75.85), (-95.0, 75.55),
    (-98.0, 75.65), (-101.0, 76.05), (-103.8, 76.55),
    (-105.0, 77.0),
)

# Bathurst Island.
add_land(
    (-102.6, 76.65), (-100.8, 76.9), (-97.2, 76.75),
    (-94.8, 76.2), (-94.0, 75.55), (-95.0, 75.05),
    (-98.0, 74.95), (-100.6, 75.35), (-102.3, 75.85),
)

# Cornwallis Island.
add_land(
    (-96.0, 76.45), (-93.5, 76.5), (-91.8, 76.0),
    (-91.5, 75.35), (-92.6, 74.8), (-94.8, 74.65),
    (-96.3, 75.1), (-96.6, 75.8),
)

# Devon Island, west and central portions in the requested frame.
add_land(
    (-91.2, 77.0), (-89.0, 77.55), (-85.5, 77.85),
    (-82.0, 77.6), (-80.8, 76.75), (-80.8, 74.8),
    (-83.0, 74.35), (-86.5, 74.15), (-89.7, 74.45),
    (-91.5, 75.1), (-90.8, 76.0),
)

# Axel Heiberg Island.
add_land(
    (-96.0, 79.0), (-95.0, 79.85), (-93.7, 80.55),
    (-91.2, 80.7), (-89.2, 80.1), (-89.0, 79.2),
    (-90.1, 78.5), (-92.0, 78.0), (-94.3, 78.25),
)

# Western Ellesmere Island and its southern projection.
add_land(
    (-90.7, 79.35), (-90.0, 80.0), (-88.0, 80.25),
    (-84.5, 80.15), (-81.0, 79.7), (-80.8, 78.3),
    (-82.0, 77.35), (-84.8, 77.0), (-87.7, 77.25),
    (-90.0, 78.0), (-91.0, 78.7),
)

# Prince of Wales Island.
add_land(
    (-103.4, 75.05), (-101.0, 75.45), (-97.5, 75.1),
    (-94.2, 74.25), (-91.7, 73.35), (-92.6, 72.6),
    (-95.5, 72.1), (-99.0, 72.1), (-101.8, 72.75),
    (-103.6, 73.65),
)

# Somerset Island.
add_land(
    (-95.0, 74.1), (-92.9, 74.45), (-89.5, 74.0),
    (-86.6, 73.25), (-86.7, 72.25), (-89.0, 71.65),
    (-92.0, 71.75), (-94.8, 72.4), (-95.7, 73.3),
)

# Boothia Peninsula and the northern mainland edge.
add_land(
    (-96.9, 73.05), (-95.8, 73.55), (-93.2, 73.6),
    (-91.0, 73.15), (-90.2, 72.35), (-91.0, 71.65),
    (-93.2, 71.15), (-95.0, 71.25), (-96.6, 72.0),
)

# Victoria Island's northern edge just enters the southern border of the box.
add_land(
    (-105.0, 73.0), (-103.5, 73.25), (-101.5, 73.05),
    (-100.0, 72.65), (-100.2, 72.25), (-102.0, 72.05),
    (-104.2, 72.25), (-105.0, 72.55),
)
# Smaller islands and exposed land fragments.
add_land(
    (-104.0, 74.65), (-102.8, 74.9), (-101.8, 74.6),
    (-102.0, 73.95), (-103.0, 73.75), (-104.0, 74.0),
)
add_land(
    (-100.0, 74.25), (-98.8, 74.45), (-97.8, 74.05),
    (-98.1, 73.45), (-99.3, 73.35), (-100.2, 73.65),
)
add_land(
    (-91.5, 74.8), (-90.6, 75.0), (-89.9, 74.7),
    (-90.2, 74.2), (-91.2, 74.1), (-91.8, 74.4),
)
add_land(
    (-88.3, 74.0), (-87.1, 74.15), (-86.3, 73.7),
    (-86.7, 73.2), (-87.8, 73.15), (-88.5, 73.5),
)
add_land(
    (-84.8, 74.0), (-83.8, 74.3), (-82.7, 74.0),
    (-82.3, 73.35), (-83.2, 73.05), (-84.4, 73.3),
)

# Banks and small islands near the Franklin Strait and Larsen Sound.
add_land(
    (-99.2, 73.05), (-98.7, 73.2), (-98.2, 72.95),
    (-98.45, 72.65), (-99.0, 72.7),
)
add_land(
    (-93.3, 73.2), (-92.8, 73.35), (-92.35, 73.05),
    (-92.6, 72.75), (-93.2, 72.85),
)
add_land(
    (-87.3, 72.75), (-86.7, 72.9), (-86.3, 72.6),
    (-86.55, 72.3), (-87.1, 72.35),
)

# Land around Lancaster Sound's northern side.
add_land(
    (-91.0, 77.6), (-90.0, 77.9), (-89.3, 77.6),
    (-89.6, 77.15), (-90.7, 77.2),
)
add_land(
    (-86.4, 78.0), (-85.3, 78.35), (-84.3, 78.05),
    (-84.5, 77.6), (-85.7, 77.55),
)


def is_land(lon, lat):
    # Explicit water polygons represent channels and sounds.  They are
    # evaluated after land so that a channel never becomes a land bridge.
    return any_polygon(lon, lat, LAND) and not any_polygon(lon, lat, WATER)


# Water passages and broad sounds inside the hand-drawn land outlines.
# Lancaster Sound, Barrow Strait, and the Parry Channel.
add_water(
    (-105.0, 77.0), (-101.0, 76.55), (-97.0, 76.25),
    (-94.0, 76.0), (-91.0, 76.25), (-90.7, 75.55),
    (-93.8, 75.45), (-97.0, 75.8), (-101.0, 76.05),
    (-105.0, 76.3),
)
add_water(
    (-96.6, 75.75), (-94.5, 75.85), (-91.7, 75.55),
    (-90.5, 75.05), (-91.0, 74.6), (-93.0, 74.85),
    (-95.0, 75.1), (-96.8, 75.25),
)

# Wellington Channel and the approaches between Cornwallis, Devon and
# Somerset islands.
add_water(
    (-94.0, 76.45), (-92.5, 76.7), (-90.7, 76.3),
    (-90.3, 75.8), (-91.3, 75.4), (-92.4, 75.55),
    (-94.0, 75.8),
)
add_water(
    (-91.0, 75.1), (-89.5, 75.1), (-87.0, 74.8),
    (-86.0, 74.2), (-86.7, 73.7), (-88.8, 74.1),
    (-90.6, 74.45),
)

# Peel Sound, Franklin Strait, and the channels around Prince of Wales.
add_water(
    (-99.5, 75.1), (-97.3, 75.0), (-95.2, 74.5),
    (-94.0, 73.9), (-94.4, 73.2), (-96.0, 73.25),
    (-97.2, 73.9), (-99.5, 74.25),
)
add_water(
    (-96.4, 73.45), (-94.9, 73.3), (-92.7, 72.9),
    (-91.3, 72.4), (-91.7, 71.9), (-93.5, 72.0),
    (-95.5, 72.55), (-96.7, 73.0),
)
# M'Clintock Channel and Larsen Sound.
add_water(
    (-102.0, 74.9), (-100.0, 75.0), (-98.5, 74.7),
    (-97.2, 74.2), (-97.5, 73.7), (-99.0, 73.9),
    (-100.5, 74.25), (-102.3, 74.35),
)
add_water(
    (-92.5, 74.65), (-90.5, 74.8), (-88.0, 74.45),
    (-86.0, 73.95), (-85.5, 73.45), (-87.1, 73.55),
    (-89.0, 73.95), (-91.0, 74.05), (-92.7, 74.25),
)

# Bellot Strait and the Gulf of Boothia approach.
add_water(
    (-95.4, 72.55), (-94.2, 72.7), (-93.0, 72.5),
    (-92.0, 72.15), (-92.3, 71.85), (-93.3, 71.95),
    (-94.6, 72.1), (-95.5, 72.3),
)
add_water(
    (-91.3, 73.0), (-89.7, 73.05), (-88.1, 72.75),
    (-87.3, 72.3), (-88.2, 71.9), (-89.7, 72.0),
    (-91.0, 72.35),
)


if __name__ == "__main__":
    result = make_grid()
    save_outputs(result)
    assert result.size == 14701
    print(f"generated {result.shape[0]}x{result.shape[1]} grid")

