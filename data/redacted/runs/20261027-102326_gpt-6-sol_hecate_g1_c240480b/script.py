"""Hand-sketched, approximate land-point map of the northern Hecate Strait.

No cartographic files or services are read; shapes are manually estimated.
Coordinates in the sketches below are (longitude east-positive, latitude).
"""
from pathlib import Path
import csv

import numpy as np
from PIL import Image


def inside(x, y, vertices):
    """Even-odd point-in-polygon test, with a consistent boundary convention."""
    hit = False
    x0, y0 = vertices[-1]
    for x1, y1 in vertices:
        if (y0 > y) != (y1 > y):
            edge_x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x <= edge_x:
                hit = not hit
        x0, y0 = x1, y1
    return hit


# Approximate ocean-facing edge of the mainland, sketched from memory.
# The detailed inlets are modeled separately as water cuts below.
mainland = [
    (-130.12, 55.15), (-130.09, 55.00), (-130.13, 54.91),
    (-130.25, 54.84), (-130.29, 54.77), (-130.39, 54.69),
    (-130.40, 54.62), (-130.30, 54.59), (-130.35, 54.52),
    (-130.28, 54.47), (-130.35, 54.39), (-130.27, 54.34),
    (-130.24, 54.27), (-130.09, 54.25), (-130.05, 54.18),
    (-129.90, 54.13), (-129.86, 54.04), (-129.74, 53.99),
    (-129.76, 53.91), (-129.60, 53.83), (-129.62, 53.77),
    (-129.45, 53.69), (-129.50, 53.62), (-129.40, 53.53),
    (-129.40, 53.47), (-129.29, 53.40), (-129.39, 53.32),
    (-129.33, 53.21), (-129.29, 53.10), (-129.34, 53.00),
    (-129.30, 52.85), (-128.70, 52.85), (-128.70, 55.15),
]

# Offshore archipelago. These schematic outlines are not sourced from maps.
islands = [
    # Dundas Island, north of the Prince Rupert approaches.
    [(-130.91, 54.62), (-130.80, 54.69), (-130.69, 54.67),
     (-130.61, 54.62), (-130.55, 54.53), (-130.62, 54.49),
     (-130.72, 54.51), (-130.83, 54.51), (-130.91, 54.56)],
    # Melville and neighboring islands.
    [(-130.78, 54.48), (-130.66, 54.47), (-130.59, 54.42),
     (-130.59, 54.37), (-130.68, 54.37), (-130.77, 54.42)],
    [(-130.52, 54.53), (-130.43, 54.51), (-130.40, 54.47),
     (-130.46, 54.44), (-130.52, 54.47)],
    # Stephens Island and smaller islands west of Kaien Island.
    [(-130.72, 54.38), (-130.61, 54.37), (-130.53, 54.30),
     (-130.49, 54.23), (-130.52, 54.17), (-130.58, 54.18),
     (-130.63, 54.24), (-130.70, 54.29)],
    [(-130.50, 54.36), (-130.39, 54.37), (-130.30, 54.32),
     (-130.29, 54.27), (-130.37, 54.24), (-130.49, 54.28)],
    # Porcher Island south of Prince Rupert.
    [(-130.63, 54.07), (-130.57, 54.15), (-130.42, 54.17),
     (-130.36, 54.12), (-130.27, 54.09), (-130.20, 54.00),
     (-130.19, 53.94), (-130.26, 53.87), (-130.36, 53.86),
     (-130.46, 53.91), (-130.55, 53.96), (-130.63, 54.01)],
    # Banks Island and Pitt Island, between Hecate Strait and the mainland.
    [(-130.74, 53.76), (-130.66, 53.84), (-130.55, 53.86),
     (-130.46, 53.81), (-130.43, 53.74), (-130.37, 53.67),
     (-130.38, 53.57), (-130.30, 53.46), (-130.29, 53.36),
     (-130.37, 53.31), (-130.46, 53.36), (-130.50, 53.46),
     (-130.57, 53.51), (-130.65, 53.63), (-130.72, 53.69)],
    [(-130.27, 54.03), (-130.17, 54.07), (-130.06, 54.04),
     (-129.99, 53.98), (-129.94, 53.86), (-129.92, 53.76),
     (-129.97, 53.69), (-130.05, 53.63), (-130.10, 53.54),
     (-130.16, 53.50), (-130.22, 53.58), (-130.24, 53.69),
     (-130.22, 53.78), (-130.25, 53.88)],
    # Smaller islands to the south and east of Banks Island.
    [(-130.19, 53.43), (-130.09, 53.47), (-129.99, 53.44),
     (-129.97, 53.38), (-130.03, 53.32), (-130.10, 53.31),
     (-130.17, 53.36)],
    [(-129.91, 53.58), (-129.80, 53.61), (-129.73, 53.56),
     (-129.74, 53.45), (-129.79, 53.37), (-129.87, 53.38),
     (-129.93, 53.46)],
    [(-129.77, 53.26), (-129.69, 53.30), (-129.60, 53.26),
     (-129.59, 53.17), (-129.65, 53.12), (-129.72, 53.15)],
]

# Navigable inlets and straits are water even inside the coarse coast polygon.
inlets = [
    [(-130.14, 54.96), (-130.02, 54.93), (-129.94, 54.88),
     (-129.91, 54.79), (-129.99, 54.78), (-130.10, 54.84)],
    [(-130.38, 54.62), (-130.26, 54.61), (-130.13, 54.55),
     (-130.00, 54.47), (-129.92, 54.42), (-129.95, 54.37),
     (-130.09, 54.45), (-130.23, 54.54), (-130.39, 54.57)],
    [(-130.26, 54.28), (-130.13, 54.27), (-130.02, 54.22),
     (-129.81, 54.18), (-129.72, 54.12), (-129.75, 54.07),
     (-129.87, 54.12), (-130.07, 54.17), (-130.26, 54.22)],
    [(-129.76, 53.94), (-129.64, 53.94), (-129.49, 53.88),
     (-129.36, 53.80), (-129.32, 53.70), (-129.40, 53.69),
     (-129.53, 53.80), (-129.66, 53.87)],
    [(-129.47, 53.53), (-129.38, 53.51), (-129.22, 53.44),
     (-129.09, 53.36), (-129.00, 53.30), (-129.00, 53.18),
     (-129.15, 53.33), (-129.34, 53.42), (-129.44, 53.47)],
    [(-129.39, 53.21), (-129.27, 53.20), (-129.14, 53.13),
     (-129.04, 53.07), (-129.00, 53.07), (-129.00, 53.01),
     (-129.13, 53.06), (-129.28, 53.14), (-129.38, 53.16)],
]


def make_matrix():
    # Row 0 is north, column 0 is west. Point locations, not cell averages.
    a = np.zeros((127, 127), dtype=np.uint8)
    for row in range(127):
        lat = 55.0 - 2.0 * row / 126.0
        for col in range(127):
            lon = -131.0 + 2.0 * col / 126.0
            land = inside(lon, lat, mainland) or any(
                inside(lon, lat, polygon) for polygon in islands
            )
            if land and any(inside(lon, lat, polygon) for polygon in inlets):
                land = False
            a[row, col] = int(land)
    return a


def main():
    a = make_matrix()
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerows(a.tolist())
    picture = Image.fromarray(a * 255, mode="L")
    picture.save(out / "candidate.png")
    big = picture.resize((1016, 1016), resample=Image.Resampling.NEAREST)
    big.save(out / "candidate_8x.png")
    with (out / "candidate.csv").open(newline="", encoding="utf-8") as fp:
        rows = list(csv.reader(fp))
    assert len(rows) == 127 and all(
        len(r) == 127 and set(r) <= {"0", "1"} for r in rows
    )
    assert set(picture.getdata()) <= {0, 255}
    assert big.size == (1016, 1016) and set(big.getdata()) <= {0, 255}
    assert np.array_equal(np.asarray(big), np.repeat(np.repeat(a * 255, 8, 0), 8, 1))
    print("Validated 16129 point values, both binary PNGs, and nearest-neighbor scaling.")


if __name__ == "__main__":
    main()
