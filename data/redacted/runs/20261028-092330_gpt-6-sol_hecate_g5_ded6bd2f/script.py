from pathlib import Path

import numpy as np
from PIL import Image

N = 127
STEP = 2.0 / 126.0


def inside(x, y, polygon):
    """Even-odd point-in-polygon test for (west longitude, latitude)."""
    hit = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        crosses = ((yi > y) != (yj > y))
        if crosses:
            at_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < at_x:
                hit = not hit
        j = i
    return hit


def ellipse(x, y, cx, cy, rx, ry, angle=0.0):
    ca, sa = np.cos(angle), np.sin(angle)
    dx, dy = x - cx, y - cy
    u = (dx * ca + dy * sa) / rx
    v = (-dx * sa + dy * ca) / ry
    return u * u + v * v <= 1.0


def polygon_land(x, y, polygons):
    return any(inside(x, y, p) for p in polygons)


# The mainland is represented by a deliberately angular hand-drawn west coast.
# The eastern and southern edges simply close the portion that is inside the box.
MAINLAND = [
    (130.34, 55.00), (129.00, 55.00), (129.00, 53.00),
    (129.34, 53.00), (129.27, 53.11), (129.46, 53.24),
    (129.31, 53.37), (129.57, 53.48), (129.46, 53.60),
    (129.71, 53.72), (129.60, 53.84), (129.86, 53.96),
    (129.74, 54.08), (129.96, 54.18), (129.83, 54.30),
    (130.06, 54.41), (129.96, 54.53), (130.15, 54.63),
    (130.01, 54.73), (130.24, 54.83), (130.12, 54.95),
    (130.34, 55.00),
]

LAND_POLYGONS = [MAINLAND]
# Major and intermediate islands in the central and southern strait.
# Coordinates are approximate west-longitude/latitude vertices.
LAND_POLYGONS += [
    # Banks Island
    [(129.93, 53.56), (129.84, 53.51), (129.82, 53.40),
     (129.86, 53.29), (129.79, 53.18), (129.84, 53.08),
     (129.94, 53.09), (130.00, 53.19), (130.00, 53.34),
     (129.98, 53.47)],
    # Small islands close to the mainland shore
    [(129.72, 53.43), (129.67, 53.39), (129.68, 53.33),
     (129.75, 53.31), (129.78, 53.36)],
    [(129.80, 53.73), (129.75, 53.68), (129.77, 53.63),
     (129.83, 53.64), (129.85, 53.69)],
    [(129.91, 54.01), (129.86, 53.98), (129.87, 53.92),
     (129.93, 53.91), (129.96, 53.96)],
    [(130.05, 54.34), (129.99, 54.30), (130.01, 54.24),
     (130.07, 54.25), (130.10, 54.29)],
    [(130.22, 54.58), (130.16, 54.54), (130.17, 54.49),
     (130.23, 54.47), (130.26, 54.53)],
    # Aristazabal Island's northern tip and nearby islets at the southern edge
    [(130.47, 53.08), (130.42, 53.03), (130.45, 53.00),
     (130.51, 53.00), (130.54, 53.05)],
    [(130.32, 53.02), (130.29, 53.00), (130.35, 53.00)],
]

# The easternmost edge of Haida Gwaii is close to the west frame. Only the
# small portions that plausibly reach the sampled box are encoded here.
LAND_POLYGONS += [
    [(131.00, 54.45), (130.97, 54.40), (130.98, 54.33),
     (131.00, 54.30)],
    [(131.00, 53.93), (130.96, 53.88), (130.97, 53.81),
     (131.00, 53.77)],
]
# Northern Hecate Strait islands and reef/islet groups.
LAND_POLYGONS += [
    [(130.66, 54.99), (130.60, 54.96), (130.62, 54.90),
     (130.68, 54.88), (130.72, 54.93)],
    [(130.78, 54.83), (130.74, 54.79), (130.77, 54.74),
     (130.83, 54.76), (130.85, 54.80)],
    [(129.98, 54.62), (129.94, 54.59), (129.96, 54.55),
     (130.01, 54.56)],
    [(129.91, 54.77), (129.88, 54.74), (129.90, 54.70),
     (129.95, 54.71)],
    [(130.36, 54.22), (130.33, 54.19), (130.36, 54.15),
     (130.40, 54.17)],
    [(130.55, 54.01), (130.51, 53.98), (130.53, 53.94),
     (130.58, 53.96)],
    [(130.17, 53.84), (130.13, 53.82), (130.15, 53.77),
     (130.20, 53.78)],
    [(130.29, 53.60), (130.25, 53.58), (130.27, 53.54),
     (130.32, 53.56)],
    [(130.14, 53.34), (130.11, 53.31), (130.14, 53.28),
     (130.18, 53.30)],
    [(130.42, 53.37), (130.39, 53.34), (130.42, 53.31),
     (130.46, 53.33)],
]

ROCKS = [
    (130.12, 54.69, 0.010, 0.012, 0.2),
    (130.43, 54.48, 0.009, 0.011, -0.3),
    (130.64, 54.28, 0.010, 0.012, 0.4),
    (130.73, 54.12, 0.008, 0.010, 0.0),
    (130.02, 53.83, 0.010, 0.012, 0.1),
    (130.61, 53.71, 0.011, 0.010, -0.2),
    (130.37, 53.48, 0.008, 0.010, 0.0),
    (130.68, 53.25, 0.010, 0.011, 0.5),
    (130.08, 53.13, 0.009, 0.012, -0.4),
]
def is_land(x, y):
    if polygon_land(x, y, LAND_POLYGONS):
        return True
    return any(ellipse(x, y, *rock) for rock in ROCKS)


def main():
    # Column zero is 131 W, then longitude increases eastward.
    longitudes = 131.0 - np.arange(N) * STEP
    # Row zero is 55 N, then latitude decreases southward.
    latitudes = 55.0 - np.arange(N) * STEP
    matrix = np.zeros((N, N), dtype=np.uint8)
    for row, lat in enumerate(latitudes):
        for col, west_lon in enumerate(longitudes):
            matrix[row, col] = int(is_land(float(west_lon), float(lat)))

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", encoding="ascii", newline="") as f:
        for row in matrix:
            f.write(",".join(str(int(v)) for v in row) + "\n")

    pixels = np.where(matrix == 1, 255, 0).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    image.save(out / "candidate.png")
    image.resize((N * 8, N * 8), resample=Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )

    # Validate exported dimensions, values, and the nearest-neighbor relation.
    lines = (out / "candidate.csv").read_text(encoding="ascii").splitlines()
    assert len(lines) == N and all(len(line.split(",")) == N for line in lines)
    assert all(value in ("0", "1") for line in lines for value in line.split(","))
    assert matrix.size == 16129
    assert set(np.unique(np.asarray(Image.open(out / "candidate.png")))) <= {0, 255}
    enlarged = np.asarray(Image.open(out / "candidate_8x.png"))
    assert enlarged.shape == (1016, 1016)
    assert np.array_equal(enlarged, np.repeat(np.repeat(pixels, 8, 0), 8, 1))


if __name__ == "__main__":
    main()
