from pathlib import Path
import csv

import numpy as np
from PIL import Image


# Hand-estimated longitude of the mainland coast, from south to north.
# The mainland occupies the area east of this irregular boundary.
COAST = [
    (53.00, -129.90), (53.08, -130.02), (53.16, -130.12),
    (53.25, -130.05), (53.34, -130.20), (53.43, -130.27),
    (53.52, -130.15), (53.61, -130.34), (53.70, -130.25),
    (53.79, -130.43), (53.88, -130.35), (53.97, -130.55),
    (54.06, -130.47), (54.15, -130.64), (54.24, -130.50),
    (54.33, -130.70), (54.42, -130.53), (54.51, -130.75),
    (54.60, -130.59), (54.69, -130.82), (54.78, -130.65),
    (54.87, -130.78), (54.96, -130.55), (55.00, -130.48),
]


def coast_lon(lat):
    for (lat0, lon0), (lat1, lon1) in zip(COAST, COAST[1:]):
        if lat0 <= lat <= lat1:
            t = (lat - lat0) / (lat1 - lat0)
            return lon0 + t * (lon1 - lon0)
    return COAST[0][1] if lat < COAST[0][0] else COAST[-1][1]


# Water polygons cut into the mainland: approximate inlets and fjords.
INLETS = [
    [(-130.20, 53.02), (-130.12, 53.14), (-129.95, 53.19),
     (-129.88, 53.15), (-129.89, 53.06), (-130.02, 53.00)],
    [(-130.38, 53.57), (-130.29, 53.68), (-130.10, 53.75),
     (-130.04, 53.83), (-130.17, 53.88), (-130.39, 53.74)],
    [(-130.58, 54.03), (-130.48, 54.12), (-130.29, 54.17),
     (-130.21, 54.25), (-130.34, 54.30), (-130.57, 54.20)],
    [(-130.66, 54.43), (-130.56, 54.51), (-130.37, 54.55),
     (-130.31, 54.63), (-130.44, 54.68), (-130.67, 54.58)],
    [(-130.56, 54.79), (-130.46, 54.86), (-130.28, 54.90),
     (-130.22, 54.98), (-130.36, 55.00), (-130.57, 54.93)],
]

# Island and reef islets approximated as hand-drawn polygons; clipped by frame.
ISLANDS = [
    [(-131.05, 53.02), (-131.00, 53.02), (-130.98, 53.10),
     (-131.00, 53.20), (-131.04, 53.28), (-131.08, 53.23),
     (-131.06, 53.12)],
    [(-131.02, 53.25), (-130.98, 53.29), (-130.97, 53.39),
     (-131.00, 53.48), (-131.05, 53.53), (-131.08, 53.47),
     (-131.05, 53.37)],
    [(-131.04, 53.50), (-131.00, 53.55), (-130.99, 53.69),
     (-131.03, 53.78), (-131.08, 53.80), (-131.10, 53.70),
     (-131.07, 53.59)],
    [(-131.08, 53.78), (-131.04, 53.84), (-131.03, 53.99),
     (-131.07, 54.10), (-131.12, 54.15), (-131.15, 54.08),
     (-131.12, 53.94)],
    [(-130.91, 53.12), (-130.86, 53.13), (-130.84, 53.18),
     (-130.90, 53.20), (-130.93, 53.17)],
    [(-130.80, 53.39), (-130.75, 53.40), (-130.73, 53.46),
     (-130.79, 53.48), (-130.82, 53.44)],
    [(-130.95, 53.88), (-130.89, 53.89), (-130.88, 53.96),
     (-130.94, 53.99), (-130.98, 53.94)],
    [(-131.00, 54.27), (-130.96, 54.29), (-130.94, 54.35),
     (-131.00, 54.38), (-131.04, 54.33)],
    [(-130.83, 54.67), (-130.77, 54.68), (-130.75, 54.74),
     (-130.81, 54.77), (-130.86, 54.73)],
    [(-130.63, 53.26), (-130.58, 53.27), (-130.56, 53.32),
     (-130.61, 53.35), (-130.66, 53.31)],
    [(-130.72, 54.09), (-130.67, 54.10), (-130.65, 54.16),
     (-130.71, 54.19), (-130.76, 54.14)],
]


def on_land(lon, lat):
    if lon >= coast_lon(lat):
        land = True
        if any(point_in_polygon(lon, lat, p) for p in INLETS):
            land = False
    else:
        land = any(point_in_polygon(lon, lat, p) for p in ISLANDS)
    return land


def point_in_polygon(x, y, polygon):
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < cross:
                inside = not inside
        j = i
    return inside


def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    lats = np.linspace(55.0, 53.0, 127)
    lons = np.linspace(-131.0, -129.0, 127)
    matrix = np.array([[int(on_land(float(x), float(y))) for x in lons]
                       for y in lats], dtype=np.uint8)
    assert matrix.shape == (127, 127)
    with (out / "candidate.csv").open("w", newline="") as f:
        csv.writer(f).writerows(matrix.tolist())
    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(out / "candidate.png")
    image.resize((1016, 1016), Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )


if __name__ == "__main__":
    main()
