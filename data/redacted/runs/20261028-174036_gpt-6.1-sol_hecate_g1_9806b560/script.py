from pathlib import Path
import csv

import numpy as np
from PIL import Image

# Approximate hand-authored land polygons in (longitude, latitude).
# Longitudes are negative west. Polygon vertices are deliberately coarse.
LAND_POLYGONS = [
    # Mainland coast east of Hecate Strait and Dixon Entrance.
    [
        (-130.15, 55.00), (-129.00, 55.00), (-129.00, 53.00),
        (-130.15, 53.00), (-130.03, 53.12), (-130.12, 53.25),
        (-129.97, 53.38), (-130.08, 53.55), (-129.93, 53.70),
        (-130.05, 53.86), (-129.91, 54.03), (-130.08, 54.18),
        (-129.95, 54.34), (-130.12, 54.49), (-129.96, 54.64),
        (-130.10, 54.79), (-129.91, 54.90), (-130.15, 55.00),
    ],
    # Graham Island, the northern and largest part of Haida Gwaii.
    [
        (-132.00, 54.98), (-131.50, 55.00), (-131.18, 54.92),
        (-131.04, 54.79), (-131.07, 54.63), (-130.98, 54.47),
        (-131.08, 54.30), (-131.00, 54.12), (-131.10, 53.98),
        (-131.29, 53.94), (-131.45, 54.02), (-131.57, 54.18),
        (-131.73, 54.31), (-131.91, 54.46), (-132.00, 54.65),
        (-132.00, 54.98),
    ],
    # Moresby Island, separated from Graham Island by narrow Skidegate Inlet.
    [
        (-132.00, 53.98), (-131.79, 54.02), (-131.62, 53.93),
        (-131.49, 53.78), (-131.37, 53.61), (-131.28, 53.40),
        (-131.21, 53.18), (-131.10, 53.00), (-132.00, 53.00),
        (-132.00, 53.98),
    ],
    # Coastal islands and the outer edge of the archipelago.
    [
        (-131.00, 54.75), (-130.88, 54.80), (-130.83, 54.70),
        (-130.90, 54.60), (-131.02, 54.63), (-131.00, 54.75),
    ],
    [
        (-131.12, 54.37), (-131.00, 54.42), (-130.92, 54.34),
        (-130.97, 54.24), (-131.09, 54.26), (-131.12, 54.37),
    ],
    [
        (-131.31, 54.05), (-131.20, 54.08), (-131.14, 53.99),
        (-131.20, 53.91), (-131.30, 53.94), (-131.31, 54.05),
    ],
    [
        (-131.69, 53.36), (-131.57, 53.42), (-131.49, 53.35),
        (-131.54, 53.24), (-131.66, 53.25), (-131.69, 53.36),
    ],
    [
        (-131.92, 53.14), (-131.80, 53.19), (-131.72, 53.11),
        (-131.78, 53.02), (-131.89, 53.04), (-131.92, 53.14),
    ],
]

# Include small nearby islands and islets with compact hand-drawn polygons.
LAND_POLYGONS += [
    [(-130.74, 54.91), (-130.68, 54.94), (-130.64, 54.89),
     (-130.68, 54.84), (-130.73, 54.86), (-130.74, 54.91)],
    [(-130.58, 54.69), (-130.52, 54.72), (-130.48, 54.67),
     (-130.52, 54.62), (-130.57, 54.64), (-130.58, 54.69)],
    [(-130.83, 54.42), (-130.77, 54.45), (-130.73, 54.40),
     (-130.77, 54.35), (-130.82, 54.37), (-130.83, 54.42)],
    [(-130.55, 54.20), (-130.49, 54.23), (-130.45, 54.18),
     (-130.49, 54.13), (-130.54, 54.15), (-130.55, 54.20)],
    [(-130.72, 53.94), (-130.65, 53.97), (-130.61, 53.92),
     (-130.65, 53.87), (-130.71, 53.89), (-130.72, 53.94)],
    [(-130.41, 53.69), (-130.35, 53.72), (-130.31, 53.67),
     (-130.35, 53.62), (-130.40, 53.64), (-130.41, 53.69)],
    [(-130.68, 53.47), (-130.62, 53.50), (-130.58, 53.45),
     (-130.62, 53.40), (-130.67, 53.42), (-130.68, 53.47)],
    [(-130.43, 53.21), (-130.37, 53.24), (-130.33, 53.19),
     (-130.37, 53.14), (-130.42, 53.16), (-130.43, 53.21)],
]


def point_in_polygon(lon, lat, polygon):
    """Ray-crossing test; boundary points count as land."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # A point on an edge is assigned to land.
        cross = (lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
        if (abs(cross) < 1e-12
                and min(x1, x2) - 1e-12 <= lon <= max(x1, x2) + 1e-12
                and min(y1, y2) - 1e-12 <= lat <= max(y1, y2) + 1e-12):
            return True

        if (y1 > lat) != (y2 > lat):
            x_cross = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < x_cross:
                inside = not inside
    return inside


def main():
    # Use exact step 2/126 degrees and include both specified endpoints.
    latitudes = np.linspace(55.0, 53.0, 127)
    longitudes = np.linspace(-131.0, -129.0, 127)

    matrix = np.zeros((127, 127), dtype=np.uint8)
    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            matrix[row, col] = int(
                any(point_in_polygon(float(lon), float(lat), polygon)
                    for polygon in LAND_POLYGONS)
            )

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)

    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # 1 (land) is white; 0 (water) is black.
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(out / "candidate.png")
    image.resize((1016, 1016), resample=Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )

    assert matrix.shape == (127, 127)
    assert matrix.size == 16129
    assert set(np.unique(matrix)).issubset({0, 1})


if __name__ == "__main__":
    main()
