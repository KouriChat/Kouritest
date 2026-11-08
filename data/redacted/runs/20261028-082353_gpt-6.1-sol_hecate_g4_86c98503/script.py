from pathlib import Path
import csv
import numpy as np
from PIL import Image

# Hand-drawn geographic approximations, coordinates are (longitude, latitude).
# Mainland west coast of northern British Columbia, including fjord mouths.
MAINLAND = [
    (-128.72, 55.00), (-129.06, 54.94), (-129.16, 54.84),
    (-129.38, 54.80), (-129.27, 54.70), (-129.48, 54.62),
    (-129.33, 54.54), (-129.60, 54.46), (-129.47, 54.37),
    (-129.72, 54.29), (-129.57, 54.20), (-129.82, 54.10),
    (-129.68, 54.02), (-129.91, 53.93), (-129.76, 53.84),
    (-129.96, 53.75), (-129.81, 53.66), (-129.98, 53.56),
    (-129.82, 53.47), (-129.96, 53.37), (-129.75, 53.29),
    (-129.84, 53.20), (-129.58, 53.12), (-129.55, 53.00),
    (-128.45, 53.00), (-128.42, 54.05), (-128.56, 54.58),
    (-128.65, 54.80), (-128.72, 55.00),
]

# Major inlets cut into the mainland coast; each is a water polygon.
INLETS = [
    [(-129.24,54.85),(-128.95,54.89),(-128.82,54.78),(-129.12,54.75)],
    [(-129.48,54.61),(-129.12,54.66),(-129.04,54.55),(-129.37,54.51)],
    [(-129.66,54.34),(-129.30,54.40),(-129.20,54.30),(-129.55,54.24)],
    [(-129.79,54.08),(-129.43,54.15),(-129.34,54.05),(-129.67,53.99)],
    [(-129.88,53.78),(-129.53,53.85),(-129.45,53.75),(-129.79,53.68)],
    [(-129.86,53.48),(-129.53,53.55),(-129.45,53.45),(-129.78,53.38)],
]
# Island and reef/islet outlines in the eastern Hecate Strait.
ISLANDS = [
    [(-129.72,54.30),(-129.57,54.33),(-129.48,54.25),(-129.52,54.13),
     (-129.61,54.09),(-129.70,54.17)],
    [(-129.86,54.10),(-129.76,54.14),(-129.69,54.07),(-129.73,53.99),
     (-129.82,53.97),(-129.88,54.03)],
    [(-129.73,53.88),(-129.56,53.91),(-129.49,53.83),(-129.52,53.69),
     (-129.62,53.62),(-129.74,53.70),(-129.78,53.81)],
    [(-129.96,53.73),(-129.83,53.77),(-129.78,53.69),(-129.84,53.60),
     (-129.94,53.63)],
    [(-129.69,53.50),(-129.54,53.52),(-129.48,53.43),(-129.54,53.34),
     (-129.66,53.33),(-129.73,53.41)],
    [(-129.92,53.40),(-129.81,53.43),(-129.75,53.36),(-129.80,53.28),
     (-129.91,53.30)],
    # Small offshore islands and reefs represented by compact polygons.
    [(-129.34,54.18),(-129.30,54.20),(-129.27,54.16),(-129.30,54.12)],
    [(-129.42,53.96),(-129.38,53.98),(-129.35,53.94),(-129.38,53.91)],
    [(-129.25,53.72),(-129.20,53.74),(-129.17,53.70),(-129.21,53.67)],
    [(-129.32,53.30),(-129.28,53.32),(-129.25,53.28),(-129.29,53.25)],
]


def inside(lon, lat, polygon):
    """Ray-crossing point-in-polygon test."""
    result = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > lat) != (yj > lat):
            crossing = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing:
                result = not result
        j = i
    return result


def is_land(lon, lat):
    if not inside(lon, lat, MAINLAND):
        return any(inside(lon, lat, island) for island in ISLANDS)
    return not any(inside(lon, lat, inlet) for inlet in INLETS)
def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the specified rational step to avoid cumulative coordinate drift.
    step = 2.0 / 126.0
    latitudes = [55.0 - row * step for row in range(127)]
    longitudes = [-131.0 + col * step for col in range(127)]
    matrix = np.array(
        [[1 if is_land(lon, lat) else 0 for lon in longitudes]
         for lat in latitudes],
        dtype=np.uint8,
    )

    with (output_dir / "candidate.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Pillow's L mode maps water to black and land to white.
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(output_dir / "candidate.png")
    image.resize((1016, 1016), resample=Image.Resampling.NEAREST).save(
        output_dir / "candidate_8x.png"
    )


if __name__ == "__main__":
    main()
