from pathlib import Path
import csv
import numpy as np
from PIL import Image


N = 127
STEP = 2.0 / 126.0
OUT = Path("outputs")

# Hand-drawn geographic approximation. Coordinates are (longitude, latitude).
# The mainland polygon follows the irregular outer coast; the remaining listed
# polygons represent prominent islands and coastal island groups in this box.
MAINLAND = [
    (-129.02, 55.00), (-129.30, 54.96), (-129.46, 54.88),
    (-129.55, 54.78), (-129.52, 54.69), (-129.62, 54.61),
    (-129.64, 54.52), (-129.57, 54.43), (-129.58, 54.34),
    (-129.70, 54.26), (-129.67, 54.17), (-129.75, 54.08),
    (-129.78, 53.98), (-129.69, 53.89), (-129.78, 53.81),
    (-129.76, 53.72), (-129.66, 53.64), (-129.72, 53.55),
    (-129.65, 53.46), (-129.62, 53.35), (-129.53, 53.27),
    (-129.48, 53.15), (-129.42, 53.00), (-128.50, 53.00),
    (-128.50, 55.00),
]

ISLANDS = [
    # Banks Island, with an uneven north-south outline.
    [(-130.17, 54.27), (-130.04, 54.28), (-129.96, 54.24),
     (-129.91, 54.16), (-129.91, 54.07), (-129.96, 53.99),
     (-129.99, 53.92), (-130.09, 53.90), (-130.17, 53.96),
     (-130.20, 54.05), (-130.20, 54.15)],
    # Smaller islands around the north and west shores of Banks Island.
    [(-130.36, 54.19), (-130.29, 54.22), (-130.24, 54.18),
     (-130.26, 54.11), (-130.33, 54.10), (-130.38, 54.14)],
    [(-130.42, 54.03), (-130.36, 54.06), (-130.32, 54.01),
     (-130.35, 53.96), (-130.41, 53.97)],
    # Island clusters north of Banks Island and near the mainland channels.
    [(-130.02, 54.43), (-129.94, 54.44), (-129.89, 54.39),
     (-129.93, 54.34), (-130.01, 54.35)],
    [(-129.91, 54.50), (-129.84, 54.53), (-129.79, 54.49),
     (-129.81, 54.44), (-129.88, 54.44)],
    [(-130.15, 53.79), (-130.08, 53.81), (-130.03, 53.77),
     (-130.05, 53.72), (-130.12, 53.73)],
    [(-129.99, 53.67), (-129.92, 53.70), (-129.87, 53.66),
     (-129.90, 53.61), (-129.96, 53.61)],
    # Pitt Island and adjacent islands enter the southern edge of the box.
    [(-129.68, 53.57), (-129.55, 53.58), (-129.48, 53.53),
     (-129.46, 53.46), (-129.50, 53.40), (-129.57, 53.37),
     (-129.64, 53.41), (-129.70, 53.48)],
    [(-129.88, 53.47), (-129.81, 53.50), (-129.76, 53.46),
     (-129.78, 53.40), (-129.85, 53.40)],
    [(-130.07, 53.40), (-129.99, 53.43), (-129.94, 53.39),
     (-129.96, 53.33), (-130.04, 53.34)],
    # Eastern edge of Haida Gwaii: only small coastal portions approach 131 W.
    [(-131.00, 54.48), (-130.96, 54.53), (-130.91, 54.52),
     (-130.88, 54.46), (-130.91, 54.39), (-130.97, 54.36),
     (-131.00, 54.39)],
]


def on_land(lon, lat):
    """Return 1 for a point inside a hand-drawn land polygon, else 0."""
    for polygon in [MAINLAND, *ISLANDS]:
        inside = False
        j = len(polygon) - 1
        for i, (xi, yi) in enumerate(polygon):
            xj, yj = polygon[j]
            if (yi > lat) != (yj > lat):
                crossing_lon = (xj - xi) * (lat - yi) / (yj - yi) + xi
                if lon < crossing_lon:
                    inside = not inside
            j = i
        if inside:
            return 1
    return 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Avoid accumulated decimal drift; rows run north-to-south and columns
    # west-to-east, matching the requested sampling grid.
    matrix = np.zeros((N, N), dtype=np.uint8)
    for row in range(N):
        lat = 55.0 - row * STEP
        for col in range(N):
            lon = -131.0 + col * STEP
            matrix[row, col] = on_land(lon, lat)

    with (OUT / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    pixels = np.where(matrix == 1, 255, 0).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    image.save(OUT / "candidate.png")
    image.resize((N * 8, N * 8), resample=Image.Resampling.NEAREST).save(
        OUT / "candidate_8x.png"
    )


if __name__ == "__main__":
    main()
