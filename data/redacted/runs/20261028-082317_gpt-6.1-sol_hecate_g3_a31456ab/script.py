from pathlib import Path
import csv
import numpy as np
from PIL import Image


# Hand-drawn geographic approximation in (longitude, latitude).
# Coordinates are intentionally approximate; no external geographic data is used.
MAINLAND = [
    (-129.00, 55.00), (-129.55, 55.00), (-129.65, 54.94),
    (-129.82, 54.91), (-129.73, 54.85), (-129.88, 54.82),
    (-129.78, 54.77), (-129.96, 54.73), (-129.83, 54.67),
    (-129.94, 54.62), (-129.79, 54.57), (-129.93, 54.52),
    (-129.79, 54.47), (-129.88, 54.41), (-129.74, 54.36),
    (-129.89, 54.31), (-129.77, 54.25), (-129.93, 54.20),
    (-129.80, 54.14), (-129.92, 54.09), (-129.78, 54.04),
    (-129.87, 53.98), (-129.70, 53.93), (-129.83, 53.87),
    (-129.72, 53.82), (-129.88, 53.77), (-129.75, 53.71),
    (-129.89, 53.66), (-129.72, 53.60), (-129.84, 53.55),
    (-129.70, 53.49), (-129.85, 53.44), (-129.72, 53.38),
    (-129.90, 53.33), (-129.75, 53.27), (-129.84, 53.21),
    (-129.67, 53.16), (-129.80, 53.10), (-129.62, 53.04),
    (-129.70, 53.00), (-129.00, 53.00),
]
MAINLAND += [
    (-129.00, 53.00), (-129.00, 55.00)
]

# Coastal islands and larger offshore islands within the requested window.
ISLANDS = [
    # Banks Island
    [(-130.34, 54.42), (-130.20, 54.48), (-130.04, 54.46),
     (-129.94, 54.37), (-129.88, 54.25), (-129.91, 54.08),
     (-130.02, 53.91), (-130.18, 53.82), (-130.34, 53.88),
     (-130.42, 54.03), (-130.40, 54.24)],
    # Pitt Island, clipped by southern frame
    [(-130.04, 53.55), (-129.91, 53.62), (-129.75, 53.58),
     (-129.66, 53.48), (-129.69, 53.30), (-129.81, 53.17),
     (-129.98, 53.16), (-130.11, 53.27), (-130.14, 53.43)],
    # Porcher Island
    [(-130.39, 53.08), (-130.23, 53.16), (-130.08, 53.11),
     (-130.05, 53.00), (-130.34, 53.00)],
    # Small islands near mainland and Banks Island
    [(-130.48, 54.43), (-130.38, 54.46), (-130.34, 54.39),
     (-130.40, 54.33), (-130.49, 54.36)],
    [(-130.25, 54.72), (-130.17, 54.76), (-130.11, 54.71),
     (-130.15, 54.64), (-130.23, 54.65)],
    [(-130.54, 53.88), (-130.47, 53.92), (-130.42, 53.86),
     (-130.47, 53.80), (-130.55, 53.82)],
    [(-130.64, 53.43), (-130.55, 53.49), (-130.49, 53.42),
     (-130.54, 53.35), (-130.63, 53.36)],
]

def inside_polygon(x, y, polygon):
    """Ray-crossing point test; polygon vertices are hand-authored above."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            crossing_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < crossing_x:
                inside = not inside
        j = i
    return inside


def main():
    rows, cols = 127, 127
    latitudes = 55.0 - np.arange(rows) * (2.0 / 126.0)
    longitudes = -131.0 + np.arange(cols) * (2.0 / 126.0)
    matrix = np.zeros((rows, cols), dtype=np.uint8)

    polygons = [MAINLAND, *ISLANDS]
    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            matrix[row, col] = int(
                any(inside_polygon(lon, lat, polygon) for polygon in polygons)
            )

    output = Path("outputs")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    image = Image.fromarray(np.where(matrix == 1, 255, 0).astype(np.uint8), mode="L")
    image.save(output / "candidate.png")
    image.resize((1016, 1016), Image.Resampling.NEAREST).save(
        output / "candidate_8x.png"
    )

    assert matrix.shape == (127, 127) and matrix.size == 16129
    assert set(np.unique(matrix)).issubset({0, 1})
    with (output / "candidate.csv").open(newline="", encoding="ascii") as f:
        readback = list(csv.reader(f))
    assert len(readback) == 127 and all(len(row) == 127 for row in readback)
    assert all(value in {"0", "1"} for row in readback for value in row)
    assert image.size == (127, 127) and set(image.getdata()).issubset({0, 255})
    enlarged = Image.open(output / "candidate_8x.png")
    assert enlarged.size == (1016, 1016)
    assert set(enlarged.getdata()).issubset({0, 255})


if __name__ == "__main__":
    main()
