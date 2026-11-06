from pathlib import Path

import numpy as np
from PIL import Image


# Approximate hand-authored coastline polygons in WGS84 lon/lat.
# The northern edges of the mainland polygons extend beyond the sampled area.
LAND_POLYGONS = [
    # Kyushu, including its eastern shore near the southern edge of the frame.
    [
        (129.0, 35.5), (132.0, 35.5), (132.0, 33.55),
        (131.85, 33.45), (131.75, 33.30), (131.65, 33.15),
        (131.45, 33.02), (131.20, 32.95), (130.90, 32.95),
        (130.50, 33.00), (130.15, 33.05), (129.75, 33.20),
        (129.35, 33.50), (129.0, 33.80),
    ],
    # Honshu: southern coast, with broad indentations for Osaka and Ise bays.
    [
        (130.75, 35.5), (137.2, 35.5), (137.2, 34.95),
        (137.05, 34.90), (136.92, 34.83), (136.85, 34.72),
        (136.78, 34.58), (136.72, 34.45), (136.78, 34.33),
        (136.70, 34.22), (136.58, 34.10), (136.43, 33.95),
        (136.28, 33.78), (136.12, 33.60), (135.95, 33.48),
        (135.78, 33.42), (135.62, 33.48), (135.50, 33.62),
        (135.40, 33.82), (135.30, 34.02),
        # Osaka Bay's southern and western shore.
        (135.18, 34.17), (135.02, 34.24), (134.88, 34.27),
        # Inland Sea north shore, westward.
        (134.72, 34.35), (134.52, 34.43), (134.30, 34.48),
        (134.08, 34.52), (133.86, 34.53), (133.64, 34.49),
        (133.42, 34.43), (133.20, 34.38), (132.98, 34.33),
        (132.76, 34.29), (132.55, 34.26), (132.34, 34.22),
        (132.12, 34.15), (131.90, 34.08), (131.68, 34.02),
        (131.45, 33.98), (131.24, 33.98), (131.02, 34.02),
        (130.85, 34.08), (130.75, 34.18),
    ],
    # Shikoku.
    [
        (132.05, 34.10), (132.28, 34.16), (132.52, 34.18),
        (132.76, 34.16), (133.00, 34.15), (133.24, 34.17),
        (133.48, 34.15), (133.72, 34.12), (133.96, 34.08),
        (134.20, 34.03), (134.42, 33.99), (134.62, 33.94),
        (134.72, 33.83), (134.68, 33.67), (134.58, 33.50),
        (134.42, 33.37), (134.20, 33.27), (133.92, 33.23),
        (133.62, 33.22), (133.32, 33.24), (133.02, 33.25),
        (132.74, 33.27), (132.48, 33.31), (132.25, 33.38),
        (132.10, 33.50), (132.00, 33.70), (131.98, 33.91),
    ],
    # Awaji Island.
    [
        (134.78, 34.78), (134.94, 34.74), (135.02, 34.62),
        (135.00, 34.47), (134.93, 34.32), (134.83, 34.25),
        (134.74, 34.35), (134.72, 34.52), (134.73, 34.67),
    ],
    # Shodoshima.
    [
        (134.12, 34.60), (134.29, 34.61), (134.38, 34.55),
        (134.35, 34.47), (134.23, 34.44), (134.12, 34.48),
        (134.07, 34.54),
    ],
    # Geiyo Islands and nearby larger islands.
    [
        (132.65, 34.28), (132.76, 34.28), (132.80, 34.20),
        (132.75, 34.13), (132.66, 34.16),
    ],
    [
        (132.82, 34.25), (132.93, 34.24), (132.97, 34.16),
        (132.91, 34.10), (132.83, 34.15),
    ],
    [
        (133.02, 34.28), (133.13, 34.26), (133.16, 34.18),
        (133.10, 34.13), (133.02, 34.17),
    ],
    [
        (133.23, 34.27), (133.34, 34.25), (133.37, 34.17),
        (133.31, 34.11), (133.23, 34.15),
    ],
    [
        (132.95, 34.08), (133.07, 34.07), (133.10, 33.99),
        (133.04, 33.95), (132.96, 34.00),
    ],
    # Other notable Inland Sea islands.
    [
        (133.55, 34.47), (133.66, 34.45), (133.68, 34.38),
        (133.60, 34.35), (133.53, 34.40),
    ],
    [
        (132.98, 34.03), (133.07, 34.02), (133.09, 33.96),
        (133.02, 33.93), (132.97, 33.97),
    ],
    # Small islands in the western Inland Sea.
    [
        (131.85, 33.98), (131.93, 33.97), (131.95, 33.91),
        (131.89, 33.89), (131.84, 33.93),
    ],
    [
        (132.20, 34.03), (132.28, 34.02), (132.30, 33.97),
        (132.24, 33.95), (132.19, 33.98),
    ],
    # Chita Peninsula, approximated as a peninsula attached to the east side
    # of Ise Bay; the surrounding bay remains water.
    [
        (136.70, 35.10), (137.2, 35.10), (137.2, 34.80),
        (137.08, 34.77), (136.98, 34.70), (136.93, 34.60),
        (136.90, 34.49), (136.85, 34.40), (136.78, 34.42),
        (136.75, 34.55), (136.73, 34.72),
    ],
]


def point_in_polygon(x, y, polygon):
    """Ray-casting point test; boundary points are treated as land."""
    inside = False
    count = len(polygon)
    for i in range(count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]

        # Treat points on an edge as land.
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if (
            abs(cross) < 1e-10
            and min(x1, x2) - 1e-10 <= x <= max(x1, x2) + 1e-10
            and min(y1, y2) - 1e-10 <= y <= max(y1, y2) + 1e-10
        ):
            return True

        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing_x:
                inside = not inside
    return inside


def main():
    rows, cols = 64, 253
    step = 2.0 / 63.0
    latitudes = [35.0 - row * step for row in range(rows)]
    longitudes = [129.0 + col * step for col in range(cols)]

    matrix = np.zeros((rows, cols), dtype=np.uint8)
    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            if any(point_in_polygon(lon, lat, polygon) for polygon in LAND_POLYGONS):
                matrix[row, col] = 1

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", encoding="ascii", newline="") as f:
        for row in matrix:
            f.write(",".join(str(int(value)) for value in row) + "\n")

    # Matrix value 1 is white (land); 0 is black (water).
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(output_dir / "candidate.png")
    enlarged = image.resize((cols * 8, rows * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")

    # Verify dimensions, CSV contents, and the exact nearest-neighbor enlargement.
    assert matrix.shape == (64, 253)
    assert int(matrix.sum() + (matrix.size - matrix.sum())) == 16192
    assert set(np.unique(matrix)).issubset({0, 1})
    with csv_path.open("r", encoding="ascii") as f:
        csv_rows = [line.rstrip("\n").split(",") for line in f]
    assert len(csv_rows) == 64
    assert all(len(row) == 253 and all(v in {"0", "1"} for v in row) for row in csv_rows)
    assert image.size == (253, 64)
    assert set(image.getdata()).issubset({0, 255})
    assert enlarged.size == (2024, 512)
    assert np.array_equal(
        np.asarray(enlarged),
        np.repeat(np.repeat(matrix * 255, 8, axis=0), 8, axis=1),
    )


if __name__ == "__main__":
    main()
