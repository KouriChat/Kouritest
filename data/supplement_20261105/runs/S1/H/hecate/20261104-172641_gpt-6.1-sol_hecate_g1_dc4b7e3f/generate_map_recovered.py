from pathlib import Path

import numpy as np
from PIL import Image


# All coordinates below use longitude in degrees west as a positive number:
# for example, 130.5 means 130.5 W.
#
# The polygons are intentionally approximate and were written from geographic
# knowledge. They describe the mainland coast and the main islands in the
# requested 131 W to 129 W, 55 N to 53 N frame.

LAND_POLYGONS = [
    # Mainland British Columbia. The first edge follows an approximate
    # north-to-south outer coastline; the remaining edges close it against
    # the eastern edge of the requested frame.
    [
        (130.42, 55.00),
        (130.32, 54.90),
        (130.23, 54.78),
        (130.28, 54.68),
        (130.43, 54.55),
        (130.40, 54.43),
        (130.25, 54.30),
        (130.08, 54.16),
        (129.88, 54.01),
        (129.68, 53.86),
        (129.47, 53.69),
        (129.27, 53.52),
        (129.10, 53.37),
        (129.04, 53.42),
        (129.00, 53.42),
        (129.00, 55.00),
    ],

    # Eastern edge of Graham Island.
    [
        (131.00, 54.27),
        (130.97, 54.20),
        (130.94, 54.06),
        (130.96, 53.88),
        (131.00, 53.70),
        (131.00, 53.52),
        (130.96, 53.43),
        (130.91, 53.48),
        (130.91, 53.65),
        (130.92, 53.84),
        (130.91, 54.04),
        (130.93, 54.21),
        (130.95, 54.27),
    ],

    # Small part of eastern Moresby Island visible at the western frame edge.
    [
        (131.00, 53.20),
        (130.96, 53.14),
        (130.92, 53.05),
        (130.90, 53.00),
        (131.00, 53.00),
    ],

    # Dundas Island.
    [
        (131.00, 54.79),
        (130.88, 54.77),
        (130.78, 54.69),
        (130.77, 54.59),
        (130.84, 54.52),
        (130.97, 54.51),
        (131.00, 54.58),
    ],

    # Stephens Island and nearby land.
    [
        (130.88, 54.70),
        (130.78, 54.67),
        (130.73, 54.60),
        (130.77, 54.54),
        (130.87, 54.56),
        (130.92, 54.63),
    ],

    # Kaien Island, including the Prince Rupert area.
    [
        (130.70, 54.47),
        (130.54, 54.49),
        (130.38, 54.43),
        (130.25, 54.33),
        (130.22, 54.24),
        (130.30, 54.15),
        (130.47, 54.13),
        (130.62, 54.22),
        (130.72, 54.35),
    ],

    # Digby Island.
    [
        (130.66, 54.36),
        (130.51, 54.36),
        (130.40, 54.29),
        (130.40, 54.22),
        (130.51, 54.18),
        (130.64, 54.22),
        (130.70, 54.30),
    ],

    # Banks Island.
    [
        (130.37, 54.30),
        (130.18, 54.35),
        (129.99, 54.32),
        (129.82, 54.24),
        (129.70, 54.09),
        (129.70, 53.94),
        (129.83, 53.83),
        (130.04, 53.84),
        (130.22, 53.94),
        (130.34, 54.10),
    ],

    # Porcher Island.
    [
        (130.16, 54.20),
        (129.93, 54.21),
        (129.70, 54.15),
        (129.48, 54.06),
        (129.30, 53.96),
        (129.23, 53.84),
        (129.36, 53.74),
        (129.58, 53.76),
        (129.80, 53.83),
        (130.00, 53.94),
        (130.16, 54.05),
    ],

    # Gil Island and nearby islands in the northern Hecate Strait area.
    [
        (129.43, 53.83),
        (129.25, 53.86),
        (129.10, 53.79),
        (129.01, 53.66),
        (129.02, 53.53),
        (129.17, 53.48),
        (129.34, 53.56),
        (129.42, 53.69),
    ],

    # Pitt Island.
    [
        (129.73, 53.78),
        (129.51, 53.82),
        (129.29, 53.76),
        (129.10, 53.63),
        (128.98, 53.45),
        (128.91, 53.22),
        (129.00, 53.08),
        (129.18, 53.03),
        (129.38, 53.13),
        (129.55, 53.31),
        (129.68, 53.53),
        (129.76, 53.68),
    ],

    # Campania Island and nearby land near the southern part of the frame.
    [
        (129.96, 53.39),
        (129.78, 53.44),
        (129.61, 53.37),
        (129.48, 53.22),
        (129.43, 53.09),
        (129.56, 53.01),
        (129.77, 53.00),
        (129.91, 53.12),
        (129.98, 53.27),
    ],

    # Western edge of Princess Royal Island.
    [
        (129.32, 53.67),
        (129.18, 53.64),
        (129.07, 53.51),
        (128.98, 53.32),
        (128.92, 53.13),
        (128.92, 53.00),
        (129.00, 53.00),
        (129.02, 53.21),
        (129.12, 53.43),
        (129.28, 53.58),
    ],

    # Several smaller islands and island groups around Dixon Entrance and
    # the northern Hecate Strait channels.
    [
        (130.56, 54.72),
        (130.47, 54.75),
        (130.40, 54.69),
        (130.43, 54.62),
        (130.53, 54.61),
    ],
    [
        (130.24, 54.61),
        (130.16, 54.65),
        (130.09, 54.59),
        (130.12, 54.52),
        (130.22, 54.53),
    ],
    [
        (130.82, 54.30),
        (130.72, 54.29),
        (130.68, 54.22),
        (130.74, 54.18),
        (130.83, 54.22),
    ],
    [
        (130.06, 53.76),
        (129.96, 53.79),
        (129.91, 53.72),
        (129.96, 53.66),
        (130.05, 53.68),
    ],
    [
        (129.84, 53.62),
        (129.75, 53.66),
        (129.70, 53.59),
        (129.74, 53.52),
        (129.83, 53.54),
    ],
]


def points_in_polygon(x, y, polygon):
    """Vectorized point-in-polygon test using the ray-crossing rule."""
    vertices = np.asarray(polygon, dtype=np.float64)
    px = vertices[:, 0]
    py = vertices[:, 1]

    inside = np.zeros(x.shape, dtype=bool)

    x0 = px
    y0 = py
    x1 = np.roll(px, -1)
    y1 = np.roll(py, -1)

    for ax, ay, bx, by in zip(x0, y0, x1, y1):
        crosses_scanline = (ay > y) != (by > y)

        denominator = by - ay
        if abs(denominator) < 1e-15:
            continue

        intersection_x = (bx - ax) * (y - ay) / denominator + ax
        inside ^= crosses_scanline & (x < intersection_x)

    return inside


def build_matrix():
    rows = 127
    cols = 127

    # Row 0 is 55 N, row 126 is 53 N.
    # Column 0 is 131 W, column 126 is 129 W.
    latitudes = 55.0 - np.arange(rows, dtype=np.float64) * (2.0 / 126.0)
    west_longitudes = 131.0 - np.arange(cols, dtype=np.float64) * (2.0 / 126.0)

    longitudes, latitudes_grid = np.meshgrid(
        west_longitudes,
        latitudes,
    )

    land = np.zeros((rows, cols), dtype=bool)

    for polygon in LAND_POLYGONS:
        land |= points_in_polygon(longitudes, latitudes_grid, polygon)

    return land.astype(np.uint8)


def validate_outputs(matrix, csv_path, original_png, enlarged_png):
    if matrix.shape != (127, 127):
        raise AssertionError(f"Unexpected matrix shape: {matrix.shape}")

    if matrix.size != 16129:
        raise AssertionError(f"Unexpected value count: {matrix.size}")

    if not np.all((matrix == 0) | (matrix == 1)):
        raise AssertionError("Matrix contains values other than 0 and 1")

    csv_lines = csv_path.read_text(encoding="ascii").splitlines()

    if len(csv_lines) != 127:
        raise AssertionError("CSV does not contain exactly 127 rows")

    for line_number, line in enumerate(csv_lines, start=1):
        fields = line.split(",")
        if len(fields) != 127:
            raise AssertionError(
                f"CSV row {line_number} does not contain exactly 127 values"
            )
        if any(field not in ("0", "1") for field in fields):
            raise AssertionError(
                f"CSV row {line_number} contains a value other than 0 or 1"
            )
        if any(field == "" for field in fields):
            raise AssertionError(f"CSV row {line_number} contains an empty value")

    loaded_csv = np.loadtxt(csv_path, dtype=np.uint8, delimiter=",")
    if loaded_csv.shape != (127, 127):
        raise AssertionError("Reloaded CSV has an unexpected shape")
    if not np.array_equal(loaded_csv, matrix):
        raise AssertionError("CSV contents differ from the generated matrix")

    with Image.open(original_png) as image:
        if image.size != (127, 127):
            raise AssertionError(f"Unexpected original image size: {image.size}")
        original_pixels = np.asarray(image.convert("L"))
        if set(np.unique(original_pixels).tolist()) != {0, 255}:
            raise AssertionError("Original PNG is not strictly black and white")

    with Image.open(enlarged_png) as image:
        if image.size != (1016, 1016):
            raise AssertionError(f"Unexpected enlarged image size: {image.size}")
        enlarged_pixels = np.asarray(image.convert("L"))
        if set(np.unique(enlarged_pixels).tolist()) != {0, 255}:
            raise AssertionError("Enlarged PNG is not strictly black and white")

        # Every 8x8 block must be constant and equal to the corresponding
        # original pixel, which verifies nearest-neighbor enlargement.
        blocks = enlarged_pixels.reshape(127, 8, 127, 8)
        block_reference = blocks[:, 0:1, :, 0:1]
        if not np.all(blocks == block_reference):
            raise AssertionError("Enlarged PNG is not nearest-neighbor expanded")

        if not np.array_equal(enlarged_pixels[::8, ::8], original_pixels):
            raise AssertionError(
                "Enlarged PNG does not preserve the original pixel grid"
            )


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = build_matrix()

    csv_path = output_dir / "candidate.csv"
    original_png = output_dir / "candidate.png"
    enlarged_png = output_dir / "candidate_8x.png"

    np.savetxt(
        csv_path,
        matrix,
        fmt="%d",
        delimiter=",",
    )

    # 1 means land and is white; 0 means water and is black.
    image_array = (matrix * 255).astype(np.uint8)
    original_image = Image.fromarray(image_array, mode="L")
    original_image.save(original_png)

    enlarged_image = original_image.resize(
        (127 * 8, 127 * 8),
        resample=Image.Resampling.NEAREST,
    )
    enlarged_image.save(enlarged_png)

    validate_outputs(
        matrix,
        csv_path,
        original_png,
        enlarged_png,
    )

    print("Generated:")
    print(f"  {csv_path}       shape={matrix.shape}, values={matrix.size}")
    print(f"  {original_png}   size=127x127")
    print(f"  {enlarged_png}  size=1016x1016")
    print("Validation passed.")


if __name__ == "__main__":
    main()
