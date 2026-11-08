from pathlib import Path
import csv
import numpy as np
from PIL import Image

# Approximate hand-authored coastline polygons in (longitude, latitude).
# The mainland polygon traces the outer coast on its west side and extends
# beyond the eastern frame edge. Island polygons represent Graham and Moresby.
LAND = [
    [
        (-130.30, 55.00), (-130.43, 54.94), (-130.60, 54.86),
        (-130.65, 54.78),
        (-130.42, 54.58), (-130.28, 54.42), (-130.15, 54.24),
        (-130.12, 54.05), (-130.05, 53.90), (-129.93, 53.72),
        (-129.90, 53.52), (-129.82, 53.34), (-129.75, 53.15),
        (-129.65, 53.00), (-128.0, 53.00), (-128.0, 55.0),
    ],
    # Eastern Graham Island and its archipelago-connected northern shore.
    [
        (-131.10, 54.46), (-130.94, 54.40), (-130.88, 54.30),
        (-130.82, 54.20), (-130.78, 54.08), (-130.83, 53.98),
        (-130.94, 53.94), (-131.08, 54.00), (-131.13, 54.12),
        (-131.18, 54.24), (-131.20, 54.36),
    ],
    # Eastern Moresby Island, clipped by the southern edge of this frame.
    [
        (-131.18, 53.62), (-131.02, 53.56), (-130.90, 53.48),
        (-130.82, 53.36), (-130.78, 53.20), (-130.76, 53.00),
        (-131.18, 53.00), (-131.30, 53.12), (-131.34, 53.28),
        (-131.31, 53.43),
    ],
    # Small coastal islands and reefs are approximated as compact land patches.
    [
        (-130.98, 54.74), (-130.95, 54.72), (-130.93, 54.69),
        (-130.97, 54.67), (-131.00, 54.70),
    ],
    [
        (-130.56, 54.08), (-130.53, 54.06), (-130.50, 54.03),
        (-130.54, 54.01), (-130.58, 54.04),
    ],
    [
        (-130.49, 53.84), (-130.46, 53.81), (-130.42, 53.79),
        (-130.46, 53.77), (-130.51, 53.80),
    ],
]


def contains(lon, lat, polygon):
    """Ray-casting point-in-polygon test; boundary points count as land."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        cross = (lon - xi) * (yj - yi) - (lat - yi) * (xj - xi)
        if abs(cross) < 1e-10 and min(xi, xj) <= lon <= max(xi, xj) and min(yi, yj) <= lat <= max(yi, yj):
            return True
        if (yi > lat) != (yj > lat):
            x_cross = xi + (lat - yi) * (xj - xi) / (yj - yi)
            if lon < x_cross:
                inside = not inside
        j = i
    return inside
ROWS = COLS = 127
STEP = 2.0 / 126.0
latitudes = 55.0 - np.arange(ROWS) * STEP
longitudes = -131.0 + np.arange(COLS) * STEP

matrix = np.zeros((ROWS, COLS), dtype=np.uint8)
for row, lat in enumerate(latitudes):
    for col, lon in enumerate(longitudes):
        matrix[row, col] = int(any(contains(lon, lat, poly) for poly in LAND))

out = Path("outputs")
out.mkdir(parents=True, exist_ok=True)
with (out / "candidate.csv").open("w", newline="", encoding="ascii") as handle:
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerows(matrix.tolist())

# White is land (1), black is water (0).
pixels = np.where(matrix == 1, 255, 0).astype(np.uint8)
image = Image.fromarray(pixels, mode="L")
image.save(out / "candidate.png")
image.resize((1016, 1016), resample=Image.Resampling.NEAREST).save(
    out / "candidate_8x.png"
)

# Validate the persisted deliverables, not just the in-memory matrix.
with (out / "candidate.csv").open("r", newline="", encoding="ascii") as handle:
    records = list(csv.reader(handle))
assert len(records) == ROWS
assert all(len(record) == COLS for record in records)
assert all(value in {"0", "1"} for record in records for value in record)
assert sum(map(len, records)) == 16129
assert image.size == (127, 127)
assert set(image.getdata()) == {0, 255}
upscaled = Image.open(out / "candidate_8x.png")
assert upscaled.size == (1016, 1016)
assert set(upscaled.getdata()) == {0, 255}
for y in range(127):
    for x in range(127):
        expected = image.getpixel((x, y))
        assert all(
            upscaled.getpixel((x * 8 + dx, y * 8 + dy)) == expected
            for dx in range(8)
            for dy in range(8)
        )
