from pathlib import Path
import csv

import numpy as np
from PIL import Image

# WGS84 sampling points: rows north-to-south, columns west-to-east.
lats = 23.0 - 0.2 * np.arange(111)
lons = 99.0 + 0.2 * np.arange(141)
lon_grid, lat_grid = np.meshgrid(lons, lats)
land = np.zeros((111, 141), dtype=bool)

# Hand-drawn approximate mainland coast, ordered around the land boundary.
# The polygon includes land beyond the north and west edges of the frame.
mainland = [
    (99.0, 23.0), (111.8, 23.0), (113.0, 22.8), (114.0, 22.2),
    (115.0, 22.5), (116.0, 22.8), (117.0, 23.2), (118.0, 23.0),
    (120.0, 23.0), (120.0, 24.0), (99.0, 24.0),
]
# Explicit coastal boundary continues south from Guangdong through Vietnam,
# Cambodia, Thailand, and the Malay Peninsula, then up the Andaman coast.
mainland = [
    (99.0, 23.0), (115.7, 23.0), (116.5, 22.8), (117.2, 23.0),
    (118.0, 22.7), (119.0, 22.0), (120.0, 21.0), (121.0, 20.0),
    (121.0, 19.0), (120.0, 18.0), (119.0, 17.0),
    (118.0, 16.0), (117.0, 15.0), (116.0, 14.0),
    (115.0, 13.0), (114.0, 12.0), (113.0, 11.0),
    (112.0, 10.5), (111.0, 10.0), (110.0, 10.0),
    (109.0, 10.5), (108.0, 11.0), (107.0, 11.5),
    (106.0, 11.0), (105.0, 10.5), (104.0, 10.0),
    (103.0, 10.0), (102.0, 10.2), (101.0, 10.5),
    (100.0, 10.7), (99.0, 10.0), (98.0, 9.0), (97.5, 8.0),
    (97.5, 7.0), (98.0, 6.0), (99.0, 5.0), (100.0, 4.0),
    (101.0, 3.0), (102.0, 2.0), (103.0, 1.0), (104.0, 0.5),
    (105.0, 0.5), (105.0, -1.0), (99.0, -1.0),
]

def inside_polygon(x, y, polygon):
    """Vectorized even-odd point-in-polygon test."""
    result = np.zeros(x.shape, dtype=bool)
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[i - 1]
        crosses = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-30) + xi
        )
        result ^= crosses
    return result

land |= inside_polygon(lon_grid, lat_grid, mainland)

# Approximate island outlines as sampled ellipses. These are geographic
# judgement rules, not imported coastlines or a land-mask dataset.
def add_ellipse(cx, cy, rx, ry, angle=0.0):
    a = np.deg2rad(angle)
    dx = lon_grid - cx
    dy = lat_grid - cy
    xx = dx * np.cos(a) + dy * np.sin(a)
    yy = -dx * np.sin(a) + dy * np.cos(a)
    land[:] |= (xx / rx) ** 2 + (yy / ry) ** 2 <= 1.0

# Hainan and Taiwan
add_ellipse(109.7, 19.2, 1.05, 2.0, -8)
add_ellipse(120.9, 22.7, 0.85, 1.45, 18)

# Philippines: Luzon, Mindoro, Palawan, Visayas, and Mindanao.
add_ellipse(121.0, 16.2, 1.35, 2.45, 12)
add_ellipse(120.9, 13.3, 0.55, 0.9, 18)
add_ellipse(120.0, 10.0, 0.32, 0.75, 8)
add_ellipse(118.7, 10.0, 0.30, 2.0, 12)
add_ellipse(122.0, 11.0, 0.8, 0.45, -10)
add_ellipse(124.0, 10.0, 1.3, 1.65, 20)
add_ellipse(125.0, 7.5, 0.55, 0.7, 20)
add_ellipse(122.0, 12.3, 0.55, 0.5)
add_ellipse(123.5, 12.2, 0.6, 0.35)
add_ellipse(124.0, 11.7, 0.45, 0.3)
add_ellipse(121.8, 9.0, 0.35, 0.35)
# Batanes and small northern Philippine islands
add_ellipse(121.9, 20.4, 0.18, 0.22)
add_ellipse(122.0, 19.9, 0.12, 0.16)

# Borneo, including the northern coastline and the Indonesian portion.
add_ellipse(114.8, 3.5, 4.8, 3.45, 5)
# Carve the broad northwest and northeast bays to approximate the coast.
land &= ~(
    ((lon_grid > 113.0) & (lon_grid < 117.5) &
     (lat_grid > 5.5) & (lat_grid < 7.2))
)
# Restore the northwestern tip of Borneo.
add_ellipse(115.0, 6.2, 1.15, 0.9, -15)

# Sumatra, including its northern portion in this frame.
add_ellipse(101.2, 2.1, 2.35, 3.25, -18)
# Java's northern edge barely enters the southern frame.
add_ellipse(105.0, 1.1, 1.25, 0.24, -8)
# Sulawesi and nearby larger islands
add_ellipse(120.0, 1.4, 0.55, 1.8, 30)
add_ellipse(121.3, 1.0, 0.55, 1.05, -20)
add_ellipse(122.7, 1.1, 0.55, 0.8, 15)
add_ellipse(119.0, 0.2, 0.85, 0.45, -10)
add_ellipse(123.5, 0.9, 0.45, 0.65, 20)
# Sulu Islands
add_ellipse(120.0, 6.0, 0.18, 0.3)
add_ellipse(120.5, 5.5, 0.16, 0.25)
add_ellipse(121.0, 5.0, 0.14, 0.24)
add_ellipse(121.5, 4.6, 0.16, 0.22)
add_ellipse(122.0, 4.2, 0.14, 0.2)

# Spratly and Paracel reefs/islets: small point-centered approximations.
for x, y, rx, ry in [
    (111.7, 16.5, .18, .10), (112.0, 16.0, .12, .10),
    (112.5, 15.8, .12, .10), (113.0, 15.2, .12, .10),
    (114.0, 12.0, .12, .10), (114.5, 11.5, .12, .10),
    (115.0, 10.5, .12, .10), (116.0, 10.0, .12, .10),
    (116.5, 9.5, .12, .10), (117.0, 8.5, .12, .10),
    (115.8, 8.0, .12, .10), (114.3, 8.7, .12, .10),
    (113.5, 9.0, .12, .10), (112.5, 10.0, .12, .10),
    (111.8, 10.3, .12, .10), (112.3, 16.7, .28, .18),
]:
    add_ellipse(x, y, rx, ry)

# Validate and write deliverables.
assert land.shape == (111, 141)
assert land.size == 15651
out = Path("outputs")
out.mkdir(parents=True, exist_ok=True)

with (out / "candidate.csv").open("w", newline="") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(land.astype(np.uint8).tolist())

# 1 means land (white); 0 means water (black).
image = Image.fromarray(np.where(land, 255, 0).astype(np.uint8), mode="L")
assert image.size == (141, 111)
image.save(out / "candidate.png")
image.resize((1128, 888), resample=Image.Resampling.NEAREST).save(
    out / "candidate_8x.png"
)

# Verify CSV shape/content and exact nearest-neighbour enlargement.
with (out / "candidate.csv").open(newline="") as f:
    rows = list(csv.reader(f))
assert len(rows) == 111 and all(len(row) == 141 for row in rows)
assert all(value in {"0", "1"} for row in rows for value in row)
large = Image.open(out / "candidate_8x.png")
assert large.size == (1128, 888)
assert set(large.getdata()) == {0, 255}
assert np.array_equal(
    np.asarray(large).reshape(111, 8, 141, 8)[:, 0, :, 0],
    np.asarray(image),
)
print("Wrote validated 111x141 CSV and 141x111 / 1128x888 PNG files.")
