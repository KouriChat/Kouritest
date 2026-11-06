import csv
import os

import numpy as np
from PIL import Image

# Grid: 45N to 31N, 9W to 37E, both inclusive at 0.2-degree steps.
lats = np.round(np.arange(45.0, 30.999, -0.2), 10)
lons = np.round(np.arange(-9.0, 37.001, 0.2), 10)
land = np.zeros((len(lats), len(lons)), dtype=bool)

# Approximate mainland outlines, drawn from general geographic knowledge.
# Each polygon is a sequence of (longitude, latitude) vertices.
polygons = [
    # Iberia and France
    [(-9, 43.8), (-8.9, 41.8), (-7.1, 41.8), (-6.9, 36.0),
     (-5.5, 36.0), (-5.2, 36.5), (-2.0, 36.0), (0.0, 36.0),
     (1.5, 36.5), (3.0, 36.7), (3.2, 42.4), (0.0, 42.7),
     (-1.8, 43.4), (-1.8, 43.8), (-5.3, 43.6), (-7.3, 43.8)],
    [(-5.2, 43.5), (-1.8, 43.5), (0.0, 42.8), (3.2, 42.4),
     (4.8, 43.0), (6.8, 43.2), (7.7, 43.8), (7.6, 45.2),
     (-5.2, 45.2)],
    # North Africa, with the Mediterranean coast approximated
    [(-9, 35.0), (-6.0, 35.0), (-5.5, 35.9), (-2.0, 35.9),
     (0.0, 36.0), (1.5, 36.5), (3.0, 36.7), (5.0, 37.1),
     (8.0, 37.1), (10.0, 37.0), (11.0, 37.4), (12.5, 37.1),
     (14.0, 37.0), (16.0, 37.2), (18.0, 37.4), (20.0, 37.0),
     (22.0, 37.0), (24.0, 37.2), (26.0, 37.0), (28.0, 36.8),
     (30.0, 36.8), (32.0, 36.8), (34.0, 36.8), (37.0, 36.8),
     (37.0, 31.0), (-9.0, 31.0)],
    # Italy and the Adriatic / Balkan mainland
    [(6.5, 45.2), (13.0, 45.2), (13.0, 44.8), (12.5, 44.5),
     (13.0, 43.5), (13.5, 42.5), (15.0, 41.9), (16.8, 41.9),
     (18.5, 40.0), (18.5, 39.0), (17.0, 38.0), (16.0, 38.0),
     (15.0, 39.0), (14.0, 40.0), (13.0, 41.0), (12.0, 42.0),
     (11.0, 43.0), (10.0, 43.8), (9.0, 44.0), (7.0, 44.0)],
    [(13.0, 45.2), (19.0, 45.2), (20.0, 44.5), (19.0, 43.0),
     (18.5, 41.5), (20.0, 40.0), (21.0, 39.0), (23.0, 38.0),
     (24.0, 37.0), (27.0, 36.5), (28.0, 37.0), (30.0, 37.0),
     (30.0, 41.0), (29.0, 42.0), (27.0, 42.0), (25.0, 42.0),
     (23.0, 42.5), (21.0, 43.0), (20.0, 44.5), (19.0, 45.2)],
    # Anatolia and Levant
    [(26.0, 41.7), (27.0, 42.0), (29.0, 42.0), (30.0, 41.0),
     (32.0, 41.0), (34.0, 41.0), (36.0, 41.0), (37.0, 41.0),
     (37.0, 35.0), (36.0, 36.0), (35.0, 36.5), (34.0, 36.5),
     (32.0, 36.0), (30.0, 36.0), (29.0, 36.0), (28.0, 36.5),
     (27.0, 36.5), (26.0, 37.0)],
    # Black Sea north coast and Crimea
    [(27.0, 45.2), (37.0, 45.2), (37.0, 44.0), (35.0, 44.0),
     (34.0, 45.0), (33.0, 46.0), (32.0, 46.2), (31.0, 46.0),
     (30.0, 45.5), (29.0, 45.2)],
]

def point_in_polygon(x, y, poly):
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-15) + xi
        ):
            inside = not inside
        j = i
    return inside

for ri, lat in enumerate(lats):
    for ci, lon in enumerate(lons):
        land[ri, ci] = any(point_in_polygon(lon, lat, p) for p in polygons)

# Elliptical island rules (center longitude, center latitude, radii in degrees).
# These deliberately use compact approximations for major islands.
islands = [
    ( -1.5, 39.7, 0.55, 0.28),  # Ibiza
    (  2.9, 39.6, 0.75, 0.38),  # Mallorca
    (  4.0, 39.9, 0.28, 0.18),  # Menorca
    (  8.0, 40.0, 0.95, 1.55),  # Corsica
    (  9.0, 40.0, 1.35, 1.75),  # Sardinia
    (  8.5, 41.6, 0.45, 0.22),  # northern Corsica
    ( 12.0, 36.8, 1.55, 1.35),  # Sicily
    ( 15.0, 39.0, 0.45, 0.20),  # Aeolian islands, broad approximation
    (  9.0, 39.2, 0.28, 0.22),  # Sardinian south
    ( 14.0, 35.9, 0.24, 0.12),  # Malta
    ( 25.0, 35.2, 1.55, 0.50),  # Crete
    ( 23.0, 37.5, 0.60, 0.28),  # Euboea
    ( 22.5, 38.0, 0.50, 0.25),  # central Greek islands
    ( 26.0, 37.0, 0.85, 0.35),  # Dodecanese / Rhodes region
    ( 28.0, 36.2, 0.35, 0.18),  # Rhodes
    ( 33.0, 35.1, 0.65, 0.28),  # Cyprus
    ( 19.8, 39.6, 0.45, 0.40),  # Corfu
    ( 20.7, 38.7, 0.70, 0.20),  # Ionian islands
    ( 23.5, 40.5, 0.85, 0.45),  # northern Aegean islands
    ( 24.8, 40.6, 0.55, 0.28),  # Thasos / Samothrace
    ( 26.0, 40.0, 0.35, 0.20),  # Lesbos vicinity
    ( 27.0, 37.7, 0.55, 0.24),  # Samos / nearby islands
    ( 29.0, 36.0, 0.32, 0.15),  # eastern Mediterranean islands
    ( 13.4, 43.6, 0.35, 0.18),  # Adriatic islands
    ( 17.8, 42.7, 0.35, 0.18),  # Dalmatian islands
    ( 18.5, 42.5, 0.28, 0.16),
    ( 31.2, 31.5, 0.30, 0.12),  # Nile Delta coastal land extension
]

for ri, lat in enumerate(lats):
    for ci, lon in enumerate(lons):
        for cx, cy, rx, ry in islands:
            if ((lon - cx) / rx) ** 2 + ((lat - cy) / ry) ** 2 <= 1.0:
                land[ri, ci] = True

# Keep the southern Iberian / Moroccan coastline and Gibraltar region plausible.
# The polygon rules above provide the broad land masses; this narrow strait remains water.
for ri, lat in enumerate(lats):
    if 35.8 <= lat <= 36.2:
        for ci, lon in enumerate(lons):
            if -5.8 <= lon <= -5.0:
                land[ri, ci] = False

out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, "candidate.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(land.astype(np.uint8).tolist())

# Pillow uses 0 for black (water), 255 for white (land).
pixels = land.astype(np.uint8) * 255
image = Image.fromarray(pixels, mode="L")
image.save(os.path.join(out_dir, "candidate.png"))

large = image.resize((231 * 8, 71 * 8), resample=Image.Resampling.NEAREST)
large.save(os.path.join(out_dir, "candidate_8x.png"))

# Validate dimensions, values, CSV shape, palette, and exact nearest-neighbor scaling.
assert land.shape == (71, 231)
assert int(land.sum() + (~land).sum()) == 16401
with open(os.path.join(out_dir, "candidate.csv"), newline="") as f:
    rows = list(csv.reader(f))
assert len(rows) == 71 and all(len(row) == 231 for row in rows)
assert all(value in ("0", "1") for row in rows for value in row)
assert set(np.unique(np.asarray(image)).tolist()) == {0, 255}
assert image.size == (231, 71)
assert large.size == (1848, 568)
assert np.array_equal(
    np.asarray(large),
    np.repeat(np.repeat(pixels, 8, axis=0), 8, axis=1),
)
