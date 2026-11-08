from pathlib import Path
import csv
import numpy as np
from PIL import Image

# Hand-authored approximate coast outlines, in (longitude, latitude).
# These are deliberately coarse knowledge-based rules, not map data.
LAND = [
    # Iberia
    [(-9.5, 43.8), (-8.9, 43.7), (-8.2, 43.6), (-7.0, 43.7), (-5.5, 43.5),
     (-3.0, 43.4), (-1.8, 43.3), (0.0, 42.8), (1.5, 42.5), (3.2, 42.4),
     (3.3, 41.8), (2.7, 41.3), (3.2, 40.0), (2.6, 39.5), (1.4, 39.7),
     (0.2, 38.7), (-0.7, 37.6), (-1.7, 36.7), (-3.0, 36.7), (-5.3, 36.0),
     (-6.3, 36.0), (-7.5, 37.0), (-7.4, 38.5), (-9.0, 41.0)],
    # France and western/central Europe, with Bay of Biscay and Gulf of Lion
    [(-5.2, 48.8), (-2.0, 48.7), (0.0, 49.0), (1.8, 50.8), (4.2, 51.0),
     (6.0, 50.8), (7.5, 49.2), (7.6, 47.5), (8.2, 46.0), (7.0, 44.2),
     (5.0, 43.2), (4.7, 43.0), (3.5, 43.3), (3.2, 42.4), (1.5, 42.5),
     (0.0, 42.8), (-1.8, 43.3), (-1.5, 44.0), (-1.2, 45.0), (-1.8, 46.0),
     (-2.2, 47.0), (-4.5, 47.5)],
    # Northern Europe eastward to the map edge
    [(7.0, 44.2), (8.2, 46.0), (7.6, 47.5), (7.5, 49.2), (6.0, 50.8),
     (8.0, 54.0), (10.0, 55.5), (12.0, 55.0), (14.0, 54.5), (16.0, 55.0),
     (18.0, 54.5), (20.0, 55.0), (22.0, 54.5), (24.0, 55.0), (26.0, 54.5),
     (28.0, 55.0), (30.0, 54.0), (32.0, 54.5), (34.0, 55.0), (37.5, 54.0),
     (37.5, 45.0), (35.0, 45.0), (32.0, 45.5), (30.0, 45.5), (29.0, 43.0),
     (26.0, 44.0), (24.0, 45.0), (22.0, 44.0), (20.0, 45.0), (18.0, 45.8),
     (16.0, 46.0), (13.5, 45.5), (10.0, 46.5), (8.2, 46.0), (7.0, 44.2)],
    # Italy, including the peninsula but excluding surrounding seas
    [(7.0, 44.2), (8.0, 43.0), (9.0, 42.0), (9.5, 41.0), (10.0, 40.0),
     (11.0, 39.0), (12.5, 38.0), (13.0, 37.0), (14.5, 38.0), (16.0, 39.0),
     (17.0, 40.0), (18.5, 40.0), (18.5, 41.0), (17.0, 42.0), (16.0, 43.0),
     (15.0, 44.0), (13.5, 45.5), (12.0, 46.0), (10.0, 46.5), (8.2, 46.0)],
    # Balkans and Greece
    [(13.5, 45.5), (15.0, 44.0), (16.0, 43.0), (17.0, 42.0), (18.5, 41.0),
     (20.0, 40.0), (21.0, 39.0), (22.0, 37.0), (23.0, 36.0), (24.5, 35.5),
     (26.0, 36.5), (27.0, 38.0), (26.0, 40.0), (27.0, 41.5), (29.0, 41.5),
     (29.0, 43.0), (26.0, 44.0), (24.0, 45.0), (22.0, 44.0), (20.0, 45.0),
     (18.0, 45.8), (16.0, 46.0)],
    # Anatolia
    [(26.0, 41.5), (27.0, 41.0), (29.0, 41.3), (31.0, 41.0), (33.0, 42.0),
     (36.0, 41.5), (37.5, 40.5), (37.5, 36.0), (36.0, 36.0), (34.0, 36.0),
     (32.0, 36.0), (30.0, 36.0), (28.0, 36.0), (27.0, 37.0), (26.0, 38.0)],
    # North Africa, Mediterranean coast and Sahara interior
    [(-9.0, 31.0), (-8.5, 32.7), (-6.0, 35.0), (-5.5, 35.8), (-3.0, 35.9),
     (-1.5, 35.8), (0.0, 36.0), (2.0, 36.7), (4.0, 37.0), (6.0, 37.0),
     (8.0, 36.8), (10.0, 37.0), (11.0, 37.0), (12.0, 37.0), (13.0, 37.0),
     (14.5, 38.0), (16.0, 37.5), (18.0, 37.0), (20.0, 33.0), (22.0, 32.9),
     (24.0, 32.0), (25.0, 31.5), (27.0, 31.5), (29.0, 31.2), (32.0, 31.0),
     (37.0, 31.0), (37.0, 29.0), (-9.0, 29.0)],
    # Crete
    [(23.4, 35.5), (24.0, 35.7), (25.7, 35.5), (26.3, 35.1), (25.0, 34.9), (23.5, 35.0)],
    # Sardinia and Corsica
    [(8.1, 41.0), (8.4, 41.2), (9.7, 41.0), (9.7, 39.0), (9.0, 38.8), (8.3, 40.0)],
    [(8.5, 42.8), (9.5, 43.0), (9.5, 41.4), (8.7, 41.4)],
    # Sicily
    [(12.4, 38.2), (13.0, 38.3), (15.6, 38.2), (15.2, 37.0), (13.0, 36.6), (12.4, 37.5)],
    # Balearic Islands
    [(1.2, 39.9), (1.7, 40.1), (4.3, 40.1), (4.3, 39.2), (2.3, 38.6), (1.2, 39.0)],
    # Cyprus
    [(32.2, 35.7), (34.6, 35.8), (34.8, 35.0), (32.3, 34.6)],
    # Malta
    [(14.1, 36.1), (14.6, 36.1), (14.6, 35.8), (14.1, 35.8)],
    # Aegean island groups (island-sized polygons, not broad sea regions)
    [(25.0, 36.8), (25.5, 37.0), (25.7, 36.5), (25.2, 36.4)],
    [(26.5, 36.8), (27.0, 37.0), (27.2, 36.4), (26.6, 36.4)],
    [(24.5, 38.5), (25.0, 38.7), (25.2, 38.3), (24.7, 38.2)],
    [(26.0, 39.0), (26.5, 39.2), (26.6, 38.7), (26.1, 38.6)],
    [(27.0, 36.9), (27.5, 37.1), (27.7, 36.6), (27.2, 36.5)],
]


def in_polygon(x, y, poly):
    inside = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < cross:
                inside = not inside
        j = i
    return inside


def main():
    lats = 45.0 - 0.2 * np.arange(71)
    lons = -9.0 + 0.2 * np.arange(231)
    matrix = np.zeros((71, 231), dtype=np.uint8)
    for r, lat in enumerate(lats):
        for c, lon in enumerate(lons):
            matrix[r, c] = any(in_polygon(lon, lat, poly) for poly in LAND)

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(matrix.tolist())
    image = Image.fromarray(np.where(matrix == 1, 255, 0).astype(np.uint8), mode="L")
    image.save(out / "candidate.png")
    image.resize((1848, 568), resample=Image.Resampling.NEAREST).save(out / "candidate_8x.png")

    assert matrix.shape == (71, 231) and matrix.size == 16401
    assert set(np.unique(matrix)) <= {0, 1}
    assert image.size == (231, 71) and set(image.getdata()) <= {0, 255}
    with (out / "candidate.csv").open(newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 71 and all(len(row) == 231 for row in rows)
    assert all(v in ("0", "1") for row in rows for v in row)
    enlarged = Image.open(out / "candidate_8x.png")
    assert enlarged.size == (1848, 568)
    assert set(enlarged.getdata()) <= {0, 255}
    print("Validated 71x231 matrix, CSV, 231x71 PNG, and nearest-neighbor 8x PNG.")


if __name__ == "__main__":
    main()
