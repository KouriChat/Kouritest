from pathlib import Path
import csv
import numpy as np
from PIL import Image

# Hand-drawn approximate land outlines in (longitude, latitude).
POLYGONS = [
    # Iberian Peninsula
    [(-9,44),(-8,42),(-9,41),(-9,38),(-9,37),(-7,37),(-6,36),
     (-5,36),(-4,37),(-2,36.7),(-1,37.5),(0,38.5),(1,40),
     (2,41),(3,41.5),(3,43),(1,43),(-2,43.5),(-5,43.5),(-7,44)],
    # Mainland Europe above the Mediterranean
    [(-2,45),(37,45),(37,42),(34,41.5),(32,42),(30,41.5),(28,41),
     (27,40.5),(26,40.5),(25,40.7),(24,40.7),(23,40.7),(22,41),
     (21,40.7),(20,41),(19,40.5),(19.5,39.8),(20,39),(21,38),
     (22,38),(23,38),(24,38),(25,37.8),(26,38.3),(27,38),(28,37),
     (28,36),(26,36),(24,36),(22,36),(20,37),(19,38),(18,39),
     (17,40),(16,40.5),(15,41),(14,41.5),(13,43),(12,44),(13,45)],
    # Italian peninsula
    [(8,45),(13,45),(13,44),(13,43),(14,42),(14,41),(16,41),
     (17,40),(18,40),(18,39),(17,40),(16,39),(16,38),(15,38),
     (15,37),(14,38),(13,38),(12,38),(13,39),(14,40),(14,41),
     (13,42),(12,43),(11,44),(9,44)],
    # Sardinia and Corsica
    [(8.1,41.2),(8.2,40.5),(8.4,39.5),(9.5,38.8),(9.8,40),
     (9.6,41),(9,41.3)],
    [(8.5,43),(8.5,42),(9.4,41.3),(9.5,42.2),(9.1,43)],
    # Sicily
    [(12.4,38),(13.5,38.2),(15.7,37.2),(15,36.7),(13,37),(12.4,37.5)],
    # North Africa
    [(-9,31),(37,31),(37,32.5),(34,31.5),(32,31.2),(30,31.5),
     (28,31.5),(25,31.7),(22,32),(20,32.5),(17,33),(15,32.8),
     (13,32.7),(12,33),(11,33.2),(10,33.7),(9,34),(8,34),
     (7,34),(6,36),(5,36.5),(3,36.8),(2,36.8),(1,36.5),
     (0,36.5),(-1,35.5),(-3,35.2),(-5,35.8),(-6,35.2),(-8,34.5)],
]
POLYGONS += [
    # Balearic Islands
    [(1.1,39.8),(1.3,39),(2.1,38.6),(3.5,39.2),(4.3,39.8),
     (3.2,40.1),(2,40)],
    # Malta, Crete, Cyprus
    [(14.2,36.1),(14.6,35.8),(14.6,36.1)],
    [(23.4,35.7),(25.8,35.7),(26.3,35.1),(24.5,34.9),(23.4,35.2)],
    [(32,35.2),(34.7,35.7),(34.6,34.6),(32.3,34.5)],
    # Greek islands and Aegean island groups
    [(19.3,39),(20.1,39.2),(20.8,38.8),(21.3,38.5),(22,38.3),
     (23,38),(24,37.8),(25,37.5),(26,37.3),(27,36.7),(28,36),
     (27,35.5),(25,36),(23,36.5),(21,37),(19,38)],
    # Crete
    [(23.4,35.7),(25.7,35.7),(26.4,35),(24.7,34.8),(23.5,35.1)],
    # Adriatic eastern shore, Balkans, Anatolia and Levant
    [(13,45),(14,45),(15,44),(16,43),(17,42),(18,41),(19,41),
     (20,40.5),(21,40),(22,39),(23,38),(24,37),(26,36),(28,36),
     (29,36),(30,36),(31,36),(32,36),(33,36),(35,36),(36,36),
     (37,36),(37,31),(35,31),(34,31.5),(32,31.2),(31,31.5),
     (30,31.5),(29,32),(28,33),(27,34),(26,35),(25,36),(24,36),
     (23,36),(22,36),(21,36),(20,37),(19,38),(18,39),(17,40),
     (16,41),(15,41),(14,42),(13,43)],
]

def inside(lon, lat, polygon):
    """Ray-cast point-in-polygon; points on an edge count as land."""
    hit = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
        if abs(cross) < 1e-10 and min(x1, x2) <= lon <= max(x1, x2) and min(y1, y2) <= lat <= max(y1, y2):
            return True
        if (y1 > lat) != (y2 > lat):
            x_cross = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < x_cross:
                hit = not hit
    return hit
def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)

    lats = [45.0 - 0.2 * i for i in range(71)]
    lons = [-9.0 + 0.2 * j for j in range(231)]
    matrix = np.zeros((71, 231), dtype=np.uint8)

    for row, lat in enumerate(lats):
        for col, lon in enumerate(lons):
            matrix[row, col] = int(any(inside(lon, lat, p) for p in POLYGONS))

    with (out / "candidate.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Pillow mode 1 gives a strictly black-and-white, one-bit PNG.
    image = Image.fromarray(matrix * 255, mode="L").convert("1")
    image.save(out / "candidate.png")
    enlarged = image.resize((1848, 568), resample=Image.Resampling.NEAREST)
    enlarged.save(out / "candidate_8x.png")

    assert matrix.shape == (71, 231)
    assert int(matrix.size) == 16401
    assert set(np.unique(matrix).tolist()) <= {0, 1}
    assert image.size == (231, 71)
    assert enlarged.size == (1848, 568)
    assert len(set(image.getdata())) == 2

if __name__ == "__main__":
    main()
