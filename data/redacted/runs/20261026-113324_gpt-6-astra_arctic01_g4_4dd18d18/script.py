from pathlib import Path
import csv
import numpy as np
from PIL import Image

# Hand-drawn geographic approximations; coordinates are (west longitude, latitude).
LAND = [
    # Ellesmere Island, western and southern coasts
    [(81, 76.9), (82, 77.0), (82.7, 77.5), (83.3, 78.1), (84.2, 78.4),
     (85, 78.9), (85.5, 79.5), (86, 80.1), (87, 81), (84, 81), (81, 81)],
    # Axel Heiberg Island
    [(84.8, 78.0), (85.2, 78.4), (86.5, 78.7), (87.8, 79.0), (88.3, 79.7),
     (88, 80.4), (87.2, 80.8), (86.3, 80.5), (85.5, 79.9), (84.7, 79.2)],
    # Amund Ringnes Island
    [(92.2, 78.3), (93, 78.5), (94, 78.5), (95.1, 78.9), (95.2, 79.5),
     (94.4, 80.0), (93.5, 80.1), (92.6, 79.7), (92.1, 79.1)],
    # Ellef Ringnes Island
    [(100.0, 77.0), (101.2, 77.2), (102.5, 77.0), (103.7, 77.3),
     (104.6, 77.8), (104.8, 78.5), (104.2, 79.2), (102.8, 79.4),
     (101.6, 79.1), (100.4, 79.3), (99.8, 78.8), (99.9, 77.9)],
    # Bathurst Island
    [(104.8, 75.1), (103.7, 75.6), (103.3, 76.2), (102.6, 76.5),
     (101.4, 76.5), (100.6, 76.9), (99.5, 76.9), (98.7, 76.5),
     (97.8, 76.5), (97.5, 76.0), (98.2, 75.6), (99.2, 75.5),
     (100.2, 75.1), (101.2, 75.0), (102.1, 74.7), (103.0, 74.9)],
    # Cornwallis Island
    [(96.1, 74.8), (96.7, 74.9), (97.2, 75.2), (97.2, 75.5),
     (96.7, 75.7), (96.1, 75.6), (95.8, 75.3)],
    # Devon Island
    [(91.2, 74.0), (90.6, 74.5), (90.4, 75.1), (89.7, 75.5),
     (88.7, 75.8), (87.6, 75.7), (86.7, 75.4), (85.5, 75.4),
     (84.7, 75.1), (83.8, 75.2), (82.8, 75.0), (81.8, 74.7),
     (81, 74.7), (81, 73.0), (84.0, 73.0), (85.0, 73.3),
     (86.1, 73.5), (87.2, 73.8), (88.2, 73.6), (89.2, 73.8),
     (90.3, 73.6)],
    # Somerset Island
    [(95.7, 73.0), (95.6, 73.7), (95.9, 74.5), (95.4, 75.0),
     (94.4, 75.3), (93.5, 75.2), (92.9, 74.8), (92.1, 74.4),
     (91.5, 73.8), (91.7, 73.0)],
    # Prince of Wales Island (eastern portion)
    [(103.0, 73.0), (102.4, 73.6), (102.0, 74.4), (101.1, 74.9),
     (100.1, 75.0), (99.4, 74.7), (98.9, 74.1), (98.3, 73.8),
     (98.2, 73.0)],
    # Northern Victoria Island
    [(105, 73.0), (104.8, 73.5), (104, 73.9), (103, 74.0),
     (102, 73.8), (101, 73.7), (100, 73.5), (99, 73.3),
     (98, 73.0)],
]

# Small islands and rugged coastal fragments within the sampled window
LAND += [
    [(95.0, 76.6), (95.5, 76.7), (95.8, 77.0), (95.4, 77.2), (94.9, 77.0)],
    [(96.0, 77.1), (96.5, 77.2), (96.8, 77.5), (96.3, 77.7), (95.9, 77.5)],
    [(97.1, 77.0), (97.8, 77.1), (98.0, 77.4), (97.5, 77.6), (97.0, 77.4)],
    [(90.0, 76.2), (90.5, 76.4), (90.3, 76.7), (89.8, 76.6)],
    [(93.8, 77.6), (94.3, 77.7), (94.6, 78.0), (94.1, 78.1), (93.7, 77.9)],
    [(82.5, 76.2), (83.0, 76.3), (83.2, 76.6), (82.7, 76.7)],
]


def inside(lon, lat, polygon):
    """Ray-crossing point-in-polygon test; boundary hits count as land."""
    hit = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
        if abs(cross) < 1e-10 and min(x1, x2) <= lon <= max(x1, x2) \
                and min(y1, y2) <= lat <= max(y1, y2):
            return True
        if (y1 > lat) != (y2 > lat):
            x_cross = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < x_cross:
                hit = not hit
    return hit


def generate():
    lats = [79.0 - 0.1 * i for i in range(61)]
    west_lons = [105.0 - 0.1 * i for i in range(241)]
    return np.array([
        [int(any(inside(w, lat, poly) for poly in LAND)) for w in west_lons]
        for lat in lats
    ], dtype=np.uint8)


def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    matrix = generate()
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(matrix.tolist())
    # The stored matrix is north-to-south and west-to-east. White is land.
    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(out / "candidate.png")
    image.resize((1928, 488), resample=Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )
    assert matrix.shape == (61, 241) and matrix.size == 14701
    assert set(np.unique(matrix)) <= {0, 1}
    with (out / "candidate.csv").open(newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 61 and all(len(row) == 241 for row in rows)
    assert all(v in ("0", "1") for row in rows for v in row)
    assert Image.open(out / "candidate.png").size == (241, 61)
    enlarged = Image.open(out / "candidate_8x.png")
    assert enlarged.size == (1928, 488)
    assert set(enlarged.getdata()) <= {(0, 0, 0), (255, 255, 255)}
    assert np.array_equal(np.asarray(enlarged)[::8, ::8], np.asarray(image))


if __name__ == "__main__":
    main()
