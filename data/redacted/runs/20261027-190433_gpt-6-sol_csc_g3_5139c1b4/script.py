from pathlib import Path
import csv
import numpy as np
from PIL import Image


# Hand-written approximate mainland outlines, in (longitude, latitude).
MAINLAND = [
    (99.0, 23.0), (120.3, 23.0), (120.0, 22.2), (119.5, 21.8),
    (119.0, 20.8), (118.3, 19.8), (117.5, 18.8), (116.9, 17.8),
    (116.5, 16.8), (116.0, 15.8), (115.5, 14.8), (115.1, 13.8),
    (114.7, 12.8), (114.3, 11.8), (113.8, 10.8), (113.0, 9.8),
    (112.0, 8.4), (110.9, 7.3), (109.5, 5.8), (108.2, 4.3),
    (107.0, 2.8), (105.4, 1.0), (99.0, 1.0),
]

INDOCHINA = [
    (99.0, 23.0), (101.0, 23.0), (101.8, 21.0), (102.5, 19.0),
    (103.2, 16.0), (104.0, 12.0), (105.0, 9.0), (105.8, 7.0),
    (105.3, 5.0), (104.2, 3.0), (102.8, 1.0), (100.2, 1.0),
    (100.5, 4.0), (100.2, 7.0), (99.8, 10.0), (99.2, 12.0),
    (99.0, 15.0), (99.3, 17.0), (99.8, 19.0), (100.3, 21.0),
    (99.5, 22.5),
]


def point_in_polygon(lon, lat, polygon):
    inside = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        cross = (lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
        if abs(cross) < 1e-10 and min(x1, x2) <= lon <= max(x1, x2):
            if min(y1, y2) <= lat <= max(y1, y2):
                return True
        if (y1 > lat) != (y2 > lat):
            if lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
    return inside


ISLANDS = [
    # Hainan and Taiwan
    [(108.5, 20.1), (109.5, 20.2), (110.4, 19.4),
     (110.1, 18.5), (109.1, 18.3), (108.5, 19.2)],
    [(120.0, 25.3), (121.0, 25.3), (121.8, 24.0),
     (121.0, 22.5), (120.0, 23.0)],
    # Borneo
    [(109.0, 7.0), (111.0, 7.0), (113.0, 6.6), (115.0, 5.9),
     (117.5, 3.5), (116.0, 1.7), (113.0, 1.0), (110.0, 1.5),
     (108.5, 3.2), (108.7, 5.7)],
    # Sumatra
    [(105.0, 5.8), (105.7, 4.5), (105.2, 2.5), (104.4, 0.5),
     (103.2, -1.5), (101.8, -4.5), (100.5, -5.8),
     (99.5, -4.5), (100.2, -1.5), (100.8, 1.5),
     (102.5, 4.5), (104.2, 5.7)],
    # Java
    [(105.0, -5.7), (108.0, -6.2), (111.0, -7.0),
     (114.8, -8.0), (114.5, -8.5), (111.5, -8.0),
     (108.5, -7.3), (105.5, -6.3)],
]

PHILIPPINES = [
    [(119.5, 18.6), (120.5, 18.5), (121.0, 17.8),
     (120.7, 17.0), (119.5, 17.5)],
    [(120.0, 16.5), (121.0, 16.5), (121.5, 15.5),
     (121.0, 14.5), (120.0, 13.8), (119.8, 14.8)],
    [(122.0, 12.8), (123.0, 12.5), (123.5, 11.5),
     (123.0, 10.8), (122.2, 11.0), (121.8, 11.8)],
    [(122.0, 11.0), (123.5, 10.8), (124.5, 10.0),
     (123.8, 9.5), (122.8, 9.7), (122.0, 10.2)],
    [(123.0, 9.5), (124.5, 9.5), (125.5, 8.8),
     (125.5, 7.5), (124.0, 6.0), (123.2, 6.5), (122.8, 7.5)],
]

ISLAND_ELLIPSES = [
    # Paracel and Spratly groups
    (112.33, 16.5, 0.22, 0.20), (111.7, 16.5, 0.12, 0.12),
    (112.7, 16.3, 0.10, 0.10), (114.3, 10.3, 0.18, 0.14),
    (114.1, 11.0, 0.12, 0.12), (115.8, 10.0, 0.16, 0.12),
    (116.0, 9.7, 0.10, 0.10), (116.7, 10.4, 0.13, 0.12),
    (116.9, 9.8, 0.12, 0.10), (117.2, 8.8, 0.12, 0.10),
    (113.9, 8.8, 0.10, 0.10), (112.9, 7.5, 0.10, 0.08),
    # Natuna, Anambas, and nearby island groups
    (109.2, 7.3, 0.12, 0.10), (108.2, 3.9, 0.20, 0.18),
    (106.0, 3.0, 0.13, 0.12), (104.5, 1.0, 0.15, 0.12),
    (108.0, 1.5, 0.12, 0.10), (102.5, 9.7, 0.10, 0.12),
    (100.7, 7.8, 0.10, 0.12),
]


def ellipse_contains(lon, lat, item):
    cx, cy, rx, ry = item
    return ((lon - cx) / rx) ** 2 + ((lat - cy) / ry) ** 2 <= 1.0


def is_land(lon, lat):
    if point_in_polygon(lon, lat, MAINLAND):
        return True
    if point_in_polygon(lon, lat, INDOCHINA):
        return True
    if any(point_in_polygon(lon, lat, p) for p in ISLANDS):
        return True
    if any(point_in_polygon(lon, lat, p) for p in PHILIPPINES):
        return True
    return any(ellipse_contains(lon, lat, item) for item in ISLAND_ELLIPSES)


def main():
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.zeros((111, 141), dtype=np.uint8)
    for row in range(111):
        lat = 23.0 - row * 0.2
        for col in range(141):
            lon = 99.0 + col * 0.2
            matrix[row, col] = int(is_land(lon, lat))

    with (out_dir / "candidate.csv").open(
            "w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(matrix.tolist())

    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(out_dir / "candidate.png")
    enlarged = image.resize((1128, 888), resample=Image.Resampling.NEAREST)
    enlarged.save(out_dir / "candidate_8x.png")

    with (out_dir / "candidate.csv").open(
            "r", newline="", encoding="ascii") as stream:
        rows = list(csv.reader(stream))
    assert matrix.shape == (111, 141) and matrix.size == 15651
    assert len(rows) == 111
    assert all(len(row) == 141 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert image.size == (141, 111)
    assert enlarged.size == (1128, 888)
    assert set(image.getdata()) == {(0, 0, 0), (255, 255, 255)}
    print("Validated matrix and images in outputs/.")


if __name__ == "__main__":
    main()
