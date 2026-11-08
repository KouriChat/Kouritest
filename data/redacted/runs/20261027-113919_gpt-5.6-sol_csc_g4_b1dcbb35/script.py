from pathlib import Path
import csv
import numpy as np
from PIL import Image


# Hand-estimated WGS84 coastline vertices. Coordinates are (longitude, latitude).
# These intentionally approximate coastlines and islands; no external data is used.
LAND = [
    # Mainland Asia, including China, Indochina, and the Malay Peninsula.
    [(90, 30), (98, 28), (103, 27), (108, 24), (112, 22), (115, 22),
     (117, 23), (119, 25), (121, 28), (124, 31), (128, 34), (132, 38),
     (136, 42), (140, 46), (145, 50), (150, 55), (150, 65), (90, 65)],
    # Myanmar and west side of Indochina: Gulf of Thailand shores are explicit.
    [(90, 23), (96, 23), (98, 22), (99.5, 20.5), (99.5, 18.5),
     (100.2, 17.5), (100.1, 16), (101, 15), (101, 13), (101.5, 11),
     (102, 10), (102.5, 8), (103, 7), (103.5, 5.5), (104, 4),
     (104.5, 2), (104.5, 0), (90, 0)],
    # Vietnam coast and Cambodia; central Vietnam bends east around Da Nang.
    [(102, 23), (106, 23), (108, 22), (109.5, 21), (108.5, 20),
     (107.5, 19), (107.2, 18), (107.8, 17), (108.2, 16), (108.8, 15),
     (109.4, 14), (109.3, 13), (109, 12), (108.5, 11), (108, 10),
     (107.5, 9), (106.8, 8.5), (106, 8.7), (105, 9.5), (104, 10),
     (103, 10), (102, 10), (101.5, 11), (101, 13), (101, 15),
     (100.1, 16), (100.2, 17.5), (99.5, 18.5), (99.5, 20.5),
     (98, 22), (98, 24)],
    # Hainan Island.
    [(108.55, 19.55), (109.2, 20.05), (110.2, 20.05), (111.05, 19.65),
     (111.05, 19.0), (110.55, 18.25), (109.7, 18.15), (109.1, 18.35),
     (108.65, 18.85)],
    # Taiwan.
    [(120.0, 25.3), (121.0, 25.3), (121.65, 24.4), (121.45, 23.2),
     (120.9, 22.5), (120.0, 22.0), (120.0, 23.0), (120.0, 24.2)],
]
LAND += [
    # Sumatra, clipped naturally by the requested frame.
    [(95, 5.5), (97, 5.8), (99, 5.2), (101, 4.2), (103, 3.0),
     (104, 1.5), (104.2, 0), (102, -2), (100, -5), (98, -4), (96, -1)],
    # Bangka, Belitung, and nearby larger islands.
    [(105.0, -2.8), (106.2, -2.7), (106.8, -3.2), (106.4, -4.0), (105.3, -4.2)],
    [(107.4, -2.5), (108.2, -2.7), (108.8, -3.2), (108.2, -3.7), (107.5, -3.4)],
    # Borneo (Kalimantan), with the north coast and Sabah represented.
    [(108.0, 7.2), (109.5, 7.0), (110.7, 6.8), (112.5, 6.6),
     (114.0, 7.0), (115.5, 6.8), (117.0, 6.2), (118.0, 5.3),
     (118.3, 4.0), (117.8, 2.5), (117.2, 1.0), (116.0, 0.5),
     (114.5, 0.8), (113.0, 1.2), (111.0, 1.0), (109.5, 1.5),
     (108.0, 2.5), (108.0, 4.0), (109.0, 5.5)],
    # Natuna and Anambas island groups, generalized as many small islands.
    [(107.3, 3.0), (108.0, 3.2), (108.5, 3.0), (108.4, 2.6), (107.7, 2.5)],
    [(105.5, 3.2), (106.0, 3.4), (106.3, 3.1), (106.1, 2.8), (105.7, 2.9)],
    # Palawan and the Calamian group.
    [(117.0, 11.0), (118.0, 11.2), (118.7, 10.7), (118.5, 10.0),
     (118.0, 9.3), (117.7, 8.3), (117.3, 7.4), (117.0, 6.8),
     (116.6, 7.4), (116.8, 8.5), (116.7, 9.5), (116.8, 10.4)],
    [(119.5, 12.2), (120.0, 12.4), (120.3, 12.0), (120.1, 11.7), (119.7, 11.8)],
    # Mindoro and Panay.
    [(120.3, 13.5), (121.0, 13.4), (121.3, 13.0), (121.0, 12.4),
     (120.5, 12.3), (120.1, 12.7)],
    [(121.7, 11.8), (122.5, 11.7), (122.8, 11.2), (122.3, 10.4),
     (121.8, 10.6), (121.4, 11.2)],
    # Negros and Cebu.
    [(122.4, 10.9), (123.1, 10.8), (123.3, 10.0), (123.0, 9.5),
     (122.6, 9.8), (122.3, 10.3)],
    [(123.4, 11.3), (123.8, 11.2), (124.0, 10.6), (123.8, 10.0),
     (123.5, 10.2)],
]
LAND += [
    # Samar and Leyte.
    [(124.0, 12.5), (124.8, 12.4), (125.2, 11.8), (125.0, 11.2),
     (124.5, 10.9), (124.1, 11.4)],
    [(124.7, 11.4), (125.3, 11.5), (125.4, 10.8), (124.9, 10.4), (124.6, 10.8)],
    # Mindanao, southern Philippines.
    [(122.0, 9.8), (122.8, 9.6), (123.5, 9.0), (124.1, 8.7),
     (124.7, 8.9), (125.2, 8.4), (125.6, 7.5), (126.3, 7.0),
     (126.3, 6.2), (125.8, 5.6), (125.0, 5.4), (124.2, 5.6),
     (123.5, 6.0), (123.0, 6.5), (122.5, 7.2), (122.0, 8.2)],
    # Sulu archipelago and Tawi-Tawi islands.
    [(120.0, 6.2), (120.5, 6.3), (120.7, 5.9), (120.4, 5.6), (120.0, 5.8)],
    [(121.0, 5.5), (121.7, 5.5), (122.0, 5.1), (121.5, 4.8), (121.1, 5.0)],
    [(119.5, 5.0), (120.1, 5.1), (120.4, 4.8), (120.1, 4.5), (119.7, 4.6)],
    # Sulawesi's northern arm and the edge of Celebes within frame.
    [(120.0, 1.0), (121.0, 1.5), (122.0, 1.8), (123.0, 1.5),
     (124.0, 1.0), (125.0, 1.0), (126.0, 1.5), (127.2, 2.0),
     (127.2, 0.5), (126.0, 0), (120.0, 0)],
    # Riau Islands, scattered just south of Singapore.
    [(103.3, 1.2), (104.0, 1.3), (104.3, 1.0), (104.0, 0.7), (103.5, 0.8)],
    [(105.0, 1.0), (105.6, 1.1), (105.8, 0.8), (105.4, 0.6), (105.1, 0.7)],
]


def on_segment(px, py, ax, ay, bx, by):
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > 1e-9:
        return False
    return (min(ax, bx) - 1e-9 <= px <= max(ax, bx) + 1e-9 and
            min(ay, by) - 1e-9 <= py <= max(ay, by) + 1e-9)
def in_polygon(lon, lat, polygon):
    inside = False
    n = len(polygon)
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]
        if on_segment(lon, lat, ax, ay, bx, by):
            return True
        if (ay > lat) != (by > lat):
            crossing_lon = ax + (lat - ay) * (bx - ax) / (by - ay)
            if lon < crossing_lon:
                inside = not inside
    return inside


def build_matrix():
    # Rows run 23N to 1N; columns run 99E to 127E in 0.2-degree steps.
    lats = [23.0 - 0.2 * row for row in range(111)]
    lons = [99.0 + 0.2 * col for col in range(141)]
    return np.array([
        [int(any(in_polygon(lon, lat, poly) for poly in LAND)) for lon in lons]
        for lat in lats
    ], dtype=np.uint8)


def main():
    outdir = Path.cwd() / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()
    assert matrix.shape == (111, 141)
    with (outdir / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Pillow uses RGB values 255 for land and 0 for water; no resampling filter
    # other than nearest-neighbor is applied for the 8x enlargement.
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(outdir / "candidate.png")
    enlarged = image.resize((141 * 8, 111 * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(outdir / "candidate_8x.png")

    with (outdir / "candidate.csv").open("r", newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 111 and all(len(row) == 141 for row in rows)
    assert all(value in ("0", "1") for row in rows for value in row)
    assert sum(map(len, rows)) == 15651
    assert image.size == (141, 111) and enlarged.size == (1128, 888)
    assert set(image.getdata()) == {0, 255}
    assert set(enlarged.getdata()) == {0, 255}
    print("Validated 111x141 CSV, binary pixels, and 141x111 / 1128x888 PNGs")


if __name__ == "__main__":
    main()
