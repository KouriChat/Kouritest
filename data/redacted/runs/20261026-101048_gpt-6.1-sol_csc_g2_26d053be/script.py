"""Generate a hand-estimated South China Sea land/water sample raster."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

LAT0, LON0, STEP = 23.0, 99.0, 0.2
ROWS, COLS = 111, 141

# Approximate hand-drawn outlines, ordered as longitude/latitude vertices.
LAND = [
    # Mainland Asia and Indochina; frame-edge portions are intentionally clipped.
    [(99,23),(127,23),(123,22.7),(121.5,22.4),(120.7,21.9),(120.2,21.2),
     (119.6,20.6),(118.8,20.2),(117.5,20),(116.5,19.7),(115.5,19.4),
     (114.8,19),(114,18.5),(113.4,17.8),(112.6,17.2),(111.8,16.9),
     (111.3,16.2),(110.8,15.7),(110.2,15.4),(109.5,15),(109.2,14.4),
     (109.4,13.8),(109.3,13),(109,12.1),(108.7,11.4),(108.4,10.7),
     (107.8,10.1),(106.9,9.5),(106.2,9),(105.8,8.6),(105.2,8.6),
     (104.8,9.1),(104.5,10),(104.2,10.8),(103.7,11.5),(103.2,12.2),
     (102.8,13.2),(102.5,14),(102,14.7),(101.5,15.5),(101,16.2),
     (100.5,17),(100,18),(99.5,19),(99,20),(99,1),(101.5,1),
     (102.2,2),(103,3.5),(103.5,5),(103.6,6.5),(103,7.5),
     (101.5,8.5),(100.5,10),(100,12),(99.5,15)],
    # Taiwan, Hainan
    [(120,21.8),(120.2,22),(120.5,22.4),(120.9,23),(121.3,23.8),
     (121.5,24.7),(121.3,25.2),(120.9,25.3),(120.5,25),(120.2,24.3),
     (120,23.5),(119.9,22.7)],
    [(108.6,18.2),(109,18.1),(109.5,18.3),(110,18.7),(110.5,19.2),
     (110.6,19.7),(110.2,20),(109.7,20.1),(109.2,20),(108.8,19.6),
     (108.6,19)],
    # Sumatra, Java, Borneo, Sulawesi
    [(95,5.8),(97,5.5),(99,3.8),(101,2.2),(103,0),(105,-2),(106,-4.8),
     (104,-5.8),(102,-5.4),(100,-3.5),(98,-1),(96,1.5),(95,4)],
    [(105,-5.7),(107,-5.8),(110,-6),(113,-6.5),(115,-7.5),(113,-8.5),
     (110,-8.8),(108,-8.4),(106,-7.2)],
    [(108,7),(109,6.8),(110,5.5),(112,4.5),(114,4),(116,4.5),
     (118,4.2),(119,2.5),(118,.5),(117,-1.5),(115,-3),(113,-3.2),
     (111,-2),(110,0),(109,2)],
    [(119.5,1.5),(121,1),(123,0),(124,-1),(123,-2.5),(122,-3.5),
     (121,-2),(120,-1)],
    # Major Philippines
    [(119.5,18.5),(120.2,18.3),(120.8,17.5),(120.7,16.5),(120.3,15.8),
     (119.8,16.1),(119.5,17.2)],
    [(120.8,18.5),(122,18),(122.4,17),(122,16),(121.2,16.1),(120.8,17)],
    [(120,15),(121.4,14.8),(122,13.8),(122,12),(121.3,10.8),(120.6,9.8),
     (119.8,10),(119.5,11.5),(119.7,13)],
    [(122,13.3),(123.5,13.5),(124,12.5),(123.5,11.5),(122.5,11.2),(121.8,12)],
    [(123,10.8),(124.5,10.5),(125.5,9.5),(126,8),(125.3,7),(124.2,7),(123.5,8.5)],
    [(124.5,7),(126,6),(126.7,5.5),(125.8,4.8),(124.8,5.3),(124.2,6.2)],
]

# Small islands, reefs, and cays represented by approximate elliptical outlines.
ISLANDS = [
    (114.35,10,.14,.12),(114.2,11,.12,.16),(114,12,.10,.14),
    (113.7,13,.12,.14),(113.5,14,.12,.14),(112.5,16,.10,.12),
    (111.5,17,.09,.12),(110.5,18,.08,.10),(117,11.5,.14,.10),
    (116,10.5,.13,.10),(115,9.5,.12,.10),
    (111.9,8.65,.10,.08),(112,8,.08,.08),(112.3,7.5,.08,.08),
    (113,8.8,.10,.08),(114.3,8,.09,.08),(114.6,9,.08,.08),
    (115.8,9.8,.08,.08),(116.7,10.3,.08,.07),(109.5,7.5,.08,.08),
    (108.7,7,.07,.07),(117.5,8.8,.07,.07),(112.3,16.5,.16,.12),
    (116.7,20.7,.10,.07),(108.2,4,.12,.12),(106,3,.16,.12),
    (105.8,2.5,.12,.10),(104.3,1.7,.10,.08),(103.8,1.4,.10,.08),
    (99,9,.10,.14),(98.5,7.5,.08,.12),(97.5,5.5,.08,.10),
    (119,5.5,.10,.10),(118,6,.08,.08),(117,6,.08,.08),
]


def inside_polygon(x, y, vertices):
    inside = False
    j = len(vertices) - 1
    for i, (xi, yi) in enumerate(vertices):
        xj, yj = vertices[j]
        if (yi > y) != (yj > y):
            cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < cross:
                inside = not inside
        j = i
    return inside


def inside_island(x, y, cx, cy, rx, ry):
    dx, dy = (x - cx) / rx, (y - cy) / ry
    angle = np.arctan2(dy, dx)
    edge = 1 + .10 * np.sin(3 * angle) + .06 * np.cos(5 * angle)
    return dx * dx + dy * dy <= edge


def build_matrix():
    matrix = np.zeros((ROWS, COLS), dtype=np.uint8)
    for row in range(ROWS):
        lat = LAT0 - STEP * row
        for col in range(COLS):
            lon = LON0 + STEP * col
            if any(inside_polygon(lon, lat, p) for p in LAND):
                matrix[row, col] = 1
            elif any(inside_island(lon, lat, *p) for p in ISLANDS):
                matrix[row, col] = 1
    return matrix


def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(matrix.tolist())
    image = Image.fromarray(matrix * np.uint8(255), mode="L")
    image.save(out / "candidate.png")
    large = image.resize((1128, 888), resample=Image.Resampling.NEAREST)
    large.save(out / "candidate_8x.png")
    with (out / "candidate.csv").open(newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 111 and all(len(r) == 141 for r in rows)
    assert all(v in {"0", "1"} for r in rows for v in r)
    assert image.size == (141, 111) and large.size == (1128, 888)
    assert set(image.getdata()) == {0, 255}
    assert np.array_equal(np.asarray(large)[::8, ::8], matrix * 255)
    print("Validated 111x141 matrix (15651 binary values) and both PNGs.")


if __name__ == "__main__":
    main()
