from pathlib import Path
import csv
import numpy as np
from PIL import Image

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
ROWS, COLS = 111, 141


def pip(x, y, poly):
    inside = False
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        cross = (x-x1)*(y2-y1) - (y-y1)*(x2-x1)
        if abs(cross) < 1e-10 and min(x1,x2) <= x <= max(x1,x2) \
                and min(y1,y2) <= y <= max(y1,y2):
            return True
        if (y1 > y) != (y2 > y):
            if x < x1 + (y-y1)*(x2-x1)/(y2-y1):
                inside = not inside
    return inside


def ellipse(x, y, cx, cy, rx, ry):
    return ((x-cx)/rx)**2 + ((y-cy)/ry)**2 <= 1


POLYS = [
    [(99,23),(117.5,23),(117.4,22.6),(117,22.1),(116.5,21.7),
     (115.8,21.35),(115,21.05),(114,20.85),(112.8,20.85),
     (111.5,21.1),(110.2,21.35),(109,21.6),(108,21.5),
     (107.4,22),(106,22.4),(104,22.7),(101,22.85),(99,22.85)],
    [(99,22.9),(107.5,22.9),(108,21.5),(107.8,20.5),(107.5,19),
     (107.2,17),(106.8,15),(106.5,13),(105.8,11),(104.8,10),
     (104,9.8),(103,10.2),(102,11),(101.3,12.5),(100.5,14.5),
     (99.7,17),(99,19.5)],
    [(99.1,19),(100,18),(100.5,16),(100.5,14),(100.2,12),
     (99.5,10),(99.2,8),(99.4,7),(100.2,7.5),(101.2,7.2),
     (102,6.6),(102.7,5.6),(103.3,4.5),(103.8,3.3),(104.2,2),
     (104.3,1),(103.2,1),(102.5,2.2),(101.6,3.3),(100.7,4.3),
     (99.8,5.5),(99.2,6.5),(98.8,7),(98.5,10),(98.5,14),
     (98.8,18)],
    [(99,1),(105,1),(104.2,1.5),(103.2,2.2),(102.5,3),
     (101.8,4),(100.5,5.2),(99.5,5),(98.5,4),(98,2)],
    [(109,1),(118,1),(118.5,2),(118.2,4),(117.5,5.5),(116,6.5),
     (114.5,7.2),(112,7.3),(110,6.5),(109.5,5),(109,3)],
    [(120,22),(120.2,23),(121,23.5),(122,23),(121.8,22.2),
     (121.3,21.8),(120.7,21.8)],
]

ELL = [(109.8,19.2,1.15,.62),(108.3,16.1,.16,.28),
       (109.3,12,.18,.3),(123.1,11,.18,.28),(124,10.2,.15,.22)]
POLYS.extend([
    [(119.8,18.5),(120.2,19.5),(120.5,20.5),(121.2,21),
     (122,20.8),(122.2,19.5),(121.8,18),(121.2,16.5),
     (120.8,15),(120.3,14),(119.5,14.5),(119.2,16),(119.3,17.5)],
    [(120,14.8),(121,14.5),(122,14),(123,13.5),(123.5,13),
     (123,12.3),(122,12.5),(121,13),(120,13.8)],
    [(120,13.5),(121.2,13.5),(121.8,12.7),(121.4,11.8),
     (120.5,11.7),(119.8,12.5)],
    [(117,11.5),(118,12),(119,11),(119.5,10),(119,8),
     (118.3,7),(117.5,8.5),(117,10)],
    [(122,10),(123,10),(124,9.5),(125.5,9),(126.5,7.5),
     (126,6),(124.5,5),(123,5.5),(122,7),(121.5,8.5)],
    [(122.5,12),(123.5,12.5),(124.5,12),(124.5,11),
     (123.5,10.5),(122.5,11)],
    [(124.5,11.5),(125.5,12),(126,11),(125.5,10),(124.5,10.5)],
])
ELL.extend([
    (111.7,16.6,.15,.11),(112,16.5,.11,.1),
    (112.3,16.8,.12,.1),(111.3,16.3,.1,.08),
    (112.7,9.7,.13,.09),(113.8,8.7,.12,.08),
    (114.1,9.9,.11,.09),(114.3,10.4,.12,.1),
    (114.3,11,.12,.09),(114.8,10,.1,.08),
    (115.5,9.5,.12,.09),(115.8,11,.11,.09),
    (116,8.5,.12,.09),(116.7,8.7,.12,.08),
    (117.2,10,.11,.09),(117.5,8.9,.1,.08),
    (110.8,12,.1,.08),(111.5,11.5,.09,.07),
    (112.5,13.5,.09,.07),(113.2,12.2,.1,.07),(115,12.5,.1,.07),
])


def is_land(lon, lat):
    return int(any(pip(lon, lat, p) for p in POLYS)
               or any(ellipse(lon, lat, *e) for e in ELL))


def generate():
    return np.array([
        [is_land(99 + c / 5, 23 - r / 5) for c in range(COLS)]
        for r in range(ROWS)
    ], dtype=np.uint8)


def save(matrix):
    with (OUT / "candidate.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerows(matrix.tolist())
    pixels = matrix * np.uint8(255)
    image = Image.fromarray(pixels)
    image.save(OUT / "candidate.png")
    image.resize((1128, 888), Image.Resampling.NEAREST).save(
        OUT / "candidate_8x.png")


def validate(matrix):
    assert matrix.shape == (111, 141) and matrix.size == 15651
    assert set(np.unique(matrix)) <= {0, 1}
    with (OUT / "candidate.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 111
    assert all(len(row) == 141 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert np.array_equal(np.array(rows, dtype=np.uint8), matrix)
    with Image.open(OUT / "candidate.png") as im:
        assert im.size == (141, 111)
        raw = np.array(im)
    with Image.open(OUT / "candidate_8x.png") as im:
        assert im.size == (1128, 888)
        large = np.array(im)
    assert set(np.unique(raw)) <= {0, 255}
    assert set(np.unique(large)) <= {0, 255}
    assert np.array_equal(raw, matrix * np.uint8(255))
    assert np.array_equal(large, raw.repeat(8, axis=0).repeat(8, axis=1))
    print("Validated: 111x141, 15651 binary values; PNGs 141x111 and 1128x888.")


if __name__ == "__main__":
    result = generate()
    save(result)
    validate(result)
