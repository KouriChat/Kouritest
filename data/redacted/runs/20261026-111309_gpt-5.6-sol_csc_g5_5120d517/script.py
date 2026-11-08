from pathlib import Path
import csv
import numpy as np
from PIL import Image


LAT_TOP, LAT_BOTTOM, LON_LEFT, LON_RIGHT, STEP = 23.0, 1.0, 99.0, 127.0, 0.2

# Hand-entered approximate coastlines. Vertices are (longitude, latitude).
POLYGONS = [
    # Southern mainland Asia and China within the frame.
    [(99, 23), (127, 23), (126, 22.7), (124, 22.5), (122.5, 21.7),
     (121, 21.5), (120, 22.4), (119, 23.5), (117.5, 23.5), (116.5, 23),
     (115, 22.7), (114, 22.5), (113, 22.2), (112, 21.8), (111, 21.2),
     (110, 20.8), (109.5, 21), (108.5, 21.5), (107.5, 21),
     (106.5, 20.5), (106, 19.5), (105.5, 18.5), (106, 17.5),
     (107, 16.5), (108, 16), (108.5, 15), (109, 14), (109.5, 12.5),
     (109, 11), (108, 10.5), (107, 10), (106, 9.5), (105.5, 9),
     (104.5, 8.5), (103.5, 8), (102.5, 7), (101.5, 6.5), (100, 7),
     (99, 8), (99, 23)],
    # Malay Peninsula.
    [(100, 6.8), (101.5, 6.5), (102.2, 5.5), (102.5, 4.5),
     (103, 3.5), (103.5, 2.5), (104, 1), (102, 1), (101.2, 2),
     (100.5, 3), (100, 4.5)],
    # Sumatra.
    [(95, 5.8), (97, 5.5), (99, 4.5), (101, 2.8), (103, 0),
     (101, -1.5), (99, -3.5), (97, -4.5), (95.5, -3), (94.5, 0),
     (94.8, 3)],
    # Java.
    [(105, -5.8), (107, -5.8), (109, -6.1), (111, -6.5),
     (113, -7.5), (114, -8.5), (112, -8.8), (110, -8.5),
     (108, -8.2), (106, -7.8)],
    # Borneo.
    [(108, 7.5), (110, 7.2), (112, 6.5), (114, 5.5), (116, 4.5),
     (117.5, 2.5), (117, 0), (115, -2), (113, -3.5), (111, -3.5),
     (109, -2), (108, 0.5), (108, 3)],
    # Taiwan, Hainan, Palawan, and major Philippine islands.
    [(120, 25.5), (121, 25.2), (121.8, 24), (121.5, 22.5),
     (120.8, 21.8), (120, 22.5)],
    [(108.5, 20), (110, 20), (111, 19.3), (110.5, 18.2),
     (109.5, 18), (108.5, 19)],
    [(117, 11.8), (118, 11.5), (119, 10.5), (119, 9), (118.5, 8),
     (117.5, 8.5), (117, 10)],
    [(119.5, 18.5), (121, 18.5), (122, 17), (123, 15), (124, 13),
     (125, 11), (125, 9), (124, 7), (123, 6), (122, 7), (121, 9),
     (120, 11), (119, 13), (119, 16)],
    [(123, 11.5), (124, 11), (125, 10), (125.5, 8), (124.5, 7),
     (123.5, 8), (123, 10)],
    [(122, 7), (123, 6), (124, 5), (125, 5), (126, 7),
     (125, 8), (123, 8)],
    # Sulawesi.
    [(119, 1), (120, 1.5), (121, 1), (122, 0), (123, -1),
     (122, -2), (121, -2.5), (120, -1), (119, -1)],
]

# Small islands and reef features, represented as (lon, lat, lon radius, lat radius).
ELLIPSES = [
    (99.2, 21, .12, .12), (120.4, 18.2, .25, .25), (118.2, 13, .2, .3),
    (121.5, 12.5, .18, .3), (122.2, 10.5, .15, .25),
    # Spratly and Paracel archipelagos, including reef tops.
    (112.3, 16.5, .14, .10), (111.8, 16.3, .10, .08),
    (114.3, 11.8, .12, .10), (114.6, 10.8, .12, .10),
    (115.0, 10.3, .11, .10), (115.5, 10.0, .10, .08),
    (116.0, 10.5, .10, .08), (116.5, 10.8, .09, .08),
    (117.0, 11.0, .10, .08), (113.8, 8.8, .10, .08),
    (114.5, 8.5, .10, .08), (115.0, 8.2, .10, .08),
    (116.0, 8.5, .10, .08), (116.5, 9.0, .09, .08),
    (117.5, 9.5, .08, .08), (111.7, 12.0, .08, .08),
    # Andaman and Nicobar islands.
    (92, 13, .3, 1), (93, 8, .25, .8),
    # Riau, Bangka-Belitung, and nearby Indonesian islands.
    (104, .5, .5, .3), (106, -2.5, .5, .8), (108, -3, .4, .6),
]


def inside_polygon(lon, lat, vertices):
    inside = False
    j = len(vertices) - 1
    for i, (xi, yi) in enumerate(vertices):
        xj, yj = vertices[j]
        if (yi > lat) != (yj > lat):
            crossing = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing:
                inside = not inside
        j = i
    return inside


def is_land(lon, lat):
    if any(inside_polygon(lon, lat, polygon) for polygon in POLYGONS):
        return True
    for cx, cy, rx, ry in ELLIPSES:
        if ((lon - cx) / rx) ** 2 + ((lat - cy) / ry) ** 2 <= 1:
            return True
    return False


def main():
    lats = np.round(np.arange(LAT_TOP, LAT_BOTTOM - 1e-8, -STEP), 1)
    lons = np.round(np.arange(LON_LEFT, LON_RIGHT + 1e-8, STEP), 1)
    matrix = np.array([[int(is_land(float(lon), float(lat))) for lon in lons]
                       for lat in lats], dtype=np.uint8)
    assert matrix.shape == (111, 141) and matrix.size == 15651

    output = Path("outputs")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

    # Pixel value 255 is land (white); 0 is water (black).
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(output / "candidate.png")
    image.resize((1128, 888), resample=Image.Resampling.NEAREST).save(
        output / "candidate_8x.png"
    )

    with (output / "candidate.csv").open(newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 111 and all(len(row) == 141 for row in rows)
    assert all(value in ("0", "1") for row in rows for value in row)
    assert set(Image.open(output / "candidate.png").getdata()) <= {0, 255}
    assert Image.open(output / "candidate.png").size == (141, 111)
    enlarged = Image.open(output / "candidate_8x.png")
    assert enlarged.size == (1128, 888)
    assert np.array_equal(np.asarray(enlarged)[::8, ::8], matrix * 255)
    print(f"Validated {matrix.shape[0]}x{matrix.shape[1]} matrix, 15651 values.")


if __name__ == "__main__":
    main()
