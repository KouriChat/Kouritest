from pathlib import Path
import csv
import numpy as np
from PIL import Image

LAT_MAX, LAT_MIN, LON_MIN, LON_MAX, STEP = 23.0, 1.0, 99.0, 127.0, 0.2

# Hand-drawn coastal outlines, ordered along the coast then closed far inland.
# Coordinates are (longitude, latitude); no external geographic data is used.
POLYGONS = [
    # Mainland Southeast Asia and southern China, including the Gulf of Tonkin.
    [(99, 28), (105, 28), (109, 25), (111, 22.5), (113, 21.5),
     (115, 21.8), (117, 22.2), (119, 24), (123, 26), (130, 28),
     (130, 40), (99, 40)],
    # Vietnam's eastern shore and the narrow Malay peninsula.
    [(99, 23.2), (102.2, 23.2), (105.5, 21.7), (107.5, 20.8),
     (108.8, 19.8), (107.8, 18.0), (107.0, 16.0), (108.0, 14.5),
     (109.5, 13.0), (109.4, 11.5), (108.0, 10.0), (106.8, 9.0),
     (105.8, 8.5), (104.8, 10.0), (103.5, 11.5), (102.0, 13.5),
     (101.0, 16.0), (100.0, 18.0), (99, 20.0)],
    # Thailand's west side; joined across the peninsula's interior.
    [(97, 23.5), (100.0, 23.5), (100.5, 21.0), (100.8, 18.0),
     (101.5, 15.0), (101.0, 12.5), (100.0, 10.5), (99.5, 9.0),
     (100.0, 7.5), (101.0, 6.0), (102.0, 5.5), (103.0, 7.0),
     (104.0, 9.0), (105.0, 11.0), (106.0, 14.0), (107, 18),
     (108, 22), (108, 26), (97, 26)],
    # Sumatra northern end and west coast.
    [(95, 6.5), (96.5, 5.8), (98, 5.2), (99.5, 4.5), (100.5, 3.8),
     (101.2, 2.8), (102.0, 1.0), (99.0, 0.0), (96, 0.0)],
    # Borneo.
    [(108.0, 7.0), (109.0, 7.2), (110.0, 6.8), (112.0, 5.5),
     (114.0, 4.5), (116.0, 4.0), (117.8, 4.2), (119.0, 5.0),
     (118.5, 6.5), (117.0, 7.0), (117.5, 8.0), (117.0, 9.0),
     (115.5, 9.5), (114.0, 8.8), (112.5, 9.0), (110.5, 8.5),
     (109.0, 8.0)],
    # Taiwan.
    [(120.0, 21.8), (120.4, 22.5), (120.9, 23.0), (121.5, 25.3),
     (122.0, 25.5), (122.1, 24.0), (121.8, 22.5), (121.2, 21.8),
     (120.6, 21.7)],
    # Luzon and the Philippine islands.
    [(119.5, 18.5), (120.0, 19.5), (121.0, 20.5), (122.0, 19.5),
     (121.8, 18.0), (122.3, 17.0), (122.0, 15.5), (121.5, 14.5),
     (121.2, 13.0), (120.5, 12.0), (120.0, 13.0), (120.0, 15.0),
     (119.5, 16.5)],
    # Mindoro, Panay, Negros, Cebu, Samar, Leyte, Mindanao.
    [(120.3, 13.5), (121.2, 13.7), (121.5, 13.0), (121.0, 12.2),
     (120.5, 12.3)],
    [(121.5, 11.8), (123.0, 12.2), (123.5, 11.5), (122.8, 10.5),
     (122.0, 10.5)],
    [(122.0, 10.5), (123.0, 10.7), (123.5, 9.5), (123.2, 8.5),
     (122.8, 8.5), (122.5, 9.5)],
    [(123.5, 12.0), (125.0, 12.5), (126.5, 11.8), (126.0, 10.0),
     (125.2, 8.0), (124.5, 6.5), (123.5, 6.0), (123.0, 7.5),
     (123.0, 9.5)],
    [(124.5, 12.5), (125.5, 13.0), (126.0, 12.2)],
    # Palawan.
    [(117.0, 11.8), (117.5, 11.0), (118.0, 10.0), (118.5, 9.0),
     (119.0, 8.0), (119.5, 7.0), (119.8, 6.0), (119.5, 5.5),
     (118.8, 6.5), (118.0, 8.0), (117.5, 9.5), (117.0, 10.5)],
    # Sulawesi's northern arm enters the southeast corner.
    [(119.0, 1.0), (120.0, 1.0), (121.0, 1.5), (122.0, 2.0),
     (123.0, 2.0), (124.0, 1.5), (125.0, 1.0), (126.0, 0.0),
     (119.0, 0.0)],
]

# Small islands and island groups represented by hand-drawn polygons.
ISLANDS = [
    # Hainan and nearby islands
    [(108.5, 19.3), (109.5, 20.0), (110.8, 20.0), (111.2, 19.0),
     (110.5, 18.0), (109.2, 18.2)],
    [(109.0, 21.1), (109.3, 21.4), (109.7, 21.2), (109.5, 20.9)],
    # Paracels
    [(111.5, 16.8), (111.9, 17.0), (112.2, 16.7), (112.0, 16.4),
     (111.6, 16.4)],
    [(112.7, 16.3), (113.0, 16.6), (113.3, 16.4), (113.1, 16.1)],
    [(112.0, 15.8), (112.3, 16.0), (112.5, 15.7), (112.2, 15.5)],
    [(111.2, 16.0), (111.5, 16.2), (111.7, 15.9), (111.4, 15.7)],
    [(113.5, 16.0), (113.8, 16.2), (114.0, 16.0), (113.7, 15.8)],
    # Macclesfield Bank / Scarborough area
    [(115.0, 15.1), (115.2, 15.3), (115.5, 15.1), (115.3, 14.9)],
    [(117.5, 15.1), (117.7, 15.3), (117.9, 15.1), (117.7, 14.9)],
    # Spratly group: substantial islands and reefs represented as point-sized land.
    [(114.3, 11.7), (114.6, 11.9), (114.8, 11.7), (114.6, 11.5)],
    [(114.0, 10.4), (114.2, 10.6), (114.4, 10.4), (114.2, 10.2)],
    [(114.8, 9.8), (115.1, 10.0), (115.3, 9.8), (115.0, 9.6)],
    [(115.8, 10.0), (116.1, 10.2), (116.3, 10.0), (116.0, 9.8)],
    [(116.5, 9.3), (116.8, 9.5), (117.0, 9.3), (116.8, 9.1)],
    [(117.2, 8.7), (117.5, 8.9), (117.7, 8.7), (117.4, 8.5)],
    [(117.8, 8.0), (118.0, 8.2), (118.3, 8.0), (118.0, 7.8)],
    [(118.5, 8.5), (118.8, 8.7), (119.0, 8.4), (118.7, 8.2)],
    [(119.3, 9.1), (119.6, 9.3), (119.8, 9.1), (119.5, 8.9)],
    [(119.8, 10.2), (120.1, 10.4), (120.3, 10.1), (120.0, 9.9)],
    [(118.5, 11.0), (118.8, 11.2), (119.0, 11.0), (118.7, 10.8)],
    [(117.0, 10.8), (117.2, 11.0), (117.4, 10.8), (117.2, 10.6)],
    [(116.0, 11.5), (116.2, 11.7), (116.5, 11.5), (116.2, 11.3)],
    [(115.4, 12.5), (115.7, 12.7), (115.9, 12.5), (115.6, 12.3)],
    # Natuna, Anambas, and Tioman/Bangka-Belitung in the southwest.
    [(108.0, 3.8), (108.8, 4.3), (109.5, 3.8), (109.0, 3.0),
     (108.2, 3.1)],
    [(106.5, 3.0), (106.8, 3.3), (107.1, 3.0), (106.8, 2.7)],
    [(104.0, 2.8), (105.0, 3.2), (105.4, 2.5), (105.0, 1.8),
     (104.2, 2.0)],
    [(103.0, 1.8), (103.5, 2.2), (104.0, 1.8), (103.7, 1.4)],
    [(104.8, 1.0), (105.4, 1.5), (106.2, 1.2), (106.0, 0.7),
     (105.2, 0.7)],
    # Andaman/Nicobar chain within the western frame.
    [(92.5, 12.0), (93.0, 13.0), (94.0, 13.2), (94.5, 12.2),
     (94.0, 11.0), (93.0, 11.0)],
    [(92.5, 10.0), (92.8, 10.5), (93.2, 10.3), (93.0, 9.8)],
    [(93.0, 8.5), (93.4, 9.0), (93.8, 8.7), (93.5, 8.2)],
    [(93.5, 7.5), (93.8, 7.8), (94.1, 7.5), (93.8, 7.2)],
]

def point_in_polygon(x, y, polygon):
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        crosses = (yi > y) != (yj > y)
        if crosses and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside

def is_land(lon, lat):
    return any(point_in_polygon(lon, lat, poly)
               for poly in POLYGONS + ISLANDS)

def main():
    lats = [LAT_MAX - STEP * i for i in range(111)]
    lons = [LON_MIN + STEP * i for i in range(141)]
    matrix = np.fromiter(
        (int(is_land(lon, lat)) for lat in lats for lon in lons),
        dtype=np.uint8,
        count=111 * 141,
    ).reshape((111, 141))
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)

    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    image = Image.fromarray(matrix * 255, mode="L").convert("1")
    image.save(out / "candidate.png")
    enlarged = image.resize((1128, 888), Image.Resampling.NEAREST)
    enlarged.save(out / "candidate_8x.png")

    assert matrix.shape == (111, 141)
    assert matrix.size == 15651 and set(np.unique(matrix)) <= {0, 1}
    with (out / "candidate.csv").open(newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 111 and all(len(row) == 141 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert image.size == (141, 111) and set(image.getdata()) <= {0, 255}
    assert enlarged.size == (1128, 888)
    assert np.array_equal(np.asarray(enlarged)[::8, ::8], np.asarray(image))
    print(f"validated matrix={matrix.shape}, values={matrix.size}, "
          f"images={image.size}/{enlarged.size}")

if __name__ == "__main__":
    main()
