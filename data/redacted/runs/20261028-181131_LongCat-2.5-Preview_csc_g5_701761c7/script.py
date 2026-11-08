# -*- coding: utf-8 -*-
"""Reconstruct a land/water map of the South China Sea and surroundings
purely from hand-coded geographic knowledge (approximate coastlines).
No network, no external map data. Standard library + numpy + Pillow only.
"""
import os
import numpy as np

LAT_TOP = 23.0
LAT_BOT = 1.0
LON_LEFT = 99.0
LON_RIGHT = 127.0
STEP = 0.2
NLAT = 111  # rows, north -> south
NLON = 141  # cols, west -> east

os.makedirs("outputs", exist_ok=True)


def lat_of_row(i):
    return LAT_TOP - i * STEP


def lon_of_col(j):
    return LON_LEFT + j * STEP


def point_in_poly(lon, lat, poly):
    """Ray-casting point-in-polygon. poly = [(lon,lat), ...]."""
    inside = False
    n = len(poly)
    for k in range(n):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n]
        if (y1 > lat) != (y2 > lat):
            xin = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < xin:
                inside = not inside
    return inside


def in_circle(lon, lat, clon, clat, rlon, rlat=None):
    if rlat is None:
        rlat = rlon
    return ((lon - clon) / rlon) ** 2 + ((lat - clat) / rlat) ** 2 <= 1.0
# ---------------------------------------------------------------------------
# Approximate coastlines (hand-coded). Polygons are closed implicitly.
# ---------------------------------------------------------------------------

# Southern China coastline (Guangxi -> Guangdong -> Fujian) + inland top edge.
CHINA_COAST = [
    (108.0, 22.0), (108.6, 21.9), (109.1, 21.4), (109.6, 21.5),
    (110.0, 21.2), (110.5, 21.6), (111.0, 21.9), (111.6, 22.3),
    (112.2, 22.6), (113.0, 22.6), (113.6, 22.5), (114.0, 22.3),
    (114.5, 22.0), (115.0, 22.4), (115.5, 22.7), (116.2, 22.9),
    (116.7, 23.3), (117.2, 23.6), (117.8, 24.0), (118.5, 24.5),
    (119.0, 25.0), (119.6, 25.5), (120.2, 26.2), (121.0, 27.0),
    (121.9, 28.0), (127.5, 29.0), (127.5, 23.0), (108.0, 23.0),
]

# Vietnam (mainland) coastline: Gulf of Tonkin -> south tip.
VIETNAM = [
    (108.0, 21.5), (106.7, 21.0), (105.8, 21.2), (105.8, 20.0),
    (106.5, 19.5), (107.3, 18.7), (108.3, 17.5), (109.2, 15.8),
    (109.4, 14.0), (109.3, 12.5), (108.5, 11.0), (107.0, 10.4),
    (106.0, 9.5), (105.0, 9.0), (104.5, 8.6), (103.9, 10.0),
    (104.0, 11.0), (105.0, 11.5), (106.0, 11.0), (107.5, 11.5),
    (108.5, 12.5), (109.3, 13.5), (109.4, 15.0), (109.0, 16.5),
    (108.2, 18.0), (107.2, 19.3), (106.3, 20.5), (105.9, 21.5),
    (108.0, 21.5),
]

# Malay Peninsula west coast (Thailand/Malaysia) down to Singapore.
MALAY = [
    (100.0, 13.6), (99.5, 12.5), (99.3, 11.5), (99.8, 10.5),
    (100.5, 9.5), (101.3, 8.5), (102.0, 7.0), (103.0, 6.0),
    (103.5, 5.0), (103.8, 4.0), (103.6, 3.0), (103.4, 2.2),
    (103.6, 1.5), (104.0, 1.3), (103.9, 2.5), (103.5, 4.0),
    (103.2, 5.0), (102.5, 6.0), (101.8, 7.2), (101.2, 8.5),
    (100.5, 9.5), (100.0, 10.8), (99.6, 12.0), (99.8, 13.5),
    (100.0, 13.6),
]

# Hainan Island (ellipse).
HAINAN = (109.8, 19.2, 1.3, 1.1)
# Taiwan (ellipse).
TAIWAN = (121.0, 23.7, 0.45, 1.1)

# Luzon (northern Philippines).
LUZON = [
    (120.0, 18.6), (120.5, 18.4), (121.0, 18.2), (121.6, 18.0),
    (122.0, 17.6), (122.2, 17.0), (122.0, 16.0), (121.8, 15.0),
    (121.6, 14.2), (121.2, 13.8), (120.8, 14.0), (120.4, 14.6),
    (120.6, 15.4), (120.4, 16.2), (120.0, 16.8), (119.8, 17.5),
    (119.9, 18.2), (120.0, 18.6),
]

# Mindoro.
MINDORO = (120.8, 12.6, 0.7, 0.55)

# Palawan (long narrow island).
PALAWAN = [
    (117.3, 12.0), (118.0, 12.3), (118.8, 11.8), (119.5, 11.3),
    (120.2, 10.8), (121.0, 10.0), (121.8, 9.3), (122.2, 8.7),
    (121.8, 8.3), (121.0, 9.0), (120.2, 9.8), (119.5, 10.5),
    (118.8, 11.0), (118.0, 11.5), (117.3, 11.8), (117.3, 12.0),
]

# Visayas group (Panay, Negros, Cebu, Bohol, Leyte, Samar).
PANAY = (122.3, 11.2, 0.6, 0.5)
NEGROS = (122.9, 10.0, 0.45, 0.9)
CEBU = (123.6, 10.5, 0.3, 0.9)
BOHOL = (124.0, 10.0, 0.55, 0.45)
LEYTE = (124.7, 10.8, 0.45, 1.1)
SAMAR = (125.0, 11.6, 0.6, 0.6)

# Mindanao (southern Philippines).
MINDANAO = [
    (122.0, 8.0), (122.5, 8.4), (123.2, 8.8), (124.0, 9.2),
    (124.8, 9.6), (125.5, 9.9), (126.3, 9.8), (126.6, 9.0),
    (126.5, 8.0), (126.3, 7.0), (126.0, 6.2), (125.5, 6.0),
    (125.0, 6.3), (124.5, 6.8), (124.0, 7.3), (123.5, 7.5),
    (123.0, 7.8), (122.5, 7.8), (122.0, 8.0),
]
# Borneo (northern part visible in frame).
BORNEO = [
    (109.5, 2.0), (110.3, 2.8), (111.5, 3.3), (112.8, 4.0),
    (114.0, 4.7), (115.5, 5.5), (116.5, 6.3), (117.5, 6.8),
    (118.1, 6.5), (118.7, 5.6), (119.1, 5.0), (119.3, 4.2),
    (119.2, 3.4), (118.8, 2.9), (118.0, 2.4), (117.0, 1.9),
    (116.0, 1.5), (115.0, 1.2), (114.0, 1.0), (112.0, 1.0),
    (111.0, 1.2), (110.2, 1.5), (109.5, 2.0),
]

# Sumatra (northern part visible in frame).
SUMATRA = [
    (99.5, 1.0), (104.8, 5.6), (105.2, 4.8), (104.0, 3.8),
    (102.8, 2.8), (101.5, 1.8), (100.5, 1.0), (99.5, 1.0),
]

# Small islands: (lon, lat, rlon, rlat)
SMALL_ISLANDS = [
    (116.7, 20.7, 0.15, 0.15),   # Pratas (Dongsha)
    (112.0, 16.5, 0.35, 0.30),   # Paracel group (Xisha)
    (114.4, 10.4, 0.15, 0.15),   # Taiping It. (Spratlys)
    (112.3, 9.5, 0.12, 0.12),    # Spratly group
    (115.8, 10.0, 0.12, 0.12),   # Spratly group
    (114.2, 8.0, 0.10, 0.10),    # Spratly group
    (117.75, 15.15, 0.10, 0.10), # Scarborough Shoal
    (108.2, 4.0, 0.30, 0.30),    # Natuna Is.
    (106.0, 3.0, 0.20, 0.20),    # Anambas Is.
    (124.6, 1.4, 0.25, 0.20),    # N. Sulawesi tip / Togian
    (119.0, 5.5, 0.15, 0.15),    # Tawau area small is.
    (109.8, 11.0, 0.10, 0.10),   # Con Dao area
    (107.0, 10.5, 0.10, 0.10),   # Phu Quoc
    (103.5, 1.0, 0.15, 0.12),    # Karimata-area small is.
]
ELLIPSES = [HAINAN, TAIWAN, MINDORO, PANAY, NEGROS, CEBU, BOHOL,
            LEYTE, SAMAR]
POLYS = [CHINA_COAST, VIETNAM, MALAY, LUZON, PALAWAN, MINDANAO,
         BORNEO, SUMATRA]


def is_land(lon, lat):
    """Point criterion: land(1)/water(0), from hand-coded rules only."""
    for poly in POLYS:
        if point_in_poly(lon, lat, poly):
            return True
    for (clon, clat, rlon, rlat) in ELLIPSES:
        if in_circle(lon, lat, clon, clat, rlon, rlat):
            return True
    for (clon, clat, rlon, rlat) in SMALL_ISLANDS:
        if in_circle(lon, lat, clon, clat, rlon, rlat):
            return True
    return False


def main():
    mat = np.zeros((NLAT, NLON), dtype=np.int8)
    for i in range(NLAT):
        lat = lat_of_row(i)
        for j in range(NLON):
            lon = lon_of_col(j)
            mat[i, j] = 1 if is_land(lon, lat) else 0

    with open("outputs/candidate.csv", "w", encoding="utf-8") as f:
        for i in range(NLAT):
            f.write(",".join(str(int(v)) for v in mat[i]) + "\n")

    img = np.where(mat == 1, 255, 0).astype(np.uint8)
    from PIL import Image
    im = Image.fromarray(img, mode="L")
    im.save("outputs/candidate.png")
    im8 = im.resize((NLON * 8, NLAT * 8), resample=Image.NEAREST)
    im8.save("outputs/candidate_8x.png")

    print("matrix:", mat.shape, "valid:", mat.size,
          "land cells:", int(mat.sum()))


if __name__ == "__main__":
    main()
