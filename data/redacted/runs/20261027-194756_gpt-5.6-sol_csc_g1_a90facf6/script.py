"""Knowledge-based, deliberately approximate South China Sea raster."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def inside_polygon(lon, lat, vertices):
    """Ray-casting point test; vertices are (longitude, latitude)."""
    hit = False
    j = len(vertices) - 1
    for i, (xi, yi) in enumerate(vertices):
        xj, yj = vertices[j]
        if (yi > lat) != (yj > lat):
            cross = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < cross:
                hit = not hit
        j = i
    return hit


def inside_ellipse(lon, lat, cx, cy, rx, ry):
    return ((lon - cx) / rx) ** 2 + ((lat - cy) / ry) ** 2 <= 1.0


# Broad continental masses, clipped naturally by the requested frame.
LAND_POLYGONS = [
    # Southern China and mainland Southeast Asia, following the coastline
    # approximately from the Red River delta toward the Malay peninsula.
    [(99, 23), (127, 23), (127, 18.2), (124, 21.7), (121, 24),
     (119, 25.5), (117.5, 24.5), (116, 23), (114.5, 22.6),
     (113.5, 22.2), (112, 21.8), (110.5, 21.2), (109.5, 20.2),
     (108.4, 19.5), (107.8, 18.2), (107.2, 16.5), (106.7, 15),
     (107.5, 13.5), (108.7, 12), (109.2, 10.8), (108.5, 9.5),
     (107.2, 8.8), (106, 8.5), (105, 9.2), (104, 10.5),
     (103, 11.5), (102, 12.5), (101, 13.5), (100, 15),
     (99.3, 16.5), (99, 18), (99, 20), (99, 23)],
    # Malay peninsula, including its southern narrowing.
    [(100.1, 7.2), (100.5, 9.5), (101.3, 11.5), (102, 13),
     (102.8, 13.5), (103.5, 12.5), (103.3, 10.5), (103.6, 8.5),
     (104.3, 6.5), (103.8, 4.5), (103.4, 2.5), (102.6, 1),
     (101.4, 1), (100.8, 2.5), (100.4, 4.5), (100.1, 7.2)],
    # Taiwan.
    [(120, 21.8), (120.7, 21.8), (121.5, 23), (121.6, 24.8),
     (121, 25.3), (120.3, 24.5), (120, 23), (120, 21.8)],
    # Hainan.
    [(108.5, 18.1), (109.8, 18.1), (110.7, 19.1), (110.4, 20.1),
     (109.2, 20.2), (108.5, 19.3), (108.5, 18.1)],
    # Borneo.
    [(109.5, 7), (109.8, 4), (110.8, 1), (114, 0.8), (117.5, 1.5),
     (119.2, 3.5), (118.5, 5.2), (117, 6.5), (115, 7.5),
     (112.5, 7.8), (110.5, 7.5), (109.5, 7)],
]


# Elliptical approximations for larger islands and island clusters.  Small
# radii are intentional: most reef and islet pixels remain water.
ISLANDS = [
    # Sumatra, Java's northwestern end, and nearby islands.
    (99.2, 1.3, 0.8, 0.5), (99.5, 2.5, 0.85, 1.0),
    (99.1, 4.2, 0.6, 0.9), (100.2, 5.0, 0.45, 0.8),
    (105.8, 1.1, 0.6, 0.25),
    # Natuna, Anambas, and southern South China Sea islands.
    (108.2, 3.9, 0.25, 0.22), (106.0, 3.0, 0.22, 0.2),
    (109.2, 4.8, 0.2, 0.2), (112.5, 7.2, 0.18, 0.14),
    # Spratly and Paracel archipelagos, represented by island/reef clusters.
    (111.7, 10.3, 0.2, 0.16), (112.3, 10.8, 0.14, 0.12),
    (113.0, 9.8, 0.13, 0.12), (114.2, 10.5, 0.12, 0.1),
    (115.0, 11.8, 0.1, 0.1), (114.3, 12.0, 0.1, 0.08),
    (112.0, 16.5, 0.35, 0.24), (111.7, 16.9, 0.18, 0.13),
    # Philippines: Palawan, Luzon, Mindoro, Visayas, Mindanao and Sulu.
    (117.0, 7.5, 0.32, 1.25), (118.0, 9.5, 0.2, 0.35),
    (121.0, 18.0, 0.85, 1.45), (122.2, 13.2, 0.6, 0.85),
    (121.0, 12.5, 0.4, 0.55), (123.5, 11.0, 0.8, 0.7),
    (124.0, 8.0, 1.0, 1.0), (122.5, 7.0, 0.5, 0.45),
    (120.5, 6.0, 0.45, 0.35), (119.5, 5.5, 0.3, 0.3),
    (125.5, 9.8, 0.25, 0.35), (126.0, 7.5, 0.25, 0.6),
    (124.0, 13.5, 0.35, 0.55), (123.5, 16.0, 0.4, 0.8),
    # Taiwan's offshore islands, Ryukyu chain, and southern Japan.
    (119.5, 23.6, 0.12, 0.1), (119.7, 26.0, 0.18, 0.15),
    (121.5, 24.5, 0.2, 0.2), (122.9, 24.5, 0.18, 0.15),
    (123.0, 24.8, 0.13, 0.12), (124.2, 24.4, 0.28, 0.18),
    (125.3, 24.8, 0.2, 0.16), (127.0, 26.5, 0.35, 0.65),
    # Northern Sulawesi and the western edge of the Moluccas.
    (123.0, 1.5, 0.45, 0.35), (125.0, 1.2, 0.22, 0.25),
]


def is_land(lon, lat):
    if any(inside_polygon(lon, lat, poly) for poly in LAND_POLYGONS):
        return True
    return any(inside_ellipse(lon, lat, *island) for island in ISLANDS)


def main():
    latitudes = [23.0 - 0.2 * i for i in range(111)]
    longitudes = [99.0 + 0.2 * j for j in range(141)]
    matrix = np.array(
        [[int(is_land(lon, lat)) for lon in longitudes] for lat in latitudes],
        dtype=np.uint8,
    )
    with (OUT / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

    image = Image.fromarray(matrix * 255, mode="L")
    image.save(OUT / "candidate.png")
    image.resize((1128, 888), resample=Image.Resampling.NEAREST).save(
        OUT / "candidate_8x.png"
    )

    with (OUT / "candidate.csv").open(newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 111 and all(len(row) == 141 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert matrix.size == 15651 and set(np.unique(matrix)) <= {0, 1}
    assert image.size == (141, 111)
    assert set(image.getdata()) <= {0, 255}
    enlarged = Image.open(OUT / "candidate_8x.png")
    assert enlarged.size == (1128, 888)
    assert set(enlarged.getdata()) <= {0, 255}
    print("Validated 111x141 matrix, CSV, binary PNG, and 8x nearest-neighbor PNG.")


if __name__ == "__main__":
    main()
