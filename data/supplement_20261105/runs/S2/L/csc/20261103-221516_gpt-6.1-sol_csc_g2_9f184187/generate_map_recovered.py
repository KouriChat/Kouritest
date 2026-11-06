from pathlib import Path

import numpy as np
from PIL import Image

# Approximate coastline and islands as hand-written polygons in (lon, lat).
# Coordinates are intentionally approximate; no external geographic data is used.
LAND_POLYGONS = [
    # Mainland Asia: southern China, Indochina, Malay Peninsula and nearby coast.
    [
        (99.0, 23.0), (99.0, 22.0), (99.5, 21.5), (100.0, 21.0),
        (100.5, 20.5), (101.0, 20.0), (101.5, 19.5), (102.0, 19.0),
        (102.5, 18.5), (103.0, 18.0), (103.5, 17.5), (104.0, 17.0),
        (104.5, 16.5), (105.0, 16.0), (105.5, 15.5), (106.0, 15.0),
        (106.0, 14.5), (105.5, 14.0), (105.0, 13.5), (104.5, 13.0),
        (104.0, 12.5), (103.5, 12.0), (103.0, 11.5), (102.5, 11.0),
        (102.0, 10.5), (101.5, 10.0), (101.0, 9.5), (100.5, 9.0),
        (100.0, 8.5), (99.5, 8.0), (99.0, 7.5), (99.0, 23.0),
    ],
    # Vietnam's eastern coastline and southern delta, joined to mainland.
    [
        (102.0, 23.0), (107.0, 23.0), (109.0, 21.5), (108.5, 20.0),
        (108.0, 18.5), (108.5, 17.0), (109.5, 15.5), (109.5, 14.0),
        (109.0, 12.5), (108.5, 11.0), (108.0, 10.0), (107.0, 9.0),
        (106.0, 8.5), (105.0, 9.0), (104.0, 10.0), (103.0, 11.0),
        (102.0, 12.0), (101.0, 13.0), (100.0, 14.0), (100.0, 16.0),
        (101.0, 18.0), (102.0, 20.0), (102.0, 23.0),
    ],
    # Hainan.
    [
        (108.55, 20.15), (109.1, 20.25), (109.8, 20.05), (110.5, 19.65),
        (110.8, 19.2), (110.3, 18.5), (109.7, 18.15), (109.1, 18.2),
        (108.7, 18.65), (108.55, 19.3), (108.55, 20.15),
    ],
    # Taiwan.
    [
        (120.0, 25.3), (121.0, 25.2), (121.7, 24.8), (121.9, 24.0),
        (121.5, 23.2), (121.2, 22.4), (120.8, 21.9), (120.2, 22.1),
        (120.0, 23.0), (120.0, 24.0), (120.0, 25.3),
    ],
    # Luzon and the northern Philippine islands.
    [
        (119.7, 18.6), (120.7, 18.4), (121.5, 17.8), (122.1, 17.0),
        (122.0, 16.0), (121.7, 15.1), (121.2, 14.3), (121.0, 13.5),
        (120.5, 13.0), (120.0, 13.5), (120.0, 14.5), (119.5, 15.5),
        (119.3, 16.7), (119.7, 18.6),
    ],
    # Mindoro.
    [
        (120.3, 13.5), (121.0, 13.4), (121.2, 12.8), (121.0, 12.1),
        (120.5, 12.0), (120.1, 12.5), (120.3, 13.5),
    ],
    # Samar and Leyte.
    [
        (124.5, 12.5), (125.2, 12.4), (125.5, 11.8), (125.3, 11.1),
        (124.8, 10.9), (124.5, 11.5), (124.5, 12.5),
    ],
    # Negros, Cebu and Panay.
    [
        (122.0, 11.9), (123.0, 11.9), (123.4, 11.4), (123.0, 10.8),
        (122.4, 10.6), (122.0, 11.0), (122.0, 11.9),
    ],
    # Mindanao.
    [
        (124.0, 9.8), (125.0, 9.7), (126.0, 8.8), (126.2, 7.8),
        (125.8, 6.8), (125.0, 6.0), (124.0, 5.5), (123.0, 5.7),
        (122.2, 6.2), (122.0, 7.2), (122.5, 8.2), (123.0, 9.0),
        (124.0, 9.8),
    ],
    # Palawan.
    [
        (117.0, 11.5), (118.0, 11.2), (118.5, 10.5), (118.0, 9.5),
        (117.5, 8.5), (117.0, 7.5), (116.8, 8.8), (116.7, 10.0),
        (117.0, 11.5),
    ],
    # Borneo, with approximate coastline.
    [
        (109.5, 7.0), (110.5, 7.2), (111.5, 6.8), (112.5, 6.5),
        (113.5, 5.5), (114.5, 4.5), (115.5, 4.0), (116.5, 4.0),
        (117.5, 4.5), (118.0, 5.5), (117.5, 6.5), (117.0, 7.5),
        (116.0, 8.0), (115.0, 7.8), (114.0, 7.5), (113.0, 7.0),
        (112.0, 6.8), (111.0, 6.5), (110.0, 6.2), (109.5, 7.0),
    ],
    # Sumatra, clipped naturally by the sampled frame at its southern end.
    [
        (105.0, 6.0), (105.5, 5.5), (105.5, 4.5), (105.0, 3.5),
        (104.5, 2.5), (104.0, 1.5), (103.5, 0.5), (102.5, -1.0),
        (101.5, -3.0), (100.5, -5.0), (99.5, -5.5), (99.0, -4.0),
        (99.0, 6.0), (105.0, 6.0),
    ],
    # Java, mostly south of the sampled frame; its northern coast grazes the frame.
    [
        (105.0, -5.8), (106.0, -6.0), (107.0, -6.2), (108.0, -6.5),
        (109.0, -6.8), (110.0, -7.0), (111.0, -7.2), (112.0, -7.5),
        (113.0, -7.8), (114.0, -8.0), (114.0, -9.0), (105.0, -9.0),
    ],
    # Malay Peninsula, connected northward to mainland.
    [
        (99.0, 7.5), (100.0, 7.0), (101.0, 6.5), (102.0, 6.0),
        (103.0, 5.5), (103.5, 4.5), (103.5, 3.5), (104.0, 2.5),
        (104.3, 1.5), (104.0, 1.0), (103.0, 1.0), (102.0, 2.0),
        (101.0, 3.0), (100.0, 4.0), (99.0, 5.0), (99.0, 7.5),
    ],
    # Andaman and Nicobar island chain (coarse island-sized polygons).
    [
        (92.0, 14.0), (94.0, 14.0), (94.0, 13.0), (92.0, 13.0),
    ],
    [
        (92.0, 12.0), (94.0, 12.0), (94.0, 11.0), (92.0, 11.0),
    ],
    [
        (92.0, 10.0), (94.0, 10.0), (94.0, 9.0), (92.0, 9.0),
    ],
    # Spratly region: only selected larger islands and reef/cay groups are represented.
    [(114.0, 10.5), (114.5, 10.5), (114.5, 10.0), (114.0, 10.0)],
    [(115.0, 11.0), (115.4, 11.0), (115.4, 10.6), (115.0, 10.6)],
    [(116.5, 10.0), (117.0, 10.0), (117.0, 9.5), (116.5, 9.5)],
    [(112.5, 9.5), (113.0, 9.5), (113.0, 9.0), (112.5, 9.0)],
    # Paracel Islands (coarse group).
    [(111.0, 17.0), (112.0, 17.0), (112.0, 16.0), (111.0, 16.0)],
]


def point_in_polygon(lon, lat, polygon):
    """Ray-casting point test; boundary points are assigned to land."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # Explicitly include points on polygon edges.
        cross = (lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
        if abs(cross) < 1e-10:
            if (min(x1, x2) - 1e-10 <= lon <= max(x1, x2) + 1e-10
                    and min(y1, y2) - 1e-10 <= lat <= max(y1, y2) + 1e-10):
                return True

        if (y1 > lat) != (y2 > lat):
            x_cross = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < x_cross:
                inside = not inside
    return inside


def is_land(lon, lat):
    return int(any(point_in_polygon(lon, lat, polygon) for polygon in LAND_POLYGONS))


def main():
    latitudes = 23.0 - 0.2 * np.arange(111)
    longitudes = 99.0 + 0.2 * np.arange(141)

    matrix = np.zeros((111, 141), dtype=np.uint8)
    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            matrix[row, col] = is_land(float(lon), float(lat))

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", encoding="ascii", newline="") as f:
        for row in matrix:
            f.write(",".join(str(int(value)) for value in row) + "\n")

    # Mode "1" gives a strict two-color image: 1 is white, 0 is black.
    image = Image.fromarray(matrix * 255, mode="L").convert("1")
    image.save(output_dir / "candidate.png")

    enlarged = image.resize((141 * 8, 111 * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")

    # Validate the generated artifacts.
    assert matrix.shape == (111, 141)
    assert int(matrix.size) == 15651
    assert set(np.unique(matrix).tolist()) <= {0, 1}

    with csv_path.open("r", encoding="ascii") as f:
        rows = [line.rstrip("\n").split(",") for line in f]
    assert len(rows) == 111
    assert all(len(row) == 141 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    check_original = Image.open(output_dir / "candidate.png")
    check_scaled = Image.open(output_dir / "candidate_8x.png")
    assert check_original.size == (141, 111)
    assert check_scaled.size == (1128, 888)
    assert set(check_original.convert("L").getdata()) <= {0, 255}
    assert set(check_scaled.convert("L").getdata()) <= {0, 255}

    expected_scaled = check_original.resize(
        (1128, 888), resample=Image.Resampling.NEAREST
    )
    assert list(check_scaled.getdata()) == list(expected_scaled.getdata())


if __name__ == "__main__":
    main()
