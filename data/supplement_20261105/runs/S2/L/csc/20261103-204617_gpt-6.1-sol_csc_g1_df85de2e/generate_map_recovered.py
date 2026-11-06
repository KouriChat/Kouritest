from pathlib import Path

import numpy as np
from PIL import Image

# Sample centers: 23N to 1N, 99E to 127E, at 0.2-degree intervals.
LAT_MAX = 23.0
LAT_MIN = 1.0
LON_MIN = 99.0
LON_MAX = 127.0
STEP = 0.2
HEIGHT = 111
WIDTH = 141

# Approximate land outlines, hand-authored from geographic knowledge.
# Vertices are (longitude, latitude), ordered around each polygon.
LAND_POLYGONS = [
    # Mainland Southeast Asia: Myanmar, Thailand, Cambodia, Vietnam, Laos.
    [
        (99.0, 23.0), (101.0, 23.0), (101.2, 22.4), (101.8, 21.8),
        (102.2, 21.0), (102.8, 20.4), (103.3, 19.7), (103.8, 18.9),
        (104.2, 18.0), (104.7, 17.3), (105.2, 16.6), (105.6, 15.8),
        (105.8, 15.0), (106.2, 14.4), (106.4, 13.8), (106.7, 13.1),
        (107.0, 12.4), (107.4, 11.8), (107.8, 11.1), (108.2, 10.5),
        (108.5, 9.8), (108.6, 9.0), (108.5, 8.2), (108.3, 7.4),
        (108.0, 6.7), (107.4, 6.2), (106.8, 5.8), (106.0, 5.5),
        (105.2, 5.3), (104.6, 5.5), (104.1, 6.0), (103.7, 6.8),
        (103.4, 7.5), (103.1, 8.3), (102.8, 9.1), (102.5, 9.8),
        (102.1, 10.5), (101.8, 11.2), (101.4, 11.9), (101.0, 12.6),
        (100.6, 13.4), (100.2, 14.2), (99.8, 15.0), (99.4, 15.8),
        (99.0, 16.5), (99.0, 23.0),
    ],
    # Malay Peninsula and southern Thailand.
    [
        (99.2, 7.0), (100.0, 7.0), (100.8, 6.8), (101.5, 6.5),
        (102.2, 6.2), (102.8, 5.8), (103.2, 5.1), (103.4, 4.4),
        (103.5, 3.7), (103.6, 3.0), (103.8, 2.4), (104.1, 1.8),
        (104.5, 1.2), (104.8, 1.0), (102.8, 1.0), (102.5, 1.6),
        (102.2, 2.2), (101.8, 2.9), (101.4, 3.6), (101.0, 4.3),
        (100.6, 5.0), (100.1, 5.7), (99.6, 6.3), (99.2, 7.0),
    ],
    # Hainan.
    [
        (108.6, 20.2), (109.1, 20.3), (109.8, 20.1), (110.3, 19.8),
        (110.7, 19.3), (110.6, 18.7), (110.2, 18.3), (109.6, 18.1),
        (109.0, 18.2), (108.6, 18.7), (108.5, 19.3), (108.6, 20.2),
    ],
    # Taiwan.
    [
        (120.0, 25.3), (121.0, 25.0), (121.7, 24.4), (121.9, 23.8),
        (121.5, 23.0), (121.1, 22.3), (120.7, 21.9), (120.3, 22.2),
        (120.0, 22.8), (119.9, 23.5), (120.0, 25.3),
    ],
    # Luzon and nearby northern Philippine islands.
    [
        (119.8, 18.5), (120.7, 18.4), (121.3, 17.8), (121.7, 17.0),
        (122.0, 16.1), (121.9, 15.2), (121.5, 14.4), (121.0, 13.8),
        (120.5, 13.4), (120.0, 13.5), (119.7, 14.2), (119.6, 15.0),
        (119.5, 16.0), (119.6, 17.0), (119.8, 18.5),
    ],
    # Mindoro.
    [
        (120.2, 13.5), (121.0, 13.4), (121.2, 13.0), (121.0, 12.5),
        (120.6, 12.2), (120.2, 12.5), (120.0, 13.0), (120.2, 13.5),
    ],
    # Palawan.
    [
        (117.0, 11.5), (117.8, 11.3), (118.0, 10.8), (117.7, 10.0),
        (117.5, 9.1), (117.2, 8.2), (117.0, 7.4), (116.8, 7.5),
        (116.9, 8.5), (116.8, 9.5), (116.7, 10.5), (117.0, 11.5),
    ],
    # Samar.
    [
        (124.0, 12.5), (125.0, 12.5), (125.4, 12.0), (125.3, 11.4),
        (124.9, 10.9), (124.4, 11.0), (124.0, 11.5), (124.0, 12.5),
    ],
    # Leyte.
    [
        (124.2, 11.4), (124.8, 11.3), (125.0, 10.8), (124.8, 10.1),
        (124.4, 9.8), (124.1, 10.3), (124.0, 10.9), (124.2, 11.4),
    ],
    # Panay.
    [
        (121.7, 11.9), (122.6, 11.9), (123.0, 11.4), (122.8, 10.7),
        (122.3, 10.4), (121.8, 10.7), (121.5, 11.2), (121.7, 11.9),
    ],
    # Negros and Cebu.
    [
        (122.7, 10.3), (123.3, 10.2), (123.5, 9.7), (123.2, 9.1),
        (122.8, 9.2), (122.6, 9.7), (122.7, 10.3),
    ],
    [
        (123.5, 11.3), (124.0, 11.2), (124.1, 10.5), (123.9, 9.8),
        (123.6, 9.7), (123.4, 10.3), (123.5, 11.3),
    ],
    # Mindanao.
    [
        (123.5, 9.8), (124.2, 9.6), (125.0, 9.0), (125.7, 8.5),
        (126.5, 7.8), (126.4, 7.0), (125.8, 6.2), (125.1, 5.6),
        (124.3, 5.5), (123.6, 6.0), (123.2, 6.8), (123.1, 7.7),
        (123.2, 8.7), (123.5, 9.8),
    ],
    # Bangka and Belitung.
    [
        (105.0, 3.5), (106.0, 3.6), (106.3, 3.2), (106.0, 2.7),
        (105.3, 2.6), (104.9, 3.0), (105.0, 3.5),
    ],
    [
        (107.4, 3.3), (108.0, 3.4), (108.3, 3.0), (108.0, 2.6),
        (107.5, 2.7), (107.3, 3.0), (107.4, 3.3),
    ],
    # Northern Borneo (Sabah and Brunei).
    [
        (109.5, 7.2), (110.1, 7.5), (110.8, 7.8), (111.4, 7.5),
        (112.0, 7.2), (112.7, 7.0), (113.3, 6.7), (114.0, 6.5),
        (114.5, 6.0), (115.0, 5.5), (115.1, 5.0), (114.5, 4.8),
        (113.8, 4.7), (113.0, 4.8), (112.3, 4.9), (111.5, 5.2),
        (110.8, 5.8), (110.2, 6.4), (109.5, 7.2),
    ],
    # Natuna Besar.
    [
        (108.0, 4.2), (108.5, 4.3), (108.8, 3.9), (108.7, 3.4),
        (108.2, 3.3), (107.9, 3.7), (108.0, 4.2),
    ],
    # Anambas and Lingga groups (simplified island points become small polygons).
    [
        (105.6, 3.2), (105.9, 3.3), (106.0, 3.0), (105.8, 2.9),
        (105.6, 3.2),
    ],
    # Spratly Islands: selected reefs/islets as tiny land areas.
    [
        (111.8, 11.3), (112.0, 11.3), (112.1, 11.1), (111.9, 11.0),
        (111.8, 11.3),
    ],
    [
        (114.3, 10.4), (114.5, 10.4), (114.6, 10.2), (114.4, 10.1),
        (114.3, 10.4),
    ],
    [
        (115.7, 9.9), (115.9, 9.9), (116.0, 9.7), (115.8, 9.6),
        (115.7, 9.9),
    ],
    [
        (117.1, 8.8), (117.3, 8.8), (117.4, 8.6), (117.2, 8.5),
        (117.1, 8.8),
    ],
    [
        (112.8, 7.8), (113.0, 7.8), (113.1, 7.6), (112.9, 7.5),
        (112.8, 7.8),
    ],
    # Paracel Islands.
    [
        (111.3, 16.6), (111.6, 16.6), (111.7, 16.4), (111.5, 16.3),
        (111.3, 16.6),
    ],
    [
        (112.0, 16.4), (112.2, 16.4), (112.3, 16.2), (112.1, 16.1),
        (112.0, 16.4),
    ],
    # Small islands in the Gulf of Thailand.
    [
        (102.5, 9.8), (102.7, 9.8), (102.8, 9.6), (102.6, 9.5),
        (102.5, 9.8),
    ],
]

# More reliable broad land masses for eastern Myanmar/Thailand/Vietnam,
# selected as additional hand-drawn polygons.
LAND_POLYGONS.extend([
    # Southern China within the northern edge of the sampling frame.
    [
        (99.0, 23.0), (109.5, 23.0), (109.0, 22.5), (108.3, 22.1),
        (107.4, 21.8), (106.8, 21.4), (106.5, 20.8), (106.3, 20.1),
        (105.8, 19.5), (105.2, 18.9), (104.8, 18.2), (104.4, 17.6),
        (103.8, 17.2), (103.1, 17.0), (102.4, 17.2), (101.7, 17.6),
        (101.0, 18.2), (100.4, 18.9), (99.8, 19.6), (99.3, 20.5),
        (99.0, 21.2), (99.0, 23.0),
    ],
    # Vietnam's eastern coastal strip.
    [
        (105.5, 23.0), (109.5, 23.0), (109.7, 22.4), (109.5, 21.8),
        (109.3, 21.0), (109.4, 20.3), (109.7, 19.6), (109.5, 18.8),
        (109.4, 18.0), (109.2, 17.2), (109.0, 16.4), (108.8, 15.6),
        (108.6, 14.8), (108.4, 14.0), (108.2, 13.2), (108.1, 12.4),
        (108.0, 11.6), (107.8, 10.8), (107.6, 10.0), (107.4, 9.2),
        (107.2, 8.4), (106.8, 7.8), (106.2, 7.5), (105.8, 8.0),
        (105.7, 9.0), (105.8, 10.0), (105.9, 11.0), (106.0, 12.0),
        (106.1, 13.0), (106.2, 14.0), (106.0, 15.0), (105.8, 16.0),
        (105.7, 17.0), (105.6, 18.0), (105.5, 19.0), (105.5, 20.0),
        (105.5, 21.0), (105.5, 23.0),
    ],
])


def point_in_polygon(lon, lat, polygon):
    """Even-odd point-in-polygon test; boundary points count as land."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        cross = (lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
        if abs(cross) < 1e-10:
            if (min(x1, x2) - 1e-10 <= lon <= max(x1, x2) + 1e-10
                    and min(y1, y2) - 1e-10 <= lat <= max(y1, y2) + 1e-10):
                return True

        if (y1 > lat) != (y2 > lat):
            crossing_lon = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < crossing_lon:
                inside = not inside
    return inside


def build_matrix():
    matrix = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for row in range(HEIGHT):
        lat = LAT_MAX - row * STEP
        for col in range(WIDTH):
            lon = LON_MIN + col * STEP
            if any(point_in_polygon(lon, lat, poly) for poly in LAND_POLYGONS):
                matrix[row, col] = 1
    return matrix


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = build_matrix()
    assert matrix.shape == (111, 141)
    assert matrix.size == 15651
    assert np.isin(matrix, (0, 1)).all()

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", encoding="ascii", newline="") as f:
        for row in matrix:
            f.write(",".join(str(int(value)) for value in row) + "\n")

    pixels = np.where(matrix == 1, 255, 0).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    image.save(output_dir / "candidate.png")

    enlarged = image.resize(
        (WIDTH * 8, HEIGHT * 8),
        resample=Image.Resampling.NEAREST,
    )
    enlarged.save(output_dir / "candidate_8x.png")

    # Verify dimensions, binary CSV contents, and monochrome image pixels.
    lines = csv_path.read_text(encoding="ascii").splitlines()
    assert len(lines) == HEIGHT
    assert all(len(line.split(",")) == WIDTH for line in lines)
    assert all(set(line.split(",")) <= {"0", "1"} for line in lines)
    assert image.size == (141, 111)
    assert set(image.getdata()) <= {0, 255}
    assert enlarged.size == (1128, 888)
    assert set(enlarged.getdata()) <= {0, 255}
    print(f"Wrote {csv_path}, {output_dir / 'candidate.png'}, "
          f"{output_dir / 'candidate_8x.png'}")


if __name__ == "__main__":
    main()
