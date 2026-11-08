from pathlib import Path

import numpy as np
from PIL import Image


# Approximate hand-drawn land outlines in longitude/latitude.
LAND_POLYGONS = [
    # Mainland Southeast Asia and southern China
    [
        (96.0, 25.0), (103.0, 25.0), (110.0, 25.0), (114.0, 24.4),
        (116.0, 24.0), (117.5, 23.5), (119.0, 25.0), (122.0, 26.0),
        (124.0, 27.0), (128.0, 27.0), (128.0, 18.0), (121.0, 18.0),
        (119.0, 16.0), (117.5, 13.5), (115.5, 11.0), (113.5, 9.0),
        (111.0, 7.0), (108.0, 5.5), (105.0, 5.0), (102.5, 6.0),
        (100.0, 8.0), (98.0, 10.0), (97.0, 14.0), (96.0, 18.0),
    ],
    # Malay Peninsula
    [
        (100.0, 7.0), (101.8, 7.0), (102.2, 6.0), (102.3, 5.0),
        (103.0, 4.0), (103.5, 2.5), (104.0, 1.0), (103.0, 0.5),
        (101.5, 1.0), (100.5, 2.5), (100.0, 4.0), (99.5, 5.5),
    ],
    # Sumatra
    [
        (95.0, 5.8), (97.0, 5.5), (99.0, 4.2), (101.0, 2.5),
        (103.0, 0.5), (105.0, -2.0), (106.0, -5.5), (104.0, -6.5),
        (101.0, -4.0), (99.0, -1.0), (97.0, 1.0), (95.0, 3.0),
    ],
    # Borneo
    [
        (108.0, 7.5), (110.0, 7.0), (112.0, 7.3), (114.5, 6.5),
        (117.5, 5.0), (119.0, 2.0), (118.0, 0.0), (115.0, -1.5),
        (112.0, -1.0), (110.0, 0.0), (109.0, 2.5), (108.0, 5.0),
    ],
    # Northern edge of Java
    [
        (105.0, -5.5), (108.0, -5.5), (111.0, -6.0), (114.0, -6.5),
        (114.0, -8.0), (110.0, -8.5), (107.0, -8.0), (105.0, -7.0),
    ],
    # Taiwan and Hainan
    [(120.0, 25.5), (121.8, 25.3), (122.0, 24.0), (121.5, 22.0),
     (120.8, 21.8), (120.0, 23.0)],
    [(108.5, 20.2), (110.8, 20.1), (111.0, 19.0), (110.0, 18.2),
     (108.7, 18.4), (108.3, 19.3)],
    # Palawan and Philippine islands
    [(117.0, 11.5), (118.0, 11.8), (119.0, 10.0), (118.5, 8.0),
     (117.5, 7.0), (116.8, 8.5)],
    [(119.5, 18.5), (121.5, 18.5), (122.5, 17.0), (123.0, 15.0),
     (122.5, 13.0), (121.0, 12.5), (120.0, 14.0), (119.5, 16.0)],
    [(120.2, 13.5), (121.2, 13.4), (121.3, 12.2), (120.5, 12.0)],
    [(121.5, 11.8), (122.8, 11.6), (123.0, 10.5), (122.0, 10.2)],
    [(124.0, 12.0), (125.5, 11.8), (125.5, 10.0), (124.2, 10.0)],
    [(124.5, 11.0), (125.5, 10.8), (125.3, 9.5), (124.5, 9.6)],
    [(123.5, 10.0), (124.2, 10.0), (124.0, 9.0), (123.5, 9.0)],
    [(122.5, 10.0), (123.5, 9.8), (123.0, 8.5), (122.2, 8.5)],
    [(123.0, 9.0), (126.0, 9.0), (126.5, 7.0), (125.0, 6.0),
     (123.0, 6.5), (122.5, 8.0)],
    # Natuna, Anambas, and selected Spratly exposed features
    [(107.0, 4.0), (108.0, 4.0), (108.0, 3.5), (107.0, 3.5)],
    [(105.5, 3.5), (106.3, 3.5), (106.3, 3.0), (105.5, 3.0)],
    [(111.8, 11.9), (112.2, 11.9), (112.2, 11.6), (111.8, 11.6)],
    [(113.0, 12.0), (113.4, 12.0), (113.4, 11.7), (113.0, 11.7)],
    [(114.2, 11.5), (114.6, 11.5), (114.6, 11.2), (114.2, 11.2)],
    [(115.0, 10.5), (115.4, 10.5), (115.4, 10.2), (115.0, 10.2)],
    [(116.0, 10.0), (116.4, 10.0), (116.4, 9.7), (116.0, 9.7)],
    [(117.0, 8.8), (117.4, 8.8), (117.4, 8.5), (117.0, 8.5)],
    [(112.5, 8.0), (112.9, 8.0), (112.9, 7.7), (112.5, 7.7)],
    [(114.0, 8.0), (114.4, 8.0), (114.4, 7.7), (114.0, 7.7)],
]


def point_in_polygon(lon, lat, polygon):
    """Ray crossing test; points exactly on a polygon edge count as land."""
    inside = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        cross = (lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
        if (abs(cross) < 1e-10 and min(x1, x2) <= lon <= max(x1, x2)
                and min(y1, y2) <= lat <= max(y1, y2)):
            return True
        if (y1 > lat) != (y2 > lat):
            crossing_lon = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < crossing_lon:
                inside = not inside
    return inside


def build_matrix():
    matrix = np.zeros((111, 141), dtype=np.uint8)
    for row in range(111):
        lat = 23.0 - 0.2 * row
        for col in range(141):
            lon = 99.0 + 0.2 * col
            matrix[row, col] = int(
                any(point_in_polygon(lon, lat, poly) for poly in LAND_POLYGONS)
            )
    return matrix


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()
    with (output_dir / "candidate.csv").open(
        "w", encoding="ascii", newline=""
    ) as stream:
        for row in matrix:
            stream.write(",".join(str(int(value)) for value in row) + "\n")

    image = Image.fromarray(np.where(matrix == 1, 255, 0).astype(np.uint8), mode="L")
    image.save(output_dir / "candidate.png")
    image.resize((1128, 888), resample=Image.Resampling.NEAREST).save(
        output_dir / "candidate_8x.png"
    )

    assert matrix.shape == (111, 141)
    assert int(matrix.size) == 15651
    assert set(np.unique(matrix)).issubset({0, 1})


if __name__ == "__main__":
    main()
