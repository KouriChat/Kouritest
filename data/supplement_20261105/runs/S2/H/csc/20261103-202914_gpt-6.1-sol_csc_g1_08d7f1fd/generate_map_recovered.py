from pathlib import Path
import numpy as np
from PIL import Image


# Sampling grid:
# latitude: 23.0, 22.8, ..., 1.0
# longitude: 99.0, 99.2, ..., 127.0
N_LAT = 111
N_LON = 141
LAT_MAX = 23.0
LON_MIN = 99.0
STEP = 0.2


# Hand-written approximate coastline polygons.
# Polygon coordinates are (longitude, latitude).
POLYGONS = [
    # Mainland Southeast Asia and southern China.
    [
        (99.0, 23.0),
        (117.5, 23.0),
        (117.3, 22.5),
        (116.8, 22.0),
        (116.3, 21.7),
        (115.8, 21.4),
        (115.2, 21.1),
        (114.4, 20.9),
        (113.5, 21.0),
        (112.8, 21.2),
        (112.1, 21.1),
        (111.4, 21.2),
        (110.7, 21.6),
        (109.8, 21.8),
        (108.7, 21.5),
        (108.5, 20.8),
        (108.8, 20.0),
        (108.7, 19.0),
        (108.5, 18.0),
        (108.2, 17.0),
        (108.0, 16.0),
        (107.8, 15.0),
        (107.5, 14.0),
        (107.1, 13.0),
        (106.8, 12.0),
        (106.2, 11.0),
        (105.6, 10.4),
        (104.8, 9.7),
        (104.0, 9.1),

        # Cambodia and the northern Gulf of Thailand.
        (103.5, 9.3),
        (103.0, 9.9),
        (102.5, 10.5),
        (101.8, 11.2),
        (101.0, 12.0),
        (100.2, 12.8),

        # Eastern coast of Thailand and Malay Peninsula.
        (99.8, 12.4),
        (100.0, 12.0),
        (100.1, 11.0),
        (100.2, 10.0),
        (100.3, 8.0),
        (100.8, 7.0),
        (101.5, 6.0),
        (102.0, 5.0),
        (102.7, 4.0),
        (103.4, 3.0),
        (104.3, 0.7),

        # Southern and western coast of the Malay Peninsula.
        (102.6, 0.7),
        (101.8, 1.5),
        (101.0, 2.5),
        (100.5, 3.7),
        (100.2, 5.0),
        (99.5, 6.3),
        (99.0, 6.6),

        # Western edge of Thailand/Myanmar inside the frame.
        (99.0, 23.0),
    ],

    # Northern Sumatra.
    [
        (99.0, 5.2),
        (99.5, 5.6),
        (100.4, 5.3),
        (101.3, 4.8),
        (102.2, 4.1),
        (103.0, 3.1),
        (103.7, 2.0),
        (104.3, 0.7),
        (99.0, 0.7),
    ],

    # Borneo.
    [
        (108.0, 0.8),
        (115.5, 0.8),
        (117.0, 1.5),
        (118.2, 2.5),
        (119.0, 4.0),
        (119.2, 5.2),
        (118.7, 6.2),
        (117.5, 7.2),
        (116.3, 7.5),
        (115.0, 7.3),
        (113.8, 7.0),
        (112.5, 6.6),
        (111.3, 6.8),
        (110.0, 6.5),
        (109.2, 5.8),
        (109.0, 4.5),
        (108.5, 3.0),
    ],

    # Hainan.
    [
        (108.6, 20.15),
        (109.5, 20.25),
        (110.4, 20.05),
        (110.9, 19.5),
        (110.5, 18.7),
        (109.8, 18.15),
        (109.1, 18.15),
        (108.6, 18.8),
        (108.5, 19.5),
    ],

    # Taiwan, only its southern part is inside the frame.
    [
        (120.0, 21.9),
        (120.8, 21.8),
        (121.5, 22.1),
        (121.9, 22.8),
        (121.7, 23.0),
        (120.0, 23.0),
    ],

    # Luzon.
    [
        (119.5, 18.5),
        (120.0, 19.0),
        (121.0, 18.8),
        (122.0, 18.5),
        (122.5, 17.5),
        (122.3, 16.0),
        (121.8, 15.0),
        (121.2, 14.0),
        (120.5, 13.0),
        (119.7, 13.0),
        (119.2, 14.2),
        (119.1, 15.5),
        (119.4, 17.0),
    ],

    # Palawan.
    [
        (117.1, 11.8),
        (118.0, 11.7),
        (119.0, 11.3),
        (119.7, 10.5),
        (119.4, 9.5),
        (118.9, 8.8),
        (118.1, 8.5),
        (117.5, 9.2),
        (117.0, 10.2),
        (116.8, 11.0),
    ],

    # Mindoro.
    [
        (120.0, 13.6),
        (120.8, 13.7),
        (121.5, 13.2),
        (121.6, 12.4),
        (121.0, 11.9),
        (120.2, 12.0),
        (119.7, 12.7),
    ],

    # Panay.
    [
        (121.5, 11.8),
        (122.5, 11.9),
        (123.2, 11.3),
        (123.2, 10.3),
        (122.6, 10.0),
        (121.8, 10.2),
        (121.4, 10.8),
    ],

    # Negros.
    [
        (122.5, 10.1),
        (123.2, 10.5),
        (123.4, 9.5),
        (123.0, 9.0),
        (122.5, 9.3),
    ],

    # Cebu.
    [
        (123.3, 11.3),
        (123.8, 11.0),
        (123.9, 10.0),
        (123.5, 9.6),
        (123.2, 10.2),
    ],

    # Samar.
    [
        (124.1, 12.5),
        (125.0, 12.7),
        (125.6, 11.8),
        (125.1, 11.0),
        (124.5, 10.8),
        (124.0, 11.5),
    ],

    # Leyte.
    [
        (124.4, 11.5),
        (125.0, 11.4),
        (125.2, 10.5),
        (124.8, 9.9),
        (124.3, 10.4),
    ],

    # Bohol.
    [
        (123.5, 10.1),
        (124.3, 10.2),
        (124.5, 9.8),
        (124.0, 9.5),
        (123.5, 9.7),
    ],

    # Mindanao.
    [
        (122.0, 10.0),
        (122.7, 10.4),
        (123.4, 10.1),
        (124.0, 9.5),
        (124.7, 9.5),
        (125.3, 8.8),
        (126.0, 8.3),
        (126.3, 7.2),
        (125.8, 6.2),
        (125.0, 5.5),
        (124.0, 5.0),
        (123.1, 5.1),
        (122.5, 5.7),
        (122.0, 6.5),
        (121.7, 7.5),
        (121.5, 8.5),
    ],
]


# Small islands, reefs, cays, and island groups.
# Each tuple is (longitude, latitude, longitude_radius, latitude_radius).
ELLIPSES = [
    # Gulf of Thailand, Vietnam, Malaysia, and Singapore.
    (103.85, 1.30, 0.13, 0.08),  # Singapore
    (104.10, 1.10, 0.16, 0.10),  # Batam
    (104.45, 1.10, 0.20, 0.12),  # Bintan
    (103.45, 1.10, 0.12, 0.10),  # Karimun
    (106.15, 3.00, 0.18, 0.22),  # Anambas
    (106.65, 2.80, 0.13, 0.16),  # Anambas group
    (108.20, 3.90, 0.35, 0.35),  # Natuna Besar
    (107.95, 4.20, 0.12, 0.14),
    (103.95, 10.25, 0.18, 0.18),  # Phu Quoc
    (106.60, 8.70, 0.10, 0.12),   # Con Dao
    (100.00, 9.55, 0.12, 0.12),    # Samui
    (99.95, 9.75, 0.08, 0.08),     # Phangan
    (102.35, 12.10, 0.12, 0.12),   # Koh Chang
    (99.80, 6.35, 0.14, 0.10),     # Langkawi
    (100.25, 5.40, 0.10, 0.12),    # Penang

    # Offshore Borneo islands.
    (115.20, 5.30, 0.12, 0.14),    # Labuan
    (117.10, 7.30, 0.20, 0.22),    # Banggi
    (116.80, 7.55, 0.10, 0.12),
    (118.00, 6.10, 0.08, 0.10),

    # Philippine islands not represented by the larger polygons.
    (121.95, 20.50, 0.13, 0.10),   # Batanes group
    (121.85, 20.30, 0.08, 0.07),
    (122.00, 12.30, 0.16, 0.18),   # Masbate
    (120.00, 12.20, 0.08, 0.08),
    (124.15, 11.80, 0.08, 0.12),
    (124.55, 11.80, 0.08, 0.12),
    (124.30, 9.65, 0.08, 0.08),
    (123.10, 9.20, 0.08, 0.10),
    (120.95, 6.50, 0.20, 0.24),    # Basilan
    (120.95, 5.95, 0.18, 0.18),    # Jolo
    (119.90, 5.20, 0.12, 0.15),    # Tawi-Tawi

    # Paracel Islands.
    (112.33, 16.83, 0.06, 0.06),
    (112.02, 16.45, 0.05, 0.05),
    (111.60, 16.50, 0.05, 0.05),
    (111.20, 15.80, 0.05, 0.05),
    (112.60, 16.40, 0.05, 0.05),
    (112.75, 16.55, 0.04, 0.04),

    # Spratly Islands and exposed cays.
    (114.37, 10.38, 0.05, 0.05),   # Taiping
    (114.30, 11.05, 0.06, 0.06),   # Thitu
    (113.83, 11.10, 0.04, 0.04),   # West York
    (115.00, 10.75, 0.04, 0.04),
    (114.42, 10.72, 0.04, 0.04),
    (114.37, 10.18, 0.04, 0.04),
    (114.47, 11.00, 0.04, 0.04),
    (114.23, 8.64, 0.04, 0.04),    # Spratly Island
    (114.50, 9.88, 0.04, 0.04),
    (114.13, 9.70, 0.04, 0.04),
    (115.50, 7.90, 0.04, 0.04),
    (115.80, 7.40, 0.04, 0.04),
    (117.70, 8.40, 0.04, 0.04),

    # Other isolated South China Sea cays.
    (109.50, 11.00, 0.04, 0.04),
    (110.30, 12.00, 0.04, 0.04),
    (108.70, 8.80, 0.04, 0.04),
]


def point_in_polygon(x, y, polygon):
    """Return True when (x, y) is inside or on the boundary."""
    inside = False
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # Boundary test.
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) <= 1e-10:
            if (
                min(x1, x2) - 1e-10 <= x <= max(x1, x2) + 1e-10
                and min(y1, y2) - 1e-10 <= y <= max(y1, y2) + 1e-10
            ):
                return True

        # Ray crossing test.
        crosses = (y1 > y) != (y2 > y)
        if crosses:
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside

    return inside


def point_in_ellipse(x, y, ellipse):
    cx, cy, rx, ry = ellipse
    dx = (x - cx) / rx
    dy = (y - cy) / ry
    return dx * dx + dy * dy <= 1.0


def is_land(longitude, latitude):
    for polygon in POLYGONS:
        if point_in_polygon(longitude, latitude, polygon):
            return True

    for ellipse in ELLIPSES:
        if point_in_ellipse(longitude, latitude, ellipse):
            return True

    return False


def build_matrix():
    matrix = np.zeros((N_LAT, N_LON), dtype=np.uint8)

    for row in range(N_LAT):
        latitude = LAT_MAX - row * STEP
        for col in range(N_LON):
            longitude = LON_MIN + col * STEP
            matrix[row, col] = 1 if is_land(longitude, latitude) else 0

    return matrix


def write_outputs(matrix):
    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "candidate.csv"
    raw_png_path = output_dir / "candidate.png"
    enlarged_png_path = output_dir / "candidate_8x.png"

    # CSV: no header, exactly 111 rows and 141 comma-separated 0/1 values.
    with csv_path.open("w", encoding="ascii", newline="\n") as f:
        for row in matrix:
            f.write(",".join(str(int(value)) for value in row))
            f.write("\n")

    # White land, black water.
    raw_pixels = (matrix * 255).astype(np.uint8)
    raw_image = Image.fromarray(raw_pixels, mode="L")
    raw_image.save(raw_png_path)

    # Nearest-neighbor enlargement by exactly 8.
    enlarged_image = raw_image.resize(
        (N_LON * 8, N_LAT * 8),
        resample=Image.NEAREST,
    )
    enlarged_image.save(enlarged_png_path)

    return csv_path, raw_png_path, enlarged_png_path


def validate_outputs(matrix, paths):
    csv_path, raw_png_path, enlarged_png_path = paths

    assert matrix.shape == (111, 141)
    assert matrix.size == 15651
    assert set(np.unique(matrix).tolist()) <= {0, 1}

    csv_lines = csv_path.read_text(encoding="ascii").splitlines()
    assert len(csv_lines) == 111

    for line in csv_lines:
        values = line.split(",")
        assert len(values) == 141
        assert all(value in {"0", "1"} for value in values)

    raw_image = Image.open(raw_png_path).convert("L")
    enlarged_image = Image.open(enlarged_png_path).convert("L")

    raw_array = np.asarray(raw_image)
    enlarged_array = np.asarray(enlarged_image)

    assert raw_image.size == (141, 111)
    assert enlarged_image.size == (1128, 888)
    assert set(np.unique(raw_array).tolist()) <= {0, 255}
    assert set(np.unique(enlarged_array).tolist()) <= {0, 255}

    # Every 8x8 block must equal its source pixel.
    expected_blocks = raw_array[:, None, :, None]
    actual_blocks = enlarged_array.reshape(111, 8, 141, 8)
    assert np.array_equal(
        actual_blocks,
        np.broadcast_to(expected_blocks, actual_blocks.shape),
    )


def main():
    matrix = build_matrix()
    paths = write_outputs(matrix)
    validate_outputs(matrix, paths)

    print("matrix:", matrix.shape, "valid values:", int(matrix.size))
    print("csv:", paths[0])
    print("raw png:", paths[1], Image.open(paths[1]).size)
    print("8x png:", paths[2], Image.open(paths[2]).size)


if __name__ == "__main__":
    main()
