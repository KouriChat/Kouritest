"""Knowledge-only approximation of the Canadian Arctic Archipelago.

All coordinate lists below are hand-authored from geographic knowledge, not
from maps, GIS files, online sources, or a land/water classification service.
Each vertex is written as west-longitude,north-latitude, in decimal degrees.
Classification is a union of these approximate island polygons, evaluated at
the requested POINTS. Lakes are not removed. Sea ice is not treated as land.
Requires Python 3, numpy, and Pillow; run from any current working directory.
"""
from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image

LAND = []


def island(name, vertices):
    polygon = []
    for token in vertices.split():
        west, north = map(float, token.split(","))
        polygon.append((-west, north))
    if len(polygon) < 3:
        raise ValueError(name)
    LAND.append((name, polygon))


def points_in_polygon(x, y, polygon):
    """Vectorized ray crossing, not independent language-model judgments."""
    inside = np.zeros(x.shape, dtype=bool)
    ax, ay = polygon[-1]
    for bx, by in polygon:
        if ay != by:
            crossing = (ay > y) != (by > y)
            intersection = ax + (y - ay) * (bx - ax) / (by - ay)
            inside ^= crossing & (x < intersection)
        ax, ay = bx, by
    return inside


island("Devon Island", """
96.91,76.18 96.75,76.40 96.40,76.58 96.01,76.72
95.78,76.94 95.15,77.08 94.42,77.06 93.84,76.97
93.38,76.83 92.90,76.88 92.35,76.70 91.87,76.67
91.45,76.57 91.07,76.70 90.58,76.68 90.16,76.55
90.45,76.39 89.85,76.33 89.17,76.39 88.48,76.29
87.88,76.35 87.29,76.19 86.64,76.28 86.10,76.30
85.63,76.35 85.11,76.40 84.65,76.35 84.15,76.47
83.64,76.42 83.11,76.30 82.62,76.35 82.03,76.23
81.50,76.14 81.03,76.03 80.72,75.87 80.26,75.75
80.02,75.50 79.79,75.36 79.95,75.12 80.27,74.98
80.33,74.75 80.74,74.59 81.25,74.50 81.65,74.55
82.12,74.50 82.42,74.56 82.66,74.68 82.90,74.67
83.06,74.87 83.14,75.11 83.04,75.29 83.26,75.39
83.48,75.23 83.51,75.03 83.70,75.13 83.87,75.34
84.10,75.30 84.02,75.05 83.90,74.84 83.87,74.65
84.18,74.51 84.66,74.45 85.08,74.47 85.43,74.56
85.94,74.52 86.40,74.57 86.84,74.46 87.34,74.45
87.86,74.50 88.30,74.55 88.54,74.70 88.45,74.92
88.71,75.03 89.08,75.05 89.22,74.84 89.22,74.64
89.63,74.61 90.03,74.64 90.48,74.60 90.84,74.54
91.18,74.60 91.53,74.65 91.69,74.78 91.91,74.78
92.05,74.96 92.22,75.12 92.21,75.32 92.53,75.49
92.67,75.66 93.10,75.77 93.72,75.82 94.04,75.96
94.60,75.98 95.11,76.02 95.56,76.07 96.00,76.09
96.40,76.05 96.72,76.09
""")

island("Cornwallis Island", """
95.91,75.59 95.60,75.71 95.02,75.79 94.67,75.70
94.31,75.75 93.94,75.64 93.46,75.61 93.26,75.44
93.43,75.28 93.53,75.04 93.75,74.92 94.16,74.86
94.41,74.76 94.72,74.69 94.96,74.64 95.31,74.72
95.65,74.87 95.78,75.01 96.11,75.10 96.45,75.18
96.56,75.29 96.18,75.37 95.99,75.46
""")
island("Little Cornwallis Island", """
97.20,75.46 97.05,75.61 96.78,75.68 96.39,75.65
96.16,75.58 96.28,75.45 96.53,75.36 96.93,75.35
""")
island("Baillie-Hamilton Island", """
95.18,75.86 94.94,75.99 94.45,76.02 94.05,75.93
94.11,75.79 94.45,75.75 94.95,75.76
""")
island("Beechey Island", """
92.02,74.73 91.88,74.76 91.78,74.72 91.89,74.68
""")
island("Griffith Island", """
95.86,74.55 95.64,74.65 95.24,74.59 95.10,74.49
95.42,74.43 95.73,74.45
""")
island("Lowther Island", """
97.68,74.55 97.48,74.64 97.13,74.63 96.94,74.54
97.12,74.48 97.49,74.47
""")

island("Bathurst Island", """
100.62,74.77 101.12,74.91 101.45,75.15 101.67,75.38
102.03,75.49 102.17,75.66 101.76,75.72 100.91,75.72
100.65,75.80 101.40,75.84 101.92,75.94 101.69,76.09
101.74,76.28 101.35,76.43 100.71,76.49 100.17,76.43
100.03,76.24 100.23,76.10 99.92,75.96 99.58,75.91
99.33,76.06 99.36,76.28 99.48,76.50 99.21,76.71
98.78,76.85 98.27,76.75 98.01,76.55 97.71,76.42
97.51,76.19 97.67,76.04 97.39,75.89 97.53,75.79
98.12,75.84 98.67,75.78 98.75,75.69 98.09,75.67
97.66,75.61 97.48,75.43 97.63,75.22 97.88,75.04
98.09,74.84 98.47,74.76 98.76,74.67 99.23,74.72
99.62,74.78 100.12,74.72
""")
island("Cameron Island", """
104.59,76.41 104.32,76.63 103.83,76.76 103.29,76.71
102.92,76.56 103.02,76.36 103.43,76.29 103.95,76.32
""")
island("Vanier Island", """
104.04,76.05 103.87,76.24 103.42,76.34 102.91,76.29
102.54,76.15 102.62,75.98 103.05,75.91 103.62,75.93
""")
island("Massey Island", """
103.61,75.84 103.26,75.92 102.82,75.88 102.62,75.78
102.88,75.68 103.33,75.70
""")
island("Alexander Island", """
102.68,75.99 102.36,76.06 102.06,76.01 101.98,75.88
102.18,75.76 102.45,75.78
""")
island("Marc Island", """
104.03,75.97 103.91,76.04 103.73,76.01 103.69,75.93
103.87,75.91
""")
island("Helena Island", """
101.57,76.69 101.21,76.80 100.73,76.77 100.38,76.64
100.67,76.57 101.14,76.58
""")
island("Seymour Island", """
101.40,76.80 101.25,76.84 101.08,76.82 101.19,76.78
""")
island("Young Island", """
98.14,74.65 97.99,74.72 97.80,74.68 97.86,74.58
98.03,74.58
""")
island("Garrett Island", """
97.82,74.78 97.67,74.82 97.50,74.75 97.66,74.70
""")
island("Hamilton Island", """
95.90,75.82 95.65,75.89 95.38,75.86 95.54,75.79
""")
island("Lougheed Island", """
105.91,77.75 105.40,77.70 105.03,77.60 104.77,77.45
104.72,77.28 104.94,77.11 105.15,77.00 105.63,77.06
105.83,77.30
""")
island("King Christian Island", """
102.12,77.77 101.91,77.88 101.49,77.91 101.10,77.85
100.78,77.73 100.94,77.60 101.38,77.56 101.84,77.64
""")
island("Ellef Ringnes Island", """
104.08,79.15 103.63,79.32 102.96,79.43 102.33,79.48
101.68,79.40 101.13,79.32 100.70,79.23 100.23,79.25
99.78,79.12 99.38,78.97 99.27,78.80 99.61,78.66
100.05,78.63 100.18,78.44 99.91,78.26 100.03,78.14
100.50,78.04 100.87,77.93 101.35,77.96 101.78,78.09
102.29,78.15 102.55,78.31 103.02,78.38 103.12,78.57
103.54,78.66 103.55,78.79 103.88,78.92 103.76,79.04
""")
island("Amund Ringnes Island", """
98.58,78.19 98.35,78.39 97.89,78.55 97.75,78.78
97.35,78.95 96.97,79.12 96.39,79.16 95.98,79.04
95.61,78.90 95.80,78.72 95.38,78.59 95.21,78.39
95.12,78.18 95.38,78.03 95.90,77.92 96.50,77.84
97.16,77.89 97.64,78.02 98.12,78.07
""")
island("Cornwall Island", """
96.24,77.65 95.92,77.80 95.40,77.90 94.79,77.92
94.22,77.80 93.61,77.77 93.15,77.64 93.10,77.51
93.62,77.43 94.28,77.40 94.78,77.47 95.47,77.43
95.91,77.51
""")
island("Graham Island", """
91.87,77.40 91.62,77.61 91.07,77.70 90.58,77.66
90.04,77.57 89.70,77.43 89.88,77.28 90.44,77.23
91.01,77.28 91.50,77.30
""")
island("Buckingham Island", """
91.48,77.14 91.25,77.25 90.94,77.23 90.74,77.12
90.91,77.04 91.22,77.04
""")
island("North Kent Island", """
90.35,76.69 90.13,76.84 89.71,76.87 89.27,76.77
88.90,76.61 89.10,76.43 89.45,76.35 89.86,76.46
""")

island("Axel Heiberg Island, southern part", """
95.0,80.2 94.8,79.50 94.35,79.20 93.83,79.02
93.42,78.91 92.85,78.95 91.95,79.08 91.73,79.01
92.22,78.90 92.96,78.80 93.17,78.73 92.82,78.59
92.62,78.44 92.29,78.33 91.83,78.22 91.12,78.16
90.73,78.33 90.73,78.56 90.32,78.83 90.03,78.95
90.12,78.65 89.83,78.45 89.31,78.56 88.94,78.71
88.60,78.85 88.15,79.02 87.72,79.27 87.35,79.62
86.5,80.2
""")
island("Ellesmere Island, southwestern part", """
88.25,79.5 86.00,80.5 70.0,80.5 70.0,75.8
78.32,76.11 79.32,76.16 80.15,76.29 81.06,76.38
81.67,76.45 82.29,76.44 82.84,76.41 83.00,76.52
82.98,76.75 83.15,76.86 83.36,76.75 83.40,76.49
83.88,76.44 84.25,76.48 84.39,76.65 84.55,76.83
84.77,76.82 84.73,76.57 85.20,76.62 85.60,76.54
86.02,76.65 86.31,76.80 86.58,76.64 87.08,76.71
87.42,76.90 87.42,77.13 87.16,77.37 87.28,77.59
87.53,77.37 87.73,77.14 88.10,76.92 88.47,76.81
88.79,76.96 89.13,77.12 89.38,77.32 89.61,77.51
89.34,77.68 88.89,77.78 88.67,77.94 88.24,78.08
87.73,78.02 87.20,78.12 86.62,78.24 86.09,78.24
85.56,78.36 84.92,78.39 84.36,78.50 83.68,78.56
83.23,78.53 82.72,78.59 82.23,78.72 82.66,78.80
83.27,78.73 83.80,78.77 84.28,78.66 84.84,78.72
85.29,78.67 85.78,78.54 86.28,78.62 86.80,78.59
87.30,78.63 87.79,78.53 88.36,78.49 88.61,78.61
88.58,78.73 88.20,78.86 87.92,78.94 87.68,79.03
87.38,79.08 86.83,79.16 86.24,79.29 86.46,79.36
87.13,79.29 87.72,79.34
""")
island("Hoved Island", """
88.12,76.97 87.96,77.07 87.73,77.08 87.60,76.98
87.80,76.91 88.00,76.91
""")
island("Stor Island in Baumann Fiord", """
86.77,78.42 86.50,78.47 86.14,78.42 86.28,78.35
86.60,78.35
""")
island("Somerset Island, northern part", """
95.67,72.4 95.54,73.02 95.78,73.36 95.61,73.67
95.31,73.90 95.08,74.08 94.62,74.15 94.20,74.10
93.76,74.16 93.18,74.05 92.66,74.02 92.15,74.03
91.64,73.96 91.16,73.94 90.72,73.98 90.39,73.86
90.27,73.75 90.60,73.64 90.89,73.51 91.03,73.32
91.42,73.16 91.71,73.03 91.89,72.86 92.7,72.3
""")
island("Prince Leopold Island", """
90.17,73.89 90.00,73.92 89.84,73.87 89.91,73.81
90.09,73.81
""")
island("Prince of Wales Island, northern part", """
103.2,72.4 102.00,73.04 101.79,73.30 101.45,73.48
101.14,73.74 100.96,73.95 100.53,74.04 99.97,73.98
99.41,73.91 99.11,73.78 98.70,73.71 98.30,73.61
98.04,73.49 97.84,73.29 97.76,73.04 97.51,72.87
97.21,72.78 96.5,72.4
""")
island("Russell Island", """
98.89,73.89 98.55,74.01 98.04,74.03 97.63,73.94
97.47,73.80 97.91,73.74 98.43,73.78
""")
island("Prescott Island", """
97.48,73.10 97.22,73.22 96.89,73.21 96.62,73.10
96.62,72.94 96.96,72.84 97.33,72.92
""")
island("Pandora Island", """
97.77,73.32 97.57,73.39 97.38,73.33 97.48,73.24
97.68,73.24
""")
island("Baffin Island, Brodeur Peninsula", """
90.2,72.3 89.29,73.02 89.09,73.31 88.85,73.56
88.45,73.74 88.04,73.84 87.60,73.92 87.18,73.86
86.78,73.78 86.51,73.60 86.35,73.40 85.97,73.29
85.72,73.11 85.30,73.03 85.20,72.87 84.7,72.3
""")
island("Baffin Island, Borden Peninsula", """
85.12,72.3 84.88,72.98 84.79,73.16 84.62,73.40
84.61,73.61 84.36,73.73 83.88,73.71 83.48,73.65
83.14,73.60 82.83,73.47 82.43,73.44 82.07,73.34
81.79,73.22 81.49,73.06 81.25,72.88 80.5,72.3
""")
island("Bylot Island, western extremity", """
80.0,74.0 80.75,73.82 81.03,73.68 81.18,73.52
81.12,73.30 80.96,73.18 80.0,72.9
""")


def main():
    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Integer tenths avoid an accumulated floating-point step error.
    latitudes = (790 - np.arange(61, dtype=np.int64)) / 10.0
    longitudes = (-1050 + np.arange(241, dtype=np.int64)) / 10.0
    x, y = np.meshgrid(longitudes, latitudes)
    land = np.zeros((61, 241), dtype=bool)
    for name, polygon in LAND:
        land |= points_in_polygon(x, y, polygon)
    matrix = land.astype(np.uint8)
    assert matrix.shape == (61, 241)
    assert matrix.size == 14701
    assert np.all((matrix == 0) | (matrix == 1))
    assert latitudes[0] == 79 and latitudes[-1] == 73
    assert longitudes[0] == -105 and longitudes[-1] == -81

    csv_path = output_dir / "candidate.csv"
    with csv_path.open("w", encoding="ascii", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(matrix.tolist())

    # No projection, smoothing, resampling of the matrix, or coast repair.
    pixels = matrix * np.uint8(255)
    original = Image.fromarray(pixels)
    assert original.mode == "L"
    assert original.size == (241, 61)
    original_path = output_dir / "candidate.png"
    original.save(original_path)
    nearest = getattr(Image, "Resampling", Image).NEAREST
    enlarged = original.resize((1928, 488), resample=nearest)
    enlarged_path = output_dir / "candidate_8x.png"
    enlarged.save(enlarged_path)

    # Reopen only this script's own outputs to verify their actual contents.
    with csv_path.open("r", encoding="ascii", newline="") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 61
    assert all(len(row) == 241 for row in rows)
    assert all(value in ("0", "1") for row in rows for value in row)
    assert sum(map(len, rows)) == 14701
    assert np.array_equal(np.asarray(rows, dtype=np.uint8), matrix)

    with Image.open(original_path) as check:
        assert check.size == (241, 61) and check.mode == "L"
        saved_original = np.asarray(check).copy()
    with Image.open(enlarged_path) as check:
        assert check.size == (1928, 488) and check.mode == "L"
        saved_enlarged = np.asarray(check).copy()
    assert set(np.unique(saved_original).tolist()) == {0, 255}
    assert set(np.unique(saved_enlarged).tolist()) == {0, 255}
    assert np.array_equal(saved_original, pixels)
    expected = np.repeat(np.repeat(saved_original, 8, axis=0), 8, axis=1)
    assert np.array_equal(saved_enlarged, expected)

    print(json.dumps({
        "validation": "passed",
        "matrix_shape": list(matrix.shape),
        "valid_values": int(matrix.size),
        "land_points": int(matrix.sum()),
        "water_points": int(matrix.size - int(matrix.sum())),
        "original_png": [241, 61],
        "enlarged_png": [1928, 488],
        "pixel_values": [0, 255],
        "nearest_neighbor_8x_verified": True,
        "source": "hand-authored knowledge-only approximate island polygons",
        "classification": "point-in-polygon union",
        "files": [str(csv_path), str(original_path), str(enlarged_path)],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
