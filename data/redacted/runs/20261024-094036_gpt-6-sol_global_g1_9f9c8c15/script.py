"""A land/water map made entirely from manually recalled approximate outlines.
Longitudes and latitudes below are handwritten vertices, not imported map data.
Water polygons are removed after continental polygons; islands are then restored.
"""
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image

LON, LAT = np.meshgrid(np.arange(-179, 180, 2), np.arange(89, -90, -2))
land_shapes = []
water_shapes = []
island_shapes = []


def put(collection, vertices):
    numbers = [float(s) for s in re.findall(r"-?\d+(?:\.\d+)?", vertices)]
    if len(numbers) < 6 or len(numbers) % 2:
        raise ValueError("Expected at least three longitude/latitude pairs")
    collection.append(list(zip(numbers[::2], numbers[1::2])))


def continent(vertices):
    put(land_shapes, vertices)


def sea(vertices):
    put(water_shapes, vertices)


def island(vertices):
    put(island_shapes, vertices)


# Alaska, Canada, the contiguous US, mainland Mexico and Central America.
continent("""
-168 65 -166 67 -163 68 -160 70 -155 71 -149 70 -143 70
-139 69 -135 69 -132 70 -126 70 -122 69 -116 69 -111 68
-107 68 -104 69 -100 70 -98 71 -96 72 -93 72 -91 69
-88 68 -85 68 -83 67 -81 66 -81 64 -79 63 -77 62
-74 60 -70 59 -65 60 -63 59 -61 57 -60 55 -57 53
-56 52 -58 51 -61 50 -63 49 -65 49 -67 48 -69 48
-71 47 -69 46 -67 45 -68 44 -71 43 -72 42 -74 41
-75 39 -76 37 -78 35 -80 34 -81 33 -81 30 -80 28
-80 25 -81 24 -82 25 -82 27 -83 29 -85 30 -88 30
-90 29 -92 29 -95 29 -97 26 -97 24 -96 22 -94 20
-92 19 -91 18 -90 19 -90 21 -87 21 -86 19 -87 17
-88 16 -89 16 -89 15 -87 15 -86 16 -84 15 -83 11
-82 9 -80 9 -79 9 -77 8 -77 7 -79 7 -81 8
-83 8 -84 10 -86 11 -87 12 -89 13 -91 14 -93 15
-95 16 -97 16 -99 17 -102 18 -104 19 -105 21 -107 23
-109 24 -110 26 -112 28 -114 30 -117 31 -117 33
-119 34 -120 37 -122 38 -123 40 -124 42 -124 46
-123 48 -124 49 -126 50 -127 52 -130 54 -133 56
-136 58 -139 59 -142 60 -145 60 -148 61 -151 60
-152 59 -155 58 -158 57 -160 58 -162 59 -164 60
-166 61 -165 63 -168 65
""")

# Hudson Bay and James Bay: water, even where the surrounding mainland is land.
sea("""
-95 60 -94 63 -91 65 -87 65 -84 64 -82 62 -80 60
-79 58 -79 56 -79 54 -81 52 -83 52 -85 54 -88 54
-90 55 -92 56 -94 57 -95 60
""")
sea("-85 64 -83 65 -81 64 -79 63 -78 62 -81 61 -83 62")

# Baja California and Newfoundland are separate from mainland at this scale.
island("""
-117 32 -116 31 -114 29 -112 27 -110 25 -109 23
-110 22 -112 24 -113 26 -115 28 -117 30
""")
island("-60 47 -57 47 -55 48 -53 47 -53 46 -55 46 -56 45 -59 46")
island("-65 45 -62 46 -60 46 -62 45 -64 44 -66 44")
island("-126 50 -124 49 -123 48 -124 48 -127 49 -128 50")

# South America; the northern tip meets Panama by a narrow isthmus.
continent("""
-77 8 -75 10 -72 11 -70 12 -67 11 -64 11 -62 10
-60 9 -58 8 -55 7 -52 5 -50 3 -50 1 -48 0
-47 -1 -44 -2 -41 -3 -38 -4 -36 -5 -35 -7
-35 -10 -37 -12 -39 -14 -39 -17 -40 -19
-41 -22 -43 -23 -45 -24 -47 -26 -49 -28
-50 -30 -52 -32 -55 -34 -58 -36 -60 -38
-63 -40 -65 -42 -66 -45 -67 -47 -68 -49
-69 -52 -68 -54 -67 -55 -69 -55 -71 -54
-72 -52 -74 -50 -74 -47 -73 -45 -73 -42
-73 -39 -72 -36 -72 -33 -71 -30 -70 -27
-70 -23 -70 -20 -72 -17 -74 -14 -76 -11
-77 -9 -79 -7 -81 -5 -81 -3 -79 -1
-79 1 -77 3 -77 8
""")
# Mainland Europe and Asia, including Arabia, India, Indochina and Kamchatka.
# The northern edge follows the rough Arctic coast rather than sea-ice extent.
continent("""
-9 36 -7 36 -5 36 -4 37 -1 37 1 39 3 41 5 43 7 44
8 43 8 42 9 41 11 40 13 38 16 38 18 39
17 41 15 42 13 44 14 45 17 45 19 43
20 42 19 40 20 39 22 38 24 38 25 40 26 41
27 40 29 40 30 38 28 37 26 36 29 36 32 36
35 36 36 34 35 33 35 31 34 30 34 28 35 26
36 23 38 20 40 18 42 16 43 13 44 12
46 13 48 14 50 15 52 17 54 17 56 20
57 23 56 25 55 25 53 24 51 24 50 26
49 27 48 29 50 30 52 28 54 27 56 26
58 25 59 23 60 22 61 24 63 25 65 25
67 24 69 24 70 22 72 21 73 19 73 16
74 14 75 11 77 8 79 8 80 10 80 13
81 16 82 18 84 19 86 20 88 21 90 22
91 22 92 20 94 18 94 16 96 16 97 15
98 13 98 10 99 8 100 7 100 5 101 3
103 1 104 2 104 4 103 6 102 8 100 10
99 12 100 14 101 14 102 12 104 10
105 9 107 10 109 12 109 15 107 16
106 18 107 20 109 21 110 20 111 21
113 22 115 23 117 24 119 25 120 27
121 29 121 31 120 32 120 34 122 36
120 37 119 39 121 40 122 39 124 39
125 38 126 35 127 34 128 36 129 39
130 42 132 43 135 44 137 45 139 48
141 50 141 53 140 54 141 55 143 56
145 58 149 59 152 59 156 60 159 60
161 61 163 63 163 61 162 59 161 56
160 54 158 53 158 55 159 57 161 60
164 61 167 61 169 62 172 63 176 64 180 65
180 69 176 69 172 70 168 70 164 69
160 70 155 70 151 71 146 72 141 72
136 71 131 71 126 73 120 74 115 73
110 73 105 73 101 76 96 76 91 75 87 74
83 72 80 73 76 72 73 71 69 69 65 68
61 69 57 70 54 72 51 72 48 71 45 68
42 67 40 67 37 68 35 69 32 70 30 70
29 69 27 70 26 69 25 68 23 69
21 70 19 70 17 69 15 69 13 68 12 66
10 64 8 62 6 60 5 58 7 58 9 58
10 57 9 55 8 54 7 53 6 53 4 52
3 51 2 51 2 50 0 50 -2 49 -5 48
-5 47 -3 47 -2 46 -1 44 -1 43
-4 43 -7 44 -9 43 -9 40 -9 36
""")

# A second polygon carries the east end of Siberia across the date line.
continent("-180 65 -176 65 -173 66 -170 66 -169 67 -172 69 -176 70 -180 69")

# Africa is bordered by the Atlantic, Mediterranean, Red Sea and Indian Ocean.
continent("""
-17 21 -17 24 -16 27 -14 29 -12 31 -10 33
-8 34 -6 35 -3 35 0 36 3 37 7 37
10 37 11 35 10 33 12 32 15 33
18 32 21 32 24 33 27 32 30 31
32 31 33 30 33 28 34 26 35 24
36 22 38 19 40 16 42 13 43 12
44 11 48 12 51 11 51 9 50 8
49 7 50 5 48 3 46 2 44 1
42 -2 42 -4 41 -6 40 -8 39 -11
37 -14 36 -17 35 -20 34 -23 33 -26
32 -29 31 -31 29 -33 27 -34 25 -33
23 -31 21 -28 19 -24 18 -21 16 -17
14 -12 13 -9 12 -6 11 -3 9 0
9 3 10 5 9 6 7 5 5 5
2 5 0 5 -3 5 -6 5 -9 5
-11 6 -13 8 -14 10 -15 11 -16 13
-17 15 -16 17 -16 19 -17 21
""")

# Greenland: rock and land-based ice are both land.
island("""
-53 60 -49 60 -46 59 -43 60 -42 62 -41 64
-39 65 -39 67 -36 69 -33 69 -29 71
-25 73 -21 76 -19 78 -21 80 -26 82
-33 83 -41 83 -47 82 -51 82 -55 80
-58 78 -61 76 -61 73 -57 72 -52 70
-50 68 -51 66 -53 64 -53 60
""")
# Australia.  The indentation in the north is the Gulf of Carpentaria.
continent("""
113 -22 114 -19 118 -18 121 -17 123 -17
125 -15 128 -15 130 -12 133 -12
136 -12 137 -14 136 -16 137 -17
139 -16 140 -13 141 -11 142 -12
142 -15 143 -17 145 -18 146 -20
148 -21 150 -23 151 -25 153 -27
153 -30 151 -33 150 -36 148 -38
145 -39 142 -39 140 -38 138 -35
136 -35 134 -33 132 -32 129 -31
127 -32 124 -33 121 -34 118 -34
116 -33 115 -31 114 -29 113 -26 113 -22
""")

# Antarctica's ice-covered ground counts as land.  The Ross and Weddell
# embayments and the floating ice shelves there do not count as land.
continent("""
-180 -78 -173 -77 -165 -78 -155 -78 -145 -77
-137 -75 -128 -74 -119 -74 -111 -73 -102 -73
-94 -73 -86 -72 -79 -72 -75 -73
-72 -71 -69 -69 -68 -67 -66 -65 -63 -63
-61 -62 -59 -63 -59 -65 -61 -68
-60 -70 -57 -72 -54 -73 -48 -75
-40 -76 -30 -76 -23 -75 -15 -73
-8 -71 0 -70 10 -70 20 -70
30 -69 40 -68 50 -67 60 -67
70 -68 80 -67 90 -66 100 -66
110 -65 120 -66 130 -66 140 -66
150 -68 160 -69 170 -72 177 -75
180 -77 180 -90 -180 -90
""")

# Arctic Archipelago of Canada, hand-drawn as several distinct islands.
island("-127 71 -125 73 -122 74 -121 75 -118 75 -116 74 -117 71 -120 70 -124 70")
island("-118 69 -115 71 -112 72 -110 74 -107 74 -104 73 -102 71 -104 69 -110 68 -114 69")
island("-106 74 -102 75 -97 75 -94 74 -96 73 -101 72 -104 73")
island("-96 72 -92 73 -89 73 -88 71 -91 70 -95 70")
island("-96 75 -91 76 -87 77 -82 77 -81 75 -86 74 -92 74")
island("-84 75 -79 76 -76 78 -72 78 -67 81 -62 82 -62 80 -66 78 -71 76 -77 75")
island("-81 63 -78 64 -77 66 -75 68 -77 71 -73 72 -69 72 -66 70 -63 67 -63 64 -66 62 -70 63 -73 64 -76 62")
island("-74 62 -70 63 -67 62 -65 60 -67 59 -71 60")
island("-100 69 -97 70 -95 69 -97 67 -99 67")

# North Atlantic and European islands.
island("-25 64 -22 66 -17 67 -13 66 -13 64 -17 63 -22 63")
island("""
-11 50 -9 52 -7 54 -6 55 -7 57 -6 58 -5 59
-4 58 -3 58 -4 56 -2 55 -1 54 0 52
1 51 1 50 -2 50 -5 50 -6 51 -5 52
-6 53 -8 53 -9 52 -10 51 -11 50
""")
island("-10 52 -9 54 -8 55 -6 55 -6 53 -8 52")
island("-4 60 -2 61 -1 60 -2 59 -4 59")
island("-8 58 -6 59 -5 58 -6 57")
island("10 55 12 57 12 58 10 57 8 57 8 55")
island("9 54 11 55 12 54 11 53")
island("12 55 16 57 20 57 22 55 19 54 16 55")
island("8 42 10 43 10 41 9 40 8 39")
island("9 39 10 40 10 39 9 38")
island("12 38 14 38 16 38 16 37 13 37 12 38")
island("19 39 20 40 21 39 20 38")
island("23 36 26 36 26 35 24 35")
island("32 35 35 35 35 34 33 34")

# High Arctic Eurasian archipelagos.
island("10 77 14 79 20 80 25 79 27 77 22 76 17 77 12 76")
island("13 74 17 75 18 74 14 73")
island("43 79 49 81 57 82 65 81 63 80 55 79 48 79")
island("51 71 54 74 56 76 61 77 67 76 65 73 61 72 57 70")
island("91 78 96 80 103 81 108 80 107 78 103 77 97 78")
island("137 74 143 76 150 76 154 75 148 73 141 73")
island("178 70 180 72 180 70")
island("-180 70 -178 72 -175 71 -176 70")

# Pacific and Atlantic North American islands.
island("-167 54 -164 55 -161 56 -160 55 -162 54")
island("-175 52 -172 53 -169 53 -168 52")
island("-180 51 -176 52 -173 52 -173 51")
island("177 51 180 52 180 51")
island("-84 22 -82 23 -79 23 -76 21 -74 20 -77 20 -79 21 -82 21")
island("-74 20 -72 20 -69 19 -68 18 -71 18 -73 18")
island("-78 19 -76 19 -76 17 -78 17")
island("-67 18 -65 19 -65 17 -67 17")
island("-62 17 -60 17 -60 16 -62 16")
island("-71 12 -67 13 -66 11 -69 11")
island("-82 13 -80 13 -79 12 -81 12")
island("-91 -1 -90 1 -89 0 -90 -2")
island("-61 -51 -58 -51 -57 -53 -60 -53 -62 -52")
# Indian Ocean, Southeast Asian, Pacific and subantarctic islands.
island("43 -12 46 -13 49 -12 50 -15 49 -19 47 -23 45 -26 43 -25 44 -21 45 -17")
island("55 -20 58 -20 57 -22 55 -22")
island("80 10 82 9 82 7 81 5 79 6 79 8")
island("73 5 74 7 73 8 72 6")
island("94 6 97 4 99 0 102 -3 105 -5 105 -6 103 -6 100 -3 97 0 95 4")
island("105 -6 108 -6 111 -7 114 -8 114 -9 110 -8 107 -8 105 -7")
island("109 1 110 3 113 5 116 7 119 5 118 2 117 -1 115 -4 112 -4 109 -2")
island("117 1 119 2 120 1 121 -2 123 -3 122 -5 120 -5 120 -3 118 -4 119 -2 117 -2")
island("115 -8 117 -8 120 -9 119 -10 116 -9")
island("119 -8 123 -8 125 -9 124 -10 120 -9")
island("125 -8 128 -8 130 -9 129 -10 126 -10")
island("131 -1 135 -2 138 -2 141 -3 144 -4 146 -5 149 -6 151 -8 151 -10 149 -10 146 -9 143 -8 141 -9 138 -9 135 -7 133 -5 130 -4")
island("129 -3 130 -2 131 -2 131 -5 129 -6")
island("121 19 122 18 123 16 124 14 123 12 121 14 120 16")
island("120 13 122 13 124 11 123 10 121 11")
island("124 10 126 8 126 6 124 5 122 6 121 8")
island("121 25 122 25 122 23 121 21 120 22 120 24")
island("109 20 111 20 111 18 109 18")
island("139 42 141 45 144 45 146 43 144 41 141 41")
island("140 41 141 39 141 38 140 36 139 35 137 35 135 34 133 34 131 33 130 34 132 35 134 35 136 36 138 38 139 40")
island("130 34 132 33 132 31 130 31 129 33")
island("133 34 135 34 135 33 133 33")
island("142 46 144 47 144 50 142 53 143 54 141 55 140 52 141 50 140 47")
island("143 -10 145 -11 146 -12 144 -12")
island("145 -40 148 -41 149 -43 147 -44 145 -43 144 -42")
island("173 -34 175 -35 177 -37 177 -39 175 -41 173 -41 173 -39 174 -38 172 -37")
island("172 -40 174 -42 173 -44 170 -46 168 -47 166 -46 167 -44 169 -43 170 -41")
island("166 -20 169 -20 169 -22 166 -23")
island("178 -16 180 -17 180 -19 178 -19 177 -17")
island("46 -46 52 -46 52 -50 47 -50")
island("69 -48 71 -49 69 -50")
island("-38 -54 -35 -54 -35 -55 -38 -55")

# Baltic and Black Seas are open saltwater.  At this 2-degree sampling scale
# small navigable straits have to be approximated by hand.
sea("""
10 54 13 54 16 55 19 55 22 55 24 57
25 59 28 60 29 61 26 61 23 60
22 62 22 65 20 65 18 62 18 60
16 58 14 56 12 56 11 55
""")
sea("28 41 30 41 32 42 35 41 38 41 41 42 42 43 39 44 36 45 34 46 31 46 30 45 29 44 28 42")
sea("46 41 49 43 52 44 52 46 49 47 47 46 46 44")
# Gulf of Bothnia has an especially long, narrow northern arm.
sea("18 60 19 62 20 65 23 66 24 64 22 62 22 60 20 59")


def inside(polygon):
    """Ray casting against every sample coordinate, not independent guesses."""
    mask = np.zeros(LON.shape, dtype=bool)
    for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + polygon[:1]):
        if y0 != y1:
            mask ^= ((y0 > LAT) != (y1 > LAT)) & (
                LON < (x1 - x0) * (LAT - y0) / (y1 - y0) + x0
            )
    return mask


def main():
    matrix = np.zeros((90, 180), dtype=bool)
    for polygon in land_shapes:
        matrix |= inside(polygon)
    for polygon in water_shapes:
        matrix &= ~inside(polygon)
    for polygon in island_shapes:
        matrix |= inside(polygon)

    output = Path("outputs")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "candidate.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(matrix.astype(np.uint8).tolist())
    image = Image.fromarray(matrix.astype(np.uint8) * 255, mode="L")
    image.save(output / "candidate.png")
    enlarged = image.resize((1440, 720), resample=Image.Resampling.NEAREST)
    enlarged.save(output / "candidate_8x.png")

    with (output / "candidate.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 90 and all(len(row) == 180 for row in rows)
    assert all(value in ("0", "1") for row in rows for value in row)
    assert np.array_equal(np.array(rows, dtype=np.uint8), matrix)
    assert image.size == (180, 90) and enlarged.size == (1440, 720)
    assert set(image.getdata()) <= {0, 255}
    assert set(enlarged.getdata()) <= {0, 255}
    assert np.array_equal(np.asarray(enlarged), np.repeat(np.repeat(np.asarray(image), 8, 0), 8, 1))
    print(f"Verified 90 x 180 = {matrix.size} samples; {matrix.sum()} land cells.")


if __name__ == "__main__":
    main()
