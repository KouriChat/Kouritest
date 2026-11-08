"""Hand-drawn geographic approximations, evaluated at the specified cell centres."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

LON, LAT = np.meshgrid(np.arange(-179, 180, 2), np.arange(89, -90, -2))
land = np.zeros((90, 180), dtype=bool)

def polygon(coords, value=True):
    """Ray-crossing classification of all sample centres for a hand-written outline."""
    p = np.asarray([float(x) for x in coords.split()], dtype=float).reshape(-1, 2)
    inside = np.zeros_like(land)
    for (x1, y1), (x2, y2) in zip(p, np.roll(p, -1, axis=0)):
        if y1 == y2:
            continue
        crossed = ((y1 > LAT) != (y2 > LAT)) & (LON < (x2-x1)*(LAT-y1)/(y2-y1)+x1)
        inside ^= crossed
    land[inside] = value

# North America, including the isthmus as far as Panama.
polygon("""
-168 65 -166 62 -161 61 -159 57 -154 57 -151 59 -147 60 -141 60
-136 59 -134 56 -130 54 -128 51 -125 49 -125 45 -124 42 -123 39
-120 36 -118 34 -117 32 -115 31 -112 29 -111 27 -109 25 -106 23
-105 20 -103 18 -100 17 -97 16 -95 15 -92 14 -90 14 -87 13 -85 11
-83 9 -80 9 -79 8 -78 9 -80 10 -82 10 -83 12 -83 16 -86 16 -87 18
-87 21 -90 21 -93 19 -96 19 -97 21 -97 25 -97 28 -94 29 -90 29
-87 30 -85 30 -82 25 -80 25 -80 30 -78 33 -76 36 -75 39
-72 41 -70 42 -70 44 -67 45 -65 44 -61 44 -60 46 -56 47 -53 47
-53 51 -56 53 -60 54 -62 57 -65 60 -69 61 -73 62 -78 62 -79 59
-79 55 -82 53 -86 52 -90 52 -94 55 -94 59 -89 63 -96 65 -103 68
-112 69 -121 70 -128 70 -135 69 -140 70 -148 70 -153 71 -158 70
-162 68 -166 68 -168 65
""")
# South America.
polygon("""
-79 9 -76 10 -73 11 -71 12 -67 11 -63 10 -60 9 -58 7 -55 6
-52 5 -50 2 -47 0 -44 -2 -40 -3 -37 -5 -35 -7 -36 -11 -38 -14
-39 -18 -41 -21 -44 -23 -47 -25 -49 -28 -51 -31 -53 -33
-56 -35 -58 -38 -62 -40 -64 -42 -66 -45 -68 -48 -68 -52 -70 -55
-73 -55 -75 -51 -74 -47 -73 -43 -73 -38 -72 -34 -71 -30 -70 -26
-70 -22 -70 -18 -73 -16 -76 -14 -77 -10 -79 -6 -80 -2 -79 1 -77 4 -79 9
""")
# Eurasia: Arctic margin, Pacific margin, Southeast Asia, India, Arabia,
# the Mediterranean shores and the Atlantic-facing coast.
polygon("""
-10 36 -9 43 -5 44 -4 47 -5 48 -2 49 0 51 4 52 7 54
8 55 10 56 11 58 7 58 5 60 7 62 11 64 14 67 18 69
22 70 26 71 29 70 32 70 36 68 40 67 45 68 50 69 55 69
60 70 65 71 71 72 76 73 83 73 88 74 94 75 100 76
107 77 113 75 118 74 125 73 132 72 137 71 143 72 149 72
156 71 162 70 169 69 177 68 179 67 178 64 174 62 168 61
163 60 160 57 162 55 161 52 159 51 156 52 155 54 154 57
151 59 146 59 143 56 141 53 141 49 138 48 135 46 132 44
130 43 130 40 129 37 129 35 127 35 126 38 125 40
122 40 121 37 120 35 121 32 120 30 121 27 119 25
116 23 113 22 110 20 108 17 108 14 109 12 107 10 105 10
105 13 103 16 102 19 100 20 100 16 99 12 100 8 102 4
104 1 103 0 100 2 98 5 98 9 97 14 94 17 93 20
90 22 88 21 86 20 83 18 81 16 80 13 78 9 76 8
74 11 73 15 72 18 70 20 68 23 69 25 71 25 73 23
69 21 66 24 62 25 59 25 56 26 53 27 51 29 49 30
48 29 49 27 51 25 54 24 57 22 59 19 57 16 54 14
50 13 46 13 43 12 42 14 41 18 39 22 37 26 35 28
34 29 33 31 35 32 36 35 33 36 30 36 27 36 26 38
27 40 29 41 32 41 35 42 39 42 42 41 41 40 37 40
32 41 29 41 26 40 24 39 23 37 24 35 22 36
21 39 19 41 16 42 15 44 14 45 13 44 12 41
15 40 16 38 15 37 13 38 11 41 9 43 7 44
5 43 3 43 1 42 -1 42 -3 41 -5 40 -7 37 -10 36
""")
# Northeast Siberia crosses the date line.
polygon("-180 66 -176 66 -172 65 -170 63 -173 61 -178 63 -180 63")
# Africa.
polygon("""
-17 15 -17 21 -16 25 -14 28 -10 32 -6 35 -1 36
5 37 10 37 15 35 20 33 25 32 30 31 33 31
34 29 33 27 36 22 38 19 42 15 44 12
48 12 51 11 51 8 49 5 46 2 43 0 42 -3
41 -6 40 -10 39 -15 37 -19 35 -24 33 -28
31 -31 28 -33 25 -34 21 -35 18 -34 17 -30
15 -25 13 -20 12 -16 11 -11 9 -6 8 -1
7 3 5 5 2 5 -1 5 -4 5 -8 5
-10 7 -13 9 -16 12 -17 15
""")
# Australia and Antarctica; Antarctica's northern edge includes the peninsula.
polygon("""
113 -22 115 -18 119 -16 123 -17 126 -14 130 -12 135 -12
136 -15 136 -17 139 -17 141 -15 142 -12 145 -15
146 -19 149 -21 152 -24 153 -28 151 -33 148 -37
144 -39 141 -38 138 -36 135 -35 131 -32 127 -33
123 -34 118 -35 115 -34 113 -29 113 -22
""")
polygon("""
-180 -78 -170 -79 -160 -78 -150 -77 -140 -75 -130 -74
-120 -73 -110 -73 -100 -73 -90 -72 -82 -72 -77 -70
-73 -67 -69 -63 -65 -62 -62 -65 -64 -68 -61 -71
-55 -73 -46 -75 -35 -76 -26 -75 -19 -73 -11 -71
0 -70 12 -69 25 -68 38 -67 50 -67 63 -67
76 -66 89 -66 102 -66 114 -66 127 -67 140 -67
151 -69 161 -71 171 -74 180 -77 180 -90 -180 -90
""")
# Greenland, Baffin, Ellesmere and Canada's Arctic archipelago.
polygon("-52 59 -47 60 -43 60 -39 65 -39 70 -34 75 -38 79 -44 82 -52 83 -60 82 -63 79 -61 75 -56 71 -52 68 -52 64")
polygon("-80 62 -75 63 -70 67 -67 71 -69 74 -75 74 -80 71 -82 67")
polygon("-82 76 -77 79 -72 81 -67 83 -71 84 -80 82 -85 79")
polygon("-92 73 -85 75 -83 78 -88 81 -97 81 -103 79 -100 75")
polygon("-114 73 -108 74 -103 77 -105 79 -112 78 -119 75")
polygon("-124 70 -117 71 -112 73 -114 75 -123 75 -128 72")
polygon("-102 68 -96 69 -93 72 -99 74 -106 72 -109 70")
# Water enclosed by the broad Eurasian outline: Baltic, White and Black seas.
polygon("10 55 13 54 16 54 20 54 22 55 25 56 28 59 30 60 30 61 28 60 26 60 24 59 22 60 22 63 20 65 18 66 17 64 16 61 13 58 10 57", False)
polygon("34 64 37 64 40 65 43 66 45 67 43 69 39 68 35 67", False)
polygon("27 42 30 41 33 41 36 41 39 41 42 41 42 43 40 45 37 46 34 46 31 46 29 45 27 43", False)
polygon("35 45 38 45 39 46 36 47 35 46", False)
# Hudson Bay is largely outlined by the mainland, with a few broad inlets.
polygon("-92 57 -91 55 -86 54 -82 54 -81 56 -83 58 -87 59", False)
# Alaska Peninsula, the Aleutians, and the Baja California peninsula.
polygon("-162 61 -158 60 -155 58 -157 56 -161 56 -165 55 -163 57 -160 59")
polygon("-167 54 -163 54 -162 53 -167 52 -170 52")
polygon("-176 52 -172 52 -171 51 -177 51")
polygon("-115 32 -112 28 -110 25 -109 22 -110 22 -112 25 -114 29 -117 32")
# Newfoundland and large islands along the Atlantic coast.
polygon("-59 51 -55 51 -52 49 -53 47 -56 46 -59 47")
polygon("-67 48 -64 47 -62 46 -61 45 -64 45 -66 47")
# British Isles, Iceland, Svalbard and Mediterranean islands.
polygon("-6 50 -5 53 -3 55 -5 57 -3 58 0 58 2 56 0 54 1 52 -1 50")
polygon("-11 52 -10 54 -8 55 -6 55 -6 53 -8 51")
polygon("-24 64 -22 66 -18 67 -14 66 -14 64 -17 63 -21 63")
polygon("10 77 15 80 22 80 25 78 21 76 15 76")
polygon("8 41 10 42 10 39 9 38 8 39")
polygon("12 38 15 38 15 37 13 36 12 37")
polygon("23 36 26 36 26 35 23 35")
polygon("32 35 34 35 34 34 32 34")
polygon("-1 39 3 40 4 39 2 38")
# Madagascar, nearby Africa, and Indian Ocean islands.
polygon("49 -12 50 -16 49 -20 47 -24 45 -26 44 -25 43 -20 44 -16 46 -13")
polygon("39 -11 41 -12 40 -13 39 -12")
polygon("55 -20 58 -20 58 -21 55 -21")
polygon("57 -4 56 -5 55 -4")
polygon("79 10 81 9 82 7 81 6 80 6 79 8")
polygon("73 5 74 5 74 3 73 3")
# East Asia: Sakhalin, Japan, Taiwan and Hainan.
polygon("143 54 144 52 143 48 142 46 141 47 141 51")
polygon("145 45 146 44 145 42 142 41 140 42 141 44")
polygon("141 41 142 39 141 37 140 35 138 35 137 34 135 34 134 35 132 34 130 33 131 35 134 36 136 38 139 40")
polygon("133 34 134 34 134 32 132 32")
polygon("131 31 132 31 131 30 130 30")
polygon("129 29 130 29 130 28 129 28")
polygon("121 25 122 25 122 23 121 21 120 22 120 24")
polygon("109 20 111 20 111 18 109 18")
# Maritime Southeast Asia: Sumatra, Java, Borneo, Sulawesi, New Guinea.
polygon("95 6 98 6 101 3 104 1 106 -5 105 -6 102 -4 100 -1 97 2")
polygon("105 -6 108 -6 112 -7 115 -8 114 -9 109 -8 106 -7")
polygon("109 1 113 2 117 5 119 5 119 1 117 -3 114 -4 110 -3")
polygon("119 2 122 2 123 0 122 -2 124 -4 123 -6 121 -5 120 -3 118 -5 119 -1")
polygon("119 -8 121 -8 122 -10 120 -10")
polygon("123 -8 126 -8 127 -9 125 -10 123 -9")
polygon("128 0 131 0 131 -3 128 -4 126 -3 127 -1")
polygon("130 -1 133 -1 136 -3 138 -2 141 -2 142 -5 145 -6 148 -6 152 -5 152 -8 147 -9 142 -9 139 -8 135 -5 132 -5")
polygon("144 -11 148 -11 149 -13 146 -14 144 -13")
# Philippine archipelago.
polygon("120 19 122 19 123 17 122 15 121 14 120 16")
polygon("123 14 124 13 125 11 123 10 121 11 122 13")
polygon("119 12 121 12 121 10 120 9 118 10")
polygon("125 10 127 9 127 7 125 6 124 8")
polygon("122 9 125 8 125 6 123 6 121 8")
# New Zealand, Tasmania, Melanesia and Hawaii.
polygon("145 -40 148 -40 149 -43 147 -44 145 -43")
polygon("173 -34 175 -37 176 -39 174 -41 172 -42 171 -40 172 -37")
polygon("172 -40 173 -42 171 -45 168 -47 166 -46 168 -43 170 -42")
polygon("160 -9 162 -9 162 -10 160 -10")
polygon("166 -15 167 -15 167 -17 166 -17")
polygon("178 -17 180 -17 180 -19 178 -19")
polygon("-158 22 -157 22 -157 21 -159 21")
polygon("-156 21 -155 21 -155 19 -156 19")
# Caribbean islands and Pacific-facing islands near the Americas.
polygon("-85 22 -83 23 -80 23 -77 22 -75 21 -76 20 -79 20 -83 21")
polygon("-79 19 -76 20 -73 20 -71 19 -72 18 -76 18")
polygon("-74 19 -71 20 -69 19 -69 18 -72 18")
polygon("-68 18 -65 18 -65 17 -67 17")
polygon("-78 19 -77 18 -76 17 -78 17")
polygon("-60 14 -60 13 -61 13")
polygon("-91 1 -89 1 -89 -1 -91 -1")
# Tierra del Fuego and the Falklands.
polygon("-74 -53 -69 -52 -66 -54 -68 -55 -73 -55")
polygon("-61 -51 -58 -51 -57 -53 -61 -53")

def main():
    dest = Path("outputs")
    dest.mkdir(exist_ok=True)
    with (dest / "candidate.csv").open("w", newline="", encoding="ascii") as fp:
        csv.writer(fp, lineterminator="\n").writerows(land.astype(np.uint8).tolist())
    original = Image.fromarray(np.where(land, 255, 0).astype(np.uint8), mode="L")
    original.save(dest / "candidate.png")
    original.resize((1440, 720), resample=Image.Resampling.NEAREST).save(dest / "candidate_8x.png")
    with (dest / "candidate.csv").open(newline="", encoding="ascii") as fp:
        rows = list(csv.reader(fp))
    assert len(rows) == 90 and all(len(row) == 180 and set(row) <= {"0", "1"} for row in rows)
    assert sum(map(len, rows)) == 16200
    with Image.open(dest / "candidate.png") as img:
        assert img.size == (180, 90) and set(img.convert("L").getdata()) <= {0, 255}
        nearest = img.resize((1440, 720), Image.Resampling.NEAREST)
    with Image.open(dest / "candidate_8x.png") as enlarged:
        assert enlarged.size == (1440, 720) and set(enlarged.convert("L").getdata()) <= {0, 255}
        assert list(enlarged.convert("L").getdata()) == list(nearest.getdata())
    print("Validated: 90 x 180, 16200 values; 180 x 90 and 1440 x 720 binary PNGs.")

if __name__ == "__main__":
    main()
