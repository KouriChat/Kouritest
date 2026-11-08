# Indonesia and the Philippines: each group is a coarse hand-sketched outline.
add("95 5; 99 3; 102 0; 105 -3; 106 -6; 103 -5; 100 -2; 97 2; 95 4")
add("105 -6; 109 -6; 112 -7; 115 -8; 113 -9; 109 -8; 106 -7")
add("114 -8; 117 -8; 119 -9; 119 -10; 116 -9")
add("119 -8; 124 -8; 125 -9; 123 -10; 120 -10")
add("125 -8; 127 -8; 127 -9; 125 -9")
add("108 6; 112 7; 116 7; 118 5; 119 2; 118 -1; 116 -4; 112 -4; 109 -2; 109 1")
add("120 1; 122 2; 124 1; 125 -1; 123 -2; 121 -2")
add("120 -2; 122 -2; 123 -4; 121 -5; 119 -4")
add("122 -3; 125 -2; 125 -5; 123 -6; 122 -5")
add("126 2; 128 2; 128 0; 126 0")
add("127 -3; 130 -3; 129 -5; 127 -4")
add("130 -2; 135 -1; 141 -2; 147 -4; 151 -6; 151 -10; 147 -10; 144 -8; 141 -9; 138 -8; 135 -6; 132 -5")
add("131 -1; 133 0; 136 -1; 135 -3; 132 -2")
add("128 -7; 131 -7; 131 -9; 128 -9")
add("118 21; 120 19; 121 17; 123 18; 122 20; 120 21")
add("119 18; 121 17; 122 14; 124 12; 123 10; 121 12; 119 15")
add("122 13; 124 13; 125 11; 126 10; 124 9; 122 10")
add("121 10; 123 10; 123 8; 122 7; 120 8")
add("124 9; 126 9; 127 7; 126 5; 124 6")
add("121 25; 122 25; 122 22; 120 21; 120 23")
# Japan, Sri Lanka, Madagascar and other distinct islands.
add("130 34; 132 35; 134 34; 136 35; 139 36; 141 39; 141 41; 143 42; 141 42; 140 40; 138 37; 135 35; 132 34")
add("140 42; 143 43; 145 45; 142 45; 140 44")
add("130 33; 132 33; 133 32; 131 31")
add("133 34; 135 35; 134 33; 132 33")
add("127 27; 129 27; 129 26; 127 26")
add("80 10; 82 9; 81 6; 80 6; 79 8")
add("44 -12; 47 -12; 50 -15; 50 -19; 48 -23; 46 -26; 44 -25; 43 -20; 43 -16")
add("55 -20; 58 -20; 58 -21; 56 -21")
add("46 -12; 47 -11; 45 -11; 44 -12")
add("73 4; 74 4; 74 0; 73 0")
add("56 -4; 56 -5; 55 -5; 55 -4")
# Australia, Tasmania, New Zealand and major South Pacific islands.
add("""
113 -34; 113 -29; 114 -25; 114 -22; 117 -20;
121 -18; 123 -16; 125 -15; 128 -14; 130 -12;
133 -11; 135 -12; 136 -15; 139 -17; 141 -16;
142 -11; 144 -13; 145 -16; 146 -17; 147 -19;
149 -22; 151 -24; 153 -27; 153 -31; 151 -35;
149 -37; 146 -39; 143 -39; 140 -38; 137 -36;
135 -35; 133 -33; 130 -32; 127 -33; 123 -34;
119 -35; 116 -35
""")
add("144 -41; 146 -40; 148 -42; 147 -44; 145 -44; 144 -42")
add("172 -35; 174 -36; 175 -38; 178 -40; 177 -42; 175 -41; 173 -39")
add("172 -40; 173 -42; 172 -44; 169 -46; 167 -47; 166 -46; 168 -44; 170 -42")
add("166 -21; 168 -20; 168 -23; 166 -23")
add("158 -7; 160 -8; 161 -10; 159 -10; 158 -8")
add("164 -10; 166 -11; 167 -12; 165 -12")
add("178 -17; 180 -17; 180 -18; 178 -18")
add("-180 -17; -179 -17; -179 -18; -180 -18")
add("152 -4; 154 -4; 154 -5; 152 -5")
add("151 -2; 153 -2; 153 -4; 151 -4")
add("143 -10; 144 -10; 144 -11; 143 -11")
# Antarctica: continental ice is land, while the offshore ice shelves are sea.
add("""
-180 -78; -175 -78; -168 -77; -159 -76; -149 -75;
-139 -74; -129 -73; -120 -73; -110 -73; -100 -73;
-90 -73; -81 -72; -76 -70; -72 -68; -69 -65;
-65 -63; -63 -65; -66 -68; -64 -70; -58 -72;
-52 -74; -42 -74; -32 -73; -23 -72; -15 -71;
-7 -70; 1 -69; 10 -69; 19 -69; 28 -69;
36 -68; 44 -67; 53 -67; 62 -67; 72 -66;
82 -66; 93 -65; 102 -65; 111 -66; 120 -66;
130 -66; 139 -67; 148 -68; 157 -70; 166 -72;
173 -75; 180 -78; 180 -90; -180 -90
""")


def inside(x, y, vertices):
    """Even-odd ray crossing for a longitude-latitude polygon."""
    result = False
    previous_x, previous_y = vertices[-1]
    for next_x, next_y in vertices:
        if (previous_y > y) != (next_y > y):
            edge_x = previous_x + (y - previous_y) * (next_x - previous_x) / (next_y - previous_y)
            if x < edge_x:
                result = not result
        previous_x, previous_y = next_x, next_y
    return result


def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for y in range(89, -90, -2):
        row = []
        for x in range(-179, 180, 2):
            row.append(int(any(inside(x, y, shape) for shape in land)
                           and not any(inside(x, y, shape) for shape in water)))
        rows.append(row)
    assert len(rows) == 90 and all(len(row) == 180 and set(row) <= {0, 1} for row in rows)
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        csv.writer(f).writerows(rows)
    image = Image.new("RGB", (180, 90))
    image.putdata([(255, 255, 255) if value else (0, 0, 0)
                   for row in rows for value in row])
    image.save(out / "candidate.png")
    image.resize((1440, 720), Image.Resampling.NEAREST).save(out / "candidate_8x.png")
    assert set(image.getdata()) <= {(0, 0, 0), (255, 255, 255)}
    assert Image.open(out / "candidate_8x.png").size == (1440, 720)


if __name__ == "__main__":
    main()
