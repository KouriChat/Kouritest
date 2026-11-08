"""Knowledge-only schematic of land near Hecate Strait; no map data used.

Run with Python 3 and Pillow: python outputs/generate_map.py
Longitude/latitude sketches are deliberately approximate, not surveyed coastlines.
"""
import csv
from pathlib import Path
from PIL import Image

OUT = Path('outputs')
OUT.mkdir(exist_ok=True)

# Each outline is drawn manually in (longitude, latitude) WGS84 coordinates.
# The first polygon is the mainland, extending east beyond the sampled frame.
outlines = [
    [(-130.13,55.10),(-130.08,54.98),(-130.18,54.94),(-130.14,54.87),
     (-130.08,54.81),(-130.15,54.74),(-130.09,54.69),(-130.11,54.63),
     (-130.19,54.60),(-130.22,54.56),(-130.16,54.53),(-130.14,54.49),
     (-130.23,54.46),(-130.16,54.43),(-130.22,54.39),(-130.24,54.35),
     (-130.20,54.33),(-130.13,54.34),(-130.09,54.27),(-130.04,54.25),
     (-129.99,54.26),(-129.93,54.24),(-129.91,54.17),(-129.95,54.13),
     (-129.88,54.09),(-129.92,54.03),(-129.85,53.98),(-129.76,53.95),
     (-129.77,53.89),(-129.69,53.88),(-129.69,53.82),(-129.62,53.78),
     (-129.69,53.73),(-129.64,53.67),(-129.54,53.66),(-129.58,53.60),
     (-129.50,53.55),(-129.47,53.46),(-129.54,53.40),(-129.55,53.32),
     (-129.50,53.28),(-129.49,53.20),(-129.40,53.18),(-129.37,53.11),
     (-129.36,53.06),(-129.28,53.01),(-128.8,52.9),(-128.8,55.1)]

# A few smaller nearby islands represented by coarse elliptical sketches.
# (center longitude, center latitude, longitude radius, latitude radius)
islets = [
    (-130.72,54.74,.026,.019),(-130.82,54.80,.016,.015),
    (-130.80,54.38,.038,.028),(-130.83,54.30,.031,.020),
    (-130.69,54.27,.025,.020),(-130.61,54.30,.028,.018),
    (-130.56,54.20,.027,.016),(-130.55,54.56,.032,.017),
    (-130.49,54.59,.025,.021),(-130.47,54.54,.024,.015),
    (-130.32,54.57,.018,.015),(-130.26,54.53,.023,.016),
    (-130.58,53.74,.023,.033),(-130.54,53.67,.020,.031),
    (-130.57,53.57,.019,.027),(-130.51,53.36,.025,.020),
    (-130.40,53.23,.030,.028),(-130.34,53.16,.020,.027),
    (-130.24,53.18,.022,.024),(-130.53,53.13,.020,.018),
    (-130.06,53.25,.018,.022),(-129.93,53.28,.023,.028),
    (-129.84,53.13,.015,.025),(-129.58,53.05,.032,.042),
    (-129.65,53.13,.021,.021),(-129.73,53.08,.018,.030),
]


def inside_polygon(x, y, vertices):
    """Even-odd crossing test, applied to coordinates, not image cells."""
    inside = False
    x0, y0 = vertices[-1]
    for x1, y1 in vertices:
        if (y0 > y) != (y1 > y):
            cross_x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < cross_x:
                inside = not inside
        x0, y0 = x1, y1
    return inside


def land(lon, lat):
    return int(any(inside_polygon(lon, lat, polygon) for polygon in outlines)
               or any(((lon-x)/rx)**2 + ((lat-y)/ry)**2 <= 1
                      for x,y,rx,ry in islets))


def main():
    # Exact point sampling; no cell average, smoothing, or geographic lookup.
    grid = [[land(-131 + 2*c/126, 55 - 2*r/126) for c in range(127)]
            for r in range(127)]
    csv_file = OUT / 'candidate.csv'
    with csv_file.open('w', newline='', encoding='ascii') as f:
        csv.writer(f, lineterminator='\n').writerows(grid)
    original = Image.new('L', (127,127))
    original.putdata([255*v for row in grid for v in row])
    original.save(OUT / 'candidate.png')
    enlarged = original.resize((1016,1016), resample=Image.Resampling.NEAREST)
    enlarged.save(OUT / 'candidate_8x.png')

    with csv_file.open(newline='', encoding='ascii') as f:
        check = list(csv.reader(f))
    assert len(check) == 127 and all(len(row) == 127 for row in check)
    assert all(v in {'0','1'} for row in check for v in row)
    assert sum(map(len, check)) == 16129
    for path, size in [('candidate.png',(127,127)),
                       ('candidate_8x.png',(1016,1016))]:
        with Image.open(OUT / path) as im:
            assert im.size == size and set(im.convert('L').getdata()) <= {0,255}
    with Image.open(OUT / 'candidate_8x.png') as im:
        assert im.tobytes() == enlarged.tobytes()
    print('Validated 127x127 CSV, monochrome PNG and 8x nearest-neighbor PNG.')


if __name__ == '__main__':
    main(),
    # Dundas Island and its neighbours, on the south side of Dixon Entrance.
    [(-130.95,54.75),(-130.88,54.75),(-130.83,54.71),(-130.84,54.66),
     (-130.76,54.62),(-130.77,54.58),(-130.80,54.56),(-130.78,54.51),
     (-130.86,54.49),(-130.92,54.54),(-130.94,54.59),(-130.98,54.62),
     (-131.00,54.69)],
    [(-130.72,54.66),(-130.67,54.65),(-130.63,54.62),(-130.66,54.58),
     (-130.71,54.59),(-130.75,54.62)],
    [(-130.71,54.48),(-130.67,54.47),(-130.62,54.42),(-130.63,54.39),
     (-130.69,54.40),(-130.75,54.45)],
    # Tsimpsean Peninsula and adjacent islands around Prince Rupert.
    [(-130.45,54.43),(-130.41,54.43),(-130.38,54.39),(-130.32,54.37),
     (-130.30,54.34),(-130.32,54.31),(-130.36,54.29),(-130.40,54.32),
     (-130.43,54.34)],
    [(-130.54,54.36),(-130.50,54.38),(-130.46,54.34),(-130.43,54.29),
     (-130.49,54.27),(-130.52,54.29)],
    [(-130.40,54.52),(-130.34,54.50),(-130.29,54.45),(-130.31,54.40),
     (-130.35,54.42),(-130.39,54.46)],
    # Porcher Island (with prominent inlet-facing indentations).
    [(-130.64,54.17),(-130.57,54.18),(-130.54,54.14),(-130.48,54.15),
     (-130.43,54.11),(-130.39,54.13),(-130.32,54.09),(-130.29,54.03),
     (-130.23,53.99),(-130.27,53.93),(-130.22,53.88),(-130.25,53.83),
     (-130.31,53.81),(-130.37,53.83),(-130.44,53.82),(-130.50,53.85),
     (-130.57,53.84),(-130.62,53.88),(-130.67,53.92),(-130.67,53.98),
     (-130.70,54.03),(-130.66,54.08),(-130.68,54.12)],
    # Islands to the east and north of Porcher Island.
    [(-130.20,54.15),(-130.14,54.17),(-130.09,54.13),(-130.11,54.07),
     (-130.17,54.04),(-130.23,54.07)],
    [(-130.05,54.01),(-130.00,54.03),(-129.96,54.00),(-129.96,53.93),
     (-130.02,53.89),(-130.08,53.93)],
    # Banks Island, northwest of Pitt Island.
    [(-130.41,53.83),(-130.33,53.81),(-130.27,53.77),(-130.21,53.73),
     (-130.17,53.67),(-130.13,53.62),(-130.12,53.54),(-130.06,53.51),
     (-130.07,53.44),(-130.11,53.41),(-130.07,53.36),(-130.12,53.30),
     (-130.19,53.28),(-130.22,53.23),(-130.29,53.23),(-130.33,53.28),
     (-130.38,53.29),(-130.38,53.35),(-130.45,53.39),(-130.47,53.47),
     (-130.50,53.53),(-130.47,53.60),(-130.49,53.65),(-130.44,53.70)],
    # Pitt Island and nearby land separated from the mainland by channels.
    [(-129.98,54.05),(-129.93,54.03),(-129.91,53.99),(-129.88,53.94),
     (-129.88,53.89),(-129.82,53.84),(-129.83,53.79),(-129.78,53.75),
     (-129.80,53.69),(-129.77,53.63),(-129.72,53.58),(-129.72,53.51),
     (-129.77,53.45),(-129.75,53.39),(-129.83,53.35),(-129.91,53.38),
     (-129.94,53.45),(-130.00,53.50),(-130.02,53.56),(-129.99,53.63),
     (-130.03,53.69),(-130.01,53.77),(-130.04,53.83),(-130.00,53.89)],
    # Aristazabal Island at the southern edge of this square.
    [(-130.19,53.13),(-130.12,53.18),(-130.06,53.19),(-130.03,53.15),
     (-129.97,53.13),(-129.95,53.08),(-129.88,53.05),(-129.87,53.00),
     (-129.92,52.95),(-130.16,52.95),(-130.23,53.01),(-130.22,53.07)],
    [(-129.80,53.29),(-129.75,53.30),(-129.71,53.25),(-129.69,53.20),
     (-129.72,53.14),(-129.77,53.13),(-129.82,53.20)],
]
