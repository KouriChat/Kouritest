"""Knowledge-only reconstruction; no map files, services, or reference data.

All coastlines below are hand-written, deliberately approximate polygons in
(longitude east, latitude north). They are not surveyed boundaries. Sampling
uses point-in-polygon rules, not independent language-model judgments. Inland
lakes are left as land. Floating ice is not added. Small unremembered islands
are necessarily omitted. Run: python outputs/generate_map.py
Dependencies: Python 3, numpy, Pillow. No network or external input is used.
"""
from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image

LAND = []
WATER = []

def land(name, points):
    LAND.append((name, points))

def water(name, points):
    WATER.append((name, points))

# Axel Heiberg: the portion north of the sampling window is only polygon
# closure. Sverdrup Channel lies to its west, Eureka Sound to its east.
land('Axel Heiberg', [
    (-96.3,80.5),(-94.8,81.25),(-91.0,81.4),(-87.2,80.85),
    (-86.3,80.0),(-86.7,79.65),(-87.25,79.4),(-87.45,79.08),
    (-87.15,78.85),(-87.7,78.55),(-88.3,78.35),(-89.15,78.2),
    (-89.65,78.25),(-90.1,78.15),(-90.55,78.23),(-90.85,78.05),
    (-91.55,78.0),(-92.25,78.12),(-92.7,78.35),(-93.25,78.53),
    (-93.55,78.85),(-94.55,79.0),(-95.15,79.25),(-95.0,79.6),
    (-95.8,79.9)])
water('Axel southwest inlets', [
    (-94.6,78.72),(-93.1,78.9),(-92.7,79.12),(-91.9,79.2),
    (-92.1,78.96),(-92.85,78.72),(-93.1,78.42)])
water('Axel southeast inlet', [
    (-88.0,78.36),(-89.2,78.6),(-90.2,78.72),(-90.6,78.95),
    (-90.05,78.92),(-89.5,78.74),(-88.35,78.64)])

# Ellesmere's western and southern shores; east/north polygon closure is
# outside the requested window and does not fill any missing sample values.
land('Ellesmere', [
    (-85.6,80.7),(-84.75,80.15),(-85.1,79.9),(-85.0,79.65),
    (-84.7,79.45),(-84.85,79.2),(-85.25,79.05),(-85.25,78.8),
    (-85.75,78.6),(-86.8,78.62),(-87.6,78.4),(-88.05,78.15),
    (-88.25,77.93),(-87.7,77.73),(-88.3,77.52),(-88.05,77.27),
    (-88.8,77.05),(-89.65,76.98),(-89.4,76.76),(-88.75,76.6),
    (-87.8,76.48),(-86.9,76.45),(-85.8,76.45),(-85.15,76.34),
    (-84.5,76.43),(-83.8,76.42),(-83.15,76.55),(-82.8,76.42),
    (-82.15,76.45),(-81.7,76.6),(-81.05,76.45),(-80.5,76.35),
    (-78.4,76.4),(-76.0,77.0),(-72.0,80.5),(-73.0,83.5),
    (-85.0,83.5)])
water('Ellesmere Bay Fiord', [
    (-88.75,77.05),(-87.45,77.13),(-86.7,77.3),(-85.95,77.35),
    (-86.45,77.17),(-87.4,76.98),(-88.7,76.83)])
water('Ellesmere Vendom Fiord', [
    (-87.9,76.42),(-87.3,76.75),(-86.4,76.94),(-85.6,77.15),
    (-85.45,77.1),(-86.0,76.8),(-86.95,76.55),(-87.1,76.38)])
water('Ellesmere southern fjord', [
    (-84.65,76.35),(-84.7,76.7),(-84.15,76.92),(-83.9,77.18),
    (-83.7,77.1),(-83.95,76.7),(-84.2,76.4)])

# Islands south and west of Axel Heiberg.
land('Amund Ringnes', [
    (-96.65,78.63),(-95.6,78.65),(-94.85,78.4),(-94.65,78.15),
    (-95.2,78.0),(-95.8,77.85),(-96.0,77.62),(-96.8,77.47),
    (-97.55,77.56),(-97.8,77.9),(-97.45,78.23),(-96.85,78.3)])
land('Ellef Ringnes', [
    (-103.25,79.35),(-101.85,79.35),(-101.0,79.15),(-100.15,79.0),
    (-99.6,78.78),(-99.3,78.5),(-99.75,78.28),(-99.45,78.04),
    (-100.1,77.83),(-100.2,77.56),(-100.7,77.66),(-101.2,77.94),
    (-102.2,78.08),(-103.3,78.25),(-103.5,78.5),(-103.0,78.73)])
water('Ellef Ringnes east bay', [
    (-99.45,78.5),(-100.5,78.63),(-101.0,78.5),(-100.85,78.37),
    (-99.7,78.27)])
land('King Christian', [
    (-102.62,77.9),(-101.9,77.96),(-101.25,77.81),(-101.17,77.57),
    (-101.8,77.46),(-102.45,77.55),(-102.72,77.72)])
land('Lougheed', [
    (-105.6,77.6),(-105.05,77.56),(-104.72,77.35),(-105.02,77.12),
    (-105.55,77.05),(-106.0,77.35)])
land('Borden', [
    (-112.0,78.6),(-109.0,78.7),(-107.6,78.55),(-106.0,78.55),
    (-105.3,78.32),(-105.05,78.13),(-105.5,78.04),(-106.3,78.1),
    (-108.0,78.0),(-111.0,78.1)])
# Norwegian Bay / Jones Sound islands, with the channels between them kept
# open by using separate polygons rather than a connected land envelope.
land('Cornwall', [
    (-96.1,77.82),(-95.25,77.84),(-94.5,77.75),(-93.65,77.62),
    (-93.45,77.43),(-94.1,77.3),(-95.05,77.3),(-95.8,77.43),
    (-96.25,77.6)])
land('Graham', [
    (-91.95,77.53),(-91.4,77.65),(-90.55,77.58),(-89.7,77.39),
    (-89.8,77.18),(-90.4,77.12),(-91.1,77.22),(-91.9,77.3)])
land('Buckingham', [
    (-91.25,77.18),(-90.9,77.2),(-90.65,77.08),(-90.85,77.0),
    (-91.25,77.05)])
land('North Kent', [
    (-90.55,76.82),(-90.05,76.88),(-89.6,76.77),(-89.15,76.61),
    (-89.25,76.43),(-89.85,76.4),(-90.3,76.5),(-90.65,76.65)])

# Devon: Grinnell Peninsula in the northwest, Jones Sound in the north,
# Lancaster Sound in the south and Wellington Channel in the west.
land('Devon', [
    (-96.65,76.72),(-96.0,76.9),(-95.15,77.03),(-94.3,76.93),
    (-93.55,76.78),(-92.85,76.63),(-92.2,76.42),(-91.5,76.52),
    (-90.85,76.46),(-90.3,76.43),(-89.6,76.35),(-88.9,76.23),
    (-88.0,76.22),(-87.25,76.12),(-86.3,76.14),(-85.35,76.02),
    (-84.5,75.95),(-83.8,75.98),(-82.9,75.87),(-82.25,75.8),
    (-81.6,75.8),(-81.05,75.73),(-80.6,75.79),(-80.5,75.66),
    (-80.05,75.62),(-79.55,75.55),(-79.85,75.3),(-79.7,75.05),
    (-80.05,74.78),(-80.55,74.6),(-81.15,74.53),(-81.8,74.5),
    (-82.45,74.52),(-83.0,74.55),(-83.55,74.58),(-84.0,74.57),
    (-84.65,74.62),(-85.35,74.55),(-86.15,74.48),(-86.8,74.48),
    (-87.45,74.5),(-88.1,74.57),(-88.7,74.56),(-89.35,74.6),
    (-90.0,74.6),(-90.6,74.64),(-91.0,74.63),(-91.55,74.72),
    (-92.05,74.78),(-92.5,74.96),(-92.55,75.2),(-92.8,75.42),
    (-92.55,75.66),(-92.8,75.85),(-93.25,75.98),(-94.05,76.03),
    (-94.85,76.15),(-95.6,76.25),(-96.2,76.45)])
water('Devon Croker Bay', [
    (-83.75,74.48),(-83.65,74.85),(-83.35,75.1),(-83.25,75.35),
    (-82.9,75.5),(-82.85,75.25),(-83.02,75.0),(-83.1,74.52)])
water('Devon Dundas Harbour', [
    (-82.8,74.45),(-82.7,74.68),(-82.43,74.74),(-82.35,74.48)])
water('Devon Maxwell Bay', [
    (-88.7,74.49),(-88.65,74.83),(-88.25,75.07),(-87.95,75.13),
    (-88.05,74.86),(-88.2,74.52)])
water('Devon Radstock Bay', [
    (-91.1,74.56),(-91.0,74.85),(-90.7,74.94),(-90.55,74.62)])
water('Devon northeast fjord', [
    (-83.8,76.03),(-83.5,75.88),(-83.75,75.66),(-83.65,75.5),
    (-83.35,75.65),(-83.2,75.9),(-83.2,76.08)])
water('Devon north-central fjord', [
    (-87.75,76.35),(-87.55,76.03),(-87.05,75.86),(-86.8,75.7),
    (-86.85,75.96),(-87.15,76.15),(-87.1,76.4)])
land('Beechey', [
    (-92.02,74.77),(-91.82,74.78),(-91.76,74.7),(-91.95,74.68)])

# Bathurst and its western satellite islands.
land('Bathurst', [
    (-101.2,75.1),(-100.85,75.4),(-101.05,75.73),(-100.7,76.02),
    (-100.55,76.35),(-100.1,76.55),(-99.35,76.62),(-98.9,76.5),
    (-98.7,76.22),(-98.15,76.55),(-97.55,76.52),(-97.2,76.3),
    (-97.5,76.02),(-97.65,75.75),(-97.55,75.5),(-97.8,75.26),
    (-98.0,75.05),(-98.55,74.92),(-99.15,74.91),(-99.55,74.75),
    (-100.05,74.75),(-100.3,74.98),(-100.85,74.95)])
water('Bathurst Erskine Inlet', [
    (-99.25,76.72),(-99.13,76.3),(-99.1,76.02),(-98.8,75.8),
    (-98.6,75.79),(-98.72,76.07),(-98.55,76.35),(-98.4,76.7)])
water('Bathurst May Inlet', [
    (-101.05,76.16),(-100.4,76.07),(-100.0,75.84),(-99.6,75.78),
    (-99.8,76.04),(-100.1,76.2),(-100.45,76.43)])
land('Cameron', [
    (-104.55,76.62),(-103.8,76.68),(-103.0,76.54),(-102.8,76.32),
    (-103.35,76.21),(-104.05,76.27),(-104.5,76.4)])
land('Vanier', [
    (-104.0,76.3),(-103.45,76.36),(-102.7,76.28),(-102.15,76.11),
    (-102.3,76.06),(-103.1,76.09),(-103.9,76.12)])
land('Massey', [
    (-103.48,76.02),(-103.02,76.05),(-102.65,76.0),(-102.65,75.86),
    (-102.95,75.82),(-103.4,75.88)])
land('Alexander', [
    (-102.6,75.93),(-102.3,75.99),(-101.85,75.92),(-101.72,75.76),
    (-102.1,75.68),(-102.55,75.73),(-102.7,75.82)])
land('Helena', [
    (-101.4,76.74),(-100.8,76.8),(-100.35,76.67),(-100.45,76.52),
    (-101.05,76.49),(-101.5,76.58)])
# Cornwallis group and smaller islands in the central channels.
land('Cornwallis', [
    (-96.1,75.5),(-95.85,75.66),(-95.3,75.7),(-94.65,75.64),
    (-94.0,75.58),(-93.65,75.42),(-93.35,75.15),(-93.4,74.95),
    (-93.8,74.78),(-94.3,74.68),(-94.85,74.68),(-95.25,74.65),
    (-95.65,74.8),(-95.85,75.05),(-96.05,75.24)])
land('Little Cornwallis', [
    (-96.95,75.62),(-96.5,75.66),(-96.2,75.5),(-96.15,75.32),
    (-96.5,75.28),(-96.85,75.4)])
land('Baillie Hamilton', [
    (-95.3,75.96),(-94.8,76.04),(-94.25,75.96),(-94.15,75.82),
    (-94.65,75.75),(-95.15,75.8)])
land('Dundas', [
    (-95.98,76.2),(-95.65,76.29),(-95.32,76.23),(-95.45,76.1),
    (-95.84,76.09)])
land('Griffith', [
    (-95.85,74.6),(-95.35,74.65),(-95.05,74.55),(-95.2,74.43),
    (-95.65,74.44)])
land('Somerville', [
    (-96.4,74.5),(-96.1,74.55),(-95.95,74.47),(-96.07,74.38),
    (-96.37,74.4)])
land('Garrett', [
    (-96.62,75.02),(-96.45,75.09),(-96.3,75.03),(-96.45,74.95)])
land('Lowther', [
    (-97.75,74.63),(-97.35,74.63),(-97.14,74.5),(-97.38,74.45),
    (-97.77,74.51)])
land('Young', [
    (-98.25,74.08),(-98.05,74.12),(-97.91,74.04),(-98.1,73.98)])
land('Hamilton', [
    (-99.13,74.28),(-98.77,74.32),(-98.57,74.22),(-98.78,74.13),
    (-99.12,74.17)])
land('Browne', [
    (-96.5,74.85),(-96.3,74.89),(-96.22,74.8),(-96.45,74.77)])

# Southern edge: northern Prince of Wales, Somerset, Baffin and Bylot.
land('Prince of Wales', [
    (-103.0,71.0),(-102.75,72.5),(-102.25,73.05),(-101.8,73.38),
    (-101.2,73.65),(-100.65,73.82),(-100.2,73.93),(-99.55,73.87),
    (-99.05,73.72),(-98.55,73.73),(-98.1,73.83),(-97.6,73.73),
    (-97.05,73.61),(-96.78,73.33),(-96.62,73.0),(-96.45,72.5),
    (-97.3,70.5),(-101.0,70.5)])
water('Prince of Wales northern bay', [
    (-100.2,74.03),(-100.1,73.75),(-99.65,73.55),(-99.22,73.35),
    (-99.12,73.55),(-99.4,73.87),(-99.5,74.03)])
land('Russell', [
    (-99.1,74.03),(-98.65,74.1),(-98.05,74.03),(-97.65,73.92),
    (-97.62,73.83),(-98.05,73.8),(-98.65,73.88),(-99.0,73.87)])
land('Somerset', [
    (-95.8,71.0),(-95.7,72.5),(-95.4,73.1),(-95.55,73.4),
    (-95.25,73.72),(-95.05,73.95),(-94.45,74.08),(-93.85,74.1),
    (-93.3,74.03),(-92.75,74.02),(-92.25,73.89),(-91.65,73.9),
    (-91.15,73.98),(-90.55,73.93),(-90.2,73.83),(-90.6,73.62),
    (-91.0,73.4),(-91.05,73.05),(-91.5,72.5),(-92.0,71.0)])
water('Somerset Cunningham Inlet', [
    (-94.15,74.2),(-94.1,73.95),(-93.8,73.8),(-93.6,73.82),
    (-93.75,74.2)])
water('Somerset Creswell Bay', [
    (-91.1,73.2),(-91.65,73.02),(-92.2,72.95),(-92.3,72.75),
    (-90.9,72.8)])
land('Baffin - Brodeur Peninsula', [
    (-90.0,70.0),(-89.8,72.3),(-89.3,73.0),(-89.0,73.35),
    (-88.6,73.6),(-88.25,73.78),(-87.6,73.86),(-86.9,73.87),
    (-86.25,73.82),(-85.7,73.7),(-85.65,73.45),(-85.9,73.2),
    (-86.1,73.0),(-86.25,72.5),(-86.0,71.0),(-85.0,70.0)])
land('Baffin - Borden Peninsula', [
    (-85.0,71.0),(-85.2,72.5),(-85.12,73.0),(-84.95,73.4),
    (-84.85,73.75),(-84.4,73.82),(-83.8,73.75),(-83.2,73.72),
    (-82.6,73.64),(-82.15,73.48),(-81.8,73.2),(-81.7,73.0),
    (-81.5,72.8),(-80.5,72.0),(-79.0,70.5)])
water('Baffin Elwin Inlet', [
    (-84.4,73.9),(-84.35,73.55),(-84.1,73.3),(-83.9,73.1),
    (-83.85,73.35),(-84.1,73.62),(-84.05,73.9)])
land('Bylot', [
    (-81.8,73.15),(-81.6,73.45),(-81.3,73.65),(-80.8,73.8),
    (-80.1,73.87),(-79.0,73.75),(-77.0,73.2),(-76.0,72.8),
    (-78.0,72.65),(-79.3,72.75),(-80.4,72.8),(-81.3,72.93)])

def contains(points, x, y):
    """Vectorized even/odd point test, without a geographic library."""
    inside = np.zeros(x.shape, dtype=bool)
    x0, y0 = points[-1]
    for x1, y1 in points:
        if y0 != y1:
            cross = ((y0 > y) != (y1 > y))
            edge_x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            inside ^= cross & (x < edge_x)
        x0, y0 = x1, y1
    return inside
land('Victoria northeastern edge', [
    (-108.5,71.5),(-108.5,73.45),(-107.2,73.55),(-106.3,73.45),
    (-105.55,73.37),(-105.05,73.24),(-104.65,73.06),(-104.45,72.8),
    (-103.6,72.5),(-103.0,71.5)])

def generate():
    # Integer tenth-degree construction avoids accumulated step errors.
    latitudes = (790 - np.arange(61, dtype=np.int32)) / 10.0
    longitudes = (-1050 + np.arange(241, dtype=np.int32)) / 10.0
    x, y = np.meshgrid(longitudes, latitudes)
    mask = np.zeros((61, 241), dtype=bool)
    for name, polygon in LAND:
        mask |= contains(polygon, x, y)
    for name, polygon in WATER:
        mask &= ~contains(polygon, x, y)
    matrix = mask.astype(np.uint8)
    assert matrix.shape == (61, 241) and matrix.size == 14701
    assert set(np.unique(matrix).tolist()) <= {0, 1}
    assert latitudes[0] == 79.0 and latitudes[-1] == 73.0
    assert longitudes[0] == -105.0 and longitudes[-1] == -81.0

    # All outputs are relative to the current working directory.
    output = Path('outputs')
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / 'candidate.csv'
    png_path = output / 'candidate.png'
    enlarged_path = output / 'candidate_8x.png'
    with csv_path.open('w', encoding='ascii', newline='') as handle:
        csv.writer(handle, lineterminator='\n').writerows(matrix.tolist())

    pixels = matrix * np.uint8(255)
    image = Image.fromarray(pixels)
    assert image.mode == 'L'
    image.save(png_path, format='PNG')
    nearest = getattr(Image, 'Resampling', Image).NEAREST
    enlarged = image.resize((1928, 488), resample=nearest)
    enlarged.save(enlarged_path, format='PNG')

    # Reopen the actual written artifacts, not just their in-memory sources.
    with csv_path.open('r', encoding='ascii', newline='') as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 61
    assert all(len(row) == 241 for row in rows)
    assert all(value in ('0', '1') for row in rows for value in row)
    restored_matrix = np.array(rows, dtype=np.uint8)
    assert restored_matrix.size == 14701
    assert np.array_equal(restored_matrix, matrix)
    with Image.open(png_path) as saved:
        assert saved.size == (241, 61) and saved.mode == 'L'
        restored_pixels = np.array(saved)
    assert set(np.unique(restored_pixels).tolist()) == {0, 255}
    assert np.array_equal(restored_pixels, pixels)
    with Image.open(enlarged_path) as saved:
        assert saved.size == (1928, 488) and saved.mode == 'L'
        restored_enlarged = np.array(saved)
    expected = np.repeat(np.repeat(pixels, 8, axis=0), 8, axis=1)
    assert set(np.unique(restored_enlarged).tolist()) == {0, 255}
    assert np.array_equal(restored_enlarged, expected)
    print(json.dumps({
        'validated': True,
        'matrix_shape': list(matrix.shape),
        'valid_values': int(matrix.size),
        'land_points': int(matrix.sum()),
        'water_points': int(matrix.size - matrix.sum()),
        'original_size': [241, 61],
        'enlarged_size': [1928, 488],
        'pixel_values': [0, 255],
        'nearest_neighbor_verified': True,
        'method': 'handwritten approximate polygons; vectorized point tests',
        'external_geographic_inputs': False,
        'files': [str(csv_path), str(png_path), str(enlarged_path)]
    }, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    generate()
