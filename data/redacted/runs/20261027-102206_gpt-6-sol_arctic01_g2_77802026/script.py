"""Memory-only, approximate, point-sampled map of the Canadian Arctic islands.

No geographic input files or online services are consulted. Coordinates in the
hand-drawn outlines below are (degrees WEST, degrees NORTH). These approximate
outlines are not surveyed shoreline data and will have local inaccuracies.
"""
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def shape(*coordinates):
    """Record a single hand-estimated island/coast outline."""
    return coordinates


# The major islands, including the parts cut off by the sampling window.
# The repeated small indentations stand for large inlets, not individual bays.
LAND = [
    # Devon Island: a broad northern island bounded by Jones and Lancaster sounds.
    shape((97.0, 75.70), (96.3, 76.12), (95.5, 76.36),
          (94.9, 76.41), (94.5, 76.70), (93.5, 76.64),
          (93.1, 76.90), (91.7, 77.08), (90.0, 77.15),
          (88.8, 77.08), (87.4, 76.94), (86.4, 76.64),
          (85.6, 76.55), (85.3, 76.21), (84.5, 76.30),
          (83.7, 76.16), (82.8, 76.03), (82.2, 75.85),
          (81.1, 75.91), (80.0, 75.62), (79.8, 74.63),
          (81.4, 74.63), (82.5, 74.51), (83.8, 74.59),
          (84.9, 74.50), (86.1, 74.69), (87.2, 74.59),
          (87.9, 74.48), (88.9, 74.54), (89.8, 74.43),
          (90.9, 74.48), (91.9, 74.67), (92.4, 74.87),
          (93.3, 74.85), (94.1, 75.07), (94.4, 75.43),
          (95.2, 75.58), (95.8, 75.43), (96.6, 75.54)),

    # Ellesmere Island: its southwestern coast along Eureka Sound and Jones Sound.
    shape((91.6, 80.0), (89.0, 80.0), (86.7, 79.80),
          (85.5, 79.56), (84.6, 79.27), (84.4, 78.99),
          (85.0, 78.83), (85.7, 78.51), (85.4, 78.24),
          (85.9, 77.96), (85.6, 77.70), (86.0, 77.43),
          (85.0, 77.19), (85.1, 76.99), (84.7, 76.75),
          (83.8, 76.63), (83.3, 76.40), (82.8, 76.32),
          (82.0, 76.43), (81.1, 76.31), (80.0, 76.33),
          (80.0, 80.0)),

    # Axel Heiberg Island, separated from Ellesmere by Eureka Sound.
    shape((98.2, 80.0), (86.4, 80.0), (86.6, 79.62),
          (87.3, 79.36), (88.3, 79.25), (88.6, 79.01),
          (89.2, 78.91), (89.0, 78.71), (89.8, 78.55),
          (90.0, 78.30), (90.6, 78.04), (91.2, 78.02),
          (91.3, 78.32), (92.0, 78.46), (92.4, 78.74),
          (93.2, 78.81), (93.6, 79.00), (94.2, 79.04),
          (94.1, 79.31), (94.9, 79.39), (95.2, 79.67),
          (96.4, 79.73), (97.5, 79.94)),

    # Ellef Ringnes Island, extending beyond the north and west edges.
    shape((106.5, 79.95), (102.6, 80.0), (100.9, 79.79),
          (100.1, 79.66), (99.5, 79.47), (99.6, 79.27),
          (99.1, 79.04), (99.6, 78.85), (100.1, 78.76),
          (101.0, 78.73), (101.6, 78.55), (102.3, 78.54),
          (103.0, 78.66), (103.6, 78.60), (104.3, 78.79),
          (104.8, 79.07), (105.4, 79.15), (106.4, 79.45)),

    # Amund Ringnes Island.
    shape((99.2, 78.63), (98.5, 78.83), (97.4, 78.86),
          (96.5, 78.82), (96.0, 78.72), (95.5, 78.78),
          (95.0, 78.57), (94.6, 78.41), (94.7, 78.15),
          (95.2, 78.03), (96.0, 77.98), (96.4, 78.05),
          (97.1, 78.03), (97.9, 78.12), (98.6, 78.25),
          (99.0, 78.42)),

    # Bathurst Island: an irregular coastline and an eastern peninsula.
    shape((104.8, 76.16), (104.2, 76.25), (103.9, 76.52),
          (103.1, 76.55), (102.8, 76.82), (102.0, 76.96),
          (101.0, 76.91), (100.7, 77.12), (99.9, 77.19),
          (99.4, 77.05), (98.8, 77.15), (98.5, 76.89),
          (97.9, 76.76), (97.6, 76.50), (97.2, 76.39),
          (97.3, 76.13), (97.9, 76.04), (98.1, 75.82),
          (98.8, 75.75), (99.5, 75.53), (100.1, 75.59),
          (100.5, 75.40), (101.4, 75.48), (101.7, 75.72),
          (102.7, 75.71), (103.0, 75.91), (104.0, 75.91)),
    # Cornwallis Island, between Wellington Channel and the western straits.
    shape((97.1, 75.75), (96.4, 75.93), (95.4, 75.90),
          (94.8, 75.82), (94.2, 75.65), (93.6, 75.66),
          (93.3, 75.45), (93.2, 75.26), (93.6, 75.14),
          (93.7, 74.90), (94.3, 74.77), (94.8, 74.54),
          (95.8, 74.54), (96.3, 74.70), (96.3, 74.98),
          (96.8, 75.15), (97.1, 75.43)),

    # Somerset Island south of Lancaster Sound and west of Prince Regent Inlet.
    shape((95.1, 74.35), (94.0, 74.28), (93.2, 74.17),
          (92.4, 74.15), (91.3, 74.02), (90.9, 73.76),
          (90.6, 73.68), (90.8, 73.38), (90.4, 73.19),
          (90.5, 72.8), (95.7, 72.8), (95.9, 73.18),
          (96.3, 73.39), (96.2, 73.70), (96.5, 73.92),
          (96.1, 74.18)),

    # Prince of Wales Island, with its northern coast in the sampling region.
    shape((102.4, 73.22), (101.7, 73.45), (100.9, 73.58),
          (100.0, 73.64), (99.3, 73.77), (98.6, 73.88),
          (97.6, 73.98), (96.9, 73.86), (96.4, 73.47),
          (96.2, 73.13), (96.1, 72.7), (103.0, 72.7)),

    # Easternmost part of Melville Island and a few western passages.
    shape((106.0, 75.88), (105.0, 75.85), (104.9, 75.56),
          (104.6, 75.39), (104.9, 75.21), (104.7, 74.99),
          (104.9, 74.80), (105.6, 74.69), (106.0, 74.68)),
    # Byam Martin Island.
    shape((105.4, 75.88), (104.4, 75.91), (104.1, 75.70),
          (104.3, 75.48), (104.8, 75.52), (105.4, 75.71)),

    # King Christian Island, in the Sverdrup Islands.
    shape((103.9, 78.08), (102.9, 78.14), (102.2, 78.07),
          (101.8, 77.93), (101.9, 77.74), (102.5, 77.61),
          (103.2, 77.65), (103.9, 77.83)),
    # Cornwall Island and associated smaller islands to its south.
    shape((95.7, 77.93), (94.9, 78.03), (94.0, 77.95),
          (93.8, 77.76), (94.0, 77.53), (94.8, 77.48),
          (95.5, 77.62)),
    # Graham Island, north of Devon.
    shape((92.2, 77.72), (91.4, 77.82), (90.6, 77.74),
          (90.3, 77.56), (90.8, 77.44), (91.7, 77.46)),
    # North Kent Island in Norwegian Bay.
    shape((91.7, 77.24), (90.7, 77.29), (90.1, 77.16),
          (90.4, 77.01), (91.2, 76.99), (91.8, 77.09)),
    # Baillie-Hamilton Island in Jones Sound.
    shape((86.3, 76.58), (85.8, 76.73), (85.1, 76.74),
          (84.6, 76.62), (84.5, 76.48), (85.0, 76.38),
          (85.6, 76.38), (86.2, 76.44)),
    # Little Cornwallis Island.
    shape((96.9, 76.07), (96.3, 76.16), (95.9, 76.08),
          (95.9, 75.94), (96.5, 75.92)),
    # Helena Island and the neighbouring land north of Bathurst.
    shape((99.0, 77.45), (98.2, 77.53), (97.6, 77.50),
          (97.6, 77.31), (98.1, 77.23), (98.8, 77.29)),
    # Seymour Island, west of Bathurst's north end.
    shape((101.5, 77.47), (101.0, 77.54), (100.6, 77.46),
          (100.8, 77.33), (101.4, 77.33)),
    # Cameron Island near western Bathurst.
    shape((103.8, 77.29), (103.1, 77.33), (102.6, 77.21),
          (102.6, 77.04), (103.3, 76.91), (104.0, 77.02)),
    # Lougheed Island, on the west boundary of the northern passages.
    shape((105.9, 77.97), (105.0, 78.00), (104.5, 77.89),
          (104.2, 77.73), (104.5, 77.55), (105.9, 77.46)),
    # Stor Island west of Amund Ringnes.
    shape((100.4, 78.43), (99.9, 78.53), (99.5, 78.49),
          (99.5, 78.34), (99.9, 78.27), (100.4, 78.33)),
    # Several explicitly represented small islands within the sampled window.
    shape((97.8, 77.71), (97.3, 77.75), (97.0, 77.64),
          (97.3, 77.53), (97.8, 77.55)),
    shape((92.7, 77.48), (92.2, 77.53), (92.0, 77.41),
          (92.3, 77.34), (92.7, 77.36)),
    shape((89.6, 77.23), (89.2, 77.29), (88.9, 77.20),
          (89.0, 77.10), (89.5, 77.10)),
    shape((98.2, 74.48), (97.6, 74.54), (97.2, 74.48),
          (97.2, 74.33), (97.9, 74.31)),
]

def points_in_outline(west, north, vertices):
    """Vectorized even-odd point test of one hand-drawn polygon.

    A vertex's second coordinate is latitude; the first is positive longitude
    west. Boundary coincidences are assigned by the ordinary crossing rule.
    """
    inside = np.zeros(west.shape, dtype=bool)
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        x1, y1 = a
        x2, y2 = b
        if y1 == y2:
            continue
        crossing = ((y1 > north) != (y2 > north)) & (
            west < x1 + (north - y1) * (x2 - x1) / (y2 - y1)
        )
        inside ^= crossing
    return inside


def main():
    out = Path('outputs')
    out.mkdir(parents=True, exist_ok=True)

    # Exact tenths avoid floating-point drift in generating the coordinates.
    longitudes_west = np.arange(1050, 809, -1) / 10.0
    latitudes_north = np.arange(790, 729, -1) / 10.0
    west, north = np.meshgrid(longitudes_west, latitudes_north)
    land = np.zeros((61, 241), dtype=bool)
    for outline in LAND:
        land |= points_in_outline(west, north, outline)
    values = land.astype(np.uint8)

    csv_path = out / 'candidate.csv'
    small_path = out / 'candidate.png'
    large_path = out / 'candidate_8x.png'

    with csv_path.open('w', newline='', encoding='ascii') as file:
        csv.writer(file).writerows(values.tolist())
    image = Image.fromarray(values * 255, mode='L')
    image.save(small_path)
    image.resize((1928, 488), resample=Image.Resampling.NEAREST).save(large_path)

    # Check the saved artifacts, not just the intermediate arrays.
    with csv_path.open(newline='', encoding='ascii') as file:
        rows = list(csv.reader(file))
    assert len(rows) == 61
    assert all(len(row) == 241 and set(row) <= {'0', '1'} for row in rows)
    assert np.array_equal(np.array(rows, dtype=np.uint8), values)
    with Image.open(small_path) as file:
        check_small = np.asarray(file.convert('L'))
        assert file.size == (241, 61)
    with Image.open(large_path) as file:
        check_large = np.asarray(file.convert('L'))
        assert file.size == (1928, 488)
    assert set(np.unique(check_small)) <= {0, 255}
    assert set(np.unique(check_large)) <= {0, 255}
    assert np.array_equal(check_large,
                          np.repeat(np.repeat(check_small, 8, axis=0), 8, axis=1))
    print('Verified: 61 x 241 points, binary CSV and PNGs, nearest-neighbor 8x.')


if __name__ == '__main__':
    main()
