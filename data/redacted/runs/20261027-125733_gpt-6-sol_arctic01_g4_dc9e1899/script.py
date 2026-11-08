"""Hand-drawn, approximate Canadian Arctic land mask, in lon/lat degrees.

No geographic data are read. Shapes encode remembered island positions and
the main straits; boundary samples are treated as land.
"""
from pathlib import Path
import csv

import numpy as np
from PIL import Image


def poly(*points):
    return points


# Longitude is negative west; latitude increases northward.
# Northernmost western islands and southern fringe of Axel Heiberg.
ISLANDS = [
    # Ellef Ringnes, with the irregular shallow embayments on its south edge
    poly((-104.3, 78.15), (-104.8, 78.43), (-104.3, 78.75),
         (-103.5, 79.00), (-102.2, 79.11), (-101.0, 78.94),
         (-100.2, 78.75), (-99.7, 78.43), (-100.1, 78.08),
         (-100.9, 77.87), (-102.0, 77.94), (-102.8, 78.06)),
    # Amund Ringnes, smaller island to the east of Ellef Ringnes
    poly((-99.2, 78.24), (-99.35, 78.52), (-98.8, 78.80),
         (-97.7, 78.92), (-96.6, 78.77), (-96.2, 78.53),
         (-96.7, 78.28), (-97.6, 78.17), (-98.4, 78.13)),
    # Axel Heiberg southern portion, cropped by the northern edge
    poly((-95.9, 79.0), (-94.9, 78.93), (-94.5, 78.75),
         (-93.6, 78.48), (-93.1, 78.53), (-92.2, 78.69),
         (-91.1, 78.73), (-90.4, 78.95), (-89.1, 79.0)),
    # southwest Axel Heiberg peninsula
    poly((-95.0, 78.95), (-94.4, 78.81), (-94.0, 78.61),
         (-94.5, 78.47), (-95.2, 78.54), (-95.6, 78.70)),
    # western Ellesmere reaches the northeast of the sampling window
    poly((-86.9, 79.0), (-86.35, 78.82), (-85.4, 78.62),
         (-84.75, 78.57), (-84.14, 78.76), (-83.75, 78.62),
         (-83.58, 78.40), (-82.60, 78.46), (-82.10, 78.69),
         (-81.65, 78.76), (-81.0, 78.88), (-81.0, 79.0)),
    # Lougheed Island, west of Bathurst
    poly((-105.0, 77.57), (-104.5, 77.75), (-103.6, 77.83),
         (-102.9, 77.74), (-102.65, 77.49), (-103.4, 77.30),
         (-104.4, 77.32), (-105.0, 77.40)),
    # Mackenzie King Island edge at far west
    poly((-105.0, 77.0), (-104.65, 77.03), (-104.38, 76.91),
         (-104.8, 76.76), (-105.0, 76.79)),
    # Borden Island
    poly((-105.0, 78.05), (-104.75, 78.04), (-104.60, 77.92),
         (-105.0, 77.88)),
    # Meighen Island near the northern limit of the map
    poly((-99.9, 79.0), (-99.6, 78.91), (-98.95, 78.92),
         (-98.7, 79.0)),
    # Cornwall Island
    poly((-96.4, 77.58), (-95.8, 77.78), (-94.65, 77.84),
         (-94.0, 77.72), (-93.8, 77.53), (-94.2, 77.35),
         (-95.4, 77.33), (-96.2, 77.41)),
    # Graham Island
    poly((-92.0, 77.57), (-91.5, 77.71), (-90.5, 77.73),
         (-90.2, 77.54), (-90.65, 77.36), (-91.5, 77.34)),
    # Buckingham Island
    poly((-91.55, 78.1), (-91.2, 78.26), (-90.65, 78.17),
         (-90.74, 78.02), (-91.3, 77.99)),
    # North Kent Island
    poly((-91.9, 76.9), (-91.3, 77.09), (-90.5, 77.07),
         (-90.1, 76.90), (-90.5, 76.70), (-91.3, 76.73)),
    # South of Ellesmere, separated from Devon by Jones Sound
    poly((-84.80, 77.85), (-84.3, 77.95), (-83.8, 77.89),
         (-83.4, 77.61), (-82.8, 77.40), (-82.7, 77.15),
         (-81.8, 77.13), (-81.0, 77.31), (-81.0, 78.96),
         (-81.6, 78.90), (-82.2, 78.70), (-83.0, 78.48),
         (-83.7, 78.42), (-84.55, 78.48), (-85.15, 78.39),
         (-85.15, 78.12)),
]
# Bathurst Island: long north and south arms, with the deep eastern inlets.
ISLANDS += [
    poly((-104.3, 76.33), (-103.9, 76.68), (-103.1, 76.89),
         (-102.0, 76.88), (-101.45, 77.00), (-100.6, 76.96),
         (-100.16, 76.72), (-99.6, 76.77), (-98.75, 76.62),
         (-98.35, 76.35), (-98.85, 76.10), (-98.28, 75.97),
         (-97.92, 75.73), (-98.4, 75.51), (-97.96, 75.30),
         (-97.80, 75.08), (-98.5, 74.91), (-99.1, 74.99),
         (-99.35, 74.79), (-100.4, 74.65), (-101.5, 74.73),
         (-102.15, 74.91), (-102.55, 75.18), (-103.17, 75.33),
         (-103.6, 75.73), (-104.1, 75.85), (-104.56, 76.12)),
    # northwest Bathurst promontory
    poly((-103.65, 76.66), (-103.15, 76.98), (-102.6, 77.1),
         (-101.9, 77.05), (-101.63, 76.84), (-102.1, 76.67)),
    # Byam Martin Island
    poly((-105.0, 75.69), (-104.7, 75.92), (-103.96, 76.02),
         (-103.5, 75.85), (-103.61, 75.55), (-104.23, 75.37),
         (-104.9, 75.46), (-105.0, 75.52)),
    # Little Cornwallis Island west of Cornwallis
    poly((-98.14, 75.65), (-97.9, 75.86), (-97.27, 75.94),
         (-96.93, 75.77), (-97.09, 75.56), (-97.76, 75.49)),
    # Cornwallis Island
    poly((-97.0, 75.34), (-96.67, 75.6), (-95.95, 75.75),
         (-94.75, 75.78), (-94.15, 75.67), (-93.8, 75.41),
         (-93.5, 75.07), (-93.92, 74.91), (-93.77, 74.64),
         (-94.31, 74.47), (-94.95, 74.52), (-95.5, 74.32),
         (-96.18, 74.45), (-96.42, 74.73), (-96.90, 74.93)),
    # small islets in western Wellington Channel
    poly((-92.86, 75.63), (-92.7, 75.74), (-92.43, 75.69),
         (-92.40, 75.56), (-92.76, 75.54)),
    # Devon Island, with Grinnell Peninsula in the west and long east shore
    poly((-94.37, 76.04), (-93.7, 76.26), (-92.7, 76.33),
         (-91.8, 76.55), (-90.55, 76.66), (-89.8, 76.48),
         (-88.85, 76.51), (-87.88, 76.67), (-86.88, 76.60),
         (-86.18, 76.32), (-85.4, 76.27), (-84.70, 76.39),
         (-83.84, 76.32), (-83.27, 76.07), (-82.45, 76.04),
         (-81.66, 76.17), (-81.0, 76.33), (-81.0, 74.70),
         (-81.62, 74.61), (-82.46, 74.57), (-83.05, 74.77),
         (-83.60, 74.73), (-83.9, 74.58), (-84.51, 74.65),
         (-85.10, 74.54), (-86.02, 74.62), (-86.92, 74.41),
         (-88.12, 74.45), (-89.15, 74.37), (-90.26, 74.38),
         (-90.62, 74.60), (-91.56, 74.66), (-92.10, 74.90),
         (-92.66, 75.05), (-93.3, 75.03), (-93.58, 75.21),
         (-93.23, 75.49), (-93.7, 75.67), (-94.53, 75.72),
         (-95.05, 75.9)),
    # Dundas and other offshore land at eastern Devon fringe
    poly((-82.25, 76.45), (-81.8, 76.61), (-81.25, 76.55),
         (-81.0, 76.40), (-81.30, 76.26), (-81.98, 76.25)),
]
# Southern islands and bays of the Northwest Passage.
ISLANDS += [
    # eastern tip of Melville Island, westernmost edge of the sampled window
    poly((-105.0, 76.18), (-104.65, 76.05), (-104.53, 75.79),
         (-104.17, 75.62), (-104.31, 75.22), (-104.72, 74.91),
         (-105.0, 74.83)),
    # Prince of Wales Island
    poly((-105.0, 73.33), (-104.42, 73.45), (-103.75, 73.50),
         (-103.2, 73.68), (-102.20, 73.82), (-101.6, 73.92),
         (-100.7, 74.05), (-99.9, 74.03), (-99.43, 73.82),
         (-98.66, 73.72), (-98.0, 73.52), (-97.45, 73.18),
         (-97.22, 73.0), (-105.0, 73.0)),
    # northern promontory of Prince of Wales Island
    poly((-100.5, 73.83), (-100.05, 74.19), (-99.45, 74.35),
         (-98.95, 74.29), (-98.91, 74.07), (-99.35, 73.87)),
    # Russell Island
    poly((-99.0, 74.25), (-98.6, 74.43), (-98.02, 74.41),
         (-97.62, 74.28), (-97.86, 74.07), (-98.4, 74.05)),
    # Prescott Island near Peel Sound
    poly((-96.99, 73.84), (-96.72, 73.97), (-96.32, 73.92),
         (-96.21, 73.74), (-96.58, 73.65), (-96.93, 73.68)),
    # Somerset Island
    poly((-95.92, 73.0), (-96.12, 73.28), (-95.90, 73.62),
         (-95.44, 73.88), (-94.72, 74.08), (-93.92, 74.11),
         (-93.07, 74.03), (-92.64, 73.85), (-92.08, 73.77),
         (-91.55, 73.58), (-91.13, 73.43), (-90.66, 73.24),
         (-89.78, 73.06), (-89.46, 73.0)),
    # Baffin Island (Brodeur Peninsula) south of Lancaster Sound
    poly((-89.25, 73.0), (-88.75, 73.08), (-88.23, 73.23),
         (-87.45, 73.40), (-86.81, 73.54), (-86.14, 73.67),
         (-85.47, 73.78), (-84.91, 73.71), (-84.44, 73.57),
         (-84.04, 73.38), (-83.58, 73.29), (-83.13, 73.05),
         (-82.97, 73.0)),
]

# Subtract broad inlets where the hand-drawn island envelope crosses water.
INLETS = [
    # Bathurst: a deep inlet entering from the east
    poly((-97.5, 76.08), (-98.5, 76.09), (-99.35, 75.97),
         (-99.78, 75.81), (-99.43, 75.70), (-98.68, 75.78),
         (-97.5, 75.70)),
    poly((-97.6, 75.44), (-98.85, 75.48), (-99.44, 75.34),
         (-99.56, 75.20), (-98.69, 75.25), (-97.5, 75.18)),
    # Cornwallis: inlets cutting the southern rim
    poly((-95.29, 74.0), (-95.18, 74.51), (-94.96, 74.78),
         (-94.76, 74.77), (-94.75, 74.51), (-94.57, 74.0)),
    # Devon: Croker Bay entering the northern coast
    poly((-84.37, 77.0), (-84.35, 76.36), (-84.14, 76.12),
         (-83.83, 76.12), (-83.74, 76.31), (-83.66, 77.0)),
    # Devon: a southern bay
    poly((-88.9, 74.0), (-88.75, 74.49), (-88.33, 74.73),
         (-88.07, 74.72), (-88.15, 74.37), (-87.9, 74.0)),
    # deep indentation in northern Somerset
    poly((-93.62, 74.45), (-93.52, 73.96), (-93.37, 73.71),
         (-93.05, 73.60), (-92.90, 73.82), (-92.99, 74.45)),
]
def contains(vertices, x, y):
    """Even/odd polygon fill, with points exactly on an edge included."""
    inside = np.zeros(x.shape, dtype=bool)
    edge = np.zeros(x.shape, dtype=bool)
    for (ax, ay), (bx, by) in zip(vertices, vertices[1:] + vertices[:1]):
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        edge |= ((np.abs(cross) < 1e-9) &
                 (x >= min(ax, bx) - 1e-9) &
                 (x <= max(ax, bx) + 1e-9) &
                 (y >= min(ay, by) - 1e-9) &
                 (y <= max(ay, by) + 1e-9))
        if ay != by:
            inside ^= ((ay > y) != (by > y)) & (
                x < ax + (bx - ax) * (y - ay) / (by - ay))
    return inside | edge


def main():
    longitude, latitude = np.meshgrid(
        -105.0 + np.arange(241) / 10.0,
        79.0 - np.arange(61) / 10.0,
    )
    land = np.zeros((61, 241), dtype=bool)
    for outline in ISLANDS:
        land |= contains(outline, longitude, latitude)
    for inlet in INLETS:
        land &= ~contains(inlet, longitude, latitude)

    result_dir = Path("outputs")
    result_dir.mkdir(exist_ok=True)
    matrix = land.astype(np.uint8)
    with (result_dir / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f)
        writer.writerows(matrix.tolist())

    original = Image.fromarray(matrix * 255, mode="L")
    original.save(result_dir / "candidate.png")
    enlarged = original.resize((1928, 488), Image.Resampling.NEAREST)
    enlarged.save(result_dir / "candidate_8x.png")

    with (result_dir / "candidate.csv").open(newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 61 and all(len(row) == 241 for row in rows)
    assert all(cell in ("0", "1") for row in rows for cell in row)
    assert np.array_equal(np.asarray(rows, dtype=np.uint8), matrix)
    assert original.size == (241, 61) and enlarged.size == (1928, 488)
    assert set(original.getdata()) <= {0, 255}
    assert set(enlarged.getdata()) <= {0, 255}
    assert np.array_equal(np.asarray(enlarged),
                          np.repeat(np.repeat(np.asarray(original), 8, axis=0),
                                    8, axis=1))
    print(f"Validated {matrix.size} points ({int(matrix.sum())} land), "
          "61x241 CSV, 241x61 PNG, 1928x488 nearest-neighbor PNG.")


if __name__ == "__main__":
    main()
