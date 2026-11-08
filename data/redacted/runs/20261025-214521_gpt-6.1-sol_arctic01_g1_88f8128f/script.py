"""Memory-based Arctic reconstruction; no external geographic data."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

# Approximate, handwritten island outlines: (longitude, latitude).
# These are rules for automated point classification, not surveyed coasts.
LAND = [
    # Bathurst Island: narrow southern body and broad northern peninsulas.
    [(-100.8,75.0),(-99.6,74.8),(-98.1,75.0),(-97.6,75.5),
     (-97.5,75.9),(-98.0,76.2),(-97.7,76.5),(-97.8,76.9),
     (-98.5,77.1),(-99.0,76.9),(-99.5,76.6),(-100.0,76.5),
     (-100.2,76.1),(-100.8,76.0),(-101.0,75.6),(-100.8,75.0)],
    # Cornwallis Island.
    [(-96.7,74.8),(-95.6,74.65),(-94.6,74.7),(-94.2,75.0),
     (-93.5,75.2),(-93.4,75.6),(-94.0,75.8),(-94.7,76.0),
     (-95.7,75.9),(-96.3,75.6),(-96.6,75.2)],
    # Devon Island; channels and fjords may be subtracted in later rules.
    [(-92.0,74.6),(-91.0,74.5),(-89.5,74.5),(-88.5,74.5),
     (-87.8,74.6),(-86.5,74.5),(-85.0,74.5),(-83.5,74.6),
     (-82.0,74.6),(-80.0,74.5),(-79.0,75.0),(-80.0,75.8),
     (-81.5,76.1),(-83.0,76.4),(-85.0,76.55),(-87.0,76.6),
     (-89.0,76.5),(-90.0,76.3),(-91.0,76.1),(-92.0,75.7),
     (-92.3,75.3),(-92.0,74.6)],
]


def contains(x, y, polygon):
    """Ray-crossing test at the sample point, without cell averaging."""
    inside = False
    for i, (ax, ay) in enumerate(polygon):
        bx, by = polygon[(i + 1) % len(polygon)]
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if (abs(cross) < 1e-10
                and min(ax, bx) - 1e-10 <= x <= max(ax, bx) + 1e-10
                and min(ay, by) - 1e-10 <= y <= max(ay, by) + 1e-10):
            return True
        if (ay > y) != (by > y):
            if x < ax + (y - ay) * (bx - ax) / (by - ay):
                inside = not inside
    return inside
LAND += [
    # Southern Ellesmere, continuing beyond the northern/eastern frame.
    [(-89.8,76.55),(-88.7,76.35),(-87.4,76.45),(-86.5,76.3),
     (-85.0,76.35),(-84.0,76.5),(-82.8,76.45),(-81.2,76.25),
     (-79.0,76.2),(-77.0,77.0),(-77.0,80.0),(-87.5,80.0),
     (-88.5,79.4),(-87.9,79.0),(-86.8,78.6),(-87.0,78.2),
     (-88.0,78.0),(-88.5,77.6),(-88.0,77.2),(-89.0,76.9)],
    # Axel Heiberg, with its southern peninsulas.
    [(-96.3,80.0),(-89.0,80.0),(-89.5,79.5),(-90.0,79.0),
     (-91.0,78.7),(-91.5,78.4),(-92.7,78.2),(-93.4,78.1),
     (-94.5,78.3),(-95.1,78.7),(-96.0,79.0),(-95.5,79.4)],
    # Ellef Ringnes.
    [(-103.9,78.1),(-103.3,77.8),(-102.5,77.6),(-101.7,77.3),
     (-100.5,77.4),(-99.5,77.7),(-99.3,78.0),(-100.0,78.3),
     (-100.5,78.7),(-101.6,78.8),(-102.5,78.5),(-103.6,78.4)],
    # Amund Ringnes.
    [(-98.8,78.0),(-98.0,77.8),(-97.0,77.8),(-96.2,78.0),
     (-95.6,78.3),(-96.1,78.6),(-97.2,78.6),(-98.2,78.4)],
    # Cornwall Island.
    [(-96.2,77.6),(-95.4,77.5),(-94.2,77.5),(-93.2,77.6),
     (-93.0,77.8),(-94.1,77.95),(-95.5,77.9)],
    # Graham Island.
    [(-90.7,77.4),(-90.1,77.25),(-89.5,77.35),(-89.6,77.55),
     (-90.4,77.6)],
    # North Kent Island.
    [(-90.0,76.6),(-89.2,76.45),(-88.5,76.5),(-88.0,76.7),
     (-88.6,76.85),(-89.5,76.8)],
    # King Christian Island.
    [(-102.3,77.6),(-101.8,77.5),(-101.0,77.6),(-100.9,77.85),
     (-101.5,77.95),(-102.1,77.85)],
    # Lougheed group near the western boundary.
    [(-105.6,77.2),(-105.0,77.05),(-104.7,77.2),(-104.9,77.5),
     (-105.3,77.6),(-105.6,77.4)],
    # Eastern Melville Island.
    [(-109.0,74.8),(-107.0,74.8),(-105.6,75.0),(-105.1,75.3),
     (-104.6,75.5),(-104.8,75.8),(-105.6,76.0),(-105.8,76.5),
     (-106.5,76.9),(-108.0,77.0),(-109.0,76.5)],
    # Byam Martin Island.
    [(-104.2,75.05),(-103.5,75.1),(-103.1,75.35),(-103.2,75.65),
     (-103.6,75.8),(-104.05,75.6)],
]
# Refine Bathurst's broad western lobes and southern peninsula.
LAND[0] = [
    (-100.6,75.0),(-99.5,74.93),(-98.3,75.05),(-97.8,75.4),
    (-97.6,75.8),(-98.0,76.05),(-97.4,76.35),(-97.2,76.65),
    (-97.8,76.9),(-98.5,76.95),(-99.1,76.65),(-99.5,76.2),
    (-100.0,76.45),(-100.3,76.85),(-101.1,76.9),(-101.8,76.7),
    (-101.7,76.35),(-102.5,76.3),(-102.7,76.0),(-102.1,75.75),
    (-101.2,75.6),(-100.7,75.4),
]
LAND += [
    # Prince of Wales Island.
    [(-102.4,71.5),(-102.0,72.5),(-102.2,73.0),(-101.9,73.5),
     (-101.0,73.85),(-100.0,73.9),(-99.5,73.7),(-98.8,73.95),
     (-97.8,74.05),(-97.2,73.85),(-97.1,73.4),(-97.4,73.0),
     (-97.0,72.4),(-97.5,71.5),(-100.0,71.0)],
    # Somerset Island, bounded by Peel Sound and Prince Regent Inlet.
    [(-95.8,71.5),(-95.5,72.5),(-95.3,73.1),(-95.7,73.6),
     (-95.0,74.0),(-94.0,74.15),(-92.8,74.13),(-91.6,74.0),
     (-90.5,73.9),(-90.7,73.5),(-91.4,73.15),(-91.8,72.7),
     (-92.5,72.3),(-92.0,71.8),(-93.0,71.5)],
    # Northern Baffin: Brodeur and Borden peninsulas.
    [(-89.5,72.0),(-89.2,72.65),(-88.5,73.1),(-87.5,73.6),
     (-86.0,73.85),(-84.8,73.8),(-84.0,73.5),(-83.5,73.2),
     (-82.5,73.65),(-81.4,73.8),(-80.2,73.6),(-79.0,73.0),
     (-79.0,71.5),(-85.0,71.0)],
    # Victoria Island's northeastern tip (separated by M'Clintock Channel).
    [(-108.0,71.5),(-107.0,72.2),(-105.4,72.8),(-104.8,73.3),
     (-104.0,73.7),(-103.2,73.7),(-102.8,73.3),(-103.3,72.7),
     (-104.0,72.2),(-105.0,71.5)],
    # Russell Island, immediately north of Prince of Wales.
    [(-98.5,74.0),(-97.9,74.02),(-97.3,74.15),(-97.5,74.3),
     (-98.2,74.25)],
    # Griffith Island and nearby southern Cornwallis satellites.
    [(-95.8,74.45),(-95.3,74.4),(-94.9,74.5),(-95.1,74.65),
     (-95.6,74.65)],
    [(-96.7,74.6),(-96.3,74.55),(-96.1,74.7),(-96.5,74.8)],
    # Bathurst's western satellite islands.
    [(-104.5,76.35),(-104.1,76.25),(-103.6,76.35),(-103.7,76.6),
     (-104.2,76.65)],
    [(-104.0,76.05),(-103.5,75.95),(-102.9,76.0),(-103.0,76.2),
     (-103.5,76.25)],
    [(-103.1,75.75),(-102.6,75.7),(-102.2,75.8),(-102.5,76.0)],
]

# Water exclusions represent remembered inlets, not inland lakes.
WATER = [
    # Admiralty Inlet, separating the northern Baffin peninsulas.
    [(-84.9,74.1),(-84.0,74.1),(-84.1,73.55),(-84.5,73.1),
     (-85.0,72.7),(-85.5,72.3),(-85.7,72.4),(-85.1,73.1)],
]


def classify(lon, lat):
    return int(any(contains(lon, lat, p) for p in LAND)
               and not any(contains(lon, lat, p) for p in WATER))


def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    # Integer tenths avoid accumulated coordinate stepping errors.
    matrix = np.array([
        [classify(lon10 / 10, lat10 / 10)
         for lon10 in range(-1050, -809)]
        for lat10 in range(790, 729, -1)
    ], dtype=np.uint8)
    assert matrix.shape == (61, 241) and matrix.size == 14701
    assert set(np.unique(matrix)) <= {0, 1}

    with (out / "candidate.csv").open(
            "w", newline="", encoding="ascii") as handle:
        csv.writer(handle, lineterminator="\n").writerows(matrix.tolist())

    pixels = matrix * np.uint8(255)
    original = Image.fromarray(pixels)
    original.save(out / "candidate.png")
    nearest = getattr(Image, "Resampling", Image).NEAREST
    enlarged = original.resize((1928, 488), resample=nearest)
    enlarged.save(out / "candidate_8x.png")

    with (out / "candidate.csv").open(
            newline="", encoding="ascii") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 61
    assert all(len(row) == 241 for row in rows)
    assert all(v in ("0", "1") for row in rows for v in row)
    assert sum(map(len, rows)) == 14701
    assert np.array_equal(np.array(rows, dtype=np.uint8), matrix)

    with Image.open(out / "candidate.png") as image:
        assert image.size == (241, 61)
        small = np.array(image)
    with Image.open(out / "candidate_8x.png") as image:
        assert image.size == (1928, 488)
        big = np.array(image)
    assert set(np.unique(small)) == {0, 255}
    assert set(np.unique(big)) == {0, 255}
    assert np.array_equal(small, pixels)
    assert np.array_equal(big, small.repeat(8, axis=0).repeat(8, axis=1))
    print("Validated: 61x241, 14701 binary values; PNG 241x61 and 1928x488.")
    print("Method: handwritten approximate polygons, point-in-polygon rules.")


if __name__ == "__main__":
    main()
