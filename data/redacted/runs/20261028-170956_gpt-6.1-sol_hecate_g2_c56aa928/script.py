from pathlib import Path
import numpy as np
from PIL import Image

SIZE = 127
LAT_NORTH, LAT_SOUTH = 55.0, 53.0
LON_WEST, LON_EAST = -131.0, -129.0

# Hand-sketched approximate land polygons; coordinates are (longitude, latitude).
MAINLAND = [
    (-130.45,55.0),(-130.17,54.93),(-130.03,54.82),(-129.94,54.70),
    (-129.84,54.60),(-129.70,54.50),(-129.58,54.38),(-129.51,54.22),
    (-129.48,54.06),(-129.45,53.91),(-129.40,53.76),(-129.33,53.61),
    (-129.25,53.43),(-129.18,53.24),(-129.13,53.0),(-129.0,53.0),
    (-129.0,55.0),
]
NORTH_ISLAND = [
    (-131,54.96),(-130.83,54.94),(-130.68,54.91),(-130.53,54.88),
    (-130.39,54.86),(-130.25,54.83),(-130.13,54.78),(-130.07,54.71),
    (-130.13,54.65),(-130.22,54.66),(-130.34,54.70),(-130.47,54.73),
    (-130.60,54.76),(-130.75,54.79),(-130.91,54.81),(-131,54.82),
]
PRINCE_OF_WALES = [
    (-131,54.68),(-130.89,54.66),(-130.80,54.61),(-130.75,54.54),
    (-130.75,54.45),(-130.79,54.37),(-130.87,54.32),(-130.98,54.33),(-131,54.34),
]
DALL_ISLAND = [
    (-131,54.14),(-130.91,54.14),(-130.84,54.10),(-130.82,54.03),
    (-130.85,53.96),(-130.93,53.93),(-131,53.94),
]
GRAHAM_ISLAND = [
    (-131,54.0),(-130.84,53.99),(-130.67,53.98),(-130.49,53.97),
    (-130.31,53.96),(-130.14,53.94),(-130.0,53.91),(-129.87,53.88),
    (-129.77,53.84),(-129.70,53.79),(-129.68,53.72),(-129.70,53.65),
    (-129.75,53.58),(-129.83,53.50),(-129.91,53.43),(-130.01,53.37),
    (-130.13,53.33),(-130.28,53.32),(-130.43,53.34),(-130.57,53.39),
    (-130.70,53.46),(-130.81,53.54),(-130.91,53.63),(-131,53.70),
]
MORSESBY_ISLAND = [
    (-131,53.63),(-130.90,53.62),(-130.80,53.58),(-130.73,53.52),
    (-130.70,53.44),(-130.70,53.36),(-130.74,53.28),(-130.80,53.20),
    (-130.88,53.13),(-130.97,53.08),(-131,53.07),
]
SMALL_ISLANDS = [
    [(-130.42,54.61),(-130.36,54.62),(-130.32,54.58),(-130.34,54.53),
     (-130.40,54.53),(-130.44,54.57)],
    [(-130.57,54.48),(-130.51,54.49),(-130.47,54.45),(-130.49,54.40),
     (-130.55,54.40),(-130.59,54.44)],
    [(-130.24,54.30),(-130.18,54.31),(-130.14,54.27),(-130.16,54.22),
     (-130.22,54.22),(-130.26,54.26)],
]
POLYGONS = [
    MAINLAND, NORTH_ISLAND, PRINCE_OF_WALES, DALL_ISLAND,
    GRAHAM_ISLAND, MORSESBY_ISLAND, *SMALL_ISLANDS,
]


def inside(lon, lat, polygon):
    """Ray casting; points on polygon edges count as land."""
    hit = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        cross = (lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
        if abs(cross) < 1e-12 and min(x1, x2) <= lon <= max(x1, x2):
            if min(y1, y2) <= lat <= max(y1, y2):
                return True
        if (y1 > lat) != (y2 > lat):
            if lon < x1 + (lat-y1) * (x2-x1) / (y2-y1):
                hit = not hit
    return hit

latitudes = np.linspace(LAT_NORTH, LAT_SOUTH, SIZE)
longitudes = np.linspace(LON_WEST, LON_EAST, SIZE)
m = np.array([[int(any(inside(lon, lat, p) for p in POLYGONS))
               for lon in longitudes] for lat in latitudes], dtype=np.uint8)
out = Path("outputs")
out.mkdir(exist_ok=True)
np.savetxt(out / "candidate.csv", m, fmt="%d", delimiter=",")
im = Image.fromarray(m * 255, mode="L")
im.save(out / "candidate.png")
im.resize((1016, 1016), Image.Resampling.NEAREST).save(out / "candidate_8x.png")
print(m.shape, int(m.sum()), len((out / "candidate.csv").read_text().splitlines()))
