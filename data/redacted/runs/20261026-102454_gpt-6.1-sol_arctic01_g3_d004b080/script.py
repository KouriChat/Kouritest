from pathlib import Path
import csv
import numpy as np
from PIL import Image

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

LATITUDES = [79.0 - 0.1 * i for i in range(61)]
LONGITUDES = [-105.0 + 0.1 * i for i in range(241)]


def inside(lon, lat, polygon):
    """Ray crossing test; points on an edge are treated as land."""
    hit = False
    j = len(polygon) - 1
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[j]
        cross = (y1 - lat) * (x2 - x1) - (x1 - lon) * (y2 - y1)
        if abs(cross) < 1e-10 and min(x1, x2) <= lon <= max(x1, x2) and min(y1, y2) <= lat <= max(y1, y2):
            return True
        if (y1 > lat) != (y2 > lat):
            at_x = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
            if lon < at_x:
                hit = not hit
        j = i
    return hit


def polygon(*points):
    return tuple(points)


# Hand-drawn outlines, in west-to-east map coordinates (longitude, latitude).
# These are deliberately coarse geographic knowledge rules, not downloaded data.
LAND = [
    # Eastern Victoria Island and its north-eastern shoulder.
    polygon((-105.0, 72.6), (-104.0, 73.3), (-103.0, 74.2),
            (-101.8, 74.5), (-100.4, 74.0), (-99.8, 72.9),
            (-100.2, 71.8), (-102.0, 71.2), (-104.0, 71.4)),
    # Prince of Wales Island.
    polygon((-103.0, 74.6), (-101.0, 75.1), (-98.5, 75.2),
            (-96.0, 74.7), (-94.0, 73.6), (-93.5, 72.4),
            (-94.3, 71.5), (-97.0, 71.3), (-99.5, 71.8),
            (-102.0, 72.7), (-103.5, 73.7)),
    # Somerset Island.
    polygon((-95.2, 74.2), (-93.5, 74.4), (-91.2, 73.8),
            (-89.3, 72.6), (-89.8, 71.6), (-91.8, 71.0),
            (-94.0, 71.4), (-95.0, 72.4)),
    # Melville Island, eastern part visible in this frame.
    polygon((-105.0, 75.2), (-103.5, 76.0), (-101.0, 76.2),
            (-98.2, 76.0), (-96.0, 75.0), (-95.0, 74.2),
            (-96.2, 73.7), (-99.0, 74.0), (-102.0, 74.5),
            (-104.5, 74.4)),
    # Boothia Peninsula and the northern Canadian mainland edge.
    polygon((-96.0, 73.0), (-95.0, 73.7), (-93.0, 73.9),
            (-91.5, 73.4), (-90.0, 72.5), (-89.0, 71.5),
            (-88.0, 70.0), (-89.0, 69.0), (-92.0, 68.6),
            (-96.0, 69.4), (-98.0, 70.8), (-98.0, 72.0)),
    # Western edge of Baffin Island at the far eastern edge of the frame.
    polygon((-87.0, 73.0), (-85.0, 73.5), (-82.0, 73.8),
            (-81.0, 73.0), (-81.0, 71.8), (-83.0, 70.8),
            (-86.0, 70.7), (-88.0, 71.7)),
]

# Major northern islands.
LAND += [
    # Banks Island, with its southern/eastern end entering the frame.
    polygon((-105.0, 77.7), (-103.8, 78.5), (-101.8, 78.8),
            (-99.2, 78.4), (-97.8, 77.4), (-98.5, 76.6),
            (-100.5, 76.0), (-102.8, 76.2), (-104.8, 76.8)),
    # North-western Victoria Island.
    polygon((-105.0, 78.8), (-103.5, 79.0), (-101.5, 79.0),
            (-100.5, 78.3), (-101.2, 77.6), (-103.0, 77.3),
            (-105.0, 77.5)),
    # Bathurst Island.
    polygon((-100.7, 77.9), (-99.5, 78.7), (-97.0, 78.9),
            (-95.0, 78.4), (-94.7, 77.5), (-96.1, 76.8),
            (-98.6, 76.7), (-100.1, 77.2)),
    # Cornwallis Island.
    polygon((-96.0, 75.9), (-94.8, 76.4), (-93.3, 76.4),
            (-92.5, 75.7), (-92.9, 75.0), (-94.8, 74.9),
            (-96.0, 75.2)),
    # Devon Island, large high-latitude landmass.
    polygon((-96.0, 78.0), (-94.3, 78.7), (-91.0, 78.8),
            (-87.0, 78.4), (-83.0, 77.8), (-82.0, 76.5),
            (-83.5, 75.4), (-86.0, 75.1), (-89.0, 75.4),
            (-92.0, 75.0), (-94.5, 75.5), (-96.0, 76.4)),
    # Ellesmere Island, western coast only within the requested box.
    polygon((-90.0, 79.0), (-86.5, 79.0), (-83.0, 78.8),
            (-81.0, 78.3), (-81.0, 76.7), (-82.0, 76.1),
            (-84.0, 76.3), (-86.0, 77.1), (-89.0, 77.4),
            (-90.5, 78.1)),
    # Axel Heiberg Island, narrow western portion in the box.
    polygon((-96.0, 79.0), (-93.0, 79.0), (-91.0, 78.5),
            (-91.5, 77.5), (-93.0, 77.2), (-95.0, 77.6)),
    # Small islands commonly defining the narrow channels.
    polygon((-99.2, 76.0), (-98.6, 76.3), (-97.8, 76.1), (-98.0, 75.7)),
    polygon((-91.0, 76.7), (-90.3, 77.0), (-89.7, 76.6), (-90.1, 76.2)),
    polygon((-87.0, 75.0), (-86.2, 75.4), (-85.5, 75.0), (-85.9, 74.6)),
]


def is_land(lon, lat):
    return int(any(inside(lon, lat, shape) for shape in LAND))


def build_matrix():
    return np.array(
        [[is_land(lon, lat) for lon in LONGITUDES] for lat in LATITUDES],
        dtype=np.uint8,
    )


def save_outputs(matrix):
    with (OUT / "candidate.csv").open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(matrix.tolist())

    raw = Image.fromarray(np.where(matrix == 1, 255, 0).astype(np.uint8), mode="L")
    raw.save(OUT / "candidate.png")
    enlarged = raw.resize((raw.width * 8, raw.height * 8), Image.Resampling.NEAREST)
    enlarged.save(OUT / "candidate_8x.png")


def validate(matrix):
    assert matrix.shape == (61, 241)
    assert matrix.size == 14701
    assert set(np.unique(matrix)) <= {0, 1}
    lines = (OUT / "candidate.csv").read_text(encoding="ascii").splitlines()
    assert len(lines) == 61
    assert all(len(line.split(",")) == 241 for line in lines)
    assert all(value in {"0", "1"} for line in lines for value in line.split(","))
    raw = Image.open(OUT / "candidate.png")
    enlarged = Image.open(OUT / "candidate_8x.png")
    assert raw.size == (241, 61)
    assert enlarged.size == (1928, 488)
    assert set(raw.getdata()) <= {0, 255}
    assert set(enlarged.getdata()) <= {0, 255}
    source = np.asarray(raw)
    scaled = np.asarray(enlarged)
    assert np.array_equal(scaled, np.repeat(np.repeat(source, 8, axis=0), 8, axis=1))


if __name__ == "__main__":
    result = build_matrix()
    save_outputs(result)
    validate(result)
    print("validated", result.shape, result.size)
