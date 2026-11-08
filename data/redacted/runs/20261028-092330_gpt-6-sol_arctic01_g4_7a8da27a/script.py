from pathlib import Path

import numpy as np
from PIL import Image


# Hand-authored approximate outlines, in (longitude, latitude).
POLYGONS = [
    [(-105, 76.9), (-103, 77.1), (-101, 76.8), (-100, 76.3),
     (-100.3, 75.7), (-102, 75.2), (-104, 75.3), (-105, 75.7)],
    [(-105, 73.8), (-104, 74.0), (-104, 74.4), (-105, 74.6)],
    [(-105, 73.6), (-104, 73.9), (-103, 74.1), (-102, 74.5),
     (-101, 74.8), (-100, 75.0), (-99.2, 75.1), (-98.7, 74.8),
     (-99.4, 74.3), (-100.5, 74.0), (-102, 73.7), (-103.5, 73.3),
     (-105, 73.2)],
    [(-101.0, 73.5), (-100.0, 74.1), (-99.3, 74.9), (-98.5, 75.4),
     (-97.5, 75.2), (-97.2, 74.5), (-97.8, 73.8), (-98.8, 73.3),
     (-100.0, 73.2)],
    [(-95.8, 72.9), (-95.4, 73.5), (-95.0, 74.4), (-94.5, 74.9),
     (-93.6, 74.9), (-92.8, 74.4), (-92.7, 73.7), (-93.2, 73.1),
     (-94.0, 72.8), (-95.0, 72.7)],
    [(-96.0, 73.0), (-95.7, 73.5), (-95.5, 73.9), (-94.8, 74.1),
     (-93.8, 73.9), (-92.7, 73.4), (-92.0, 73.0), (-91.3, 72.8),
     (-91.0, 72.5), (-92.0, 72.3), (-93.3, 72.4), (-94.7, 72.6)],
    [(-98.0, 73.0), (-97.7, 73.4), (-97.2, 73.6), (-96.4, 73.4),
     (-96.0, 73.0), (-96.4, 72.8), (-97.3, 72.8)],
    [(-98.0, 75.4), (-97.6, 75.7), (-97.0, 75.6), (-96.8, 75.2),
     (-97.2, 75.0), (-97.7, 75.1)],
    [(-99.8, 75.6), (-99.5, 76.1), (-98.9, 76.7), (-98.0, 77.0),
     (-96.9, 76.8), (-96.2, 76.3), (-96.5, 75.8), (-97.4, 75.5),
     (-98.5, 75.4), (-99.3, 75.3)],
    [(-96.7, 75.0), (-96.1, 75.5), (-95.3, 75.8), (-94.5, 75.7),
     (-94.2, 75.3), (-94.8, 75.0), (-95.8, 74.9)],
    [(-93.9, 74.7), (-93.6, 75.1), (-92.9, 75.5), (-92.5, 76.2),
     (-91.5, 76.6), (-90.3, 76.5), (-89.2, 76.2), (-88.0, 76.1),
     (-86.8, 76.0), (-85.5, 75.8), (-84.3, 75.4), (-83.2, 75.0),
     (-82.2, 74.8), (-81.0, 74.6), (-81.0, 76.5), (-83.0, 76.8),
     (-85.0, 77.0), (-88.0, 77.0), (-90.0, 76.9), (-92.0, 76.5),
     (-93.2, 75.9)],
    [(-91.0, 76.0), (-90.0, 76.5), (-89.8, 77.2), (-89.2, 77.8),
     (-88.2, 78.3), (-87.0, 78.7), (-85.5, 78.8), (-84.0, 78.7),
     (-82.5, 78.4), (-81.0, 78.0), (-81.0, 79.0), (-91.0, 79.0)],
    [(-92.2, 78.0), (-91.6, 78.5), (-90.7, 79.0), (-89.6, 79.4),
     (-88.2, 79.5), (-86.8, 79.3), (-86.0, 78.9), (-86.3, 78.5),
     (-87.4, 78.2), (-89.0, 78.0), (-90.7, 77.8)],
    [(-101.5, 77.0), (-101.0, 77.4), (-100.2, 77.5), (-99.8, 77.2),
     (-100.4, 76.9)],
    [(-98.4, 77.3), (-97.8, 77.7), (-97.0, 77.8), (-96.6, 77.4),
     (-97.2, 77.1)],
    [(-84.0, 76.8), (-83.4, 77.2), (-82.8, 77.3), (-82.3, 77.0),
     (-82.7, 76.7)],
]
def point_in_polygon(lon, lat, polygon):
    """Ray-casting point test for an authored outline."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > lat) != (yj > lat):
            crossing_x = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing_x:
                inside = not inside
        j = i
    return inside


def main():
    latitudes = np.round(np.arange(79.0, 72.9, -0.1), 1)
    longitudes = np.round(np.arange(-105.0, -80.9, 0.1), 1)
    matrix = np.zeros((len(latitudes), len(longitudes)), dtype=np.uint8)
    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            matrix[row, col] = any(
                point_in_polygon(float(lon), float(lat), polygon)
                for polygon in POLYGONS
            )

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        for row in matrix:
            f.write(",".join(map(str, row.tolist())) + "\n")

    pixels = np.repeat((matrix[:, :, None] * 255).astype(np.uint8), 3, axis=2)
    image = Image.fromarray(pixels, mode="RGB")
    image.save(out / "candidate.png")
    image.resize((241 * 8, 61 * 8), resample=Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )

    assert matrix.shape == (61, 241) and matrix.size == 14701
    assert set(np.unique(matrix).tolist()) <= {0, 1}
    assert image.size == (241, 61)
    assert set(image.getdata()) <= {(0, 0, 0), (255, 255, 255)}
    assert Image.open(out / "candidate_8x.png").size == (1928, 488)


if __name__ == "__main__":
    main()
