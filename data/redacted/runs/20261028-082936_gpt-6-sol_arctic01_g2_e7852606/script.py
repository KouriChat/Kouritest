from pathlib import Path
import csv
import numpy as np
from PIL import Image

# Hand-authored approximate coastline rings, coordinates are (longitude, latitude).
# Rings are deliberately simplified; no external geographic data is used.
RINGS = [
    # Ellesmere Island, southern and eastern coast in the sampling window
    [(-86,79.7),(-82.2,79.5),(-80.9,78.9),(-81.4,78.2),(-82.1,77.5),
     (-82.2,76.8),(-83.0,76.2),(-84.4,75.8),(-85.3,75.1),(-87.3,74.7),
     (-89.0,74.9),(-90.3,75.5),(-90.8,76.2),(-89.8,76.7),(-89.8,77.3),
     (-88.5,77.6),(-88.4,78.2),(-87.5,78.4),(-87.4,79.0),(-86,79.7)],
    # Axel Heiberg Island
    [(-92.8,79.5),(-90.8,79.2),(-89.5,78.7),(-89.3,78.0),(-90.0,77.2),
     (-91.1,76.6),(-92.6,76.5),(-93.7,77.0),(-94.2,77.8),(-94.0,78.6),
     (-92.8,79.5)],
    # Ellesmere's northwestern portion
    [(-96.8,79.4),(-94.9,79.6),(-93.5,79.2),(-94.1,78.6),(-95.5,78.3),
     (-96.9,78.6),(-98.0,79.0),(-96.8,79.4)],
    # Devon Island
    [(-85.0,75.6),(-82.5,75.1),(-81.1,74.5),(-82.0,73.7),(-84.0,73.3),
     (-87.1,73.4),(-89.2,73.8),(-90.0,74.3),(-88.2,74.9),(-86.8,75.3),
     (-85.0,75.6)],
    # Cornwallis Island and nearby larger islands
    [(-96.1,75.1),(-94.8,75.0),(-94.4,74.6),(-95.1,74.3),(-96.2,74.5),
     (-96.1,75.1)],
    [(-98.1,75.2),(-97.0,75.0),(-96.8,74.6),(-97.7,74.4),(-98.5,74.7),
     (-98.1,75.2)],
    # Somerset Island
    [(-96.7,74.9),(-94.7,74.8),(-93.3,74.2),(-92.5,73.4),(-93.2,72.9),
     (-95.8,73.0),(-97.2,73.6),(-97.5,74.3),(-96.7,74.9)],
    # Prince of Wales Island
    [(-101.8,73.9),(-100.1,73.7),(-98.8,73.2),(-98.0,72.5),(-98.7,71.8),
     (-100.2,71.6),(-101.7,72.1),(-102.4,72.8),(-101.8,73.9)],
    # King William Island
    [(-98.8,70.7),(-97.2,70.8),(-96.2,70.3),(-96.3,69.5),(-97.4,69.2),
     (-98.8,69.6),(-99.1,70.2),(-98.8,70.7)],
    # Boothia Peninsula, reaching the southern edge
    [(-95.0,73.0),(-93.4,72.7),(-91.6,71.9),(-90.7,71.0),(-90.8,70.0),
     (-91.8,69.4),(-93.0,69.5),(-94.0,70.4),(-95.0,71.2),(-96.1,71.8),
     (-95.0,73.0)],
    # Victoria Island eastern edge
    [(-105.0,72.9),(-103.4,72.8),(-101.7,72.4),(-101.0,71.6),(-101.6,70.8),
     (-103.2,70.5),(-105.0,70.9),(-106.2,71.6),(-105.0,72.9)],
    # Bylot Island
    [(-79.9,73.6),(-78.4,73.7),(-77.5,73.3),(-78.1,72.8),(-79.4,72.9),
     (-79.9,73.6)],
    # Baffin Island's small westernmost edge in the southeast corner
    [(-81.0,73.2),(-80.5,73.1),(-80.1,72.8),(-80.7,72.5),(-81.0,72.7),
     (-81.0,73.2)],
    # Bathurst Island
    [(-98.7,75.5),(-96.8,75.7),(-95.4,75.3),(-95.1,74.9),(-96.0,74.7),
     (-97.5,74.8),(-98.7,75.1),(-98.7,75.5)],
    # Borden Island
    [(-113.0,78.8),(-111.0,79.0),(-108.8,78.7),(-108.3,78.2),(-110.0,77.9),
     (-112.0,78.1),(-113.0,78.8)],
    # Mackenzie King Island (easternmost trace)
    [(-105.0,78.8),(-103.8,78.9),(-102.7,78.5),(-103.1,78.0),(-104.5,78.0),
     (-105.0,78.4),(-105.0,78.8)],
    # Eglinton Island
    [(-105.0,76.8),(-104.0,76.9),(-103.5,76.4),(-104.3,76.1),(-105.0,76.3),
     (-105.0,76.8)],
    # Small Sverdrup group island fragments represented at their approximate scale
    [(-101.7,78.0),(-100.8,78.1),(-100.3,77.8),(-101.0,77.5),(-101.7,77.7),
     (-101.7,78.0)],
    # North Somerset / Prince of Wales channel-side islands
    [(-99.2,73.7),(-98.7,73.8),(-98.4,73.4),(-98.9,73.2),(-99.2,73.7)],
]

def inside_ring(lon, lat, ring):
    """Even-odd point-in-polygon test."""
    inside = False
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat):
            cross = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < cross:
                inside = not inside
        j = i
    return inside

def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    lats = [79.0 - 0.1 * i for i in range(61)]
    lons = [-105.0 + 0.1 * j for j in range(241)]
    matrix = np.zeros((61, 241), dtype=np.uint8)
    for row, lat in enumerate(lats):
        for col, lon in enumerate(lons):
            matrix[row, col] = int(any(inside_ring(lon, lat, ring) for ring in RINGS))

    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(matrix.tolist())
    # Array value 1 is land, rendered white; water (0) is black.
    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(out / "candidate.png")
    image.resize((1928, 488), Image.Resampling.NEAREST).save(out / "candidate_8x.png")

    assert matrix.shape == (61, 241) and matrix.size == 14701
    assert set(np.unique(matrix)).issubset({0, 1})
    assert image.size == (241, 61)
    assert set(image.getdata()).issubset({(0, 0, 0), (255, 255, 255)})
    assert Image.open(out / "candidate_8x.png").size == (1928, 488)

if __name__ == "__main__":
    main()
