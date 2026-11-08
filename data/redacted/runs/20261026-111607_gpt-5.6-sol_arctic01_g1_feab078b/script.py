from pathlib import Path
import csv
import numpy as np
from PIL import Image

# Hand-drawn approximate coastlines. Coordinates are (longitude, latitude).
POLYGONS = [
    # Ellesmere Island, including its northern and eastern reaches
    [(-91.8, 76.2), (-89.8, 77.0), (-89.2, 78.2), (-87.6, 79.0),
     (-86.5, 80.4), (-84.4, 81.3), (-82.7, 82.5), (-78.0, 83.0),
     (-75.1, 82.2), (-76.1, 80.8), (-77.8, 79.6), (-78.4, 78.5),
     (-79.3, 77.7), (-80.3, 76.8), (-82.5, 76.3), (-85.4, 76.0),
     (-88.2, 75.8), (-90.3, 75.9)],
    # Axel Heiberg Island
    [(-92.6, 78.0), (-91.4, 78.8), (-91.0, 80.1), (-89.6, 80.8),
     (-87.5, 80.4), (-86.8, 79.5), (-87.5, 78.3), (-89.1, 77.9),
     (-90.7, 77.7)],
    # Devon Island
    [(-90.9, 74.7), (-89.5, 75.3), (-87.3, 75.8), (-84.9, 75.9),
     (-83.0, 75.4), (-82.1, 74.7), (-83.0, 74.0), (-85.0, 73.8),
     (-87.6, 74.0), (-89.8, 74.1)],
    # Bathurst Island
    [(-97.4, 75.0), (-96.5, 75.8), (-95.0, 76.2), (-93.8, 75.8),
     (-93.0, 75.0), (-94.2, 74.6), (-96.2, 74.6)],
    # Cornwallis Island
    [(-96.5, 74.6), (-95.7, 75.0), (-94.7, 75.1), (-93.9, 74.7),
     (-94.5, 74.3), (-95.7, 74.3)],
    # Prince of Wales Island
    [(-101.1, 73.0), (-100.3, 74.2), (-99.8, 75.0), (-98.6, 75.6),
     (-97.2, 75.4), (-96.1, 74.7), (-96.5, 73.8), (-97.7, 73.1),
     (-99.1, 72.8), (-100.3, 72.8)],
]
POLYGONS += [
    # Victoria Island (only its eastern edge is inside this frame)
    [(-116.0, 72.0), (-113.0, 73.0), (-110.0, 74.0), (-108.0, 75.2),
     (-106.0, 75.8), (-104.5, 75.3), (-103.8, 74.1), (-104.5, 72.8),
     (-106.0, 71.5), (-110.0, 70.7), (-114.0, 71.0)],
    # Melville Island, clipping against the western frame
    [(-112.0, 74.2), (-109.0, 75.4), (-106.5, 76.0), (-104.0, 75.7),
     (-102.5, 75.1), (-103.5, 74.3), (-106.0, 73.9), (-109.0, 73.8)],
    # Somerset Island
    [(-95.1, 73.0), (-94.3, 74.0), (-93.7, 74.8), (-92.2, 74.9),
     (-91.1, 74.3), (-91.4, 73.4), (-92.5, 72.8), (-94.0, 72.7)],
    # Boothia Peninsula tip
    [(-95.8, 73.0), (-95.0, 73.5), (-94.4, 74.1), (-93.5, 74.2),
     (-92.8, 73.8), (-93.0, 73.0), (-94.0, 72.5), (-95.1, 72.4)],
    # Baffin Island's northwest coast at the eastern border
    [(-81.0, 72.0), (-79.8, 72.7), (-78.8, 73.6), (-77.5, 74.4),
     (-76.7, 75.3), (-75.6, 76.0), (-73.8, 76.4), (-72.0, 76.2),
     (-72.0, 71.0), (-81.0, 71.0)],
    # Small northern islands and coastal fragments
    [(-100.8, 75.8), (-100.1, 76.3), (-99.3, 76.2), (-99.0, 75.7),
     (-99.8, 75.5)],
    [(-93.0, 76.0), (-92.4, 76.4), (-91.7, 76.3), (-91.6, 75.9),
     (-92.3, 75.7)],
]
def contains(lon, lat, polygon):
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > lat) != (yj > lat):
            crossing = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing:
                inside = not inside
        j = i
    return inside


def main():
    latitudes = [79.0 - row / 10 for row in range(61)]
    longitudes = [-105.0 + col / 10 for col in range(241)]
    matrix = np.zeros((61, 241), dtype=np.uint8)
    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            matrix[row, col] = any(
                contains(lon, lat, polygon) for polygon in POLYGONS
            )

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # White land (1), black water (0); the array is already north-to-south.
    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(output_dir / "candidate.png")
    image.resize((241 * 8, 61 * 8), Image.Resampling.NEAREST).save(
        output_dir / "candidate_8x.png"
    )

    assert matrix.shape == (61, 241) and matrix.size == 14701
    assert set(np.unique(matrix)) <= {0, 1}
    assert Image.open(output_dir / "candidate.png").size == (241, 61)
    enlarged = Image.open(output_dir / "candidate_8x.png")
    assert enlarged.size == (1928, 488)
    assert set(enlarged.getdata()) <= {(0, 0, 0), (255, 255, 255)}
    print("Validated 61x241 matrix, CSV, binary PNG, and nearest-neighbor 8x PNG.")


if __name__ == "__main__":
    main()
