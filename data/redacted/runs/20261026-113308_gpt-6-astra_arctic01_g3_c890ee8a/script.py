from pathlib import Path
import csv

import numpy as np
from PIL import Image


# Hand-drawn approximate island outlines in (latitude, longitude).
# These rules are deliberately approximate and use no external map data.
ISLANDS = [
    # Victoria Island
    [
        (74.8, -111.0), (74.9, -108.0), (74.8, -105.0), (74.3, -103.3),
        (73.8, -102.0), (73.2, -101.0), (72.5, -101.0), (71.8, -102.0),
        (71.2, -103.5), (70.6, -105.0), (70.5, -108.0), (71.0, -111.0),
        (72.0, -114.5), (73.0, -115.5), (74.0, -114.0),
    ],
    # Prince of Wales Island
    [
        (75.4, -100.5), (75.1, -98.5), (74.4, -97.3), (73.6, -97.0),
        (72.8, -97.5), (72.1, -98.5), (71.7, -100.0), (72.0, -101.5),
        (72.8, -102.7), (73.7, -102.8), (74.5, -102.0), (75.1, -101.5),
    ],
    # Somerset Island
    [
        (74.8, -95.8), (74.7, -94.2), (74.1, -93.2), (73.2, -93.4),
        (72.6, -94.3), (72.4, -95.4), (72.8, -96.2), (73.6, -96.4),
        (74.3, -96.2),
    ],
    # Boothia Peninsula
    [
        (72.7, -95.4), (72.4, -94.3), (71.8, -93.8), (71.2, -94.0),
        (70.4, -94.4), (69.5, -95.2), (68.8, -96.8), (68.7, -98.0),
        (69.1, -98.5), (70.0, -97.7), (71.0, -97.0), (72.0, -96.4),
    ],
    # Melville Island
    [
        (76.4, -113.5), (76.7, -111.0), (76.5, -108.2), (76.1, -106.0),
        (75.5, -105.7), (74.8, -106.7), (74.2, -108.4), (74.0, -110.8),
        (74.4, -113.2), (75.2, -114.2), (75.9, -114.0),
    ],
# Bathurst Island
    [
        (76.4, -101.9), (76.3, -99.8), (76.0, -97.8), (75.5, -96.7),
        (75.0, -97.0), (74.7, -98.2), (74.8, -100.0), (75.1, -101.6),
        (75.7, -102.5), (76.1, -102.4),
    ],
    # Cornwallis Island
    [
        (75.4, -96.7), (75.3, -94.7), (74.9, -93.6), (74.5, -93.7),
        (74.4, -95.0), (74.6, -96.3), (75.0, -96.8),
    ],
    # Devon Island
    [
        (78.4, -90.2), (78.6, -87.0), (78.4, -84.5), (77.8, -82.7),
        (77.0, -82.1), (76.2, -83.2), (75.3, -84.6), (74.8, -86.2),
        (75.0, -88.3), (75.8, -89.8), (76.7, -90.4), (77.6, -90.6),
    ],
    # Ellesmere Island, approximated to extend beyond the map bounds
    [
        (79.4, -83.0), (79.0, -81.0), (78.0, -80.0), (77.0, -79.0),
        (76.0, -78.0), (75.0, -77.0), (74.0, -76.0), (73.0, -75.0),
        (73.0, -70.0), (79.0, -70.0), (80.0, -83.0),
    ],
    # Axel Heiberg Island
    [
        (79.0, -92.5), (78.8, -90.5), (78.3, -88.5), (77.7, -87.6),
        (77.0, -88.3), (76.6, -89.8), (76.8, -91.4), (77.4, -92.3),
        (78.2, -93.0), (78.8, -93.0),
    ],
]
# Additional western Arctic islands and point-in-polygon routine.
ISLANDS.extend([
    # Amund Ringnes Island
    [
        (78.6, -99.0), (78.4, -97.4), (78.0, -96.2),
        (77.6, -96.5), (77.4, -98.0), (77.7, -99.3), (78.2, -99.7),
    ],
    # Ellef Ringnes Island
    [
        (78.9, -104.5), (78.7, -102.7), (78.3, -101.5),
        (77.8, -101.7), (77.4, -103.0), (77.6, -104.5), (78.2, -105.0),
    ],
    # Borden Island
    [
        (78.8, -112.0), (78.5, -110.8), (78.1, -111.0),
        (78.0, -112.0), (78.3, -112.6),
    ],
    # Lougheed Island
    [
        (77.9, -106.8), (77.7, -105.8), (77.4, -106.0),
        (77.3, -106.8), (77.6, -107.2),
    ],
    # Northwestern Baffin Island and Bylot Island
    [
        (73.9, -81.0), (74.5, -82.0), (74.8, -81.0), (75.0, -78.0),
        (73.0, -75.0), (72.0, -75.0), (72.0, -80.0), (73.0, -81.0),
    ],
])


def contains(lat, lon, polygon):
    inside = False
    j = len(polygon) - 1
    for i, (yi, xi) in enumerate(polygon):
        yj, xj = polygon[j]
        if (yi > lat) != (yj > lat):
            crossing_lon = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing_lon:
                inside = not inside
        j = i
    return inside
def main():
    latitudes = [79.0 - 0.1 * row for row in range(61)]
    longitudes = [-105.0 + 0.1 * col for col in range(241)]
    matrix = np.zeros((61, 241), dtype=np.uint8)

    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            matrix[row, col] = int(
                any(contains(lat, lon, polygon) for polygon in ISLANDS)
            )

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate.csv").open(
        "w", newline="", encoding="ascii"
    ) as file:
        csv.writer(file, lineterminator="\n").writerows(matrix.tolist())

    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(out / "candidate.png")
    image.resize((1928, 488), Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )

    assert matrix.shape == (61, 241) and matrix.size == 14701
    assert set(np.unique(matrix)).issubset({0, 1})


if __name__ == "__main__":
    main()
