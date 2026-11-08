from pathlib import Path

import numpy as np
from PIL import Image


# Hand-drawn approximate coastlines in (longitude, latitude), west negative.
LAND_POLYGONS = [
    [(-113,73),(-112,73.7),(-110.5,74.3),(-108.5,74.6),(-106.8,74.4),
     (-105.3,73.9),(-104.7,73.3),(-105.1,72.7),(-106.3,72.5),
     (-108,72.7),(-110,72.5),(-112,72.7)],
    [(-103,72.8),(-102.8,73.5),(-102.1,74.1),(-101.5,74.8),
     (-100.7,75.2),(-99.5,75.1),(-98.6,74.7),(-98.2,74),
     (-98.5,73.3),(-99,72.7),(-100,72.4),(-101.2,72.4),(-102.2,72.5)],
    [(-96.4,72.6),(-96.1,73.2),(-96,74),(-95.6,74.7),(-94.9,75),
     (-94,74.8),(-93.6,74.2),(-93.8,73.6),(-94.1,73),(-94.6,72.6),
     (-95.5,72.4)],
    [(-97,73),(-96.3,73.4),(-95.6,73.6),(-94.8,73.3),(-94.2,72.8),
     (-93.5,72.5),(-93,72),(-93,70),(-97,70)],
    [(-113,75.4),(-112,75.9),(-111.5,76.5),(-110.4,76.9),(-108.8,76.9),
     (-107.8,76.5),(-106.4,76.4),(-105.5,75.9),(-106,75.4),
     (-107.5,75.2),(-109,75.3),(-110.4,75),(-112,75.1)],
    [(-102.8,75.3),(-102.5,75.9),(-102,76.5),(-101.5,77.2),(-100.5,77.5),
     (-99.4,77.3),(-98.6,76.9),(-98.2,76.3),(-98.5,75.8),(-99.2,75.5),
     (-100.2,75.3),(-101.3,75.1)],
    [(-96.2,74.8),(-95.8,75.1),(-95,75.2),(-94.4,75),(-94.3,74.7),
     (-94.8,74.5),(-95.6,74.5),(-96.1,74.6)],
    [(-90,74.5),(-89.5,75),(-88.8,75.7),(-88,76.3),(-86.8,76.9),
     (-85.5,77.2),(-84,77),(-82.8,76.7),(-81.5,76.3),(-80.5,75.9),
     (-80.2,75.3),(-81,74.8),(-82.2,74.6),(-84,74.5),(-86,74.4),(-88,74.3)],
    [(-82,76),(-81,76.5),(-80,77),(-79,77.7),(-78,78.5),(-77,79.2),
     (-75,79.8),(-73,80),(-70,80),(-70,73),(-81,73),(-81.8,74)],
    [(-91.5,77),(-90.8,77.7),(-89.8,78.4),(-88.5,79),(-87,79.5),
     (-85.8,79.3),(-84.9,78.8),(-85.5,78.2),(-86.5,77.7),
     (-87.6,77.3),(-89,77),(-90.5,76.8)],
    [(-100,77),(-99.2,77.5),(-98.5,78.2),(-97.5,78.7),(-96.2,78.8),
     (-95,78.4),(-94.2,77.9),(-94.5,77.4),(-95.5,77.1),(-97,76.9),(-98.5,76.8)],
    [(-98.5,78.5),(-98,79),(-97,79.5),(-95.8,79.7),(-94.8,79.3),
     (-94.5,78.8),(-95.2,78.4),(-96.4,78.2),(-97.5,78.2)],
    [(-112,77.3),(-111.3,77.8),(-110.3,78.1),(-109.4,77.9),
     (-109.2,77.4),(-110,77.1),(-111,77)],
    [(-112,78),(-111.5,78.6),(-110.5,79),(-109.3,79),(-108.7,78.5),
     (-109.4,78.1),(-110.6,77.9)],
    [(-108,78),(-107.5,78.6),(-106.5,78.8),(-105.7,78.4),
     (-106,77.9),(-107,77.7)],
    [(-104,76.8),(-103.5,77.2),(-102.8,77.3),(-102.4,76.9),
     (-102.8,76.6),(-103.6,76.6)],
    [(-103.5,77.8),(-103,78.2),(-102.2,78.3),(-101.8,77.9),
     (-102.2,77.6),(-103,77.5)],
    [(-92,75.4),(-91.6,75.7),(-91,75.7),(-90.8,75.4),(-91.2,75.2),(-91.8,75.2)],
    [(-81,73),(-80.2,73),(-80.2,73.5),(-80.5,74),(-81,74.3),(-81.2,73.7)],
]

def point_in_polygon(lon, lat, polygon):
    inside = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        cross = (lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
        if (
            abs(cross) < 1e-10
            and min(x1, x2) - 1e-10 <= lon <= max(x1, x2) + 1e-10
            and min(y1, y2) - 1e-10 <= lat <= max(y1, y2) + 1e-10
        ):
            return True
        if (y1 > lat) != (y2 > lat):
            crossing_lon = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < crossing_lon:
                inside = not inside
    return inside


def main():
    latitudes = np.round(np.arange(79.0, 72.9, -0.1), 1)
    longitudes = np.round(np.arange(-105.0, -80.9, 0.1), 1)
    matrix = np.zeros((len(latitudes), len(longitudes)), dtype=np.uint8)
    for row, lat in enumerate(latitudes):
        for col, lon in enumerate(longitudes):
            matrix[row, col] = int(
                any(point_in_polygon(lon, lat, polygon)
                    for polygon in LAND_POLYGONS)
            )
    if matrix.shape != (61, 241):
        raise RuntimeError(f"Unexpected matrix shape: {matrix.shape}")

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "candidate.csv").open(
        "w", encoding="ascii", newline=""
    ) as f:
        for row in matrix:
            f.write(",".join(str(int(value)) for value in row) + "\n")

    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(output_dir / "candidate.png")
    image.resize((1928, 488), resample=Image.Resampling.NEAREST).save(
        output_dir / "candidate_8x.png"
    )

    if not np.isin(matrix, (0, 1)).all() or matrix.size != 14701:
        raise RuntimeError("Matrix contains invalid values")
    with (output_dir / "candidate.csv").open("r", encoding="ascii") as f:
        rows = [line.rstrip("\n").split(",") for line in f]
    if len(rows) != 61 or any(
        len(row) != 241 or any(value not in ("0", "1") for value in row)
        for row in rows
    ):
        raise RuntimeError("CSV validation failed")

    original = Image.open(output_dir / "candidate.png")
    enlarged = Image.open(output_dir / "candidate_8x.png")
    if original.size != (241, 61) or enlarged.size != (1928, 488):
        raise RuntimeError("PNG dimensions are invalid")
    colors = set(map(tuple, np.unique(
        np.asarray(original).reshape(-1, 3), axis=0
    )))
    if colors != {(0, 0, 0), (255, 255, 255)}:
        raise RuntimeError("Original PNG contains unexpected colors")
    if not np.array_equal(
        np.asarray(enlarged),
        np.repeat(np.repeat(np.asarray(original), 8, axis=0), 8, axis=1),
    ):
        raise RuntimeError("Enlarged PNG is not nearest-neighbor scaled")
    print("Validated 14,701 values and all output dimensions and pixel colors.")


if __name__ == "__main__":
    main()
