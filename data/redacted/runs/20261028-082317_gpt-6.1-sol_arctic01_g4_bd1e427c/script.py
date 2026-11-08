from pathlib import Path

import numpy as np
from PIL import Image


# Hand-drawn geographic approximations; coordinates are (longitude, latitude).
LAND = [
    # Ellesmere Island, including its northern part beyond the frame.
    [(-91.8, 76.0), (-90.0, 76.5), (-88.5, 77.0), (-86.8, 77.1),
     (-85.7, 77.5), (-84.4, 78.1), (-83.0, 78.5), (-81.0, 78.8),
     (-79.0, 79.2), (-78.0, 81.0), (-80.0, 82.7), (-84.5, 83.0),
     (-88.0, 82.0), (-90.5, 80.5), (-92.0, 78.5)],
    # Axel Heiberg Island.
    [(-96.0, 78.0), (-94.9, 78.3), (-94.0, 79.0), (-92.6, 79.4),
     (-91.4, 80.2), (-89.6, 80.0), (-88.5, 79.1), (-89.0, 78.1),
     (-90.0, 77.7), (-91.1, 77.8), (-92.3, 77.5), (-94.0, 77.4),
     (-95.1, 77.6)],
    # Devon Island.
    [(-89.0, 74.8), (-87.9, 75.1), (-86.9, 75.6), (-85.8, 75.9),
     (-84.5, 76.5), (-82.8, 76.7), (-81.0, 76.4), (-80.2, 75.8),
     (-81.5, 75.3), (-83.0, 75.0), (-84.3, 74.6), (-86.0, 74.5),
     (-87.5, 74.5)],
    # Bathurst Island.
    [(-98.5, 75.8), (-97.7, 76.1), (-96.5, 76.2), (-95.7, 76.0),
     (-95.0, 75.7), (-95.5, 75.3), (-96.6, 75.2), (-97.8, 75.4)],
    # Cornwallis Island.
    [(-96.8, 75.0), (-96.0, 75.2), (-95.0, 75.1), (-94.5, 74.8),
     (-95.1, 74.5), (-96.1, 74.5), (-96.8, 74.7)],

    # North Somerset Island.
    [(-95.5, 73.0), (-94.8, 73.4), (-93.8, 73.8), (-92.5, 74.0),
     (-91.0, 73.8), (-90.0, 73.4), (-90.5, 72.9), (-92.0, 72.7),
     (-93.5, 72.8), (-94.8, 72.7)],
    # Prince of Wales Island, only its northern end reaches the frame.
    [(-99.3, 73.0), (-98.7, 73.4), (-97.7, 73.7), (-96.6, 73.6),
     (-95.6, 73.3), (-95.0, 72.9), (-96.0, 72.6), (-97.5, 72.5),
     (-98.6, 72.7)],
    # Baffin Island's western fringe at the eastern edge of this map.
    [(-81.1, 73.0), (-80.8, 73.7), (-80.5, 74.4), (-80.0, 75.1),
     (-79.2, 75.8), (-78.3, 76.0), (-77.5, 75.2), (-77.6, 74.3),
     (-78.0, 73.5), (-78.1, 72.8)],
    # Smaller islands of the central Queen Elizabeth Islands.
    [(-101.0, 76.2), (-100.0, 76.6), (-98.9, 76.4), (-98.3, 75.9),
     (-99.1, 75.6), (-100.2, 75.7)],
    [(-103.0, 75.7), (-102.0, 76.0), (-101.4, 75.7), (-101.8, 75.3),
     (-102.8, 75.3)],
    [(-105.0, 75.1), (-104.0, 75.4), (-103.2, 75.2), (-103.6, 74.8),
     (-104.7, 74.8)],
    # Small islands in Norwegian Bay and the Sverdrup group.
    [(-93.0, 76.2), (-92.4, 76.5), (-91.7, 76.3), (-91.9, 75.9),
     (-92.7, 75.9)],
    [(-90.7, 76.8), (-90.0, 77.1), (-89.3, 76.9), (-89.7, 76.5),
     (-90.4, 76.5)],
    [(-87.3, 77.2), (-86.7, 77.4), (-86.1, 77.1), (-86.5, 76.8),
     (-87.1, 76.9)],
    # Small islands near the northern coast of Devon and Ellesmere.
    [(-82.7, 77.2), (-82.0, 77.5), (-81.3, 77.2), (-81.8, 76.9),
     (-82.4, 76.9)],
    [(-84.1, 76.9), (-83.7, 77.1), (-83.2, 76.9), (-83.5, 76.6),
     (-84.0, 76.6)],
]



def polygon_mask(lon_grid, lat_grid, polygon):
    """Return points inside a lon/lat polygon using an even-odd ray test."""
    inside = np.zeros(lon_grid.shape, dtype=bool)
    x0, y0 = polygon[-1]
    for x1, y1 in polygon:
        crosses = (y0 > lat_grid) != (y1 > lat_grid)
        x_intersect = (x1 - x0) * (lat_grid - y0) / (y1 - y0 + 1e-15) + x0
        inside ^= crosses & (lon_grid < x_intersect)
        x0, y0 = x1, y1
    return inside


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    latitudes = np.round(np.arange(79.0, 72.99, -0.1), 1)
    longitudes = np.round(np.arange(-105.0, -80.99, 0.1), 1)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    land = np.zeros((len(latitudes), len(longitudes)), dtype=bool)
    for polygon in LAND:
        land |= polygon_mask(lon_grid, lat_grid, polygon)
    matrix = land.astype(np.uint8)

    assert matrix.shape == (61, 241)
    assert int(matrix.size) == 14701
    np.savetxt(output_dir / "candidate.csv", matrix, fmt="%d", delimiter=",")

    raw = Image.fromarray(np.where(matrix == 1, 255, 0).astype(np.uint8), mode="L")
    raw.save(output_dir / "candidate.png")
    enlarged = raw.resize((1928, 488), resample=Image.Resampling.NEAREST)
    enlarged.save(output_dir / "candidate_8x.png")

    # Verify exported dimensions and strict binary contents.
    with (output_dir / "candidate.csv").open("r", encoding="ascii") as stream:
        rows = [line.rstrip("\n").split(",") for line in stream]
    assert len(rows) == 61 and all(len(row) == 241 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert raw.size == (241, 61) and enlarged.size == (1928, 488)
    assert set(raw.getdata()) == {0, 255}
    assert enlarged.resize(raw.size, Image.Resampling.NEAREST).tobytes() == raw.tobytes()


if __name__ == "__main__":
    main()
