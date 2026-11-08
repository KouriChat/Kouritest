"""Hand-drawn land mask approximation for the Canadian Arctic Archipelago.

Coordinates are WGS84 longitude/latitude, with west longitudes negative.
Polygons are deliberately approximate and contain no externally sourced data.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

OUT = Path("outputs")

# Each ring sketches one island or connected landmass. The polygons are
# hand-entered geographic approximations; points are tested, not cell areas.
LAND = [
    # Banks Island
    [(-125, 73), (-124, 75.1), (-123.4, 76.1), (-122.1, 76.9),
     (-120.5, 77.8), (-119, 78.2), (-117, 78.1), (-115, 77.5),
     (-114.3, 76.8), (-115.2, 75.6), (-117.1, 74.8), (-119.5, 73.7),
     (-121, 73), (-125, 73)],
    # Victoria Island
    [(-117.5, 73), (-117.5, 74.4), (-116, 75.3), (-115.4, 76.1),
     (-114.8, 76.9), (-113.5, 77.4), (-112, 78), (-109.5, 78.4),
     (-107.5, 78.3), (-106.3, 77.8), (-105, 77.6), (-103.5, 77.1),
     (-102.5, 76.3), (-103, 75.5), (-104.5, 74.8), (-106, 74.3),
     (-108, 73.9), (-110, 73.4), (-112, 73), (-117.5, 73)],
    # Prince of Wales Island
    [(-103.5, 73), (-103.4, 74.1), (-102.8, 75.2), (-101.5, 76),
     (-100, 76.8), (-98.2, 77.2), (-96.5, 76.9), (-95.8, 76.2),
     (-96.4, 75.5), (-97.5, 74.9), (-98, 73.9), (-98.8, 73),
     (-103.5, 73)],
    # Somerset Island
    [(-95.8, 73), (-95.7, 74), (-95.4, 75), (-94.7, 75.9),
     (-93.8, 76.6), (-92.7, 76.9), (-91.8, 76.4), (-91.3, 75.7),
     (-91.5, 74.8), (-92, 74), (-92.3, 73), (-95.8, 73)],
    # Boothia Peninsula and mainland north of the grid's lower edge
    [(-97.3, 73), (-97, 73.7), (-96.9, 74.6), (-96.4, 75.4),
     (-95.6, 76.2), (-94.8, 76.5), (-94, 76), (-93.5, 75),
     (-93, 74), (-92.6, 73), (-97.3, 73)],
    # King William Island
    [(-98.5, 69), (-97.7, 70), (-96.7, 70.8), (-95.4, 71.4),
     (-94.3, 71.8), (-93.7, 72.4), (-94.2, 73), (-97.5, 73),
     (-98.4, 72), (-99, 71), (-98.5, 69)],
    # Melville Island
    [(-114.5, 75.7), (-113, 76.5), (-111, 76.7), (-109.5, 76.3),
     (-108, 76.2), (-106.5, 76.5), (-104.7, 76.7), (-103.4, 76.3),
     (-103, 75.8), (-104.2, 75.3), (-106, 75), (-108, 74.9),
     (-110, 75.1), (-112, 75.2), (-114.5, 75.7)],
]
# Northern islands and the eastern edge of the sampled region
LAND += [
    # Ellesmere Island, including its southern coast in the map frame
    [(-94, 78.8), (-93, 79), (-90, 79), (-88, 78.7), (-86, 78.4),
     (-84, 78.4), (-82, 78.6), (-80.5, 79), (-79, 80), (-77, 82),
     (-75, 84), (-72, 86), (-68, 88), (-60, 90), (-100, 90),
     (-100, 80), (-97, 79.5), (-94, 78.8)],
    # Axel Heiberg Island
    [(-92.8, 78.8), (-92.2, 79.4), (-91.4, 80.3), (-90, 80.8),
     (-88.4, 80.6), (-87.5, 79.9), (-87.4, 79.1), (-88.4, 78.5),
     (-90, 78.4), (-91.5, 78.5), (-92.8, 78.8)],
    # Devon Island
    [(-89.5, 74.6), (-88.4, 75.3), (-87.3, 76), (-85.8, 76.5),
     (-84.3, 76.8), (-82.7, 76.6), (-81.8, 76), (-81.2, 75.4),
     (-80.8, 74.8), (-81.5, 74.4), (-83, 74.2), (-85, 74.1),
     (-87, 74.2), (-89.5, 74.6)],
    # Baffin Island at the eastern edge of the frame
    [(-81, 73), (-81, 79), (-78, 79), (-75, 78.5), (-73, 77),
     (-71, 75), (-69, 73), (-81, 73)],
    # Cornwallis Island
    [(-96.2, 74.3), (-95.7, 74.7), (-95, 74.9), (-94.3, 74.6),
     (-94, 74.1), (-94.4, 73.7), (-95.3, 73.6), (-96, 73.8),
     (-96.2, 74.3)],
    # Bathurst Island
    [(-99.3, 75.2), (-98.5, 75.8), (-97.4, 76.2), (-96.3, 76.1),
     (-95.6, 75.6), (-96.1, 75), (-97.2, 74.8), (-98.5, 74.9),
     (-99.3, 75.2)],
    # Prince Patrick Island
    [(-121, 76.1), (-120, 76.8), (-118.8, 77.2), (-117.5, 77),
     (-116.8, 76.5), (-117.8, 75.9), (-119.4, 75.6), (-121, 76.1)],
    # Eglinton Island
    [(-119.2, 74.6), (-118.4, 75), (-117.6, 74.8), (-117.4, 74.3),
     (-118.2, 74), (-119, 74.1), (-119.2, 74.6)],
    # King Christian Island
    [(-103.3, 77.7), (-102.7, 78.1), (-101.8, 78), (-101.4, 77.6),
     (-102.1, 77.3), (-102.9, 77.4), (-103.3, 77.7)],
]

# Tiny channels and bays remain water: only explicit polygon interiors count.
def inside(lon, lat, ring):
    """Even-odd point-in-polygon test."""
    hit = False
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat):
            crossing = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing:
                hit = not hit
        j = i
    return hit


def land_at(lon, lat):
    return int(any(inside(lon, lat, ring) for ring in LAND))
# The mainland enters the southern boundary around the Amundsen Gulf and
# Boothia Peninsula; its northern coastal fringe is sketched here.
LAND += [[
    (-125, 69), (-90, 69), (-90, 73.35), (-93, 73.5), (-95, 73.45),
    (-97, 73.3), (-99, 73.15), (-101, 73.12), (-103, 73.08),
    (-105, 73.1), (-107, 73.2), (-109, 73.35), (-112, 73.4),
    (-115, 73.2), (-118, 73), (-125, 73), (-125, 69)
]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lats = [79.0 - 0.1 * i for i in range(61)]
    lons = [-105.0 + 0.1 * j for j in range(241)]
    matrix = np.array(
        [[land_at(lon, lat) for lon in lons] for lat in lats],
        dtype=np.uint8,
    )

    with (OUT / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # 0 maps to black water and 1 to white land.
    image = Image.fromarray(matrix * 255, mode="L").convert("RGB")
    image.save(OUT / "candidate.png")
    enlarged = image.resize((241 * 8, 61 * 8), resample=Image.Resampling.NEAREST)
    enlarged.save(OUT / "candidate_8x.png")

    assert matrix.shape == (61, 241)
    assert int(matrix.size) == 14701 and set(np.unique(matrix)) <= {0, 1}
    assert image.size == (241, 61)
    assert set(image.getdata()) <= {(0, 0, 0), (255, 255, 255)}
    assert enlarged.size == (1928, 488)
    assert all(v in ("0", "1") for row in
               (OUT / "candidate.csv").read_text("ascii").splitlines()
               for v in row.split(","))
    print(f"Validated {matrix.shape[0]}x{matrix.shape[1]} mask; "
          f"{matrix.size} points; PNGs {image.size} and {enlarged.size}.")


if __name__ == "__main__":
    main()
