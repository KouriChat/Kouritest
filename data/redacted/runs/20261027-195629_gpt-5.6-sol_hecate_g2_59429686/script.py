"""Knowledge-based schematic land mask for Hecate Strait; no external data."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

N = 127
OUT = Path("outputs")

# Vertices are (longitude, latitude), in WGS84 decimal degrees.
# These are hand-estimated coastline/island outlines, not surveyed shorelines.
LAND = [
    # Mainland coast, including the inlets around the north coast of BC.
    [(-129.00, 53.00), (-129.00, 55.00), (-129.30, 55.00),
     (-129.38, 54.90), (-129.34, 54.80), (-129.48, 54.70),
     (-129.43, 54.61), (-129.56, 54.54), (-129.50, 54.44),
     (-129.65, 54.34), (-129.57, 54.24), (-129.78, 54.13),
     (-129.72, 54.02), (-129.84, 53.93), (-129.77, 53.84),
     (-129.91, 53.75), (-129.84, 53.66), (-129.96, 53.56),
     (-129.89, 53.47), (-130.02, 53.37), (-129.93, 53.28),
     (-130.02, 53.17), (-129.90, 53.07), (-129.92, 53.00)],
    # Graham Island: the western part lies beyond the frame; east-coast
    # outline includes the Masset and Skidegate inlet openings.
    [(-132.0, 55.0), (-131.0, 55.0), (-131.0, 54.88),
     (-130.88, 54.84), (-130.96, 54.78), (-130.88, 54.71),
     (-130.91, 54.63), (-130.84, 54.58), (-130.87, 54.51),
     (-130.79, 54.45), (-130.85, 54.39), (-130.77, 54.34),
     (-130.81, 54.28), (-130.70, 54.24), (-130.69, 54.18),
     (-130.77, 54.12), (-130.70, 54.05), (-130.81, 53.99),
     (-130.73, 53.93), (-130.80, 53.87), (-130.70, 53.82),
     (-130.76, 53.76), (-130.66, 53.70), (-130.69, 53.63),
     (-130.81, 53.59), (-130.91, 53.62), (-131.0, 53.70),
     (-131.0, 55.0)],
    # Moresby Island, with its eastern shore in the southwest corner.
    [(-132.0, 53.88), (-131.00, 53.88), (-131.00, 53.75),
     (-130.91, 53.69), (-130.84, 53.60), (-130.86, 53.50),
     (-130.93, 53.40), (-130.91, 53.28), (-131.00, 53.16),
     (-131.00, 53.00), (-132.0, 53.00)],
    # Banks Island.
    [(-130.18, 54.48), (-130.05, 54.45), (-129.93, 54.37),
     (-129.89, 54.24), (-129.92, 54.10), (-130.02, 53.99),
     (-130.16, 53.95), (-130.27, 54.01), (-130.32, 54.13),
     (-130.31, 54.28), (-130.27, 54.40)],
    # Dundas Island and nearby islands.
    [(-130.92, 54.66), (-130.83, 54.68), (-130.77, 54.63),
     (-130.79, 54.56), (-130.87, 54.54), (-130.94, 54.59)],
    [(-130.79, 54.71), (-130.73, 54.72), (-130.69, 54.68),
     (-130.71, 54.64), (-130.77, 54.65)],
    [(-130.73, 54.56), (-130.68, 54.58), (-130.64, 54.54),
     (-130.67, 54.50), (-130.72, 54.51)],
    # Aristazabal, Campania and smaller outer-coast islands.
    [(-129.92, 53.90), (-129.85, 53.91), (-129.80, 53.85),
     (-129.83, 53.78), (-129.90, 53.77), (-129.95, 53.83)],
    [(-129.78, 53.64), (-129.71, 53.67), (-129.65, 53.62),
     (-129.68, 53.54), (-129.75, 53.52), (-129.80, 53.57)],
    [(-130.05, 53.30), (-129.98, 53.34), (-129.91, 53.29),
     (-129.93, 53.22), (-130.01, 53.21), (-130.07, 53.25)],
]
def inside_polygon(x, y, polygon):
    """Vectorized ray crossing; each sample is classified independently."""
    hit = np.zeros(x.shape, dtype=bool)
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        crosses = (yi > y) != (yj > y)
        boundary_x = (xj - xi) * (y - yi) / ((yj - yi) or 1e-30) + xi
        hit ^= crosses & (x < boundary_x)
        j = i
    return hit


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # The stated sampling interval is exactly 2/126 degrees.
    lat = np.linspace(55.0, 53.0, N)
    lon = np.linspace(-131.0, -129.0, N)
    xx, yy = np.meshgrid(lon, lat)
    land = np.zeros((N, N), dtype=bool)
    for polygon in LAND:
        land |= inside_polygon(xx, yy, polygon)
    matrix = land.astype(np.uint8)

    with (OUT / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    pixels = np.where(land, 255, 0).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L").convert("RGB")
    image.save(OUT / "candidate.png")
    enlarged = image.resize((N * 8, N * 8), Image.Resampling.NEAREST)
    enlarged.save(OUT / "candidate_8x.png")

    # Validate file dimensions, binary values, and nearest-neighbor replication.
    assert matrix.shape == (127, 127) and int(matrix.size) == 16129
    rows = list(csv.reader((OUT / "candidate.csv").open(encoding="ascii")))
    assert len(rows) == 127 and all(len(r) == 127 for r in rows)
    assert all(v in ("0", "1") for r in rows for v in r)
    assert set(image.getdata()) <= {(0, 0, 0), (255, 255, 255)}
    assert enlarged.size == (1016, 1016)
    enlarged_pixels = np.asarray(enlarged)
    assert np.array_equal(enlarged_pixels[::8, ::8], np.asarray(image))
    assert np.all(enlarged_pixels == np.repeat(
        np.repeat(np.asarray(image), 8, axis=0), 8, axis=1))
    print("validated: 127x127 / 16,129 binary samples; PNG 127x127; "
          "nearest-neighbor PNG 1016x1016")


if __name__ == "__main__":
    main()
