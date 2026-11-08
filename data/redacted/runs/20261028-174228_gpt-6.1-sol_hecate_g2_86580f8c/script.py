from pathlib import Path
import csv
import numpy as np
from PIL import Image

# Hand-estimated mainland outline and islands; x is degrees east from 131W.
LAND_POLYGONS = [
    [(1.48,55),(2,55),(2,53),(.98,53),(1.05,53.16),(1.20,53.30),
     (1.29,53.47),(1.18,53.61),(1.15,53.76),(1.28,53.88),
     (1.18,54),(1.04,54.12),(.92,54.23),(1.04,54.34),
     (1.24,54.42),(1.36,54.55),(1.31,54.68),(1.43,54.79),
     (1.39,54.89),(1.48,55)],
    [(1.67,53.29),(1.78,53.27),(1.91,53.31),(2,53.38),(2,53.79),
     (1.90,53.85),(1.78,53.83),(1.67,53.74),(1.61,53.59),
     (1.63,53.43),(1.67,53.29)],
    [(.88,54.57),(.94,54.55),(.99,54.59),(.98,54.64),(.92,54.65)],
    [(.70,54.37),(.77,54.34),(.82,54.38),(.80,54.43),(.74,54.44)],
    [(.57,54.26),(.62,54.24),(.66,54.28),(.63,54.32),(.58,54.31)],
    [(.19,53.59),(.27,53.57),(.32,53.61),(.31,53.67),(.25,53.70),(.20,53.66)],
    [(.37,53.53),(.42,53.52),(.45,53.56),(.42,53.59),(.38,53.58)],
]

def inside(x, y, polygon):
    result = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            if x < (xj-xi)*(y-yi)/(yj-yi)+xi:
                result = not result
        j = i
    return result
def main():
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    lats = np.linspace(55.0, 53.0, 127)
    lons_west = np.linspace(131.0, 129.0, 127)
    grid = np.zeros((127, 127), dtype=np.uint8)
    for r, lat in enumerate(lats):
        for c, lon in enumerate(lons_west):
            x = 131.0 - lon
            grid[r, c] = int(any(inside(x, float(lat), p)
                                 for p in LAND_POLYGONS))
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        csv.writer(f, lineterminator="\n").writerows(grid.tolist())
    image = Image.fromarray(grid * 255, mode="L")
    image.save(out / "candidate.png")
    image.resize((1016, 1016), Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )

if __name__ == "__main__":
    main()
