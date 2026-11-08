"""Knowledge-only approximation of land around Hecate Strait (53–55 N, 131–129 W).

No reference map or geographic database is accessed. Coordinates below are
hand-estimated vertices; straight segments approximate the shorelines.
"""
from pathlib import Path
import csv
import math
import numpy as np
from PIL import Image

# Each polygon is (longitude, latitude). The first polygon represents the
# eastern mainland; the rest are estimated islands, including Haida Gwaii's
# eastern Graham Island shore, Dundas, Porcher, Banks and Pitt.
polygons = [
    # Mainland: Portland Inlet, Prince Rupert/Skeena estuary, coastal fjords.
    [(-129.00,55.00),(-129.00,53.00),(-129.08,53.00),
     (-129.12,53.11),(-129.20,53.21),(-129.24,53.30),
     (-129.21,53.42),(-129.30,53.48),(-129.37,53.52),
     (-129.41,53.62),(-129.39,53.70),(-129.29,53.78),
     (-129.27,53.85),(-129.34,53.92),(-129.31,53.99),
     (-129.38,54.05),(-129.44,54.10),(-129.51,54.11),
     (-129.59,54.16),(-129.65,54.21),(-129.77,54.22),
     (-129.79,54.27),(-129.86,54.31),(-129.85,54.35),
     (-129.94,54.38),(-129.88,54.44),(-129.83,54.49),
     (-129.81,54.56),(-129.77,54.61),(-129.69,54.63),
     (-129.62,54.67),(-129.63,54.72),(-129.68,54.77),
     (-129.75,54.79),(-129.80,54.84),(-129.82,54.91),
     (-129.88,54.97),(-129.88,55.00)],
    # Graham Island: eastern shore in the western edge of the window.
    [(-131.00,54.37),(-130.96,54.36),(-130.92,54.33),
     (-130.89,54.28),(-130.91,54.24),(-130.95,54.20),
     (-130.99,54.17),(-131.00,54.14)],
    # Dundas Island.
    [(-130.90,54.62),(-130.83,54.67),(-130.73,54.69),
     (-130.65,54.68),(-130.62,54.63),(-130.65,54.59),
     (-130.69,54.55),(-130.76,54.52),(-130.83,54.53),
     (-130.89,54.57)],
    # Melville Island, plus nearby eastern islands in the Dixon Entrance.
    [(-130.67,54.52),(-130.61,54.55),(-130.54,54.53),
     (-130.48,54.49),(-130.51,54.45),(-130.59,54.45)],
    [(-130.54,54.62),(-130.49,54.66),(-130.44,54.64),
     (-130.47,54.59)],
    # Kaien Island (Prince Rupert).
    [(-130.42,54.34),(-130.37,54.37),(-130.30,54.37),
     (-130.25,54.32),(-130.28,54.27),(-130.36,54.28)],
    # Digby Island, west of Prince Rupert.
    [(-130.48,54.36),(-130.43,54.38),(-130.39,54.36),
     (-130.40,54.29),(-130.45,54.25),(-130.49,54.27)],
    # Porcher Island, to the south of Prince Rupert.
    [(-130.50,54.11),(-130.46,54.16),(-130.39,54.18),
     (-130.34,54.16),(-130.33,54.12),(-130.27,54.10),
     (-130.27,54.05),(-130.21,54.02),(-130.23,53.97),
     (-130.18,53.93),(-130.22,53.88),(-130.28,53.84),
     (-130.34,53.85),(-130.36,53.90),(-130.41,53.93),
     (-130.44,53.99),(-130.49,54.04)],
    # Banks Island, elongated northwest to southeast.
    [(-130.43,53.84),(-130.37,53.84),(-130.31,53.80),
     (-130.27,53.76),(-130.24,53.73),(-130.22,53.68),
     (-130.17,53.64),(-130.15,53.60),(-130.11,53.57),
     (-130.09,53.51),(-130.04,53.47),(-130.04,53.42),
     (-130.00,53.38),(-130.05,53.35),(-130.10,53.38),
     (-130.15,53.43),(-130.20,53.47),(-130.22,53.53),
     (-130.28,53.58),(-130.30,53.65),(-130.35,53.68),
     (-130.40,53.74),(-130.46,53.78)],
    # Pitt Island, separated from Banks by Principe Channel.
    [(-130.03,53.82),(-129.96,53.87),(-129.89,53.87),
     (-129.84,53.83),(-129.81,53.77),(-129.79,53.72),
     (-129.75,53.67),(-129.75,53.62),(-129.70,53.57),
     (-129.70,53.50),(-129.65,53.45),(-129.66,53.39),
     (-129.72,53.35),(-129.78,53.38),(-129.80,53.45),
     (-129.85,53.50),(-129.86,53.57),(-129.91,53.64),
     (-129.93,53.70),(-129.99,53.75)],
    # Coastal islands bordering eastern channels.
    [(-129.65,54.05),(-129.61,54.10),(-129.55,54.11),
     (-129.51,54.08),(-129.53,54.02),(-129.59,53.99)],
    [(-129.69,53.93),(-129.64,53.98),(-129.57,53.97),
     (-129.51,53.91),(-129.51,53.84),(-129.57,53.80),
     (-129.62,53.83)],
    [(-129.56,53.76),(-129.51,53.79),(-129.44,53.75),
     (-129.44,53.68),(-129.50,53.65),(-129.55,53.68)],
    [(-129.50,53.54),(-129.46,53.59),(-129.40,53.57),
     (-129.36,53.51),(-129.38,53.45),(-129.45,53.45)],
    # Aristazabal Island and southern channels.
    [(-129.99,53.19),(-129.96,53.25),(-129.91,53.29),
     (-129.85,53.27),(-129.82,53.21),(-129.76,53.17),
     (-129.70,53.11),(-129.67,53.05),(-129.71,53.01),
     (-129.78,53.03),(-129.83,53.08),(-129.90,53.11),
     (-129.95,53.14)],
]

def on_land(x, y, vertices):
    inside = False
    x0, y0 = vertices[-1]
    for x1, y1 in vertices:
        if (y1 > y) != (y0 > y):
            crossing = x1 + (y - y1) * (x0 - x1) / (y0 - y1)
            if x < crossing:
                inside = not inside
        x0, y0 = x1, y1
    return inside

def main():
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    matrix = np.array([[int(any(on_land(-131 + col / 63, 55 - row / 63, p)
                                for p in polygons))
                        for col in range(127)] for row in range(127)], dtype=np.uint8)
    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as file:
        csv.writer(file).writerows(matrix.tolist())
    image = Image.fromarray(matrix * 255, "L")
    image.save(out / "candidate.png")
    image.resize((1016, 1016), Image.Resampling.NEAREST).save(out / "candidate_8x.png")
    with (out / "candidate.csv").open(newline="", encoding="ascii") as file:
        readback = [row for row in csv.reader(file)]
    assert len(readback) == 127 and all(len(row) == 127 for row in readback)
    assert all(value in ("0", "1") for row in readback for value in row)
    assert set(image.getdata()) <= {0, 255}
    enlarged = Image.open(out / "candidate_8x.png")
    assert enlarged.size == (1016, 1016) and set(enlarged.getdata()) <= {0, 255}
    assert np.array_equal(np.asarray(enlarged)[::8, ::8], np.asarray(image))
    assert np.array_equal(np.asarray(enlarged), np.repeat(np.repeat(np.asarray(image), 8, 0), 8, 1))

if __name__ == "__main__":
    main()
