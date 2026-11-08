"""Knowledge-based, hand-drawn approximation of the Canadian Arctic Archipelago.

Coordinates below are (degrees WEST, degrees NORTH); no external geographic
files, services, coastlines, or land/water classifiers are used.
"""
from pathlib import Path
import csv

import numpy as np
from PIL import Image


lat = 79.0 - np.arange(61, dtype=float)[:, None] / 10
west = 105.0 - np.arange(241, dtype=float)[None, :] / 10
land = np.zeros((61, 241), dtype=bool)


def polygon(points):
    """Point-in-polygon test at the sampling points (not area averaging)."""
    pts = np.asarray(points, dtype=float)
    result = np.zeros((61, 241), dtype=bool)
    for (x1, y1), (x2, y2) in zip(pts, np.roll(pts, -1, axis=0)):
        if y1 == y2:
            continue
        crosses = ((y1 > lat) != (y2 > lat))
        xx = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
        result ^= crosses & (west < xx)
    return result


def add(*points):
    global land
    land |= polygon(points)


def remove(*points):
    global land
    land &= ~polygon(points)


def oval(x, y, rx, ry, theta=0.0):
    a = np.deg2rad(theta)
    dx, dy = west - x, lat - y
    xp = dx * np.cos(a) + dy * np.sin(a)
    yp = -dx * np.sin(a) + dy * np.cos(a)
    return (xp / rx) ** 2 + (yp / ry) ** 2 <= 1


def add_oval(x, y, rx, ry, theta=0):
    global land
    land |= oval(x, y, rx, ry, theta)


def remove_oval(x, y, rx, ry, theta=0):
    global land
    land &= ~oval(x, y, rx, ry, theta)


# Axel Heiberg Island: west coast across Norwegian Bay; the eastern shore
# faces Eureka Sound. Only its southern end and middle lie in this window.
add(
    (95.85, 79.20), (95.60, 79.02), (95.20, 78.88), (94.65, 78.82),
    (94.48, 78.63), (94.08, 78.59), (93.94, 78.40), (93.46, 78.31),
    (93.02, 78.20), (92.47, 78.14), (91.83, 78.08), (91.51, 78.17),
    (91.06, 78.14), (90.71, 78.26), (90.40, 78.23), (90.05, 78.34),
    (89.87, 78.47), (89.36, 78.58), (89.22, 78.72), (88.75, 78.84),
    (88.10, 79.04), (87.96, 79.25), (88.45, 79.34), (89.08, 79.46),
    (90.17, 79.47), (91.10, 79.55), (93.25, 79.59), (95.8, 79.5),
)
# A number of long, narrow head-of-fjord cuts in its southern coast.
remove(
    (93.40, 78.11), (93.31, 78.37), (93.20, 78.49),
    (92.99, 78.51), (93.09, 78.30), (93.07, 78.14),
)
remove(
    (91.80, 78.05), (91.92, 78.46), (91.80, 78.57),
    (91.59, 78.55), (91.69, 78.33), (91.57, 78.12),
)
remove(
    (90.20, 78.19), (90.27, 78.51), (90.20, 78.70),
    (89.98, 78.71), (90.06, 78.49), (89.94, 78.30),
)

# Ellef Ringnes: distinctly separate from Amund Ringnes farther east.
add(
    (105.4, 79.33), (103.89, 79.25), (103.29, 79.18),
    (102.76, 79.03), (102.14, 78.94), (101.68, 78.81),
    (101.51, 78.57), (101.76, 78.43), (102.35, 78.39),
    (102.86, 78.29), (103.30, 78.27), (103.83, 78.43),
    (104.23, 78.49), (104.37, 78.74), (104.88, 78.84),
    (105.4, 78.81),
)
remove_oval(102.45, 78.57, 0.16, 0.17)
remove_oval(103.85, 78.74, 0.13, 0.12)
# Amund Ringnes Island, separated from Axel Heiberg by open water.
add(
    (98.80, 78.61), (98.20, 78.70), (97.62, 78.69),
    (97.32, 78.59), (96.87, 78.65), (96.38, 78.59),
    (95.69, 78.48), (95.23, 78.32), (95.32, 78.12),
    (95.83, 78.00), (96.22, 77.91), (96.91, 77.95),
    (97.41, 77.89), (97.86, 78.00), (98.20, 78.13),
    (98.10, 78.30), (98.59, 78.37), (98.85, 78.50),
)
remove_oval(96.44, 78.13, 0.21, 0.15, 25)

# Cornwall Island (not Cornwallis), King Christian and neighbouring islets.
add(
    (96.49, 77.99), (95.72, 77.95), (95.16, 77.80),
    (94.75, 77.71), (94.55, 77.52), (94.95, 77.43),
    (95.36, 77.52), (95.97, 77.60), (96.52, 77.83),
)
add(
    (103.67, 78.02), (103.22, 77.98), (102.71, 77.87),
    (102.38, 77.75), (102.57, 77.48), (102.90, 77.38),
    (103.50, 77.42), (104.06, 77.59), (104.03, 77.78),
)
add_oval(101.62, 77.35, 0.27, 0.13, -20)
add_oval(100.44, 77.51, 0.29, 0.12, 15)
add_oval(98.71, 77.55, 0.18, 0.09)

# Southern Ellesmere Island. Deep embayments are cut separately below;
# this island continues beyond the northern and eastern image borders.
add(
    (87.20, 79.5), (87.04, 79.03), (86.43, 78.93),
    (86.45, 78.64), (86.01, 78.58), (85.86, 78.38),
    (86.04, 78.22), (86.50, 78.09), (86.85, 77.93),
    (87.03, 77.73), (86.71, 77.54), (86.42, 77.42),
    (85.95, 77.33), (85.71, 77.12), (85.28, 76.94),
    (84.74, 76.85), (84.43, 76.77), (84.07, 76.88),
    (83.72, 76.81), (83.26, 76.73), (82.92, 76.75),
    (82.52, 76.68), (82.04, 76.75), (81.62, 76.72),
    (80.5, 76.67), (80.5, 79.5),
)
# Narrow fjords from the open sea up the southwestern side of Ellesmere.
remove(
    (86.97, 77.67), (85.74, 77.85), (84.72, 78.14),
    (84.65, 78.25), (85.14, 78.13), (86.22, 77.86),
    (87.16, 77.82),
)
remove(
    (86.26, 77.31), (85.04, 77.61), (84.32, 77.66),
    (83.99, 77.77), (84.48, 77.72), (85.51, 77.72),
    (86.79, 77.56),
)
remove(
    (84.99, 76.85), (84.25, 77.07), (83.46, 77.32),
    (83.01, 77.65), (83.18, 77.68), (83.75, 77.34),
    (84.70, 77.01), (85.39, 76.90),
)
remove(
    (83.25, 76.55), (82.91, 76.79), (82.33, 77.18),
    (82.24, 77.34), (82.45, 77.29), (83.08, 76.97),
    (83.81, 76.82),
)
add(
    (91.60, 77.67), (91.16, 77.61), (90.62, 77.51),
    (90.21, 77.38), (90.25, 77.17), (90.75, 77.17),
    (91.43, 77.37), (91.82, 77.55),
)
add_oval(89.78, 77.40, 0.26, 0.14, 20)
add_oval(91.66, 76.80, 0.42, 0.13, 10)
add_oval(90.46, 76.93, 0.51, 0.20, 8)  # North Kent group
add_oval(88.99, 77.01, 0.20, 0.11)
add_oval(87.23, 76.60, 0.24, 0.11)
# Devon Island, between Jones Sound and Lancaster Sound. Numerous projections
# and bays are represented explicitly, rather than smoothing raster pixels.
add(
    (95.78, 75.83), (95.10, 75.87), (94.54, 75.80),
    (94.04, 75.92), (93.48, 75.86), (93.11, 75.69),
    (92.75, 75.79), (92.25, 75.98), (91.68, 76.06),
    (91.06, 76.00), (90.48, 75.94), (90.04, 76.05),
    (89.51, 76.17), (88.99, 76.15), (88.43, 76.29),
    (87.80, 76.34), (87.35, 76.22), (86.90, 76.16),
    (86.56, 76.20), (85.85, 76.35), (85.38, 76.29),
    (84.71, 76.24), (84.17, 76.39), (83.60, 76.40),
    (82.92, 76.38), (82.31, 76.39), (81.58, 76.42),
    (80.5, 76.44), (80.5, 74.26), (81.30, 74.23),
    (81.95, 74.20), (82.63, 74.09), (83.17, 74.12),
    (83.71, 74.19), (84.16, 74.23), (84.79, 74.18),
    (85.40, 74.13), (86.04, 74.17), (86.71, 74.15),
    (87.49, 74.20), (87.98, 74.22), (88.62, 74.26),
    (89.19, 74.30), (89.82, 74.27), (90.45, 74.30),
    (90.88, 74.33), (91.42, 74.37), (92.07, 74.39),
    (92.47, 74.50), (92.91, 74.59), (93.04, 74.73),
    (93.60, 74.75), (93.78, 74.90), (93.86, 75.03),
    (94.45, 75.13), (94.93, 75.24), (95.12, 75.34),
    (95.79, 75.42), (95.95, 75.56), (95.78, 75.70),
)
# Devon west coast's large, deep inlet from Wellington Channel.
remove(
    (96.08, 75.63), (95.09, 75.56), (94.19, 75.62),
    (93.49, 75.49), (93.41, 75.37), (94.13, 75.48),
    (94.95, 75.45), (96.01, 75.40),
)
# Northeast-going channels entering Devon's northern coastline.
remove(
    (90.84, 76.22), (90.67, 75.96), (90.55, 75.84),
    (90.30, 75.73), (90.15, 75.72), (90.31, 75.89),
    (90.37, 76.16),
)
remove(
    (87.54, 76.50), (87.62, 76.12), (87.48, 75.98),
    (87.27, 75.91), (87.39, 76.08), (87.23, 76.43),
)
remove(
    (85.22, 76.56), (85.18, 76.19), (85.41, 75.99),
    (85.29, 75.83), (85.09, 75.86), (85.13, 76.06),
    (84.86, 76.45),
)
remove(
    (83.90, 76.58), (83.70, 76.22), (83.19, 76.00),
    (82.79, 75.91), (82.91, 76.02), (83.50, 76.35),
    (83.53, 76.59),
)
# Indentations along the south side of Devon are small at this resolution.
remove(
    (91.73, 74.24), (91.44, 74.56), (91.14, 74.74),
    (91.20, 74.52), (91.39, 74.30),
)
remove(
    (86.33, 74.00), (86.22, 74.32), (86.05, 74.51),
    (85.99, 74.33), (86.13, 74.10),
)
remove(
    (83.24, 74.01), (83.05, 74.32), (82.91, 74.38),
    (82.94, 74.13),
)
# Coastal islands off Devon's northwest and western shores.
add_oval(92.76, 76.08, 0.39, 0.14, -8)
add_oval(94.13, 76.05, 0.44, 0.17, -8)
add_oval(94.84, 74.60, 0.37, 0.15, -7)  # Griffith Island
add_oval(93.44, 74.37, 0.14, 0.07)
# Bathurst Island: deeply divided northern coasts and smaller satellite islands.
add(
    (103.65, 76.31), (103.05, 76.39), (102.65, 76.51),
    (102.14, 76.56), (101.73, 76.81), (101.16, 76.92),
    (100.69, 76.89), (100.38, 76.98), (99.87, 76.99),
    (99.71, 76.79), (99.29, 76.73), (98.78, 76.84),
    (98.43, 76.80), (98.20, 76.63), (97.77, 76.60),
    (97.43, 76.46), (97.22, 76.30), (97.35, 76.15),
    (97.14, 75.97), (97.44, 75.79), (97.76, 75.69),
    (97.97, 75.53), (98.53, 75.45), (98.85, 75.30),
    (99.31, 75.27), (99.67, 75.30), (100.15, 75.35),
    (100.46, 75.30), (101.13, 75.43), (101.74, 75.47),
    (102.12, 75.60), (102.58, 75.69), (102.88, 75.86),
    (103.30, 76.01), (103.49, 76.12),
)
# Bathurst's inlets mostly enter from the north; leave the southern coast
# relatively continuous, as required by the island's broad core.
remove(
    (101.54, 77.03), (101.45, 76.68), (101.08, 76.47),
    (100.92, 76.28), (101.06, 76.27), (101.35, 76.47),
    (101.81, 76.62), (102.23, 76.90),
)
remove(
    (99.84, 77.04), (99.62, 76.70), (99.29, 76.62),
    (99.26, 76.41), (99.46, 76.48), (99.86, 76.70),
    (100.20, 77.04),
)
remove(
    (98.16, 76.89), (98.07, 76.60), (98.54, 76.42),
    (98.80, 76.28), (98.96, 76.30), (98.76, 76.47),
    (98.39, 76.68), (98.50, 76.85),
)
remove(
    (103.90, 76.00), (103.08, 75.97), (102.16, 75.91),
    (101.76, 75.99), (101.76, 76.06), (102.39, 76.00),
    (103.60, 76.16),
)
remove(
    (97.34, 75.88), (98.10, 76.07), (98.72, 76.04),
    (99.13, 76.20), (99.16, 76.29), (98.62, 76.17),
    (97.99, 76.17), (97.22, 76.04),
)
add_oval(104.41, 76.62, 0.59, 0.27, 35)  # Cameron Island
add_oval(103.83, 76.81, 0.25, 0.12, 15)
add_oval(96.62, 76.12, 0.32, 0.14, -8)
add_oval(99.50, 77.16, 0.28, 0.12)

# The eastern extremity of Melville Island, on the western image boundary;
# neighbouring Byam Martin Island lies farther south and east.
add(
    (105.5, 76.93), (104.99, 76.75), (104.92, 76.49),
    (104.65, 76.33), (104.79, 76.13), (104.54, 76.03),
    (104.62, 75.87), (104.27, 75.72), (104.19, 75.56),
    (104.44, 75.43), (104.72, 75.44), (105.5, 75.30),
)
add_oval(104.30, 75.23, 0.48, 0.20, 20)  # Byam Martin
add_oval(103.61, 75.19, 0.14, 0.08)

# Cornwallis Island, east of McDougall Sound, west of Wellington Channel.
add(
    (96.91, 75.64), (96.51, 75.60), (96.17, 75.53),
    (95.71, 75.55), (95.36, 75.45), (95.19, 75.34),
    (94.91, 75.32), (94.66, 75.19), (94.30, 75.07),
    (94.05, 74.92), (94.20, 74.71), (94.41, 74.50),
    (94.84, 74.29), (95.51, 74.18), (96.13, 74.22),
    (96.65, 74.31), (97.01, 74.44), (97.26, 74.65),
    (97.39, 74.94), (97.13, 75.10), (97.20, 75.33),
    (96.99, 75.42),
)
remove(
    (97.20, 75.29), (96.71, 75.13), (96.30, 75.05),
    (96.08, 74.96), (96.18, 75.11), (96.59, 75.25),
    (97.36, 75.40),
)
remove(
    (94.05, 74.86), (94.72, 74.97), (94.95, 75.07),
    (94.89, 75.16), (94.56, 75.05), (93.90, 75.03),
)
add_oval(96.82, 75.77, 0.35, 0.14)  # Little Cornwallis
add_oval(97.70, 75.35, 0.15, 0.09)
# Prince of Wales Island, immediately south of Viscount Melville Sound.
add(
    (101.38, 74.13), (100.85, 74.12), (100.46, 74.04),
    (99.96, 74.03), (99.32, 73.93), (98.73, 73.86),
    (98.29, 73.78), (98.04, 73.63), (97.74, 73.48),
    (97.33, 73.33), (97.19, 73.02), (96.96, 72.5),
    (102.40, 72.5), (102.81, 73.00), (102.94, 73.18),
    (103.30, 73.37), (103.37, 73.60), (102.94, 73.75),
    (102.43, 73.87), (102.00, 74.02),
)
# M'Clintock Channel indents the western shore. Minor inlets penetrate
# the northern coast, leaving the wide waterway north of Prince of Wales.
remove(
    (103.49, 73.67), (102.87, 73.68), (102.48, 73.48),
    (102.31, 73.29), (102.37, 73.53), (102.79, 73.78),
    (103.6, 73.82),
)
remove(
    (100.55, 74.22), (100.46, 73.91), (100.19, 73.77),
    (100.14, 73.72), (100.23, 73.96), (100.27, 74.21),
)
remove(
    (98.99, 73.96), (99.00, 73.74), (98.73, 73.61),
    (98.57, 73.58), (98.76, 73.81), (98.66, 73.99),
)
add_oval(98.47, 74.02, 0.50, 0.15, -12)  # Russell Island
add_oval(99.53, 74.18, 0.18, 0.08)

# Somerset Island on the other side of Peel Sound. Prince Regent Inlet
# remains water to its east, even when the image cuts off its south shore.
add(
    (96.52, 74.10), (95.86, 74.08), (95.46, 74.14),
    (95.05, 74.08), (94.63, 74.13), (94.17, 74.09),
    (93.78, 74.02), (93.28, 73.94), (92.82, 73.88),
    (92.40, 73.74), (91.92, 73.66), (91.31, 73.63),
    (90.78, 73.48), (90.16, 73.34), (89.75, 73.14),
    (89.52, 72.5), (96.14, 72.5), (96.30, 73.04),
    (96.56, 73.37), (96.81, 73.48), (96.97, 73.72),
    (96.78, 73.90),
)
remove(
    (96.76, 74.20), (95.98, 74.01), (95.68, 73.79),
    (95.62, 73.69), (95.87, 73.86), (96.31, 74.04),
    (96.91, 74.07),
)
remove(
    (94.49, 74.23), (94.27, 73.98), (94.00, 73.89),
    (93.83, 73.88), (94.11, 74.08), (94.10, 74.20),
)
remove_oval(95.01, 73.54, 0.31, 0.14, 20)
add_oval(96.45, 74.27, 0.23, 0.07)
add_oval(93.15, 74.19, 0.13, 0.06)

# Northwest Baffin Island (Brodeur Peninsula and the land east of Admiralty
# Inlet). The inlet itself, cut from the land below, contains seawater.
add(
    (89.80, 73.64), (89.19, 73.59), (88.42, 73.64),
    (87.96, 73.68), (87.21, 73.64), (86.61, 73.70),
    (85.91, 73.66), (85.38, 73.69), (84.89, 73.72),
    (84.29, 73.78), (83.62, 73.76), (83.16, 73.80),
    (82.76, 73.82), (82.05, 73.93), (81.29, 73.94),
    (80.5, 73.97), (80.5, 72.5), (89.86, 72.5),
)
remove(
    (84.72, 72.5), (84.75, 73.01), (84.54, 73.26),
    (84.11, 73.46), (83.87, 73.62), (83.65, 73.93),
    (83.06, 73.99), (83.34, 73.61), (83.57, 73.43),
    (83.62, 73.19), (83.39, 72.5),
)
remove(
    (87.62, 73.82), (87.53, 73.61), (87.31, 73.51),
    (87.02, 73.42), (87.21, 73.57), (87.18, 73.80),
)
add_oval(89.40, 73.68, 0.22, 0.09)
# Deliverables are computed solely by testing each of the 14,701 coordinate
# points against the above hand-specified islands and coastal indentations.
def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    matrix = land.astype(np.uint8)
    assert matrix.shape == (61, 241)
    assert set(np.unique(matrix).tolist()) <= {0, 1}

    csv_file = out / "candidate.csv"
    with csv_file.open("w", encoding="ascii", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(matrix.tolist())

    base = Image.fromarray(matrix * 255, mode="L")
    base.save(out / "candidate.png")
    enlarged = base.resize((1928, 488), resample=Image.Resampling.NEAREST)
    enlarged.save(out / "candidate_8x.png")

    with csv_file.open(encoding="ascii", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 61 and all(len(row) == 241 for row in rows)
    assert all(v in ("0", "1") for row in rows for v in row)
    assert np.array_equal(np.asarray(rows, dtype=np.uint8), matrix)
    with Image.open(out / "candidate.png") as img:
        assert img.size == (241, 61)
        assert set(img.getdata()) <= {0, 255}
        original = np.array(img)
    with Image.open(out / "candidate_8x.png") as img:
        assert img.size == (1928, 488)
        assert set(img.getdata()) <= {0, 255}
        assert np.array_equal(np.array(img), np.repeat(np.repeat(
            original, 8, axis=0), 8, axis=1))
    print(f"Validated {matrix.size} points; {matrix.sum()} land, "
          f"{matrix.size - matrix.sum()} water.")


if __name__ == "__main__":
    main()
