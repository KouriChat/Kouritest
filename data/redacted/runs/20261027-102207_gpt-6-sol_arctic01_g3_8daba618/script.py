"""An approximate Canadian Arctic Archipelago land mask, hand-drawn from memory.

The contours below are estimates, not a coastline dataset. Coordinates are
(degrees west, degrees north); rasterization uses the coordinate of each point.
"""
from pathlib import Path
import csv

import numpy as np
from PIL import Image


W = np.broadcast_to((105.0 - np.arange(241, dtype=float) / 10)[None, :], (61, 241))
N = np.broadcast_to((79.0 - np.arange(61, dtype=float) / 10)[:, None], (61, 241))
land = np.zeros((61, 241), dtype=bool)


def polygon(vertices, erase=False):
    """Even-odd point-in-polygon; erase is used for open straits/inlets."""
    inside = np.zeros_like(land)
    last_w, last_n = vertices[-1]
    for west, north in vertices:
        if (last_n > 73) != (north > 73) or last_n != north:
            if last_n != north:
                intersects = ((last_n > N) != (north > N))
                intersection = last_w + (N - last_n) * (west - last_w) / (north - last_n)
                inside ^= intersects & (W < intersection)
        last_w, last_n = west, north
    if erase:
        land[inside] = False
    else:
        land[inside] = True


def island(west, north, radius_w, radius_n, tilt=0):
    """Small approximate oval, possibly tilted in west/latitude coordinates."""
    t = np.linspace(0, 2 * np.pi, 19)[:-1]
    polygon([(west + radius_w * np.cos(v) + tilt * np.sin(v),
              north + radius_n * np.sin(v)) for v in t])


# Southern Axel Heiberg Island, west of Eureka Sound. Most of it lies north of
# the raster. The shallow coves are incorporated as vertices, not smoothing.
polygon([
    (99.2, 80.0), (97.3, 79.7), (96.1, 79.4), (94.4, 79.2),
    (93.4, 78.95), (92.9, 78.72), (92.0, 78.53), (91.65, 78.38),
    (91.85, 78.25), (92.6, 78.22), (93.3, 78.31), (94.2, 78.48),
    (94.9, 78.48), (95.2, 78.61), (95.8, 78.57), (96.7, 78.73),
    (97.3, 78.9), (99.2, 79.03), (100.5, 79.5), (101.5, 80.0)
])

# Ellesmere Island south coast of Jones Sound and the west shore of Eureka
# Sound: a wide northern mainland, with a jagged southern boundary.
polygon([
    (87.45, 79.6), (87.6, 79.1), (88.1, 78.86), (88.35, 78.65),
    (88.9, 78.44), (88.85, 78.20), (89.3, 78.04), (89.9, 77.91),
    (90.45, 77.81), (90.6, 77.67), (90.05, 77.45),
    (89.2, 77.35), (88.8, 77.17), (87.75, 77.13),
    (86.75, 77.02), (85.9, 77.1), (85.25, 76.99),
    (84.75, 77.02), (84.1, 76.9), (83.35, 76.91),
    (82.5, 76.77), (81.75, 76.84), (80.8, 76.77),
    (79.8, 76.83), (78.5, 77.0), (78.0, 79.8)
])

# Coastal inlets incised into the west of Ellesmere.
polygon([(89.1, 78.12), (88.5, 78.19), (87.85, 78.4),
         (87.9, 78.52), (88.65, 78.37), (89.1, 78.29)], erase=True)

# Ellef Ringnes Island, west of Hassel Sound; only its southern end enters.
polygon([
    (103.65, 79.55), (102.8, 79.36), (102.25, 79.02),
    (101.15, 78.86), (100.65, 78.61), (100.9, 78.39),
    (101.85, 78.27), (102.75, 78.38), (103.65, 78.63),
    (104.1, 78.91), (104.7, 79.6)
])

# Amund Ringnes Island, lower centre of the Sverdrup Islands.
polygon([
    (98.15, 79.27), (97.15, 79.13), (96.5, 78.97),
    (95.8, 78.8), (95.35, 78.55), (95.55, 78.33),
    (96.05, 78.18), (96.8, 78.13), (97.55, 78.25),
    (98.2, 78.45), (98.65, 78.69), (98.65, 79.0)
])

# Small islands in the channels north of Bathurst.
island(103.9, 77.75, 0.48, 0.30, -0.10)  # Lougheed, eastern fringe
island(100.6, 77.66, 0.32, 0.21)
island(98.5, 77.72, 0.27, 0.18)
island(97.55, 77.54, 0.36, 0.14)
island(92.2, 77.28, 0.39, 0.14)
# Melville Island's eastern tip reaches into the western edge of this window.
polygon([
    (108.0, 76.95), (105.5, 76.83), (104.95, 76.64),
    (104.6, 76.43), (104.58, 76.15), (104.9, 75.96),
    (105.27, 75.88), (106.5, 75.95), (108.0, 76.15)
])

# Cameron, Vanier and the western satellites of Bathurst Island.
polygon([(104.13, 76.86), (103.55, 76.78), (103.03, 76.5),
         (103.13, 76.26), (103.85, 76.22), (104.25, 76.5)])
polygon([(103.55, 76.14), (103.14, 76.13), (102.67, 75.98),
         (102.69, 75.83), (103.37, 75.82), (103.7, 75.95)])
island(103.9, 75.67, 0.43, 0.16)
island(104.67, 75.15, 0.38, 0.33, -0.12)  # Byam Martin Island

# Bathurst Island. The northwest and southeast peninsulas are left connected
# where the deep northern embayments terminate.
polygon([
    (102.75, 77.21), (102.0, 77.21), (101.38, 77.09),
    (100.93, 76.87), (100.25, 76.77), (99.8, 76.88),
    (99.36, 77.06), (98.94, 77.16), (98.5, 77.04),
    (98.55, 76.78), (98.24, 76.52), (98.62, 76.27),
    (98.08, 76.08), (97.87, 75.87), (98.12, 75.68),
    (98.72, 75.61), (99.0, 75.8), (99.59, 75.91),
    (99.81, 75.76), (100.5, 75.69), (100.93, 75.77),
    (101.03, 76.05), (101.59, 76.17), (101.83, 76.47),
    (102.43, 76.65), (102.81, 76.92)
])
polygon([(101.62, 77.32), (101.1, 77.28), (100.9, 77.04),
         (101.52, 76.98), (101.95, 77.12)])
island(97.75, 76.22, 0.37, 0.22)
island(99.63, 77.15, 0.28, 0.20)
island(101.96, 75.48, 0.24, 0.17)

# Cornwallis Island is separated from Bathurst by McDougall Sound and from
# Devon by Wellington Channel; its southern edge faces Barrow Strait.
polygon([
    (96.78, 75.68), (96.20, 75.64), (95.88, 75.48),
    (95.24, 75.45), (94.81, 75.5), (94.37, 75.4),
    (94.29, 75.16), (94.37, 74.91), (94.06, 74.78),
    (94.31, 74.6), (94.87, 74.57), (95.33, 74.67),
    (95.86, 74.61), (96.35, 74.75), (96.56, 75.03),
    (96.92, 75.21), (97.09, 75.42)
])
island(97.66, 74.85, 0.35, 0.13)   # small Barrow Strait islands
island(93.78, 74.68, 0.22, 0.08)   # Beechey vicinity

# Devon Island: continuous large island between Jones Sound (north) and
# Lancaster Sound (south); irregular lobes follow major peninsulas and bays.
polygon([
    (95.55, 76.55), (94.95, 76.56), (94.51, 76.43),
    (93.75, 76.55), (93.28, 76.48), (92.6, 76.49),
    (91.95, 76.62), (91.15, 76.57), (90.56, 76.56),
    (89.84, 76.48), (89.14, 76.42), (88.53, 76.42),
    (88.0, 76.59), (87.5, 76.45), (86.81, 76.50),
    (86.05, 76.37), (85.25, 76.42), (84.75, 76.37),
    (84.13, 76.27), (83.59, 76.27), (82.95, 76.30),
    (82.43, 76.21), (81.8, 76.24), (81.23, 76.26),
    (80.62, 76.09), (79.82, 76.0), (79.72, 75.70),
    (80.08, 75.38), (80.37, 75.12), (80.37, 74.88),
    (80.71, 74.72), (81.41, 74.68), (82.17, 74.52),
    (82.83, 74.57), (83.16, 74.40), (83.85, 74.4),
    (84.55, 74.52), (85.28, 74.52), (86.04, 74.57),
    (86.74, 74.52), (87.22, 74.65), (87.84, 74.59),
    (88.56, 74.68), (89.15, 74.54), (89.75, 74.58),
    (90.45, 74.48), (91.15, 74.6), (91.84, 74.56),
    (92.5, 74.70), (93.2, 74.77), (93.5, 74.95),
    (93.47, 75.24), (93.76, 75.39), (93.63, 75.66),
    (93.94, 75.81), (94.22, 75.94), (94.65, 75.88),
    (95.02, 76.11), (95.63, 76.25)
])

# Major Devon fjords on its Jones Sound side and on the southern coast.
polygon([(91.13, 76.76), (90.61, 76.49), (90.63, 76.19),
         (90.96, 76.02), (91.21, 76.24)], erase=True)
polygon([(85.46, 76.62), (85.05, 76.35), (85.1, 76.06),
         (85.37, 76.0), (85.53, 76.31)], erase=True)
polygon([(83.35, 74.29), (83.46, 74.67), (83.8, 74.9),
         (84.07, 74.78), (83.91, 74.4)], erase=True)

# Northern reaches of Devon-adjacent islands in Jones Sound.
island(90.4, 76.94, 0.35, 0.17)
island(79.95, 75.84, 0.46, 0.24)
# Northern Prince of Wales Island and the separated Russell Island. The broad
# passage to its west is M'Clintock Channel, to its east Peel Sound.
polygon([
    (101.57, 74.15), (101.18, 74.04), (100.67, 74.09),
    (100.27, 74.31), (99.78, 74.25), (99.19, 74.20),
    (98.67, 74.24), (98.13, 74.13), (97.72, 73.96),
    (97.42, 73.67), (97.63, 73.38), (97.48, 73.13),
    (97.64, 72.89), (100.98, 72.8), (101.11, 73.06),
    (101.32, 73.26), (101.36, 73.57), (101.76, 73.85)
])
island(99.12, 74.49, 0.43, 0.18)
island(98.27, 74.45, 0.31, 0.15)
island(96.64, 73.76, 0.17, 0.20)

# Somerset Island, running along the south side of Barrow Strait and the
# Lancaster Sound entrance. Its southwest shore faces Peel Sound.
polygon([
    (95.21, 74.45), (94.72, 74.49), (94.2, 74.39),
    (93.72, 74.37), (93.33, 74.26), (92.93, 74.3),
    (92.25, 74.24), (91.59, 74.26), (91.16, 74.15),
    (90.7, 73.92), (90.34, 73.85), (89.98, 73.5),
    (90.22, 73.30), (90.43, 73.00), (95.41, 72.8),
    (95.5, 73.05), (95.32, 73.26), (95.62, 73.45),
    (95.58, 73.73), (95.32, 73.91), (95.46, 74.18)
])
polygon([(92.62, 74.5), (92.11, 74.22), (92.26, 74.0),
         (92.69, 73.99), (92.82, 74.19)], erase=True)
polygon([(90.24, 73.75), (90.8, 73.79), (91.14, 73.55),
         (90.97, 73.31), (90.46, 73.44)], erase=True)

# Small islands and islets on the south side of Lancaster Sound.
island(88.45, 73.83, 0.42, 0.23)
island(86.74, 73.75, 0.30, 0.11)
island(83.85, 73.71, 0.18, 0.12)

# Northern Baffin peninsula enters the bottom-right edge of the selected
# window; marine inlets separate its lobes near the frame edge.
polygon([
    (86.25, 72.5), (86.09, 73.04), (85.56, 73.18),
    (85.02, 73.26), (84.49, 73.13), (84.09, 73.20),
    (83.72, 73.11), (83.14, 73.12), (82.91, 73.03),
    (82.49, 73.17), (81.98, 73.11), (81.55, 73.18),
    (81.19, 73.04), (80.8, 72.95), (80.0, 72.5)
])
island(81.25, 73.56, 0.29, 0.16)


def main():
    out = Path('outputs')
    out.mkdir(parents=True, exist_ok=True)
    mask = land.astype(np.uint8)
    assert mask.shape == (61, 241) and mask.size == 14701
    with (out / 'candidate.csv').open('w', newline='', encoding='ascii') as f:
        csv.writer(f).writerows(mask.tolist())
    original = Image.fromarray(mask * 255, mode='L')
    original.save(out / 'candidate.png')
    resampling = Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST
    enlarged = original.resize((1928, 488), resample=resampling)
    enlarged.save(out / 'candidate_8x.png')

    with (out / 'candidate.csv').open(newline='', encoding='ascii') as f:
        rows = list(csv.reader(f))
    assert len(rows) == 61 and all(len(r) == 241 and set(r) <= {'0', '1'} for r in rows)
    assert original.size == (241, 61) and enlarged.size == (1928, 488)
    assert set(original.getdata()) <= {0, 255} and set(enlarged.getdata()) <= {0, 255}
    assert np.array_equal(np.asarray(enlarged), np.repeat(np.repeat(np.asarray(original), 8, 0), 8, 1))
    print('Verified: 61 x 241 = 14,701 points; PNGs 241 x 61 and 1928 x 488; binary nearest-neighbor.')


if __name__ == '__main__':
    main()
