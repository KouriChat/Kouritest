"""Memory-only reconstruction; no geographic files, services, or databases.

Run from the current working directory. Dependencies: numpy and Pillow.
Coordinates below are hand-written (degrees west, degrees north).
These approximate polygons are evaluated mechanically at the sample points;
they are not 16,129 independent language-model judgements.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

N = 127
west, north = np.meshgrid(
    131.0 - np.arange(N, dtype=np.float64) * (2.0 / 126.0),
    55.0 - np.arange(N, dtype=np.float64) * (2.0 / 126.0),
)

def polygon(vertices):
    """Even/odd point-in-polygon test at the exact sampling coordinates."""
    result = np.zeros((N, N), dtype=bool)
    x0, y0 = vertices[-1]
    for x1, y1 in vertices:
        if y1 != y0:
            crosses = (y0 > north) != (y1 > north)
            intersection = x0 + (north - y0) * (x1 - x0) / (y1 - y0)
            result ^= crosses & (west < intersection)
        x0, y0 = x1, y1
    return result

# The mainland coast is deliberately approximate, with separately cut inlets.
mainland_polygons = [
    # Alaska mainland, Cape Fox and the west side of Portland Canal.
    [(130.86,55.06),(130.82,54.98),(130.85,54.90),
     (130.87,54.82),(130.82,54.78),(130.76,54.76),
     (130.68,54.78),(130.63,54.82),(130.57,54.84),
     (130.53,54.89),(130.47,54.94),(130.40,54.98),
     (130.35,55.06)],
    # BC mainland: Portland Inlet, Tsimpsean Peninsula, Skeena, Grenville.
    [(129.91,55.06),(129.94,55.00),(129.91,54.97),
     (129.98,54.93),(130.02,54.87),(130.10,54.82),
     (130.12,54.76),(130.24,54.72),(130.34,54.69),
     (130.42,54.66),(130.44,54.61),(130.46,54.56),
     (130.43,54.53),(130.40,54.49),(130.39,54.44),
     (130.42,54.39),(130.43,54.36),(130.38,54.34),
     (130.30,54.35),(130.25,54.33),(130.22,54.28),
     (130.18,54.25),(130.17,54.20),(130.13,54.16),
     (130.10,54.12),(130.04,54.08),(130.00,54.04),
     (129.94,53.99),(129.89,53.94),(129.83,53.89),
     (129.79,53.84),(129.73,53.78),(129.68,53.73),
     (129.61,53.67),(129.56,53.62),(129.50,53.57),
     (129.44,53.52),(129.39,53.48),(129.30,53.44),
     (129.24,53.43),(129.17,53.47),(129.13,53.53),
     (129.12,53.60),(129.13,53.67),(129.08,53.74),
     (129.04,53.79),(128.95,53.84),(128.90,55.06)],
]

inlets = [
    # Work Channel, cutting into the Tsimpsean Peninsula's eastern side.
    [(130.44,54.69),(130.36,54.60),(130.30,54.56),
     (130.25,54.50),(130.19,54.45),(130.14,54.40),
     (130.10,54.34),(130.07,54.31),(130.065,54.35),
     (130.11,54.41),(130.16,54.47),(130.22,54.52),
     (130.27,54.58),(130.34,54.64),(130.39,54.70)],
    # Khutzeymateen Inlet.
    [(130.30,54.68),(130.18,54.67),(130.07,54.67),
     (129.97,54.68),(129.88,54.70),(129.81,54.72),
     (129.76,54.76),(129.77,54.78),(129.84,54.74),
     (129.92,54.73),(130.00,54.71),(130.11,54.71),
     (130.21,54.72),(130.30,54.71)],
    # Kwinamass / eastern Portland Inlet embayment.
    [(130.13,54.79),(130.04,54.79),(129.98,54.80),
     (129.91,54.83),(129.86,54.87),(129.87,54.89),
     (129.94,54.85),(130.01,54.84),(130.10,54.83)],
    # Skeena estuary and the principal tidal river channel.
    [(130.26,54.17),(130.16,54.21),(130.07,54.23),
     (130.00,54.26),(129.91,54.29),(129.83,54.33),
     (129.74,54.38),(129.65,54.43),(129.58,54.47),
     (129.56,54.45),(129.64,54.40),(129.72,54.35),
     (129.80,54.29),(129.89,54.25),(129.98,54.21),
     (130.06,54.18),(130.13,54.13),(130.20,54.12)],
    # Small fjords opening from Grenville Channel.
    [(129.76,53.83),(129.68,53.85),(129.60,53.88),
     (129.54,53.92),(129.53,53.90),(129.59,53.85),
     (129.67,53.82),(129.73,53.80)],
    [(129.56,53.63),(129.48,53.64),(129.43,53.68),
     (129.39,53.71),(129.38,53.69),(129.42,53.65),
     (129.47,53.61),(129.53,53.60)],
]

land = np.zeros((N, N), dtype=bool)
for outline in mainland_polygons:
    land |= polygon(outline)
for outline in inlets:
    land &= ~polygon(outline)

islands = [
    # Wales, Pearse, and Somerville: Portland entrance islands.
    [(130.64,54.85),(130.59,54.92),(130.52,54.93),
     (130.46,54.89),(130.43,54.83),(130.45,54.76),
     (130.50,54.72),(130.56,54.72),(130.61,54.78)],
    [(130.43,55.07),(130.30,55.07),(130.22,55.02),
     (130.19,54.97),(130.25,54.91),(130.29,54.85),
     (130.36,54.80),(130.41,54.81),(130.44,54.86),
     (130.39,54.94),(130.40,55.00)],
    [(130.19,55.07),(130.06,55.07),(130.02,55.01),
     (130.07,54.96),(130.10,54.92),(130.17,54.88),
     (130.22,54.89),(130.22,54.93),(130.16,55.00)],
