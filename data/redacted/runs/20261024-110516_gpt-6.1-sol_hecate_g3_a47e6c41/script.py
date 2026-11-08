"""Knowledge-only approximation of the northern British Columbia coast.

No maps, GIS files, coastline services, or external reference data are used.
All geographic vertices below are hand-written estimates from prior knowledge.
Classification is point-in-polygon plus explicitly drawn marine channels,
not independent language-model reasoning for the individual sample points.
Requires Python 3, numpy and Pillow. Run from the desired working directory.
"""
from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image

N = 127
LONGITUDES = -131.0 + np.arange(N, dtype=float) * (2.0 / 126.0)
LATITUDES = 55.0 - np.arange(N, dtype=float) * (2.0 / 126.0)
X, Y = np.meshgrid(LONGITUDES, LATITUDES)
land = np.zeros((N, N), dtype=bool)
LAND_RULES = []
WATER_RULES = []


def polygon_mask(vertices):
    """Ray-crossing test at the sample point, without area averaging."""
    inside = np.zeros((N, N), dtype=bool)
    x0, y0 = vertices[-1]
    for x1, y1 in vertices:
        if y0 != y1:
            crossing = ((y0 > Y) != (y1 > Y))
            intercept = x0 + (Y - y0) * (x1 - x0) / (y1 - y0)
            inside ^= crossing & (X < intercept)
        x0, y0 = x1, y1
    return inside


def island(name, vertices):
    LAND_RULES.append((name, vertices))
    land[:] |= polygon_mask(vertices)


def marine_polygon(name, vertices):
    WATER_RULES.append((name, vertices))
    land[:] &= ~polygon_mask(vertices)


def channel(name, vertices):
    """Marine path; each (longitude, latitude, half-width in km) is manual.

    A locally constant metric is used only to construct the approximate rule.
    The path is sampled analytically by projection, not by pixel smoothing.
    """
    WATER_RULES.append((name, vertices))
    for a, b in zip(vertices[:-1], vertices[1:]):
        ax, ay, ra = a
        bx, by, rb = b
        dx, dy = (bx - ax) * 65.5, (by - ay) * 111.2
        px, py = (X - ax) * 65.5, (Y - ay) * 111.2
        denom = dx * dx + dy * dy
        if denom == 0:
            t = np.zeros_like(X)
        else:
            t = np.clip((px * dx + py * dy) / denom, 0.0, 1.0)
        r = ra + t * (rb - ra)
        wet = (px - t * dx) ** 2 + (py - t * dy) ** 2 <= r * r
        land[wet] = False


# Mainland south of the Alaska panhandle: Portland Inlet, Tsimpsean
# Peninsula, Skeena estuary and the eastern shore of Grenville Channel.
island("British Columbia mainland", [
    (-130.04,55.30),(-128.60,55.30),(-128.60,53.43),
    (-128.93,53.40),(-129.03,53.44),(-129.10,53.50),
    (-129.17,53.54),(-129.23,53.60),(-129.29,53.65),
    (-129.37,53.69),(-129.43,53.76),(-129.50,53.81),
    (-129.58,53.89),(-129.64,53.96),(-129.73,54.03),
    (-129.83,54.09),(-129.94,54.12),(-130.02,54.16),
    (-130.08,54.20),(-130.17,54.25),(-130.20,54.31),
    (-130.27,54.35),(-130.33,54.38),(-130.39,54.43),
    (-130.43,54.49),(-130.47,54.54),(-130.44,54.59),
    (-130.40,54.63),(-130.42,54.66),(-130.33,54.70),
    (-130.23,54.73),(-130.15,54.77),(-130.04,54.81),
    (-129.96,54.85),(-129.88,54.90),(-129.79,54.95),
    (-129.77,55.00),(-129.85,55.07),(-129.94,55.14),
])

# Cape Fox mainland. The top and side vertices intentionally extend past
# the requested rectangle; only the requested sample coordinates are tested.
island("Cape Fox mainland", [
    (-131.25,55.30),(-130.16,55.30),(-130.22,55.15),
    (-130.30,55.04),(-130.40,54.95),(-130.49,54.89),
    (-130.57,54.84),(-130.62,54.79),(-130.67,54.77),
    (-130.74,54.80),(-130.81,54.82),(-130.87,54.85),
    (-130.91,54.88),(-130.90,54.93),(-130.96,54.97),
    (-131.00,55.04),(-131.10,55.13),
])

island("Portland-Observatory mainland tongue", [
    (-130.11,55.30),(-129.91,55.30),(-129.87,55.16),
    (-129.84,55.05),(-129.86,54.98),(-129.93,54.93),
    (-130.00,54.92),(-130.05,54.97),(-130.08,55.08),
])
