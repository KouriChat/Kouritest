"""Knowledge-only, hand-drawn approximation of the northern BC coast.

No map, geographic dataset, network service, or land/water library is used.
Coordinates below are approximate remembered outlines, not surveyed coastlines.
The entire grid is classified by polygon containment, not independent LLM
reasoning at each point. Inland lakes are deliberately not subtracted.
Run from the desired working directory. Dependencies: numpy and Pillow.
"""
from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image

N = 127
longitudes = -131.0 + np.arange(N, dtype=np.float64) * (2.0 / 126.0)
latitudes = 55.0 - np.arange(N, dtype=np.float64) * (2.0 / 126.0)
X, Y = np.meshgrid(longitudes, latitudes)
land = np.zeros((N, N), dtype=bool)
outlines = []


def polygon_mask(vertices):
    """Even-odd containment of sampling POINTS; no area averaging."""
    inside = np.zeros((N, N), dtype=bool)
    x1, y1 = vertices[-1]
    for x2, y2 in vertices:
        if y2 != y1:
            crossing = ((y1 > Y) != (y2 > Y))
            cut_x = x1 + (Y - y1) * (x2 - x1) / (y2 - y1)
            inside ^= crossing & (X <= cut_x)
        x1, y1 = x2, y2
    return inside


def add(name, vertices):
    outlines.append((name, vertices))
    land[:] |= polygon_mask(vertices)


def sea(name, vertices):
    land[:] &= ~polygon_mask(vertices)


# Mainland southeast Alaska, including the Cape Fox/Tree Point coast.
# Vertices outside the box only close polygons; no outside samples are made.
add("Alaska mainland", [
    (-131.20,55.35),(-131.08,55.17),(-131.03,55.06),
    (-131.015,54.995),(-130.976,54.945),(-130.963,54.883),
    (-130.957,54.841),(-130.932,54.802),(-130.897,54.797),
    (-130.874,54.827),(-130.848,54.834),(-130.822,54.812),
    (-130.785,54.802),(-130.763,54.779),(-130.711,54.772),
    (-130.670,54.760),(-130.628,54.772),(-130.593,54.789),
    (-130.567,54.815),(-130.513,54.833),(-130.465,54.841),
    (-130.424,54.862),(-130.391,54.900),(-130.345,54.921),
    (-130.314,54.953),(-130.254,54.976),(-130.224,55.014),
    (-130.209,55.075),(-130.205,55.200),(-130.12,55.35),
])

# BC mainland: Portland Inlet, Tsimpsean Peninsula, Skeena mouth,
# eastern side of Grenville Channel, and western shore of Douglas Channel.
add("British Columbia mainland", [
    (-129.98,55.35),(-129.965,55.12),(-129.974,55.040),
    (-129.943,54.983),(-129.870,54.957),(-129.783,54.950),
    (-129.846,54.914),(-129.901,54.888),(-129.957,54.858),
    (-130.023,54.825),(-130.077,54.787),(-130.123,54.753),
    (-130.172,54.724),(-130.222,54.705),(-130.280,54.683),
    (-130.343,54.667),(-130.408,54.662),(-130.454,54.640),
    (-130.474,54.605),(-130.459,54.578),(-130.438,54.547),
    (-130.433,54.515),(-130.448,54.486),(-130.431,54.457),
    (-130.444,54.429),(-130.424,54.403),(-130.444,54.371),
    (-130.426,54.346),(-130.389,54.350),(-130.359,54.374),
    (-130.320,54.379),(-130.284,54.397),(-130.244,54.411),
    (-130.203,54.444),(-130.187,54.435),(-130.210,54.394),
    (-130.235,54.358),(-130.231,54.322),(-130.206,54.293),
    (-130.195,54.255),(-130.174,54.234),(-130.125,54.226),
    (-130.089,54.206),(-130.068,54.173),(-130.063,54.137),
    (-130.039,54.105),(-130.052,54.078),(-130.021,54.047),
    (-129.992,54.012),(-129.962,53.974),(-129.947,53.939),
    (-129.907,53.905),(-129.880,53.868),(-129.847,53.830),
    (-129.813,53.788),(-129.785,53.751),(-129.750,53.710),
    (-129.720,53.666),(-129.680,53.625),(-129.651,53.589),
    (-129.613,53.553),(-129.582,53.512),(-129.537,53.475),
    (-129.486,53.440),(-129.427,53.418),(-129.378,53.431),
    (-129.364,53.466),(-129.388,53.501),(-129.353,53.537),
    (-129.317,53.575),(-129.307,53.623),(-129.278,53.663),
    (-129.227,53.698),(-129.193,53.739),(-129.144,53.769),
    (-129.073,53.785),(-129.020,53.818),(-128.959,53.849),
    (-128.862,53.898),(-128.65,54.02),(-128.5,55.35),
])
