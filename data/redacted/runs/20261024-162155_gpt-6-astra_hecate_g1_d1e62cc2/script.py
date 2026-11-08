"""Knowledge-only approximate coast reconstruction; no geographic data inputs.

Dependencies: Python 3, numpy, Pillow.
Run from the working directory. All output paths are relative to that directory.
The coordinates below are hand-written approximations, not surveyed coastlines.
Classification is polygon/corridor computation, not independent point reasoning.
Inland lakes are intentionally not removed from land.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

N = 127
LON, LAT = np.meshgrid(-131.0 + np.arange(N) * (2.0 / 126.0),
                      55.0 - np.arange(N) * (2.0 / 126.0))
POLYGONS = []
CHANNELS = []

def land(name, vertices):
    POLYGONS.append((name, vertices))

def channel(name, vertices, widths):
    """Full channel widths in km at hand-written centreline vertices."""
    assert len(vertices) == len(widths)
    CHANNELS.append((name, vertices, widths))

def inside(vertices):
    result = np.zeros((N, N), dtype=bool)
    x0, y0 = vertices[-1]
    for x1, y1 in vertices:
        if y1 != y0:
            crossing = ((y1 > LAT) != (y0 > LAT))
            intercept = x0 + (LAT - y0) * (x1 - x0) / (y1 - y0)
            result ^= crossing & (LON < intercept)
        x0, y0 = x1, y1
    return result

def corridor(vertices, widths):
    # Local metric only for approximating inlet widths; no geographic library.
    x, y = LON * 65.4, LAT * 111.2
    result = np.zeros((N, N), dtype=bool)
    for k in range(len(vertices) - 1):
        ax, ay = vertices[k][0] * 65.4, vertices[k][1] * 111.2
        bx, by = vertices[k+1][0] * 65.4, vertices[k+1][1] * 111.2
        dx, dy = bx - ax, by - ay
        t = np.clip(((x-ax)*dx + (y-ay)*dy) / (dx*dx+dy*dy), 0, 1)
        radius = (widths[k] * (1-t) + widths[k+1] * t) / 2
        result |= (x-ax-t*dx)**2 + (y-ay-t*dy)**2 <= radius**2
    return result

# The mainland's western coast, including the Tsimpsean peninsula.
# Closing vertices outside the frame only close the polygon; no outside
# sampling points are generated.
land("British Columbia mainland", [
    (-128.70,55.15),(-129.68,55.15),(-129.77,55.04),
    (-129.86,55.00),(-129.93,54.94),(-130.02,54.89),
    (-130.10,54.82),(-130.18,54.77),(-130.28,54.72),
    (-130.39,54.67),(-130.44,54.61),(-130.44,54.55),
    (-130.49,54.49),(-130.48,54.44),(-130.45,54.39),
    (-130.42,54.35),(-130.37,54.33),(-130.33,54.29),
    (-130.31,54.24),(-130.25,54.20),(-130.18,54.17),
    (-130.12,54.12),(-130.08,54.085),(-130.035,54.045),
    (-130.005,53.995),(-129.955,53.94),(-129.91,53.89),
    (-129.855,53.83),(-129.805,53.77),(-129.76,53.71),
    (-129.70,53.65),(-129.655,53.60),(-129.60,53.56),
    (-129.555,53.52),(-129.51,53.48),(-129.465,53.45),
    (-129.40,53.425),(-129.34,53.42),(-129.285,53.44),
    (-129.245,53.475),(-129.23,53.515),(-129.17,53.55),
    (-129.135,53.59),(-129.095,53.63),(-129.02,53.67),
    (-128.93,53.70),(-128.70,53.75)])

land("Alaska mainland near Cape Fox", [
    (-131.10,55.20),(-130.26,55.20),(-130.275,55.03),
    (-130.30,54.98),(-130.32,54.94),(-130.36,54.90),
    (-130.40,54.865),(-130.46,54.85),(-130.49,54.81),
    (-130.55,54.785),(-130.61,54.79),(-130.65,54.77),
    (-130.705,54.755),(-130.73,54.79),(-130.73,54.835),
    (-130.77,54.87),(-130.785,54.92),(-130.85,54.96),
    (-130.88,55.015),(-130.98,55.07)])

land("Wales Island", [
    (-130.50,54.89),(-130.435,54.89),(-130.39,54.86),
    (-130.375,54.82),(-130.405,54.77),(-130.45,54.72),
    (-130.49,54.685),(-130.55,54.69),(-130.59,54.72),
    (-130.615,54.755),(-130.58,54.785),(-130.55,54.83)])
land("Pearse Island", [
    (-130.255,55.075),(-130.17,55.095),(-130.11,55.065),
    (-130.12,55.015),(-130.14,54.97),(-130.19,54.915),
    (-130.235,54.865),(-130.30,54.80),(-130.35,54.80),
    (-130.37,54.835),(-130.33,54.895),(-130.29,54.95)])
land("Sitklan Island", [
    (-130.69,54.77),(-130.65,54.79),(-130.615,54.775),
    (-130.62,54.75),(-130.66,54.735),(-130.69,54.745)])
land("Kannaghunut Island", [
    (-130.65,54.73),(-130.60,54.745),(-130.575,54.725),
    (-130.60,54.705),(-130.64,54.71)])
