"""Memory-only reconstruction of the Canadian Arctic Archipelago.

No maps, geographic databases, coastline libraries, network access, or answer
files are used. All vertices below are hand-written approximations from prior
geographical knowledge, not measured coordinates. Classification is a
vectorized point-in-polygon calculation, NOT independent reasoning at each
sample. Inland lakes are retained as land; explicit cuts represent seawater.

Run: python outputs/generate_map.py
Dependencies: Python 3, numpy, Pillow.
Coordinates in the hand-written rules are (degrees WEST, degrees NORTH).
"""
from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image

LAT = (790 - np.arange(61, dtype=np.int32)) / 10.0
LON = (-1050 + np.arange(241, dtype=np.int32)) / 10.0
WEST, NORTH = np.meshgrid(-LON, LAT)
land = np.zeros((61, 241), dtype=bool)
rules = []


def polygon(vertices):
    """Even-odd point test; a point exactly on an edge counts as inside."""
    inside = np.zeros(land.shape, dtype=bool)
    boundary = np.zeros(land.shape, dtype=bool)
    x0, y0 = vertices[-1]
    for x1, y1 in vertices:
        dx, dy = x1 - x0, y1 - y0
        cross = (WEST - x0) * dy - (NORTH - y0) * dx
        boundary |= (
            (np.abs(cross) < 1e-10)
            & (WEST >= min(x0, x1) - 1e-10)
            & (WEST <= max(x0, x1) + 1e-10)
            & (NORTH >= min(y0, y1) - 1e-10)
            & (NORTH <= max(y0, y1) + 1e-10)
        )
        if dy != 0:
            crossing = ((y0 > NORTH) != (y1 > NORTH))
            intercept = x0 + (NORTH - y0) * dx / dy
            inside ^= crossing & (WEST < intercept)
        x0, y0 = x1, y1
    return inside | boundary


def add(name, vertices):
    rules.append(("land", name, vertices))
    land[:] |= polygon(vertices)


def sea(name, vertices):
    rules.append(("water", name, vertices))
    land[:] &= ~polygon(vertices)


# Ellef Ringnes: the northern end extends beyond the requested window.
add("Ellef Ringnes", [
    (102.10,79.55),(100.30,79.55),(99.55,79.30),(99.28,79.05),
    (99.45,78.88),(99.90,78.79),(99.58,78.58),(99.82,78.40),
    (100.35,78.34),(100.32,78.16),(100.87,78.10),(101.02,77.92),
    (101.50,77.88),(101.72,77.65),(102.15,77.53),(102.65,77.59),
    (102.68,77.78),(102.40,77.95),(102.95,78.02),(103.20,78.25),
    (102.92,78.42),(103.35,78.46),(103.67,78.66),(103.54,78.86),
    (103.15,78.94),(103.18,79.10),(102.65,79.25),(102.35,79.30),
])
sea("Ellef Ringnes western bays", [
    (103.80,78.10),(102.85,78.17),(102.20,78.42),(102.35,78.51),
    (102.90,78.38),(103.80,78.47),
])
add("Amund Ringnes", [
    (97.62,78.91),(96.88,78.84),(96.38,78.75),(95.60,78.70),
    (95.22,78.51),(95.36,78.29),(94.76,78.18),(94.88,78.02),
    (95.30,77.92),(95.41,77.74),(96.02,77.62),(96.49,77.68),
    (96.80,77.86),(97.32,77.83),(97.69,78.03),(97.35,78.26),
    (97.74,78.43),(97.91,78.66),
])
add("King Christian", [
    (102.16,77.84),(101.72,77.96),(101.06,77.95),(100.61,77.84),
    (100.69,77.65),(101.07,77.56),(101.65,77.62),(102.07,77.70),
])
add("Lougheed", [
    (105.75,77.60),(105.10,77.62),(104.72,77.48),(104.55,77.25),
    (104.68,77.03),(105.04,76.96),(105.31,77.13),(105.73,77.17),
])
add("Southern Lougheed islets", [
    (105.19,76.93),(104.97,76.94),(104.84,76.80),(105.03,76.74),
])
add("Cornwall", [
    (96.14,77.78),(95.53,77.93),(94.73,77.91),(94.27,77.79),
    (93.40,77.75),(93.03,77.60),(93.42,77.43),(94.06,77.35),
    (94.64,77.32),(95.19,77.41),(95.57,77.57),(96.00,77.60),
])
