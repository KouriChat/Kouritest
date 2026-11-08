"""Knowledge-only approximation of the Canadian Arctic Archipelago.

No geographic data, coastline files, network requests or reference maps are used.
Coordinates below are hand-written estimates, not surveyed coastline vertices.
Sampling uses point-in-polygon tests, not independent reasoning per point.
Dependencies: Python 3, numpy, Pillow.
Run from the working directory: python outputs/generate_map.py
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

LAND = []
WATER = []


def island(name, vertices):
    """Vertices use positive degrees west, followed by degrees north."""
    LAND.append((name, np.asarray(vertices, dtype=np.float64)))


def inlet(name, vertices):
    WATER.append((name, np.asarray(vertices, dtype=np.float64)))


# Eastern Melville Island, cut by the western edge of the requested window.
island("Melville", [
    (110.0, 74.65), (108.0, 74.75), (106.5, 74.88), (105.60, 75.08),
    (105.12, 75.35), (104.60, 75.38), (104.26, 75.64), (104.48, 75.91),
    (104.05, 76.05), (104.15, 76.32), (104.62, 76.49), (105.00, 76.45),
    (105.36, 76.75), (106.20, 76.83), (107.15, 76.68), (108.30, 76.85),
    (110.0, 76.80),
])

# Byam Martin Island is separated from Melville and Bathurst by open channels.
island("Byam Martin", [
    (104.28, 75.13), (103.78, 75.11), (103.35, 75.25), (103.06, 75.46),
    (103.19, 75.67), (103.67, 75.81), (104.03, 75.69), (104.18, 75.43),
])

# Bathurst: the southern body and its deeply divided northern peninsulas.
island("Bathurst", [
    (102.23, 75.13), (101.83, 75.03), (101.28, 75.07), (100.78, 75.00),
    (100.17, 75.06), (99.79, 75.26), (99.12, 75.30), (98.55, 75.39),
    (98.04, 75.53), (97.51, 75.61), (97.23, 75.85), (97.42, 76.09),
    (97.22, 76.30), (97.44, 76.47), (97.39, 76.65), (97.70, 76.83),
    (97.52, 77.03), (97.77, 77.20), (98.21, 77.18), (98.60, 77.04),
    (98.92, 76.81), (98.81, 76.56), (99.15, 76.41), (99.48, 76.30),
    (99.61, 76.12), (99.88, 76.18), (99.83, 76.54), (99.44, 76.81),
    (99.48, 77.02), (99.96, 77.19), (100.49, 77.21), (100.91, 77.08),
    (101.10, 76.80), (100.91, 76.61), (100.60, 76.45), (100.81, 76.26),
    (101.29, 76.38), (101.65, 76.63), (102.10, 76.63), (102.39, 76.44),
    (102.22, 76.22), (101.93, 76.05), (102.12, 75.80), (102.41, 75.60),
    (102.42, 75.36),
])

# Islands along Bathurst's western shore.
island("Vanier", [
    (103.42, 76.03), (102.99, 76.00), (102.59, 76.12), (102.52, 76.32),
    (102.86, 76.51), (103.24, 76.47), (103.48, 76.26),
])
island("Cameron", [
    (103.73, 76.43), (103.10, 76.48), (102.78, 76.63), (102.88, 76.83),
    (103.38, 76.92), (103.83, 76.73),
])
island("Massey", [
    (103.16, 75.91), (102.64, 75.91), (102.49, 75.99), (102.66, 76.08),
    (103.14, 76.04),
])
island("Marc", [
    (103.18, 75.80), (102.90, 75.78), (102.83, 75.86), (103.10, 75.89),
])
island("Alexander", [
    (103.16, 75.67), (102.72, 75.61), (102.47, 75.72), (102.54, 75.87),
    (102.98, 75.91), (103.20, 75.81),
])

# Cornwallis and the small islands east of Bathurst.
island("Cornwallis", [
    (96.55, 74.99), (96.19, 74.80), (95.65, 74.71), (95.04, 74.69),
    (94.63, 74.75), (94.37, 74.91), (93.92, 75.00), (93.52, 75.23),
    (93.48, 75.47), (93.70, 75.60), (94.14, 75.63), (94.49, 75.53),
    (94.92, 75.64), (95.42, 75.60), (95.80, 75.44), (96.24, 75.39),
    (96.52, 75.18),
])
island("Little Cornwallis", [
    (96.87, 75.33), (96.47, 75.34), (96.14, 75.48), (96.20, 75.67),
    (96.60, 75.71), (96.94, 75.59),
])
island("Griffith", [
    (95.68, 74.49), (95.12, 74.48), (94.90, 74.59), (95.18, 74.67),
    (95.64, 74.62),
])
island("Lowther", [
    (97.72, 74.49), (97.18, 74.44), (96.91, 74.54), (97.22, 74.66),
    (97.62, 74.62),
])
island("Garrett", [
    (96.45, 74.58), (96.20, 74.56), (96.12, 74.63), (96.35, 74.68),
])
island("Young", [
    (98.08, 75.02), (97.64, 75.02), (97.54, 75.11), (97.91, 75.18),
])
island("Davy", [
    (96.74, 75.85), (96.46, 75.82), (96.30, 75.91), (96.55, 75.99),
])
