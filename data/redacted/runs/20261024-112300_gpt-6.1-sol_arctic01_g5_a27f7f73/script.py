"""Memory-only Canadian Arctic reconstruction; no geographic data are read.

Run: python outputs/generate_map.py
Dependencies: numpy, Pillow.
Coordinates in the hand-written outlines are (degrees WEST, degrees NORTH).
These are approximate polygons, not independent judgments at every grid point.
Inland lakes are deliberately not subtracted. Sea ice is not treated as land.
"""
from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image

LAND = []
SEA = []


def outline(name, coordinates, water=False):
    vertices = [
        tuple(map(float, pair.split(",")))
        for pair in coordinates.split()
    ]
    assert len(vertices) >= 3, name
    (SEA if water else LAND).append((name, vertices))


# Devon: Grinnell Peninsula, the main island and its south/east coasts.
outline("Devon", """
96.78,76.63 96.46,76.82 95.82,76.94 95.12,77.02
94.43,76.96 93.79,76.88 93.19,76.79 92.59,76.68
92.02,76.58 91.31,76.48 90.66,76.40 90.03,76.34
89.52,76.27 89.20,76.24 88.81,76.38 88.06,76.43
87.45,76.48 86.77,76.48 86.09,76.44 85.43,76.43
84.77,76.38 84.18,76.37 83.64,76.33 83.00,76.31
82.46,76.24 81.95,76.21 81.45,76.17 80.92,76.12
80.40,75.99 80.13,75.81 79.79,75.67 79.60,75.40
79.77,75.15 80.11,74.94 80.51,74.76 81.00,74.61
81.46,74.53 82.02,74.54 82.39,74.57 82.89,74.54
83.25,74.58 83.76,74.57 84.14,74.62 84.66,74.55
85.22,74.53 85.82,74.49 86.42,74.47 87.04,74.46
87.61,74.48 88.19,74.49 88.78,74.53 89.36,74.56
89.91,74.57 90.50,74.61 90.98,74.59 91.45,74.62
91.91,74.66 92.35,74.63 92.85,74.64 93.29,74.66
93.62,74.71 93.66,74.93 93.79,75.13 94.01,75.31
94.29,75.51 94.67,75.68 95.06,75.85 95.53,76.01
96.04,76.11 96.47,76.27 96.64,76.44
""")

# Cornwallis, west of Wellington Channel.
outline("Cornwallis", """
96.27,75.47 95.97,75.60 95.57,75.63 95.17,75.59
94.78,75.62 94.39,75.52 94.12,75.36 93.93,75.15
93.79,74.96 93.89,74.79 94.16,74.70 94.55,74.69
94.87,74.70 95.16,74.75 95.44,74.81 95.66,74.96
95.89,75.03 96.09,75.19 96.21,75.32
""")

outline("Beechey", """
92.03,74.74 91.86,74.75 91.78,74.71 91.86,74.68 92.00,74.69
""")
outline("Griffith", """
95.87,74.72 95.57,74.76 95.22,74.65 95.13,74.53
95.40,74.48 95.71,74.52 95.88,74.62
""")
outline("Somerville", """
96.77,74.73 96.46,74.73 96.29,74.63 96.44,74.54
96.70,74.57 96.85,74.65
""")
outline("Lowther", """
97.84,74.66 97.53,74.65 97.27,74.55 97.29,74.44
97.58,74.40 97.83,74.49
""")
outline("Brown", """
96.91,74.87 96.62,74.88 96.52,74.79 96.73,74.74 96.93,74.78
""")
