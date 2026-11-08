"""Knowledge-only Mediterranean reconstruction; no geographic files or services.

All outlines below are manually written approximations from geographic knowledge.
Classification is polygon containment, not independent reasoning at each point.
Inland lakes are deliberately not subtracted from the land polygons.
Run with Python 3, numpy and Pillow: python outputs/generate_map.py
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

POLYGONS = []

def polygon(name, coordinates):
    vertices = [tuple(map(float, item.split(",")))
                for item in coordinates.split()]
    assert len(vertices) >= 3, name
    POLYGONS.append((name, vertices))

# Mainland Europe: Atlantic, Iberia, France, Italy, Adriatic, Greece,
# Turkish Thrace, and the western Black Sea. Coordinates are (longitude, latitude).
europe = """
-12,48 -1.1,48 -1.1,45.6 -1.22,45.55 -1.15,45.35
-1.10,45.15 -1.05,45.05 -0.8,45.04 -0.65,45.1
-0.78,45.2 -0.9,45.3 -1.0,45.45 -1.02,45.6
-0.6,46 1,47 10,48 20,49 32,49
"""
# The northern closure is added at the end; start the actual coast at Gironde.
europe = """
-1.1,46 -1.12,45.7 -1.22,45.55 -1.15,45.35
-1.10,45.15 -1.05,45.05 -0.75,45.03 -0.65,45.10
-0.78,45.25 -1.00,45.48 -0.98,45.58 -0.8,45.6
-0.65,45.5 -0.53,45.2 -0.55,44.9 -0.65,44.85
-1.02,44.95 -1.16,45.0 -1.20,44.83 -1.25,44.65
-1.18,44.68 -1.07,44.73 -1.03,44.65 -1.13,44.58
-1.24,44.57 -1.25,44.3 -1.32,44.0 -1.43,43.7
-1.48,43.51 -1.66,43.40 -1.79,43.38 -1.98,43.32
-2.2,43.32 -2.43,43.38 -2.75,43.43 -2.97,43.38
-3.17,43.35 -3.43,43.45 -3.8,43.47 -4.0,43.48
-4.4,43.43 -4.75,43.42 -5.0,43.47 -5.4,43.55
-5.66,43.55 -5.84,
