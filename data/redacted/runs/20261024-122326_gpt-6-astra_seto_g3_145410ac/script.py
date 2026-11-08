"""Knowledge-only approximate coastline experiment; no geographic inputs.
Run from the working directory: python outputs/generate_map.py
Dependencies: numpy, Pillow. Inland lakes are deliberately classified as land.
Coordinates below are hand-written approximations, not independently inferred
labels for each sample. Ray casting applies these rules to all sample points.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)
LON, LAT = np.meshgrid(129 + np.arange(253) * (2 / 63),
                      35 - np.arange(64) * (2 / 63))
LAND = np.zeros((64, 253), dtype=bool)

def polygon(text, land=True):
    """Classify sample points against a manually specified lon,lat polygon."""
    vertices = [tuple(map(float, p.split(","))) for p in text.split()]
    inside = np.zeros(LAND.shape, dtype=bool)
    x0, y0 = vertices[-1]
    for x1, y1 in vertices:
        if y0 != y1:
            crossing = ((y0 > LAT) != (y1 > LAT))
            xcross = (x1 - x0) * (LAT - y0) / (y1 - y0) + x0
            inside ^= crossing & (LON < xcross)
        x0, y0 = x1, y1
    if land:
        LAND[inside] = True
    else:
        LAND[inside] = False

def island(lon, lat, a, b, angle=0):
    """Small-island fallback: a hand-sized, rotated elliptic footprint."""
    theta = np.deg2rad(angle)
    dx, dy = LON - lon, LAT - lat
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    LAND[(u / a) ** 2 + (v / b) ** 2 <= 1] = True

# Main Honshu: Japan Sea coast, then the Seto Inland Sea coast eastwards.
polygon("""
138.5,36.5 133.0,36.5 132.72,35.43 132.64,35.34
132.55,35.27 132.45,35.18 132.35,35.12 132.26,35.07
132.17,35.01 132.08,34.94 132.00,34.89 131.93,34.86
131.86,34.79 131.78,34.74 131.72,34.70 131.65,34.66
131.60,34.65 131.56,34.60 131.49,34.56 131.43,34.48
131.40,34.43 131.34,34.42 131.28,34.43 131.24,34.47
131.18,34.45 131.12,34.42 131.09,34.38 131.02,34.40
130.99,34.43 130.94,34.41 130.93,34.36 130.88,34.34
130.88,34.30 130.91,34.27 130.88,34.24 130.92,34.21
130.90,34.17 130.90,34.13 130.87,34.08 130.89,34.04
130.92,34.00 130.94,33.96 130.96,33.94 130.99,33.96
131.00,33.98 131.03,33.99 131.06,34.01 131.12,34.02
131.16,34.00 131.18,33.96 131.20,33.94 131.22,33.92
131.26,33.93 131.29,33.95 131.33,33.96 131.38,34.00
131.42,34.02 131.45,34.00 131.43,33.96 131.40,33.94
