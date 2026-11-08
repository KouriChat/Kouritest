"""Memory-only Mediterranean reconstruction; no geographic data inputs.
Run from the working directory: python outputs/generate_map.py
Dependencies: numpy, Pillow. Lakes are land by the task's definition.
All coast vertices below are handwritten approximations, not map extracts.
Classification is point-in-polygon, NOT independent model inference per point.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)
LONS = (np.arange(231, dtype=np.int32) - 45) / 5.0
LATS = (225 - np.arange(71, dtype=np.int32)) / 5.0
X, Y = np.meshgrid(LONS, LATS)
LAND = np.zeros(X.shape, dtype=bool)
POLYGONS = {}

def polygon(name, text, water=False):
    """Union a closed polygon; exact boundary points count as polygon."""
    vertices = [tuple(map(float, s.split(","))) for s in text.split()]
    assert len(vertices) >= 3
    inside = np.zeros(X.shape, dtype=bool)
    boundary = np.zeros(X.shape, dtype=bool)
    x0, y0 = vertices[-1]
    for x1, y1 in vertices:
        cross = (X-x0)*(y1-y0) - (Y-y0)*(x1-x0)
        boundary |= ((np.abs(cross) < 1e-10)
                     & (X >= min(x0,x1)-1e-10)
                     & (X <= max(x0,x1)+1e-10)
                     & (Y >= min(y0,y1)-1e-10)
                     & (Y <= max(y0,y1)+1e-10))
        if y1 != y0:
            inside ^= (((y0 > Y) != (y1 > Y))
                       & (X < x0+(Y-y0)*(x1-x0)/(y1-y0)))
        x0, y0 = x1, y1
    mask = inside | boundary
    if water:
        LAND[mask] = False
    else:
        LAND[mask] = True
    POLYGONS[name] = vertices

# Vertices use longitude,latitude. Exterior closures are outside the frame.
# Main western-European coast, including Italy; closure across the Alps.
