"""Knowledge-only approximation; no external maps, datasets, or network access."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

LON, LAT = np.meshgrid(99 + np.arange(141) / 5,
                      23 - np.arange(111) / 5)
land = np.zeros((111, 141), dtype=bool)
features = []

def polygon(name, text):
    """Union a hand-written polygon, testing the sampling points themselves."""
    points = np.array([list(map(float, p.split()))
                       for p in text.strip().split(",")])
    x, y = LON, LAT
    inside = np.zeros(x.shape, dtype=bool)
    boundary = np.zeros(x.shape, dtype=bool)
    for i in range(len(points)):
        ax, ay = points[i - 1]
        bx, by = points[i]
        if ay != by:
            cross_x = ax + (y - ay) * (bx - ax) / (by - ay)
            inside ^= ((ay > y) != (by > y)) & (x < cross_x)
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        boundary |= ((np.abs(cross) < 1e-10) &
                     (x >= min(ax, bx) - 1e-10) &
                     (x <= max(ax, bx) + 1e-10) &
                     (y >= min(ay, by) - 1e-10) &
                     (y <= max(ay, by) + 1e-10))
    land[:] |= inside | boundary
    features.append(name)

def island(name, lon, lat, rx, ry):
    """Small-island approximation by a manually specified ellipse."""
    land[:] |= ((LON - lon) / rx)**2 + ((LAT - lat) / ry)**2 <= 1
    features.append(name)

# Mainland: the outer closure lies outside the sampling window.
polygon("Asian mainland", """
90 30,125 30,121 25,119 24.5,118 24.1,117.3 23.7,
116.8 23.45,116.6 23.25,116.5 23.05,116.25 22.95,
116 22.95,115.85 22.8,115.6 22.8,115.55 22.7,
115.2 22.8,115 22.76,114.75 22.8,114.87 22.65,
114.75 22.6,114.6 22.65,114.55 22.45,114.48 22.5,
114.3 22.55,114.25 22.5,114.35 22.3,114.25 22.25,
114.15 22.3,114.02 22.36,113.9 22.35,113.85 22.48,
113.68 22.55,113.65 22.8,113.55 22.95,113.5 22.8,
113.6 22.6,113.55 22.4,113.5 22.2,113.4 22.18,
113.3 22.35,113.2 22.55,113.08 22.55,113.08 22.2,
112.98 22.1,112.8 21.9,112.5 21.75,112.2 21.85,
111.9 21.85,111.65 21.75,111.3 21.5,111 21.5,
110.75 21.4,110.55 21.2,110.45 21.05,110.35 21.15,
110.4 21.3,110.2 21.25,110.15 21.05,110.4 20.95,
110.5 20.75,110.4 20.5,110.4 20.25,110.16 20.23,
