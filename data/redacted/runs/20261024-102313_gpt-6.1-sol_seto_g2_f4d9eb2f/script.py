"""Knowledge-only, hand-drawn approximation; no geographic data is read."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)
lon = 129.0 + np.arange(253, dtype=np.float64) * (2.0 / 63.0)
lat = 35.0 - np.arange(64, dtype=np.float64) * (2.0 / 63.0)
X, Y = np.meshgrid(lon, lat)
land = np.zeros((64, 253), dtype=bool)

def polygon(text, water=False):
    """Ray crossing at the sample points, not pixel-area rasterization."""
    vertices = [tuple(map(float, p.split(","))) for p in text.split()]
    inside = np.zeros_like(land)
    x0, y0 = vertices[-1]
    for x1, y1 in vertices:
        if y1 != y0:
            crossing = (y0 > Y) != (y1 > Y)
            cut = x0 + (Y - y0) * (x1 - x0) / (y1 - y0)
            inside ^= crossing & (X < cut)
        x0, y0 = x1, y1
    if water:
        land[inside] = False
    else:
        land[inside] = True

def island(cx, cy, rx, ry, angle=0.0, n=20):
    """Small islands approximated by oriented ellipses."""
    a = np.arange(n) * (2 * np.pi / n)
    t = np.deg2rad(angle)
    dx, dy = rx * np.cos(a), ry * np.sin(a)
    xx = cx + dx * np.cos(t) - dy * np.sin(t)
    yy = cy + dx * np.sin(t) + dy * np.cos(t)
    polygon(" ".join(f"{x:.7f},{y:.7f}" for x, y in zip(xx, yy)))

# All subsequent vertices are manually recalled approximations (longitude,
# latitude). Lakes remain land. Main-island outlines are independent polygons.
