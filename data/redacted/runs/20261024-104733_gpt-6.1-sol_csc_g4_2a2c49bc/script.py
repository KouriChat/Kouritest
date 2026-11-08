"""Knowledge-only approximate South China Sea land mask.
No maps, geographic datasets, network requests, or reference masks are used.
All polygon vertices below were manually estimated from existing knowledge.
Coordinates are longitude,latitude; inland lakes are deliberately not cut out.
Classification is vectorized point-in-polygon, not independent model reasoning.
Run with Python 3, numpy and Pillow: python outputs/generate_map.py
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

POLYGONS = []

def add(name, text):
    vertices = [tuple(map(float, item.split(","))) for item in text.split()]
    assert len(vertices) >= 3, name
    POLYGONS.append((name, vertices))

# Mainland: clockwise from China along the coast, then around Malaya.
# Vertices outside the frame only close polygons; no outside cells are output.
add("Asian mainland", """
92,30 120,30 119.7,25.8 119.2,25.3 118.6,24.6 118.1,24.45
117.8,24.0 117.5,23.75 117.2,23.6 116.9,23.55 116.8,23.35
116.65,23.25 116.55,23.05 116.3,22.95 116.05,22.85 115.8,22.75
115.6,22.65 115.4,22.7 115.25,22.8 115.1,22.7 114.9,22.6
114.75,22.65 114.6,22.65 114.55,22.5 114.4,22.6 114.3,22.55
114.25,22.4 114.1,22.35 114.0,22.4 113.9,22.45 113.98,22.55
113.87,22.7 113.72,22.76 113.65,22.65 113.58,22.47 113.55,22.23
113.48,22.16 113.35,22.18 113.22,22.05 113.1,22.05 113.0,21.9
112.8,21.9 112.65,21.
