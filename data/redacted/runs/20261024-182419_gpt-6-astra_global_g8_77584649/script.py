"""Knowledge-only approximate world land mask; no geographic data input.

All outlines below are handwritten from geographic knowledge, not retrieved
coastlines. Grid cells are classified by polygon rules, not independent
language-model judgments. Inland lakes remain land; sea ice remains water.
Dependencies: numpy, Pillow. Run: python outputs/generate_map.py
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

LAND = []
WATER = []

def polygon(name, coordinates, water=False):
    values = [float(v) for v in coordinates.replace(",", " ").split()]
    assert len(values) >= 6 and len(values) % 2 == 0, name
    points = np.asarray(values, dtype=np.float64).reshape(-1, 2)
    (WATER if water else LAND).append((name, points))

# North America: Pacific coast, Mexico, Atlantic coast, Labrador, Hudson Bay,
# Canadian mainland Arctic coast, and Alaska. Lakes are not cut out.
polygon("North American mainland", """
-168,66 -166,65 -164,64.5 -161,64.5 -161,63 -164,63
-166,61.5 -165,60 -163,59 -161,58.5 -158.5,58.5 -157,58
-159,56 -162,55 -165,54.5 -164,54 -161,55 -158,56
-156,57 -154,58.5 -153,59 -151,59 -150,60 -148,60.5
-146,60.5 -144,60 -141,59.5 -139,59 -137,58.5 -136,57
-135,56 -133,55 -131,54.5 -130,53.5 -129,52 -127.5,51
-125,50 -123.5,49.5 -123,48.5 -122.7,48 -122.5,47.3
-123,47 -123.5,48.2 -124.6,48.4 -124.5,46 -124,43
-124.4,42 -124,41 -123,39 -122.5,38 -122.5,37
-121.8,36.5 -121,35 -120.5,34.5 -119,34 -118.5,34
-117.2,32.5 -116,31 -114.7,30 -114.5,28 -113,27
-112,26 -112,24.5 -110,23 -109.5,23 -109.5,24
-111,26 -112,27 -112.8,28.5 -114.7,30.5 -114.8,31.8
-113,31 -112,29 -111,27.5 -110,27 -109,25.5 -108,25
-107,24 -106,23 -105.3,21.5 -105.7,20.5 -104.8,19.3
-103,18.5 -101,17.5 -99.5,16.5 -98,16 -96,15.7
-94.8,16.2 -93.5,15.5 -92.2,14.5 -90.5,13.8 -89,13.3
-87.8,13 -87.5,12.5 -86.5,11.8 -85.7,10.8 -85.7,9.8
-84.8,9.6 -84,9 -83.5,8.5 -83,8.3 -82.7,8.1
-82,8.2 -81.7,7.5 -80.8,7.2 -80,7.5 -80.5,8.2
-79.5,9 -79,8.5 -78.2,8.1 -77.3,7.5 -77.1,8.6
-78,9.4 -79.3,9.6 -80,9.3 -81,8.9 -82,9.3
-82.6,9.6 -83.5,10.5 -83.7,12 -83.5,13 -83.2,14.5
-83,15 -84.5,15.8 -86,15.8 -87.5,15.8 -88.5,15.8
-88.2,17 -88,18.5 -87.5,19.5 -86.8,21.3 -88,21.6
-90.3,21 -90.5,20 -91.5,18.5 -93,18.4 -94.5,18.2
-95.8,19 -96.5,20 -97.5,22 -97.8,24 -97.2,26
-97.3,27 -96,28 -95,29 -93.5,29.8 -92,29.5 -91,29.2
-90,29.2 -89.3,29 -89.5,30.2 -88,30.3 -86,30.4
