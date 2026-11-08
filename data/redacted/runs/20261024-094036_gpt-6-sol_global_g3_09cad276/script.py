"""A hand-sketched, deliberately approximate world land/sea raster.

All boundaries below are original approximate longitude/latitude sketches;
no external coastline, geographic data set, or reference raster is loaded.
Lakes are intentionally treated as land. Floating sea ice is water.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

lon, lat = np.meshgrid(np.arange(-179, 180, 2), np.arange(89, -90, -2))
land = np.zeros((90, 180), dtype=bool)

def poly(vertices, value=True):
    """Paint an approximate polygon by testing all grid-point coordinates."""
    v = np.asarray([float(n) for n in vertices.split()], dtype=float).reshape(-1, 2)
    inside = np.zeros_like(land)
    for (x1, y1), (x2, y2) in zip(v, np.roll(v, -1, axis=0)):
        if y1 == y2:
            continue
        inside ^= ((y1 > lat) != (y2 > lat)) & (
            lon < (x2-x1)*(lat-y1)/(y2-y1)+x1
        )
    land[inside] = value

def oval(x, y, rx, ry, value=True):
    land[((lon-x)/rx)**2 + ((lat-y)/ry)**2 <= 1] = value

# North America, from Alaska and the Arctic coast to the Isthmus of Panama.
poly("""
-168 64 -165 67 -158 70 -149 71 -141 70 -136 70 -130 71
-124 70 -119 70 -113 68 -108 68 -102 69 -98 72 -94 72
-91 70 -87 68 -82 67 -78 66 -76 64 -70 62 -67 60
-63 59 -61 56 -59 54 -56 53 -56 51 -60 50 -62 48
-64 47 -67 45 -69 44 -70 42 -72 41 -73 40 -75 38
-76 36 -78 35 -80 33 -81 30 -81 28 -80 25 -82 24
-84 25 -85 29 -88 30 -90 30 -92 29 -95 29 -97 26
-97 24 -94 21 -91 19 -88 21 -87 19 -88 16 -90 16
-92 15 -91 14 -89 15 -87 15 -85 13 -84 10 -82 9
-80 8 -79 9 -78 9 -77 8 -78 7 -80 8 -82 9
-84 10 -85 11 -87 12 -90 13 -92 15 -94 16 -96 16
-98 17 -101 18 -104 19 -105 21 -107 23 -110 24
-111 27 -113 29 -114 31 -116 32 -117 33 -117 35
-120 36 -122 38 -123 40 -124 43 -124 46 -127 49
-128 51 -130 53 -134 55 -137 58 -141 60 -144 60
-148 59 -152 58 -155 57 -160 56 -164 54 -168 53
-166 56 -163 58 -161 60 -165 62
""")
# Baja California, separated from Sonora by the narrow Gulf of California.
poly("-117 33 -115 32 -113 30 -112 27 -111 25 -109 23 -110 24 -112 27 -114 28 -116 31")
# Alaska Peninsula and some eastern Aleutians.
poly("-163 59 -161 58 -159 57 -157 57 -155 56 -153 56 -151 58 -156 59")
for x, y, rx in [(-166,54,2.1), (-171,53,2.0), (-176,52,1.5),
                 (-179,52,1.0), (-161,55,1.1)]:
    oval(x, y, rx, 0.9)
# Hudson Bay and its open northern passage (Great Lakes are counted as land).
poly("""
-96 64 -94 61 -93 58 -91 56 -90 53 -87 52 -85 51
-82 52 -80 54 -79 57 -79 60 -81 63 -85 65 -90 65
""", False)
poly("-82 55 -80 54 -79 52 -78 51 -80 50 -82 51", False)
poly("-86 65 -86 68 -83 70 -79 71 -76 69 -78 66", False)
# Labrador inlet, and Newfoundland/Nova Scotia.
poly("-62 59 -62 56 -59 54 -56 53 -56 51 -59 50 -61 52", False)
poly("-59 52 -55 52 -53 49 -55 47 -59 47 -60 49")
poly("-66 46 -63 46 -60 45 -61 44 -63 45")
# Greenland: high Arctic land/ice, not the surrounding sea ice.
poly("""
-44 59 -40 60 -40 63 -38 65 -34 67 -29 70 -25 73
-22 76 -18 78 -12 81 -19 82 -29 83 -40 83 -49 82
-55 81 -60 79 -62 76 -58 74 -57 71 -54 68 -52 65
-49 62
""")
# Large Canadian Arctic Archipelago islands, with open water between them.
poly("-82 62 -76 64 -70 67 -64 68 -62 72 -67 74 -72 73 -76 70 -81 68")
poly("-88 76 -83 77 -79 80 -72 82 -65 82 -61 81 -68 79 -72 76 -79 75")
poly("-96 73 -91 73 -86 75 -87 77 -94 77 -98 75")
poly("-111 69 -106 69 -103 71 -104 74 -109 75 -114 72")
poly("-124 70 -118 70 -115 72 -117 74 -122 74 -126 72")
poly("-101 68 -96 68 -94 71 -99 72 -104 70")
poly("-118 76 -114 77 -108 78 -102 77 -105 75 -113 74")
poly("-102 78 -96 79 -94 80 -101 81 -107 80")
