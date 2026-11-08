"""Knowledge-only reconstruction of land in 99--127 E, 1--23 N.

No map, GIS package, network service, or external input is used.  The coast
is represented by hand-written polygons, ellipses, and narrow island chains.
Every requested sample point is then tested against the same rules.
"""
from pathlib import Path
import csv
import math
import numpy as np
from PIL import Image


def in_poly(x, y, p):
    """Even/odd point-in-polygon test (boundary is treated as land)."""
    inside = False
    n = len(p)
    for i in range(n):
        x1, y1 = p[i]
        x2, y2 = p[(i + 1) % n]
        # Explicit point-on-segment test, with a small numerical tolerance.
        cross = (x-x1)*(y2-y1) - (y-y1)*(x2-x1)
        if abs(cross) < 1e-10 and min(x1,x2)-1e-10 <= x <= max(x1,x2)+1e-10 \
                and min(y1,y2)-1e-10 <= y <= max(y1,y2)+1e-10:
            return True
        if (y1 > y) != (y2 > y):
            xi = x1 + (y-y1) * (x2-x1) / (y2-y1)
            if x < xi:
                inside = not inside
    return inside


def ellipse(x, y, cx, cy, rx, ry, angle=0.0):
    a = math.radians(angle)
    dx, dy = x-cx, y-cy
    u = dx*math.cos(a) + dy*math.sin(a)
    v = -dx*math.sin(a) + dy*math.cos(a)
    return (u/rx)**2 + (v/ry)**2 <= 1.0


POLYS = []
ELLIPSES = []

# Continental SE Asia, including the Malay peninsula.  Vertices follow the
# remembered outer coast clockwise, with the Gulf of Thailand indentation.
POLYS.append([
 (99,23),(117.05,23),(116.75,22.85),(116.55,22.70),(116.15,22.62),
 (115.55,22.70),(115.15,22.78),(114.75,22.72),(114.48,22.57),
 (114.30,22.45),(114.05,22.38),(113.82,22.24),(113.55,22.18),
 (113.30,22.30),(112.95,21.98),(112.55,21.75),(112.05,21.62),
 (111.60,21.45),(111.05,21.25),(110.55,21.22),(110.15,21.05),
 (109.65,21.08),(109.10,21.42),(108.55,21.58),(108.10,21.55),
 (107.75,21.18),(107.42,20.92),(107.05,20.62),(106.72,20.25),
 (106.47,19.80),(106.10,19.35),(105.80,18.80),(105.72,18.25),
 (105.85,17.78),(106.22,17.35),(106.60,16.95),(107.12,16.55),
 (107.55,16.05),(108.15,15.45),(108.52,14.98),(108.82,14.35),
 (109.05,13.75),(109.22,13.15),(109.28,12.55),(109.18,11.90),
 (109.28,11.35),(109.10,10.85),(108.65,10.52),(107.95,10.42),
 (107.35,10.28),(106.78,10.05),(106.45,9.72),(106.02,9.58),
 (105.55,9.20),(105.05,8.65),(104.85,9.25),(104.88,9.85),
 (104.45,10.25),(103.95,10.38),(103.50,10.55),(103.15,10.88),
 (102.85,11.35),(102.55,11.75),(102.22,12.10),(101.72,12.62),
 (101.30,12.68),(100.92,13.05),(100.72,13.38),(100.48,13.56),
 (100.25,13.42),(100.05,13.02),(99.98,12.55),(99.96,11.98),
 (99.80,11.35),(99.70,10.72),(99.55,10.12),(99.30,9.52),
 (99.15,9.05),(99.45,8.55),(99.70,8.05),(99.82,7.55),
 (100.08,7.05),(100.35,6.55),(100.72,6.12),(101.05,5.65),
 (101.28,5.08),(101.65,4.48),(102.05,3.85),(102.42,3.25),
 (102.78,2.78),(103.18,2.25),(103.42,1.72),(103.62,1.30),
 (103.35,1.05),(102.92,1.15),(102.55,1.55),(102.05,2.05),
 (101.58,2.68),(101.15,3.25),(100.78,3.85),(100.55,4.40),
 (100.38,5.00),(100.15,5.55),(100.02,6.08),(99.73,6.55),
 (99.28,7.10),(99.02,7.85),(99,8.25)
])
