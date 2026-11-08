"""Knowledge-only, deliberately approximate 2-degree world land mask.

No files are read and no geographic package or data set is used.  The
coordinates below are hand-written recollections of coastlines.  Every sample
is classified by the same polygon/ellipse rules (not by 16,200 separate
judgements).
"""
from pathlib import Path
import csv
import math
import numpy as np
from PIL import Image

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

def inside_poly(x, y, p):
    """Even/odd ray test; boundary is treated as land."""
    hit = False
    x0, y0 = p[-1]
    for x1, y1 in p:
        # explicit point-on-segment test
        cross = (x-x0)*(y1-y0) - (y-y0)*(x1-x0)
        if abs(cross) < 1e-9 and min(x0,x1) <= x <= max(x0,x1) and min(y0,y1) <= y <= max(y0,y1):
            return True
        if (y0 > y) != (y1 > y):
            xi = x0 + (y-y0)*(x1-x0)/(y1-y0)
            if x < xi:
                hit = not hit
        x0, y0 = x1, y1
    return hit

def ellipse(x, y, cx, cy, rx, ry, angle=0):
    a = math.radians(angle)
    u = (x-cx)*math.cos(a) + (y-cy)*math.sin(a)
    v = -(x-cx)*math.sin(a) + (y-cy)*math.cos(a)
    return (u/rx)**2 + (v/ry)**2 <= 1

P = []
def poly(coords):
    P.append(coords)

# North America proper: hand-traced clockwise from Alaska's Arctic coast.
poly([(-168,72),(-158,71),(-150,70),(-141,69),(-134,69),(-127,71),
 (-120,74),(-110,75),(-99,76),(-92,74),(-84,73),(-79,70),(-82,66),
 (-88,63),(-91,60),(-87,57),(-82,55),(-79,51),(-73,47),(-69,46),
 (-66,44),(-67,40),(-74,39),(-75,36),(-80,33),(-81,29),(-83,25),
 (-88,21),(-91,19),(-96,16),(-101,17),(-105,20),(-108,24),(-112,29),
 (-117,32),(-120,35),(-124,39),(-125,44),(-126,49),(-130,54),
 (-135,58),(-141,60),(-148,60),(-154,58),(-160,59),(-166,62),
 (-170,65),(-168,72)])
# Alaska west/south and Aleutian base
poly([(-141,69),(-150,70),(-158,71),(-166,70),(-168,66),(-164,62),
 (-158,58),(-153,57),(-149,60),(-145,60),(-139,57),(-134,57),
 (-130,55),(-134,61),(-141,69)])
# Labrador/Newfoundland-like eastern extensions
poly([(-80,60),(-74,62),(-66,61),(-61,57),(-58,53),(-61,49),
 (-67,47),(-73,49),(-78,53),(-80,60)])
poly([(-59.5,51.5),(-54,49.5),(-53,47),(-56,46),(-59,47),(-59.5,51.5)])
# Florida
poly([(-83,31),(-80,30),(-80,26),(-81,24),(-83,26),(-83,31)])
# Central America (wide enough to classify the odd-degree samples sensibly)
poly([(-96,18),(-91,18),(-88,16),(-86,16),(-84,12),(-82,10),
 (-79,9),(-77,8),(-78,6),(-82,8),(-85,10),(-87,13),(-91,15),(-96,16)])

# Canadian Arctic islands (coarse individual masses; channels remain water).
poly([(-125,74),(-118,77),(-111,78),(-109,75),(-114,72),(-121,71)])
poly([(-123,78),(-116,82),(-103,82),(-96,79),(-101,76),(-113,77)])
poly([(-99,77),(-94,81),(-86,82),(-80,79),(-87,76),(-95,75)])
poly([(-86,68),(-80,72),(-74,73),(-68,70),(-72,67),(-80,66)])
poly([(-80,75),(-74,79),(-64,81),(-61,77),(-68,74)])
