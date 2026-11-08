"""Knowledge-only reconstruction of a world land/water mask.

All vertices below are hand-written geographic approximations.  The program
does not read, download, or query any geographic data.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image


def inside(x, y, p):
    """Even/odd point-in-polygon test."""
    c = False
    j = len(p) - 1
    for i, (xi, yi) in enumerate(p):
        xj, yj = p[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi) + xi
        ):
            c = not c
        j = i
    return c


def ellipse(x, y, cx, cy, rx, ry, tilt=0.0):
    """A small-island primitive; tilt is the eastward lean per degree north."""
    xx = x - cx - tilt * (y - cy)
    return (xx / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1


# Polygon coordinates are (longitude, latitude).  Coastline detail is
# intentionally limited to what was recalled when writing this script.
POLYS = [
    # Alaska, Canada and the conterminous United States
    [(-168,71),(-155,72),(-141,70),(-136,68),(-128,70),(-120,74),
     (-110,76),(-96,77),(-82,75),(-71,70),(-64,62),(-58,55),
     (-61,50),(-67,47),(-70,44),(-75,41),(-74,39),(-77,35),
     (-80,32),(-81,26),(-82,25),(-90,29),(-97,26),(-97,29),
     (-105,30),(-111,32),(-117,33),(-124,40),(-125,49),(-130,54),
     (-137,58),(-146,60),(-153,58),(-160,59),(-166,64)],
    # Alaska's long south-west peninsula
    [(-165,66),(-158,64),(-153,60),(-153,57),(-158,56),(-164,58),
     (-170,59),(-177,56),(-173,53),(-164,54),(-157,57),(-151,59),
     (-141,60),(-136,64),(-141,69),(-154,71)],
    # Labrador/Newfoundland connection and maritime Canada
    [(-79,58),(-70,60),(-63,59),(-56,53),(-58,48),(-64,46),
     (-67,44),(-72,45),(-77,49)],
    # Mexico
    [(-117,33),(-111,31),(-106,31),(-103,29),(-100,26),(-97,26),
     (-96,20),(-91,18),(-88,21),(-87,19),(-91,16),(-96,16),
     (-101,18),(-105,21),(-106,24),(-111,27),(-114,29)],
    # Baja California
    [(-117,32),(-114,30),(-111,25),(-109,22),(-111,22),(-114,26)],
    # Central America
    [(-92,18),(-88,18),(-86,15),(-83,15),(-82,10),(-78,9),
     (-77,7),(-80,7),(-83,9),(-85,11),(-88,13),(-91,14)],
    # Greenland
    [(-73,78),(-61,82),(-43,84),(-22,82),(-17,77),(-22,70),
     (-29,65),(-40,60),(-49,60),(-56,65),(-61,70),(-69,75)],
    # Baffin Island
    [(-80,73),(-74,77),(-66,78),(-61,74),(-64,68),(-70,64),
     (-76,66)],
    # Ellesmere Island
    [(-85,83),(-74,84),(-61,82),(-63,77),(-73,76),(-82,78)],
    # Victoria and Banks islands
    [(-125,74),(-115,75),(-104,72),(-105,68),(-114,68),(-122,71)],
    [(-128,74),(-121,75),(-116,72),(-119,69),(-126,70)],
]
