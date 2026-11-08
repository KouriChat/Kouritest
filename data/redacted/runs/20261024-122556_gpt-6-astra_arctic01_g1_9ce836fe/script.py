"""Knowledge-only, approximate Canadian Arctic land/water reconstruction.

All coastlines below are hand-written estimates from geographic knowledge.
No maps, geographic databases, coastline files, or network access are used.
Classification is polygon membership at the requested POINTS, not cell voting
or independent language-model reasoning for each point. Inland lakes remain
land. Explicit marine inlets are water, irrespective of sea ice.

Run: python outputs/generate_map.py
Dependencies: Python 3, numpy, Pillow.
Outputs are relative to the current working directory.
"""
from pathlib import Path
import csv
import numpy as np

