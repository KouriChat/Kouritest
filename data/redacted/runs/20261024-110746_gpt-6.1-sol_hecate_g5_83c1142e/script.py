"""Memory-only reconstruction of the northern Hecate Strait coast.

No map, geographic database, network, or reference mask is consulted.
Handwritten polygons and channel centre-lines classify the sample points;
these are not independent per-point language-model judgements.
Requires Python 3, numpy, Pillow. Run from the working directory.
"""
from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image

N = 127
LON, LAT = np.meshgrid(-131.0 + np.arange(N) * 2.0 / 126.0,
                       55.0 - np.arange(N) * 2.0 / 126.0)
land = np.zeros((N, N), dtype=np.uint8)


def polygon(vertices, value=1):
    """Ray crossing, evaluated at the requested points, not cell areas."""
    inside = np.zeros((N, N), dtype=bool)
    x0, y0 = vertices[-1]
    for x1, y1 in vertices:
        if y0 != y1:
            crosses = (y0 > LAT) != (y1 > LAT)
            x_at_y = x0 + (LAT - y0) * (x1 - x0) / (y1 - y0)
            inside ^= crosses & (LON < x_at_y)
        x0, y0 = x1, y1
    land[inside] = value


def channel(nodes):
    """Carve a sea inlet; nodes are (longitude, latitude, halfwidth_km)."""
    px = LON * 65.4
    py = LAT * 111.2
    for a, b in zip(nodes[:-1], nodes[1:]):
        ax, ay, ar = a[0] * 65.4, a[1] * 111.2, a[2]
        bx, by, br = b[0] * 65.4, b[1] * 111.2, b[2]
        dx, dy = bx - ax, by - ay
        t = np.clip(((px-ax)*dx + (py-ay)*dy)/(dx*dx+dy*dy), 0, 1)
        radius = ar + t * (br-ar)
        hit = (px-ax-t*dx)**2 + (py-ay-t*dy)**2 <= radius**2
        land[hit] = 0


# Canadian mainland, including the Tsimpsean peninsula.  The lower edge
# follows Grenville Channel, then turns northeast along Douglas Channel.
polygon([
    (-128.6,55.2),(-129.91,55.2),(-129.92,55.04),
    (-129.88,55.005),(-129.95,54.966),(-130.012,54.940),
    (-130.062,54.889),(-130.125,54.850),(-130.185,54.795),
    (-130.224,54.750),(-130.310,54.695),(-130.371,54.668),
    (-130.455,54.636),(-130.465,54.610),(-130.432,54.587),
    (-130.437,54.565),(-130.471,54.551),(-130.452,54.524),
    (-130.479,54.505),(-130.452,54.477),(-130.437,54.460),
    (-130.455,54.435),(-130.429,54.406),(-130.446,54.383),
    (-130.407,54.361),(-130.371,54.347),(-130.337,54.357),
    (-130.298,54.337),(-130.252,54.345),(-130.228,54.315),
    (-130.232,54.282),(-130.201,54.259),(-130.185,54.225),
    (-130.153,54.218),(-130.121,54.187),(-130.058,54.178),
    (-130.031,54.150),(-130.018,54.115),(-130.017,54.072),
    (-130.044,54.047),(-130.007,54.010),(-129.972,53.976),
    (-129.943,53.940),(-129.902,53.903),(-129.872,53.865),
    (-129.833,53.827),(-129.802,53.785),(-129.770,53.753),
    (-129.735,53.714),(-129.709,53.672),(-129.670,53.631),
    (-129.634,53.590),(-129.596,53.553),(-129.558,53.513),
    (-129.521,53.479),(-129.473,53.453),(-129.456,53.421),
    (-129.419,53.409),(-129.373,53.410),(-129.338,53.436),
    (-129.297,53.449),(-129.270,53.480),(-129.234,53.507),
    (-129.220,53.550),(-129.182,53.574),(-129.155,53.620),
    (-129.104,53.650),(-129.067,53.695),(-129.019,53.727),
    (-128.970,53.777),(-128.925,53.824),(-128.6,53.85)
])

# Alaska mainland, west of the Portland/Pearse waterways.
polygon([
    (-131.3,55.25),(-130.24,55.25),(-130.305,55.082),
    (-130.359,55.024),(-130.409,54.986),(-130.468,54.975),
    (-130.511,54.956),(-130.567,54.939),(-130.604,54.900),
    (-130.646,54.872),(-130.661,54.839),(-130.711,54.806),
    (-130.749,54.790),(-130.771,54.810),(-130.797,54.804),
    (-130.816,54.777),(-130.872,54.763),(-130.908,54.752),
    (-130.952,54.765),(-130.987,54.785),(-130.982,54.817),
    (-131.015,54.859),(-131.046,54.913),(-131.12,54.976),
    (-131.3,55.00)
])
