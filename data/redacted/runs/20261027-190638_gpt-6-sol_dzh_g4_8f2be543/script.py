from pathlib import Path
import numpy as np
from PIL import Image


def inside(x, y, polygon):
    """Point-in-polygon test for hand-drawn longitude/latitude boundaries."""
    hit = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) < 1e-10 and (
            min(x1, x2) - 1e-10 <= x <= max(x1, x2) + 1e-10
            and min(y1, y2) - 1e-10 <= y <= max(y1, y2) + 1e-10
        ):
            return True
        if (y1 > y) != (y2 > y):
            if x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                hit = not hit
    return hit


def rect(x1, y1, x2, y2):
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


# Approximate European mainland and the Iberian Peninsula.
iberia = [
    (-9, 43.8), (-8, 43.8), (-7, 43.7), (-6, 43.5), (-5, 43.5),
    (-4, 43.4), (-3, 43.3), (-2, 43.3), (-1, 43), (-1, 42.5),
    (0, 42.4), (1, 42.5), (2, 42.6), (3.3, 42.3), (3.2, 41.5),
    (2.8, 41.2), (2, 41), (1.5, 40.5), (1, 40), (0, 39.5),
    (-1, 38.5), (-2, 37.5), (-3, 36.8), (-4, 36.5), (-5, 36.2),
    (-6, 36), (-7, 36), (-8, 36.2), (-9, 37),
]

north_europe = [
    (-5.6, 45), (37, 45), (37, 42.1), (34, 42), (31, 42),
    (29, 41.6), (27, 41.2), (25, 41.1), (23, 41), (22, 41.1),
    (21, 41.7), (20, 42.1), (19, 42.7), (18, 43.4), (17, 44),
    (16, 44.7), (15, 45), (13.3, 45.4), (12, 45.2), (11, 44.5),
    (10, 44), (9, 43.8), (8, 43.6), (7, 43.4), (6, 43.2),
    (5, 43.1), (4, 43), (3, 42.8), (2, 42.5), (1, 42.3),
    (-1, 42.8), (-3, 43.2), (-5.6, 43.4),
]
