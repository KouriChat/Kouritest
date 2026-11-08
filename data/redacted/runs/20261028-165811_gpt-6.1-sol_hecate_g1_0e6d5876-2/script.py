import csv
import os
import numpy as np
from PIL import Image

# Hand-authored coastline approximation for the Hecate Strait region.
# Coordinates are (longitude, latitude); no external geographic data is used.
LAND_POLYGONS = [
    # Mainland coast: from the Alaskan panhandle through the mainland fjords.
    [(-131.05,55.10),(-130.82,55.02),(-130.63,54.92),(-130.55,54.78),
     (-130.46,54.67),(-130.42,54.48),(-130.35,54.34),(-130.25,54.23),
     (-130.27,54.05),(-130.17,53.91),(-130.19,53.72),(-130.04,53.55),
     (-129.98,53.36),(-129.89,53.20),(-129.85,53.00),(-129.55,52.95),
     (-129.45,55.10)],
    # Graham Island, northern and largest part of Haida Gwaii.
    [(-131.02,54.30),(-130.91,54.33),(-130.77,54.30),(-130.64,54.25),
     (-130.52,54.21),(-130.47,54.11),(-130.49,53.99),(-130.56,53.90),
     (-130.62,53.79),(-130.71,53.72),(-130.80,53.75),(-130.88,53.84),
     (-130.95,53.95),(-131.04,54.06)],
    # Moresby Island, separated from Graham by narrow Skidegate Channel.
    [(-131.01,53.77),(-130.91,53.75),(-130.82,53.68),(-130.78,53.56),
     (-130.79,53.42),(-130.84,53.27),(-130.89,53.14),(-130.96,53.04),
     (-131.06,53.00),(-131.10,53.18),(-131.09,53.38),(-131.05,53.58)],
]

ISLANDS = [
    # Small coastal islands and islets, deliberately generalized.
    (-130.71,54.49,0.055,0.10), (-130.55,54.47,0.045,0.08),
    (-130.39,54.61,0.035,0.07), (-130.35,54.02,0.035,0.08),
    (-130.23,53.82,0.035,0.08), (-130.15,53.62,0.035,0.07),
    (-130.93,54.48,0.040,0.08), (-130.59,53.35,0.025,0.07),
    (-130.48,53.23,0.025,0.06), (-130.37,53.12,0.025,0.05),
    (-130.77,53.04,0.030,0.06), (-130.64,53.02,0.025,0.05),
]


def inside_polygon(x, y, polygon):
    """Ray-crossing point-in-polygon test."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            crossing_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < crossing_x:
                inside = not inside
        j = i
    return inside


def is_land(lon, lat):
    if any(inside_polygon(lon, lat, poly) for poly in LAND_POLYGONS):
        return True
    # Elliptical island approximations; these add land, never fill channels.
    for cx, cy, rx, ry in ISLANDS:
        if ((lon - cx) / rx) ** 2 + ((lat - cy) / ry) ** 2 <= 1.0:
            return True
    return False

def main():
    count = 127
    step = 2.0 / 126.0
    latitudes = [55.0 - i * step for i in range(count)]
    longitudes = [-131.0 + i * step for i in range(count)]
    matrix = np.array(
        [[1 if is_land(lon, lat) else 0 for lon in longitudes]
         for lat in latitudes],
        dtype=np.uint8,
    )

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/candidate.csv", "w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Matrix 1 is land: map it to white, and water 0 to black.
    image = Image.fromarray(matrix * np.uint8(255), mode="L").convert("RGB")
    image.save("outputs/candidate.png")
    image.resize((1016, 1016), resample=Image.Resampling.NEAREST).save(
        "outputs/candidate_8x.png"
    )

    assert matrix.shape == (127, 127)
    assert int(matrix.sum() + (matrix == 0).sum()) == 16129


if __name__ == "__main__":
    main()
