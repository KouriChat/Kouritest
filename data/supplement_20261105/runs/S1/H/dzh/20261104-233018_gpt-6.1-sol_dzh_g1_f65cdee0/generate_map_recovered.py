from pathlib import Path
import csv

import numpy as np
from PIL import Image


WEST = -9.0
EAST = 37.0
NORTH = 45.0
SOUTH = 31.0
STEP = 0.2
ROWS = 71
COLS = 231

# Each ring is a hand-written approximation in (longitude, latitude).
# These rules describe broad coastlines and selected islands; they are not
# sourced from a map or geographic database.
LAND_RINGS = [
    # Iberia, France, Italy, the Balkans, Greece, and mainland Europe.
    [
        (-9.0, 45.0), (-9.0, 43.8), (-8.2, 43.6), (-8.9, 42.1),
        (-9.0, 41.9), (-8.7, 41.8), (-8.9, 40.8), (-8.9, 39.6),
        (-9.0, 38.7), (-9.4, 37.0), (-7.4, 37.0), (-7.0, 37.2),
        (-6.0, 36.9), (-5.6, 36.0), (-5.2, 36.0), (-4.5, 36.7),
        (-3.0, 36.7), (-1.4, 36.7), (0.0, 37.1), (1.4, 38.7),
        (0.5, 40.5), (0.0, 41.0), (0.7, 41.3), (1.7, 42.5),
        (3.2, 42.4), (3.2, 43.0), (4.8, 43.4), (5.4, 43.0),
        (6.2, 43.1), (7.0, 43.7), (7.6, 43.8), (8.0, 44.1),
        (9.3, 44.3), (10.0, 44.0), (10.5, 43.8), (11.1, 42.9),
        (12.5, 41.8), (13.0, 41.0), (15.0, 40.0), (15.8, 39.0),
        (16.2, 38.0), (17.2, 37.9), (18.5, 39.0), (18.5, 40.0),
        (17.5, 40.7), (16.5, 41.9), (16.0, 42.5), (15.5, 43.0),
        (14.0, 44.0), (13.0, 45.0), (15.0, 45.0), (16.5, 45.6),
        (18.0, 45.8), (19.0, 45.0), (20.0, 44.8), (21.0, 43.8),
        (22.0, 43.0), (22.8, 41.9), (23.3, 41.0), (24.0, 40.8),
        (25.0, 40.9), (26.2, 40.7), (27.0, 41.0), (28.0, 41.3),
        (29.0, 41.2), (30.0, 41.3), (31.0, 41.1), (32.0, 41.5),
        (33.0, 42.0), (34.0, 42.3), (35.0, 43.0), (37.0, 44.0),
        (37.0, 45.0),
        # Close along the northern map boundary.
        (-9.0, 45.0),
    ],

    # North Africa and the Sahara within the sampling frame.
    [
        (-9.0, 31.0), (37.0, 31.0), (34.2, 31.3), (33.5, 31.2),
        (32.5, 31.1), (31.5, 31.5), (30.0, 31.4), (29.0, 31.2),
        (27.5, 31.2), (26.0, 31.5), (25.0, 31.7), (24.0, 31.7),
        (23.0, 32.0), (22.0, 32.6), (21.0, 32.9), (20.0, 32.9),
        (19.0, 32.9), (18.0, 32.7), (17.0, 32.5), (16.0, 32.4),
        (15.0, 32.4), (14.0, 32.5), (13.0, 32.8), (12.0, 33.2),
        (11.0, 33.3), (10.5, 33.9), (10.0, 34.5), (11.0, 34.8),
        (11.5, 35.2), (11.0, 36.0), (10.5, 36.8), (10.0, 37.2),
        (9.0, 37.3), (8.0, 36.9), (7.0, 36.8), (6.0, 37.1),
        (5.0, 36.9), (4.0, 36.8), (3.0, 36.8), (2.0, 36.7),
        (1.0, 36.6), (0.0, 36.0), (-1.5, 35.3), (-3.0, 35.2),
        (-4.5, 35.2), (-5.5, 35.8), (-6.5, 35.2), (-7.5, 34.8),
        (-8.5, 33.5), (-9.0, 31.0),
    ],

    # Anatolia and the eastern Mediterranean coast of Asia.
    [
        (26.2, 40.7), (26.0, 40.2), (26.3, 39.7), (26.5, 39.0),
        (26.7, 38.5), (27.0, 38.0), (27.3, 37.6), (27.5, 36.9),
        (28.0, 36.6), (29.0, 36.2), (30.0, 36.1), (31.0, 36.0),
        (32.0, 36.0), (33.0, 36.0), (34.0, 36.0), (35.0, 36.0),
        (36.0, 36.0), (37.0, 36.0), (37.0, 45.0), (35.0, 43.0),
        (34.0, 42.3), (33.0, 42.0), (32.0, 41.5), (31.0, 41.1),
        (30.0, 41.3), (29.0, 41.2), (28.0, 41.3), (27.0, 41.0),
        (26.2, 40.7),
    ],

    # Corsica.
    [(8.5, 43.0), (9.6, 43.0), (9.6, 41.4), (8.5, 41.4)],

    # Sardinia.
    [(8.1, 41.3), (9.8, 41.3), (9.8, 38.8), (8.1, 38.8)],

    # Sicily.
    [(12.3, 38.3), (15.7, 38.3), (15.7, 36.6), (12.3, 36.6)],

    # Balearic Islands.
    [(1.1, 40.0), (4.5, 40.0), (4.5, 38.5), (1.1, 38.5)],

    # Crete.
    [(23.4, 35.7), (26.4, 35.7), (26.4, 34.8), (23.4, 34.8)],

    # Cyprus.
    [(32.2, 35.7), (34.7, 35.7), (34.7, 34.5), (32.2, 34.5)],

    # Malta.
    [(14.1, 36.1), (14.7, 36.1), (14.7, 35.7), (14.1, 35.7)],

    # Larger Aegean islands and nearby island groups.
    [(24.0, 40.0), (26.2, 40.0), (26.2, 38.5), (24.0, 38.5)],
    [(25.0, 38.5), (27.0, 38.5), (27.0, 36.3), (25.0, 36.3)],
    [(26.0, 36.8), (28.0, 36.8), (28.0, 35.7), (26.0, 35.7)],
    [(19.0, 39.0), (21.0, 39.0), (21.0, 37.5), (19.0, 37.5)],

    # Ionian islands.
    [(19.4, 40.0), (20.3, 40.0), (20.3, 38.0), (19.4, 38.0)],

    # Adriatic islands, represented as a narrow coastal band.
    [(14.0, 45.0), (16.8, 45.0), (18.0, 43.0), (17.0, 42.0),
     (15.0, 43.0)],

    # Djerba and other larger near-shore islands in the southern frame.
    [(10.2, 33.9), (11.0, 33.9), (11.0, 33.5), (10.2, 33.5)],
]


def point_in_ring(lon, lat, ring):
    """Return whether a point lies inside a polygon using ray crossing."""
    inside = False
    count = len(ring)
    j = count - 1

    for i in range(count):
        xi, yi = ring[i]
        xj, yj = ring[j]

        # Treat points on a polygon edge as land.
        cross = (lon - xi) * (yj - yi) - (lat - yi) * (xj - xi)
        if (
            abs(cross) < 1e-10
            and min(xi, xj) - 1e-10 <= lon <= max(xi, xj) + 1e-10
            and min(yi, yj) - 1e-10 <= lat <= max(yi, yj) + 1e-10
        ):
            return True

        intersects = ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / (yj - yi) + xi
        )
        if intersects:
            inside = not inside
        j = i

    return inside


def is_land(lon, lat):
    return int(any(point_in_ring(lon, lat, ring) for ring in LAND_RINGS))


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    lats = [NORTH - STEP * row for row in range(ROWS)]
    lons = [WEST + STEP * col for col in range(COLS)]
    matrix = np.zeros((ROWS, COLS), dtype=np.uint8)

    for row, lat in enumerate(lats):
        for col, lon in enumerate(lons):
            matrix[row, col] = is_land(lon, lat)

    with (output_dir / "candidate.csv").open(
        "w", newline="", encoding="ascii"
    ) as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Matrix value 1 (land) is white; 0 (water) is black.
    image = Image.fromarray(matrix * 255, mode="L")
    image.save(output_dir / "candidate.png")

    enlarged = image.resize(
        (COLS * 8, ROWS * 8),
        resample=Image.Resampling.NEAREST,
    )
    enlarged.save(output_dir / "candidate_8x.png")

    # Validate dimensions, values, CSV shape, and the nearest-neighbor result.
    assert matrix.shape == (71, 231)
    assert int(matrix.size) == 16401
    assert set(np.unique(matrix).tolist()) <= {0, 1}
    assert image.size == (231, 71)
    assert set(image.getdata()) <= {0, 255}
    assert enlarged.size == (1848, 568)

    with (output_dir / "candidate.csv").open(
        "r", newline="", encoding="ascii"
    ) as csv_file:
        rows = list(csv.reader(csv_file))
    assert len(rows) == 71
    assert all(len(row) == 231 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)

    # Verify that every output pixel is an 8x8 replication of its source pixel.
    enlarged_array = np.asarray(enlarged)
    source_array = np.asarray(image)
    assert np.array_equal(
        enlarged_array[::8, ::8],
        source_array,
    )
    assert np.array_equal(
        enlarged_array,
        np.repeat(np.repeat(source_array, 8, axis=0), 8, axis=1),
    )


if __name__ == "__main__":
    main()
