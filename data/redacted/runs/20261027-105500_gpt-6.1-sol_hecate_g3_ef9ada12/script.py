"""Knowledge-only approximation of the northern Hecate Strait coast.

No maps, geographical datasets, network requests, or reference masks are used.
All coordinates below are handwritten approximations from existing geographical
knowledge, not surveyed shorelines. Point classifications are computed from
these shared outline rules, not from independent reasoning about 16,129 points.
Inland lakes are deliberately not subtracted: this task counts them as land.
Requirements: Python 3, numpy, Pillow. Run from the desired working directory.
"""
from pathlib import Path
import csv
import math
import numpy as np
from PIL import Image

N = 127
LONGITUDE = np.linspace(-131.0, -129.0, N, dtype=np.float64)
LATITUDE = np.linspace(55.0, 53.0, N, dtype=np.float64)
X, Y = np.meshgrid(LONGITUDE, LATITUDE)
LAND = np.zeros((N, N), dtype=bool)
OUT = Path.cwd() / "outputs"


def polygon(vertices):
    """Even/odd point-in-polygon test at the actual sampling coordinates."""
    inside = np.zeros((N, N), dtype=bool)
    x0, y0 = vertices[-1]
    for x1, y1 in vertices:
        if y0 != y1:
            crossing = (y0 > Y) != (y1 > Y)
            edge_x = x0 + (Y - y0) * (x1 - x0) / (y1 - y0)
            inside ^= crossing & (X < edge_x)
        x0, y0 = x1, y1
    return inside


def land(name, vertices):
    """Add a named approximate land outline."""
    LAND[:] |= polygon(vertices)


def sea(name, vertices):
    """Subtract a named marine inlet, bay, or channel."""
    LAND[:] &= ~polygon(vertices)


def channel(name, route):
    """Subtract a waterway; route entries are (longitude, latitude, halfwidth_km).

    Distances use a fixed local equirectangular scale solely as a handwritten
    geometric approximation, not as a geographical-data query.
    """
    east_scale = 111.32 * math.cos(math.radians(54.0))
    xx, yy = X * east_scale, Y * 111.32
    wet = np.zeros_like(LAND)
    for (a, b, ra), (c, d, rb) in zip(route[:-1], route[1:]):
        ax, ay = a * east_scale, b * 111.32
        bx, by = c * east_scale, d * 111.32
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        if length2 == 0:
            continue
        t = np.clip(((xx - ax) * dx + (yy - ay) * dy) / length2, 0, 1)
        radius = ra + t * (rb - ra)
        wet |= (xx - ax - t * dx) ** 2 + (yy - ay - t * dy) ** 2 <= radius ** 2
    LAND[:] &= ~wet


def islet(name, lon, lat, east_radius, north_radius):
    """An explicitly placed elliptical small island; radii are degrees."""
    LAND[:] |= ((X - lon) / east_radius) ** 2 + ((Y - lat) / north_radius) ** 2 <= 1

# The bounding box does not intersect Graham or Moresby Island: their eastern
# shores are west of 131 W. Dixon Entrance and Hecate Strait start as water.

# Mainland east of Chatham Sound, Skeena estuary and Grenville Channel.
# The long indentation in the southern boundary is Douglas Channel.
land("British Columbia mainland", [
    (-128.70,55.25),(-130.105,55.25),(-130.125,55.07),
    (-130.145,55.00),(-130.155,54.953),(-130.110,54.925),
    (-130.025,54.910),(-129.950,54.925),(-129.965,54.883),
    (-130.025,54.845),(-130.085,54.801),(-130.120,54.766),
    (-130.210,54.715),(-130.310,54.670),(-130.400,54.635),
    (-130.445,54.597),(-130.465,54.565),(-130.440,54.533),
    (-130.454,54.493),(-130.425,54.461),(-130.470,54.420),
    (-130.449,54.380),(-130.439,54.346),(-130.402,54.329),
    (-130.345,54.350),(-130.287,54.355),(-130.251,54.386),
    (-130.226,54.420),(-130.187,54.425),(-130.171,54.398),
    (-130.201,54.358),(-130.192,54.315),(-130.149,54.283),
    (-130.161,54.250),(-130.128,54.227),(-129.967,54.249),
    (-129.774,54.283),(-129.557,54.337),(-129.401,54.389),
    (-129.382,54.370),(-129.552,54.312),(-129.777,54.253),
    (-129.892,54.193),(-129.947,54.147),(-129.935,54.113),
    (-129.993,54.086),(-130.054,54.070),(-130.075,54.034),
    (-130.040,53.997),(-129.969,53.955),(-129.927,53.917),
    (-129.894,53.869),(-129.829,53.818),(-129.785,53.763),
    (-129.717,53.709),(-129.666,53.648),(-129.605,53.593),
    (-129.568,53.550),(-129.495,53.495),(-129.419,53.451),
    (-129.333,53.408),(-129.274,53.406),(-129.284,53.458),
    (-129.321,53.515),(-129.319,53.577),(-129.287,53.653),
    (-129.263,53.706),(-129.202,53.760),(-129.141,53.812),
    (-129.082,53.859),(-129.005,53.910),(-128.825,53.983),
    (-128.70,54.00),
])

# Alaska's mainland shore near Boca de Quadra, Nakat Inlet and Cape Fox.
land("Alaska mainland", [
    (-131.35,55.25),(-130.175,55.25),(-130.202,55.112),
    (-130.259,55.051),(-130.330,55.006),(-130.383,54.968),
    (-130.425,54.925),(-130.450,54.886),(-130.511,54.847),
    (-130.559,54.819),(-130.591,54.778),(-130.623,54.769),
    (-130.653,54.789),(-130.688,54.822),(-130.747,54.852),
    (-130.805,54.869),(-130.897,54.900),(-130.958,54.940),
    (-131.022,54.965),(-131.105,55.011),(-131.35,55.04),
])
land("Pearse Island", [
    (-130.239,55.079),(-130.204,55.053),(-130.201,55.003),
    (-130.227,54.963),(-130.249,54.920),(-130.281,54.878),
    (-130.333,54.832),(-130.388,54.802),(-130.425,54.822),
    (-130.405,54.867),(-130.367,54.910),(-130.345,54.956),
    (-130.303,55.002),(-130.285,55.042),
])
land("Wales Island", [
    (-130.548,54.814),(-130.487,54.802),(-130.434,54.774),
    (-130.395,54.740),(-130.424,54.706),(-130.466,54.681),
    (-130.527,54.657),(-130.575,54.671),(-130.612,54.700),
    (-130.641,54.738),(-130.608,54.773),(-130.574,54.783),
])
land("Dundas Island", [
    (-130.965,54.706),(-130.907,54.702),(-130.873,54.680),
    (-130.817,54.668),(-130.778,54.632),(-130.756,54.596),
    (-130.790,54.575),(-130.782,54.545),(-130.833,54.523),
    (-130.889,54.528),(-130.907,54.549),(-130.958,54.532),
    (-130.997,54.558),(-131.020,54.596),(-131.006,54.640),
    (-130.980,54.660),
])
land("Melville Island", [
    (-130.918,54.525),(-130.872,54.514),(-130.831,54.481),
    (-130.850,54.449),(-130.893,54.455),(-130.945,54.485),
])
land("Baron Island", [
    (-130.806,54.525),(-130.771,54.513),(-130.733,54.477),
    (-130.754,54.446),(-130.802,54.465),(-130.832,54.496),
])
land("Dunira Island", [
    (-130.996,54.506),(-130.964,54.496),(-130.937,54.466),
    (-130.956,54.446),(-130.987,54.466),(-131.006,54.489),
])
islet("Tongass Island", -130.751,54.773,0.024,0.018)
islet("Sitklan Island", -130.688,54.771,0.035,0.026)
islet("Kanagunut Island", -130.731,54.814,0.029,0.023)
islet("Green Island", -130.709,54.565,0.014,0.010)
islet("Lucy Island", -130.613,54.296,0.014,0.009)

land("Stephens Island", [
    (-130.713,54.344),(-130.658,54.326),(-130.627,54.288),
    (-130.576,54.260),(-130.580,54.215),(-130.619,54.183),
    (-130.646,54.157),(-130.688,54.175),(-130.719,54.204),
    (-130.773,54.216),(-130.795,54.252),(-130.759,54.290),
])
land("Prescott Island", [
    (-130.722,54.166),(-130.664,54.159),(-130.608,54.137),
    (-130.624,54.113),(-130.683,54.124),(-130.737,54.141),
])
land("Digby Island", [
    (-130.464,54.378),(-130.429,54.353),(-130.411,54.328),
    (-130.402,54.292),(-130.415,54.265),(-130.448,54.253),
    (-130.477,54.275),(-130.493,54.311),(-130.485,54.340),
])
land("Kaien Island", [
    (-130.331,54.342),(-130.289,54.345),(-130.249,54.319),
    (-130.221,54.290),(-130.238,54.265),(-130.272,54.249),
    (-130.310,54.261),(-130.339,54.283),(-130.352,54.310),
])
land("Ridley Island", [
    (-130.347,54.277),(-130.321,54.258),(-130.312,54.229),
    (-130.331,54.217),(-130.355,54.242),
])
land("Smith Island", [
    (-130.298,54.223),(-130.245,54.218),(-130.215,54.193),
    (-130.224,54.161),(-130.265,54.135),(-130.298,54.153),
    (-130.319,54.186),
])
land("De Horsey Island", [
    (-130.144,54.219),(-130.107,54.218),(-130.073,54.191),
    (-130.095,54.163),(-130.142,54.172),(-130.168,54.193),
])
land("Kennedy Island", [
    (-130.064,54.171),(-130.010,54.172),(-129.976,54.145),
    (-129.988,54.116),(-130.032,54.096),(-130.080,54.113),
])
land("Porcher Island", [
    (-130.689,54.117),(-130.615,54.103),(-130.565,54.124),
    (-130.517,54.100),(-130.473,54.113),(-130.412,54.093),
    (-130.354,54.081),(-130.319,54.047),(-130.293,54.010),
    (-130.316,53.977),(-130.308,53.931),(-130.340,53.893),
    (-130.389,53.866),(-130.452,53.853),(-130.506,53.873),
    (-130.563,53.846),(-130.608,53.875),(-130.635,53.917),
    (-130.686,53.947),(-130.704,53.980),(-130.664,54.014),
    (-130.701,54.047),(-130.724,54.083),
])
land("Dolphin Island", [
    (-130.479,53.849),(-130.432,53.848),(-130.389,53.818),
    (-130.357,53.791),(-130.384,53.766),(-130.426,53.775),
    (-130.466,53.794),(-130.494,53.815),
])
land("Goschen Island", [
    (-130.630,53.836),(-130.581,53.831),(-130.533,53.803),
    (-130.516,53.766),(-130.555,53.733),(-130.603,53.752),
    (-130.639,53.790),
])
land("McCauley Island", [
    (-130.213,54.062),(-130.155,54.050),(-130.109,54.010),
    (-130.058,53.974),(-130.022,53.931),(-130.032,53.899),
    (-130.079,53.876),(-130.107,53.832),(-130.155,53.792),
    (-130.198,53.804),(-130.225,53.839),(-130.226,53.893),
    (-130.246,53.941),(-130.221,53.991),
])
land("Arthur Island", [
    (-130.280,53.977),(-130.253,53.963),(-130.246,53.931),
    (-130.257,53.894),(-130.285,53.882),(-130.296,53.924),
])
islet("Rachel islands", -130.334,54.118,0.018,0.020)
islet("Lawyer Islands", -130.351,54.098,0.016,0.013)
islet("Grace Island", -130.377,54.253,0.014,0.011)
islet("Kitson Island", -130.135,54.156,0.017,0.010)
islet("Spicer Island", -130.338,53.779,0.025,0.023)
islet("Kitkatla outer islet", -130.516,53.854,0.013,0.013)
islet("Browning Entrance islet", -130.670,53.769,0.018,0.012)

land("Pitt Island", [
    (-130.014,53.950),(-129.988,53.934),(-129.966,53.887),
    (-129.913,53.837),(-129.855,53.781),(-129.803,53.732),
    (-129.743,53.675),(-129.680,53.623),(-129.623,53.569),
    (-129.572,53.522),(-129.501,53.473),(-129.440,53.430),
    (-129.373,53.404),(-129.357,53.369),(-129.408,53.336),
    (-129.486,53.304),(-129.542,53.314),(-129.609,53.346),
    (-129.674,53.388),(-129.730,53.419),(-129.811,53.468),
    (-129.863,53.511),(-129.909,53.549),(-129.971,53.585),
    (-130.024,53.637),(-130.060,53.676),(-130.103,53.708),
    (-130.126,53.746),(-130.160,53.781),(-130.124,53.817),
    (-130.082,53.855),(-130.064,53.897),(-130.041,53.925),
])
land("Banks Island", [
    (-130.468,53.711),(-130.409,53.686),(-130.371,53.650),
    (-130.328,53.624),(-130.295,53.579),(-130.236,53.542),
    (-130.193,53.493),(-130.131,53.459),(-130.093,53.410),
    (-130.029,53.370),(-129.999,53.321),(-129.935,53.285),
    (-129.896,53.241),(-129.845,53.211),(-129.797,53.164),
    (-129.837,53.135),(-129.892,53.150),(-129.951,53.193),
    (-130.015,53.236),(-130.072,53.280),(-130.137,53.322),
    (-130.192,53.374),(-130.265,53.417),(-130.322,53.467),
    (-130.391,53.511),(-130.444,53.561),(-130.510,53.592),
    (-130.553,53.637),(-130.551,53.677),(-130.505,53.696),
])
land("Dewdney Island", [
    (-129.746,53.372),(-129.690,53.354),(-129.649,53.317),
    (-129.608,53.282),(-129.592,53.236),(-129.636,53.201),
    (-129.683,53.213),(-129.729,53.247),(-129.772,53.289),
    (-129.789,53.337),
])
land("Trutch Island", [
    (-129.825,53.241),(-129.774,53.223),(-129.725,53.198),
    (-129.692,53.159),(-129.709,53.117),(-129.763,53.095),
    (-129.813,53.119),(-129.853,53.153),(-129.868,53.201),
])
land("Barnard Island", [
    (-129.846,53.121),(-129.792,53.100),(-129.745,53.077),
    (-129.700,53.036),(-129.685,52.990),(-129.742,52.965),
    (-129.799,52.989),(-129.849,53.029),(-129.879,53.075),
])
land("Campania Island", [
    (-129.531,53.260),(-129.486,53.239),(-129.458,53.194),
    (-129.440,53.149),(-129.411,53.110),(-129.401,53.066),
    (-129.362,53.020),(-129.351,52.964),(-129.407,52.938),
    (-129.453,52.976),(-129.483,53.026),(-129.505,53.073),
    (-129.535,53.114),(-129.554,53.167),(-129.562,53.217),
])
land("Gil Island", [
    (-129.245,53.366),(-129.196,53.342),(-129.153,53.301),
    (-129.112,53.277),(-129.088,53.226),(-129.079,53.176),
    (-129.049,53.123),(-129.071,53.073),(-129.114,53.043),
    (-129.161,53.069),(-129.196,53.112),(-129.241,53.155),
    (-129.276,53.199),(-129.284,53.254),(-129.271,53.311),
])
land("Hawkesbury Island", [
    (-129.099,53.790),(-129.034,53.760),(-128.978,53.704),
    (-128.919,53.640),(-128.890,53.554),(-128.941,53.463),
    (-129.008,53.428),(-129.056,53.470),(-129.085,53.515),
    (-129.143,53.566),(-129.195,53.621),(-129.189,53.676),
    (-129.135,53.734),
])
land("Gribbell Island", [
    (-128.951,53.559),(-128.790,53.516),(-128.760,53.389),
    (-128.844,53.316),(-128.935,53.328),(-129.013,53.364),
    (-129.104,53.405),(-129.112,53.445),(-129.040,53.488),
])
land("Princess Royal Island northwest coast", [
    (-128.790,53.351),(-128.910,53.310),(-128.951,53.251),
    (-128.974,53.198),(-128.986,53.136),(-129.010,53.075),
    (-129.039,53.010),(-129.030,52.900),(-128.70,52.90),
    (-128.70,53.35),
])
islet("Bonilla Island", -130.638,53.491,0.028,0.019)
islet("Triple Islands", -130.880,54.294,0.009,0.006)
islet("Prior Island", -129.938,53.073,0.030,0.034)
islet("Estevan outer islet", -129.931,53.132,0.017,0.017)

# Marine cutouts are applied after island outlines. They are geometric rules,
# not pixel editing, shoreline smoothing, or post-processing of the matrix.
channel("Boca de Quadra", [
    (-131.065,54.974,1.9),(-130.961,54.992,1.6),
    (-130.850,55.014,1.4),(-130.750,55.046,1.3),
    (-130.640,55.100,1.1),(-130.570,55.170,0.9),
])
channel("Nakat Inlet", [
    (-130.707,54.826,1.4),(-130.641,54.854,1.1),
    (-130.607,54.893,0.9),(-130.579,54.934,0.6),
])
channel("Portland Canal", [
    (-130.177,54.965,1.0),(-130.169,55.014,0.9),
    (-130.137,55.078,0.8),(-130.113,55.145,0.7),
])
channel("Observatory Inlet", [
    (-130.011,54.951,1.4),(-129.990,54.999,1.6),
    (-129.947,55.056,1.6),(-129.876,55.123,1.3),
])
sea("Nass Bay", [
    (-130.004,54.946),(-129.938,54.979),(-129.819,54.997),
    (-129.696,55.018),(-129.602,55.039),(-129.575,55.020),
    (-129.687,54.986),(-129.782,54.958),(-129.860,54.937),
    (-129.975,54.919),
])
channel("Khutzeymateen Inlet", [
    (-130.198,54.735,1.3),(-130.104,54.710,1.2),
    (-129.995,54.687,1.0),(-129.884,54.663,0.8),
    (-129.789,54.645,0.5),
])
channel("Kwinamass Bay", [
    (-130.091,54.809,1.0),(-129.984,54.797,1.0),
    (-129.867,54.793,0.7),(-129.752,54.814,0.4),
])
channel("Work Channel", [
    (-130.440,54.602,1.0),(-130.358,54.554,0.9),
    (-130.263,54.508,0.8),(-130.161,54.464,0.7),
    (-130.078,54.426,0.7),(-130.025,54.392,0.4),
])
channel("Quottoon Inlet", [
    (-130.105,54.440,0.65),(-130.029,54.466,0.65),
    (-129.956,54.483,0.4),
])
channel("Skeena River tidal continuation", [
    (-129.560,54.325,0.8),(-129.398,54.379,0.7),
    (-129.267,54.407,0.5),(-129.110,54.432,0.4),
    (-128.970,54.451,0.4),
])
channel("Dundas northeastern inlet", [
    (-130.804,54.664,0.65),(-130.849,54.646,0.55),
    (-130.906,54.636,0.4),
])
channel("Hunts Inlet", [
    (-130.463,54.126,0.8),(-130.438,54.085,0.7),
    (-130.457,54.050,0.45),
])
channel("Porcher southern inlet", [
    (-130.489,53.851,1.2),(-130.470,53.896,1.0),
    (-130.465,53.946,0.8),(-130.503,53.985,0.5),
])
channel("Welcome Harbour indentation", [
    (-130.700,54.003,0.8),(-130.642,54.009,0.7),
    (-130.599,54.027,0.4),
])
channel("Porcher eastern indentation", [
    (-130.287,54.010,0.6),(-130.349,54.006,0.5),
    (-130.395,54.023,0.35),
])
channel("Kumealon Inlet", [
    (-129.926,53.882,0.8),(-129.831,53.891,0.8),
    (-129.764,53.921,0.6),(-129.726,53.947,0.4),
])
channel("Klewnuggit Inlet", [
    (-129.716,53.682,0.8),(-129.632,53.696,0.8),
    (-129.564,53.715,0.6),(-129.515,53.745,0.35),
])
channel("Lowe Inlet", [
    (-129.601,53.558,0.7),(-129.548,53.567,0.6),
    (-129.508,53.592,0.35),
])
channel("Pitt western inlet approximation", [
    (-130.088,53.677,0.8),(-130.012,53.685,0.7),
    (-129.928,53.714,0.45),
])
channel("Banks eastern harbour approximation", [
    (-130.088,53.424,0.7),(-130.145,53.440,0.55),
    (-130.183,53.465,0.35),
])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    matrix = LAND.astype(np.uint8)
    assert matrix.shape == (127, 127) and matrix.size == 16129
    assert np.all((matrix == 0) | (matrix == 1))
    assert (LONGITUDE[0], LONGITUDE[-1]) == (-131.0, -129.0)
    assert (LATITUDE[0], LATITUDE[-1]) == (55.0, 53.0)
    assert np.all(np.diff(LONGITUDE) > 0)
    assert np.all(np.diff(LATITUDE) < 0)

    csv_path = OUT / "candidate.csv"
    original_path = OUT / "candidate.png"
    enlarged_path = OUT / "candidate_8x.png"
    with csv_path.open("w", encoding="ascii", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

    # Each matrix entry becomes exactly one pixel, without interpolation,
    # cropping, labels, antialiasing, repairs, or coastline beautification.
    pixels = matrix * np.uint8(255)
    original = Image.fromarray(pixels)
    assert original.mode == "L"
    original.save(original_path, format="PNG")
    nearest = getattr(Image, "Resampling", Image).NEAREST
    enlarged = original.resize((1016, 1016), resample=nearest)
    enlarged.save(enlarged_path, format="PNG")

    # Reopen the actual serialized deliverables rather than checking only
    # their in-memory precursors. No external geographical files are read.
    with csv_path.open("r", encoding="ascii", newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 127
    assert all(len(row) == 127 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    assert sum(len(row) for row in rows) == 16129
    reloaded = np.array(rows, dtype=np.uint8)
    assert np.array_equal(reloaded, matrix)

    with Image.open(original_path) as im:
        assert im.format == "PNG" and im.mode == "L"
        assert im.size == (127, 127)
        small_pixels = np.array(im)
    with Image.open(enlarged_path) as im:
        assert im.format == "PNG" and im.mode == "L"
        assert im.size == (1016, 1016)
        big_pixels = np.array(im)
    assert set(np.unique(small_pixels).tolist()) == {0, 255}
    assert set(np.unique(big_pixels).tolist()) == {0, 255}
    assert np.array_equal(small_pixels, reloaded * np.uint8(255))
    expected = np.repeat(np.repeat(small_pixels, 8, axis=0), 8, axis=1)
    assert np.array_equal(big_pixels, expected), "Not exact nearest-neighbor 8x"
    assert np.array_equal(big_pixels[::8, ::8], small_pixels)
    print("Validated: 127x127 CSV; 16,129 binary values; no empty cells.")
    print("Validated: black/white 127x127 PNG; exact nearest-neighbor 1016x1016 PNG.")
    print("Method: handwritten knowledge-only polygons and marine-channel rules.")
    print("Dependencies: numpy, Pillow. No geographical reference data used.")


if __name__ == "__main__":
    main()
