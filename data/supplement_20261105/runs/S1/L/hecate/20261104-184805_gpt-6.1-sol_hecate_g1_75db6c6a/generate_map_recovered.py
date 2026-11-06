from pathlib import Path
import csv
import numpy as np
from PIL import Image

# Knowledge-only reconstruction of the northern British Columbia coast.
# All vertices below are manually specified approximations (longitude, latitude).
# No external data, maps, network requests, or geographic libraries are used.
#
# Inland lakes are implicitly classified as land.
# Coastal channels are represented as explicit water polygons.
# This is a rule-based reconstruction, not independent reasoning per sample.

N = 127
OUT = Path("outputs")

# The actual sampling coordinates use the exact requested interval, 2/126 degree.
longitude = np.linspace(-131.0, -129.0, N)
latitude = np.linspace(55.0, 53.0, N)
X, Y = np.meshgrid(longitude, latitude)

LAND = [
    # Alaskan mainland near Cape Fox and Tree Point.
    [
        (-131.20,55.15),(-130.10,55.15),
        (-130.13,55.00),(-130.23,54.94),
        (-130.29,54.88),(-130.39,54.83),
        (-130.49,54.79),(-130.57,54.76),
        (-130.65,54.77),(-130.72,54.80),
        (-130.79,54.82),(-130.86,54.86),
        (-130.91,54.91),(-130.96,54.96),
        (-131.04,55.00),(-131.20,55.01),
    ],
    # Canadian coastal land mass. Narrow coastal waterways are cut out below.
    [
        (-129.99,55.15),(-128.75,55.15),(-128.75,52.85),
        (-129.18,52.85),(-129.22,53.00),
        (-129.29,53.10),(-129.34,53.19),
        (-129.40,53.25),(-129.44,53.34),
        (-129.51,53.42),(-129.57,53.49),
        (-129.61,53.58),(-129.67,53.65),
        (-129.71,53.73),(-129.77,53.81),
        (-129.83,53.87),(-129.87,53.95),
        (-129.93,54.00),(-129.99,54.06),
        (-130.09,54.12),(-130.16,54.18),
        (-130.23,54.23),(-130.29,54.27),
        (-130.34,54.32),(-130.38,54.39),
        (-130.43,54.45),(-130.46,54.51),
        (-130.44,54.57),(-130.40,54.62),
        (-130.38,54.67),(-130.32,54.70),
        (-130.27,54.74),(-130.23,54.79),
        (-130.17,54.84),(-130.12,54.90),
        (-130.07,54.95),(-130.03,55.01),
    ],
    # Dundas Island.
    [
        (-131.025,54.650),(-130.956,54.666),
        (-130.888,54.645),(-130.841,54.608),
        (-130.813,54.566),(-130.825,54.525),
        (-130.861,54.487),(-130.905,54.470),
        (-130.956,54.485),(-130.994,54.524),
        (-131.015,54.581),
    ],
    # Baron Island.
    [
        (-130.872,54.465),(-130.813,54.481),
        (-130.760,54.453),(-130.738,54.415),
        (-130.779,54.390),(-130.830,54.409),
    ],
    # Dunira Island.
    [
        (-130.977,54.433),(-130.930,54.448),
        (-130.887,54.418),(-130.902,54.381),
        (-130.950,54.376),(-130.980,54.401),
    ],
    # Melville Island.
    [
        (-130.861,54.371),(-130.802,54.380),
        (-130.755,54.347),(-130.782,54.318),
        (-130.833,54.325),
    ],
    # Stephens Island.
    [
        (-130.751,54.366),(-130.697,54.382),
        (-130.640,54.349),(-130.603,54.300),
        (-130.601,54.249),(-130.628,54.203),
        (-130.663,54.172),(-130.708,54.189),
        (-130.744,54.239),(-130.763,54.301),
    ],
    # Prescott Island.
    [
        (-130.607,54.229),(-130.559,54.247),
        (-130.525,54.221),(-130.529,54.172),
        (-130.561,54.149),(-130.593,54.174),
    ],
    # Digby Island, west of Prince Rupert.
    [
        (-130.490,54.367),(-130.437,54.380),
        (-130.395,54.347),(-130.385,54.309),
        (-130.418,54.273),(-130.458,54.282),
        (-130.480,54.319),
    ],
    # Kaien Island.
    [
        (-130.346,54.331),(-130.303,54.334),
        (-130.269,54.302),(-130.262,54.265),
        (-130.300,54.244),(-130.334,54.268),
        (-130.352,54.301),
    ],
    # Ridley Island.
    [
        (-130.366,54.256),(-130.338,54.257),
        (-130.316,54.220),(-130.332,54.200),
        (-130.359,54.221),
    ],
    # Smith Island, in the Skeena estuary.
    [
        (-130.285,54.188),(-130.226,54.206),
        (-130.164,54.178),(-130.134,54.137),
        (-130.165,54.108),(-130.221,54.126),
        (-130.271,54.143),
    ],
    # Porcher Island.
    [
        (-130.661,54.082),(-130.604,54.121),
        (-130.542,54.111),(-130.480,54.072),
        (-130.436,54.046),(-130.383,54.027),
        (-130.352,53.974),(-130.321,53.928),
        (-130.304,53.868),(-130.338,53.821),
        (-130.396,53.789),(-130.462,53.791),
        (-130.503,53.827),(-130.537,53.864),
        (-130.591,53.894),(-130.628,53.947),
        (-130.665,53.989),
    ],
    # Edye Passage / outer Porcher coastal islands.
    [
        (-130.735,54.044),(-130.696,54.066),
        (-130.666,54.038),(-130.671,54.004),
        (-130.705,53.988),(-130.733,54.012),
    ],
    [
        (-130.666,53.926),(-130.626,53.929),
        (-130.600,53.899),(-130.614,53.872),
        (-130.650,53.881),
    ],
    # McCauley Island.
    [
        (-130.341,53.844),(-130.286,53.862),
        (-130.224,53.829),(-130.174,53.788),
        (-130.160,53.732),(-130.184,53.688),
        (-130.232,53.675),(-130.274,53.714),
        (-130.305,53.767),
    ],
    # Dolphin Island.
    [
        (-130.384,53.796),(-130.349,53.799),
        (-130.317,53.756),(-130.325,53.714),
        (-130.354,53.701),(-130.384,53.747),
    ],
    # Banks Island.
    [
        (-130.314,53.650),(-130.258,53.675),
        (-130.196,53.652),(-130.137,53.602),
        (-130.085,53.541),(-130.039,53.469),
        (-129.999,53.394),(-129.961,53.313),
        (-129.932,53.230),(-129.925,53.157),
        (-129.955,53.103),(-130.009,53.094),
        (-130.065,53.145),(-130.107,53.207),
        (-130.146,53.281),(-130.182,53.354),
        (-130.229,53.426),(-130.268,53.503),
        (-130.301,53.576),
    ],
    # Pitt Island, between Principe and Grenville channels.
    [
        (-130.038,53.861),(-129.975,53.886),
        (-129.920,53.851),(-129.881,53.804),
        (-129.846,53.742),(-129.800,53.680),
        (-129.754,53.606),(-129.717,53.536),
        (-129.673,53.464),(-129.626,53.395),
        (-129.581,53.330),(-129.566,53.268),
        (-129.599,53.237),(-129.643,53.264),
        (-129.695,53.322),(-129.744,53.389),
        (-129.799,53.452),(-129.846,53.518),
        (-129.899,53.584),(-129.946,53.651),
        (-129.987,53.723),(-130.018,53.786),
    ],
    # Small islands in northern Principe Channel.
    [
        (-130.126,53.752),(-130.090,53.769),
        (-130.063,53.739),(-130.071,53.705),
        (-130.104,53.696),
    ],
    # Campania Island.
    [
        (-129.515,53.160),(-129.454,53.178),
        (-129.409,53.141),(-129.382,53.082),
        (-129.358,53.015),(-129.360,52.948),
        (-129.407,52.933),(-129.449,52.986),
        (-129.483,53.047),
    ],
    # Estevan Group, northern portions within the sampling frame.
    [
        (-129.758,53.111),(-129.707,53.125),
        (-129.658,53.088),(-129.638,53.037),
        (-129.658,52.978),(-129.709,52.967),
        (-129.747,53.020),
    ],
    [
        (-129.858,53.054),(-129.816,53.074),
        (-129.780,53.043),(-129.780,52.984),
        (-129.823,52.961),(-129.854,53.000),
    ],
]

WATER = [
    # Portland Inlet and Portland Canal, separating Alaska and Canada.
    [
        (-130.650,54.720),(-130.566,54.752),
        (-130.470,54.782),(-130.365,54.823),
        (-130.287,54.873),(-130.226,54.934),
        (-130.178,55.008),(-130.149,55.100),
        (-130.093,55.100),(-130.120,55.002),
        (-130.169,54.917),(-130.224,54.852),
        (-130.292,54.796),(-130.389,54.750),
        (-130.503,54.712),(-130.590,54.681),
    ],
    # Observatory Inlet branch.
    [
        (-130.253,54.835),(-130.165,54.867),
        (-130.089,54.923),(-130.019,54.988),
        (-129.970,55.060),(-129.927,55.070),
        (-129.978,54.970),(-130.055,54.902),
        (-130.136,54.843),(-130.225,54.807),
    ],
    # Work Channel, east of the Tsimpsean Peninsula.
    [
        (-130.323,54.683),(-130.256,54.657),
        (-130.197,54.598),(-130.154,54.536),
        (-130.115,54.466),(-130.079,54.408),
        (-130.056,54.368),(-130.035,54.373),
        (-130.057,54.426),(-130.092,54.489),
        (-130.129,54.553),(-130.177,54.617),
        (-130.239,54.677),(-130.298,54.708),
    ],
    # Prince Rupert harbour and eastern Kaien Island passage.
    [
        (-130.374,54.361),(-130.346,54.351),
        (-130.327,54.315),(-130.339,54.279),
        (-130.362,54.255),(-130.379,54.267),
        (-130.357,54.302),(-130.366,54.333),
    ],
    [
        (-130.280,54.334),(-130.252,54.334),
        (-130.237,54.294),(-130.241,54.258),
        (-130.274,54.230),(-130.291,54.243),
        (-130.265,54.274),(-130.263,54.303),
    ],
    # Skeena River estuary: only the marine / tidal estuary is cut out.
    [
        (-130.153,54.155),(-130.077,54.168),
        (-129.987,54.187),(-129.912,54.205),
        (-129.861,54.201),(-129.915,54.173),
        (-129.989,54.150),(-130.066,54.130),
        (-130.131,54.120),
    ],
    # Grenville Channel.
    [
        (-129.975,53.961),(-129.922,53.917),
        (-129.867,53.846),(-129.819,53.771),
        (-129.771,53.694),(-129.723,53.619),
        (-129.671,53.540),(-129.618,53.462),
        (-129.563,53.383),(-129.507,53.306),
        (-129.477,53.272),(-129.491,53.254),
        (-129.532,53.293),(-129.587,53.371),
        (-129.642,53.449),(-129.696,53.528),
        (-129.749,53.607),(-129.798,53.682),
        (-129.845,53.760),(-129.894,53.833),
        (-129.948,53.902),(-129.995,53.941),
    ],
    # Douglas Channel, cutting the mainland near the eastern frame edge.
    [
        (-129.365,53.365),(-129.279,53.427),
        (-129.201,53.500),(-129.130,53.584),
        (-129.056,53.668),(-128.980,53.755),
        (-128.934,53.739),(-129.025,53.636),
        (-129.098,53.553),(-129.171,53.474),
        (-129.254,53.399),(-129.332,53.339),
    ],
    # Verney Passage and the water north of Princess Royal Island.
    [
        (-129.493,53.285),(-129.404,53.317),
        (-129.316,53.356),(-129.221,53.383),
        (-129.121,53.398),(-129.075,53.380),
        (-129.192,53.354),(-129.290,53.329),
        (-129.382,53.289),(-129.469,53.259),
    ],
    # Laredo Channel along the western shore of Princess Royal Island.
    [
        (-129.466,53.246),(-129.410,53.199),
        (-129.370,53.133),(-129.334,53.065),
        (-129.300,52.980),(-129.264,52.960),
        (-129.302,53.074),(-129.340,53.149),
        (-129.386,53.218),(-129.437,53.264),
    ],
    # A coastal inlet north of the Skeena.
    [
        (-130.000,54.498),(-129.932,54.521),
        (-129.860,54.566),(-129.793,54.607),
        (-129.770,54.601),(-129.846,54.548),
        (-129.921,54.504),(-129.987,54.480),
    ],
]


def inside_polygon(vertices):
    """Vectorized point-in-polygon using the even-odd crossing rule."""
    result = np.zeros(X.shape, dtype=bool)
    x0, y0 = vertices[-1]
    for x1, y1 in vertices:
        if y1 != y0:
            crosses_y = (y0 > Y) != (y1 > Y)
            intersection_x = x0 + (Y - y0) * (x1 - x0) / (y1 - y0)
            result ^= crosses_y & (X < intersection_x)
        x0, y0 = x1, y1
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    land_mask = np.zeros((N, N), dtype=bool)
    for polygon in LAND:
        land_mask |= inside_polygon(polygon)
    for polygon in WATER:
        land_mask &= ~inside_polygon(polygon)

    matrix = land_mask.astype(np.uint8)
    assert matrix.shape == (127, 127)
    assert matrix.size == 16129
    assert np.isin(matrix, (0, 1)).all()
    assert latitude[0] == 55.0 and latitude[-1] == 53.0
    assert longitude[0] == -131.0 and longitude[-1] == -129.0

    csv_path = OUT / "candidate.csv"
    with csv_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Direct conversion: every sample becomes exactly one black or white pixel.
    pixels = matrix * np.uint8(255)
    original = Image.fromarray(pixels)
    original.save(OUT / "candidate.png")

    nearest = (
        Image.Resampling.NEAREST
        if hasattr(Image, "Resampling")
        else Image.NEAREST
    )
    enlarged = original.resize((1016, 1016), resample=nearest)
    enlarged.save(OUT / "candidate_8x.png")

    # Verify the files actually written, rather than only in-memory dimensions.
    with csv_path.open("r", newline="", encoding="ascii") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 127
    assert all(len(row) == 127 for row in rows)
    assert all(value in {"0", "1"} for row in rows for value in row)
    restored = np.asarray([[int(value) for value in row] for row in rows],
                          dtype=np.uint8)
    assert np.array_equal(restored, matrix)

    with Image.open(OUT / "candidate.png") as image:
        assert image.size == (127, 127)
        saved_original = np.asarray(image.convert("L"))
    with Image.open(OUT / "candidate_8x.png") as image:
        assert image.size == (1016, 1016)
        saved_enlarged = np.asarray(image.convert("L"))

    assert set(np.unique(saved_original).tolist()) <= {0, 255}
    assert set(np.unique(saved_enlarged).tolist()) <= {0, 255}
    assert np.array_equal(saved_original, pixels)
    expected = np.repeat(np.repeat(saved_original, 8, axis=0), 8, axis=1)
    assert np.array_equal(saved_enlarged, expected)

    print("Verified: 127 x 127 matrix; 16,129 valid binary samples.")
    print("Verified: CSV has 127 rows, each containing 127 binary values.")
    print("Verified: PNGs contain only black and white.")
    print("Verified: 1016 x 1016 PNG is an exact 8x nearest-neighbor enlargement.")
    print(f"Land samples: {int(matrix.sum())}")
    print(f"Water samples: {int(matrix.size - matrix.sum())}")
    for name in ("candidate.csv", "candidate.png", "candidate_8x.png"):
        print(OUT / name)


if __name__ == "__main__":
    main()
