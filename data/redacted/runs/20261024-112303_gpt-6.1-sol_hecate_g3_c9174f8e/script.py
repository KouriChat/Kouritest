"""Memory-only approximation: all geometry below is handwritten, not map data.

Run from the working directory with Python 3, numpy and Pillow installed.
Classification is point-in-polygon plus handwritten marine-channel exclusions,
not independent language-model reasoning at each sample. Inland lakes are land.
"""
from pathlib import Path
import csv
import json
import math
import numpy as np
from PIL import Image

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)
N = 127
longitude = -131.0 + np.arange(N, dtype=float) * (2.0 / 126.0)
latitude = 55.0 - np.arange(N, dtype=float) * (2.0 / 126.0)
X, Y = np.meshgrid(longitude, latitude)
land = np.zeros((N, N), dtype=bool)


def polygon(vertices):
    """Ray crossing at sample coordinates; vertices are (longitude, latitude)."""
    inside = np.zeros(X.shape, dtype=bool)
    x0, y0 = vertices[-1]
    for x1, y1 in vertices:
        if y0 != y1:
            crossing = ((y0 > Y) != (y1 > Y))
            edge_x = x0 + (Y - y0) * (x1 - x0) / (y1 - y0)
            inside ^= crossing & (X < edge_x)
        x0, y0 = x1, y1
    return inside


def add(vertices):
    land[:] |= polygon(vertices)


def remove(vertices):
    land[:] &= ~polygon(vertices)


# Northern coastal mainland envelope. Narrow straits will separate the
# memory-sketched Pearse and Wales islands from this envelope below.
add([
    (-130.86,55.12),(-130.85,55.00),(-130.80,54.96),
    (-130.79,54.91),(-130.73,54.88),(-130.71,54.82),
    (-130.65,54.77),(-130.60,54.755),(-130.55,54.775),
    (-130.51,54.746),(-130.46,54.735),(-130.41,54.755),
    (-130.35,54.775),(-130.30,54.805),(-130.24,54.830),
    (-130.18,54.853),(-130.12,54.882),(-130.04,54.904),
    (-129.98,54.929),(-129.90,54.945),(-129.84,54.974),
    (-129.78,55.045),(-129.70,55.12),
])

# Canadian mainland, Tsimpsean Peninsula, Skeena coast and the mainland
# northeast of Grenville Channel. Vertices outside the box only close shapes;
# the script classifies and exports exclusively the requested sample points.
add([
    (-128.80,55.14),(-129.53,55.14),(-129.58,55.015),
    (-129.69,54.954),(-129.79,54.917),(-129.87,54.865),
    (-129.97,54.824),(-130.07,54.779),(-130.17,54.742),
    (-130.27,54.711),(-130.35,54.689),(-130.43,54.674),
    (-130.465,54.650),(-130.455,54.611),(-130.427,54.576),
    (-130.447,54.553),(-130.453,54.520),(-130.420,54.487),
    (-130.434,54.454),(-130.397,54.420),(-130.391,54.386),
    (-130.354,54.367),(-130.352,54.337),(-130.366,54.312),
    (-130.341,54.282),(-130.310,54.261),(-130.287,54.232),
    (-130.246,54.222),(-130.205,54.197),(-130.152,54.190),
    (-130.103,54.182),(-130.018,54.178),(-129.973,54.157),
    (-129.952,54.115),(-129.958,54.072),(-129.978,54.035),
    (-129.965,53.993),(-129.989,53.954),(-129.978,53.914),
    (-129.936,53.870),(-129.890,53.823),(-129.847,53.783),
    (-129.787,53.737),(-129.738,53.689),(-129.686,53.646),
    (-129.634,53.594),(-129.577,53.552),(-129.534,53.519),
    (-129.477,53.481),(-129.412,53.447),(-129.357,53.423),
    (-129.304,53.435),(-129.264,53.479),(-129.238,53.535),
    (-129.179,53.570),(-129.130,53.610),(-129.072,53.646),
    (-129.017,53.685),(-128.955,53.720),(-128.80,53.75),
])

# Dundas group, from remembered relative positions and approximate dimensions.
add([
    (-130.96,54.703),(-130.89,54.697),(-130.85,54.669),
    (-130.81,54.651),(-130.795,54.614),(-130.779,54.591),
    (-130.800,54.557),(-130.786,54.533),(-130.823,54.506),
    (-130.835,54.474),(-130.879,54.461),(-130.919,54.481),
    (-130.943,54.511),(-130.965,54.532),(-130.951,54.565),
    (-130.992,54.592),(-131.019,54.623),(-130.999,54.650),
    (-131.015,54.672),
])
# Dunira, Baron and Melville, with water gaps retained between the islands.
add([(-131.071,54.612),(-131.030,54.605),(-131.000,54.574),
     (-131.012,54.550),(-131.054,54.544),(-131.083,54.576)])
add([(-131.009,54.491),(-130.971,54.488),(-130.942,54.458),
     (-130.946,54.430),(-130.981,54.420),(-131.009,54.449)])
add([(-130.954,54.423),(-130.903,54.425),(-130.863,54.410),
     (-130.840,54.381),(-130.852,54.355),(-130.890,54.339),
     (-130.925,54.354),(-130.947,54.381)])

# Stephens Island, facing the northern shore of Porcher across Edye Passage.
add([
    (-130.714,54.301),(-130.675,54.289),(-130.653,54.259),
    (-130.611,54.246),(-130.585,54.214),(-130.577,54.179),
    (-130.608,54.158),(-130.613,54.124),(-130.653,54.103),
    (-130.691,54.110),(-130.715,54.142),(-130.729,54.174),
    (-130.766,54.191),(-130.748,54.222),(-130.758,54.256),
])

# Digby Island, west of Prince Rupert Harbour.
add([(-130.475,54.376),(-130.430,54.365),(-130.416,54.339),
     (-130.391,54.319),(-130.399,54.285),(-130.421,54.267),
     (-130.430,54.235),(-130.470,54.223),(-130.491,54.254),
     (-130.525,54.275),(-130.503,54.302),(-130.516,54.329)])

# Porcher Island. Its marine embayments are subtracted in the last part.
add([
    (-130.686,54.112),(-130.642,54.105),(-130.600,54.115),
    (-130.555,54.100),(-130.519,54.113),(-130.482,54.089),
    (-130.439,54.074),(-130.421,54.044),(-130.387,54.017),
    (-130.379,53.978),(-130.352,53.952),(-130.374,53.926),
    (-130.366,53.891),(-130.403,53.862),(-130.445,53.850),
    (-130.465,53.819),(-130.506,53.811),(-130.541,53.832),
    (-130.589,53.842),(-130.617,53.861),(-130.669,53.867),
    (-130.690,53.899),(-130.731,53.919),(-130.724,53.958),
    (-130.758,53.978),(-130.745,54.014),(-130.759,54.044),
    (-130.717,54.066),(-130.730,54.090),
])

# Islands in and around the Skeena estuary.
add([(-130.297,54.187),(-130.248,54.182),(-130.203,54.160),
     (-130.196,54.127),(-130.218,54.099),(-130.204,54.066),
     (-130.238,54.025),(-130.272,54.041),(-130.291,54.082),
     (-130.320,54.112),(-130.315,54.153)])
add([(-130.164,54.122),(-130.132,54.139),(-130.091,54.123),
     (-130.035,54.125),(-130.001,54.096),(-130.013,54.054),
     (-130.031,54.022),(-130.018,53.992),(-130.055,53.960),
     (-130.094,53.950),(-130.121,53.983),(-130.122,54.028),
     (-130.150,54.070)])
add([(-130.145,54.181),(-130.105,54.174),(-130.089,54.150),
     (-130.119,54.144),(-130.153,54.157)])

# McCauley, Goschen, and Dolphin islands south/east of Porcher.
add([(-130.251,53.967),(-130.209,53.958),(-130.169,53.932),
     (-130.135,53.902),(-130.137,53.866),(-130.103,53.836),
     (-130.106,53.796),(-130.077,53.765),(-130.113,53.742),
     (-130.162,53.754),(-130.180,53.784),(-130.220,53.811),
     (-130.231,53.848),(-130.263,53.877),(-130.253,53.915)])
add([(-130.487,53.780),(-130.446,53.777),(-130.413,53.750),
     (-130.373,53.738),(-130.374,53.703),(-130.413,53.688),
     (-130.457,53.708),(-130.485,53.740)])
add([(-130.462,53.817),(-130.413,53.820),(-130.379,53.795),
     (-130.411,53.778),(-130.450,53.789)])

# Banks Island: a long NW--SE island, distinct from Pitt across Principe.
add([
    (-130.526,53.698),(-130.477,53.685),(-130.442,53.655),
    (-130.439,53.623),(-130.401,53.595),(-130.368,53.561),
    (-130.323,53.540),(-130.291,53.510),(-130.244,53.486),
    (-130.209,53.451),(-130.168,53.425),(-130.134,53.385),
    (-130.086,53.354),(-130.060,53.313),(-130.014,53.281),
    (-129.985,53.245),(-129.943,53.211),(-129.915,53.173),
    (-129.865,53.143),(-129.837,53.107),(-129.788,53.076),
    (-129.787,53.041),(-129.826,53.026),(-129.876,53.048),
    (-129.909,53.079),(-129.958,53.098),(-130.000,53.126),
    (-130.037,53.165),(-130.090,53.196),(-130.132,53.237),
    (-130.176,53.265),(-130.207,53.309),(-130.263,53.347),
    (-130.309,53.390),(-130.352,53.423),(-130.409,53.460),
    (-130.449,53.503),(-130.501,53.536),(-130.552,53.577),
    (-130.588,53.611),(-130.568,53.647),(-130.570,53.677),
])

# Pitt Island. The eastern shore follows the southwest side of Grenville.
add([
    (-130.113,53.956),(-130.081,53.932),(-130.055,53.902),
    (-130.030,53.874),(-129.987,53.836),(-129.959,53.810),
    (-129.920,53.770),(-129.886,53.742),(-129.843,53.705),
    (-129.803,53.668),(-129.769,53.630),(-129.723,53.600),
    (-129.681,53.567),(-129.644,53.535),(-129.607,53.505),
    (-129.570,53.476),(-129.532,53.448),(-129.488,53.421),
    (-129.451,53.391),(-129.412,53.366),(-129.397,53.331),
    (-129.370,53.303),(-129.381,53.272),(-129.425,53.246),
    (-129.473,53.245),(-129.515,53.268),(-129.552,53.293),
    (-129.603,53.312),(-129.652,53.345),(-129.697,53.379),
    (-129.741,53.405),(-129.781,53.445),(-129.827,53.469),
    (-129.872,53.503),(-129.912,53.535),(-129.957,53.571),
    (-129.978,53.610),(-130.022,53.646),(-130.041,53.682),
    (-130.085,53.720),(-130.091,53.759),(-130.112,53.791),
    (-130.093,53.827),(-130.116,53.863),(-130.104,53.899),
    (-130.138,53.928),
])

# Campania Island, south of Pitt beyond Otter Channel.
add([
    (-129.426,53.215),(-129.384,53.201),(-129.354,53.173),
    (-129.328,53.150),(-129.304,53.115),(-129.280,53.083),
    (-129.262,53.043),(-129.245,53.013),(-129.243,52.974),
    (-129.284,52.953),(-129.326,52.970),(-129.351,53.001),
    (-129.380,53.039),(-129.397,53.073),(-129.424,53.105),
    (-129.437,53.139),(-129.456,53.174),
])

# Gil Island and the western part of Gribbell Island.
add([(-129.137,53.337),(-129.090,53.319),(-129.056,53.282),
     (-129.035,53.249),(-129.006,53.221),(-128.982,53.178),
     (-128.985,53.134),(-129.009,53.099),(-129.037,53.072),
     (-129.071,53.053),(-129.116,53.076),(-129.149,53.112),
     (-129.169,53.149),(-129.190,53.184),(-129.198,53.224),
     (-129.179,53.263),(-129.172,53.303)])
add([(-128.987,53.586),(-128.937,53.567),(-128.869,53.519),
     (-128.845,53.443),(-128.865,53.365),(-128.941,53.309),
     (-128.993,53.308),(-129.042,53.336),(-129.072,53.366),
     (-129.106,53.407),(-129.115,53.445),(-129.084,53.486),
     (-129.066,53.525),(-129.021,53.556)])
# A small edge of Princess Royal Island; the polygon is clipped by sampling.
add([(-128.95,53.098),(-128.89,53.060),(-128.84,52.90),
     (-129.10,52.90),(-129.063,52.983),(-129.025,53.027)])

# Only a few confidently remembered small island/reef groups are included.
# Ellipses are handwritten rules, not geographic lookups.
def ellipse(lon, lat, rx, ry):
    land[:] |= ((X-lon)/rx)**2 + ((Y-lat)/ry)**2 <= 1.0

ellipse(-130.638,53.492,0.033,0.024)  # Bonilla
ellipse(-130.611,54.296,0.013,0.009)  # Lucy group
ellipse(-130.884,54.296,0.010,0.007)  # Triple Islands
ellipse(-130.866,54.182,0.011,0.009)  # Tree Nob group
ellipse(-130.737,54.748,0.023,0.016)  # Tongass
ellipse(-130.340,54.044,0.017,0.012)  # Lawyer group

# Southwest end of Wales Island, protruding beyond the northern envelope.
add([(-130.531,54.778),(-130.493,54.817),(-130.442,54.853),
     (-130.382,54.863),(-130.331,54.831),(-130.362,54.795),
     (-130.411,54.768),(-130.460,54.740),(-130.494,54.714),
     (-130.530,54.736)])


def channel(nodes):
    """Exclude a marine corridor; nodes are lon, lat, half-width in km.

    A local equirectangular metric is sufficient for these approximate shapes.
    This does not resample pixels, use cell areas, or smooth a raster coast.
    """
    sx = 111.32 * math.cos(math.radians(54.0))
    sy = 111.32
    px, py = X*sx, Y*sy
    water = np.zeros_like(land)
    for a, b in zip(nodes[:-1], nodes[1:]):
        ax, ay, ra = a[0]*sx, a[1]*sy, a[2]
        bx, by, rb = b[0]*sx, b[1]*sy, b[2]
        dx, dy = bx-ax, by-ay
        t = np.clip(((px-ax)*dx+(py-ay)*dy)/(dx*dx+dy*dy),0,1)
        distance2 = (px-ax-t*dx)**2 + (py-ay-t*dy)**2
        water |= distance2 <= (ra+t*(rb-ra))**2
    land[:] &= ~water


# Pearse Canal, Wales Passage, Portland Canal and Observatory/Nass branches.
channel([(-130.57,54.755,1.1),(-130.51,54.824,0.8),
         (-130.45,54.878,0.9),(-130.35,54.940,1.0),
         (-130.26,54.991,1.1),(-130.19,55.055,1.0)])
channel([(-130.455,54.882,0.65),(-130.370,54.867,0.65),
         (-130.270,54.844,0.8),(-130.205,54.841,1.1)])
channel([(-130.128,54.879,1.8),(-130.102,54.927,1.4),
         (-130.079,54.977,1.3),(-130.042,55.040,1.4)])
channel([(-129.864,54.954,1.7),(-129.826,54.997,1.5),
         (-129.800,55.050,1.3)])
channel([(-129.784,54.945,1.5),(-129.680,54.973,1.5),
         (-129.577,55.005,1.0),(-129.470,55.044,0.5)])

# Work Channel, the south side of Somerville, and Khutzeymateen Inlet.
channel([(-130.337,54.705,1.0),(-130.290,54.645,0.9),
         (-130.218,54.585,0.85),(-130.145,54.512,0.7),
         (-130.073,54.439,0.6),(-130.015,54.383,0.3)])
channel([(-130.292,54.648,0.9),(-130.222,54.659,0.8),
         (-130.165,54.706,0.9),(-130.153,54.749,1.2)])
channel([(-130.165,54.701,1.0),(-130.068,54.679,0.9),
         (-129.970,54.653,0.7),(-129.860,54.624,0.35)])

# Prince Rupert's tidal connections and the Skeena estuarine river.
channel([(-130.381,54.347,0.45),(-130.299,54.349,0.4),
         (-130.248,54.338,0.5),(-130.234,54.285,0.4),
         (-130.259,54.234,0.5)])
channel([(-130.248,54.340,0.6),(-130.213,54.385,0.6),
         (-130.182,54.420,0.4)])
channel([(-130.135,54.179,1.1),(-130.021,54.188,1.3),
         (-129.897,54.217,1.2),(-129.792,54.261,1.0),
         (-129.654,54.295,0.9),(-129.540,54.325,0.8),
         (-129.425,54.370,0.6),(-129.294,54.434,0.55),
         (-129.178,54.485,0.5),(-128.970,54.552,0.4)])

# A few remembered marine indentations, deliberately not inland-lake masks.
channel([(-130.746,53.952,0.8),(-130.681,53.970,0.6)])
channel([(-130.442,53.850,1.0),(-130.482,53.909,1.1),
         (-130.541,53.960,0.6)])
channel([(-129.608,53.553,0.7),(-129.548,53.568,0.6),
         (-129.514,53.585,0.3)])
channel([(-129.480,53.584,0.65),(-129.412,53.559,0.55),
         (-129.330,53.533,0.65),(-129.239,53.517,0.9)])
channel([(-130.030,53.651,0.75),(-129.948,53.660,0.7),
         (-129.876,53.673,0.3)])
channel([(-129.791,53.438,0.8),(-129.714,53.451,0.6)])

# Exact endpoint-inclusive WGS84 sampling; the mask contains point values only.
assert X.shape == Y.shape == land.shape == (127,127)
assert longitude[0] == -131.0 and longitude[-1] == -129.0
assert latitude[0] == 55.0 and latitude[-1] == 53.0
matrix = land.astype(np.uint8)
csv_path = OUT / "candidate.csv"
with csv_path.open("w",newline="",encoding="ascii") as f:
    writer = csv.writer(f,lineterminator="\n")
    writer.writerows(matrix.tolist())

pixels = matrix * np.uint8(255)
original = Image.fromarray(pixels)
assert original.mode == "L"
original.save(OUT / "candidate.png")
nearest = getattr(Image,"Resampling",Image).NEAREST
enlarged = original.resize((1016,1016),resample=nearest)
enlarged.save(OUT / "candidate_8x.png")

# Validate the files themselves, including every replicated 8-by-8 block.
with csv_path.open("r",newline="",encoding="ascii") as f:
    rows = list(csv.reader(f))
assert len(rows) == 127
assert all(len(row) == 127 for row in rows)
assert all(value in ("0","1") for row in rows for value in row)
reloaded = np.array(rows,dtype=np.uint8)
assert reloaded.size == 16129
assert np.array_equal(reloaded,matrix)

with Image.open(OUT / "candidate.png") as im:
    assert im.size == (127,127)
    assert im.mode == "L"
    small = np.array(im)
with Image.open(OUT / "candidate_8x.png") as im:
    assert im.size == (1016,1016)
    assert im.mode == "L"
    large = np.array(im)
assert set(np.unique(small).tolist()) == {0,255}
assert set(np.unique(large).tolist()) == {0,255}
assert np.array_equal(small,matrix*255)
assert np.array_equal(large,np.repeat(np.repeat(small,8,axis=0),8,axis=1))

report = {
    "method": "Handwritten approximate polygons and marine-channel corridors, "
              "based only on pre-existing geographic knowledge; vectorized "
              "point predicates, not independent reasoning for each point.",
    "dependencies": ["Python 3","numpy","Pillow"],
    "shape": [127,127],
    "valid_values": int(matrix.size),
    "land_points": int(matrix.sum()),
    "water_points": int(matrix.size-matrix.sum()),
    "csv_no_header_no_missing_binary_only": True,
    "original_png_size": [127,127],
    "enlarged_png_size": [1016,1016],
    "png_intensities": [0,255],
    "nearest_neighbor_exact_verified": True,
    "external_geographic_data_used": False,
}
(OUT / "validation.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2))
