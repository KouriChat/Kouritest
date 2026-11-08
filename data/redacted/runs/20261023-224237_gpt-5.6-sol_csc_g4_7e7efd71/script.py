# Moderately sized outlying islands, represented by remembered centre, size,
# and orientation.  Radii are in degrees and are intentionally not inflated to
# fill pixels: this is a point test rather than an area-majority raster.
ISLAND_ELLIPSES = [
    # Philippine outliers and islands omitted from the larger polygons
    (124.20,13.72,.28,.48,-18), # Catanduanes
    (121.95,14.82,.13,.28,-15), # Polillo
    (122.02,13.40,.25,.20,-20), # Marinduque
    (122.35,12.55,.10,.15,0),(122.02,12.32,.10,.13,0), # Romblon group
    (121.38,19.32,.12,.17,5),(121.90,18.92,.10,.14,-10),
    (121.95,19.52,.10,.10,0),(121.90,20.42,.09,.16,-10),
    (121.85,20.75,.08,.13,0), # Babuyan and Batanes
    (125.92,10.05,.17,.50,-12),(126.05,9.82,.23,.35,12), # Dinagat, Siargao
    (124.70,9.15,.12,.16,0),(122.02,6.55,.32,.18,-12), # Camiguin, Basilan
    (121.05,5.98,.32,.20,5),(120.20,5.35,.22,.14,0),
    (119.88,5.02,.38,.20,-10),(118.75,5.05,.13,.10,0), # Sulu chain
    # Islands of the Sunda shelf and around the Malay peninsula
    (108.20,3.92,.34,.52,-20),(106.25,3.15,.18,.26,15), # Natuna, Anambas
    (105.78,2.95,.13,.19,-15),(107.55,1.18,.15,.12,0),
    (104.48,1.08,.28,.24,-15),(104.02,1.12,.28,.17,0), # Bintan, Batam
    (103.82,1.30,.27,.10,0),(104.17,2.80,.08,.18,-15), # Singapore, Tioman
    (109.10,1.62,.20,.15,0),(117.18,7.25,.20,.27,-20), # Karimata, Banggi
    (115.22,5.30,.07,.10,0),
    # Islands close to the continental coast
    (103.98,10.25,.16,.29,-5),(102.35,12.05,.16,.26,8), # Phu Quoc, Ko Chang
    (100.00,9.52,.13,.16,0),(106.60,8.68,.08,.18,-35), # Samui, Con Dao
    (108.50,15.95,.05,.08,0),(114.20,22.25,.16,.10,0) # Cham, Hong Kong
]

# Tiny coral islands and reef platforms.  Most contain no 0.2-degree sample
# point; retaining their true small scale avoids claiming surrounding sea.
REEFS = [
    (116.72,20.70,.025,.018),(112.33,16.83,.035,.025),
    (111.62,16.52,.045,.025),(111.20,15.78,.035,.022),
    (111.68,17.08,.040,.025),(112.73,16.05,.060,.025),
    (114.37,10.37,.035,.025),(114.28,11.45,.045,.025),
    (113.85,10.03,.040,.025),(114.92,9.72,.055,.030),
    (115.87,10.82,.035,.025),(115.82,9.88,.055,.025),
    (113.93,9.70,.040,.025),(112.88,9.55,.035,.025),
    (112.22,8.65,.035,.025),(111.92,7.88,.035,.025),
    (113.70,8.17,.040,.025),(114.48,7.38,.050,.025),
    (116.00,7.00,.040,.025),(116.68,8.85,.045,.025)
]

# Inland waters large enough to affect this sampling lattice.
TONLE_SAP = [
    (103.45,13.85),(103.42,13.45),(103.62,12.92),(103.82,12.50),
    (104.15,12.25),(104.48,12.28),(104.62,12.52),(104.38,12.92),
    (104.18,13.30),(103.85,13.70)
]


def is_land(x, y):
    land = any(in_polygon(x, y, p) for p in BIG_POLYGONS)
    if not land:
        land = any(ellipse(x, y, *e) for e in ISLAND_ELLIPSES)
    if not land:
        land = any(ellipse(x, y, cx, cy, rx, ry)
                   for cx, cy, rx, ry in REEFS)
    if land:
        # These are water even though enclosed by one of the land polygons.
        if in_polygon(x, y, TONLE_SAP):
            return False
        if ellipse(x, y, 121.22,14.40,.34,.23,-10):  # Laguna de Bay
            return False
        if ellipse(x, y, 124.25,7.90,.16,.20,0):     # Lake Lanao
            return False
        if ellipse(x, y, 100.38,7.20,.10,.42,-8):    # Songkhla Lake
            return False
    return land


matrix = np.array([[1 if is_land(float(x), float(y)) else 0 for x in LONS]
                   for y in LATS], dtype=np.uint8)

out = Path("outputs")
out.mkdir(parents=True, exist_ok=True)
with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
    csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

raw = Image.fromarray(matrix * 255, mode="L")
raw.save(out / "candidate.png")
large = raw.resize((1128, 888), resample=Image.Resampling.NEAREST)
large.save(out / "candidate_8x.png")

# Fail loudly if any delivery invariant is violated.
assert matrix.shape == (111, 141) and set(np.unique(matrix)) <= {0, 1}
assert raw.size == (141, 111) and large.size == (1128, 888)
assert set(raw.getdata()) <= {0, 255} and set(large.getdata()) <= {0, 255}
assert np.array_equal(np.asarray(large), np.repeat(np.repeat(
    np.asarray(raw), 8, axis=0), 8, axis=1))
print("created", matrix.shape, "land points:", int(matrix.sum()))
