# Smaller archipelagos.  At this 0.2-degree sampling only islands capable of
# containing a sample point have much effect, but remembered islets are kept.
small_islands = [
 # Ionian
 (19.88,39.62,.16,.55,-8),(20.70,38.72,.12,.27,-12),
 (20.57,38.22,.30,.38,-18),(20.88,37.78,.20,.25,-8),
 (23.00,36.22,.18,.28,-10),
 # Euboea and Sporades
 (23.65,38.58,.26,.92,-18),(23.47,39.17,.12,.19,0),
 (24.10,39.15,.13,.10,0),
 # Northern/eastern Aegean
 (25.55,40.48,.23,.30,8),(25.15,39.92,.20,.13,0),
 (26.20,39.18,.38,.36,5),(26.02,38.38,.18,.45,-5),
 (26.82,37.75,.43,.17,5),(27.15,36.88,.38,.13,4),
 # Cyclades and Saronic islands
 (24.85,37.82,.13,.33,-12),(25.17,37.60,.13,.20,-12),
 (25.35,37.45,.15,.13,0),(25.48,37.05,.20,.37,-10),
 (25.15,37.03,.18,.24,-10),(24.93,37.43,.13,.17,0),
 (24.45,36.72,.33,.16,5),(25.43,36.40,.17,.13,0),
 (24.35,37.30,.17,.13,0),(23.45,37.72,.17,.24,-5),
 # Dodecanese
 (26.98,37.30,.11,.18,0),(26.55,37.12,.14,.22,0),
 (26.85,36.60,.17,.15,0),(27.35,36.60,.13,.11,0),
 # Dalmatian archipelago
 (14.45,44.85,.18,.35,-8),(14.75,44.70,.18,.37,-12),
 (15.05,44.40,.13,.33,-12),(15.45,44.15,.14,.42,-15),
 (16.20,43.95,.12,.34,-25),(16.50,43.15,.18,.34,-28),
 (16.72,42.98,.20,.37,-25),(17.05,42.77,.12,.30,-25),
 # Sicily's Aeolian/Egadi chain
 (14.95,38.48,.09,.09,0),(14.82,38.58,.08,.08,0),
 (13.00,38.02,.08,.08,0),(12.32,37.93,.12,.09,0),
 # North African offshore islands
 (11.00,33.82,.32,.24,-15),(11.15,34.72,.15,.24,0),
 (8.75,36.95,.15,.09,0),(3.20,37.10,.07,.07,0),
 # Sea of Marmara islands
 (27.60,40.52,.15,.27,-5),(28.05,40.62,.14,.09,0),
]

# Sinai completes the small portion north of 31 N.
sinai = [(32.2,30.7),(32.4,31.3),(33.2,31.1),(34.9,31.9),
         (34.5,31.1),(34.3,30.7)]

def is_land(x, y):
    if any(inside(x, y, p) for p in main_polys) or inside(x, y, sinai):
        return 1
    if any(ellipse(x, y, *q) for q in islands):
        return 1
    if any(ellipse(x, y, *q) for q in small_islands):
        return 1
    return 0

matrix = np.array([[is_land(float(x), float(y)) for x in LONS]
                   for y in LATS], dtype=np.uint8)

# The remembered outlines are the complete decision rule; serialize directly,
# without smoothing, resampling, majority voting, or post-processing.
out = Path("outputs")
out.mkdir(exist_ok=True)
with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
    csv.writer(f).writerows(matrix.tolist())

raw = Image.fromarray(matrix * 255, mode="L")
raw.save(out / "candidate.png")
large = raw.resize((231*8, 71*8), resample=Image.Resampling.NEAREST)
large.save(out / "candidate_8x.png")

# Fail loudly if any delivery invariant is not met.
assert matrix.shape == (71, 231) and matrix.size == 16401
assert set(np.unique(matrix)).issubset({0, 1})
assert raw.size == (231, 71) and large.size == (1848, 568)
assert set(raw.getdata()).issubset({0, 255})
assert np.array_equal(np.asarray(large), np.repeat(np.repeat(
    np.asarray(raw), 8, axis=0), 8, axis=1))
print("generated", matrix.shape, "land points", int(matrix.sum()))
