"""Knowledge-only reconstruction of Mediterranean land and water.

All vertices below were typed from geographic memory.  No external files,
map services, coastline packages, or geographic databases are read.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

LONS = np.round(np.arange(-9.0, 37.0001, 0.2), 1)
LATS = np.round(np.arange(45.0, 30.9999, -0.2), 1)
X, Y = np.meshgrid(LONS, LATS)
land = np.ones(X.shape, dtype=bool)

def inside(poly):
    """Vectorized even/odd point-in-polygon test, boundary biased inward."""
    px = X.ravel()
    py = Y.ravel()
    hit = np.zeros(px.size, dtype=bool)
    xj, yj = poly[-1]
    for xi, yi in poly:
        cross = ((yi > py) != (yj > py))
        xcut = (xj-xi) * (py-yi) / ((yj-yi) + 1e-300) + xi
        hit ^= cross & (px <= xcut)
        xj, yj = xi, yi
    return hit.reshape(X.shape)

def paint(value, poly):
    land[inside(poly)] = value

def ellipse(value, cx, cy, rx, ry):
    land[((X-cx)/rx)**2 + ((Y-cy)/ry)**2 <= 1.0] = value

# Atlantic Ocean.  The right-hand edge follows the remembered coasts of
# France, Iberia and Atlantic Morocco.
atlantic = [
    (-9,45),(-1.45,45),(-1.35,44.5),(-1.65,43.55),
    (-3.0,43.45),(-4.5,43.45),(-6.0,43.65),(-7.4,43.55),
    (-8.5,43.3),(-8.9,42.7),(-8.75,41.8),(-8.75,41.0),
    (-8.9,40.0),(-9.05,39.0),(-9.05,38.0),(-8.8,37.1),
    (-8.0,37.0),(-7.3,37.2),(-6.35,36.65),(-5.62,36.02),
    (-5.35,35.85),(-5.9,35.45),(-6.45,34.65),(-6.8,34.0),
    (-7.35,33.45),(-7.7,32.8),(-8.25,32.0),(-9,31.45)
]
paint(False, atlantic)

# Main connected Mediterranean water body.  Peninsulas and islands that
# project into this broad outline are restored as land farther below.
med = [
    (-5.62,36.02),(-4.8,36.45),(-4.0,36.7),(-3.0,36.75),
    (-2.0,36.75),(-1.25,37.45),(-0.7,37.65),(-0.15,38.35),
    (0.2,39.0),(0.05,39.8),(0.55,40.65),(1.35,41.1),
    (2.15,41.3),(3.15,42.35),(3.15,42.9),(3.8,43.3),
    (4.8,43.4),(5.35,43.25),(6.15,43.1),(7.4,43.75),
    (8.7,44.35),(10.0,44.75),(13.5,45.15),(14.4,45.15),
    (18.4,43.0),(19.6,41.8),(20.0,40.0),(21.4,39.4),
    (22.7,40.5),(24.2,41.0),(26.3,41.15),(26.65,40.45),
    (26.4,39.8),(26.7,39.0),(26.7,38.25),(27.3,37.85),
    (27.2,37.0),(28.0,36.55),(29.1,36.2),(30.5,36.15),
    (32.0,36.0),(33.5,35.95),(34.2,36.15),(35.0,35.95),
    (36.0,36.15),(36.15,35.55),(35.75,34.75),(35.55,34.25),
    (35.6,33.35),(35.25,32.55),(34.65,31.75),(34.3,31.25),
    (33.0,31.05),(31.8,31.25),(31.25,31.5),(30.45,31.35),
    (29.8,31.15),(28.8,30.8),(19.5,30.8),(19.0,31.3),
    (18.1,31.8),(16.8,32.3),(15.1,32.45),(13.2,32.9),
    (12.0,33.25),(11.2,33.65),(10.0,33.8),(10.45,34.45),
    (10.75,35.15),(11.05,36.0),(11.0,36.85),(10.2,37.15),
    (9.4,37.3),(8.5,36.9),(7.0,36.95),(5.2,36.75),
    (3.2,36.8),(1.3,36.55),(0.0,35.9),(-0.8,35.35),
    (-1.7,35.05),(-2.7,35.15),(-3.6,35.35),(-4.5,35.2),
    (-5.35,35.85)
]
paint(False, med)
# Italian peninsula, from the Ligurian shore clockwise around the boot.
italy = [
    (7.35,45.2),(13.65,45.2),(13.2,44.65),(12.55,44.2),
    (12.3,43.6),(13.0,43.0),(13.75,42.65),(14.7,42.4),
    (15.45,42.15),(16.1,41.75),(17.15,41.2),(17.85,41.0),
    (18.05,40.55),(18.5,40.15),(18.45,39.8),(18.0,39.95),
    (17.55,40.35),(16.95,40.45),(16.65,40.2),(16.9,39.7),
    (16.65,39.05),(16.15,38.35),(15.7,38.05),(15.65,38.5),
    (15.85,39.15),(15.65,39.65),(15.15,40.0),(14.7,40.3),
    (14.1,40.55),(13.75,40.85),(13.1,41.05),(12.65,41.2),
    (12.0,41.55),(11.35,42.05),(10.75,42.55),(10.25,43.05),
    (9.85,43.55),(9.45,44.05),(8.75,44.35),(7.8,43.9)
]
paint(True, italy)

# Eastern Adriatic coast, mainland Greece, Chalkidiki and Thrace.
balkans = [
    (13.55,45.2),(27.0,45.2),(27.0,41.2),(26.65,40.55),
    (26.25,40.6),(26.1,40.05),(25.65,40.0),(25.2,40.3),
    (24.7,40.6),(24.0,40.75),(23.4,40.65),(24.25,40.35),
    (24.0,39.95),(23.65,40.05),(23.35,40.45),(23.05,40.55),
    (22.75,40.6),(22.65,40.25),(22.95,39.85),(22.75,39.35),
    (23.15,38.9),(23.75,38.45),(24.05,37.75),(23.65,37.7),
    (23.25,38.0),(22.65,38.25),(22.0,38.3),(21.25,38.35),
    (20.85,38.65),(20.6,39.05),(20.25,39.45),(19.85,39.75),
    (19.55,40.15),(19.35,40.55),(19.4,41.0),(19.1,41.85),
    (18.55,42.0),(18.45,42.45),(17.75,42.65),(17.2,42.9),
    (16.65,42.9),(16.15,43.15),(15.55,43.35),(15.2,43.75),
    (14.85,44.2),(14.3,44.5),(13.75,44.5),(13.15,44.8)
]
paint(True, balkans)

# Peloponnese, attached at the narrow Corinth isthmus.
peloponnese = [
    (20.55,38.25),(21.2,38.4),(22.1,38.35),(23.0,38.15),
    (23.25,37.75),(23.0,37.3),(23.0,36.75),(22.8,36.45),
    (22.25,36.4),(21.85,36.75),(21.55,36.4),(21.0,36.65),
    (21.15,37.15),(21.0,37.55),(20.65,37.85)
]
paint(True, peloponnese)

# Recut the Gulf of Corinth, while retaining the isthmus at its east end.
corinth_gulf = [
    (20.75,38.25),(21.25,38.35),(22.0,38.3),(22.75,38.15),
    (22.9,38.05),(22.55,38.0),(21.9,38.1),(21.25,38.15)
]
paint(False, corinth_gulf)

# Anatolia.  Black Sea and Marmara water are cut from it in the next section.
anatolia = [
    (25.9,40.55),(26.35,39.7),(26.65,39.05),(26.65,38.3),
    (27.25,37.85),(27.15,37.0),(27.9,36.5),(28.9,36.15),
    (30.4,36.05),(31.7,35.85),(32.8,35.8),(33.6,35.85),
    (34.15,36.1),(35.0,35.9),(36.0,36.1),(37.2,36.55),
    (37.2,42.4),(29.0,42.4),(27.8,41.6),(26.7,41.25)
]
paint(True, anatolia)

# A little more definition to the Gallipoli peninsula.
gallipoli = [
    (26.0,40.0),(26.25,40.05),(26.7,40.45),(26.65,40.75),
    (26.35,40.65),(26.1,40.35)
]
paint(True, gallipoli)
# Black Sea, including the part north of the plotted 45-degree edge.
black_sea = [
    (29.55,45.2),(37.2,45.2),(37.2,40.85),(36.6,41.15),
    (36.0,41.65),(35.25,41.75),(34.5,41.95),(33.6,41.95),
    (32.8,41.75),(32.1,41.65),(31.4,41.25),(30.5,41.1),
    (29.8,41.1),(29.05,41.05),(28.8,41.45),(28.45,41.85),
    (27.95,42.15),(27.75,42.7),(28.05,43.2),(28.55,43.65),
    (29.2,44.1)
]
paint(False, black_sea)

# Crimea projects into the broad Black Sea cut.
crimea = [
    (32.3,45.2),(32.45,44.75),(32.8,44.55),(33.35,44.4),
    (33.8,44.6),(34.35,44.5),(34.8,44.35),(35.4,44.55),
    (35.85,44.8),(36.5,45.15)
]
paint(True, crimea)

# The extreme corner of the Sea of Azov visible east of Crimea.
azov = [(36.45,45.2),(37.2,45.2),(37.2,44.75),(36.75,44.8)]
paint(False, azov)

# Sea of Marmara, Bosporus and Dardanelles represented at grid scale.
marmara = [
    (26.15,40.0),(26.35,40.15),(26.55,40.35),(27.1,40.35),
    (27.65,40.4),(28.2,40.6),(28.75,40.8),(29.05,41.15),
    (29.15,41.05),(28.95,40.75),(28.55,40.55),(27.85,40.65),
    (27.2,40.65),(26.65,40.5),(26.4,40.3)
]
paint(False, marmara)

# Largest western and central Mediterranean islands.
mallorca = [(2.25,39.25),(2.45,39.75),(2.9,40.0),(3.35,39.8),
            (3.5,39.45),(3.1,39.25),(2.65,39.25)]
paint(True, mallorca)
menorca = [(3.75,39.8),(4.1,40.1),(4.45,40.05),(4.35,39.85)]
paint(True, menorca)
ellipse(True, 1.42, 38.98, .28, .24)       # Ibiza
ellipse(True, 1.43, 38.70, .12, .13)       # Formentera

corsica = [
    (8.55,43.0),(9.25,43.05),(9.5,42.7),(9.4,42.15),
    (9.55,41.65),(9.2,41.35),(8.75,41.45),(8.65,42.0)
]
paint(True, corsica)
sardinia = [
    (8.15,41.25),(8.8,41.3),(9.55,41.0),(9.8,40.45),
    (9.65,39.8),(9.45,39.15),(8.85,38.85),(8.35,39.05),
    (8.25,39.65),(8.45,40.1),(8.15,40.55)
]
paint(True, sardinia)

sicily = [
    (12.25,38.0),(12.7,38.2),(13.55,38.15),(14.25,38.0),
    (15.2,38.15),(15.65,38.05),(15.3,37.65),(15.05,37.2),
    (14.45,36.75),(13.65,36.65),(12.8,37.0),(12.45,37.45)
]
paint(True, sicily)

ellipse(True, 14.4,35.9,.22,.18)            # Malta
ellipse(True, 14.34,36.03,.11,.08)          # Gozo
ellipse(True, 12.0,35.5,.14,.20)            # Pantelleria
ellipse(True, 12.6,35.5,.10,.08)            # Linosa
ellipse(True, 12.58,35.85,.20,.08)          # Lampedusa
# Crete and Cyprus.
crete = [
    (23.35,35.55),(24.0,35.65),(24.75,35.55),(25.35,35.35),
    (26.25,35.25),(26.35,35.05),(25.8,34.95),(25.05,35.05),
    (24.4,35.15),(23.75,35.2),(23.35,35.35)
]
paint(True, crete)
cyprus = [
    (32.25,35.05),(32.65,35.45),(33.2,35.7),(33.85,35.65),
    (34.55,35.65),(34.15,35.35),(33.95,34.95),(33.4,34.55),
    (32.8,34.6),(32.35,34.85)
]
paint(True, cyprus)

# Euboea and the larger Ionian islands.
euboea = [
    (23.35,38.9),(23.65,39.1),(24.1,38.8),(24.55,38.05),
    (24.25,37.85),(23.95,38.25),(23.65,38.55)
]
paint(True, euboea)
ellipse(True, 19.82,39.62,.22,.48)          # Corfu
ellipse(True, 20.72,38.65,.15,.22)          # Lefkada
ellipse(True, 20.55,38.22,.32,.39)          # Kefalonia
ellipse(True, 20.72,38.40,.10,.18)          # Ithaca
ellipse(True, 20.73,37.78,.19,.32)          # Zakynthos

# Northern and eastern Aegean islands.
ellipse(True, 24.58,40.68,.18,.25)          # Thasos
ellipse(True, 25.55,40.47,.28,.20)          # Samothrace
ellipse(True, 25.18,39.92,.36,.31)          # Lemnos
ellipse(True, 26.28,39.18,.43,.33)          # Lesbos
ellipse(True, 26.05,38.42,.20,.42)          # Chios
ellipse(True, 26.82,37.75,.34,.16)          # Samos
ellipse(True, 26.05,39.52,.09,.11)          # Agios Efstratios

# Cyclades, Sporades and Dodecanese (small islands deliberately retained
# whenever a 0.2-degree sample can plausibly strike them).
ellipse(True, 23.48,39.17,.12,.24)          # Skiathos
ellipse(True, 23.70,39.12,.15,.30)          # Skopelos
ellipse(True, 24.05,39.18,.11,.19)          # Alonnisos
ellipse(True, 24.82,37.85,.17,.42)          # Andros
ellipse(True, 25.15,37.62,.13,.22)          # Tinos
ellipse(True, 25.35,37.45,.16,.13)          # Mykonos
ellipse(True, 24.92,37.42,.18,.35)          # Syros
ellipse(True, 24.43,37.60,.12,.20)          # Kea
ellipse(True, 24.72,37.35,.10,.17)          # Kythnos
ellipse(True, 24.48,37.15,.17,.28)          # Serifos
ellipse(True, 24.68,36.98,.20,.17)          # Sifnos
ellipse(True, 24.42,36.72,.34,.18)          # Milos
ellipse(True, 25.38,36.42,.20,.18)          # Santorini
ellipse(True, 25.48,37.05,.16,.20)          # Ios
ellipse(True, 25.48,37.12,.09,.17)          # Sikinos
ellipse(True, 25.85,36.92,.09,.18)          # Amorgos
ellipse(True, 25.48,37.62,.10,.14)          # Delos area
ellipse(True, 25.47,37.08,.35,.43)          # Naxos
ellipse(True, 25.18,37.05,.19,.32)          # Paros
ellipse(True, 26.58,37.58,.12,.23)          # Ikaria
ellipse(True, 26.98,36.90,.13,.10)          # Patmos
ellipse(True, 26.88,37.15,.12,.15)          # Leros
ellipse(True, 27.02,36.95,.18,.27)          # Kalymnos
ellipse(True, 27.12,36.62,.30,.15)          # Kos
ellipse(True, 27.85,36.58,.12,.24)          # Nisyros
rhodes = [(27.7,36.45),(28.15,36.5),(28.3,36.2),(28.05,35.85),
          (27.75,35.95),(27.65,36.2)]
paint(True, rhodes)
ellipse(True, 27.85,35.55,.10,.17)          # Karpathos
ellipse(True, 26.92,35.42,.10,.18)          # Kasos
ellipse(True, 29.58,36.15,.10,.08)          # Kastellorizo
# Elba and the principal small Italian archipelagos.
ellipse(True, 10.28,42.78,.28,.18)          # Elba
ellipse(True, 9.83,42.58,.08,.08)           # Capraia
ellipse(True, 10.90,42.37,.09,.12)          # Giglio
ellipse(True, 13.92,40.73,.09,.08)           # Ischia
ellipse(True, 14.23,40.55,.09,.06)           # Capri
ellipse(True, 14.95,38.48,.10,.12)           # Lipari/Vulcano
ellipse(True, 15.22,38.58,.07,.08)           # Stromboli
ellipse(True, 14.35,38.55,.08,.07)           # Filicudi
ellipse(True, 13.18,38.72,.08,.13)           # Ustica
ellipse(True, 15.08,36.70,.06,.06)           # Cape Passero islets

# Dalmatian and Kvarner islands.
croatian_islands = [
    [(14.28,45.2),(14.35,44.9),(14.55,44.55),(14.45,44.25),
     (14.25,44.5),(14.2,44.9)],              # Cres
    [(14.55,44.95),(14.85,45.05),(14.95,44.75),(14.75,44.55)],
    [(14.65,44.55),(14.85,44.65),(14.78,44.2),(14.55,43.95)],
    [(15.0,44.55),(15.25,44.6),(15.2,44.1),(14.95,44.0)], # Pag
    [(14.85,44.0),(15.15,44.15),(15.1,43.75),(14.75,43.35)],
    [(15.0,43.9),(15.5,44.15),(15.45,43.85),(15.0,43.45)], # Dugi
    [(15.3,43.5),(16.2,43.75),(16.15,43.5),(15.55,43.3)],
    [(16.05,43.3),(16.95,43.45),(16.75,43.15),(16.2,43.05)], # Brac
    [(16.1,43.15),(17.35,43.15),(17.05,42.95),(16.35,42.9)], # Hvar
    [(16.8,42.95),(17.2,43.05),(17.15,42.75),(16.85,42.7)],
    [(16.85,42.85),(17.7,42.95),(17.6,42.75),(17.0,42.6)], # Korcula
    [(17.35,42.75),(18.0,42.8),(17.75,42.55),(17.35,42.55)]
]
for p in croatian_islands:
    paint(True, p)

# North African near-shore islands.
ellipse(True, 11.15,33.78,.34,.25)          # Djerba
ellipse(True, 11.15,34.70,.16,.25)          # Kerkennah
ellipse(True, 8.65,36.93,.13,.09)           # La Galite
ellipse(True, 3.22,36.78,.08,.06)           # Algiers islets
ellipse(True, -2.18,35.18,.08,.05)          # Chafarinas
ellipse(True, -3.0,35.42,.06,.05)           # Alhucemas vicinity

# A few western Mediterranean islets with a chance of hitting this grid.
ellipse(True, -5.30,35.90,.04,.05)          # Perejil
ellipse(True, -2.95,35.93,.07,.05)           # Alboran
ellipse(True, 4.30,39.88,.07,.05)            # Menorca off-islets

# Sanity conditions required by the task.
assert land.shape == (71, 231)
matrix = land.astype(np.uint8)
assert matrix.size == 16401
assert np.all((matrix == 0) | (matrix == 1))

out = Path("outputs")
out.mkdir(parents=True, exist_ok=True)
with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
    csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

raw_array = matrix * np.uint8(255)
raw = Image.fromarray(raw_array, mode="L")
raw.save(out / "candidate.png")
large = raw.resize((1848, 568), resample=Image.Resampling.NEAREST)
large.save(out / "candidate_8x.png")

# Verify dimensions, bilevel content, CSV row lengths, and exact 8x replication.
assert raw.size == (231, 71) and large.size == (1848, 568)
assert set(np.unique(np.asarray(raw))).issubset({0, 255})
assert np.array_equal(np.asarray(large), np.repeat(np.repeat(raw_array, 8, 0), 8, 1))
with (out / "candidate.csv").open(encoding="ascii") as f:
    rows = list(csv.reader(f))
assert len(rows) == 71 and all(len(r) == 231 for r in rows)
assert all(v in ("0", "1") for r in rows for v in r)
