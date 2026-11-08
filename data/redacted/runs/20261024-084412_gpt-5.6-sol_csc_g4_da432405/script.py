from pathlib import Path
import csv
import numpy as np
from PIL import Image

# All geometry below is a hand-written, deliberately approximate reconstruction
# from geographic knowledge.  No external map or geographic data is read.
POLYGONS = []
ELLIPSES = []

def poly(points):
    POLYGONS.append(tuple(points))

def ellipse(lon, lat, rx, ry):
    ELLIPSES.append((lon, lat, rx, ry))

def inside_polygon(x, y, p):
    inside = False
    x0, y0 = p[-1]
    for x1, y1 in p:
        # Count points on an edge as land.
        dx, dy = x1-x0, y1-y0
        cross = (x-x0)*dy - (y-y0)*dx
        if abs(cross) < 1e-10 and min(x0,x1)-1e-10 <= x <= max(x0,x1)+1e-10 and min(y0,y1)-1e-10 <= y <= max(y0,y1)+1e-10:
            return True
        if (y0 > y) != (y1 > y):
            xi = x0 + (y-y0)*(x1-x0)/(y1-y0)
            if x < xi:
                inside = not inside
        x0, y0 = x1, y1
    return inside

# Continental SE Asia, including the Malay Peninsula.  The long listed edge is
# the South China Sea/Gulf of Thailand coast; the other edge is outside the box.
poly([
 (99,23.5),(118,23.5),(117.35,23.05),(116.85,22.85),(116.25,22.75),
 (115.65,22.75),(115.15,22.55),(114.55,22.55),(114.15,22.35),
 (113.65,22.30),(113.05,21.95),(112.35,21.75),(111.55,21.55),
 (110.75,21.55),(110.15,21.25),(110.05,20.75),(109.75,20.25),
 (109.45,20.10),(109.05,20.25),(108.55,20.65),(108.05,21.15),
 (107.55,21.35),(107.05,20.95),(106.65,20.25),(106.25,19.75),
 (105.85,19.15),(105.75,18.55),(106.05,17.95),(106.45,17.25),
 (106.75,16.75),(107.35,16.15),(108.05,15.55),(108.45,15.05),
 (108.75,14.35),(109.05,13.65),(109.25,12.75),(109.25,11.65),
 (109.05,10.75),(108.55,10.25),(107.65,10.05),(106.75,10.00),
 (106.25,9.55),(105.45,9.55),(104.75,9.85),(104.15,10.25),
 (103.55,10.45),(103.05,10.75),(102.75,11.45),(102.35,11.85),
 (101.85,12.25),(101.35,12.65),(100.85,13.15),(100.65,13.55),
 (100.25,13.65),(100.05,13.35),(100.05,12.75),(99.85,12.15),
 (99.55,11.55),(99.25,11.05),(99.15,10.55),(99.35,10.15),
 (99.55,9.75),(99.65,9.15),(99.85,8.65),(100.15,8.05),
 (100.35,7.45),(100.65,6.85),(101.15,6.35),(101.85,6.15),
 (102.35,5.85),(102.75,5.35),(103.15,4.75),(103.45,4.15),
 (103.45,3.55),(103.65,3.05),(103.85,2.45),(104.15,1.95),
 (104.45,1.45),(104.25,1.15),(103.75,1.15),(103.35,1.45),
 (102.95,1.75),(102.55,2.05),(102.15,2.45),(101.75,2.95),
 (101.45,3.45),(101.15,4.05),(100.85,4.65),(100.55,5.25),
 (100.35,5.85),(100.05,6.35),(99.75,6.75),(99.35,7.15),
 (99.00,7.35),(98.7,8.0),(98.5,10.0),(98.8,12.0),(98.8,23.5)
])

# Hainan
poly([(108.60,19.35),(108.70,18.75),(109.10,18.25),(109.65,18.15),
      (110.25,18.35),(110.70,18.75),(111.00,19.25),(110.90,19.75),
      (110.45,20.05),(109.85,20.15),(109.30,20.05),(108.85,19.75)])
# Taiwan (only its southern portion is in the requested frame)
poly([(119.90,23.55),(120.00,22.95),(120.20,22.45),(120.55,22.05),
      (120.85,21.85),(121.10,22.05),(121.30,22.45),(121.50,22.95),
      (121.70,23.55)])

# Sumatra, clipped by the western and southern limits.
poly([(97.15,5.75),(98.05,5.15),(98.70,4.55),(99.45,3.95),
      (100.20,3.35),(100.85,2.75),(101.55,2.25),(102.25,1.65),
      (103.15,1.05),(104.25,0.45),(104.5,-1),(98.0,-1),
      (97.75,0.8),(97.65,1.8),(97.55,2.9),(97.15,4.3)])

# Borneo/Kalimantan.
poly([(107.8,-1),(120.2,-1),(119.25,1.20),(118.75,2.10),
      (118.55,3.05),(118.25,3.75),(118.70,4.35),(119.10,4.75),
      (118.65,5.15),(118.25,5.65),(117.75,6.05),(117.45,6.55),
      (117.05,6.95),(116.55,7.05),(116.10,6.55),(115.75,6.10),
      (115.45,5.55),(114.95,5.15),(114.55,4.65),(113.95,4.20),
      (113.25,3.85),(112.55,3.65),(111.85,3.25),(111.20,3.05),
      (110.55,2.75),(110.05,2.15),(109.65,1.65),(109.55,1.05),
      (109.15,0.35)])
# The Philippine archipelago: separate hand-sketched island outlines preserve
# the major straits rather than joining everything into one coarse blob.
# Luzon and Bicol
poly([(120.55,18.70),(121.15,18.90),(121.75,18.60),(122.15,18.25),
      (122.45,17.55),(122.35,16.85),(122.00,16.15),(121.75,15.65),
      (121.75,15.05),(122.25,14.55),(123.05,14.15),(123.65,13.65),
      (123.90,13.15),(123.55,12.65),(123.05,12.90),(122.65,13.25),
      (122.10,13.55),(121.70,13.85),(121.25,13.65),(120.85,13.55),
      (120.55,13.75),(120.60,14.20),(120.25,14.55),(120.05,15.10),
      (119.75,15.75),(119.85,16.45),(120.10,17.10),(120.25,17.85)])
# Mindoro
poly([(120.35,13.55),(120.85,13.55),(121.25,13.25),(121.45,12.75),
      (121.15,12.25),(120.65,12.30),(120.35,12.75),(120.20,13.15)])
# Palawan
poly([(119.25,11.65),(119.20,11.10),(118.90,10.55),(118.65,10.05),
      (118.30,9.55),(117.95,9.05),(117.60,8.55),(117.15,8.20),
      (117.00,8.45),(117.25,8.85),(117.55,9.25),(117.85,9.80),
      (118.15,10.35),(118.50,10.90),(118.80,11.55)])
# Panay
poly([(121.90,11.75),(122.45,11.95),(122.75,11.55),(122.70,10.95),
      (122.35,10.40),(121.95,10.45),(121.70,10.85),(121.70,11.35)])
# Negros
poly([(122.75,11.05),(123.15,10.75),(123.40,10.20),(123.30,9.55),
      (122.95,9.05),(122.60,9.15),(122.40,9.70),(122.55,10.30)])
# Cebu
poly([(123.55,11.30),(123.85,11.05),(123.95,10.55),(123.75,9.85),
      (123.55,9.45),(123.35,9.65),(123.45,10.30),(123.35,10.80)])
# Bohol
poly([(123.75,10.15),(124.20,10.20),(124.60,9.90),(124.40,9.55),
      (123.90,9.55),(123.60,9.80)])
# Leyte
poly([(124.75,11.50),(125.10,11.25),(125.25,10.65),(125.15,10.10),
      (124.80,9.75),(124.55,10.15),(124.60,10.70),(124.45,11.15)])
# Samar
poly([(124.60,12.60),(125.15,12.65),(125.60,12.30),(125.75,11.75),
      (125.45,11.25),(124.95,11.15),(124.55,11.55),(124.30,12.05)])
# Masbate and Ticao
poly([(123.05,12.55),(123.55,12.65),(124.05,12.30),(123.85,11.85),
      (123.40,11.65),(123.05,11.95)])
poly([(123.55,12.95),(123.85,12.90),(123.90,12.45),(123.70,12.35)])

# Mindanao, including its narrow westward Zamboanga peninsula.
poly([(123.10,8.95),(123.70,8.55),(124.20,8.45),(124.70,8.75),
      (125.10,9.25),(125.55,9.75),(125.85,9.65),(125.65,9.05),
      (126.00,8.55),(126.40,8.25),(126.55,7.70),(126.45,7.05),
      (126.20,6.45),(125.75,5.90),(125.35,6.05),(125.15,6.35),
      (124.80,5.75),(124.45,5.55),(124.05,5.95),(123.70,6.40),
      (123.40,6.80),(122.85,6.95),(122.25,6.75),(121.95,6.90),
      (122.00,7.25),(122.50,7.45),(122.95,7.75),(123.15,8.20)])

# Medium-sized islands north and south of the main Philippine groups.
poly([(121.00,20.85),(121.25,20.75),(121.35,20.30),(121.10,20.15),
      (120.85,20.35)])                         # Batan
ellipse(122.00, 19.10, .18, .28)              # Babuyan
ellipse(121.60, 19.50, .15, .24)              # Calayan
ellipse(122.10, 18.00, .18, .34)              # Palaui vicinity
ellipse(119.95, 12.20, .24, .30)              # Busuanga
ellipse(120.20, 11.85, .16, .23)              # Culion
ellipse(121.00, 11.05, .14, .25)              # Cuyo
ellipse(122.05, 12.90, .18, .40)              # Tablas
ellipse(122.55, 12.45, .14, .30)              # Romblon/Sibuyan
ellipse(121.95, 12.30, .13, .27)              # Sibuyan channel islands
ellipse(122.10, 10.10, .16, .26)              # Guimaras
ellipse(123.10, 9.55, .13, .20)               # Siquijor
ellipse(125.65, 10.05, .13, .22)              # Dinagat
ellipse(125.20, 9.75, .12, .22)               # Siargao
ellipse(124.00, 9.15, .12, .24)               # Camiguin (Bohol Sea proxy)
ellipse(117.00, 7.90, .24, .25)               # Balabac
# Other substantial islands around the margins and in the two gulfs.
ellipse(121.95, 13.40, .28, .34)              # Marinduque
ellipse(124.20, 13.75, .30, .48)              # Catanduanes
ellipse(121.95, 14.75, .17, .42)              # Polillo
ellipse(124.70, 9.20, .18, .25)               # Camiguin
ellipse(122.00, 6.55, .30, .22)               # Basilan
ellipse(121.05, 6.05, .34, .25)               # Jolo
ellipse(120.00, 5.35, .20, .16)               # Tawi-Tawi group
ellipse(120.80, 5.80, .12, .12)
ellipse(121.45, 6.25, .12, .13)

ellipse(104.00, 10.25, .30, .47)              # Phu Quoc
ellipse(100.00, 9.52, .15, .22)               # Ko Samui
ellipse(99.98, 9.72, .08, .13)                # Ko Pha-ngan
ellipse(102.32, 11.98, .15, .23)              # Ko Chang
ellipse(99.78, 6.32, .24, .25)                # Langkawi
ellipse(103.02, 5.78, .10, .18)               # Redang
ellipse(104.17, 2.80, .16, .28)               # Tioman
ellipse(106.60, 8.68, .12, .17)               # Con Dao
ellipse(108.93, 10.52, .09, .14)              # Phu Quy

# Riau, Anambas and Natuna groups between Sumatra and Borneo.
ellipse(103.85, 1.12, .35, .18)               # Batam
ellipse(104.48, 1.02, .34, .40)               # Bintan
ellipse(104.05, 1.42, .15, .11)               # Singapore/Ubin
ellipse(105.78, 2.92, .12, .16)               # Anambas western group
ellipse(106.22, 3.20, .13, .17)
ellipse(106.05, 2.55, .10, .12)
poly([(107.90,4.20),(108.20,4.25),(108.40,4.00),(108.32,3.65),
      (108.05,3.55),(107.85,3.80)])            # Great Natuna
ellipse(107.55, 2.02, .12, .20)               # Tambelan

# Islands fringing northern Borneo.
ellipse(117.18, 7.22, .23, .25)               # Banggi
ellipse(116.92, 7.30, .13, .20)               # Balambangan
ellipse(118.45, 6.35, .10, .14)
ellipse(118.72, 5.65, .10, .12)
ellipse(115.20, 5.30, .09, .16)               # Labuan

# Taiwan's southeast outliers.
ellipse(121.55, 22.05, .10, .18)              # Orchid Island
ellipse(121.48, 22.67, .08, .12)              # Green Island

# Hand-positioned coral islands/reef flats.  Their very small radii mean a
# sample is marked only when the 0.2-degree lattice happens to pass very close.
for item in [
 (111.20,15.80,.045,.035), (112.40,16.80,.045,.035), # Paracels
 (111.60,16.45,.035,.035), (112.75,16.65,.035,.035),
 (114.40,10.40,.055,.035), (114.30,11.05,.035,.030), # Spratlys
 (114.10,10.90,.045,.035), (115.50,9.90,.045,.035),
 (112.90,9.55,.045,.035), (112.00,8.65,.040,.030),
 (113.85,7.90,.035,.030), (116.70,11.40,.035,.030),
 (117.75,15.80,.040,.030),                     # Scarborough area
]:
    ellipse(*item)

def is_land(x, y):
    for p in POLYGONS:
        if inside_polygon(x, y, p):
            return 1
    for cx, cy, rx, ry in ELLIPSES:
        if ((x-cx)/rx)**2 + ((y-cy)/ry)**2 <= 1.0 + 1e-12:
            return 1
    return 0

# Integer arithmetic defines the requested WGS84 lattice exactly.
lons = [99 + j/5 for j in range(141)]
lats = [23 - i/5 for i in range(111)]
matrix = np.array([[is_land(x, y) for x in lons] for y in lats],
                  dtype=np.uint8)

out = Path("outputs")
out.mkdir(parents=True, exist_ok=True)
with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
    csv.writer(f, lineterminator="\n").writerows(matrix.tolist())

raw = Image.fromarray(matrix * 255, mode="L")
raw.save(out / "candidate.png")
large = raw.resize((1128, 888), resample=Image.Resampling.NEAREST)
large.save(out / "candidate_8x.png")

# Fail loudly if any contractual property was accidentally changed.
assert matrix.shape == (111, 141) and matrix.size == 15651
assert set(np.unique(matrix).tolist()) <= {0, 1}
assert raw.size == (141, 111) and set(raw.getdata()) <= {0, 255}
assert large.size == (1128, 888) and set(large.getdata()) <= {0, 255}
assert np.array_equal(np.asarray(large), np.repeat(np.repeat(
    np.asarray(raw), 8, axis=0), 8, axis=1))
