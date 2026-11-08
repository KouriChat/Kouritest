from pathlib import Path
import csv
import numpy as np
from PIL import Image

OUT = Path('outputs')
OUT.mkdir(exist_ok=True)

# Hand-written approximations, in (longitude, latitude).  These are deliberately
# polygonal: the answer is the result of one point-in-polygon rule, not a set of
# independent guesses for the 15,651 cells.
POLYS = []

def poly(*pts):
    POLYS.append([(float(x), float(y)) for x, y in pts])

def many(items):
    for p in items:
        poly(*p)

# Mainland China, Indochina, Thailand and the Malay peninsula.  The eastern
# boundary follows the broad coastal outline; the west and north are clipped by
# the requested frame.
poly((99,23),(113.7,23),(114.2,22.7),(114.6,22.2),(113.7,21.7),
     (112.7,21.1),(111.7,20.4),(110.8,19.3),(109.9,18.1),(109.2,16.6),
     (108.7,15.0),(108.3,13.0),(107.1,11.0),(105.7,9.7),(104.5,9.0),
     (103.7,9.0),(103.0,9.5),(102.0,10.2),(101.0,11.5),(100.2,13.2),
     (100.0,15.5),(99.5,18.5),(99,20.5))

# The peninsula is continued around the Gulf of Thailand and down to Singapore.
poly((103.7,9.0),(104.4,8.7),(104.6,7.7),(104.5,6.7),(104.2,5.4),
     (104.0,4.2),(104.1,3.0),(104.3,2.0),(104.2,1.25),(103.5,1.15),
     (102.9,2.3),(102.0,4.0),(101.1,5.8),(100.3,7.4),(100.0,8.8),
     (100.8,10.2),(102.0,10.2),(103.0,9.5))

# Hainan, including the large bays but retaining its detached-island status.
poly((108.65,20.05),(109.1,20.35),(110.0,20.25),(110.8,19.85),
     (111.05,19.2),(110.85,18.55),(110.25,18.1),(109.45,18.12),
     (108.85,18.55),(108.55,19.25))

# Taiwan and a few offshore islands visible at this resolution.
poly((120.05,23.0),(120.7,23.05),(121.45,22.75),(121.7,22.25),
     (121.45,21.85),(120.85,21.75),(120.25,22.0),(119.95,22.45))
poly((119.55,23.0),(119.75,23.0),(119.75,22.8),(119.55,22.8))
poly((121.35,21.45),(121.55,21.55),(121.55,21.75),(121.35,21.7))

# Sumatra (only its eastern/northern portion is inside the western frame).
poly((99.0,5.8),(99.7,5.6),(100.6,5.0),(101.5,4.2),(102.3,3.2),
     (103.2,2.2),(104.25,1.15),(104.8,1.0),(103.4,0.8),(101.8,1.0),
     (100.3,1.35),(99.4,2.4),(99.0,3.7))

# Borneo / Kalimantan, whose southern edge just touches the bottom of the box.
poly((109.45,7.35),(110.6,7.35),(112.1,7.1),(114.0,7.3),(116.0,7.0),
     (117.7,6.5),(118.7,5.3),(119.2,4.1),(118.7,2.8),(117.8,1.55),
     (116.5,1.15),(114.7,1.0),(112.7,1.0),(111.0,1.15),(109.6,1.0),
     (109.1,2.2),(109.15,3.7),(109.0,5.3))

# Bangka, Belitung and the Natuna/Anambas groups.
poly((105.5,3.1),(106.2,3.0),(106.55,2.45),(106.25,1.9),(105.65,2.0),
     (105.25,2.5))
poly((107.3,3.5),(108.0,3.4),(108.35,2.9),(108.05,2.5),(107.45,2.65))
poly((107.9,4.2),(108.7,4.25),(109.05,3.8),(108.8,3.45),(108.15,3.55))

# Philippines: larger islands are represented separately so that intervening
# straits remain water at the 0.2-degree sampling interval.
poly((120.0,18.55),(121.05,18.65),(121.9,18.15),(122.0,17.3),
     (121.55,16.3),(121.9,15.4),(123.0,14.7),(123.55,13.8),
     (123.3,13.05),(122.65,13.0),(122.0,13.45),(121.0,13.45),
     (120.4,14.2),(120.0,15.3),(119.65,16.4),(119.85,17.5))

# Batanes and the Babuyan islands.
poly((121.75,20.95),(122.0,21.15),(121.9,21.35),(121.65,21.2))
poly((121.75,19.8),(122.05,19.9),(122.0,20.15),(121.7,20.05))
poly((121.45,19.25),(121.75,19.35),(121.65,19.55),(121.35,19.5))

# Mindoro, Palawan, and the Calamian group.
poly((120.0,13.5),(120.65,13.55),(121.25,13.05),(121.25,12.2),
     (120.75,11.75),(120.05,12.0),(119.65,12.65))
poly((119.5,11.8),(119.0,11.2),(118.7,10.2),(118.45,9.2),(117.9,8.1),
     (117.5,7.1),(117.25,6.2),(117.55,5.9),(118.05,6.7),(118.5,7.8),
     (119.05,8.8),(119.55,9.9),(119.9,10.8),(120.0,11.55))
poly((119.7,12.0),(120.2,12.0),(120.45,11.7),(120.2,11.45),(119.8,11.55))

# Visayas: Panay, Negros, Cebu, Bohol, Samar and Leyte.
poly((121.0,11.0),(121.7,11.25),(122.15,10.85),(122.0,10.3),
     (121.45,10.0),(120.9,10.25),(120.65,10.65))
poly((122.55,10.1),(123.05,10.2),(123.35,9.7),(123.2,9.0),
     (122.75,8.7),(122.45,9.2))
poly((123.55,12.3),(124.15,12.15),(124.35,11.4),(124.05,10.6),
     (123.65,10.55),(123.45,11.3))
poly((124.2,11.25),(124.7,11.0),(124.9,10.35),(124.6,9.95),
     (124.2,10.15))
poly((123.8,9.0),(124.25,9.0),(124.45,8.55),(124.15,8.2),(123.8,8.4))

# Mindanao, including the large western and southern lobes.
poly((122.7,9.9),(123.45,10.2),(124.2,9.95),(125.0,9.65),(126.0,9.35),
     (126.65,8.65),(126.45,7.65),(126.0,6.9),(125.35,6.35),(124.7,6.15),
     (124.05,6.45),(123.45,6.1),(122.7,6.55),(122.35,7.25),(122.55,8.1))

# A few small islands at the eastern margin of the archipelago.
poly((126.45,10.8),(126.8,10.85),(126.9,10.55),(126.55,10.45))
poly((125.0,11.7),(125.35,11.8),(125.45,11.5),(125.1,11.4))
poly((125.9,6.2),(126.25,6.35),(126.35,6.0),(126.0,5.85))

# Western and central Sulawesi, including the northern arm which enters the
# lower-right part of the frame.
poly((119.15,1.0),(119.25,2.0),(119.8,2.9),(120.2,3.8),(120.8,4.55),
     (121.35,4.75),(121.7,4.35),(121.55,3.65),(121.0,3.0),(120.7,2.2),
     (120.5,1.35),(121.0,1.0))
poly((121.2,1.0),(121.65,1.25),(122.1,1.8),(122.7,2.05),(123.0,1.75),
     (122.65,1.25),(122.3,1.0))
poly((122.8,1.0),(123.25,1.3),(123.75,1.65),(124.15,1.55),(124.1,1.15),
     (123.7,1.0))

# Sangihe, Talaud and Halmahera fragments near the eastern edge.
poly((125.35,3.5),(125.65,3.65),(125.7,3.25),(125.45,3.05),(125.25,3.2))
poly((126.0,4.3),(126.3,4.4),(126.4,4.05),(126.1,3.9))
poly((127.0,1.0),(126.7,1.3),(126.65,2.1),(126.9,2.8),(127.0,2.9))

# South China Sea islands, cays and reef tops.  At this scale a small box is a
# useful honest approximation of a named feature; surrounding cells remain 0.
many([
 ((111.15,16.55),(111.45,16.55),(111.45,16.8),(111.15,16.8)),
 ((111.75,16.45),(112.05,16.45),(112.05,16.7),(111.75,16.7)),
 ((112.25,16.65),(112.5,16.65),(112.5,16.9),(112.25,16.9)),
 ((112.55,16.35),(112.8,16.35),(112.8,16.55),(112.55,16.55)),
 ((111.45,16.15),(111.7,16.15),(111.7,16.35),(111.45,16.35)),
])

# Dongsha / Pratas.
poly((116.65,20.65),(116.95,20.65),(117.0,20.9),(116.7,20.95))

# Spratly islands and several substantial reef crowns.
many([
 ((112.85,9.45),(113.15,9.45),(113.15,9.7),(112.85,9.7)),
 ((113.75,11.0),(114.05,11.0),(114.05,11.25),(113.75,11.25)),
 ((114.2,10.9),(114.45,10.9),(114.45,11.15),(114.2,11.15)),
 ((114.15,10.25),(114.45,10.25),(114.45,10.5),(114.15,10.5)),
 ((114.25,9.75),(114.55,9.75),(114.55,10.0),(114.25,10.0)),
 ((114.15,8.5),(114.4,8.5),(114.4,8.75),(114.15,8.75)),
 ((115.6,10.9),(115.9,10.9),(115.9,11.15),(115.6,11.15)),
 ((116.0,9.6),(116.35,9.6),(116.35,9.85),(116.0,9.85)),
 ((116.7,9.7),(117.0,9.7),(117.0,9.95),(116.7,9.95)),
 ((115.0,8.6),(115.3,8.6),(115.3,8.85),(115.0,8.85)),
 ((117.0,11.0),(117.25,11.0),(117.25,11.25),(117.0,11.25)),
])

# A few tiny islands and cays along the coast, large enough to be represented
# by at least one possible sample point.
many([
 ((108.9,10.3),(109.1,10.3),(109.1,10.5),(108.9,10.5)),
 ((109.65,12.0),(109.9,12.0),(109.9,12.25),(109.65,12.25)),
 ((110.0,13.5),(110.2,13.5),(110.2,13.75),(110.0,13.75)),
 ((113.8,15.4),(114.05,15.4),(114.05,15.65),(113.8,15.65)),
 ((117.3,14.0),(117.55,14.0),(117.55,14.25),(117.3,14.25)),
 ((118.2,15.0),(118.45,15.0),(118.45,15.25),(118.2,15.25)),
])

def inside(x, y, p):
    # Standard ray crossing with a half-open edge convention.
    hit = False
    j = len(p) - 1
    for i, (xi, yi) in enumerate(p):
        xj, yj = p[j]
        if (yi > y) != (yj > y):
            cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < cross:
                hit = not hit
        j = i
    return hit

def is_land(x, y):
    return any(inside(x, y, p) for p in POLYS)

def build():
    lons = [99.0 + 0.2*i for i in range(141)]
    lats = [23.0 - 0.2*i for i in range(111)]
    return np.array([[1 if is_land(x, y) else 0 for x in lons]
                     for y in lats], dtype=np.uint8)

def main():
    a = build()
    assert a.shape == (111, 141)
    assert set(np.unique(a)).issubset({0, 1})
    with (OUT / 'candidate.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerows(a.tolist())
    raw = Image.fromarray(a * 255, mode='L')
    raw.save(OUT / 'candidate.png')
    raw.resize((1128, 888), Image.Resampling.NEAREST).save(OUT / 'candidate_8x.png')
    print('shape=', a.shape, 'values=', int(a.sum()), '/', a.size)
    print('candidate.png=', raw.size, 'candidate_8x.png=', (1128, 888))

if __name__ == '__main__':
    main()
