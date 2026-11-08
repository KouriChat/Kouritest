#!/usr/bin/env python3
"""Knowledge-based, deliberately approximate South China Sea land mask."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

# Hand-drawn geographic outlines: (longitude, latitude), WGS84 degrees.
# No source data is read. Polygons are deliberately coarse approximations.
POLYGONS = [
    # Mainland Asia: Gulf of Tonkin through southern China coast.
    [(99,23),(102,23),(104,22.7),(105.5,21.8),(106.7,21.2),(107.5,20.3),
     (108.2,19.8),(109.2,19.7),(109.8,20.1),(110.5,20.2),(111.5,21),
     (112.5,21.5),(113.4,22.1),(114.2,22.5),(115.3,22.8),(116.4,22.8),
     (117.5,23),(118.5,24.5),(122,26),(127,27),(127,30),(99,30)],
    # Vietnam, Cambodia, Thailand, Myanmar and Malay Peninsula, west side.
    [(99,23),(104,23),(105.5,22.4),(106.5,21.8),(107.5,20.5),(108.2,19.5),
     (108.8,18.2),(109.2,17),(108.8,15.5),(108.3,14),(107.8,12.5),
     (107.3,11.2),(106.8,10),(106.2,9.5),(105.5,10),(104.8,10.5),
     (103.5,10.8),(102.5,11.5),(101.8,12.5),(101.2,13.5),(100.5,14.5),
     (100,15.5),(99,16.5)],
    # Southward extent of Malay peninsula; west coast, then east coast return.
    [(99,7.5),(100.1,7.2),(101,6.6),(101.8,5.8),(102,4.8),(101.8,3.8),
     (101.3,2.8),(100.5,1.5),(100.2,1),(104.4,1),(103.8,2),(103.6,3),
     (103.4,4),(103.2,5),(102.8,6),(102.4,7),(102,8),(101.8,9),
     (101.5,10),(101,11),(100.5,12),(100,13),(99,14)],
    # Taiwan.
    [(120,21.8),(120.5,22),(121.1,22.6),(121.5,23.2),(121.8,24),(121.9,24.8),
     (121.5,25.2),(121,25.3),(120.7,24.8),(120.3,24),(120,23),(119.9,22.5)],
    # Hainan.
    [(108.6,18.1),(109.2,18.1),(110,18.2),(110.6,18.7),(111,19.3),
     (110.8,19.8),(110.2,20.1),(109.5,20.1),(109,19.7),(108.7,19)],
    # Sumatra.
    [(99,5.5),(100,5.8),(101,5.5),(102,4.8),(103,3.8),(104,2.5),
     (104.5,1.2),(104.5,-1),(103.8,-3),(102.8,-5),(101.8,-5.8),
     (100.5,-4.7),(99.8,-2.5),(99.2,0),(98.8,2),(98.8,4)],
    # Borneo, coasts of Sabah/Sarawak/Brunei, Kalimantan.
    [(109.5,7.3),(110.8,7.1),(112,7),(113,6.7),(114,6.2),(115,5.2),
     (116,4.5),(117,4.2),(118,4.5),(118.5,3.5),(118.2,2.2),(117.5,1.2),
     (116.5,0.5),(115,-1),(114,-2),(113,-3),(112,-3.4),(111,-2.5),
     (110,-1),(109,0),(108.5,1),(109,2),(109.5,3.5),(109,5)],
]
POLYGONS += [
    # Palawan and Balabac chain.
    [(116.7,12.3),(117.2,12.2),(118,11.2),(118.7,10.2),(119.2,9.2),
     (119.1,8.3),(118.5,8),(118,8.7),(117.6,9.5),(117.2,10.5),(116.8,11.3)],
    # Luzon (western/southern portion in frame).
    [(119.7,18.6),(120.2,19),(121,18.8),(122,18.5),(123,17.8),(123.7,16.8),
     (123.6,15.8),(122.8,14.8),(122,14),(121,13.6),(120.4,14.2),
     (120.5,15.2),(120.2,16.2),(119.7,17.2)],
    # Mindoro.
    [(120.2,13.5),(120.8,13.5),(121.4,13),(121.5,12.3),(121.1,11.8),
     (120.4,12),(120.1,12.6)],
    # Panay.
    [(121.8,11.9),(122.5,12.1),(123.1,11.4),(123,10.6),(122.3,10.4),
     (121.8,10.8)],
    # Negros.
    [(122.8,11),(123.3,11),(123.6,10.2),(123.4,9.2),(122.9,9.1),(122.6,10)],
    # Cebu.
    [(123.5,11.3),(123.9,11),(123.9,9.5),(123.5,9.3),(123.4,10.2)],
    # Bohol.
    [(123.7,10.1),(124.4,10.2),(124.7,9.7),(124.2,9.3),(123.7,9.5)],
    # Leyte.
    [(124.6,12),(125,11.6),(125.3,10.7),(125.1,9.8),(124.7,10),(124.5,11)],
    # Samar.
    [(124.5,12.7),(125.3,12.8),(125.8,12.2),(125.7,11.3),(125.1,11),
     (124.6,11.5)],
    # Mindanao.
    [(122,8.3),(122.5,8.8),(123.2,8.6),(124,8),(124.8,8.5),(125.7,9.3),
     (126.5,9),(126.8,8),(126.4,7),(126.1,6),(125.3,5.5),(124.5,6),
     (123.7,6.5),(123,6.3),(122.2,6.8)],
    # Sulu archipelago broad islands: Basilan, Jolo, Tawi-Tawi.
    [(121.8,7.2),(122.3,7.1),(122.4,6.5),(121.9,6.3),(121.5,6.7)],
    [(120.7,6.3),(121.2,6.2),(121.3,5.8),(120.8,5.6),(120.5,5.9)],
    [(119.7,5.3),(120.1,5.2),(120.2,4.8),(119.7,4.7),(119.4,5)],
    # Bangka / Belitung / Riau islands near southern boundary.
    [(105,1),(106,1),(106.5,0),(106,-1),(105,-1.5),(104.5,-0.5)],
    [(107,1),(108,1),(108.5,0),(108,-1),(107,-1)],
]

# Small islands/reef platforms represented by ellipses (lon, lat, rx, ry).
ELLIPSES = [
    (114.17,22.28,.32,.22), (113.55,22.2,.22,.18), # HK/Macau vicinity
    (119.55,23.57,.45,.42),                         # Penghu
    (118.32,24.45,.14,.12), (118.42,24.7,.12,.10), # Kinmen area
    (115.85,20.7,.10,.08),                         # Dongsha/Pratas
    (112.3,16.83,.16,.12),                         # Woody/Paracels
    (111.65,16.45,.10,.08),(112.75,16.5,.10,.08),(111.2,15.8,.08,.07),
    (114.3,10.35,.10,.08),(114.7,11.45,.10,.08),(115.5,10.8,.10,.08),
    (116,9.75,.10,.08),(116.7,10.7,.10,.08),(113.85,9.7,.10,.08),
    (112.9,8.65,.10,.08),(114.7,8.0,.10,.08),(115.8,7.85,.10,.08),
    (117.3,7.0,.10,.08),                           # Spratly reef platforms
    (114.0,15.0,.08,.07),(117.75,15.75,.08,.07),  # Scarborough/Macclesfield-like
    (109.3,12.25,.10,.10),(109.25,13.7,.10,.10),(109.1,15.4,.10,.10),
    (104.0,10.0,.12,.10),(103.8,9.5,.10,.08),(104.4,8.6,.10,.08),
    (106.6,20.8,.12,.10),(107.7,20.8,.12,.10),     # Gulf/Tonkin islands
    (104.1,2.7,.18,.14),(104.2,3.2,.14,.12),(104.5,4.2,.16,.12),
    (108.2,3.8,.16,.12),(106.3,3.2,.12,.10),       # Natuna/Anambas
]
def on_segment(x, y, a, b, eps=1e-10):
    (x1, y1), (x2, y2) = a, b
    cross = (x-x1)*(y2-y1) - (y-y1)*(x2-x1)
    return abs(cross) <= eps and min(x1,x2)-eps <= x <= max(x1,x2)+eps and min(y1,y2)-eps <= y <= max(y1,y2)+eps

def in_polygon(x, y, poly):
    inside = False
    j = len(poly)-1
    for i in range(len(poly)):
        a, b = poly[j], poly[i]
        if on_segment(x, y, a, b):
            return True
        xi, yi = b
        xj, yj = a
        if (yi > y) != (yj > y):
            xin = (xj-xi)*(y-yi)/(yj-yi) + xi
            if x < xin:
                inside = not inside
        j = i
    return inside

def is_land(lon, lat):
    if any(in_polygon(lon, lat, p) for p in POLYGONS):
        return 1
    for cx, cy, rx, ry in ELLIPSES:
        if ((lon-cx)/rx)**2 + ((lat-cy)/ry)**2 <= 1.0:
            return 1
    return 0

def main():
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    lats = [23.0 - .2*i for i in range(111)]
    lons = [99.0 + .2*j for j in range(141)]
    mask = np.array([[is_land(lon, lat) for lon in lons] for lat in lats], dtype=np.uint8)
    assert mask.shape == (111, 141)
    assert set(np.unique(mask)).issubset({0, 1})

    with (out / "candidate.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerows(mask.tolist())

    pixels = (mask * 255).astype(np.uint8)
    original = Image.fromarray(pixels, mode="L")
    assert original.size == (141, 111)
    original.save(out / "candidate.png")
    enlarged = original.resize((1128, 888), resample=Image.Resampling.NEAREST)
    enlarged.save(out / "candidate_8x.png")

    # Deterministic validation, including exact nearest-neighbour block replication.
    with (out / "candidate.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 111 and all(len(r) == 141 for r in rows)
    assert all(v in {"0", "1"} for r in rows for v in r)
    a = np.asarray(Image.open(out / "candidate.png"))
    b = np.asarray(Image.open(out / "candidate_8x.png"))
    assert a.shape == (111,141) and b.shape == (888,1128)
    assert set(np.unique(a)).issubset({0,255}) and set(np.unique(b)).issubset({0,255})
    assert np.array_equal(b, np.repeat(np.repeat(a, 8, axis=0), 8, axis=1))
    print(f"validated: {mask.shape[0]}x{mask.shape[1]}, land={int(mask.sum())}, water={int(mask.size-mask.sum())}")

if __name__ == "__main__":
    main()
