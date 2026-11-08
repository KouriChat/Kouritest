import csv, os
import numpy as np
from PIL import Image

W, H = 141, 111
LONS = [99.0 + 0.2*i for i in range(W)]
LATS = [23.0 - 0.2*j for j in range(H)]
POLYS = []

def poly(*pts):
    POLYS.append([(float(x), float(y)) for x, y in pts])

def ellipse(cx, cy, rx, ry, n=20):
    import math
    poly(*[(cx + rx*math.cos(2*math.pi*k/n),
            cy + ry*math.sin(2*math.pi*k/n)) for k in range(n)])

def inside(x, y, p):
    hit = False
    j = len(p)-1
    for i in range(len(p)):
        xi, yi = p[i]; xj, yj = p[j]
        if ((yi > y) != (yj > y)) and (x < (xj-xi)*(y-yi)/(yj-yi)+xi):
            hit = not hit
        j = i
    return hit

# Mainland Southeast Asia: the western/top side is clipped by the requested box.
poly((99,23),(110.0,23.0),(109.5,22.2),(108.7,21.4),
     (107.5,21.0),(106.8,20.5),(107.0,19.3),(108.0,18.0),
     (109.2,16.2),(109.5,14.6),(108.6,13.0),(107.0,11.7),
     (105.4,10.5),(103.8,10.0),(102.7,10.5),(102.0,11.4),
     (100.8,12.1),(99.0,12.7))

# Malay peninsula (west and east coasts, with the Gulf of Thailand outside it).
poly((99.3,7.0),(100.2,7.1),(101.0,6.8),(102.2,6.7),
     (103.7,6.6),(104.4,6.0),(104.1,5.2),(103.8,4.2),
     (103.5,3.0),(103.5,1.0),(101.5,1.0),(101.4,2.0),
     (101.0,3.0),(100.4,4.2),(100.0,5.4),(99.3,6.0))

# Sumatra, only its northern/eastern portion is in the frame.
poly((99.0,1.0),(104.6,1.0),(105.0,1.8),(104.8,2.6),
     (104.0,3.6),(103.0,4.7),(101.7,5.5),(100.3,5.9),
     (99.0,5.8))

# Borneo (Kalimantan, Sabah, Sarawak and Brunei).
poly((108.7,1.0),(117.2,1.0),(118.1,2.3),(118.8,4.1),
     (119.1,5.7),(117.8,7.1),(116.0,7.4),(114.2,7.0),
     (113.0,6.0),(111.8,5.0),(110.4,4.2),(109.4,2.8))

# A few large islands near the western edge of the sea.
ellipse(105.8,2.8,0.75,0.55)       # Bangka
ellipse(107.0,2.7,0.58,0.42)       # Belitung
ellipse(106.7,3.1,0.23,0.18)
ellipse(106.3,2.2,0.18,0.14)
ellipse(103.9,10.3,0.27,0.22)      # Phu Quoc
ellipse(108.3,3.8,0.75,0.62)       # Natuna Besar
ellipse(107.1,3.0,0.42,0.30)       # Anambas group
# Hainan and the nearby Chinese coast islands.
poly((108.6,19.9),(109.4,20.2),(110.4,20.0),(111.1,19.3),
     (110.9,18.5),(110.1,18.1),(109.2,18.2),(108.5,18.8))
ellipse(109.1,21.4,0.16,0.10)       # small coast island
ellipse(112.0,16.4,0.20,0.14)       # Paracel principal island
ellipse(111.7,16.7,0.10,0.08)
ellipse(112.3,16.0,0.10,0.07)
ellipse(111.5,15.9,0.07,0.06)

# Taiwan enters the northeast corner of the frame.
poly((120.0,23.0),(122.0,23.0),(121.8,22.4),(121.5,21.9),
     (120.8,21.8),(120.3,22.2),(120.0,22.7))

# Luzon and the northern Philippine islands.
poly((119.7,18.5),(120.2,19.3),(121.2,19.5),(122.0,19.0),
     (122.6,18.1),(122.3,17.1),(121.6,16.2),(120.7,15.0),
     (119.8,15.6),(119.4,16.8))
ellipse(121.7,20.5,0.16,0.10)       # Babuyan/Camiguin area
ellipse(122.1,20.8,0.10,0.08)
ellipse(122.0,20.2,0.08,0.06)
ellipse(121.2,19.9,0.08,0.06)

# Palawan and its long chain of small southern islands.
poly((117.0,11.9),(117.6,12.0),(118.5,11.5),(119.1,10.5),
     (119.5,9.4),(119.2,8.6),(118.7,8.5),(118.1,9.1),
     (117.8,10.1),(117.2,11.1))
ellipse(117.0,8.0,0.18,0.10)
ellipse(117.7,7.5,0.16,0.09)
ellipse(118.5,7.1,0.18,0.10)
ellipse(119.2,6.7,0.12,0.08)
ellipse(120.0,6.4,0.12,0.08)
ellipse(120.8,6.1,0.15,0.09)

# Mindoro and the central Philippine islands.
poly((120.1,13.5),(121.1,13.7),(121.8,13.0),(121.8,12.2),
     (121.2,11.8),(120.5,12.0),(120.0,12.7))
poly((121.0,11.8),(121.8,12.0),(122.7,11.6),(123.0,10.8),
     (122.5,10.2),(121.6,10.3),(121.1,10.9))  # Panay
poly((122.6,11.0),(123.2,11.2),(123.7,10.5),(123.6,9.5),
     (123.0,9.0),(122.6,9.6))                  # Negros
poly((123.4,11.2),(123.8,11.3),(124.1,10.5),(123.9,9.5),
     (123.6,9.5),(123.5,10.4))                # Cebu
poly((124.1,12.4),(124.8,12.8),(125.5,12.2),(125.6,11.2),
     (125.1,10.7),(124.5,10.8),(124.2,11.5)) # Samar
poly((124.2,10.8),(124.8,10.9),(125.0,10.0),(124.7,9.2),
     (124.1,9.1),(123.9,9.8))                 # Leyte
ellipse(123.8,9.6,0.35,0.22)                   # Bohol
# Mindanao and the southern Philippines.
poly((123.0,9.2),(123.7,9.8),(124.6,9.7),(125.4,9.1),
     (126.1,8.2),(126.3,7.1),(125.8,6.1),(125.1,5.3),
     (124.0,5.0),(123.3,5.7),(122.9,6.8),(122.8,8.0))
ellipse(122.6,6.2,0.23,0.16)       # Basilan
ellipse(121.8,6.0,0.13,0.09)
ellipse(122.3,5.6,0.10,0.07)
ellipse(121.3,5.8,0.12,0.08)
ellipse(120.2,6.0,0.10,0.07)

# Low islands and reefs of the Sulu Sea.
ellipse(120.9,5.1,0.11,0.07)
ellipse(121.4,5.0,0.10,0.07)
ellipse(121.9,5.2,0.10,0.07)
ellipse(122.5,4.9,0.09,0.06)
ellipse(120.5,6.6,0.11,0.08)

# Dongsha/Pratas and the principal naturally exposed Spratly features.
ellipse(116.7,20.7,0.13,0.08)
ellipse(114.3,11.1,0.12,0.08)       # Thitu
ellipse(114.4,10.7,0.10,0.07)       # Loaita
ellipse(114.4,10.4,0.11,0.08)       # Itu Aba
ellipse(114.2,10.0,0.12,0.08)       # Subi / nearby reef
ellipse(115.6,9.9,0.13,0.08)        # Mischief
ellipse(115.9,9.9,0.09,0.06)
ellipse(114.3,9.7,0.10,0.07)        # Johnson
ellipse(114.3,9.5,0.10,0.06)
ellipse(112.9,9.6,0.12,0.08)        # Fiery Cross
ellipse(114.0,8.7,0.09,0.06)        # Spratly Island area
ellipse(113.8,8.3,0.10,0.07)
ellipse(112.7,7.9,0.10,0.07)        # Amboyna Cay
ellipse(113.8,7.5,0.11,0.07)
ellipse(113.9,7.0,0.10,0.06)
ellipse(115.1,8.1,0.08,0.06)
ellipse(116.0,8.5,0.08,0.06)

# Additional small islands around the Spratly archipelago and Palawan.
ellipse(110.8,8.8,0.09,0.06)
ellipse(111.2,9.0,0.08,0.06)
ellipse(111.6,10.0,0.08,0.06)
ellipse(112.3,10.4,0.08,0.06)
ellipse(116.4,10.2,0.08,0.06)
ellipse(117.0,10.8,0.08,0.06)

def land(x, y):
    return any(inside(x, y, p) for p in POLYS)

arr = np.array([[1 if land(x, y) else 0 for x in LONS] for y in LATS],
               dtype=np.uint8)
assert arr.shape == (111, 141)
assert int(arr.sum()) >= 0

os.makedirs('outputs', exist_ok=True)
with open('outputs/candidate.csv', 'w', newline='', encoding='ascii') as f:
    w = csv.writer(f, lineterminator='\n')
    w.writerows(arr.tolist())

im = Image.fromarray(arr * 255, mode='L')
assert im.size == (141, 111)
im.save('outputs/candidate.png')
big = im.resize((141*8, 111*8), resample=Image.Resampling.NEAREST)
assert big.size == (1128, 888)
big.save('outputs/candidate_8x.png')

# Internal checks for the requested binary, complete exports.
with open('outputs/candidate.csv', encoding='ascii') as f:
    rows = list(csv.reader(f))
assert len(rows) == 111 and all(len(r) == 141 for r in rows)
assert all(v in ('0', '1') for r in rows for v in r)
assert set(im.getdata()) <= {0, 255}
