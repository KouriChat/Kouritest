"""Knowledge-based, approximate land mask for the Seto Inland Sea region.

No external geographic data is read. Coordinates are hand-authored WGS84
longitude/latitude outlines; polygon edges are intentionally approximate.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

LON0, LON1, NLON = 129.0, 137.0, 253
LAT0, LAT1, NLAT = 35.0, 33.0, 64

# Each ring traces a broad landmass counter-clockwise. Some small bays and
# peninsulas are simplified at the roughly 3.5 km sample spacing.
LAND = [
    # Kyushu, including the north coast facing the western Inland Sea.
    [(129.0,34.40),(129.35,34.20),(129.75,34.00),(130.05,33.88),
     (130.18,33.62),(130.35,33.50),(130.55,33.48),(130.70,33.30),
     (130.93,33.20),(131.10,33.00),(129.0,33.0)],
    # Northern/eastern Kyushu and the Kanmon approach.
    [(130.0,34.10),(130.25,34.12),(130.52,34.08),(130.72,34.04),
     (130.82,33.96),(130.94,33.92),(131.02,33.98),(131.00,34.12),
     (130.85,34.22),(130.64,34.28),(130.35,34.32),(130.0,34.40)],
    # Honshu: San-in side at the north, Seto coast south, then Kii/Ise.
    [(130.82,35.0),(137.0,35.0),(137.0,34.43),(136.82,34.42),
     (136.67,34.48),(136.54,34.57),(136.45,34.70),(136.32,34.78),
     (136.08,34.83),(135.86,34.82),(135.66,34.77),(135.45,34.70),
     (135.23,34.67),(135.05,34.65),(134.86,34.65),(134.66,34.66),
     (134.47,34.67),(134.25,34.68),(134.05,34.68),(133.85,34.66),
     (133.66,34.62),(133.48,34.59),(133.28,34.58),(133.10,34.59),
     (132.92,34.58),(132.75,34.55),(132.58,34.52),(132.42,34.48),
     (132.25,34.44),(132.10,34.39),(131.94,34.34),(131.78,34.30),
     (131.62,34.27),(131.45,34.25),(131.28,34.22),(131.12,34.17),
     (130.98,34.12),(130.88,34.12),(130.82,34.22)],
    # Shikoku: northern coast along the Inland Sea, southern Pacific coast.
    [(132.12,34.04),(132.30,34.09),(132.48,34.14),(132.67,34.17),
     (132.86,34.18),(133.04,34.17),(133.22,34.16),(133.40,34.17),
     (133.58,34.18),(133.76,34.18),(133.94,34.17),(134.12,34.17),
     (134.30,34.18),(134.47,34.18),(134.63,34.16),(134.76,34.09),
     (134.79,33.96),(134.68,33.84),(134.50,33.78),(134.32,33.75),
     (134.12,33.72),(133.92,33.70),(133.72,33.69),(133.52,33.70),
     (133.32,33.69),(133.12,33.68),(132.92,33.66),(132.72,33.64),
     (132.52,33.62),(132.32,33.60),(132.12,33.60),(131.96,33.65),
     (131.88,33.76),(131.94,33.88),(132.04,33.97)],
]
# Large islands and peninsulas in the Inland Sea and its approaches.
LAND += [
    # Awaji Island.
    [(134.72,34.64),(134.83,34.66),(134.93,34.62),(134.97,34.53),
     (134.95,34.42),(134.91,34.31),(134.86,34.22),(134.80,34.20),
     (134.75,34.26),(134.72,34.38),(134.70,34.51)],
    # Shodoshima.
    [(134.08,34.56),(134.20,34.58),(134.31,34.55),(134.40,34.51),
     (134.42,34.45),(134.35,34.43),(134.25,34.45),(134.15,34.48),
     (134.07,34.51)],
    # Naoshima and nearby islands.
    [(133.94,34.49),(134.00,34.50),(134.03,34.47),(133.99,34.44),
     (133.94,34.45)],
    [(133.80,34.52),(133.85,34.53),(133.87,34.49),(133.84,34.47),
     (133.80,34.48)],
    # Islands of the Hiroshima bay approaches.
    [(132.34,34.30),(132.45,34.32),(132.51,34.27),(132.48,34.19),
     (132.42,34.16),(132.35,34.20)],
    [(132.25,34.18),(132.34,34.20),(132.38,34.14),(132.34,34.08),
     (132.27,34.09),(132.22,34.14)],
    [(132.62,34.20),(132.76,34.22),(132.82,34.17),(132.80,34.09),
     (132.72,34.06),(132.64,34.10)],
    # Geiyo island chain, west to east: Itsukushima through Oshima.
    [(132.15,34.25),(132.22,34.27),(132.27,34.23),(132.25,34.16),
     (132.20,34.13),(132.15,34.17)],
    [(132.78,34.18),(132.88,34.21),(132.98,34.18),(133.02,34.11),
     (132.97,34.04),(132.90,34.02),(132.83,34.07)],
    [(133.08,34.20),(133.19,34.22),(133.26,34.17),(133.24,34.09),
     (133.17,34.06),(133.10,34.11)],
    [(133.28,34.25),(133.41,34.27),(133.49,34.22),(133.48,34.14),
     (133.41,34.10),(133.33,34.14)],
    [(133.51,34.27),(133.65,34.28),(133.73,34.24),(133.72,34.16),
     (133.65,34.13),(133.56,34.17)],
    [(133.73,34.30),(133.84,34.32),(133.91,34.28),(133.90,34.20),
     (133.83,34.17),(133.76,34.21)],
    [(133.94,34.29),(134.04,34.31),(134.10,34.27),(134.08,34.20),
     (134.01,34.17),(133.95,34.21)],
    [(134.11,34.27),(134.22,34.28),(134.27,34.23),(134.24,34.16),
     (134.17,34.14),(134.12,34.19)],
    # Small but prominent inland islands around Kagawa and Hiroshima.
    [(133.44,34.36),(133.52,34.38),(133.57,34.34),(133.55,34.28),
     (133.49,34.27),(133.44,34.31)],
    [(133.72,34.40),(133.79,34.41),(133.83,34.37),(133.80,34.32),
     (133.74,34.33)],
    [(134.00,34.33),(134.07,34.35),(134.10,34.31),(134.07,34.27),
     (134.01,34.28)],
    [(134.48,34.35),(134.55,34.36),(134.59,34.32),(134.56,34.27),
     (134.50,34.29)],
]
# Islands around the Kii Channel, Osaka Bay, and Ise Bay.
LAND += [
    # Kii Peninsula is represented by its west coast and south edge.
    [(134.78,34.00),(134.94,34.03),(135.10,34.05),(135.27,34.04),
     (135.42,33.98),(135.54,33.90),(135.63,33.79),(135.69,33.65),
     (135.70,33.50),(135.62,33.35),(135.50,33.22),(135.38,33.00),
     (135.18,33.00),(135.06,33.20),(135.08,33.42),(135.14,33.61),
     (135.16,33.78),(135.11,33.90),(134.98,33.96),(134.86,33.96)],
    # Awaji south-east and the Naruto-side islands.
    [(134.58,34.23),(134.67,34.25),(134.72,34.21),(134.71,34.15),
     (134.65,34.11),(134.59,34.14)],
    [(134.54,34.10),(134.61,34.12),(134.64,34.08),(134.61,34.03),
     (134.55,34.04)],
    # Shikoku offshore islands near Takamatsu and Matsuyama.
    [(134.00,34.24),(134.08,34.25),(134.12,34.21),(134.10,34.16),
     (134.04,34.15),(134.00,34.19)],
    [(134.36,34.18),(134.43,34.20),(134.46,34.16),(134.43,34.12),
     (134.37,34.13)],
    [(132.72,33.98),(132.82,34.01),(132.87,33.96),(132.84,33.90),
     (132.77,33.89),(132.72,33.93)],
    # Iyo/Northern Kyushu approach islands.
    [(131.82,33.99),(131.90,34.03),(131.96,33.99),(131.95,33.93),
     (131.88,33.91),(131.82,33.95)],
    [(131.62,34.08),(131.71,34.11),(131.76,34.06),(131.73,34.00),
     (131.65,34.01)],
    # Osaka Bay and Kii Channel small islands.
    [(134.94,34.45),(135.02,34.48),(135.08,34.44),(135.06,34.38),
     (134.99,34.36),(134.94,34.40)],
    [(135.10,34.37),(135.16,34.39),(135.20,34.35),(135.18,34.30),
     (135.12,34.31)],
    [(135.06,34.13),(135.13,34.15),(135.17,34.11),(135.14,34.06),
     (135.08,34.07)],
    # Ise Bay: Chita peninsula and a few offshore islands.
    [(136.55,35.00),(136.75,35.00),(136.74,34.88),(136.66,34.79),
     (136.58,34.72),(136.52,34.64),(136.47,34.54),(136.42,34.46),
     (136.33,34.42),(136.27,34.46),(136.30,34.56),(136.37,34.66),
     (136.43,34.76),(136.49,34.86)],
    [(136.78,34.32),(136.87,34.34),(136.92,34.29),(136.88,34.23),
     (136.80,34.24)],
    [(136.92,34.54),(137.00,34.55),(137.00,34.46),(136.94,34.45)],
]

def point_in_ring(x, y, ring):
    """Even-odd point test; boundary points are treated as land."""
    inside = False
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        cross = (x - xi) * (yj - yi) - (y - yi) * (xj - xi)
        if abs(cross) < 1e-12 and min(xi, xj) <= x <= max(xi, xj) \
                and min(yi, yj) <= y <= max(yi, yj):
            return True
        if (yi > y) != (yj > y):
            hit_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < hit_x:
                inside = not inside
        j = i
    return inside

def is_land(lon, lat):
    return any(point_in_ring(lon, lat, ring) for ring in LAND)
def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    lons = np.linspace(LON0, LON1, NLON)
    lats = np.linspace(LAT0, LAT1, NLAT)
    matrix = np.zeros((NLAT, NLON), dtype=np.uint8)
    for row, lat in enumerate(lats):
        for col, lon in enumerate(lons):
            matrix[row, col] = int(is_land(float(lon), float(lat)))

    with (out / "candidate.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(matrix.tolist())

    # Pillow mode 1 gives a two-color raster: 0 black water, 255 white land.
    image = Image.fromarray(matrix * 255, mode="L").convert("1")
    image.save(out / "candidate.png")
    image.resize((NLON * 8, NLAT * 8), resample=Image.Resampling.NEAREST).save(
        out / "candidate_8x.png"
    )

    assert matrix.shape == (64, 253) and matrix.size == 16192
    assert set(np.unique(matrix).tolist()) <= {0, 1}
    assert image.size == (253, 64)
    assert set(image.getdata()) <= {0, 255}
    enlarged = Image.open(out / "candidate_8x.png")
    assert enlarged.size == (2024, 512)
    assert set(enlarged.getdata()) <= {0, 255}
    with (out / "candidate.csv").open(newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 64 and all(len(r) == 253 for r in rows)
    assert all(v in ("0", "1") for r in rows for v in r)
    print(f"validated {matrix.shape[0]}x{matrix.shape[1]} mask, "
          f"{matrix.sum()} land samples; outputs in {out.resolve()}")

if __name__ == "__main__":
    main()
