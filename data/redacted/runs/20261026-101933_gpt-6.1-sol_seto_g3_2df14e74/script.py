from pathlib import Path
import math
from PIL import Image

WIDTH = 253
HEIGHT = 64
LON0 = 129.0
LAT0 = 35.0
STEP = 2.0 / 63.0

LAND = []
WATER = []


def add(poly):
    LAND.append(poly)


def add_water(poly):
    WATER.append(poly)


def ellipse(cx, cy, rx, ry, n=24, phase=0.0):
    return [
        (
            cx + rx * math.cos(phase + 2.0 * math.pi * i / n),
            cy + ry * math.sin(phase + 2.0 * math.pi * i / n),
        )
        for i in range(n)
    ]


def inside(x, y, poly):
    min_x = min(p[0] for p in poly)
    max_x = max(p[0] for p in poly)
    min_y = min(p[1] for p in poly)
    max_y = max(p[1] for p in poly)

    if x < min_x or x > max_x or y < min_y or y > max_y:
        return False

    hit = False
    j = len(poly) - 1

    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]

        dx = xj - xi
        dy = yj - yi
        cross = (x - xi) * dy - (y - yi) * dx

        if abs(cross) < 1e-10:
            if (
                min(xi, xj) - 1e-10 <= x <= max(xi, xj) + 1e-10
                and min(yi, yj) - 1e-10 <= y <= max(yi, yj) + 1e-10
            ):
                return True

        if (yi > y) != (yj > y):
            x_cross = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < x_cross:
                hit = not hit

        j = i

    return hit


def add_ellipses(items):
    for item in items:
        add(ellipse(*item))


# Northern Kyushu, using the north coast as the upper boundary.
add([
    (129.18, 33.48),
    (129.38, 33.58),
    (129.62, 33.68),
    (129.88, 33.82),
    (130.15, 33.90),
    (130.45, 33.92),
    (130.70, 33.86),
    (130.92, 33.74),
    (131.10, 33.56),
    (131.27, 33.34),
    (131.45, 33.15),
    (131.65, 33.00),
    (129.45, 33.00),
])

# A small western Kyushu coastal block.
add([
    (129.45, 33.00),
    (129.18, 33.48),
    (129.00, 33.43),
    (129.00, 33.00),
])

# Shikoku.
add([
    (131.90, 33.36),
    (132.15, 33.52),
    (132.42, 33.72),
    (132.62, 33.92),
    (132.92, 34.10),
    (133.24, 34.18),
    (133.60, 34.27),
    (133.98, 34.35),
    (134.34, 34.38),
    (134.55, 34.28),
    (134.70, 34.05),
    (134.65, 33.00),
    (131.90, 33.00),
])

# Western and central Honshu south coast.
add([
    (130.80, 34.00),
    (131.12, 34.05),
    (131.58, 34.14),
    (131.98, 34.22),
    (132.32, 34.30),
    (132.70, 34.35),
    (133.08, 34.38),
    (133.48, 34.43),
    (133.88, 34.50),
    (134.20, 34.58),
    (134.50, 34.63),
    (134.80, 34.68),
    (135.05, 34.72),
    (135.30, 34.75),
    (135.52, 34.75),
    (135.52, 35.10),
    (130.80, 35.10),
])

# Mainland around Kyoto, Osaka, and the northern side of Osaka Bay.
add([
    (135.28, 34.70),
    (135.45, 34.65),
    (135.58, 34.70),
    (135.73, 34.82),
    (135.90, 34.96),
    (136.08, 35.08),
    (136.18, 35.10),
    (135.28, 35.10),
])

# Kii Peninsula and the Mie coast.
add([
    (135.05, 34.24),
    (135.25, 34.27),
    (135.45, 34.37),
    (135.63, 34.54),
    (135.78, 34.68),
    (135.92, 34.80),
    (136.08, 34.94),
    (136.28, 35.10),
    (136.02, 35.10),
    (135.76, 34.92),
    (135.52, 34.76),
    (135.30, 34.55),
    (135.12, 34.38),
])

# Eastern side of Ise Bay and Mikawa.
add([
    (136.72, 34.66),
    (136.88, 34.72),
    (137.00, 34.72),
    (137.00, 35.10),
    (136.50, 35.10),
    (136.62, 34.90),
])

# Lake Biwa's visible southern portion.
add_water([
    (135.84, 35.00),
    (136.22, 35.00),
    (136.20, 35.08),
    (135.88, 35.08),
])

# Tsushima.
add([
    (129.15, 34.70),
    (129.22, 34.82),
    (129.37, 34.77),
    (129.48, 34.61),
    (129.45, 34.42),
    (129.38, 34.25),
    (129.25, 34.10),
    (129.15, 34.17),
    (129.09, 34.38),
])

# Iki.
add([
    (129.67, 33.96),
    (129.88, 33.98),
    (130.02, 33.88),
    (130.00, 33.72),
    (129.87, 33.66),
    (129.68, 33.70),
    (129.61, 33.83),
])

# Hirado and nearby islands.
add([
    (129.40, 33.48),
    (129.55, 33.54),
    (129.66, 33.46),
    (129.62, 33.30),
    (129.50, 33.23),
    (129.39, 33.30),
])
add_ellipses([
    (129.52, 33.57, 0.055, 0.045, 16, 0.2),
    (129.72, 33.55, 0.035, 0.025, 14, 0.0),
    (129.84, 33.53, 0.030, 0.020, 14, 0.4),
])

# Islands near the northern Kyushu coast.
add_ellipses([
    (129.92, 33.99, 0.045, 0.025, 16, 0.1),
    (130.15, 33.99, 0.050, 0.025, 16, 0.3),
    (130.38, 34.00, 0.042, 0.022, 16, 0.0),
    (130.65, 33.97, 0.030, 0.022, 16, 0.5),
    (131.00, 33.89, 0.035, 0.022, 16, 0.2),
])

# Suo-nada and Hiroshima Bay islands.
add_ellipses([
    (131.58, 34.06, 0.075, 0.040, 18, 0.1),
    (131.90, 34.04, 0.055, 0.032, 16, 0.4),
    (132.16, 34.14, 0.070, 0.045, 18, 0.0),
    (132.34, 34.30, 0.085, 0.055, 20, 0.2),
    (132.43, 34.22, 0.115, 0.080, 22, 0.4),
    (132.49, 34.10, 0.090, 0.070, 20, 0.1),
    (132.62, 34.04, 0.045, 0.032, 16, 0.0),
    (132.75, 34.22, 0.042, 0.030, 16, 0.2),
])

# Geiyo and Kurushima Strait island chain.
add_ellipses([
    (132.93, 34.30, 0.080, 0.055, 20, 0.3),
    (133.04, 34.25, 0.105, 0.075, 22, 0.1),
    (133.13, 34.20, 0.050, 0.035, 16, 0.4),
    (133.18, 34.10, 0.120, 0.070, 22, 0.2),
    (133.31, 34.13, 0.050, 0.035, 16, 0.0),
    (133.42, 34.20, 0.040, 0.028, 16, 0.5),
    (133.50, 34.30, 0.045, 0.030, 16, 0.1),
    (133.62, 34.34, 0.060, 0.035, 18, 0.2),
    (133.73, 34.30, 0.035, 0.025, 14, 0.0),
])

# Islands in Bisan Seto.
add_ellipses([
    (133.82, 34.33, 0.055, 0.035, 16, 0.1),
    (133.91, 34.37, 0.042, 0.028, 16, 0.3),
    (134.02, 34.39, 0.045, 0.030, 16, 0.0),
    (134.13, 34.38, 0.060, 0.035, 18, 0.2),
    (134.24, 34.35, 0.040, 0.025, 14, 0.4),
])

# Shodoshima.
add([
    (134.10, 34.57),
    (134.23, 34.65),
    (134.43, 34.64),
    (134.57, 34.54),
    (134.57, 34.42),
    (134.43, 34.34),
    (134.22, 34.36),
    (134.10, 34.45),
])

# Naoshima, Teshima, and nearby islands.
add_ellipses([
    (133.94, 34.46, 0.090, 0.055, 20, 0.2),
    (134.00, 34.52, 0.060, 0.040, 18, 0.5),
    (134.10, 34.55, 0.040, 0.030, 16, 0.1),
    (134.18, 34.58, 0.028, 0.020, 14, 0.4),
    (134.31, 34.43, 0.035, 0.024, 14, 0.0),
])

# Awaji Island.
add([
    (134.70, 34.66),
    (134.82, 34.75),
    (134.98, 34.66),
    (135.02, 34.48),
    (134.98, 34.27),
    (134.88, 34.08),
    (134.75, 34.12),
    (134.66, 34.30),
    (134.63, 34.50),
])

# Small islands near Awaji and the Kii Channel.
add_ellipses([
    (134.55, 34.77, 0.055, 0.030, 16, 0.2),
    (134.67, 34.82, 0.038, 0.024, 14, 0.1),
    (134.86, 34.82, 0.032, 0.022, 14, 0.4),
    (135.02, 34.28, 0.045, 0.028, 16, 0.0),
    (135.10, 34.25, 0.030, 0.020, 14, 0.3),
])

# Islands along the southern coast of Shikoku.
add_ellipses([
    (132.12, 33.39, 0.055, 0.030, 16, 0.2),
    (132.38, 33.45, 0.042, 0.025, 16, 0.0),
    (132.80, 33.62, 0.035, 0.022, 14, 0.4),
    (133.38, 33.70, 0.032, 0.020, 14, 0.1),
    (133.88, 33.78, 0.035, 0.022, 14, 0.2),
    (134.38, 33.83, 0.040, 0.025, 16, 0.0),
])

# A few small islands in Ise Bay and the Kii Channel.
add_ellipses([
    (136.35, 34.55, 0.030, 0.020, 14, 0.2),
    (136.56, 34.48, 0.025, 0.018, 14, 0.4),
    (136.72, 34.62, 0.030, 0.020, 14, 0.1),
])


def make_matrix():
    matrix = []

    for row in range(HEIGHT):
        lat = LAT0 - row * STEP
        values = []

        for col in range(WIDTH):
            lon = LON0 + col * STEP
            land = False

            for poly in LAND:
                if inside(lon, lat, poly):
                    land = True
                    break

            if land:
                for poly in WATER:
                    if inside(lon, lat, poly):
                        land = False
                        break

            values.append(1 if land else 0)

        matrix.append(values)

    return matrix


def write_csv(matrix, path):
    with path.open("w", encoding="ascii", newline="\n") as f:
        for row in matrix:
            f.write(",".join(str(value) for value in row))
            f.write("\n")


def write_images(matrix, output_dir):
    pixels = bytearray()

    for row in matrix:
        for value in row:
            pixels.append(255 if value == 1 else 0)

    image = Image.frombytes("L", (WIDTH, HEIGHT), bytes(pixels))
    image.save(output_dir / "candidate.png", format="PNG")

    enlarged = image.resize(
        (WIDTH * 8, HEIGHT * 8),
        resample=Image.Resampling.NEAREST,
    )
    enlarged.save(output_dir / "candidate_8x.png", format="PNG")


def validate(matrix, output_dir):
    assert len(matrix) == 64
    assert all(len(row) == 253 for row in matrix)
    assert all(value in (0, 1) for row in matrix for value in row)

    csv_lines = (output_dir / "candidate.csv").read_text(
        encoding="ascii"
    ).splitlines()

    assert len(csv_lines) == 64
    assert all(len(line.split(",")) == 253 for line in csv_lines)
    assert all(
        token in ("0", "1")
        for line in csv_lines
        for token in line.split(",")
    )

    original = Image.open(output_dir / "candidate.png").convert("L")
    enlarged = Image.open(output_dir / "candidate_8x.png").convert("L")

    assert original.size == (253, 64)
    assert enlarged.size == (2024, 512)

    assert set(original.getdata()).issubset({0, 255})
    assert set(enlarged.getdata()).issubset({0, 255})

    for y in range(64):
        for x in range(253):
            expected = original.getpixel((x, y))
            block = enlarged.crop((x * 8, y * 8, x * 8 + 8, y * 8 + 8))
            assert set(block.getdata()) == {expected}


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = make_matrix()
    write_csv(matrix, output_dir / "candidate.csv")
    write_images(matrix, output_dir)
    validate(matrix, output_dir)

    print("Generated 64 x 253 =", 64 * 253, "samples")
    print("outputs/candidate.csv")
    print("outputs/candidate.png: 253 x 64")
    print("outputs/candidate_8x.png: 2024 x 512")


if __name__ == "__main__":
    main()
