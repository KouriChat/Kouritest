"""Knowledge-only approximation of the Canadian Arctic Archipelago.

No geographic data, network access, reference images, or coastline libraries.
Vertices below are hand-estimated from geographical knowledge, not surveyed.
Coordinates in the rules are (degrees WEST, degrees NORTH).
All samples use point-in-polygon classification, not cell-area voting.
Inland lakes are intentionally included in land; ocean ice remains water.
Dependencies: Python 3, numpy, Pillow.
Run from the working directory: python outputs/generate_map.py
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

LAND = []
SEA = []

def land(name, points):
    LAND.append((name, points))

def sea(name, points):
    SEA.append((name, points))

# Bathurst: manually estimated outer coast, including the major northern inlet.
land("Bathurst", [
    (99.20,74.65),(100.05,74.75),(100.80,74.92),(101.35,75.14),
    (101.83,75.39),(102.15,75.58),(102.08,75.77),(102.32,75.94),
    (102.22,76.13),(102.48,76.31),(102.64,76.47),(102.37,76.65),
    (101.87,76.77),(101.53,76.65),(101.62,76.45),(101.28,76.24),
    (100.98,76.08),(100.65,75.96),(100.40,75.94),(100.32,76.08),
    (100.50,76.23),(100.18,76.41),(100.25,76.62),(100.02,76.81),
    (99.51,76.83),(99.22,76.66),(99.36,76.50),(98.97,76.40),
    (98.72,76.59),(98.26,76.61),(97.92,76.44),(97.66,76.22),
    (97.76,76.02),(98.16,75.91),(98.66,75.86),(99.20,75.75),
    (99.72,75.72),(99.84,75.62),(99.28,75.56),(98.70,75.63),
    (98.09,75.56),(97.63,75.40),(97.48,75.18),(97.77,75.01),
    (98.20,74.88),(98.60,74.77)])
sea("Bathurst western inlet", [
    (102.5,75.35),(101.53,75.34),(101.02,75.47),(100.47,75.46),
    (100.12,75.56),(100.55,75.58),(101.12,75.56),(101.72,75.47),
    (102.5,75.48)])
sea("Bathurst southeast inlet", [
    (97.2,75.03),(98.16,75.07),(98.73,75.20),(99.05,75.24),
    (98.97,75.31),(98.39,75.24),(97.2,75.20)])
land("Byam Martin", [
    (104.55,74.93),(104.89,75.06),(105.02,75.24),(104.74,75.43),
    (104.34,75.49),(103.91,75.37),(103.65,75.17),(103.88,75.02)])
land("Cameron", [
    (104.18,76.31),(104.53,76.46),(104.26,76.60),(103.72,76.65),
    (103.08,76.61),(102.82,76.48),(103.18,76.34),(103.70,76.29)])
land("Vanier", [
    (103.83,76.06),(104.08,76.18),(103.87,76.35),(103.22,76.39),
    (102.75,76.28),(102.55,76.13),(102.94,76.04),(103.35,76.00)])
land("Massey", [
    (103.58,75.87),(103.64,75.98),(103.25,76.035),(102.78,75.99),
    (102.57,75.88),(103.03,75.82)])
land("Alexander", [
    (103.39,75.65),(103.53,75.77),(103.16,75.89),(102.67,75.84),
    (102.37,75.73),(102.57,75.61),(103.04,75.59)])
land("Helena", [
    (101.42,76.59),(101.27,76.75),(100.87,76.80),(100.46,76.69),
    (100.58,76.55),(101.00,76.51)])
land("Seymour", [
    (101.46,76.78),(101.42,76.85),(101.15,76.86),(101.08,76.80)])
land("Lougheed, eastern portion", [
    (106.0,77.14),(105.76,77.39),(105.47,77.62),(105.01,77.58),
    (104.66,77.39),(104.79,77.19),(105.20,77.04),(105.69,77.07)])
land("Edmund Walker, eastern portion", [
    (105.65,76.94),(105.36,77.12),(104.98,77.13),(104.91,77.03),
    (105.20,76.93)])
