from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import run_map_experiments as exp

ROOT = Path(__file__).resolve().parents[1]


class ArcticTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = exp.load_tasks(exp.DEFAULT_TASKS)
        cls.world = exp.load_csv_matrix(ROOT / "quant" / "refs" / "template_land_water_90x180.csv")

    def test_tasks_registered(self):
        self.assertIn("arctic01", self.tasks)
        self.assertNotIn("arctic02", self.tasks)
        self.assertEqual(self.tasks["arctic01"].shape, (61, 241))
        self.assertEqual(self.tasks["arctic01"].upscale, (1928, 488))

    def test_reference_and_pngs(self):
        for key in ("arctic01",):
            task = self.tasks[key]
            truth = exp.load_truth(task)
            self.assertEqual(truth.shape, task.shape)
            self.assertTrue(set(np.unique(truth)).issubset({0, 1}))
            native = Image.open(task.ref_path.with_suffix(".png"))
            native.load()
            self.assertEqual(native.size, (task.cols, task.rows))
            self.assertTrue(np.array_equal(np.asarray(native), truth * 255))
            big_path = task.ref_path.with_name(
                f"template_region_arctic_nwp_{task.cols * 8}x{task.rows * 8}.png"
            )
            big = Image.open(big_path)
            big.load()
            self.assertEqual(big.size, task.upscale)
            expected = np.kron(truth * 255, np.ones((8, 8), dtype=np.uint8))
            self.assertTrue(np.array_equal(np.asarray(big), expected))

    def test_coarse_subgrid_matches_global(self):
        for key in ("arctic01",):
            task = self.tasks[key]
            truth = exp.load_truth(task)
            ratio = 20
            lat_top = 79.0
            lon_left = -105.0
            r0 = int(round((89.0 - lat_top) / 2.0))
            c0 = int(round((lon_left + 179.0) / 2.0))
            h = (task.rows - 1) // ratio + 1
            w = (task.cols - 1) // ratio + 1
            sub = truth[::ratio, ::ratio]
            coarse = self.world[r0:r0 + h, c0:c0 + w]
            self.assertEqual(sub.shape, coarse.shape)
            self.assertTrue(np.array_equal(sub, coarse))

    def test_perfect_score_is_normalized_to_global(self):
        for key in ("arctic01",):
            task = self.tasks[key]
            truth = exp.load_truth(task)
            score = exp.score_matrix(truth, truth)
            self.assertEqual(score["score"], task.cells)
            self.assertEqual(score["normalized"], 16200.0)
            self.assertAlmostEqual(score["accuracy_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
