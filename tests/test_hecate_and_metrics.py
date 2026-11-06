from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import run_map_experiments as exp

ROOT = Path(__file__).resolve().parents[1]


class HecateTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = exp.load_tasks(exp.DEFAULT_TASKS)
        cls.world = exp.load_csv_matrix(
            ROOT / "quant" / "refs" / "template_land_water_90x180.csv"
        )

    def test_task_registered(self):
        task = self.tasks["hecate"]
        self.assertEqual(task.shape, (127, 127))
        self.assertEqual(task.upscale, (1016, 1016))
        self.assertEqual(task.cells, 16129)
        self.assertIn("赫卡特", task.label)

    def test_reference_pngs_and_coarse_grid(self):
        task = self.tasks["hecate"]
        truth = exp.load_truth(task)
        self.assertEqual(truth.shape, (127, 127))
        self.assertTrue(set(np.unique(truth)).issubset({0, 1}))

        native = Image.open(task.ref_path.with_suffix(".png"))
        native.load()
        self.assertEqual(native.size, (127, 127))
        self.assertTrue(np.array_equal(np.asarray(native), truth * 255))

        big = Image.open(
            ROOT / "quant" / "refs" / "template_region_hecate_strait_1016x1016.png"
        )
        big.load()
        expected = np.kron(truth * 255, np.ones((8, 8), dtype=np.uint8))
        self.assertEqual(big.size, (1016, 1016))
        self.assertTrue(np.array_equal(np.asarray(big), expected))

        sub = truth[::126, ::126]
        coarse = self.world[17:19, 24:26]
        self.assertTrue(np.array_equal(sub, coarse))

    def test_perfect_map_metrics(self):
        task = self.tasks["hecate"]
        truth = exp.load_truth(task)
        cand = exp.Candidate(
            matrix=truth,
            kind="exact",
            source=task.ref_path.with_suffix(".png"),
            how="unit-test",
            meta={"levels": [0, 255]},
        )
        metrics = exp.map_quality_metrics(cand, truth)
        self.assertAlmostEqual(metrics["accuracy_pct"], 100.0)
        self.assertAlmostEqual(metrics["hss"], 1.0)
        self.assertAlmostEqual(metrics["boundary_f1_d1"], 1.0)
        self.assertAlmostEqual(metrics["narrow_water_keep"], 1.0)
        self.assertAlmostEqual(metrics["island_iou_median"], 1.0)
        self.assertAlmostEqual(metrics["island_iou_area_weighted"], 1.0)
        self.assertAlmostEqual(metrics["island_iou_ge50"], 1.0)

    def test_prompt_and_normalized_score(self):
        task = self.tasks["hecate"]
        prompt = task.prompt_path.read_text(encoding="utf-8")
        self.assertIn("2/126°", prompt)
        self.assertIn("127×127", prompt)
        self.assertIn("1016×1016", prompt)
        truth = exp.load_truth(task)
        score = exp.score_matrix(truth, truth)
        self.assertEqual(score["normalized"], 16200.0)

    def test_summary_excludes_arctic02(self):
        self.assertNotIn("arctic02", exp.TASK_ORDER)
        self.assertIn("arctic01", exp.TASK_ORDER)
        self.assertFalse(any("北极航道0.2" in h for h in exp.SUMMARY_HEADERS))
        base = {
            "status": "ok",
            "model_display": "unit-test",
            "provider_display": "unit",
            "model": "unit",
            "provider_alias": "",
            "group": 1,
            "score": {"normalized": 16200.0, "accuracy_pct": 100.0},
        }
        recs = [
            {**base, "task": "arctic01", "task_label": "北极航道 0.1°"},
            {**base, "task": "arctic02", "task_label": "北极航道 0.2°"},
        ]
        aggregates = exp.aggregate_records(recs)
        self.assertEqual(len(aggregates), 1)
        self.assertEqual(set(aggregates[0]["task_scores"]), {"arctic01"})


if __name__ == "__main__":
    unittest.main()

