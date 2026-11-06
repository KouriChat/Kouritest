from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import run_map_experiments as exp

ROOT = Path(__file__).resolve().parents[1]


class ExcelNameTests(unittest.TestCase):
    def test_parser_defaults_to_summary_xlsx(self):
        args = exp.build_parser().parse_args(["--report-only"])
        self.assertEqual(args.excel_name, "summary.xlsx")

    def test_custom_name_gets_xlsx_suffix(self):
        self.assertEqual(exp.normalize_excel_name("batch-a"), "batch-a.xlsx")
        self.assertEqual(exp.normalize_excel_name("批次A.xlsx"), "批次A.xlsx")

    def test_rejects_paths(self):
        for value in ("../summary.xlsx", "sub/summary.xlsx", r"sub\summary.xlsx", ""):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    exp.normalize_excel_name(value)

    def test_write_reports_uses_custom_filename(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
            root = Path(tmp)
            exp.write_reports(root, [], excel_name="custom-report.xlsx")
            self.assertTrue((root / "custom-report.xlsx").is_file())
            self.assertFalse((root / "summary.xlsx").exists())

    def test_write_reports_keeps_default_filename(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
            root = Path(tmp)
            exp.write_reports(root, [])
            self.assertTrue((root / "summary.xlsx").is_file())


if __name__ == "__main__":
    unittest.main()
