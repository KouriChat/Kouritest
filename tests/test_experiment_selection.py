from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

import run_map_experiments as exp


def provider() -> exp.Provider:
    return exp.Provider(
        id="provider-1",
        name="Alias OR",
        app_type="codex",
        is_current=True,
        model="openai/gpt-6-astra",
        base_url="https://example.test/v1",
        wire_api="responses",
        requires_openai_auth=False,
        token="test",
        auth={},
        api_format="openai_responses",
        model_catalog={},
        config_text="",
        meta={},
        alias="or",
    )


def record(task: str, group: int, status: str = "failed", *, model: str = "openai/gpt-6-astra") -> dict:
    return {
        "provider_id": "provider-1",
        "provider_name": "Alias OR",
        "provider_display": "OR",
        "provider_alias": "or",
        "model": model,
        "model_display": "GPT 6 Astra OR",
        "task": task,
        "task_label": task,
        "group": group,
        "status": status,
        "score": {"normalized": 16000.0, "accuracy_pct": 98.0} if status == "ok" else None,
        "finished_at": f"2026-10-23T20:00:{group:02d}+08:00",
    }


def write_summary(root: Path, counts: dict[str, int], filename: str = "summary.xlsx") -> None:
    wb = Workbook()
    overview = wb.active
    overview.title = "总览"
    overview.append(["模型名", "来源渠道", "测试次数"])
    for model, count in counts.items():
        overview.append([model, "OR", count])
    detail = wb.create_sheet("明细")
    detail.append(["模型名", "来源渠道"])
    for model, count in counts.items():
        sheet = wb.create_sheet(f"{model}的结果")
        sheet.append(exp.SUMMARY_HEADERS)
        for group in range(1, count + 1):
            row = [model, "OR"]
            for task in exp.TASK_ORDER:
                # Every map succeeded except global#2 for the first model.
                blank = model == "GPT 6 Astra OR" and group == 2 and task == "global"
                row.extend([None, None] if blank else [15000, 0.95])
                if task == "global":
                    row.append(None if blank else 0.05)
            row.extend([15000, 0.95, group])
            sheet.append(row)
    wb.save(root / filename)


class ExtraGroupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = exp.load_tasks(exp.DEFAULT_TASKS)

    def test_extra_group_continues_after_existing_number(self):
        records = [record("csc", 3, "ok"), record("csc", 4)]
        plan = exp.build_plan(
            [provider()], self.tasks, "csc", "openai/gpt-6-astra",
            1, 1, "auto", Path("experiments"), extra_group=2, records=records,
        )
        self.assertEqual([spec.group for spec in plan], [5, 6])
        self.assertEqual([spec.archive_stem for spec in plan], [
            "gpt-6-astra_csc@or#5", "gpt-6-astra_csc@or#6",
        ])

    def test_extra_group_uses_archived_group_even_if_failed(self):
        records = [record("csc", 3, "ok"), record("csc", 4)]
        plan = exp.build_plan(
            [provider()], self.tasks, "csc", "openai/gpt-6-astra",
            1, 1, "auto", Path("experiments"), extra_group=2,
            records=records,
        )
        self.assertEqual([spec.group for spec in plan], [5, 6])


class FulfillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = exp.load_tasks(exp.DEFAULT_TASKS)

    def test_fulfill_finds_blank_attempted_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root, {"GPT 6 Astra OR": 2})
            records = [record("global", 2)]
            targets = exp.load_fulfillment_targets(root, records, self.tasks)
        self.assertEqual(len(targets), 1)
        self.assertEqual((targets[0]["task"], targets[0]["group"]), ("global", 2))

    def test_fulfill_reads_custom_excel_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root, {"GPT 6 Astra OR": 2}, filename="batch-results.xlsx")
            records = [record("global", 2)]
            targets = exp.load_fulfillment_targets(
                root, records, self.tasks, excel_name="batch-results.xlsx"
            )
        self.assertEqual(len(targets), 1)
        self.assertEqual((targets[0]["task"], targets[0]["group"]), ("global", 2))

    def test_fulfill_ignores_blank_cell_without_failed_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root, {"GPT 6 Astra OR": 2})
            records = [record("csc", 2)]
            targets = exp.load_fulfillment_targets(root, records, self.tasks)
        self.assertEqual(targets, [])

    def test_fulfill_ignores_failure_outside_workbook_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root, {"GPT 6 Astra OR": 2})
            records = [record("global", 2), record("csc", 5)]
            targets = exp.load_fulfillment_targets(root, records, self.tasks)
        self.assertEqual(
            [(target["task"], target["group"]) for target in targets],
            [("global", 2)],
        )

    def test_fulfill_limits_all_failed_group_to_summary_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root, {"GPT 6 Astra OR": 2})
            records = [record("global", 2), record("csc", 5)]
            targets = exp.load_fulfillment_targets(root, records, self.tasks)
        self.assertEqual(
            [(target["task"], target["group"]) for target in targets],
            [("global", 2)],
        )


    def test_fulfill_plan_keeps_group_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root, {"GPT 6 Astra OR": 2})
            records = [record("global", 2)]
            targets = exp.load_fulfillment_targets(root, records, self.tasks)
            targets = exp.filter_fulfillment_targets(targets, [provider()], "all")
            plan = exp.build_fulfillment_plan(
                [provider()], self.tasks, targets, root, "api-tools", "direct",
            )
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].group, 2)
        self.assertEqual(plan[0].archive_stem, "gpt-6-astra@or#2")


if __name__ == "__main__":
    unittest.main()
