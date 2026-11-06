from __future__ import annotations

import unittest

import run_experiments as entry


class PythonEntrypointTests(unittest.TestCase):
    def test_default_task_set_is_injected(self):
        args = entry.normalise_args(["--dry-run"])
        self.assertEqual(args, ["--tasks", entry.DEFAULT_TASKS, "--dry-run"])

    def test_powershell_tasks_parameter(self):
        args = entry.normalise_args(["-Tasks", "hecate", "--dry-run"])
        self.assertEqual(args, ["--tasks", "hecate", "--dry-run"])

    def test_python_tasks_option_wins_over_native_parameter(self):
        args = entry.normalise_args(["-Tasks", "seto", "--tasks=global,csc"])
        self.assertEqual(args, ["--tasks=global,csc"])

    def test_all_and_blank_select_default_tasks(self):
        self.assertEqual(entry.resolve_task_selection("all"), entry.DEFAULT_TASKS)
        self.assertEqual(entry.resolve_task_selection("  "), entry.DEFAULT_TASKS)

    def test_retired_arctic_task_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "arctic02 is retired"):
            entry.normalise_args(["--tasks", "arctic02"])

    def test_unknown_task_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown task 'atlantis'"):
            entry.normalise_args(["--tasks", "atlantis"])

    def test_powershell_fulfill_parameter_is_preserved(self):
        args = entry.normalise_args(["-Fulfill", "--providers", "Alias"])
        self.assertEqual(args[0], "--fulfill")
        self.assertEqual(args[1:3], ["--tasks", entry.DEFAULT_TASKS])

    def test_powershell_extra_group_parameter_is_preserved(self):
        args = entry.normalise_args(["-ExtraGroup", "2", "--dry-run"])
        self.assertEqual(args[:4], ["--extra-group", "2", "--tasks", entry.DEFAULT_TASKS])

    def test_python_extra_group_option_is_normalised(self):
        args = entry.normalise_args(["--dry-run", "--extra-group=3"])
        self.assertEqual(args[:4], ["--extra-group", "3", "--tasks", entry.DEFAULT_TASKS])
        self.assertEqual(args[-1], "--dry-run")

    def test_fulfill_and_extra_group_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "cannot be used together"):
            entry.normalise_args(["--fulfill", "--extra-group", "1"])
        with self.assertRaisesRegex(ValueError, "cannot be used together"):
            entry.normalise_args(["-Fulfill", "-ExtraGroup", "1"])

    def test_negative_extra_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be >= 0"):
            entry.normalise_args(["-ExtraGroup", "-1"])

    def test_extra_group_followed_by_another_option_is_not_skipped(self):
        args = entry.normalise_args(["-ExtraGroup", "2", "--dry-run"])
        self.assertIn("--dry-run", args)


if __name__ == "__main__":
    unittest.main()
