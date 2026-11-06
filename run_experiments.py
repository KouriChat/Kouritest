#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable Python entry point for the map experiment harness.

This is a behaviour-preserving replacement for run_experiments.ps1:

* the default task set is injected when --tasks is omitted;
* PowerShell-style -Tasks, -Fulfill and -ExtraGroup parameters are accepted;
* --fulfill and --extra-group are normalised before argparse sees them;
* retired and unknown tasks are rejected with the same guidance.

It is also the entry point compiled by Nuitka.  Use ``--package-self-test``
from a packaged executable to verify that bundled task assets and the helper
Python runtime (numpy + Pillow) are available.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import run_map_experiments as experiment

DEFAULT_TASKS = "global,csc,dzh,seto,arctic01,hecate"
ALLOWED_TASKS = ("global", "csc", "dzh", "seto", "arctic01", "hecate")


def resolve_task_selection(value: str) -> str:
    """Match Resolve-TaskSelection in run_experiments.ps1."""
    trimmed = (value or "").strip()
    if not trimmed or trimmed.lower() == "all":
        return DEFAULT_TASKS
    items = [item.strip() for item in trimmed.split(",") if item.strip()]
    if not items:
        return DEFAULT_TASKS
    allowed = set(ALLOWED_TASKS)
    for item in items:
        lowered = item.lower()
        if lowered == "arctic02":
            raise ValueError(
                "arctic02 is retired. Use arctic01 (the 0.1-degree Arctic task) instead."
            )
        if lowered not in allowed:
            raise ValueError(
                f"Unknown task '{item}'. Allowed: {', '.join(ALLOWED_TASKS)}."
            )
    return ",".join(items)


def _take_value(args: list[str], index: int, option: str) -> tuple[str, int]:
    if index + 1 >= len(args):
        raise ValueError(f"{option} requires a value.")
    value = args[index + 1]
    del args[index : index + 2]
    return value, index


def _replace_tasks_option(args: list[str]) -> list[str]:
    for index, value in enumerate(args):
        if value == "--tasks":
            if index + 1 >= len(args):
                raise ValueError("--tasks requires a value.")
            args[index + 1] = resolve_task_selection(args[index + 1])
            return args
        if value.startswith("--tasks="):
            args[index] = "--tasks=" + resolve_task_selection(value.split("=", 1)[1])
            return args
    return args


def _consume_powershell_parameters(args: list[str]) -> tuple[list[str], str | None, bool, int]:
    """Consume the three parameters declared by run_experiments.ps1."""
    remaining: list[str] = []
    tasks: str | None = None
    fulfill = False
    extra_group = 0
    index = 0
    while index < len(args):
        value = args[index]
        lowered = value.lower()
        if lowered == "-tasks":
            tasks_value, index = _take_value(args, index, "-Tasks")
            tasks = resolve_task_selection(tasks_value)
            continue
        if lowered.startswith("-tasks="):
            tasks = resolve_task_selection(value.split("=", 1)[1])
            index += 1
            continue
        if lowered == "-fulfill":
            fulfill = True
            index += 1
            continue
        if lowered == "-extragroup":
            extra_value, index = _take_value(args, index, "-ExtraGroup")
            try:
                extra_group = int(extra_value)
            except ValueError as exc:
                raise ValueError("-ExtraGroup must be an integer.") from exc
            continue
        if lowered.startswith("-extragroup="):
            try:
                extra_group = int(value.split("=", 1)[1])
            except ValueError as exc:
                raise ValueError("-ExtraGroup must be an integer.") from exc
            index += 1
            continue
        remaining.append(value)
        index += 1
    return remaining, tasks, fulfill, extra_group


def _normalise_python_options(
    args: list[str], fulfill: bool, extra_group: int
) -> list[str]:
    """Normalise one --fulfill / --extra-group exactly like the wrapper."""
    index = 0
    while index < len(args):
        value = args[index]
        if value == "--fulfill":
            fulfill = True
            del args[index]
            continue
        if value.startswith("--extra-group="):
            try:
                extra_group = int(value.split("=", 1)[1])
            except ValueError as exc:
                raise ValueError("--extra-group must be an integer.") from exc
            del args[index]
            continue
        if value == "--extra-group":
            if index + 1 >= len(args):
                raise ValueError("--extra-group requires a value.")
            try:
                extra_group = int(args[index + 1])
            except ValueError as exc:
                raise ValueError("--extra-group must be an integer.") from exc
            del args[index : index + 2]
            continue
        index += 1

    if extra_group < 0:
        raise ValueError("--extra-group must be >= 0.")
    if fulfill and extra_group > 0:
        raise ValueError("--fulfill and --extra-group cannot be used together.")
    if fulfill:
        args.insert(0, "--fulfill")
    if extra_group > 0:
        args[0:0] = ["--extra-group", str(extra_group)]
    return args


def normalise_args(argv: list[str] | None = None) -> list[str]:
    args = list(sys.argv[1:] if argv is None else argv)
    args, native_tasks, native_fulfill, native_extra_group = _consume_powershell_parameters(args)
    args = _replace_tasks_option(args)
    if not any(value == "--tasks" or value.startswith("--tasks=") for value in args):
        args[0:0] = ["--tasks", native_tasks if native_tasks is not None else DEFAULT_TASKS]
    return _normalise_python_options(args, native_fulfill, native_extra_group)


def _package_self_test() -> int:
    """Validate files and the helper interpreter that must survive packaging."""
    root = experiment.ROOT
    expected = [
        experiment.DEFAULT_TASKS,
        experiment.DEFAULT_ALIASES,
        root / "quant" / "refs" / "template_land_water_90x180.csv",
    ]
    failures: list[str] = []
    for path in expected:
        if not path.is_file():
            failures.append(f"missing asset: {path}")
    try:
        tasks = experiment.load_tasks(experiment.DEFAULT_TASKS)
        if set(tasks) != set(ALLOWED_TASKS):
            failures.append(f"unexpected tasks: {sorted(tasks)}")
    except Exception as exc:
        failures.append(f"task config failed: {type(exc).__name__}: {exc}")
    try:
        quant = experiment._quant_module()
        truth = experiment.load_truth(tasks["global"])
        perfect = quant.accuracy(truth, truth)
        if float(perfect) < 0.999999:
            failures.append(f"diagnostic self-score was not perfect: {perfect}")
        quant.boundary_f1(truth.astype("uint8"), truth.astype("uint8"), 1)
    except Exception as exc:
        failures.append(f"diagnostic module failed: {type(exc).__name__}: {exc}")

    try:
        from openpyxl import Workbook

        with tempfile.TemporaryDirectory() as tmp:
            workbook_path = Path(tmp) / "self-test.xlsx"
            workbook = Workbook()
            workbook.active["A1"] = "ok"
            workbook.save(workbook_path)
            if workbook_path.stat().st_size <= 0:
                failures.append("openpyxl produced an empty workbook")
    except Exception as exc:
        failures.append(f"openpyxl failed: {type(exc).__name__}: {exc}")

    python_exe = Path(experiment.DEFAULT_PYTHON)
    if not python_exe.is_file():
        failures.append(f"helper Python not found: {python_exe}")
    else:
        probe = (
            "import sys, numpy, PIL; "
            "print(sys.version.split()[0], numpy.__version__, PIL.__version__)"
        )
        try:
            result = subprocess.run(
                [str(python_exe), "-c", probe],
                capture_output=True,
                text=True,
                timeout=30,
                env=experiment._python_env(python_exe),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode != 0:
                failures.append(
                    "helper Python import failed: "
                    + ((result.stderr or result.stdout or "").strip()[-2000:])
                )
            else:
                print(f"helper Python: {result.stdout.strip()}")
        except Exception as exc:
            failures.append(f"helper Python could not start: {type(exc).__name__}: {exc}")

    if failures:
        print("package self-test: FAILED", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"package self-test: OK ({len(tasks)} tasks)")
    print(f"asset root: {root}")
    print(f"archive root: {experiment.DEFAULT_ARCHIVE}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--package-self-test" in args:
        return _package_self_test()
    try:
        normalised = normalise_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return experiment.main(normalised)


if __name__ == "__main__":
    raise SystemExit(main())
