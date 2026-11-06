#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the land/water map experiment end to end.

The script mirrors the original manual workflow:

  1. read the frozen prompts from experiment_config/prompts/
  2. start one isolated Codex session per provider/model/task/group
  3. let the model write its script, CSV and PNGs in a temporary work directory
  4. decode and score the candidate with the same 1 wrong cell = 1 point rule
  5. archive the result under experiments/<model>/ with the historical naming
     convention model[_task][@provider][#group].png

CC Switch is used as the provider registry.  The script reads
~/.cc-switch/cc-switch.db read-only.  In `proxy` mode it sends requests through
the CC Switch local proxy (the currently selected provider).  In `direct` mode
it creates a temporary CODEX_HOME from that provider's stored configuration.
`auto` uses the proxy when the provider is current, otherwise direct for
Responses-API providers.

Examples
--------
  python run_map_experiments.py --list-providers
  python run_map_experiments.py --dry-run --providers "Local Provider" --tasks seto --groups 2
  python run_map_experiments.py --providers "Local Provider,Alias OR" --tasks all --groups 1
  python run_map_experiments.py --providers all --tasks all --groups 2

Backends
  api-tools  (default) provider-native tool loop with host-side confined tools
  api        one-shot response; harness extracts and runs the returned script
  codex      codex exec agent; requires a working Windows sandbox
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import tomllib
import traceback
import urllib.error
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

def _detect_resource_root() -> Path:
    """Locate bundled prompts and references in source and Nuitka builds."""
    module_root = Path(__file__).resolve().parent
    executable_root = Path(sys.argv[0]).resolve().parent
    for candidate in (module_root, executable_root):
        if (candidate / "experiment_config" / "tasks.json").is_file():
            return candidate
    return module_root


def _detect_application_root(resource_root: Path) -> Path:
    """Keep writable experiment output outside Nuitka onefile's temp folder."""
    override = os.environ.get("MAP_EXPERIMENT_APP_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    executable = Path(sys.argv[0]).resolve()
    if executable.stem.lower() in {"python", "pythonw", "py"}:
        return resource_root
    return executable.parent


def _detect_python_runtime(resource_root: Path, app_root: Path) -> Path:
    override = os.environ.get("MAP_EXPERIMENT_PYTHON", "").strip()
    if override:
        return Path(override).expanduser()
    for base in (resource_root, app_root):
        for relative in (
            Path("_python") / "python.exe",
            Path("_python_runtime") / "python.exe",
            Path("python-runtime") / "python.exe",
        ):
            candidate = base / relative
            if candidate.is_file():
                return candidate
    return Path(sys.executable)


ROOT = _detect_resource_root()
APP_ROOT = _detect_application_root(ROOT)
GLOBAL_CELLS = 16200
DEFAULT_TASKS = ROOT / "experiment_config" / "tasks.json"
DEFAULT_ALIASES = ROOT / "experiment_config" / "providers.json"
DEFAULT_ARCHIVE = APP_ROOT / "experiments"
DEFAULT_DB = Path.home() / ".cc-switch" / "cc-switch.db"
DEFAULT_PROXY = "http://127.0.0.1:15721/v1"
DEFAULT_PYTHON = _detect_python_runtime(ROOT, APP_ROOT)
DEFAULT_CATALOG = Path.home() / ".codex" / "cc-switch-model-catalog.json"
API_USER_AGENT = "codex/0.164.0"
TASK_SUFFIX = {
    "global": "",
    "csc": "_csc",
    "dzh": "_dzh",
    "seto": "_seto",
    "arctic01": "_arc01",
    "hecate": "_hec",
}
TASK_LABEL = {
    "global": "全球 2°",
    "csc": "南海 0.2°",
    "dzh": "地中海 0.2°",
    "seto": "濑户内海 2/63°",
    "arctic01": "北极航道 0.1°",
    "hecate": "赫卡特海峡 2/126°",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "provider"


def normalize_model(name: str) -> str:
    """Historical archive model name: strip vendor prefix and -codex/-code."""
    n = (name or "").strip()
    if "/" in n:
        n = n.rsplit("/", 1)[-1]
    for suffix in ("-codex", "-code"):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
            break
    return n or "unknown-model"


def resolve_path(value: str | Path, base: Path = ROOT) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass
class Task:
    key: str
    label: str
    prompt_path: Path
    shape: tuple[int, int]          # rows, cols
    upscale: tuple[int, int]        # width, height
    ref_path: Path
    map_key: str
    archive_suffix: str

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def cols(self) -> int:
        return self.shape[1]

    @property
    def cells(self) -> int:
        return self.rows * self.cols


@dataclass
class Provider:
    id: str
    name: str
    app_type: str
    is_current: bool
    model: str
    base_url: str
    wire_api: str
    requires_openai_auth: bool
    token: str
    auth: dict[str, Any]
    api_format: str
    model_catalog: Any
    config_text: str
    meta: dict[str, Any]
    alias: str = ""

    @property
    def runnable(self) -> bool:
        return bool(self.model and self.base_url)

    @property
    def display(self) -> str:
        return f"{self.name} ({self.model or 'no model'})"


@dataclass
class RunSpec:
    provider: Provider
    model: str
    task: Task
    group: int
    archive_stem: str
    archive_dir: Path
    mode: str
    skip_reason: str = ""


def load_tasks(path: Path) -> dict[str, Task]:
    raw = read_json(path)
    tasks: dict[str, Task] = {}
    for key, cfg in raw.items():
        prompt = resolve_path(cfg["prompt"])
        if not prompt.exists():
            raise FileNotFoundError(f"task {key}: prompt not found: {prompt}")
        shape = tuple(int(x) for x in cfg["shape"])
        upscale = tuple(int(x) for x in cfg["upscale"])
        ref = resolve_path(cfg["ref"])
        if not ref.exists():
            raise FileNotFoundError(f"task {key}: reference not found: {ref}")
        tasks[key] = Task(
            key=key,
            label=str(cfg.get("label", TASK_LABEL.get(key, key))),
            prompt_path=prompt,
            shape=(shape[0], shape[1]),
            upscale=(upscale[0], upscale[1]),
            ref_path=ref,
            map_key=str(cfg.get("map_key", key)),
            archive_suffix=str(cfg.get("archive_suffix", TASK_SUFFIX.get(key, f"_{key}"))),
        )
    return tasks


def _extract_model_catalog(settings: dict[str, Any], meta: dict[str, Any]) -> Any:
    return settings.get("modelCatalog") or meta.get("modelCatalog") or None


def _extract_auth(settings: dict[str, Any]) -> dict[str, Any]:
    auth = settings.get("auth") or {}
    return auth if isinstance(auth, dict) else {}


def load_providers(db_path: Path, aliases: dict[str, str]) -> list[Provider]:
    if not db_path.exists():
        raise FileNotFoundError(f"CC Switch database not found: {db_path}")
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, app_type, name, settings_config, meta, is_current "
            "FROM providers WHERE app_type='codex' "
            "ORDER BY COALESCE(sort_index, 999999), name"
        ).fetchall()
    finally:
        conn.close()

    providers: list[Provider] = []
    for row in rows:
        try:
            settings = json.loads(row["settings_config"] or "{}")
        except json.JSONDecodeError:
            settings = {}
        try:
            meta = json.loads(row["meta"] or "{}")
        except json.JSONDecodeError:
            meta = {}
        config_text = settings.get("config") or ""
        try:
            doc = tomllib.loads(config_text)
        except Exception:
            doc = {}
        provider_key = str(doc.get("model_provider", "custom"))
        provider_cfg = (doc.get("model_providers") or {}).get(provider_key, {})
        if not isinstance(provider_cfg, dict):
            provider_cfg = {}
        model = str(doc.get("model") or "")
        base_url = str(provider_cfg.get("base_url") or "")
        wire_api = str(provider_cfg.get("wire_api") or "responses")
        requires = bool(provider_cfg.get("requires_openai_auth", False))
        auth = _extract_auth(settings)
        token = str(provider_cfg.get("experimental_bearer_token") or "")
        if not token:
            token = str(auth.get("OPENAI_API_KEY") or "")
        if not token and isinstance(auth.get("tokens"), dict):
            token = str(auth["tokens"].get("access_token") or "")
        api_format = str(meta.get("apiFormat") or "openai_responses")
        alias = aliases.get(row["name"], slugify(row["name"]))
        providers.append(Provider(
            id=str(row["id"]),
            name=str(row["name"]),
            app_type=str(row["app_type"]),
            is_current=bool(row["is_current"]),
            model=model,
            base_url=base_url,
            wire_api=wire_api,
            requires_openai_auth=requires,
            token=token,
            auth=auth,
            api_format=api_format,
            model_catalog=_extract_model_catalog(settings, meta),
            config_text=config_text,
            meta=meta,
            alias=alias,
        ))
    return providers


def provider_models(provider: Provider) -> list[str]:
    models: list[str] = []
    if provider.model:
        models.append(provider.model)
    catalog = provider.model_catalog or {}
    if isinstance(catalog, dict):
        for item in catalog.get("models", []):
            if not isinstance(item, dict):
                continue
            name = item.get("slug") or item.get("model") or item.get("id")
            if name and str(name).strip() and str(name).strip() not in models:
                models.append(str(name).strip())
    return models


def select_providers(providers: list[Provider], selector: str | None) -> list[Provider]:
    selector = (selector or "all").strip()
    if selector.lower() == "all":
        return providers
    if selector.lower() == "current":
        return [p for p in providers if p.is_current]
    wanted = [x.strip() for x in selector.split(",") if x.strip()]
    out: list[Provider] = []
    for token in wanted:
        matches = [p for p in providers if p.id == token or p.name.lower() == token.lower()]
        if not matches:
            raise SystemExit(f"unknown CC Switch provider: {token!r}; use --list-providers")
        out.extend(matches)
    # de-duplicate while preserving order
    seen: set[str] = set()
    unique: list[Provider] = []
    for p in out:
        if p.id not in seen:
            seen.add(p.id)
            unique.append(p)
    return unique


def proxy_health(proxy_url: str, timeout: float = 5.0) -> tuple[bool, str]:
    url = proxy_url.rstrip("/") + "/models"
    try:
        req = urllib.request.Request(url, headers={"Authorization": "Bearer PROXY_MANAGED"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(4096).decode("utf-8", "ignore")
        return True, f"HTTP {getattr(resp, 'status', '?')}; {body[:80]}"
    except Exception as exc:
        return False, str(exc)


def choose_mode(provider: Provider, requested: str, proxy_ok: bool, allow_switch: bool = True) -> tuple[str, str]:
    requested = (requested or "auto").lower()
    if requested == "proxy":
        if not proxy_ok:
            return "", "CC Switch proxy is not reachable"
        if not provider.is_current and not allow_switch:
            return "", "provider is not current and --no-switch-cc-provider was given"
        return "proxy", ""
    if requested == "direct":
        if provider.api_format == "openai_chat":
            return "", "direct Codex requires Responses API; provider is openai_chat"
        if not provider.base_url:
            return "", "provider has no base_url"
        return "direct", ""
    # auto: prefer the local proxy when switching is allowed, otherwise use
    # the current provider through the proxy and direct Responses providers.
    if proxy_ok and (provider.is_current or allow_switch):
        return "proxy", ""
    if provider.api_format == "openai_chat":
        return "", "openai_chat provider is not current; select it in CC Switch or use --provider-mode proxy"
    if not provider.base_url:
        return "", "provider has no base_url"
    return "direct", ""


def resolve_api_transport(provider: Provider, requested: str, proxy_ok: bool) -> tuple[str, str]:
    requested = (requested or "auto").lower()
    if requested == "direct":
        if provider.api_format != "openai_responses":
            return "", "direct transport requires an OpenAI Responses provider"
        if not provider.base_url or not provider.token:
            return "", "direct transport requires base_url and API key"
        return "direct", ""
    if requested == "proxy":
        if not proxy_ok:
            return "", "CC Switch proxy is not reachable"
        if not provider.is_current:
            return "", "provider is not current; select it in CC Switch UI first"
        return "proxy", ""
    # auto: avoid writing CC Switch's database; use the provider endpoint directly
    # for non-current Responses providers, and the local proxy for the current one.
    if provider.is_current and proxy_ok:
        return "proxy", ""
    if provider.api_format == "openai_responses" and provider.base_url and provider.token:
        return "direct", ""
    return "", "provider is not current and cannot be called directly (openai_chat)"


def _next_group_number(
    records: list[dict[str, Any]],
    provider: Provider,
    model: str,
    task_key: str,
) -> int:
    """Continue numbering from the largest existing group for this cell.

    Existing #N filenames always win, including failed attempts, because the
    new run must not overwrite them.  If no archive exists, numbering starts
    at #1 (no suffix).
    """
    groups: list[int] = []
    for rec in records:
        if str(rec.get("task")) != task_key:
            continue
        if str(rec.get("model")) != str(model):
            continue
        if str(rec.get("provider_id") or "") != provider.id and str(rec.get("provider_name") or "") != provider.name:
            continue
        try:
            groups.append(int(rec.get("group", 1)))
        except (TypeError, ValueError):
            continue
    return (max(groups) + 1) if groups else 1


def build_plan(
    providers: list[Provider],
    tasks: dict[str, Task],
    task_selector: str,
    model_selector: str | None,
    groups: int,
    start_group: int,
    provider_mode: str,
    archive_root: Path,
    allow_switch: bool = True,
    backend: str = "api",
    api_transport: str = "auto",
    extra_group: int = 0,
    records: list[dict[str, Any]] | None = None,
) -> list[RunSpec]:
    if task_selector.lower() == "all":
        selected_tasks = list(tasks.values())
    else:
        selected_tasks = []
        for key in [x.strip() for x in task_selector.split(",") if x.strip()]:
            if key not in tasks:
                raise SystemExit(f"unknown task {key!r}; available: {', '.join(tasks)}")
            selected_tasks.append(tasks[key])

    proxy_ok, proxy_msg = proxy_health(DEFAULT_PROXY)
    plan: list[RunSpec] = []
    for provider in providers:
        if not provider.runnable:
            plan.append(RunSpec(
                provider=provider, model=provider.model or "?",
                task=selected_tasks[0] if selected_tasks else next(iter(tasks.values())),
                group=start_group, archive_stem="", archive_dir=archive_root,
                mode="", skip_reason="provider has no model/base_url",
            ))
            continue
        if model_selector:
            raw_models = [x.strip() for x in model_selector.split(",") if x.strip()]
            if len(raw_models) == 1 and raw_models[0].lower() == "all":
                models = provider_models(provider)
            else:
                models = raw_models
        else:
            models = [provider.model]
        if backend in ("api", "api-tools"):
            mode, reason = resolve_api_transport(provider, api_transport, proxy_ok)
        else:
            mode, reason = choose_mode(provider, provider_mode, proxy_ok, allow_switch)
        for model in models:
            model_norm = normalize_model(model)
            for task in selected_tasks:
                if extra_group > 0:
                    first_group = _next_group_number(
                        records or [], provider, model, task.key
                    )
                    group_numbers = range(first_group, first_group + extra_group)
                else:
                    group_numbers = range(start_group, start_group + groups)
                for group in group_numbers:
                    provider_suffix = f"@{provider.alias}" if provider.alias else ""
                    group_suffix = "" if group == 1 else f"#{group}"
                    stem = f"{model_norm}{task.archive_suffix}{provider_suffix}{group_suffix}"
                    plan.append(RunSpec(
                        provider=provider,
                        model=model,
                        task=task,
                        group=group,
                        archive_stem=stem,
                        archive_dir=archive_root / model_norm,
                        mode=mode,
                        skip_reason=reason,
                    ))
    return plan

# ---------------------------------------------------------------------------
# Reference loading, PNG decoding and scoring
# ---------------------------------------------------------------------------

def load_csv_matrix(path: Path) -> np.ndarray:
    rows: list[list[int]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p for p in re.split(r"[,\s;]+", line) if p]
        try:
            vals = [int(float(p)) for p in parts]
        except ValueError:
            continue
        if vals:
            rows.append(vals)
    if not rows:
        raise ValueError(f"{path}: no numeric rows")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{path}: ragged CSV")
    arr = np.asarray(rows, dtype=np.uint8)
    if int(arr.min()) < 0 or int(arr.max()) > 1:
        raise ValueError(f"{path}: values must be 0/1")
    return arr


def decode_image(path: Path, task: Task) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode exactly like score_map.py: convert('L'), then >127.5."""
    im = Image.open(path)
    im.load()
    meta: dict[str, Any] = {
        "mode": im.mode,
        "size": list(im.size),
        "info_bits": im.info.get("bits"),
    }
    g = im.convert("L")
    arr = np.asarray(g, dtype=np.uint8)
    levels = sorted(int(x) for x in np.unique(arr).tolist())
    meta["levels"] = levels
    land = arr > 127.5
    h, w = task.rows, task.cols
    ih, iw = land.shape
    if (ih, iw) == (h, w):
        grid = land
        how = f"{iw}x{ih} -> used as-is"
        kind = "exact"
    elif ih % h == 0 and iw % w == 0:
        by, bx = ih // h, iw // w
        grid = land.reshape(h, by, w, bx).mean(axis=(1, 3)) >= 0.5
        how = f"{iw}x{ih} -> {by}x{bx} block majority vote"
        kind = "upscale" if (ih, iw) == (task.upscale[1], task.upscale[0]) else "scaled"
    else:
        small = Image.fromarray((land * 255).astype(np.uint8), mode="L").resize(
            (w, h), Image.Resampling.BOX
        )
        grid = np.asarray(small, dtype=np.uint8) > 127.5
        how = f"{iw}x{ih} -> box-filter resample (lossy)"
        kind = "other"
    meta["kind"] = kind
    meta["how"] = how
    meta["land_fraction"] = float(grid.mean())
    return grid.astype(np.uint8), meta


def score_matrix(cand: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    diff = cand != truth
    n = int(diff.size)
    bad = int(diff.sum())
    correct = n - bad
    acc = 100.0 * correct / n if n else 0.0
    flip_v = float(np.mean(np.flipud(cand) == truth))
    flip_h = float(np.mean(np.fliplr(cand) == truth))
    return {
        "errors": bad,
        "correct": correct,
        "score": correct,
        "max_score": n,
        "cells": n,
        "accuracy_pct": acc,
        "normalized": correct * GLOBAL_CELLS / n if n else 0.0,
        "extra_land": int((cand & ~truth).sum()),
        "missing_land": int((~cand & truth).sum()),
        "orientation_ok": bool((correct / n) + 1e-12 >= flip_v and (correct / n) + 1e-12 >= flip_h),
        "flip_vertical_accuracy": flip_v * 100.0,
        "flip_horizontal_accuracy": flip_h * 100.0,
        "diff": diff,
    }


def save_diff(diff: np.ndarray, cand: np.ndarray, path: Path) -> None:
    h, w = diff.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[cand.astype(bool) & ~diff] = (60, 60, 60)
    rgb[(~cand.astype(bool)) & diff] = (255, 60, 60)
    rgb[cand.astype(bool) & diff] = (60, 160, 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path)
    k = max(1, int(round(900 / max(w, 1))))
    Image.fromarray(rgb, mode="RGB").resize((w * k, h * k), Image.Resampling.NEAREST).save(
        path.with_name(path.stem + f"_x{k}.png")
    )


def write_matrix_png(matrix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((matrix.astype(np.uint8) * 255), mode="L").save(path)


def write_matrix_csv(matrix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in matrix:
            f.write(",".join("1" if int(v) else "0" for v in row) + "\n")


def nn_upscale(matrix: np.ndarray, k: int = 8) -> np.ndarray:
    return np.repeat(np.repeat(matrix.astype(np.uint8), k, axis=0), k, axis=1)


def _is_internal_path(path: Path) -> bool:
    internal = {".codex-home", ".tmp", ".tools", ".git", "__pycache__"}
    return any(part.lower() in internal for part in path.parts)


def _noise_png(path: Path) -> bool:
    n = path.name.lower()
    return any(x in n for x in ("diff", "template", "reference", "uncertainty"))


@dataclass
class Candidate:
    matrix: np.ndarray
    kind: str
    source: Path | None
    how: str
    meta: dict[str, Any]
    csv_path: Path | None = None
    csv_match: bool = False
    upscale_path: Path | None = None
    upscale_match: bool = False


def _csv_candidates(run_dir: Path, task: Task) -> list[tuple[Path, np.ndarray]]:
    out: list[tuple[Path, np.ndarray]] = []
    for path in run_dir.rglob("*.csv"):
        if _is_internal_path(path):
            continue
        if any(x in path.name.lower() for x in ("template", "reference", "uncertainty")):
            continue
        try:
            m = load_csv_matrix(path)
        except Exception:
            continue
        if m.shape == (task.rows, task.cols):
            out.append((path, m))
    return out


def _candidate_rank(cand: Candidate, run_dir: Path) -> tuple:
    kind_rank = {"exact": 0, "upscale": 1, "scaled": 2, "other": 3, "csv": 4}.get(cand.kind, 9)
    in_outputs = 0
    if cand.source is not None:
        try:
            in_outputs = 0 if "outputs" in [p.lower() for p in cand.source.relative_to(run_dir).parts] else 1
        except Exception:
            in_outputs = 1
    mtime = cand.source.stat().st_mtime if cand.source and cand.source.exists() else 0.0
    return (0 if cand.csv_match else 1, kind_rank, in_outputs, -mtime)


def _find_upscale_match(run_dir: Path, task: Task, matrix: np.ndarray) -> Path | None:
    target = nn_upscale(matrix).astype(bool)
    for path in run_dir.rglob("*.png"):
        if _is_internal_path(path) or _noise_png(path):
            continue
        try:
            im = Image.open(path)
            im.load()
            if im.size != (task.upscale[0], task.upscale[1]):
                continue
            arr = np.asarray(im.convert("L"), dtype=np.uint8) > 127.5
            if np.array_equal(arr, target):
                return path
        except Exception:
            continue
    return None


def find_candidate(run_dir: Path, task: Task) -> Candidate | None:
    csvs = _csv_candidates(run_dir, task)
    images: list[Candidate] = []
    for path in run_dir.rglob("*.png"):
        if _is_internal_path(path) or _noise_png(path):
            continue
        try:
            matrix, meta = decode_image(path, task)
        except Exception:
            continue
        if matrix.shape != (task.rows, task.cols):
            continue
        csv_path = None
        csv_match = False
        for cp, cm in csvs:
            if cm.shape == matrix.shape and np.array_equal(cm, matrix):
                csv_path = cp
                csv_match = True
                break
        images.append(Candidate(
            matrix=matrix,
            kind=str(meta.get("kind", "other")),
            source=path,
            how=str(meta.get("how", "")),
            meta=meta,
            csv_path=csv_path,
            csv_match=csv_match,
        ))
    if images:
        chosen = sorted(images, key=lambda c: _candidate_rank(c, run_dir))[0]
        up = _find_upscale_match(run_dir, task, chosen.matrix)
        if up is not None:
            chosen.upscale_path = up
            chosen.upscale_match = True
        return chosen
    if csvs:
        cp, cm = sorted(csvs, key=lambda t: t[0].stat().st_mtime, reverse=True)[0]
        return Candidate(
            matrix=cm,
            kind="csv",
            source=None,
            how="CSV matrix (PNG missing)",
            meta={"mode": "csv", "size": [task.cols, task.rows], "levels": [0, 1]},
            csv_path=cp,
            csv_match=True,
        )
    return None


def validate_candidate(cand: Candidate, task: Task, truth: np.ndarray, score: dict[str, Any]) -> dict[str, Any]:
    levels = set(cand.meta.get("levels") or [])
    checks = {
        "png_present": cand.source is not None,
        "png_exact_shape": cand.kind == "exact",
        "png_binary": levels.issubset({0, 255}),
        "csv_present": cand.csv_path is not None,
        "csv_agrees_with_png": bool(cand.csv_match),
        "upscale_present": cand.upscale_path is not None,
        "upscale_exact_nn": bool(cand.upscale_match),
        "orientation": bool(score.get("orientation_ok", False)),
    }
    required = ("png_present", "png_exact_shape", "png_binary", "csv_present",
                "csv_agrees_with_png", "upscale_present", "upscale_exact_nn", "orientation")
    status = "ok" if all(checks[k] for k in required) else "partial"
    return {"status": status, "checks": checks}


def load_truth(task: Task) -> np.ndarray:
    truth = load_csv_matrix(task.ref_path)
    if truth.shape != (task.rows, task.cols):
        raise ValueError(f"{task.key}: reference shape {truth.shape} != {(task.rows, task.cols)}")
    return truth


# ---------------------------------------------------------------------------
# CC Switch integration
# ---------------------------------------------------------------------------

MODEL_DISPLAY = {
    "gpt-6-astra": "GPT 6 Astra",
    "gpt-6.1-sol": "GPT 6.1 Sol",
    "gpt-6-sol": "GPT 6 Sol",
    "gpt-5.6-sol": "GPT 5.6 Sol",
    "deepseek-v4.1-flash": "Deepseek v4.1 Flash",
    "deepseek-v4.1-flash-1017": "Deepseek v4.1 Flash 1017",
    "deepseek-v4-pro": "Deepseek v4 Pro",
    "gemini-4-argon": "Gemini 4 Argon",
    "gemini-3.8-flash": "Gemini 3.8 Flash",
    "grok-4.6": "Grok 4.6",
    "glm-5.3-flash": "GLM 5.3 Flash",
}


def display_model_name(model: str, provider: Provider) -> str:
    norm = normalize_model(model)
    base = MODEL_DISPLAY.get(norm)
    if not base:
        base = norm.replace("-", " ").replace("_", " ").title()
    suffix = provider.alias.upper() if provider.alias else ""
    if suffix and suffix not in base.upper().split():
        base = f"{base} {suffix}"
    return base


# Rename providers here if the local registry uses different display names.
PROVIDER_DISPLAY: dict[str, str] = {}


def display_provider_name(provider: Provider) -> str:
    return PROVIDER_DISPLAY.get(provider.name, provider.name)


def _read_settings(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _write_settings_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".map-exp.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def cc_switch_snapshot(db_path: Path, settings_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT id FROM providers WHERE app_type='codex' AND is_current=1 LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    settings = _read_settings(settings_path)
    return {
        "provider_id": row[0] if row else None,
        "settings_current_provider": settings.get("currentProviderCodex"),
        "captured_at": now_iso(),
    }


def switch_cc_provider(provider: Provider, db_path: Path, settings_path: Path) -> None:
    """Select a provider in CC Switch's SQLite database and settings file.

    This is only used in proxy mode.  The caller must snapshot and restore the
    previous selection in a finally block.
    """
    conn = sqlite3.connect(db_path, timeout=20)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("UPDATE providers SET is_current=0 WHERE app_type='codex'")
        conn.execute(
            "UPDATE providers SET is_current=1 WHERE id=? AND app_type='codex'",
            (provider.id,),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    settings = _read_settings(settings_path)
    settings["currentProviderCodex"] = provider.id
    _write_settings_atomic(settings_path, settings)
    time.sleep(0.6)


def restore_cc_provider(snapshot: dict[str, Any], db_path: Path, settings_path: Path) -> None:
    provider_id = snapshot.get("provider_id")
    if not provider_id:
        return
    conn = sqlite3.connect(db_path, timeout=20)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("UPDATE providers SET is_current=0 WHERE app_type='codex'")
        conn.execute(
            "UPDATE providers SET is_current=1 WHERE id=? AND app_type='codex'",
            (provider_id,),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    settings = _read_settings(settings_path)
    settings["currentProviderCodex"] = provider_id
    _write_settings_atomic(settings_path, settings)
    time.sleep(0.4)


def pending_state_path(archive_root: Path) -> Path:
    return archive_root / "_cc-switch-state.json"


def write_pending_state(archive_root: Path, snapshot: dict[str, Any]) -> None:
    write_json(pending_state_path(archive_root), snapshot)


def clear_pending_state(archive_root: Path) -> None:
    p = pending_state_path(archive_root)
    if p.exists():
        p.unlink()


def restore_pending_state(archive_root: Path, db_path: Path, settings_path: Path) -> None:
    p = pending_state_path(archive_root)
    if not p.exists():
        return
    try:
        snapshot = read_json(p)
        restore_cc_provider(snapshot, db_path, settings_path)
        clear_pending_state(archive_root)
        print(f"[cc-switch] restored provider {snapshot.get('provider_id')} from interrupted run")
    except Exception as exc:
        print(f"[warn] could not restore interrupted CC Switch state: {exc}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Run preparation and execution
# ---------------------------------------------------------------------------

def _toml_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _catalog_entry(template: dict[str, Any], model: str) -> dict[str, Any]:
    item = json.loads(json.dumps(template))
    item["slug"] = model
    item["display_name"] = MODEL_DISPLAY.get(normalize_model(model), model)
    item["description"] = item.get("description") or item["display_name"]
    item["priority"] = int(item.get("priority", 1000))
    item.setdefault("context_window", 1_000_000)
    item.setdefault("max_context_window", 1_000_000)
    item.setdefault("effective_context_window_percent", 95)
    item.setdefault("default_reasoning_level", "high")
    item.setdefault("default_reasoning_summary", "none")
    item.setdefault("supported_reasoning_levels", [
        {"description": "Greater reasoning depth for complex problems", "effort": "high"},
    ])
    item.setdefault("input_modalities", ["text", "image"])
    item.setdefault("supported_in_api", True)
    item.setdefault("supports_reasoning_summaries", True)
    item.setdefault("visibility", "list")
    return item


def prepare_model_catalog(provider: Provider, model: str, home: Path, default_catalog: Path) -> Path:
    catalog: dict[str, Any] = {"models": []}
    if default_catalog.exists():
        try:
            raw = read_json(default_catalog)
            if isinstance(raw, dict) and isinstance(raw.get("models"), list):
                catalog = {"models": [x for x in raw["models"] if isinstance(x, dict) and x.get("slug")]}
        except Exception:
            pass
    if not catalog["models"]:
        template = {
            "slug": model,
            "display_name": model,
            "description": model,
            "context_window": 1_000_000,
            "max_context_window": 1_000_000,
            "effective_context_window_percent": 95,
            "default_reasoning_level": "high",
            "default_reasoning_summary": "none",
            "supported_reasoning_levels": [
                {"description": "Greater reasoning depth for complex problems", "effort": "high"},
            ],
            "input_modalities": ["text", "image"],
            "supported_in_api": True,
            "supports_reasoning_summaries": True,
            "visibility": "list",
            "priority": 1000,
        }
        catalog = {"models": [template]}
    existing = {str(x.get("slug")) for x in catalog["models"]}
    if model not in existing:
        catalog["models"].append(_catalog_entry(catalog["models"][0], model))
    path = home / "model_catalog.json"
    write_json(path, catalog)
    return path


def _codex_base_url(base_url: str) -> str:
    """Codex appends /responses, so direct provider roots need /v1."""
    value = (base_url or "").rstrip("/")
    if not value:
        return value
    parsed_path = value.split("://", 1)[-1]
    path = "/" + parsed_path.split("/", 1)[1] if "/" in parsed_path else ""
    if path in ("", "/"):
        return value + "/v1"
    return value


def prepare_codex_home(
    provider: Provider,
    model: str,
    run_dir: Path,
    mode: str,
    default_catalog: Path,
) -> Path:
    home = run_dir / ".codex-home"
    home.mkdir(parents=True, exist_ok=True)
    if mode == "proxy":
        auth = {"OPENAI_API_KEY": "PROXY_MANAGED"}
        base_url = DEFAULT_PROXY
        requires = False
        token = "PROXY_MANAGED"
    else:
        auth = dict(provider.auth or {})
        if provider.token and not auth.get("OPENAI_API_KEY"):
            auth["OPENAI_API_KEY"] = provider.token
        base_url = _codex_base_url(provider.base_url)
        requires = provider.requires_openai_auth
        token = provider.token
    (home / "auth.json").write_text(json.dumps(auth, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    catalog_path = prepare_model_catalog(provider, model, home, default_catalog)
    provider_name = provider.name or "CCSwitchProvider"
    lines = [
        'model_provider = "custom"',
        f"model = {_toml_string(model)}",
        'model_reasoning_effort = "high"',
        f"model_catalog_json = {_toml_string(str(catalog_path))}",
        "",
        "[model_providers.custom]",
        f"name = {_toml_string(provider_name)}",
        f"base_url = {_toml_string(base_url)}",
        'wire_api = "responses"',
        f"requires_openai_auth = {'true' if requires else 'false'}",
    ]
    if token:
        lines.append(f"experimental_bearer_token = {_toml_string(token)}")
    lines.extend([
        "",
        "[features]",
        "plugins = false",
        "remote_plugin = false",
        "apps = false",
        "",
        "[sandbox_workspace_write]",
        "network_access = false",
        "",
        "[windows]",
        'sandbox = "elevated"',
        "",
    ])
    (home / "config.toml").write_text("\n".join(lines), encoding="utf-8")
    return home


def delivery_block(task: Task, run_dir: Path, backend: str = "codex") -> str:
    if backend == "api":
        return f"""

---
本次自动评测不授予工具写权限，也不会让你执行 shell。请不要尝试创建文件或调用工具。
请直接在你的最终回答中给出完整、可独立运行的 Python 生成脚本（用 ```python 代码块包裹）。脚本必须：
- 自动生成 {task.rows} 行 × {task.cols} 列、仅含 0/1 的矩阵；
- 写出 outputs/candidate.csv（无表头）；
- 写出 outputs/candidate.png（{task.cols}×{task.rows}，白=陆地、黑=水域）；
- 写出 outputs/candidate_8x.png（最近邻放大 8 倍，{task.upscale[0]}×{task.upscale[1]}）；
- 仅使用 Python 标准库、numpy、Pillow，不得联网、读取外部地图或标准答案；
- 脚本应在当前工作目录下运行，并自行创建 outputs/ 目录。
外层评测器会保存并运行你给出的脚本，然后按 1 错格 = 1 分评分。
"""
    return f"""

---
本次自动运行的交付目录（当前工作目录）：{run_dir}
请在当前目录创建 outputs/ 子目录，并将以下文件明确保存为：
- outputs/candidate.csv ：{task.rows} 行 × {task.cols} 列，仅含 0/1，无表头；
- outputs/candidate.png ：{task.cols}×{task.rows} 像素，白=陆地、黑=水域；
- outputs/candidate_8x.png ：最近邻放大 8 倍，{task.upscale[0]}×{task.upscale[1]} 像素；
- outputs/generate_map.py ：生成上述矩阵与图片的可运行 Python 脚本。
不要把标准答案、模板或任何外部地图数据复制进本目录；不要读取本目录之外的地图文件。
完成后请直接运行脚本并检查上述四个文件确实存在；一旦验证通过就立即结束，不要继续查看图片、美化结果或做额外分析。
如果当前沙箱阻止写入文件，请不要只回复“无法完成”：请在最终消息中给出完整、可独立运行的 Python 生成脚本（用 ```python 代码块包裹），外层自动脚本会代你运行它。若连脚本也无法给出，再给出完整的 0/1 矩阵代码块。
"""


def make_run_id(spec: RunSpec) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    digest = sha256_bytes(
        f"{spec.provider.id}|{spec.model}|{spec.task.key}|{spec.group}|{stamp}".encode("utf-8")
    )[:8]
    safe_model = normalize_model(spec.model).replace("/", "-")
    return f"{stamp}_{safe_model}_{spec.task.key}_g{spec.group}_{digest}"


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            proc.kill()
        proc.wait(timeout=20)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _python_env(python_exe: Path) -> dict[str, str]:
    env = dict(os.environ)
    python_dir = str(python_exe.parent)
    env["PATH"] = python_dir + os.pathsep + env.get("PATH", "")
    # The desktop app exports its own managed permission profile.  A child
    # Codex process must not inherit that profile, otherwise --sandbox is
    # overridden and every write is rejected as "blocked by policy".
    for key in list(env):
        if key.startswith("CODEX_"):
            env.pop(key, None)
    return env


def run_codex(
    spec: RunSpec,
    codex_exe: str,
    python_exe: Path,
    timeout_min: int,
    default_catalog: Path,
    keep_work: bool,
    archive_root: Path,
) -> dict[str, Any]:
    run_id = make_run_id(spec)
    run_dir = (archive_root / "_runs" / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs").mkdir(exist_ok=True)
    prompt = spec.task.prompt_path.read_text(encoding="utf-8-sig").rstrip() + delivery_block(spec.task, run_dir, "codex")
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    home = prepare_codex_home(spec.provider, spec.model, run_dir, spec.mode, default_catalog)
    env = _python_env(python_exe)
    env["CODEX_HOME"] = str(home)
    last_message = run_dir / "last_message.txt"
    log_path = run_dir / "codex.log"
    command = [codex_exe]
    if _codex_supports_no_daemon(codex_exe):
        command.append("--no-daemon")
    command += [
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "workspace-write",
        "-C",
        str(run_dir),
        "--add-dir",
        str(run_dir),
        "-m",
        spec.model,
        "--json",
        "-o",
        str(last_message),
        prompt,
    ]
    (run_dir / "command.txt").write_text(
        " ".join(
            _toml_string(x) if (" " in x or "\n" in x) else x
            for x in command[:-1]
        ) + "\n",
        encoding="utf-8",
    )
    started = time.time()
    timed_out = False
    early_complete = False
    returncode: int | None = None
    error = ""
    expected = [
        run_dir / "outputs" / "candidate.csv",
        run_dir / "outputs" / "candidate.png",
        run_dir / "outputs" / "candidate_8x.png",
    ]
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            proc = subprocess.Popen(
                command,
                cwd=str(run_dir),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            deadline = started + max(1, timeout_min) * 60
            stable_signature: tuple[int, ...] | None = None
            stable_since = 0.0
            while True:
                if proc.poll() is not None:
                    returncode = proc.returncode
                    break
                if all(p.is_file() and p.stat().st_size > 0 for p in expected):
                    signature = tuple(p.stat().st_size for p in expected)
                    if signature == stable_signature:
                        if time.time() - stable_since >= 8.0:
                            early_complete = True
                            returncode = 0
                            _terminate_process_tree(proc)
                            break
                    else:
                        stable_signature = signature
                        stable_since = time.time()
                else:
                    stable_signature = None
                if time.time() >= deadline:
                    timed_out = True
                    _terminate_process_tree(proc)
                    returncode = proc.returncode if proc.returncode is not None else -1
                    error = f"timeout after {timeout_min} min"
                    break
                time.sleep(2)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    duration = time.time() - started
    truth = load_truth(spec.task)
    candidate = find_candidate(run_dir, spec.task)
    recovery: dict[str, Any] | None = None
    if candidate is None:
        try:
            recovery = recover_artifacts(run_dir, spec.task, python_exe)
            candidate = find_candidate(run_dir, spec.task)
        except Exception as exc:
            recovery = {"attempted": True, "error": f"{type(exc).__name__}: {exc}"}
    score: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    quality: dict[str, Any] | None = None
    if candidate is not None:
        score = score_matrix(candidate.matrix, truth)
        candidate.meta["orientation_ok"] = bool(score.get("orientation_ok", False))
        validation = validate_candidate(candidate, spec.task, truth, score)
        try:
            quality = map_quality_metrics(candidate, truth)
        except Exception as exc:
            quality = {"error": f"{type(exc).__name__}: {exc}"}
    else:
        if not error and returncode not in (None, 0):
            error = f"codex exited with code {returncode}"
        error = error or "no candidate CSV/PNG found"

    status = "ok" if (
        returncode == 0
        and not timed_out
        and candidate is not None
        and validation is not None
        and validation.get("status") == "ok"
    ) else "partial"
    if candidate is None:
        status = "failed"

    result: dict[str, Any] = {
        "run_id": run_id,
        "started_at": datetime.fromtimestamp(started).astimezone().isoformat(timespec="seconds"),
        "finished_at": now_iso(),
        "duration_sec": round(duration, 1),
        "status": status,
        "returncode": returncode,
        "timed_out": timed_out,
        "early_complete": early_complete,
        "error": error,
        "provider_id": spec.provider.id,
        "provider_name": spec.provider.name,
        "provider_display": display_provider_name(spec.provider),
        "provider_alias": spec.provider.alias,
        "provider_mode": spec.mode,
        "model": spec.model,
        "model_display": display_model_name(spec.model, spec.provider),
        "task": spec.task.key,
        "task_label": spec.task.label,
        "group": spec.group,
        "archive_stem": spec.archive_stem,
        "prompt_sha256": sha256_file(spec.task.prompt_path),
        "run_dir": str(run_dir),
        "candidate": None,
        "score": None,
        "validation": validation,
        "map_metrics": quality,
        "recovery": recovery,
    }
    if candidate is not None:
        result["candidate"] = {
            "kind": candidate.kind,
            "how": candidate.how,
            "source": str(candidate.source) if candidate.source else None,
            "csv": str(candidate.csv_path) if candidate.csv_path else None,
            "upscale": str(candidate.upscale_path) if candidate.upscale_path else None,
            "csv_match": bool(candidate.csv_match),
            "upscale_match": bool(candidate.upscale_match),
            "sha256": sha256_file(candidate.source) if candidate.source else None,
            "meta": candidate.meta,
        }
        result["score"] = {k: v for k, v in score.items() if k != "diff"} if score else None
    write_json(run_dir / "result.json", result)
    return result

# ---------------------------------------------------------------------------
# Per-map diagnostic metrics
# ---------------------------------------------------------------------------

_QUANT_MODULE: Any = None


def _quant_module() -> Any:
    """Import the diagnostic module statically so Nuitka can compile it."""
    global _QUANT_MODULE
    if _QUANT_MODULE is None:
        from quant import quantify as module

        _QUANT_MODULE = module
    return _QUANT_MODULE


def map_quality_metrics(cand: Candidate, truth: np.ndarray) -> dict[str, Any]:
    """Per-map diagnostic metrics for every task.

    Accuracy and HSS are cell-classification metrics. Boundary F1, narrow-water
    retention and island IoU are structural metrics, so a map that paints large
    blocks of the right class cannot hide poor coastlines behind a high accuracy.
    """
    q = _quant_module()
    P = cand.matrix.astype(np.uint8)
    R = truth.astype(np.uint8)
    cfg = q.CONFIG
    cstats = q.component_stats(
        P, R, int(cfg["component_min_cells"]), float(cfg["component_iou_match"])
    )
    bf1 = {
        int(d): float(q.boundary_f1(P, R, int(d))[2])
        for d in cfg["boundary_tolerances"]
    }
    nw_keep, nw_tot = q.narrow_water_preservation(
        P, R, int(cfg["narrow_water_radius"])
    )
    return {
        "accuracy_pct": round(100.0 * float(q.accuracy(P, R)), 6),
        "hss": round(float(q.hss(P, R)), 6),
        "boundary_f1_d0": round(bf1.get(0, 0.0), 6),
        "boundary_f1_d1": round(bf1.get(1, 0.0), 6),
        "boundary_f1_d2": round(bf1.get(2, 0.0), 6),
        "narrow_water_keep": round(float(nw_keep), 6) if nw_keep is not None else None,
        "narrow_water_cells": int(nw_tot),
        "island_iou_median": round(float(cstats["island_iou_median"]), 6),
        "island_iou_area_weighted": round(float(cstats["island_iou_area_weighted"]), 6),
        "island_iou_ge50": round(float(cstats["island_iou_ge50"]), 6),
        "component_recall": round(float(cstats["component_recall"]), 6),
        "n_ref_components": int(cstats["n_ref_components"]),
        "n_pred_components": int(cstats["n_pred_components"]),
        "config": {
            "component_min_cells": cfg["component_min_cells"],
            "component_iou_match": cfg["component_iou_match"],
            "boundary_tolerances": cfg["boundary_tolerances"],
            "narrow_water_radius": cfg["narrow_water_radius"],
        },
    }


# ---------------------------------------------------------------------------
# Archiving
# ---------------------------------------------------------------------------

def _clean_old_archive(archive_dir: Path, stem: str) -> None:
    exact = [
        archive_dir / f"{stem}.png",
        archive_dir / f"{stem}.csv",
        archive_dir / f"{stem}.log",
        archive_dir / f"{stem}.meta.json",
        archive_dir / f"{stem}_8x.png",
        archive_dir / f"{stem}_diff.png",
        archive_dir / f"{stem}_script.py",
        archive_dir / f"{stem}_last_message.txt",
        archive_dir / f"{stem}_prompt.txt",
        archive_dir / f"{stem}_retry_prompt.txt",
        archive_dir / f"{stem}_response.json",
        archive_dir / f"{stem}_response_error.txt",
    ]
    for path in exact:
        if path.is_file():
            path.unlink()
    for path in archive_dir.glob(f"{stem}_diff_x*.png"):
        if path.is_file():
            path.unlink()
    script_dir = archive_dir / f"{stem}_scripts"
    if script_dir.is_dir():
        shutil.rmtree(script_dir, ignore_errors=True)


def unique_archive_stem(archive_dir: Path, stem: str, overwrite: bool) -> str:
    if overwrite or not any(archive_dir.glob(f"{stem}.*")):
        return stem
    index = 2
    while any(archive_dir.glob(f"{stem}_r{index}.*")):
        index += 1
    return f"{stem}_r{index}"


def _copy_if_exists(src: Path | None, dst: Path) -> str | None:
    if src is None or not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def _find_scripts(run_dir: Path) -> list[Path]:
    scripts: list[Path] = []
    for path in run_dir.rglob("*.py"):
        if _is_internal_path(path):
            continue
        scripts.append(path)
    return sorted(scripts, key=lambda p: p.stat().st_size, reverse=True)


def archive_result(
    result: dict[str, Any],
    spec: RunSpec,
    candidate: Candidate | None,
    truth: np.ndarray,
    archive_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    archive_dir = spec.archive_dir
    archive_dir.mkdir(parents=True, exist_ok=True)
    stem = unique_archive_stem(archive_dir, spec.archive_stem, overwrite)
    if overwrite:
        _clean_old_archive(archive_dir, stem)
    files: dict[str, str | None] = {}

    if candidate is not None:
        if candidate.source is not None:
            files["png"] = _copy_if_exists(candidate.source, archive_dir / f"{stem}.png")
        if candidate.csv_path is not None:
            files["csv"] = _copy_if_exists(candidate.csv_path, archive_dir / f"{stem}.csv")
        if candidate.upscale_path is not None:
            files["upscale"] = _copy_if_exists(candidate.upscale_path, archive_dir / f"{stem}_8x.png")
        score = result.get("score") or {}
        if score:
            diff_path = archive_dir / f"{stem}_diff.png"
            # Recompute diff because result.json intentionally excludes the array.
            diff = candidate.matrix != truth
            save_diff(diff, candidate.matrix, diff_path)
            files["diff"] = str(diff_path)

    run_dir = Path(result["run_dir"])
    scripts = _find_scripts(run_dir)
    if scripts:
        files["script"] = _copy_if_exists(scripts[0], archive_dir / f"{stem}_script.py")
        script_dir = archive_dir / f"{stem}_scripts"
        script_dir.mkdir(exist_ok=True)
        copied = []
        for path in scripts:
            rel = path.relative_to(run_dir)
            dst = script_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
            copied.append(str(dst))
        files["scripts"] = copied

    last = run_dir / "last_message.txt"
    if last.exists():
        files["last_message"] = _copy_if_exists(last, archive_dir / f"{stem}_last_message.txt")
    log = run_dir / "codex.log"
    if log.exists():
        files["log"] = _copy_if_exists(log, archive_dir / f"{stem}.log")
    response = run_dir / "response.json"
    if response.exists():
        files["response"] = _copy_if_exists(response, archive_dir / f"{stem}_response.json")
    response_error = run_dir / "response_error.txt"
    if response_error.exists():
        files["response_error"] = _copy_if_exists(response_error, archive_dir / f"{stem}_response_error.txt")
    prompt = run_dir / "prompt.txt"
    if prompt.exists():
        files["prompt"] = _copy_if_exists(prompt, archive_dir / f"{stem}_prompt.txt")
    retry_prompt = run_dir / "retry_prompt.txt"
    if retry_prompt.exists():
        files["retry_prompt"] = _copy_if_exists(retry_prompt, archive_dir / f"{stem}_retry_prompt.txt")

    result["archive"] = {
        "stem": stem,
        "dir": str(archive_dir),
        "files": files,
        "archived_at": now_iso(),
    }
    result["archive_stem"] = stem
    meta_path = archive_dir / f"{stem}.meta.json"
    write_json(meta_path, result)
    files["meta"] = str(meta_path)
    result["archive"]["files"] = files
    write_json(meta_path, result)
    return result

# ---------------------------------------------------------------------------
# Summary tables and Excel export
# ---------------------------------------------------------------------------

# Summary task order. Arctic 0.2 is intentionally retired and excluded.
TASK_ORDER = ["global", "csc", "dzh", "seto", "arctic01", "hecate"]
TASK_HEADERS = {
    "global": ("全球得分", "全球得分率"),
    "csc": ("南海得分", "南海得分率"),
    "dzh": ("地中海得分", "地中海得分率"),
    "seto": ("濑户内海得分", "濑户内海得分率"),
    "arctic01": ("北极航道0.1°得分", "北极航道0.1°得分率"),
    "hecate": ("赫卡特海峡得分", "赫卡特海峡得分率"),
}
# The original four columns keep their historical order; Arctic 0.1 and
# Hecate Strait are appended before the aggregate columns.
SUMMARY_HEADERS = [
    "模型名", "来源渠道",
    "全球得分", "全球得分率", "最大全球错误率",
    "南海得分", "南海得分率",
    "地中海得分", "地中海得分率",
    "濑户内海得分", "濑户内海得分率",
    "北极航道0.1°得分", "北极航道0.1°得分率",
    "赫卡特海峡得分", "赫卡特海峡得分率",
    "平均分", "平均得分率", "测试次数",
]


def load_archive_records(archive_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(archive_root.rglob("*.meta.json")):
        if "_runs" in {p.lower() for p in path.parts}:
            continue
        try:
            rec = read_json(path)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        rec["_meta_path"] = str(path)
        records.append(rec)
    return records


def _record_sort_key(rec: dict[str, Any]) -> tuple:
    return (
        str(rec.get("provider_name", "")),
        str(rec.get("model_display", "")),
        TASK_ORDER.index(rec["task"]) if rec.get("task") in TASK_ORDER else 99,
        int(rec.get("group", 1)),
        str(rec.get("started_at", "")),
    )


def _scored(rec: dict[str, Any]) -> bool:
    score = rec.get("score")
    return bool(rec.get("status") == "ok" and isinstance(score, dict) and score.get("normalized") is not None)


def _summary_scored(rec: dict[str, Any]) -> bool:
    """Scored records that belong in the current summary task set."""
    return _scored(rec) and rec.get("task") in TASK_ORDER


def _summary_cell_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _summary_task_columns(headers: list[Any]) -> dict[int, str]:
    header_to_task = {TASK_HEADERS[key][0]: key for key in TASK_ORDER}
    out: dict[int, str] = {}
    for index, value in enumerate(headers):
        task = header_to_task.get(str(value))
        if task:
            out[index] = task
    return out


def _record_for_summary_cell(
    records: list[dict[str, Any]],
    model_display: str,
    provider_display: str,
    task_key: str,
    group: int,
) -> dict[str, Any] | None:
    """Find the latest failed archive attempt behind one summary cell."""
    matches: list[dict[str, Any]] = []
    for rec in records:
        if str(rec.get("task")) != task_key or _scored(rec):
            continue
        try:
            rec_group = int(rec.get("group", 1))
        except (TypeError, ValueError):
            continue
        if rec_group != group:
            continue
        rec_model = str(rec.get("model_display") or rec.get("model") or "")
        rec_provider = str(rec.get("provider_display") or rec.get("provider_name") or "")
        if rec_model == model_display and rec_provider == provider_display:
            matches.append(rec)
    if not matches:
        return None
    return max(matches, key=lambda rec: str(rec.get("finished_at") or rec.get("started_at") or ""))


def _target_from_record(rec: dict[str, Any], source: str) -> dict[str, Any]:
    try:
        group = int(rec.get("group", 1))
    except (TypeError, ValueError):
        group = 1
    return {
        "provider_id": str(rec.get("provider_id") or ""),
        "provider_name": str(rec.get("provider_name") or ""),
        "provider_display": str(rec.get("provider_display") or rec.get("provider_name") or ""),
        "model": str(rec.get("model") or ""),
        "model_display": str(rec.get("model_display") or rec.get("model") or ""),
        "task": str(rec.get("task") or ""),
        "group": group,
        "source": source,
    }


def _target_key(target: dict[str, Any]) -> tuple[str, str, str, int]:
    provider = target.get("provider_id") or target.get("provider_name") or target.get("provider_display") or ""
    return (
        str(provider),
        str(target.get("model") or target.get("model_display") or ""),
        str(target.get("task") or ""),
        int(target.get("group", 1)),
    )


def _record_target_key(rec: dict[str, Any]) -> tuple[str, str, str, int]:
    provider = rec.get("provider_id") or rec.get("provider_name") or rec.get("provider_display") or ""
    return (
        str(provider),
        str(rec.get("model") or rec.get("model_display") or ""),
        str(rec.get("task") or ""),
        int(_target_from_record(rec, "key")["group"]),
    )


def _record_matches_provider(rec: dict[str, Any], provider: Provider) -> bool:
    return (
        str(rec.get("provider_id") or "") == provider.id
        or str(rec.get("provider_name") or "") == provider.name
    )


def _provider_for_record(rec: dict[str, Any], providers: list[Provider]) -> Provider | None:
    rec_id = str(rec.get("provider_id") or "")
    rec_name = str(rec.get("provider_name") or "")
    rec_display = str(rec.get("provider_display") or "")
    for provider in providers:
        if rec_id and provider.id == rec_id:
            return provider
    for provider in providers:
        if rec_name and provider.name == rec_name:
            return provider
    for provider in providers:
        if rec_display and display_provider_name(provider) == rec_display:
            return provider
    return None


def _selector_values(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items or (len(items) == 1 and items[0].lower() == "all"):
        return None
    return items


def _record_matches_model(rec: dict[str, Any], model_selector: str | None) -> bool:
    wanted = _selector_values(model_selector)
    if wanted is None:
        return True
    allowed = {normalize_model(item) for item in wanted} | set(wanted)
    model = str(rec.get("model") or "")
    display = str(rec.get("model_display") or "")
    return model in allowed or normalize_model(model) in allowed or display in allowed


def load_fulfillment_targets(
    archive_root: Path,
    records: list[dict[str, Any]],
    tasks: dict[str, Task],
    model_selector: str | None = None,
    excel_name: str = "summary.xlsx",
) -> list[dict[str, Any]]:
    """Return failed cells in the selected summary workbook that can be retried.

    The workbook is the source of truth for which map cells are incomplete.
    ``records`` supplies the exact model slug/provider identity and proves
    that the cell was attempted (so a never-run map is not scheduled).
    """
    summary_path = excel_report_path(archive_root, excel_name)
    if not summary_path.exists():
        raise FileNotFoundError(
            f"summary workbook not found: {summary_path}. Run --report-only first."
        )
    try:
        from openpyxl import load_workbook
        wb = load_workbook(summary_path, data_only=True, read_only=True)
    except Exception as exc:
        raise RuntimeError(f"cannot read {summary_path}: {exc}") from exc

    found: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    summary_rows: set[tuple[str, str, int]] = set()
    summary_counts: dict[tuple[str, str], int] = {}
    known_sheets = {"总览", "明细", "地图指标", "地图指标明细"}
    try:
        # 总览's 测试次数 is the current experiment range.  It is used only
        # when a task has no successful group in its per-model sheet (e.g. a
        # group where every map failed, so the group row was never emitted).
        if "总览" in wb.sheetnames:
            overview_rows = wb["总览"].iter_rows(values_only=True)
            try:
                overview_headers = list(next(overview_rows))
            except StopIteration:
                overview_headers = []
            try:
                overview_model_col = overview_headers.index("模型名")
                overview_provider_col = overview_headers.index("来源渠道")
                overview_count_col = overview_headers.index("测试次数")
            except ValueError:
                overview_model_col = overview_provider_col = overview_count_col = -1
            if overview_count_col >= 0:
                for overview_row in overview_rows:
                    if not overview_row or overview_count_col >= len(overview_row):
                        continue
                    try:
                        count = int(overview_row[overview_count_col])
                    except (TypeError, ValueError):
                        continue
                    model_name = str(
                        overview_row[overview_model_col]
                        if overview_model_col < len(overview_row) else ""
                    )
                    provider_name = str(
                        overview_row[overview_provider_col]
                        if overview_provider_col < len(overview_row) else ""
                    )
                    if model_name:
                        summary_counts[(model_name, provider_name)] = max(
                            count, summary_counts.get((model_name, provider_name), 0)
                        )

        for ws in wb.worksheets:
            if ws.title in known_sheets:
                continue
            rows = ws.iter_rows(values_only=True)
            try:
                headers = list(next(rows))
            except StopIteration:
                continue
            try:
                model_col = headers.index("模型名")
                provider_col = headers.index("来源渠道")
                group_col = headers.index("测试次数")
            except ValueError:
                continue
            task_cols = _summary_task_columns(headers)
            if not task_cols:
                continue
            for row in rows:
                if not row or group_col >= len(row) or _summary_cell_blank(row[group_col]):
                    continue
                try:
                    group = int(row[group_col])
                except (TypeError, ValueError):
                    continue
                model_display = str(row[model_col] if model_col < len(row) else "")
                provider_display = str(row[provider_col] if provider_col < len(row) else "")
                summary_rows.add((model_display, provider_display, group))
                for col, task_key in task_cols.items():
                    value = row[col] if col < len(row) else None
                    if not _summary_cell_blank(value):
                        continue
                    if task_key not in tasks:
                        continue
                    # Only retry cells that were actually attempted: a blank
                    # cell with no failed meta is a map this model has never
                    # run, which belongs to a normal run or --extra-group.
                    rec = _record_for_summary_cell(
                        records, model_display, provider_display, task_key, group
                    )
                    if rec is None or not _record_matches_model(rec, model_selector):
                        continue
                    target = _target_from_record(rec, "summary.xlsx")
                    target["task"] = task_key
                    target["group"] = group
                    found[_target_key(target)] = target

        # If every map in a group failed, that group has no row in its model
        # sheet.  Include those failed archive attempts only when no successful
        # result exists for the exact same cell.
        successes = {_record_target_key(rec) for rec in records if _scored(rec)}
        successes_by_task: dict[tuple[str, str, str], set[int]] = {}
        for rec in records:
            if not _scored(rec):
                continue
            key3 = (
                str(rec.get("provider_id") or rec.get("provider_name") or rec.get("provider_display") or ""),
                str(rec.get("model") or rec.get("model_display") or ""),
                str(rec.get("task") or ""),
            )
            try:
                group = int(rec.get("group", 1))
            except (TypeError, ValueError):
                continue
            successes_by_task.setdefault(key3, set()).add(group)
        table_max: dict[tuple[str, str, str], int] = {}
        for key3, groups in successes_by_task.items():
            if groups:
                table_max[key3] = max(groups)
        for rec in records:
            task_key = str(rec.get("task") or "")
            if task_key not in tasks or _scored(rec):
                continue
            if not _record_matches_model(rec, model_selector):
                continue
            key = _record_target_key(rec)
            if key in successes:
                continue
            model_display = str(rec.get("model_display") or rec.get("model") or "")
            provider_display = str(rec.get("provider_display") or rec.get("provider_name") or "")
            if (model_display, provider_display, key[3]) in summary_rows:
                # A workbook row exists, so its blank task cells were already
                # enumerated above.  Do not re-add stale failures for cells
                # that the workbook currently marks as successful.
                continue
            key3 = key[:3]
            # Ignore stale failures beyond the current experiment range.
            max_group = table_max.get(key3)
            summary_count = summary_counts.get((model_display, provider_display))
            group_limit = max_group if max_group is not None else summary_count
            if group_limit is not None and key[3] > group_limit:
                continue
            target = _target_from_record(rec, "meta")
            found.setdefault(key, target)
    finally:
        wb.close()

    return sorted(found.values(), key=lambda t: (
        str(t.get("provider_name") or t.get("provider_display") or ""),
        str(t.get("model") or ""),
        TASK_ORDER.index(t["task"]) if t.get("task") in TASK_ORDER else 99,
        int(t.get("group", 1)),
    ))


def filter_fulfillment_targets(
    targets: list[dict[str, Any]],
    providers: list[Provider],
    task_selector: str,
) -> list[dict[str, Any]]:
    """Resolve each target to a live provider and apply the task selector."""
    selected_tasks = (
        set(TASK_ORDER) if task_selector.strip().lower() == "all"
        else {x.strip() for x in task_selector.split(",") if x.strip()}
    )
    out: list[dict[str, Any]] = []
    unresolved = 0
    for target in targets:
        if str(target.get("task") or "") not in selected_tasks:
            continue
        provider = _provider_for_record(target, providers)
        if provider is None:
            unresolved += 1
            continue
        item = dict(target)
        item["provider_id"] = provider.id
        item["provider_name"] = provider.name
        item["provider_display"] = display_provider_name(provider)
        out.append(item)
    if unresolved:
        print(
            f"[fulfill] skipped {unresolved} targets whose provider is not configured in CC Switch",
            file=sys.stderr,
        )
    return out


def build_fulfillment_plan(
    providers: list[Provider],
    tasks: dict[str, Task],
    targets: list[dict[str, Any]],
    archive_root: Path,
    backend: str,
    api_transport: str,
    provider_mode: str = "auto",
    allow_switch: bool = True,
) -> list[RunSpec]:
    """Build one retry spec per missing (provider, model, task, group) cell."""
    proxy_ok, _proxy_msg = proxy_health(DEFAULT_PROXY)
    plan: list[RunSpec] = []
    for target in targets:
        provider = _provider_for_record(target, providers)
        task = tasks.get(str(target.get("task") or ""))
        model = str(target.get("model") or "")
        if provider is None or task is None or not model:
            continue
        try:
            group = int(target.get("group", 1))
        except (TypeError, ValueError):
            continue
        if backend in ("api", "api-tools"):
            mode, reason = resolve_api_transport(provider, api_transport, proxy_ok)
        else:
            mode, reason = choose_mode(provider, provider_mode, proxy_ok, allow_switch)
        model_norm = normalize_model(model)
        provider_suffix = f"@{provider.alias}" if provider.alias else ""
        group_suffix = "" if group == 1 else f"#{group}"
        stem = f"{model_norm}{task.archive_suffix}{provider_suffix}{group_suffix}"
        plan.append(RunSpec(
            provider=provider,
            model=model,
            task=task,
            group=group,
            archive_stem=stem,
            archive_dir=archive_root / model_norm,
            mode=mode,
            skip_reason=reason,
        ))
    return plan



def _group_stats(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one (model, provider, group) or all groups of a model."""
    task_scores: dict[str, float] = {}
    task_rates: dict[str, float] = {}
    global_rates: list[float] = []
    for task in TASK_ORDER:
        sub = [r for r in recs if r.get("task") == task and _scored(r)]
        if not sub:
            continue
        scores = [float(r["score"]["normalized"]) for r in sub]
        rates = [float(r["score"]["accuracy_pct"]) / 100.0 for r in sub]
        task_scores[task] = float(np.mean(scores))
        task_rates[task] = float(np.mean(rates))
        if task == "global":
            global_rates.extend(rates)
    if not task_scores:
        return {}
    return {
        "task_scores": task_scores,
        "task_rates": task_rates,
        "max_global_error": (1.0 - min(global_rates)) if global_rates else None,
        "average_score": float(np.mean(list(task_scores.values()))),
        "average_rate": float(np.mean(list(task_rates.values()))),
        "completed_tasks": len(task_scores),
    }


def aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for rec in records:
        key = (
            str(rec.get("model_display", rec.get("model", "?"))),
            str(rec.get("provider_display") or rec.get("provider_name", "?")),
            str(rec.get("model", "?")),
            str(rec.get("provider_alias", "")),
        )
        groups.setdefault(key, []).append(rec)

    aggregates: list[dict[str, Any]] = []
    for (display, provider_name, model, alias), recs in groups.items():
        stats = _group_stats(recs)
        if not stats:
            continue
        ok_groups = sorted({int(r.get("group", 1)) for r in recs if _scored(r)})
        # Per-group rows, mirroring the template's "<model>的结果" sheet.
        group_rows: list[dict[str, Any]] = []
        for group in ok_groups:
            sub = [r for r in recs if int(r.get("group", 1)) == group]
            gstats = _group_stats(sub)
            if not gstats:
                continue
            group_rows.append({"group": group, **gstats})
        aggregates.append({
            "model_display": display,
            "provider_name": provider_name,
            "model": model,
            "provider_alias": alias,
            "task_scores": stats["task_scores"],
            "task_rates": stats["task_rates"],
            "max_global_error": stats["max_global_error"],
            "average_score": stats["average_score"],
            "average_rate": stats["average_rate"],
            "test_count": max(ok_groups) if ok_groups else 0,
            "completed_tasks": stats["completed_tasks"],
            "run_count": len(recs),
            "group_rows": group_rows,
            "records": recs,
        })
    aggregates.sort(key=lambda x: (-x["average_score"], x["model_display"], x["provider_name"]))
    for index, item in enumerate(aggregates, 1):
        item["rank"] = index
    return aggregates


def _summary_row(agg: dict[str, Any], test_count: int | None = None) -> list[Any]:
    row: list[Any] = [agg["model_display"], agg["provider_name"]]
    for task in TASK_ORDER:
        score = agg["task_scores"].get(task)
        rate = agg["task_rates"].get(task)
        # Scores are displayed as integers, exactly like the template.
        row.append(int(round(float(score))) if score is not None else None)
        row.append(round(float(rate), 6) if rate is not None else None)
        if task == "global":
            err = agg.get("max_global_error")
            row.append(round(float(err), 6) if err is not None else None)
    row.append(round(float(agg["average_score"]), 2))
    row.append(round(float(agg["average_rate"]), 6))
    row.append(int(agg["test_count"] if test_count is None else test_count))
    return row


MAP_METRIC_HEADERS = [
    "模型名", "来源渠道", "任务", "测试次数",
    "准确率", "HSS", "边界F1@1", "窄水道保留率",
    "岛屿IoU中位数", "岛屿IoU面积加权", "岛屿IoU≥0.5比例",
]


def _map_metric_values(rec: dict[str, Any]) -> dict[str, Any]:
    """Return per-map metrics, with fallbacks for archives created earlier."""
    metrics = rec.get("map_metrics") or {}
    score = rec.get("score") or {}
    return {
        "accuracy": metrics.get("accuracy_pct", score.get("accuracy_pct")),
        "hss": metrics.get("hss"),
        "boundary_f1_d1": metrics.get("boundary_f1_d1"),
        "narrow_water_keep": metrics.get("narrow_water_keep"),
        "island_iou_median": metrics.get("island_iou_median"),
        "island_iou_area_weighted": metrics.get("island_iou_area_weighted"),
        "island_iou_ge50": metrics.get("island_iou_ge50"),
    }


def _map_metric_row(rec: dict[str, Any]) -> list[Any]:
    m = _map_metric_values(rec)
    return [
        rec.get("model_display", ""),
        rec.get("provider_display") or rec.get("provider_name", ""),
        rec.get("task_label", rec.get("task", "")),
        int(rec.get("group", 1)),
        (float(m["accuracy"]) / 100.0) if m.get("accuracy") is not None else None,
        m.get("hss"),
        m.get("boundary_f1_d1"),
        m.get("narrow_water_keep"),
        m.get("island_iou_median"),
        m.get("island_iou_area_weighted"),
        m.get("island_iou_ge50"),
    ]


def aggregate_map_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average each per-map metric across independent groups."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for rec in records:
        if not _summary_scored(rec):
            continue
        key = (
            str(rec.get("model_display", rec.get("model", "?"))),
            str(rec.get("provider_display") or rec.get("provider_name", "?")),
            str(rec.get("task", "")),
        )
        groups.setdefault(key, []).append(rec)
    out: list[dict[str, Any]] = []
    for (display, provider, task), recs in groups.items():
        values = [_map_metric_values(r) for r in recs]
        row: dict[str, Any] = {
            "model_display": display,
            "provider_display": provider,
            "task": task,
            "task_label": recs[0].get("task_label", task),
            "groups": len(recs),
        }
        for key in (
            "accuracy", "hss", "boundary_f1_d1", "narrow_water_keep",
            "island_iou_median", "island_iou_area_weighted", "island_iou_ge50",
        ):
            vals = [float(v[key]) for v in values if v.get(key) is not None]
            row[key] = float(np.mean(vals)) if vals else None
        out.append(row)
    out.sort(key=lambda r: (
        r["model_display"], r["provider_display"],
        TASK_ORDER.index(r["task"]) if r["task"] in TASK_ORDER else 99,
    ))
    return out


def _map_metric_aggregate_row(row: dict[str, Any]) -> list[Any]:
    return [
        row["model_display"], row["provider_display"], row["task_label"], row["groups"],
        (row["accuracy"] / 100.0) if row.get("accuracy") is not None else None,
        row.get("hss"), row.get("boundary_f1_d1"),
        row.get("narrow_water_keep"),
        row.get("island_iou_median"),
        row.get("island_iou_area_weighted"),
        row.get("island_iou_ge50"),
    ]


def write_summary_metrics_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(MAP_METRIC_HEADERS)
        for rec in sorted((r for r in records if _summary_scored(r)), key=_record_sort_key):
            writer.writerow(_map_metric_row(rec))
    agg_path = path.with_name("summary_metrics_by_map.csv")
    with agg_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(MAP_METRIC_HEADERS)
        for row in aggregate_map_metrics(records):
            writer.writerow(_map_metric_aggregate_row(row))


def write_summary_csv(path: Path, aggregates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_HEADERS)
        for agg in aggregates:
            writer.writerow(_summary_row(agg))


def _fmt_score(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def _fmt_rate(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def write_summary_markdown(path: Path, aggregates: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    lines = [
        "# 地图实验汇总",
        "",
        f"生成时间：{now_iso()}",
        f"有效运行数：{sum(1 for r in records if _summary_scored(r))}；模型/渠道组合数：{len(aggregates)}",
        "",
        "评分规则：错 1 格扣 1 分；区域得分按 16,200 归一化后与其他图比较。",
        "",
        "| " + " | ".join(SUMMARY_HEADERS) + " |",
        "|" + "|".join(["---"] * len(SUMMARY_HEADERS)) + "|",
    ]
    for agg in aggregates:
        cells = [agg["model_display"], agg["provider_name"]]
        for task in TASK_ORDER:
            cells.append(_fmt_score(agg["task_scores"].get(task)))
            cells.append(_fmt_rate(agg["task_rates"].get(task)))
            if task == "global":
                cells.append(_fmt_rate(agg.get("max_global_error")))
        cells.append(f"{agg['average_score']:.2f}")
        cells.append(_fmt_rate(agg["average_rate"]))
        cells.append(str(agg["test_count"]))
        lines.append("| " + " | ".join(cells) + " |")

    group_headers = (
        ["模型", "渠道", "测试次数"]
        + [TASK_HEADERS[task][0] for task in TASK_ORDER]
        + ["平均分", "平均得分率"]
    )
    lines.extend([
        "",
        "## 分组明细",
        "",
        "| " + " | ".join(group_headers) + " |",
        "|" + "|".join(["---"] * 3 + ["---:"] * (len(TASK_ORDER) + 2)) + "|",
    ])
    for agg in aggregates:
        for grp in agg["group_rows"]:
            cells = [
                agg["model_display"],
                agg["provider_name"],
                str(grp["group"]),
                *[_fmt_score(grp["task_scores"].get(task)) for task in TASK_ORDER],
                f"{grp['average_score']:.2f}",
                _fmt_rate(grp["average_rate"]),
            ]
            lines.append("| " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## 每图指标",
        "",
        "准确率、窄水道保留率与 IoU≥0.5 分量比例为百分比；HSS、边界F1@1 与 IoU 为 0–1 指标。",
        "",
        "| 模型 | 渠道 | 任务 | 测试次数 | 准确率 | HSS | 边界F1@1 | 窄水道保留率 | IoU中位数 | IoU面积加权 | IoU≥0.5比例 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    def pct(v):
        return "-" if v is None else f"{float(v) * 100:.2f}%"

    def num(v):
        return "-" if v is None else f"{float(v):.3f}"

    for row in aggregate_map_metrics(records):
        lines.append("| " + " | ".join([
            str(row["model_display"]),
            str(row["provider_display"]),
            str(row["task_label"]),
            str(row["groups"]),
            pct((row["accuracy"] / 100.0) if row.get("accuracy") is not None else None),
            num(row.get("hss")),
            num(row.get("boundary_f1_d1")),
            pct(row.get("narrow_water_keep")),
            num(row.get("island_iou_median")),
            num(row.get("island_iou_area_weighted")),
            pct(row.get("island_iou_ge50")),
        ]) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _apply_template_layout(sheet, ncols: int = len(SUMMARY_HEADERS)) -> None:
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(bold=True, color="FFFFFF")
    thin = Side(style="thin", color="B7B7B7")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for col in range(1, ncols + 1):
        cell = sheet.cell(row=1, column=col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border
    sheet.row_dimensions[1].height = 30
    sheet.freeze_panes = "C2"
    sheet.auto_filter.ref = f"A1:{get_column_letter(ncols)}{max(1, sheet.max_row)}"

    widths = [26, 16]
    score_cols: set[int] = set()
    rate_cols: set[int] = set()
    max_global_error_col: int | None = None
    col = 3
    for task in TASK_ORDER:
        score_cols.add(col)
        widths.append(13 if task.startswith("arctic") else 11)
        col += 1
        rate_cols.add(col)
        widths.append(13 if task.startswith("arctic") else 12)
        col += 1
        if task == "global":
            max_global_error_col = col
            widths.append(15)
            col += 1
    average_score_col = col
    widths.append(11)
    col += 1
    average_rate_col = col
    widths.append(12)
    col += 1
    test_count_col = col
    widths.append(10)

    for index, width in enumerate(widths[:ncols], 1):
        sheet.column_dimensions[get_column_letter(index)].width = width

    for row in range(2, sheet.max_row + 1):
        for col in range(1, ncols + 1):
            cell = sheet.cell(row=row, column=col)
            cell.border = border
            if col in (1, 2):
                cell.alignment = Alignment(horizontal="left", vertical="center")
            else:
                cell.alignment = Alignment(horizontal="center", vertical="center")
            if col in score_cols:
                cell.number_format = "0"
            elif col in rate_cols or col == max_global_error_col or col == average_rate_col:
                cell.number_format = "0.00%"
            elif col == average_score_col:
                cell.number_format = "0.00"
            elif col == test_count_col:
                cell.number_format = "0"
    sheet.sheet_view.showGridLines = True


def write_excel(path: Path, aggregates: list[dict[str, Any]], records: list[dict[str, Any]]) -> bool:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter
    except Exception as exc:
        print(f"[warn] openpyxl unavailable, skipped xlsx: {exc}", file=sys.stderr)
        return False

    wb = Workbook()
    ws = wb.active
    ws.title = "总览"
    ws.append(SUMMARY_HEADERS)
    for agg in aggregates:
        ws.append(_summary_row(agg))
    _apply_template_layout(ws)

    # Per-model sheets mirroring the template's "<model>的结果" layout:
    # one row per test/group, with 测试次数 = group number.
    for agg in aggregates:
        title = re.sub(r"[\[\]:*?/\\]", "_", agg["model_display"])[:28] + "的结果"
        sheet = wb.create_sheet(title)
        sheet.append(SUMMARY_HEADERS)
        for grp in agg["group_rows"]:
            sheet.append(_summary_row({**agg, **grp}, test_count=grp["group"]))
        _apply_template_layout(sheet)

    detail = wb.create_sheet("明细")
    detail_headers = [
        "模型名", "来源渠道", "模型标识", "任务", "测试次数", "状态",
        "原始得分", "满分", "得分率", "错误数", "虚报陆地", "漏报陆地",
        "归一化得分", "归档 PNG", "开始时间", "错误信息",
    ]
    detail.append(detail_headers)
    # Keep failed attempts in 明细 too, so a later --fulfill run can point to
    # the exact (model, provider, task, group) cell that still needs work.
    for rec in sorted((r for r in records if r.get("task") in TASK_ORDER), key=_record_sort_key):
        score = rec.get("score") or {}
        files = (rec.get("archive") or {}).get("files") or {}
        status = rec.get("status", "")
        rate = score.get("accuracy_pct")
        detail.append([
            rec.get("model_display", ""),
            rec.get("provider_display") or rec.get("provider_name", ""),
            rec.get("model", ""),
            rec.get("task_label", rec.get("task", "")),
            rec.get("group", 1),
            status,
            score.get("score"),
            score.get("max_score"),
            (float(rate) / 100.0) if rate is not None else None,
            score.get("errors"),
            score.get("extra_land"),
            score.get("missing_land"),
            score.get("normalized"),
            files.get("png"),
            rec.get("started_at", ""),
            rec.get("error", ""),
        ])
    _apply_dark_header(detail, len(detail_headers))
    for i, width in enumerate([26, 16, 24, 16, 10, 10, 11, 10, 10, 9, 10, 10, 12, 46, 22, 55], 1):
        detail.column_dimensions[get_column_letter(i)].width = width
    for row in range(2, detail.max_row + 1):
        for col in range(1, len(detail_headers) + 1):
            cell = detail.cell(row=row, column=col)
            if col in (7, 8, 13):
                cell.number_format = "0.##"
            elif col == 9:
                cell.number_format = "0.00%"
            if col not in (1, 2, 14):
                cell.alignment = Alignment(horizontal="center", vertical="center")

    for sheet_name, rows in (
        ("地图指标", [ _map_metric_aggregate_row(r) for r in aggregate_map_metrics(records) ]),
        ("地图指标明细", [ _map_metric_row(r) for r in sorted((x for x in records if _summary_scored(x)), key=_record_sort_key) ]),
    ):
        metrics_sheet = wb.create_sheet(sheet_name)
        metrics_sheet.append(MAP_METRIC_HEADERS)
        for row in rows:
            metrics_sheet.append(row)
        _apply_dark_header(metrics_sheet, len(MAP_METRIC_HEADERS))
        for i, width in enumerate([26, 16, 18, 10, 12, 11, 12, 14, 15, 16, 16], 1):
            metrics_sheet.column_dimensions[get_column_letter(i)].width = width
        for row in range(2, metrics_sheet.max_row + 1):
            for col in range(1, len(MAP_METRIC_HEADERS) + 1):
                cell = metrics_sheet.cell(row=row, column=col)
                if col in (5, 8, 11):
                    cell.number_format = "0.00%"
                elif col in (6, 7, 9, 10):
                    cell.number_format = "0.000"
                if col not in (1, 2, 3):
                    cell.alignment = Alignment(horizontal="center", vertical="center")

    try:
        wb.calculation.fullCalcOnLoad = True
    except Exception:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return True


def _apply_dark_header(sheet, ncols: int) -> None:
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(bold=True, color="FFFFFF")
    thin = Side(style="thin", color="B7B7B7")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for col in range(1, ncols + 1):
        cell = sheet.cell(row=1, column=col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border
    sheet.row_dimensions[1].height = 30
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = f"A1:{get_column_letter(ncols)}{max(1, sheet.max_row)}"
    for row in range(2, sheet.max_row + 1):
        for col in range(1, ncols + 1):
            sheet.cell(row=row, column=col).border = border


def write_manifest(archive_root: Path, records: list[dict[str, Any]], aggregates: list[dict[str, Any]]) -> None:
    manifest = {
        "generated_at": now_iso(),
        "archive_root": str(archive_root),
        "task_order": TASK_ORDER,
        "summary_headers": SUMMARY_HEADERS,
        "map_metric_headers": MAP_METRIC_HEADERS,
        "run_count": len(records),
        "aggregate_count": len(aggregates),
        "runs": sorted(records, key=_record_sort_key),
        "aggregates": aggregates,
    }
    write_json(archive_root / "manifest.json", manifest)


def normalize_excel_name(value: str | None) -> str:
    """Return a safe workbook filename below the archive root."""
    name = str(value or "").strip()
    if not name:
        raise ValueError("--excel-name cannot be empty")
    if any(char in name for char in '<>:"/\\|?*') or any(ord(char) < 32 for char in name):
        raise ValueError("--excel-name must be a filename, not a path")
    if name in (".", "..") or Path(name).name != name:
        raise ValueError("--excel-name must be a filename, not a path")
    if not name.lower().endswith(".xlsx"):
        name += ".xlsx"
    return name


def excel_report_path(archive_root: Path, excel_name: str | None = "summary.xlsx") -> Path:
    return archive_root.resolve() / normalize_excel_name(excel_name)


def write_reports(
    archive_root: Path,
    records: list[dict[str, Any]],
    excel_name: str = "summary.xlsx",
) -> list[dict[str, Any]]:
    aggregates = aggregate_records(records)
    write_summary_csv(archive_root / "summary.csv", aggregates)
    write_summary_metrics_csv(archive_root / "summary_metrics.csv", records)
    write_summary_markdown(archive_root / "summary.md", aggregates, records)
    write_manifest(archive_root, records, aggregates)
    write_excel(excel_report_path(archive_root, excel_name), aggregates, records)
    return aggregates

# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------

def print_providers(providers: list[Provider]) -> None:
    print("name\tid\tcurrent\tmodel\tapi_format\talias\tbase_url\tmodels")
    for p in providers:
        print("\t".join([
            p.name,
            p.id,
            "yes" if p.is_current else "no",
            p.model or "-",
            p.api_format or "-",
            p.alias or "-",
            p.base_url or "-",
            ",".join(provider_models(p)) or "-",
        ]))


def print_plan(plan: list[RunSpec], groups: int) -> None:
    print(f"planned runs: {len(plan)} (groups={groups})")
    print("provider\tmodel\ttask\tgroup\tmode\tarchive\tnote")
    for spec in plan:
        print("\t".join([
            spec.provider.name,
            spec.model,
            spec.task.key,
            str(spec.group),
            spec.mode or "-",
            spec.archive_stem or "-",
            spec.skip_reason or "-",
        ]))


def _archive_meta_for_spec(spec: RunSpec) -> Path:
    return spec.archive_dir / f"{spec.archive_stem}.meta.json"


def _failed_result(spec: RunSpec, index: int, error: str, status: str = "failed") -> dict[str, Any]:
    return {
        "run_id": f"{status}-{index}",
        "status": status,
        "error": error,
        "provider_id": spec.provider.id,
        "provider_name": spec.provider.name,
        "provider_display": display_provider_name(spec.provider),
        "provider_alias": spec.provider.alias,
        "provider_mode": spec.mode,
        "model": spec.model,
        "model_display": display_model_name(spec.model, spec.provider),
        "task": spec.task.key,
        "task_label": spec.task.label,
        "group": spec.group,
        "archive_stem": spec.archive_stem,
        "score": None,
        "map_metrics": None,
        "finished_at": now_iso(),
    }


def _execute_spec(
    spec: RunSpec,
    index: int,
    total: int,
    args: argparse.Namespace,
    archive_root: Path,
    print_lock: threading.Lock,
) -> dict[str, Any]:
    prefix = f"[{index}/{total}] {spec.provider.name} | {spec.model} | {spec.task.key} | g{spec.group}"
    run_dir: Path | None = None
    if spec.skip_reason:
        with print_lock:
            print(f"{prefix} -> SKIP: {spec.skip_reason}")
        return _failed_result(spec, index, spec.skip_reason, "skipped")
    if args.resume and _archive_meta_for_spec(spec).exists():
        try:
            old = read_json(_archive_meta_for_spec(spec))
            if old.get("status") == "ok":
                with print_lock:
                    print(f"{prefix} -> SKIP: already archived ({spec.archive_stem})")
                return old
        except Exception:
            pass
    with print_lock:
        print(f"{prefix} -> running ({spec.mode})")
    stop = threading.Event()
    run_dir_holder: list[Path | None] = [None]
    reporter = start_progress_reporter(
        prefix, stop, archive_root, spec, print_lock, run_dir_holder
    )

    def progress(message: str) -> None:
        with print_lock:
            print(f"{prefix} -> {message}")

    try:
        if args.backend == "api-tools":
            result = run_api_tools(
                spec=spec,
                python_exe=args.python,
                timeout_min=args.timeout_min,
                archive_root=archive_root,
                transport=spec.mode,
                reasoning_effort=args.reasoning_effort,
                use_previous_response=args.previous_response,
                progress=progress,
                max_output_tokens=args.max_output_tokens,
                request_retries=args.request_retries,
                retry_base_sec=args.retry_base_sec,
                provider_concurrency=args.provider_concurrency,
            )
        elif args.backend == "api":
            result = run_api(
                spec=spec,
                python_exe=args.python,
                timeout_min=args.timeout_min,
                archive_root=archive_root,
                transport=spec.mode,
                reasoning_effort=args.reasoning_effort,
            )
        else:
            result = run_codex(
                spec=spec,
                codex_exe=args.codex,
                python_exe=args.python,
                timeout_min=args.timeout_min,
                default_catalog=args.default_catalog,
                keep_work=args.keep_work,
                archive_root=archive_root,
            )
        run_dir = Path(result["run_dir"])
        run_dir_holder[0] = run_dir
        candidate = find_candidate(run_dir, spec.task)
        result = archive_result(
            result=result,
            spec=spec,
            candidate=candidate,
            truth=load_truth(spec.task),
            archive_root=archive_root,
            overwrite=args.overwrite,
        )
        score = result.get("score") or {}
        with print_lock:
            if result.get("status") == "ok":
                print(f"{prefix} -> ok {score.get('score')}/{score.get('max_score')} "
                      f"({score.get('accuracy_pct', 0):.2f}%) -> {result['archive']['stem']}")
            else:
                print(f"{prefix} -> {result.get('status')}: {result.get('error') or 'partial deliverables'}")
        return result
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        with print_lock:
            print(f"{prefix} -> FAIL: {msg}", file=sys.stderr)
            traceback.print_exc()
        failed = _failed_result(spec, index, msg, "failed")
        if spec.archive_dir and spec.archive_stem:
            spec.archive_dir.mkdir(parents=True, exist_ok=True)
            write_json(_archive_meta_for_spec(spec), failed)
        return failed
    finally:
        stop.set()
        reporter.join(timeout=2)
        if not args.keep_work and args.backend != "codex":
            pass
        elif not args.keep_work and run_dir is not None:
            try:
                if run_dir.exists():
                    shutil.rmtree(run_dir, ignore_errors=True)
            except Exception:
                pass


def run_batch(
    args: argparse.Namespace,
    tasks: dict[str, Task],
    providers: list[Provider],
) -> int:
    archive_root: Path = args.archive_root.resolve()
    archive_root.mkdir(parents=True, exist_ok=True)
    db_path: Path = args.db
    settings_path: Path = args.settings
    if not args.no_restore:
        restore_pending_state(archive_root, db_path, settings_path)

    records = load_archive_records(archive_root)
    if args.fulfill:
        if args.start_group != 1 or args.groups != 1:
            print(
                "[warn] --fulfill ignores --groups/--start-group; it retries exact "
                "failed cells from the selected summary workbook",
                file=sys.stderr,
            )
        if args.extra_group:
            print(
                "[warn] --extra-group is ignored with --fulfill; use one mode at a time",
                file=sys.stderr,
            )
        targets = load_fulfillment_targets(
            archive_root, records, tasks, model_selector=args.models,
            excel_name=args.excel_name,
        )
        selected_provider_keys = {
            provider.id for provider in providers
        } | {provider.name for provider in providers}
        targets = [
            target for target in targets
            if str(target.get("provider_id") or "") in selected_provider_keys
            or str(target.get("provider_name") or "") in selected_provider_keys
        ]
        targets = filter_fulfillment_targets(targets, providers, args.tasks)
        seen: set[tuple[str, str, str, int]] = set()
        unique_targets = []
        for target in targets:
            key = _target_key(target)
            if key not in seen:
                seen.add(key)
                unique_targets.append(target)
        targets = unique_targets
        plan = build_fulfillment_plan(
            providers=providers,
            tasks=tasks,
            targets=targets,
            archive_root=archive_root,
            backend=args.backend,
            api_transport=args.api_transport,
            provider_mode=args.provider_mode,
            allow_switch=args.switch_cc_provider,
        )
        if not plan:
            print("[fulfill] no failed/missing summary cells matched the selection")
            if not args.dry_run:
                write_reports(archive_root, records, excel_name=args.excel_name)
            return 0
    else:
        plan = build_plan(
            providers=providers,
            tasks=tasks,
            task_selector=args.tasks,
            model_selector=args.models,
            groups=args.groups,
            start_group=args.start_group,
            provider_mode=args.provider_mode,
            archive_root=archive_root,
            allow_switch=args.switch_cc_provider,
            backend=args.backend,
            api_transport=args.api_transport,
            extra_group=args.extra_group,
            records=records,
        )
    if args.dry_run:
        if args.fulfill:
            print(f"[fulfill] {len(plan)} failed/missing cells selected")
        elif args.extra_group:
            print(f"[extra-group] {len(plan)} continuation runs selected")
        print_plan(plan, args.extra_group if args.extra_group else args.groups)
        if args.jobs > 1:
            print(f"[parallel] would run up to {args.jobs} jobs concurrently")
        return 0

    proxy_ok, proxy_msg = proxy_health(DEFAULT_PROXY)
    print(f"[proxy] {'ok' if proxy_ok else 'unavailable'}: {proxy_msg}")
    if not proxy_ok:
        print("[warn] proxy is not reachable; direct mode will be used where possible", file=sys.stderr)

    snapshot = cc_switch_snapshot(db_path, settings_path)
    selected_provider_id = snapshot.get("provider_id")
    switched_any = False
    results: list[dict[str, Any]] = []
    exit_code = 0
    print_lock = threading.Lock()

    try:
        jobs = max(1, int(args.jobs))
        if jobs > len(plan):
            jobs = max(1, len(plan))

        # CC Switch provider switching is global and incompatible with parallel
        # proxy requests. Direct transport and api-tools are safe to parallelize.
        needs_switch = any(
            s.mode == "proxy" and s.provider.id != selected_provider_id and s.provider.id != (
                selected_provider_id or ""
            )
            for s in plan
        )
        if jobs > 1 and needs_switch:
            print("[warn] proxy provider switching cannot be parallel; reducing to 1 job", file=sys.stderr)
            jobs = 1

        if jobs == 1:
            for index, spec in enumerate(plan, 1):
                if spec.mode == "proxy" and spec.provider.id != selected_provider_id:
                    if not args.switch_cc_provider:
                        with print_lock:
                            print(f"[{index}/{len(plan)}] -> SKIP: provider is not selected in CC Switch; rerun with --switch-cc-provider")
                        results.append(_failed_result(spec, index, "provider is not selected in CC Switch", "skipped"))
                        continue
                    try:
                        if not switched_any:
                            write_pending_state(archive_root, snapshot)
                        switch_cc_provider(spec.provider, db_path, settings_path)
                        selected_provider_id = spec.provider.id
                        switched_any = True
                        with print_lock:
                            print(f"[cc-switch] selected {spec.provider.name}")
                    except Exception as exc:
                        msg = f"CC Switch provider switch failed: {exc}"
                        with print_lock:
                            print(f"[cc-switch] {msg}", file=sys.stderr)
                        results.append(_failed_result(spec, index, msg))
                        exit_code = 2
                        continue
                res = _execute_spec(spec, index, len(plan), args, archive_root, print_lock)
                results.append(res)
                if res.get("status") not in ("ok", "skipped"):
                    exit_code = 2
        else:
            print(f"[parallel] running up to {jobs} jobs concurrently")
            with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="map-exp") as pool:
                futures = {
                    pool.submit(_execute_spec, spec, index, len(plan), args, archive_root, print_lock): spec
                    for index, spec in enumerate(plan, 1)
                }
                for future in as_completed(futures):
                    spec = futures[future]
                    try:
                        res = future.result()
                    except Exception as exc:
                        msg = f"{type(exc).__name__}: {exc}"
                        with print_lock:
                            print(f"[batch] FAIL {spec.archive_stem or spec.model}: {msg}", file=sys.stderr)
                        res = _failed_result(spec, 0, msg)
                    results.append(res)
                    if res.get("status") not in ("ok", "skipped"):
                        exit_code = 2
    finally:
        if switched_any:
            try:
                restore_cc_provider(snapshot, db_path, settings_path)
                clear_pending_state(archive_root)
                print("[cc-switch] restored original provider selection")
            except Exception as exc:
                print(f"[warn] failed to restore CC Switch provider: {exc}", file=sys.stderr)

    records = load_archive_records(archive_root)
    aggregates = write_reports(archive_root, records, excel_name=args.excel_name)
    excel_path = excel_report_path(archive_root, args.excel_name)
    print(f"[summary] {len(records)} runs, {len(aggregates)} model/provider rows -> {excel_path}")
    return exit_code


# ---------------------------------------------------------------------------
# Re-score existing archive without re-running models
# ---------------------------------------------------------------------------

def rescore_archives(
    archive_root: Path, tasks: dict[str, Task], excel_name: str = "summary.xlsx"
) -> int:
    records = load_archive_records(archive_root)
    if not records:
        print("no archive records found", file=sys.stderr)
        return 2
    updated = 0
    for rec in records:
        task_key = rec.get("task")
        if task_key not in tasks:
            continue
        task = tasks[task_key]
        # Backfill newly added metrics on old archives without calling models.
        metrics = rec.get("map_metrics")
        if _scored(rec) and isinstance(metrics, dict) and "island_iou_area_weighted" not in metrics:
            png = (rec.get("archive") or {}).get("files", {}).get("png")
            if png and Path(png).exists():
                try:
                    candidate = Candidate(
                        matrix=decode_image(Path(png), task)[0],
                        kind="exact",
                        source=Path(png),
                        how="archive rescore",
                        meta={"levels": [0, 255]},
                    )
                    quality = map_quality_metrics(candidate, load_truth(task))
                    metrics.update({
                        "island_iou_median": quality["island_iou_median"],
                        "island_iou_area_weighted": quality["island_iou_area_weighted"],
                        "island_iou_ge50": quality["island_iou_ge50"],
                    })
                    meta_path = rec.get("_meta_path")
                    if meta_path:
                        write_json(Path(meta_path), rec)
                    updated += 1
                except Exception as exc:
                    print(
                        f"[warn] cannot backfill IoU metrics for {rec.get('archive_stem')}: {exc}",
                        file=sys.stderr,
                    )
        files = (rec.get("archive") or {}).get("files") or {}
        png = Path(files["png"]) if files.get("png") else None
        csv_path = Path(files["csv"]) if files.get("csv") else None
        upscale = Path(files["upscale"]) if files.get("upscale") else None
        cand: Candidate | None = None
        if png and png.exists():
            try:
                matrix, meta = decode_image(png, task)
                meta["orientation_ok"] = True
                cand = Candidate(
                    matrix=matrix,
                    kind=str(meta.get("kind", "other")),
                    source=png,
                    how=str(meta.get("how", "")),
                    meta=meta,
                )
            except Exception as exc:
                print(f"[warn] {png}: {exc}", file=sys.stderr)
        if cand is None and csv_path and csv_path.exists():
            try:
                matrix = load_csv_matrix(csv_path)
                if matrix.shape == (task.rows, task.cols):
                    cand = Candidate(
                        matrix=matrix,
                        kind="csv",
                        source=None,
                        how="archived CSV",
                        meta={"mode": "csv", "size": [task.cols, task.rows], "levels": [0, 1]},
                        csv_path=csv_path,
                        csv_match=True,
                    )
            except Exception as exc:
                print(f"[warn] {csv_path}: {exc}", file=sys.stderr)
        if cand is None:
            rec["status"] = "failed"
            rec["error"] = "no archived candidate"
            write_json(Path(rec["_meta_path"]), {k: v for k, v in rec.items() if k != "_meta_path"})
            continue
        truth = load_truth(task)
        score = score_matrix(cand.matrix, truth)
        cand.meta["orientation_ok"] = bool(score["orientation_ok"])
        if csv_path and csv_path.exists():
            try:
                cm = load_csv_matrix(csv_path)
                cand.csv_path = csv_path
                cand.csv_match = np.array_equal(cm, cand.matrix)
            except Exception:
                pass
        if upscale and upscale.exists():
            try:
                um, _ = decode_image(upscale, task)
                cand.upscale_path = upscale
                cand.upscale_match = np.array_equal(um, cand.matrix)
            except Exception:
                pass
        rec["score"] = {k: v for k, v in score.items() if k != "diff"}
        rec["validation"] = validate_candidate(cand, task, truth, score)
        try:
            rec["map_metrics"] = map_quality_metrics(cand, truth)
        except Exception as exc:
            rec["map_metrics"] = {"error": f"{type(exc).__name__}: {exc}"}
        rec["status"] = "ok" if rec["validation"].get("status") == "ok" else "partial"
        out = {k: v for k, v in rec.items() if k != "_meta_path"}
        write_json(Path(rec["_meta_path"]), out)
        updated += 1
    records = load_archive_records(archive_root)
    aggregates = write_reports(archive_root, records, excel_name=excel_name)
    print(f"[rescore] updated {updated} records; {len(aggregates)} summary rows")
    return 0

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_ALIASES = ROOT / "experiment_config" / "providers.json"


def load_aliases(path: Path) -> dict[str, str]:
    # Aliases map a provider name as stored in the local registry to the short
    # suffix used in file names.  Edit experiment_config/providers.json instead
    # of hard-coding names here; the defaults below are only illustrative.
    defaults: dict[str, str] = {
        "Local Provider": "",
        "Alias OR": "or",
        "Alias AH": "ah",
    }
    if path.exists():
        try:
            data = read_json(path)
            if isinstance(data, dict):
                defaults.update({str(k): str(v) for k, v in data.items()})
        except Exception as exc:
            print(f"[warn] cannot read aliases {path}: {exc}", file=sys.stderr)
    return defaults


def print_tasks(tasks: dict[str, Task]) -> None:
    print("task\tlabel\tshape\tcells\tupscale\tprompt\treference")
    for key, task in tasks.items():
        print("\t".join([
            key,
            task.label,
            f"{task.rows}x{task.cols}",
            str(task.cells),
            f"{task.upscale[0]}x{task.upscale[1]}",
            str(task.prompt_path),
            str(task.ref_path),
        ]))


def _default_codex() -> str:
    """Pick the newest Codex CLI, preferring the desktop-managed build.

    The desktop app and older Program Files installs both put a `codex.exe` on
    PATH.  The old build does not support --no-daemon and has a different
    Windows sandbox behavior, so always prefer the newest binary explicitly.
    """
    candidates: list[Path] = []
    env_path = os.environ.get("CODEX_CLI_PATH")
    if env_path and Path(env_path).is_file():
        candidates.append(Path(env_path))
    local = os.environ.get("LOCALAPPDATA")
    if local:
        for pattern in (
            r"[OI]\Codex\bin\*\codex.exe",
            r"Programs\[OI]\Codex\bin\codex.exe",
        ):
            candidates.extend(Path(local).glob(pattern))
    which = shutil.which("codex")
    if which:
        candidates.append(Path(which))
    existing = [p for p in candidates if p.is_file()]
    if not existing:
        return "codex"
    existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(existing[0])


_CODEX_FEATURES: dict[str, bool] = {}


def _codex_supports_no_daemon(codex_exe: str) -> bool:
    if codex_exe in _CODEX_FEATURES:
        return _CODEX_FEATURES[codex_exe]
    supported = False
    try:
        proc = subprocess.run(
            [codex_exe, "--help"],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        supported = "--no-daemon" in (proc.stdout or "")
    except Exception:
        supported = False
    _CODEX_FEATURES[codex_exe] = supported
    return supported


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automate the land/water map experiment, score results, and archive them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--providers", default=None,
                        help="CC Switch provider names/ids, comma separated, or all/current "
                             "(default: current; all when --fulfill is used)")
    parser.add_argument("--models", default=None,
                        help="model slugs, comma separated, or all (default: provider's configured model)")
    parser.add_argument("--tasks", default="all",
                        help="task keys (global,csc,dzh,seto,arctic01,hecate), comma separated, or all")
    parser.add_argument("--groups", type=int, default=1, help="number of independent groups per task")
    parser.add_argument("--start-group", type=int, default=1, help="first group number")
    parser.add_argument("--extra-group", type=int, default=0,
                        help="append this many groups per task after the model's current 测试次数")
    parser.add_argument("--fulfill", action="store_true",
                        help="retry failed/missing cells recorded in the selected summary workbook")
    parser.add_argument("--previous-response", action=argparse.BooleanOptionalAction, default=False,
                        help="use previous_response_id when the provider persists responses (stateless gateways do not)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="maximum concurrent runs (default 1); only direct/api-tools is safe to parallelize")
    parser.add_argument("--backend", choices=["api-tools", "api", "codex"], default="api-tools",
                        help="api-tools=provider tool loop with host-side tools (recommended); api=one-shot; codex=Codex CLI agent")
    parser.add_argument("--api-transport", choices=["auto", "proxy", "direct"], default="auto",
                        help="auto=direct for non-current Responses providers, proxy for current; proxy=CC Switch local proxy; direct=provider endpoint")
    parser.add_argument("--max-output-tokens", type=int, default=12000,
                        help="max output tokens per model turn; smaller values force smaller chunks")
    parser.add_argument("--request-retries", type=int, default=4,
                        help="retry a failed API turn (403/429/5xx/stream/network) this many times; default 4")
    parser.add_argument("--retry-base-sec", type=float, default=5.0,
                        help="base exponential backoff in seconds; 403 uses at least 10 seconds")
    parser.add_argument("--provider-concurrency", type=int, default=4,
                        help="maximum in-flight API requests per provider; default 4")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="high",
                        help="reasoning effort for API requests (lower finishes faster)")
    parser.add_argument("--provider-mode", choices=["auto", "proxy", "direct"], default="auto",
                        help="auto=proxy with switching when possible, proxy=CC Switch local proxy, direct=provider endpoint")
    parser.add_argument("--switch-cc-provider", action=argparse.BooleanOptionalAction, default=False,
                        help="temporarily select each provider in CC Switch for proxy runs (default: true)")
    parser.add_argument("--keep-work", action=argparse.BooleanOptionalAction, default=True,
                        help="keep the temporary CODEX_HOME and full run log (default: true)")
    parser.add_argument("--four-dim", action=argparse.BooleanOptionalAction, default=True,
                        help="legacy compatibility flag; per-map diagnostics are always recorded")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing archive names")
    parser.add_argument("--resume", action="store_true", help="skip runs already archived with status ok")
    parser.add_argument("--dry-run", action="store_true", help="print the run plan without calling a model")
    parser.add_argument("--score-only", action="store_true", help="re-score existing archives without rerunning")
    parser.add_argument("--report-only", action="store_true", help="rebuild summary files from existing archives")
    parser.add_argument("--excel-name", default="summary.xlsx", metavar="NAME",
                        help="Excel filename under --archive-root (default: summary.xlsx)")
    parser.add_argument("--list-providers", action="store_true", help="list CC Switch Codex providers and models")
    parser.add_argument("--list-tasks", action="store_true", help="list configured tasks")
    parser.add_argument("--check-proxy", action="store_true", help="check the CC Switch local proxy")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="CC Switch SQLite database")
    parser.add_argument("--settings", type=Path, default=Path.home() / ".cc-switch" / "settings.json",
                        help="CC Switch settings.json")
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE,
                        help="archive directory (default: ./experiments)")
    parser.add_argument("--tasks-config", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON,
                        help="Python interpreter made available to the tested model")
    parser.add_argument("--codex", default=_default_codex(),
                        help="Codex CLI executable")
    parser.add_argument("--default-catalog", type=Path, default=DEFAULT_CATALOG,
                        help="model catalog used for the isolated CODEX_HOME")
    parser.add_argument("--timeout-min", type=int, default=60, help="per-run timeout in minutes")
    parser.add_argument("--no-restore", action="store_true",
                        help="do not restore a pending CC Switch state at startup")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.groups < 1:
        parser.error("--groups must be >= 1")
    if args.start_group < 1:
        parser.error("--start-group must be >= 1")
    if args.extra_group < 0:
        parser.error("--extra-group must be >= 0")
    if args.fulfill and args.extra_group:
        parser.error("--fulfill and --extra-group cannot be used together")
    if args.models and args.models.strip().lower() == "all":
        args.models = "all"
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.max_output_tokens < 1000:
        parser.error("--max-output-tokens must be >= 1000")
    if args.request_retries < 0:
        parser.error("--request-retries must be >= 0")
    if args.retry_base_sec < 0:
        parser.error("--retry-base-sec must be >= 0")
    if args.provider_concurrency < 1:
        parser.error("--provider-concurrency must be >= 1")
    try:
        args.excel_name = normalize_excel_name(args.excel_name)
    except ValueError as exc:
        parser.error(str(exc))

    tasks = load_tasks(args.tasks_config)
    aliases = load_aliases(args.aliases)

    if args.list_tasks:
        print_tasks(tasks)
        return 0

    try:
        providers = load_providers(args.db, aliases)
    except Exception as exc:
        print(f"cannot read CC Switch providers: {exc}", file=sys.stderr)
        return 2

    if args.list_providers:
        print_providers(providers)
        return 0

    if args.check_proxy:
        ok, msg = proxy_health(DEFAULT_PROXY)
        print(f"proxy: {'ok' if ok else 'unavailable'}")
        print(msg)
        return 0 if ok else 2

    if args.report_only:
        records = load_archive_records(args.archive_root)
        aggregates = write_reports(args.archive_root, records, excel_name=args.excel_name)
        excel_path = excel_report_path(args.archive_root, args.excel_name)
        print(f"[report] {len(records)} runs, {len(aggregates)} summary rows -> {excel_path}")
        return 0

    if args.score_only:
        return rescore_archives(args.archive_root, tasks, excel_name=args.excel_name)

    provider_selector = args.providers or ("all" if args.fulfill else "current")
    try:
        selected = select_providers(providers, provider_selector)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2
    if not selected:
        print("no providers selected", file=sys.stderr)
        return 2

    return run_batch(args, tasks, selected)



# ---------------------------------------------------------------------------
# Fallback artifact recovery
# ---------------------------------------------------------------------------

def _read_model_text(run_dir: Path) -> str:
    parts: list[str] = []
    last = run_dir / "last_message.txt"
    if last.exists():
        parts.append(last.read_text(encoding="utf-8", errors="replace"))
    log = run_dir / "codex.log"
    if log.exists():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                obj = json.loads(line)
            except Exception:
                continue
            item = obj.get("item") or {}
            if item.get("type") == "agent_message" and item.get("text"):
                parts.append(str(item["text"]))
    return "\n\n".join(parts)


def _code_blocks(text: str) -> list[str]:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S | re.I)
    return [b.strip() for b in blocks if b.strip()]


def _looks_like_generator(code: str) -> bool:
    markers = ("Image", "PIL", "csv", "numpy", "write_text", "write_bytes", "savetxt", "tofile")
    return len(code) > 80 and sum(1 for m in markers if m in code) >= 1


def _recover_csv_block(text: str, task: Task) -> np.ndarray | None:
    for block in _code_blocks(text):
        rows: list[list[int]] = []
        for line in block.splitlines():
            s = line.strip().strip("[]")
            if not s or s.startswith("#"):
                continue
            parts = [p for p in re.split(r"[,\s]+", s) if p]
            if len(parts) != task.cols:
                if rows:
                    break
                continue
            try:
                vals = [int(float(p)) for p in parts]
            except ValueError:
                if rows:
                    break
                continue
            if any(v not in (0, 1) for v in vals):
                if rows:
                    break
                continue
            rows.append(vals)
            if len(rows) == task.rows:
                return np.asarray(rows, dtype=np.uint8)
        if len(rows) >= task.rows:
            return np.asarray(rows[: task.rows], dtype=np.uint8)
    return None


def recover_artifacts(run_dir: Path, task: Task, python_exe: Path) -> dict[str, Any]:
    """Recover deliverables when the model's sandbox blocked file writes.

    The model is asked to print its generator script as a fallback.  This
    function extracts that script, runs it in the isolated run directory, and
    then falls back to a plain 0/1 code block if needed.
    """
    text = _read_model_text(run_dir)
    out_dir = run_dir / "outputs"
    out_dir.mkdir(exist_ok=True)
    info: dict[str, Any] = {"attempted": True, "script": None, "csv_block": False, "error": ""}
    blocks = _code_blocks(text)
    scripts = [b for b in blocks if _looks_like_generator(b)]
    if scripts:
        script = max(scripts, key=len)
        script_path = out_dir / "generate_map_recovered.py"
        script_path.write_text(script + "\n", encoding="utf-8")
        info["script"] = str(script_path)
        env = _python_env(python_exe)
        try:
            proc = subprocess.run(
                [str(python_exe), str(script_path)],
                cwd=str(run_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            info["script_returncode"] = proc.returncode
            info["script_stdout"] = proc.stdout[-4000:]
            info["script_stderr"] = proc.stderr[-4000:]
            if proc.returncode != 0:
                info["error"] = f"recovered script exited {proc.returncode}"
        except Exception as exc:
            info["error"] = f"recovered script failed: {type(exc).__name__}: {exc}"

    if find_candidate(run_dir, task) is None:
        matrix = _recover_csv_block(text, task)
        if matrix is not None:
            write_matrix_csv(matrix, out_dir / "candidate.csv")
            write_matrix_png(matrix, out_dir / "candidate.png")
            Image.fromarray((nn_upscale(matrix) * 255).astype(np.uint8), mode="L").save(
                out_dir / "candidate_8x.png"
            )
            info["csv_block"] = True
    return info

# ---------------------------------------------------------------------------
# API tool loop (host-side tools, no Windows sandbox)
# ---------------------------------------------------------------------------

@dataclass
class ToolSession:
    root: Path
    started_at: float
    call_count: int = 0


def _safe_workspace_path(root: Path, value: str) -> Path:
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        raise ValueError(f"path escapes the run workspace: {value!r}")
    return candidate


def _tool_list_files(session: ToolSession, args: dict[str, Any]) -> str:
    directory = _safe_workspace_path(session.root, str(args.get("path") or "."))
    if not directory.is_dir():
        return f"not a directory: {directory}"
    items = sorted(directory.iterdir(), key=lambda p: (p.is_dir(), p.name.lower()))
    lines = []
    for item in items[:300]:
        kind = "dir" if item.is_dir() else "file"
        size = "" if item.is_dir() else str(item.stat().st_size)
        lines.append(f"{kind}\t{size}\t{item.relative_to(session.root)}")
    return "\n".join(lines) or "(empty)"


def _tool_write_file(session: ToolSession, args: dict[str, Any]) -> str:
    path = _safe_workspace_path(session.root, str(args.get("path") or ""))
    content = args.get("content")
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    path.parent.mkdir(parents=True, exist_ok=True)
    append = bool(args.get("append"))
    if args.get("encoding") == "base64":
        import base64
        mode = "ab" if append else "wb"
        with path.open(mode) as handle:
            handle.write(base64.b64decode(content))
    else:
        mode = "a" if append else "w"
        with path.open(mode, encoding=str(args.get("encoding") or "utf-8")) as handle:
            handle.write(content)
    return f"wrote {path.relative_to(session.root)} ({path.stat().st_size} bytes)"


def _tool_read_file(session: ToolSession, args: dict[str, Any]) -> str:
    path = _safe_workspace_path(session.root, str(args.get("path") or ""))
    if not path.is_file():
        raise ValueError(f"not a file: {args.get('path')}")
    if path.stat().st_size > 500_000:
        raise ValueError("file too large to read")
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        import base64
        return "base64:" + base64.b64encode(raw).decode("ascii")


def _tool_run_python(session: ToolSession, args: dict[str, Any], python_exe: Path) -> str:
    script = args.get("path")
    cwd_arg = args.get("cwd") or "."
    if not script:
        raise ValueError("path is required")
    script_path = _safe_workspace_path(session.root, str(script))
    cwd = _safe_workspace_path(session.root, str(cwd_arg))
    if not script_path.is_file():
        raise ValueError(f"script not found: {script}")
    if not cwd.is_dir():
        raise ValueError(f"cwd not found: {cwd_arg}")
    proc = subprocess.run(
        [str(python_exe), str(script_path)],
        cwd=str(cwd),
        env=_python_env(python_exe),
        capture_output=True,
        text=True,
        timeout=int(args.get("timeout_sec") or 300),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    output = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    return f"exit_code={proc.returncode}\n{output[-20000:]}"


def _tool_delete_file(session: ToolSession, args: dict[str, Any]) -> str:
    path = _safe_workspace_path(session.root, str(args.get("path") or ""))
    if path.is_file() or path.is_symlink():
        path.unlink()
        return f"deleted {path.relative_to(session.root)}"
    if path.is_dir():
        raise ValueError("delete_file only removes files")
    return "already absent"


# ---------------------------------------------------------------------------
# Codex-compatible tools: exec_command / apply_patch / write_stdin
# ---------------------------------------------------------------------------

_SHELL_SESSIONS: dict[str, subprocess.Popen] = {}
_SHELL_LOCK = threading.Lock()


def _shell_exe() -> str:
    return shutil.which("pwsh") or shutil.which("powershell") or "powershell.exe"


def _tool_exec_command(
    session: ToolSession,
    args: dict[str, Any],
    python_exe: Path,
) -> str:
    cmd = str(args.get("cmd") or "").strip()
    if not cmd:
        raise ValueError("cmd is required")
    workdir = _safe_workspace_path(session.root, str(args.get("workdir") or "."))
    if not workdir.is_dir():
        raise ValueError(f"workdir not found: {args.get('workdir')}")
    timeout_sec = int(args.get("timeout_sec") or 600)
    yield_ms = int(args.get("yield_time_ms") or 10000)
    max_chars = int(args.get("max_output_chars") or 20000)
    env = _python_env(python_exe)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.Popen(
        [_shell_exe(), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", cmd],
        cwd=str(workdir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    session.call_count += 1
    try:
        out, _ = proc.communicate(timeout=max(1, min(yield_ms, timeout_sec * 1000)) / 1000)
        return f"exit_code={proc.returncode}\n{(out or '')[-max_chars:]}"
    except subprocess.TimeoutExpired:
        sid = f"shell-{len(_SHELL_SESSIONS) + 1}"
        with _SHELL_LOCK:
            _SHELL_SESSIONS[sid] = proc
        return (
            f"still_running session_id={sid}\n"
            "Use write_stdin with this session_id to poll or send input."
        )


def _tool_write_stdin(session: ToolSession, args: dict[str, Any]) -> str:
    sid = str(args.get("session_id") or "")
    chars = str(args.get("chars") or "")
    with _SHELL_LOCK:
        proc = _SHELL_SESSIONS.get(sid)
    if proc is None:
        raise ValueError(f"unknown session_id: {sid}")
    if chars:
        try:
            assert proc.stdin is not None
            proc.stdin.write(chars)
            proc.stdin.flush()
        except Exception as exc:
            raise ValueError(f"cannot write to stdin: {exc}")
    try:
        out, _ = proc.communicate(timeout=float(args.get("yield_time_ms") or 5000) / 1000)
        with _SHELL_LOCK:
            _SHELL_SESSIONS.pop(sid, None)
        return f"exit_code={proc.returncode}\n{(out or '')[-20000:]}"
    except subprocess.TimeoutExpired:
        return f"still_running session_id={sid}"


def _apply_patch_text(session: ToolSession, text: str) -> str:
    lines = text.replace("\r\n", "\n").split("\n")
    if not lines or lines[0].strip() != "*** Begin Patch":
        raise ValueError("patch must start with '*** Begin Patch'")
    i = 1
    changed: list[str] = []
    while i < len(lines):
        line = lines[i]
        if line.strip() == "*** End Patch":
            break
        if line.startswith("*** Add File: "):
            path = _safe_workspace_path(session.root, line[len("*** Add File: "):].strip())
            if path.exists():
                raise ValueError(f"file already exists: {path.relative_to(session.root)}")
            i += 1
            body: list[str] = []
            while i < len(lines) and not lines[i].startswith("*** "):
                if lines[i].startswith("+"):
                    body.append(lines[i][1:])
                elif lines[i].strip() == "":
                    body.append("")
                else:
                    break
                i += 1
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(body) + ("\n" if body else ""), encoding="utf-8")
            changed.append(f"added {path.relative_to(session.root)}")
            continue
        if line.startswith("*** Delete File: "):
            path = _safe_workspace_path(session.root, line[len("*** Delete File: "):].strip())
            if path.exists():
                path.unlink()
                changed.append(f"deleted {path.relative_to(session.root)}")
            i += 1
            continue
        if line.startswith("*** Update File: "):
            path = _safe_workspace_path(session.root, line[len("*** Update File: "):].strip())
            if not path.is_file():
                raise ValueError(f"file not found: {path.relative_to(session.root)}")
            i += 1
            move_to = None
            if i < len(lines) and lines[i].startswith("*** Move to: "):
                move_to = _safe_workspace_path(session.root, lines[i][len("*** Move to: "):].strip())
                i += 1
            content = path.read_text(encoding="utf-8")
            while i < len(lines) and not lines[i].startswith("*** "):
                if not lines[i].startswith("@@"):
                    i += 1
                    continue
                i += 1
                old_lines: list[str] = []
                new_lines: list[str] = []
                while i < len(lines) and not lines[i].startswith(("@@", "*** ")):
                    entry = lines[i]
                    if entry.startswith("+"):
                        new_lines.append(entry[1:])
                    elif entry.startswith("-"):
                        old_lines.append(entry[1:])
                    elif entry.startswith(" "):
                        old_lines.append(entry[1:])
                        new_lines.append(entry[1:])
                    elif entry.strip() == "":
                        old_lines.append("")
                        new_lines.append("")
                    else:
                        break
                    i += 1
                old_block = "\n".join(old_lines)
                new_block = "\n".join(new_lines)
                if old_block and old_block in content:
                    content = content.replace(old_block, new_block, 1)
                elif not old_block:
                    content = new_block
                else:
                    raise ValueError(
                        "patch context did not match; re-read the file and send a smaller patch"
                    )
            if move_to is not None:
                move_to.parent.mkdir(parents=True, exist_ok=True)
                move_to.write_text(content, encoding="utf-8")
                path.unlink()
                changed.append(f"updated {path.relative_to(session.root)} -> {move_to.relative_to(session.root)}")
            else:
                path.write_text(content, encoding="utf-8")
                changed.append(f"updated {path.relative_to(session.root)}")
            continue
        i += 1
    if not changed:
        raise ValueError("patch contained no changes")
    return "Applied patch:\n" + "\n".join(f"- {x}" for x in changed)


def _api_tools() -> list[dict[str, Any]]:
    """The same tool surface Codex exposes to the model."""
    return [
        {
            "type": "function",
            "name": "exec_command",
            "description": (
                "Run a shell command in the workspace and return its output. "
                "Use PowerShell syntax. Supports long-running commands with yield_time_ms; "
                "if it returns a session_id, poll it with write_stdin."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "workdir": {"type": "string"},
                    "yield_time_ms": {"type": "integer", "minimum": 1000, "maximum": 30000},
                    "timeout_sec": {"type": "integer", "minimum": 1, "maximum": 1800},
                    "max_output_chars": {"type": "integer", "minimum": 1000, "maximum": 60000},
                },
                "required": ["cmd", "workdir", "yield_time_ms", "timeout_sec", "max_output_chars"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "custom",
            "name": "apply_patch",
            "description": (
                "Apply a patch to files. Use the standard Codex patch format: "
                "*** Begin Patch / *** Add File: path / *** Update File: path / "
                "*** Delete File: path / @@ hunks / *** End Patch."
            ),
            "format": {"type": "text"},
        },
        {
            "type": "function",
            "name": "write_stdin",
            "description": "Poll or send input to a running exec_command session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "chars": {"type": "string"},
                    "yield_time_ms": {"type": "integer", "minimum": 100, "maximum": 300000},
                },
                "required": ["session_id", "chars", "yield_time_ms"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "view_image",
            "description": "Check whether a local image exists and return its dimensions.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]


def _dispatch_tool(
    session: ToolSession,
    name: str,
    raw_args: str,
    python_exe: Path,
    is_custom: bool = False,
) -> tuple[str, bool]:
    try:
        if name == "apply_patch":
            # Codex's apply_patch is a freeform custom tool: the body is the
            # patch text, not a JSON object.
            return _apply_patch_text(session, raw_args), False
        args = json.loads(raw_args or "{}")
        if not isinstance(args, dict):
            raise ValueError("tool arguments must be a JSON object")
        if name == "exec_command":
            return _tool_exec_command(session, args, python_exe), False
        if name == "write_stdin":
            return _tool_write_stdin(session, args), False
        if name == "view_image":
            path = _safe_workspace_path(session.root, str(args.get("path") or ""))
            if not path.is_file():
                raise ValueError(f"not a file: {args.get('path')}")
            with Image.open(path) as im:
                return f"{path.relative_to(session.root)}: {im.size[0]}x{im.size[1]} {im.mode}", False
        # Legacy tools kept for backward compatibility with older archives.
        if name == "write_file":
            return _tool_write_file(session, args), False
        if name == "read_file":
            return _tool_read_file(session, args), False
        if name == "list_files":
            return _tool_list_files(session, args), False
        if name == "run_python":
            return _tool_run_python(session, args, python_exe), False
        if name == "delete_file":
            return _tool_delete_file(session, args), False
        return f"unknown tool: {name}", True
    except Exception as exc:
        return f"tool error: {type(exc).__name__}: {exc}", True


def _active_run_hint(archive_root: Path, spec: RunSpec, run_dir: Path | None) -> str:
    """Report this run's own progress.

    Matching by model alone made every parallel job print the same run, which
    is why the console looked like only one job existed.
    """
    if run_dir is None or not run_dir.is_dir():
        return ""
    stream = run_dir / "response_stream.log"
    size = stream.stat().st_size if stream.exists() else 0
    outputs = run_dir / "outputs"
    files = len([p for p in outputs.iterdir() if p.is_file()]) if outputs.is_dir() else 0
    if size == 0 and files == 0:
        return ""
    return f"stream={size // 1024}KB files={files}"


def start_progress_reporter(
    label: str,
    stop: threading.Event,
    archive_root: Path,
    spec: RunSpec,
    print_lock: threading.Lock,
    run_dir_holder: list[Path | None],
) -> threading.Thread:
    def loop() -> None:
        last = ""
        while not stop.wait(20):
            hint = _active_run_hint(archive_root, spec, run_dir_holder[0])
            if hint and hint != last:
                with print_lock:
                    print(f"{label} -> {hint}")
                last = hint
    thread = threading.Thread(target=loop, name=f"progress-{spec.task.key}-{spec.group}", daemon=True)
    thread.start()
    return thread


_RETRYABLE_HTTP_STATUS = frozenset({403, 408, 409, 425, 429, 500, 502, 503, 504})
_RETRYABLE_RUNTIME_MARKERS = (
    "stream ended",
    "response incomplete",
    "response.failed",
    "upstream_error",
    "bad gateway",
    "gateway timeout",
    "service unavailable",
    "temporarily unavailable",
    "rate limit",
    "too many requests",
    "connection reset",
    "connection aborted",
    "remote end closed",
    "incomplete read",
    "timed out",
    "timeout",
)
_PROVIDER_REQUEST_SLOTS: dict[tuple[str, int], threading.BoundedSemaphore] = {}
_PROVIDER_REQUEST_SLOTS_LOCK = threading.Lock()


def _provider_request_slot(provider_id: str, limit: int) -> threading.BoundedSemaphore:
    """Return a shared per-provider request gate for one batch process."""
    key = (provider_id, max(1, int(limit)))
    with _PROVIDER_REQUEST_SLOTS_LOCK:
        slot = _PROVIDER_REQUEST_SLOTS.get(key)
        if slot is None:
            slot = threading.BoundedSemaphore(key[1])
            _PROVIDER_REQUEST_SLOTS[key] = slot
        return slot


def _http_error_body(exc: urllib.error.HTTPError) -> str:
    """Read an HTTPError body once and cache it for fallback/retry logging."""
    cached = getattr(exc, "_map_experiment_body", None)
    if cached is not None:
        return str(cached)
    try:
        body = exc.read().decode("utf-8", "replace")
    except Exception:
        body = ""
    setattr(exc, "_map_experiment_body", body)
    return body


def _http_error_headers(exc: urllib.error.HTTPError) -> dict[str, str]:
    headers = getattr(exc, "headers", None)
    if headers is None:
        return {}
    interesting = (
        "retry-after", "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests", "x-request-id", "cf-ray", "server", "date",
        "content-type",
    )
    result: dict[str, str] = {}
    for name in interesting:
        value = headers.get(name)
        if value:
            result[name] = str(value)
    return result


def _is_retryable_http_error(exc: urllib.error.HTTPError) -> bool:
    return int(exc.code) in _RETRYABLE_HTTP_STATUS


def _is_retryable_runtime_error(message: str) -> bool:
    text = (message or "").lower()
    return any(marker in text for marker in _RETRYABLE_RUNTIME_MARKERS)


def _retry_after_seconds(exc: BaseException | None) -> float | None:
    if isinstance(exc, urllib.error.HTTPError):
        headers = getattr(exc, "headers", None)
        value = headers.get("Retry-After") if headers is not None else None
        if value:
            try:
                return max(0.0, float(value))
            except ValueError:
                pass
    return None


def _retry_delay_seconds(
    exc: BaseException | None, attempt: int, base_sec: float
) -> float:
    """Exponential backoff with jitter; 403 gets a longer cooldown."""
    supplied = _retry_after_seconds(exc)
    if supplied is not None:
        return supplied
    status = int(exc.code) if isinstance(exc, urllib.error.HTTPError) else 0
    base = max(0.0, float(base_sec))
    if status == 403:
        base = max(base, 10.0)
    delay = min(120.0, base * (2 ** max(0, attempt)))
    return delay * random.uniform(0.85, 1.15)


def _record_request_retry(
    run_dir: Path,
    turn: int,
    retry_number: int,
    exc: BaseException,
    body: str,
    delay: float,
) -> None:
    record: dict[str, Any] = {
        "at": now_iso(),
        "turn": turn,
        "retry": retry_number,
        "delay_sec": round(delay, 2),
        "error_type": type(exc).__name__,
        "error": str(exc)[:2000],
        "body": (body or "")[:4000],
    }
    if isinstance(exc, urllib.error.HTTPError):
        record["status"] = int(exc.code)
        record["headers"] = _http_error_headers(exc)
    with (run_dir / "request_retries.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _prompt_cache_key(
    spec: RunSpec,
    reasoning_effort: str,
    backend: str,
) -> str:
    """Build a stable cache namespace shared by every group of one cell.

    The key deliberately excludes group, timestamp and run directory. Changing
    the frozen task prompt or model/reasoning configuration creates a new key.
    """
    prompt_hash = sha256_file(spec.task.prompt_path)
    material = "\0".join((
        "map-exp",
        "v1",
        spec.provider.id,
        spec.model,
        spec.task.key,
        reasoning_effort,
        backend,
        prompt_hash,
    ))
    digest = sha256_bytes(material.encode("utf-8"))[:20]
    return f"map-exp:v1:{spec.task.key}:{digest}"


def run_api_tools(
    spec: RunSpec,
    python_exe: Path,
    timeout_min: int,
    archive_root: Path,
    transport: str = "auto",
    reasoning_effort: str = "high",
    use_previous_response: bool = False,
    progress: Any = None,
    max_output_tokens: int = 12000,
    request_retries: int = 4,
    retry_base_sec: float = 5.0,
    provider_concurrency: int = 4,
) -> dict[str, Any]:
    """Run a provider-native tool loop with host-side, workspace-confined tools.

    This matches the interface's short-turn model behaviour without requiring
    the Windows restricted-token sandbox used by `codex exec`.
    """
    run_id = make_run_id(spec)
    run_dir = (archive_root / "_runs" / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs").mkdir(exist_ok=True)
    prompt = spec.task.prompt_path.read_text(encoding="utf-8-sig").rstrip() + delivery_block(
        spec.task, run_dir, "api"
    )
    prompt += """

You are working in the run workspace with the same tools as Codex:
exec_command, apply_patch, write_stdin, view_image.

Rules that keep every response short:
1. NEVER print the generator script in chat. Write files with apply_patch.
2. Build the script in small parts: outputs/_part01.py, _part02.py, ... with
   at most ~120 lines per apply_patch call. One part per turn.
3. Then concatenate with exec_command, e.g.
   Get-Content outputs/_part*.py | Set-Content -Encoding utf8 outputs/generate_map.py
4. Run it with exec_command, fix errors with small apply_patch hunks, and re-run.
5. When outputs/candidate.csv, outputs/candidate.png and outputs/candidate_8x.png
   exist and validate, stop immediately. Do not describe the whole method.
"""
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    if transport in ("", "auto"):
        if spec.provider.is_current:
            transport = "proxy"
        elif spec.provider.api_format == "openai_responses" and spec.provider.base_url and spec.provider.token:
            transport = "direct"
        else:
            transport = "proxy"
    url, headers = _responses_target(spec, transport)
    store_response, effective_previous_response, store_reason = _response_store_policy(
        transport, use_previous_response
    )
    if store_reason:
        (run_dir / "response_state_policy.txt").write_text(
            store_reason + "\n", encoding="utf-8"
        )
    cache_key = _prompt_cache_key(spec, reasoning_effort, "api-tools")
    (run_dir / "request_target.txt").write_text(
        f"transport: {transport}\nurl: {url}\nmodel: {spec.model}\n"
        f"backend: api-tools\nstore: {str(store_response).lower()}\n"
        f"previous_response: {str(effective_previous_response).lower()}\n"
        f"prompt_cache_key: {cache_key}\n"
        f"user_agent: {API_USER_AGENT}\n",
        encoding="utf-8",
    )
    session = ToolSession(root=run_dir, started_at=time.time())
    started = session.started_at

    def report(message: str) -> None:
        if progress is not None:
            try:
                progress(message)
            except Exception:
                pass
    status_code: int | None = None
    error = ""
    request_retry_count = 0
    final_text = ""
    events: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
    ]
    deadline = started + max(1, timeout_min) * 60
    body: dict[str, Any] = {
        "model": spec.model,
        "reasoning": {"effort": reasoning_effort},
        "tools": _api_tools(),
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "max_output_tokens": max_output_tokens,
        "stream": True,
        "store": store_response,
        "prompt_cache_key": cache_key,
    }
    try:
        turns = 0
        nudges = 0
        previous_response_id: str | None = None
        expected = (
            run_dir / "outputs" / "candidate.csv",
            run_dir / "outputs" / "candidate.png",
            run_dir / "outputs" / "candidate_8x.png",
        )

        def outputs_ready() -> bool:
            return all(x.is_file() and x.stat().st_size > 0 for x in expected)

        while time.time() < deadline:
            turns += 1
            if effective_previous_response and previous_response_id:
                body["input"] = []
                body["previous_response_id"] = previous_response_id
            else:
                body["input"] = history
                body.pop("previous_response_id", None)
            (run_dir / f"request_turn_{turns}.json").write_text(
                json.dumps(body, ensure_ascii=False), encoding="utf-8"
            )
            report(f"turn={turns} request")

            data: dict[str, Any] | None = None
            attempt = 0
            while True:
                try:
                    slot = _provider_request_slot(spec.provider.id, provider_concurrency)
                    with slot:
                        status_code, text, data, _ = _post_responses(
                            url, headers, body, run_dir, max(1, int((deadline - time.time()) / 60))
                        )
                    report(f"turn={turns} response_received")
                    break
                except urllib.error.HTTPError as exc:
                    body_text = _http_error_body(exc)
                    if _is_store_unsupported_error(exc.code, body_text):
                        store_response = False
                        effective_previous_response = False
                        previous_response_id = None
                        body["store"] = False
                        body["input"] = history
                        body.pop("previous_response_id", None)
                        (run_dir / "store_fallback.txt").write_text(
                            body_text, encoding="utf-8"
                        )
                        report(f"turn={turns} store_unsupported -> store=false, full history")
                        continue
                    if (
                        effective_previous_response
                        and previous_response_id
                        and exc.code == 400
                        and ("response_binding_not_found" in body_text
                             or "Previous response" in body_text)
                    ):
                        effective_previous_response = False
                        previous_response_id = None
                        (run_dir / "previous_response_fallback.txt").write_text(
                            body_text, encoding="utf-8"
                        )
                        body["input"] = history
                        body.pop("previous_response_id", None)
                        report(f"turn={turns} previous_response_unsupported -> full history")
                        continue
                    if not _is_retryable_http_error(exc) or attempt >= request_retries:
                        raise
                    delay = _retry_delay_seconds(exc, attempt, retry_base_sec)
                    if time.time() + delay >= deadline:
                        raise
                    attempt += 1
                    request_retry_count += 1
                    _record_request_retry(run_dir, turns, attempt, exc, body_text, delay)
                    (run_dir / f"retry_turn_{turns}_{attempt}.txt").write_text(
                        f"HTTP {exc.code}: {body_text or exc.reason}", encoding="utf-8"
                    )
                    report(
                        f"turn={turns} retry {attempt}/{request_retries} "
                        f"after {delay:.1f}s (HTTP {exc.code})"
                    )
                    time.sleep(delay)
                    continue
                except RuntimeError as exc:
                    message = str(exc)
                    if _is_store_unsupported_error(400, message):
                        store_response = False
                        effective_previous_response = False
                        previous_response_id = None
                        body["store"] = False
                        body["input"] = history
                        body.pop("previous_response_id", None)
                        (run_dir / "store_fallback.txt").write_text(
                            message, encoding="utf-8"
                        )
                        report(f"turn={turns} store_unsupported(stream) -> store=false, full history")
                        continue
                    if not _is_retryable_runtime_error(message) or attempt >= request_retries:
                        raise
                    delay = _retry_delay_seconds(None, attempt, retry_base_sec)
                    if time.time() + delay >= deadline:
                        raise
                    attempt += 1
                    request_retry_count += 1
                    _record_request_retry(run_dir, turns, attempt, exc, message, delay)
                    (run_dir / f"retry_turn_{turns}_{attempt}.txt").write_text(
                        message, encoding="utf-8"
                    )
                    if ("stream ended" in message.lower()
                            or "incomplete" in message.lower()):
                        history.append({
                            "role": "user",
                            "content": [{
                                "type": "input_text",
                                "text": (
                                    "Your previous response was cut off because it was too large. "
                                    "Retry that step and send a MUCH smaller chunk: at most 60 lines "
                                    "in one apply_patch call, then continue."
                                ),
                            }],
                        })
                        body["input"] = history
                        body.pop("previous_response_id", None)
                    report(
                        f"turn={turns} retry {attempt}/{request_retries} "
                        f"after {delay:.1f}s (stream error)"
                    )
                    time.sleep(delay)
                    continue
                except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                    if attempt >= request_retries:
                        raise
                    delay = _retry_delay_seconds(exc, attempt, retry_base_sec)
                    if time.time() + delay >= deadline:
                        raise
                    attempt += 1
                    request_retry_count += 1
                    _record_request_retry(run_dir, turns, attempt, exc, str(exc), delay)
                    report(
                        f"turn={turns} retry {attempt}/{request_retries} "
                        f"after {delay:.1f}s ({type(exc).__name__})"
                    )
                    time.sleep(delay)
                    continue
            if data is None:
                raise RuntimeError(f"turn {turns} produced no response")

            events.append(data)
            output_items = [
                item for item in (data.get("output") or [])
                if isinstance(item, dict)
            ]
            report(f"turn={turns} http={status_code} output_items={len(output_items)}")
            if text:
                final_text = text
                (run_dir / "last_message.txt").write_text(text + "\n", encoding="utf-8")
            if effective_previous_response and data.get("id"):
                previous_response_id = str(data["id"])
            calls = [
                item for item in output_items
                if item.get("type") in ("function_call", "custom_tool_call")
            ]
            report(f"turn={turns} tool_calls={len(calls)}")

            if not calls:
                if outputs_ready() or session.call_count >= 80:
                    break
                if nudges >= 6:
                    error = "model never called a tool"
                    break
                nudges += 1
                (run_dir / f"nudge_{nudges}.txt").write_text(
                    "model returned no tool call", encoding="utf-8"
                )
                history.extend(output_items)
                history.append({
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": (
                            "You must call a tool now. Do not answer with text. "
                            "Call apply_patch to add the next small part "
                            "(at most 60 lines) or exec_command to run/validate."
                        ),
                    }],
                })
                body["input"] = history
                body.pop("previous_response_id", None)
                report(f"turn={turns} nudge={nudges} (no tool call)")
                continue

            history.extend(output_items)
            new_items: list[dict[str, Any]] = []
            for item in calls:
                name = str(item.get("name") or "")
                is_custom = item.get("type") == "custom_tool_call"
                raw_args = str(item.get("input") if is_custom else item.get("arguments") or "{}")
                output, failed = _dispatch_tool(session, name, raw_args, python_exe, is_custom)
                new_items.append({
                    "type": "custom_tool_call_output" if is_custom else "function_call_output",
                    "call_id": str(item.get("call_id") or item.get("id") or ""),
                    "output": output,
                })
                events.append({
                    "tool": name,
                    "args": raw_args[:4000],
                    "output": output[:4000],
                    "failed": failed,
                })
                report(f"turn={turns} tool={name} {'failed' if failed else 'ok'}")
            history.extend(new_items)
            if effective_previous_response:
                body["input"] = new_items
            if outputs_ready() or session.call_count >= 80:
                break
    except urllib.error.HTTPError as exc:
        status_code = int(exc.code)
        try:
            raw = exc.read().decode("utf-8", "replace")
        except Exception:
            raw = ""
        error = f"HTTPError {exc.code}: {raw[:2000] or exc.reason}"
        (run_dir / "response_error.txt").write_text(
            f"status={exc.code}\nurl={url}\nreason={exc.reason}\nbody={raw}\n",
            encoding="utf-8",
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    duration = time.time() - started
    (run_dir / "tool_events.json").write_text(
        json.dumps(events, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    truth = load_truth(spec.task)
    candidate = find_candidate(run_dir, spec.task)
    recovery: dict[str, Any] | None = None
    if candidate is None:
        try:
            recovery = recover_artifacts(run_dir, spec.task, python_exe)
            candidate = find_candidate(run_dir, spec.task)
        except Exception as exc:
            recovery = {"attempted": True, "error": f"{type(exc).__name__}: {exc}"}
    score: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    quality: dict[str, Any] | None = None
    if candidate is not None:
        score = score_matrix(candidate.matrix, truth)
        candidate.meta["orientation_ok"] = bool(score.get("orientation_ok", False))
        validation = validate_candidate(candidate, spec.task, truth, score)
        try:
            quality = map_quality_metrics(candidate, truth)
        except Exception as exc:
            quality = {"error": f"{type(exc).__name__}: {exc}"}
    else:
        error = error or "no candidate CSV/PNG found"
    status = "ok" if (
        status_code == 200
        and candidate is not None
        and validation is not None
        and validation.get("status") == "ok"
        and not error
    ) else "partial"
    if candidate is None:
        status = "failed"
    result: dict[str, Any] = {
        "run_id": run_id,
        "started_at": datetime.fromtimestamp(started).astimezone().isoformat(timespec="seconds"),
        "finished_at": now_iso(),
        "duration_sec": round(duration, 1),
        "status": status,
        "returncode": status_code,
        "timed_out": False,
        "early_complete": False,
        "error": error,
        "provider_id": spec.provider.id,
        "provider_name": spec.provider.name,
        "provider_display": display_provider_name(spec.provider),
        "provider_alias": spec.provider.alias,
        "provider_mode": transport,
        "transport": transport,
        "backend": "api-tools",
        "prompt_cache_key": cache_key,
        "tool_calls": session.call_count,
        "request_retries": request_retry_count,
        "model": spec.model,
        "model_display": display_model_name(spec.model, spec.provider),
        "task": spec.task.key,
        "task_label": spec.task.label,
        "group": spec.group,
        "archive_stem": spec.archive_stem,
        "prompt_sha256": sha256_file(spec.task.prompt_path),
        "run_dir": str(run_dir),
        "candidate": None,
        "score": None,
        "validation": validation,
        "map_metrics": quality,
        "recovery": recovery,
    }
    if candidate is not None:
        result["candidate"] = {
            "kind": candidate.kind,
            "how": candidate.how,
            "source": str(candidate.source) if candidate.source else None,
            "csv": str(candidate.csv_path) if candidate.csv_path else None,
            "upscale": str(candidate.upscale_path) if candidate.upscale_path else None,
            "csv_match": bool(candidate.csv_match),
            "upscale_match": bool(candidate.upscale_match),
            "sha256": sha256_file(candidate.source) if candidate.source else None,
            "meta": candidate.meta,
        }
        result["score"] = {k: v for k, v in score.items() if k != "diff"} if score else None
    write_json(run_dir / "result.json", result)
    return result


# ---------------------------------------------------------------------------
# Direct Responses-API backend (through the CC Switch local proxy)
# ---------------------------------------------------------------------------

def _extract_response_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return str(data["output_text"])
    parts: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for content in item.get("content") or []:
                if isinstance(content, dict) and content.get("type") in ("output_text", "text"):
                    text = content.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        elif item.get("type") in ("output_text", "text") and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "\n".join(parts).strip()


def _responses_target(spec: RunSpec, transport: str) -> tuple[str, dict[str, str]]:
    if transport == "direct":
        base = spec.provider.base_url.rstrip("/")
        url = base + "/responses" if base.endswith("/v1") else base + "/v1/responses"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": API_USER_AGENT,
        }
        if spec.provider.token:
            headers["Authorization"] = "Bearer " + spec.provider.token
        return url, headers
    return DEFAULT_PROXY.rstrip("/") + "/responses", {
        "Authorization": "Bearer PROXY_MANAGED",
        "Content-Type": "application/json",
        "User-Agent": API_USER_AGENT,
    }


def _response_store_policy(
    transport: str, use_previous_response: bool
) -> tuple[bool, bool, str]:
    """Return (store, previous_response_enabled, reason).

    The CC Switch proxy is stateless: it rejects ``store: true`` with
    "Each response request is independent".  Only ask for stored responses on
    a direct Responses endpoint and only when previous_response_id is requested.
    """
    if transport == "proxy":
        return False, False, "proxy responses are independent; store=false"
    if use_previous_response:
        return True, True, ""
    return False, False, ""


def _is_store_unsupported_error(status: int, body: str) -> bool:
    text = (body or "").lower()
    return bool(
        status == 400
        and "store" in text
        and "not supported" in text
        and ("true" in text or "proxy" in text)
    )


def _post_responses(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    run_dir: Path,
    timeout_min: int,
) -> tuple[int, str, dict[str, Any], float]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=max(60, timeout_min * 60)) as resp:
        status = int(getattr(resp, "status", 200))
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "application/json" in ctype and "event-stream" not in ctype:
            raw = resp.read().decode("utf-8", "replace")
            (run_dir / "response.json").write_text(raw, encoding="utf-8")
            data = json.loads(raw)
            return status, _extract_response_text(data), data, time.time() - started

        deltas: list[str] = []
        final: dict[str, Any] | None = None
        stream_path = run_dir / "response_stream.log"
        with stream_path.open("w", encoding="utf-8") as log:
            for raw_line in resp:
                line = raw_line.decode("utf-8", "replace").rstrip("\r\n")
                log.write(line + "\n")
                log.flush()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                etype = str(event.get("type", ""))
                if etype == "response.output_text.delta":
                    delta = event.get("delta")
                    if isinstance(delta, str) and delta:
                        deltas.append(delta)
                elif etype in ("response.completed", "response.done"):
                    value = event.get("response")
                    if isinstance(value, dict):
                        final = value
                elif etype in ("response.failed", "error"):
                    err = event.get("error") or event
                    raise RuntimeError(json.dumps(err, ensure_ascii=False)[:2000])
        text = _extract_response_text(final or {}) or "".join(deltas)
        if final is None:
            raise RuntimeError("stream ended before response.completed")
        final_status = str(final.get("status", ""))
        if final_status and final_status != "completed":
            details = final.get("incomplete_details") or final.get("error") or {}
            raise RuntimeError(f"response {final_status}: {json.dumps(details, ensure_ascii=False)[:1000]}")
        data = final
        (run_dir / "response.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return status, text, data, time.time() - started


def run_api(
    spec: RunSpec,
    python_exe: Path,
    timeout_min: int,
    archive_root: Path,
    transport: str = "auto",
    reasoning_effort: str = "high",
) -> dict[str, Any]:
    run_id = make_run_id(spec)
    run_dir = (archive_root / "_runs" / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs").mkdir(exist_ok=True)
    prompt = spec.task.prompt_path.read_text(encoding="utf-8-sig").rstrip() + delivery_block(spec.task, run_dir, "api")
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    if transport in ("", "auto"):
        if spec.provider.is_current:
            transport = "proxy"
        elif spec.provider.api_format == "openai_responses" and spec.provider.base_url and spec.provider.token:
            transport = "direct"
        else:
            transport = "proxy"
    url, headers = _responses_target(spec, transport)
    cache_key = _prompt_cache_key(spec, reasoning_effort, "api")
    (run_dir / "request_target.txt").write_text(
        f"transport: {transport}\nurl: {url}\nmodel: {spec.model}\n"
        f"prompt_cache_key: {cache_key}\n",
        encoding="utf-8",
    )
    started = time.time()
    error = ""
    status_code: int | None = None
    body: dict[str, Any] = {
        "model": spec.model,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "reasoning": {"effort": reasoning_effort},
        "tools": [],
        "max_output_tokens": 30000,
        "stream": True,
        "prompt_cache_key": cache_key,
    }

    def request_once(text: str) -> tuple[int | None, str]:
        body["input"] = [{"role": "user", "content": [{"type": "input_text", "text": text}]}]
        last_error = ""
        last_status: int | None = None
        for attempt in range(2):
            try:
                status, answer, data, elapsed = _post_responses(
                    url, headers, body, run_dir, timeout_min
                )
                last_status = status
                (run_dir / "last_message.txt").write_text(answer + "\n", encoding="utf-8")
                try:
                    (run_dir / "response_error.txt").unlink(missing_ok=True)
                except Exception:
                    pass
                if data.get("error"):
                    last_error = json.dumps(data["error"], ensure_ascii=False)[:2000]
                if not answer.strip() and not last_error:
                    last_error = "empty response"
                return last_status, last_error
            except urllib.error.HTTPError as exc:
                last_status = int(exc.code)
                try:
                    raw_err = exc.read().decode("utf-8", "replace")
                except Exception:
                    raw_err = ""
                details = (
                    f"status={exc.code}\n"
                    f"url={url}\n"
                    f"reason={exc.reason}\n"
                    f"content-type={exc.headers.get('Content-Type', '') if exc.headers else ''}\n"
                    f"body={raw_err}\n"
                )
                try:
                    (run_dir / "response_error.txt").write_text(details, encoding="utf-8")
                except Exception:
                    pass
                last_error = f"HTTPError {exc.code}: {raw_err[:2000] or exc.reason}"
                if exc.code in (429, 503) and attempt == 0:
                    time.sleep(5)
                    continue
                return last_status, last_error
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                no_retry = (
                    isinstance(exc, TimeoutError)
                    or "stream ended" in last_error
                    or "response incomplete" in last_error
                    or "response failed" in last_error
                )
                if attempt == 0 and not no_retry:
                    time.sleep(3)
                    continue
                return last_status, last_error
        return last_status, last_error

    status_code, error = request_once(prompt)
    duration = time.time() - started

    truth = load_truth(spec.task)
    candidate = find_candidate(run_dir, spec.task)
    recovery: dict[str, Any] | None = None
    if candidate is None:
        try:
            recovery = recover_artifacts(run_dir, spec.task, python_exe)
            candidate = find_candidate(run_dir, spec.task)
        except Exception as exc:
            recovery = {"attempted": True, "error": f"{type(exc).__name__}: {exc}"}
    if candidate is None:
        retry_prompt = prompt + """
\n【重要】上一次回答没有给出可执行的 Python 脚本。请只输出一个完整的 ```python 代码块，不要解释，不要调用工具。脚本必须自行创建 outputs/ 并写出 candidate.csv、candidate.png、candidate_8x.png。\n"""
        (run_dir / "retry_prompt.txt").write_text(retry_prompt, encoding="utf-8")
        status_code, error = request_once(retry_prompt)
        try:
            recovery = recover_artifacts(run_dir, spec.task, python_exe)
            candidate = find_candidate(run_dir, spec.task)
        except Exception as exc:
            recovery = {"attempted": True, "retry": True, "error": f"{type(exc).__name__}: {exc}"}
    score: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    quality: dict[str, Any] | None = None
    if candidate is not None:
        score = score_matrix(candidate.matrix, truth)
        candidate.meta["orientation_ok"] = bool(score.get("orientation_ok", False))
        validation = validate_candidate(candidate, spec.task, truth, score)
        try:
            quality = map_quality_metrics(candidate, truth)
        except Exception as exc:
            quality = {"error": f"{type(exc).__name__}: {exc}"}
    else:
        error = error or "no candidate CSV/PNG found after extracting the response"

    status = "ok" if (
        status_code == 200
        and candidate is not None
        and validation is not None
        and validation.get("status") == "ok"
        and not error
    ) else "partial"
    if candidate is None:
        status = "failed"
    result: dict[str, Any] = {
        "run_id": run_id,
        "started_at": datetime.fromtimestamp(started).astimezone().isoformat(timespec="seconds"),
        "finished_at": now_iso(),
        "duration_sec": round(duration, 1),
        "status": status,
        "returncode": status_code,
        "timed_out": False,
        "error": error,
        "provider_id": spec.provider.id,
        "provider_name": spec.provider.name,
        "provider_display": display_provider_name(spec.provider),
        "provider_alias": spec.provider.alias,
        "provider_mode": transport,
        "transport": transport,
        "reasoning_effort": reasoning_effort,
        "model": spec.model,
        "model_display": display_model_name(spec.model, spec.provider),
        "task": spec.task.key,
        "task_label": spec.task.label,
        "group": spec.group,
        "archive_stem": spec.archive_stem,
        "prompt_sha256": sha256_file(spec.task.prompt_path),
        "run_dir": str(run_dir),
        "candidate": None,
        "score": None,
        "validation": validation,
        "map_metrics": quality,
        "recovery": recovery,
        "backend": "api",
        "prompt_cache_key": cache_key,
    }
    if candidate is not None:
        result["candidate"] = {
            "kind": candidate.kind,
            "how": candidate.how,
            "source": str(candidate.source) if candidate.source else None,
            "csv": str(candidate.csv_path) if candidate.csv_path else None,
            "upscale": str(candidate.upscale_path) if candidate.upscale_path else None,
            "csv_match": bool(candidate.csv_match),
            "upscale_match": bool(candidate.upscale_match),
            "sha256": sha256_file(candidate.source) if candidate.source else None,
            "meta": candidate.meta,
        }
        result["score"] = {k: v for k, v in score.items() if k != "diff"} if score else None
    write_json(run_dir / "result.json", result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

