from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import run_map_experiments as exp

ROOT = Path(__file__).resolve().parents[1]


class ResponseStorePolicyTests(unittest.TestCase):
    def test_proxy_never_requests_stored_responses(self):
        store, previous, reason = exp._response_store_policy("proxy", True)
        self.assertFalse(store)
        self.assertFalse(previous)
        self.assertIn("independent", reason)

    def test_direct_store_only_for_previous_response(self):
        self.assertEqual(exp._response_store_policy("direct", False), (False, False, ""))
        self.assertEqual(exp._response_store_policy("direct", True), (True, True, ""))

    def test_detects_exact_proxy_error(self):
        body = "store: true is not supported on this proxy. Each response request is independent."
        self.assertTrue(exp._is_store_unsupported_error(400, body))
        self.assertFalse(exp._is_store_unsupported_error(400, "invalid model"))
        self.assertFalse(exp._is_store_unsupported_error(500, body))

    def test_responses_requests_use_codex_user_agent(self):
        task = exp.load_tasks(exp.DEFAULT_TASKS)["hecate"]
        provider = exp.Provider(
            id="local-provider", name="Local Provider", app_type="codex", is_current=False,
            model="gpt-6.1-sol", base_url="https://example.invalid", wire_api="responses",
            requires_openai_auth=True, token="token", auth={},
            api_format="openai_responses", model_catalog=None,
            config_text="", meta={}, alias="ah",
        )
        spec = exp.RunSpec(
            provider=provider, model="gpt-6.1-sol", task=task, group=1,
            archive_stem="gpt-6.1-sol_hec@ah", archive_dir=Path("."), mode="direct",
        )
        _, direct = exp._responses_target(spec, "direct")
        self.assertEqual(direct["User-Agent"], exp.API_USER_AGENT)
        self.assertNotIn("Python-urllib", direct["User-Agent"])
        _, proxy = exp._responses_target(spec, "proxy")
        self.assertEqual(proxy["User-Agent"], exp.API_USER_AGENT)

    def test_prompt_cache_key_is_stable_across_groups(self):
        task = exp.load_tasks(exp.DEFAULT_TASKS)["hecate"]
        provider = exp.Provider(
            id="unit", name="unit", app_type="codex", is_current=False,
            model="unit", base_url="https://example.invalid/v1", wire_api="responses",
            requires_openai_auth=False, token="token", auth={},
            api_format="openai_responses", model_catalog=None,
            config_text="", meta={}, alias="",
        )

        def key(group: int) -> str:
            spec = exp.RunSpec(
                provider=provider, model="unit", task=task, group=group,
                archive_stem=f"unit_hec#{group}", archive_dir=Path("."), mode="direct",
            )
            return exp._prompt_cache_key(spec, "high", "api-tools")

        first = key(1)
        self.assertEqual(first, key(8))
        self.assertLessEqual(len(first), 64)
        self.assertTrue(first.startswith("map-exp:v1:hecate:"))

    def test_proxy_tool_request_has_store_false(self):
        task = exp.load_tasks(exp.DEFAULT_TASKS)["hecate"]
        truth = exp.load_truth(task)
        seen: list[dict] = []

        def fake_post(url, headers, body, run_dir, timeout_min):
            seen.append(json.loads(json.dumps(body)))
            out = Path(run_dir) / "outputs"
            out.mkdir(parents=True, exist_ok=True)
            np.savetxt(out / "candidate.csv", truth, fmt="%d", delimiter=",")
            Image.fromarray(truth * 255).save(out / "candidate.png")
            Image.fromarray(truth * 255).resize(task.upscale, Image.Resampling.NEAREST).save(
                out / "candidate_8x.png"
            )
            return 200, "", {"id": "fake", "output": []}, 0.0

        provider = exp.Provider(
            id="unit", name="unit", app_type="codex", is_current=True,
            model="unit", base_url="", wire_api="responses",
            requires_openai_auth=False, token="", auth={},
            api_format="openai_responses", model_catalog=None,
            config_text="", meta={}, alias="",
        )
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
            spec = exp.RunSpec(
                provider=provider, model="unit", task=task, group=1,
                archive_stem="unit_hec", archive_dir=Path(tmp), mode="proxy",
            )
            with mock.patch.object(exp, "_post_responses", side_effect=fake_post):
                result = exp.run_api_tools(
                    spec, Path(__file__).resolve(), 1, Path(tmp),
                    transport="proxy", use_previous_response=True,
                )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(seen[0]["store"], False)
            self.assertNotIn("previous_response_id", seen[0])
            self.assertTrue(seen[0]["prompt_cache_key"].startswith("map-exp:v1:hecate:"))
            target = (Path(tmp) / "_runs" / result["run_id"] / "request_target.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("store: false", target)
            self.assertIn("previous_response: false", target)
            self.assertIn("prompt_cache_key: map-exp:v1:hecate:", target)

    def test_retries_403_then_succeeds(self):
        task = exp.load_tasks(exp.DEFAULT_TASKS)["hecate"]
        truth = exp.load_truth(task)
        seen: list[dict] = []

        def fake_post(url, headers, body, run_dir, timeout_min):
            seen.append(json.loads(json.dumps(body)))
            if len(seen) == 1:
                raise urllib.error.HTTPError(
                    url, 403, "Forbidden", {"Retry-After": "0"}, io.BytesIO(b"")
                )
            out = Path(run_dir) / "outputs"
            out.mkdir(parents=True, exist_ok=True)
            np.savetxt(out / "candidate.csv", truth, fmt="%d", delimiter=",")
            Image.fromarray(truth * 255).save(out / "candidate.png")
            Image.fromarray(truth * 255).resize(task.upscale, Image.Resampling.NEAREST).save(
                out / "candidate_8x.png"
            )
            return 200, "", {"id": "fake", "output": []}, 0.0

        provider = exp.Provider(
            id="unit", name="unit", app_type="codex", is_current=False,
            model="unit", base_url="https://example.invalid/v1", wire_api="responses",
            requires_openai_auth=False, token="token", auth={},
            api_format="openai_responses", model_catalog=None,
            config_text="", meta={}, alias="",
        )
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
            spec = exp.RunSpec(
                provider=provider, model="unit", task=task, group=1,
                archive_stem="unit_hec", archive_dir=Path(tmp), mode="direct",
            )
            with mock.patch.object(exp, "_post_responses", side_effect=fake_post):
                with mock.patch.object(exp, "_retry_delay_seconds", return_value=0.0):
                    result = exp.run_api_tools(
                        spec, Path(__file__).resolve(), 1, Path(tmp),
                        transport="direct", request_retries=2,
                    )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(len(seen), 2)
            self.assertEqual(result["request_retries"], 1)
            retry_log = Path(tmp) / "_runs" / result["run_id"] / "request_retries.jsonl"
            records = [json.loads(line) for line in retry_log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[0]["status"], 403)
            self.assertEqual(records[0]["retry"], 1)

    def test_adaptive_fallback_after_stream_error(self):
        task = exp.load_tasks(exp.DEFAULT_TASKS)["hecate"]
        truth = exp.load_truth(task)
        seen: list[dict] = []

        def fake_post(url, headers, body, run_dir, timeout_min):
            seen.append(json.loads(json.dumps(body)))
            if len(seen) == 1:
                raise RuntimeError(
                    '{"error":{"message":"store: true is not supported on this proxy. '
                    'Each response request is independent."}}'
                )
            out = Path(run_dir) / "outputs"
            out.mkdir(parents=True, exist_ok=True)
            np.savetxt(out / "candidate.csv", truth, fmt="%d", delimiter=",")
            Image.fromarray(truth * 255).save(out / "candidate.png")
            Image.fromarray(truth * 255).resize(task.upscale, Image.Resampling.NEAREST).save(
                out / "candidate_8x.png"
            )
            return 200, "", {"id": "fake", "output": []}, 0.0

        provider = exp.Provider(
            id="unit", name="unit", app_type="codex", is_current=False,
            model="unit", base_url="https://example.invalid/v1", wire_api="responses",
            requires_openai_auth=False, token="token", auth={},
            api_format="openai_responses", model_catalog=None,
            config_text="", meta={}, alias="",
        )
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
            spec = exp.RunSpec(
                provider=provider, model="unit", task=task, group=1,
                archive_stem="unit_hec", archive_dir=Path(tmp), mode="direct",
            )
            with mock.patch.object(exp, "_post_responses", side_effect=fake_post):
                result = exp.run_api_tools(
                    spec, Path(__file__).resolve(), 1, Path(tmp),
                    transport="direct", use_previous_response=True,
                )
            self.assertEqual(result["status"], "ok")
            self.assertTrue(seen[0]["store"])
            self.assertFalse(seen[1]["store"])

    def test_adaptive_fallback_after_proxy_rejects_store_true(self):
        task = exp.load_tasks(exp.DEFAULT_TASKS)["hecate"]
        truth = exp.load_truth(task)
        seen: list[dict] = []

        def fake_post(url, headers, body, run_dir, timeout_min):
            seen.append(json.loads(json.dumps(body)))
            if len(seen) == 1:
                msg = "store: true is not supported on this proxy. Each response request is independent."
                raise urllib.error.HTTPError(
                    url, 400, "Bad Request", {}, io.BytesIO(msg.encode("utf-8"))
                )
            out = Path(run_dir) / "outputs"
            out.mkdir(parents=True, exist_ok=True)
            np.savetxt(out / "candidate.csv", truth, fmt="%d", delimiter=",")
            Image.fromarray(truth * 255).save(out / "candidate.png")
            Image.fromarray(truth * 255).resize(task.upscale, Image.Resampling.NEAREST).save(
                out / "candidate_8x.png"
            )
            return 200, "", {"id": "fake", "output": []}, 0.0

        provider = exp.Provider(
            id="unit", name="unit", app_type="codex", is_current=False,
            model="unit", base_url="https://example.invalid/v1", wire_api="responses",
            requires_openai_auth=False, token="token", auth={},
            api_format="openai_responses", model_catalog=None,
            config_text="", meta={}, alias="",
        )
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
            spec = exp.RunSpec(
                provider=provider, model="unit", task=task, group=1,
                archive_stem="unit_hec", archive_dir=Path(tmp), mode="direct",
            )
            with mock.patch.object(exp, "_post_responses", side_effect=fake_post):
                result = exp.run_api_tools(
                    spec, Path(__file__).resolve(), 1, Path(tmp),
                    transport="direct", use_previous_response=True,
                )
            self.assertEqual(result["status"], "ok")
            self.assertTrue(seen[0]["store"])
            self.assertFalse(seen[1]["store"])
            fallback = Path(tmp) / "_runs" / result["run_id"] / "store_fallback.txt"
            self.assertTrue(fallback.is_file())


if __name__ == "__main__":
    unittest.main()
