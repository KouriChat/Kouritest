#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the optional Flask relay.

Flask is not installed in this workspace, so the module is imported against a
tiny stand-in that records registered routes. Run:

    python webapp/tests/server.test.py
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

WEBAPP = Path(__file__).resolve().parents[1]

# Minimal Flask stand-ins so server.py can be imported without the dependency.
flask = types.ModuleType("flask")


class _App:
    def __init__(self, *a, **k):
        self.routes = {}

    def after_request(self, fn):
        return fn

    def _register(self, method, rule):
        def deco(fn):
            self.routes[(method, rule)] = fn
            return fn
        return deco

    def get(self, rule):
        return self._register("GET", rule)

    def post(self, rule):
        return self._register("POST", rule)

    def options(self, rule):
        return self._register("OPTIONS", rule)


flask.Flask = _App
flask.jsonify = lambda *a, **k: ("json", a, k)
flask.request = types.SimpleNamespace(get_json=lambda **k: None)
flask.send_from_directory = lambda *a, **k: "file"
sys.modules["flask"] = flask

sys.path.insert(0, str(WEBAPP))
server = importlib.import_module("server")

fails = 0


def check(label, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + label + ((" :: " + str(extra)) if extra else ""))
    if not ok:
        fails += 1


def _try(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as exc:  # noqa: BLE001 - we assert on the error type
        return exc


check("chat url", server._join_url("https://x.test/v1", "chat") == "https://x.test/v1/chat/completions")
check("responses url", server._join_url("https://x.test/v1", "responses") == "https://x.test/v1/responses")
check("full url kept", server._join_url("https://x.test/v1/chat/completions", "chat") == "https://x.test/v1/chat/completions")
check("rejects non-http", "ValueError" in repr(_try(server._join_url, "ftp://x", "chat")))
check("rejects empty base", "ValueError" in repr(_try(server._join_url, "", "chat")))

wire, body = server._build_body(
    {"model": "m", "wire": "chat", "maxTokens": 100, "tokenField": "max_completion_tokens"}, "p"
)
check("chat body wires max_completion_tokens",
      wire == "chat" and body["max_completion_tokens"] == 100 and body["messages"][1]["content"] == "p")
check("chat body omits max_tokens", "max_tokens" not in body)

wire_default, body_default = server._build_body({"model": "m", "wire": "chat", "maxTokens": 10}, "p")
check("chat body defaults to max_tokens", body_default["max_tokens"] == 10)

wire2, body2 = server._build_body(
    {"model": "m", "wire": "responses", "maxTokens": 50, "reasoning": "high"}, "p"
)
check("responses body", wire2 == "responses" and body2["max_output_tokens"] == 50
      and body2["reasoning"] == {"effort": "high"} and body2["input"][0]["content"][0]["type"] == "input_text")
check("missing model rejected", "ValueError" in repr(_try(server._build_body, {"wire": "chat"}, "p")))

check("chat text extract", server._extract_chat_text({"choices": [{"message": {"content": "abc"}}]}) == "abc")
check("chat text extract (list)", server._extract_chat_text(
    {"choices": [{"message": {"content": [{"text": "a"}, {"text": "b"}]}}]}) == "ab")
check("responses text extract", server._extract_responses_text(
    {"output": [{"type": "message", "content": [{"text": "xyz"}]}]}) == "xyz")
check("responses text extract (output_text)",
      server._extract_responses_text({"output_text": "direct"}) == "direct")

check("allow-list excludes server.py", "server.py" not in server.ALLOWED_STATIC)
check("allow-list excludes tests", "tests/smoke.e2e.js" not in server.ALLOWED_STATIC)
check("allow-list includes page", set(server.ALLOWED_STATIC) >=
      {"index.html", "app.css", "js/app.js", "js/assets-data.js"})

app = server.create_app()
check("proxy route registered", ("POST", "/api/proxy") in app.routes)
check("health route registered", ("GET", "/api/health") in app.routes)
check("index route registered", ("GET", "/") in app.routes)

print("all checks passed" if not fails else f"{fails} check(s) failed")
sys.exit(1 if fails else 0)