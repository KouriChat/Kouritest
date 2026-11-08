#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional local relay for the map-evaluation web page.

The page works without any backend when the provider allows browser CORS.
This Flask app exists only to (a) serve the static page and (b) relay requests
that the browser cannot send directly because of CORS. It binds to loopback by
default and never stores the API key.

Usage:
    python webapp/server.py
    python webapp/server.py --port 8788 --no-browser

Requires: Flask (pip install flask)
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

try:
    from flask import Flask, jsonify, request, send_from_directory
except ImportError as exc:  # pragma: no cover - message only
    raise SystemExit(
        "Flask is required for the optional relay.\n"
        "Install it with:  pip install flask\n"
        "The page itself still works from file:// if the provider allows CORS."
    ) from exc

ROOT = Path(__file__).resolve().parent
DEFAULT_TIMEOUT = 20 * 60


def _join_url(base: str, wire: str) -> str:
    base = (base or "").strip().rstrip("/")
    if not base:
        raise ValueError("baseUrl is required")
    if not (base.startswith("http://") or base.startswith("https://")):
        raise ValueError("baseUrl must start with http:// or https://")
    if base.endswith("/responses") or base.endswith("/chat/completions"):
        return base
    return base + ("/responses" if wire == "responses" else "/chat/completions")


def _build_body(payload: dict[str, Any], prompt: str) -> tuple[str, dict[str, Any]]:
    wire = "responses" if payload.get("wire") == "responses" else "chat"
    model = str(payload.get("model") or "").strip()
    if not model:
        raise ValueError("model is required")
    max_tokens = int(payload.get("maxTokens") or 30000)
    reasoning = str(payload.get("reasoning") or "").strip()

    if wire == "responses":
        body: dict[str, Any] = {
            "model": model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "max_output_tokens": max_tokens,
            "stream": False,
        }
        if reasoning:
            body["reasoning"] = {"effort": reasoning}
    else:
        body = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是严谨的地理知识地图生成助手。严格按用户要求输出，不要联网或读取外部地图。",
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        token_field = str(payload.get("tokenField") or "max_tokens")
        if token_field not in ("max_tokens", "max_completion_tokens"):
            token_field = "max_tokens"
        body[token_field] = max_tokens
        if reasoning:
            body["reasoning_effort"] = reasoning
    return wire, body


def _extract_chat_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    choice = choices[0] or {}
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
        return "".join(parts)
    return str(choice.get("text") or "")


def _extract_responses_text(data: dict[str, Any]) -> str:
    direct = data.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    parts: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict):
                text = content.get("text") or content.get("output_text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)


def _post_json(url: str, headers: dict[str, str], body: dict[str, Any], timeout: int) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8", "replace")
        status = int(getattr(response, "status", 200))
        return status, json.loads(raw) if raw.strip() else {}


ALLOWED_STATIC = (
    "index.html",
    "app.css",
    "README.md",
    "js/api.js",
    "js/app.js",
    "js/assets-data.js",
    "js/image.js",
    "js/render.js",
    "js/score.js",
    "js/tasks.js",
)


def create_app() -> Flask:
    # static_folder=None: assets are served from an explicit allow-list so that
    # tests/ and this file are never reachable over HTTP.
    app = Flask(__name__, static_folder=None)

    @app.after_request
    def add_cors(response):  # noqa: ANN001
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return send_from_directory(str(ROOT), "index.html")

    @app.get("/<path:filename>")
    def static_files(filename: str):
        if filename not in ALLOWED_STATIC:
            return jsonify({"error": "not found"}), 404
        return send_from_directory(str(ROOT), filename)

    @app.options("/api/proxy")
    def proxy_options():
        return ("", 204)

    @app.post("/api/proxy")
    def proxy():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON body required"}), 400
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return jsonify({"error": "prompt is required"}), 400
        try:
            wire, body = _build_body(payload, prompt)
            url = _join_url(str(payload.get("baseUrl") or ""), wire)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "codex-map-webapp/1.0",
        }
        key = str(payload.get("apiKey") or "").strip()
        if key:
            headers["Authorization"] = "Bearer " + key
        timeout = int(payload.get("timeoutSec") or DEFAULT_TIMEOUT)
        started = time.time()
        try:
            status, data = _post_json(url, headers, body, max(30, timeout))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            try:
                detail = json.loads(raw)
            except Exception:
                detail = raw[:4000]
            return jsonify({"error": f"HTTP {exc.code}", "detail": detail}), 502
        except Exception as exc:  # noqa: BLE001 - relay reports upstream failure
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 502

        text = _extract_responses_text(data) if wire == "responses" else _extract_chat_text(data)
        if not text:
            upstream_error = data.get("error") if isinstance(data, dict) else None
            if upstream_error:
                return jsonify({"error": str(upstream_error)}), 502
        return jsonify({
            "text": text,
            "raw": data,
            "usage": data.get("usage") if isinstance(data, dict) else None,
            "elapsedSec": round(time.time() - started, 3),
            "upstreamStatus": status,
        })

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the map-evaluation page and optional CORS relay.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"Map evaluation web page: {url}")
    print("Relay endpoint: /api/proxy   (Ctrl+C to stop)")
    create_app().run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()