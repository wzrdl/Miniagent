"""
Lightweight HTTP server that exposes registered tools.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from agent_rl.tools.registry import ToolRegistry


class _ToolRequestHandler(BaseHTTPRequestHandler):
    registry: ToolRegistry | None = None

    def do_POST(self) -> None:  # noqa: N802
        if not self.registry:
            self.send_error(500, "Tool registry not configured.")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        payload = self.rfile.read(content_length).decode("utf-8")
        data = json.loads(payload)
        tool_name = data.get("name")
        arguments = data.get("arguments", {})

        try:
            tool = self.registry.get(tool_name)
            result = tool(**arguments)
            response = {"status": "ok", "result": result}
            self._send_response(200, response)
        except Exception as exc:  # pragma: no cover - protective
            self._send_response(500, {"status": "error", "message": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        # Silence default stderr logging; integrate with minimind logger later.
        return

    def _send_response(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class ToolServer:
    """
    Minimal blocking HTTP server for local tool calls.
    """

    def __init__(self, registry: ToolRegistry, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.registry = registry
        self.host = host
        self.port = port
        _ToolRequestHandler.registry = registry
        self._server = HTTPServer((self.host, self.port), _ToolRequestHandler)

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def shutdown(self) -> None:
        self._server.shutdown()


__all__ = ["ToolServer"]

