from __future__ import annotations

import sys

from repo2env.openenv import server


def test_server_main_honors_host_and_port(monkeypatch) -> None:
    called: dict[str, object] = {}

    def fake_run_server(*, host: str, port: int) -> None:
        called["host"] = host
        called["port"] = port

    monkeypatch.setattr(server, "run_server", fake_run_server)
    monkeypatch.setattr(sys, "argv", ["repo2env-openenv-server", "--host", "127.0.0.1", "--port", "8011"])

    server.main()

    assert called == {"host": "127.0.0.1", "port": 8011}
