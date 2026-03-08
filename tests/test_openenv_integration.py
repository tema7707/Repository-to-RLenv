from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from repo2env.openenv.server import app


def test_openenv_websocket_reset_step_and_state() -> None:
    repo_path = str(Path("examples/repo_a").resolve())

    with TestClient(app) as client:
        schema_response = client.get("/schema")
        assert schema_response.status_code == 200
        schema_payload = schema_response.json()
        assert "tool" in json.dumps(schema_payload["action"])

        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"type": "reset", "data": {"repo_source": repo_path}})
            reset_payload = websocket.receive_json()
            assert reset_payload["type"] == "observation"
            reset_data = reset_payload["data"]
            assert reset_data["observation"]["repo_name"] == "repo_a"
            assert reset_data["observation"]["repo_source"] == repo_path
            assert reset_data["observation"]["task_summary"]["task_mode"] == "repo_editing"
            assert reset_data["done"] is False

            websocket.send_json(
                {"type": "step", "data": {"tool": "list_files", "args": {"limit": 10}}}
            )
            step_payload = websocket.receive_json()
            assert step_payload["type"] == "observation"
            step_data = step_payload["data"]
            assert step_data["observation"]["tool_result"]["count"] >= 2
            assert step_data["observation"]["current_metrics"]["passing_tests"] >= 2

            websocket.send_json({"type": "state"})
            state_payload = websocket.receive_json()
            assert state_payload["type"] == "state"
            assert state_payload["data"]["repo_name"] == "repo_a"
            assert state_payload["data"]["task_mode"] == "repo_editing"
            assert state_payload["data"]["step_count"] == 1
