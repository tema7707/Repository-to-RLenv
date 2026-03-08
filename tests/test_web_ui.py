from __future__ import annotations

import time

from fastapi.testclient import TestClient

from repo2env.app.inference import INVALID_ACTION_PENALTY
from repo2env.openenv.models import Repo2EnvObservation, Repo2EnvState
from repo2env.openenv.server import app
from repo2env.openenv import server, webui
from repo2env.openenv.webui import InferenceSessionRequest, UISessionManager


def _observation(**overrides):
    payload = {
        "repo_source": "/tmp/repo",
        "repo_name": "demo_repo",
        "repo_summary": {"repo_name": "demo_repo"},
        "task_summary": {"task_mode": "repo_editing"},
        "setup_summary": {"success": True},
        "selected_source_files": [{"path": "src/demo.py", "preview": "def demo():\n    return 1\n"}],
        "selected_test_files": [{"path": "tests/test_demo.py", "preview": ""}],
        "latest_pytest_summary": {"passed": 0, "failed": 1, "errors": 0, "output": "1 failed"},
        "latest_coverage_summary": None,
        "recent_action_history": [],
        "step_count": 0,
        "max_steps": 6,
        "tool_result": {},
        "reward_breakdown": {},
        "current_metrics": {"passing_tests": 0, "failed_tests": 1, "invalid_tests": 0},
        "allowed_tools": ["read_file", "replace_in_file", "run_tests", "submit"],
        "tool_schemas": {"run_tests": {"args": {}}},
        "done": False,
        "reward": None,
        "metadata": {"episode_id": "ep-1"},
    }
    payload.update(overrides)
    return Repo2EnvObservation.model_validate(payload)


class FakePolicy:
    def __init__(self, **_: object) -> None:
        self.calls = 0

    def next_action(self, observation: Repo2EnvObservation):
        del observation
        self.calls += 1
        return server.Repo2EnvAction(tool="run_tests", args={})


class InvalidPolicy:
    def __init__(self, **_: object) -> None:
        self.calls = 0

    def next_action(self, observation: Repo2EnvObservation):
        del observation
        self.calls += 1
        if self.calls == 1:
            raise ValueError("bad command")
        return server.Repo2EnvAction(tool="submit", args={})


class FakeEnvironment:
    def __init__(self) -> None:
        self._state = Repo2EnvState.model_validate(
            {
                "episode_id": "session-1",
                "step_count": 1,
                "repo_source": "/tmp/repo",
                "repo_name": "demo_repo",
                "task_mode": "repo_editing",
                "current_metrics": {"passing_tests": 1, "failed_tests": 0, "invalid_tests": 0},
                "latest_tool_result": {"passed": 1, "failed": 0},
                "latest_reward_breakdown": {"delta_passing_tests": 1, "step_penalty": 0.2},
                "setup_summary": {"success": True},
                "allowed_tools": ["run_tests", "submit"],
            }
        )

    def reset(self, **kwargs):
        assert kwargs["max_steps"] == 6
        return _observation()

    def step(self, action):
        assert action.tool == "run_tests"
        return _observation(
            step_count=1,
            latest_pytest_summary={"passed": 1, "failed": 0, "errors": 0, "output": "1 passed"},
            tool_result={"passed": 1, "failed": 0},
            reward_breakdown={"delta_passing_tests": 1, "step_penalty": 0.2},
            current_metrics={"passing_tests": 1, "failed_tests": 0, "invalid_tests": 0},
            reward=0.8,
            done=True,
        )

    @property
    def state(self):
        return self._state

    def close(self) -> None:
        return None


class StubSession:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id


class StubManager:
    def __init__(self) -> None:
        self._streamed = False

    def start_session(self, request: InferenceSessionRequest):
        assert request.policy_kind == "model"
        assert request.model_name == "gpt-5"
        return StubSession("session-123")

    def get_session(self, session_id: str):
        if session_id != "session-123":
            return None
        return webui.UISessionRecord(
            session_id=session_id,
            policy_kind="model",
            model_name="gpt-5",
            api_base_url="https://api.openai.com/v1",
            max_steps=12,
            status="completed",
            status_message="Episode finished",
            repo_name="demo_repo",
            task_mode="repo_editing",
            current_metrics={"passing_tests": 1, "failed_tests": 0, "invalid_tests": 0},
            latest_pytest_summary={"passed": 1, "failed": 0, "errors": 0},
            latest_reward_breakdown={"delta_passing_tests": 1, "step_penalty": 0.2},
            events=[{"type": "step", "tool": "run_tests"}],
            final_state={"episode_id": "session-123"},
        )

    def wait_for_event(self, session_id: str, after_revision: int, timeout_s: float = 15.0):
        del after_revision, timeout_s
        if session_id != "session-123":
            return "missing", None
        if self._streamed:
            return "timeout", None
        self._streamed = True
        return "update", self.get_session(session_id).to_public_dict()


def test_ui_session_manager_records_live_run(monkeypatch) -> None:
    monkeypatch.setattr(webui, "OpenAICompatiblePolicySession", FakePolicy)
    manager = UISessionManager(lambda: FakeEnvironment())
    session = manager.start_session(
        InferenceSessionRequest(
            model_name="gpt-5",
            api_key="secret",
            max_steps=6,
        )
    )

    deadline = time.time() + 3.0
    while time.time() < deadline:
        current = manager.get_session(session.session_id)
        assert current is not None
        if current.status == "completed":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("session did not complete in time")

    assert current.repo_name == "demo_repo"
    assert current.current_metrics["passing_tests"] == 1
    assert current.latest_pytest_summary["passed"] == 1
    assert [event["type"] for event in current.events] == ["reset", "step"]
    assert current.events[-1]["tool"] == "run_tests"


def test_ui_session_manager_penalizes_invalid_action_without_crashing(monkeypatch) -> None:
    monkeypatch.setattr(webui, "OpenAICompatiblePolicySession", InvalidPolicy)
    manager = UISessionManager(lambda: FakeEnvironment())
    session = manager.start_session(
        InferenceSessionRequest(
            model_name="gpt-5",
            api_key="secret",
            max_steps=6,
        )
    )

    deadline = time.time() + 3.0
    while time.time() < deadline:
        current = manager.get_session(session.session_id)
        assert current is not None
        if current.status in {"completed", "failed"}:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("session did not complete in time")

    assert current is not None
    assert current.status == "completed"
    assert current.events[1]["type"] == "invalid_action"
    assert current.events[1]["reward"] == -INVALID_ACTION_PENALTY


def test_web_ui_routes(monkeypatch) -> None:
    monkeypatch.setattr(server, "ui_sessions", StubManager())
    client = TestClient(app)

    html = client.get("/web")
    assert html.status_code == 200
    assert "Repo2Env Live UI" in html.text
    assert "Watch one repo run." in html.text
    assert "Baseline" in html.text
    assert "Start Episode" in html.text
    assert "Download JSON" in html.text
    assert "gpt-5.2" in html.text
    assert "gpt-5.2-pro" in html.text
    assert "gpt-5.2-codex" in html.text
    assert "gpt-5-pro" in html.text
    assert "gpt-5-codex" in html.text
    assert "gpt-5-mini" in html.text
    assert "gpt-4.1-nano" in html.text

    created = client.post(
        "/api/ui/sessions",
        json={
            "policy_kind": "model",
            "model_name": "gpt-5",
            "api_key": "secret",
            "max_steps": 12,
        },
    )
    assert created.status_code == 200
    assert created.json() == {"session_id": "session-123"}

    fetched = client.get("/api/ui/sessions/session-123")
    assert fetched.status_code == 200
    body = fetched.json()
    assert body["repo_name"] == "demo_repo"
    assert body["current_metrics"]["passing_tests"] == 1
    assert body["events"][0]["tool"] == "run_tests"

    streamed = client.get("/api/ui/sessions/session-123/stream")
    assert streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("text/event-stream")
    assert '"session_id": "session-123"' in streamed.text

    missing = client.get("/api/ui/sessions/missing")
    assert missing.status_code == 404


def test_start_ui_session_rejects_missing_model_credentials(monkeypatch) -> None:
    monkeypatch.setattr(server, "ui_sessions", UISessionManager(lambda: FakeEnvironment()))
    client = TestClient(app)

    response = client.post(
        "/api/ui/sessions",
        json={
            "policy_kind": "model",
            "model_name": "gpt-5",
            "max_steps": 12,
        },
    )

    assert response.status_code == 400
    assert "api_key" in response.json()["detail"]
