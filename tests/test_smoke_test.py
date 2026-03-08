from __future__ import annotations

from openenv.core.client_types import StepResult

from repo2env.app.smoke_test import run_smoke_test
from repo2env.openenv.models import Repo2EnvObservation, Repo2EnvState


def _observation(**overrides):
    payload = {
        "repo_source": "/tmp/repo",
        "repo_name": "demo_repo",
        "repo_summary": {"repo_name": "demo_repo"},
        "task_summary": {"task_mode": "repo_editing"},
        "setup_summary": {},
        "selected_source_files": [{"path": "src/demo.py", "preview": "def demo():\n    return 1\n"}],
        "selected_test_files": [{"path": "tests/test_demo.py", "preview": ""}],
        "latest_pytest_summary": {"passed": 2, "failed": 1, "errors": 0},
        "latest_coverage_summary": None,
        "recent_action_history": [],
        "step_count": 0,
        "max_steps": 6,
        "tool_result": {},
        "reward_breakdown": {},
        "current_metrics": {"passing_tests": 2, "failed_tests": 1, "invalid_tests": 0},
        "allowed_tools": ["list_files", "read_file", "run_tests", "submit"],
        "done": False,
        "reward": None,
        "metadata": {},
    }
    payload.update(overrides)
    return Repo2EnvObservation.model_validate(payload)


class FakeClient:
    def __init__(self) -> None:
        self.actions: list[str] = []

    def reset(self, **kwargs):
        assert kwargs["repo_source"] == "/tmp/repo"
        return StepResult(observation=_observation(), reward=None, done=False)

    def step(self, action, **kwargs):
        del kwargs
        self.actions.append(action.tool)
        if action.tool == "list_files":
            return StepResult(
                observation=_observation(
                    step_count=1,
                    tool_result={"count": 4, "files": ["src/demo.py", "tests/test_demo.py"]},
                ),
                reward=-0.2,
                done=False,
            )
        if action.tool == "read_file":
            return StepResult(
                observation=_observation(
                    step_count=2,
                    tool_result={"path": "src/demo.py", "content": "def demo():\n    return 1\n"},
                ),
                reward=-0.2,
                done=False,
            )
        return StepResult(
            observation=_observation(
                step_count=3,
                tool_result={"passed": 3, "failed": 0},
                current_metrics={"passing_tests": 3, "failed_tests": 0, "invalid_tests": 0},
            ),
            reward=0.8,
            done=False,
        )

    def state(self):
        return Repo2EnvState.model_validate(
            {
                "episode_id": "episode-1",
                "step_count": 3,
                "repo_source": "/tmp/repo",
                "repo_name": "demo_repo",
                "task_mode": "repo_editing",
                "current_metrics": {"passing_tests": 3, "failed_tests": 0, "invalid_tests": 0},
                "latest_tool_result": {"passed": 3, "failed": 0},
                "latest_reward_breakdown": {"delta_passing_tests": 1, "step_penalty": 0.2},
                "setup_summary": {},
                "allowed_tools": ["list_files", "read_file", "run_tests", "submit"],
            }
        )


def test_run_smoke_test_executes_expected_actions() -> None:
    result = run_smoke_test(
        FakeClient(),
        base_url="http://127.0.0.1:8000",
        repo_source="/tmp/repo",
    )

    assert result.repo_name == "demo_repo"
    assert result.task_mode == "repo_editing"
    assert [entry["tool"] for entry in result.action_trace[1:]] == [
        "list_files",
        "read_file",
        "run_tests",
    ]
    assert result.final_metrics["passing_tests"] == 3
