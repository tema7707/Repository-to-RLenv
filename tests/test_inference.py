from __future__ import annotations

from openenv.core.client_types import StepResult

from repo2env.app.inference import (
    ACTION_JSON_SCHEMA,
    HeuristicPolicySession,
    INVALID_ACTION_PENALTY,
    OpenAICompatiblePolicySession,
    _extract_json_object,
    _render_observation_prompt,
    _sanitize_action,
    run_inference_episode,
    safe_next_action,
    should_auto_submit_after_green_tests,
)
from repo2env.openenv.models import Repo2EnvAction, Repo2EnvObservation, Repo2EnvState


def _observation(**overrides):
    payload = {
        "repo_source": "/tmp/repo",
        "repo_name": "demo_repo",
        "repo_summary": {"repo_name": "demo_repo"},
        "task_summary": {
            "task_mode": "repo_editing",
            "starter_paths": ["src/demo.py", "tests/test_demo.py"],
        },
        "setup_summary": {},
        "selected_source_files": [
            {
                "path": "src/demo.py",
                "preview": 'def demo(x):\n    return x + 1\n',
            }
        ],
        "selected_test_files": [{"path": "tests/test_demo.py", "preview": ""}],
        "latest_pytest_summary": {"passed": 2, "failed": 1, "errors": 0},
        "latest_coverage_summary": None,
        "recent_action_history": [],
        "step_count": 0,
        "max_steps": 8,
        "tool_result": {},
        "reward_breakdown": {},
        "current_metrics": {"passing_tests": 2, "failed_tests": 1, "invalid_tests": 0},
        "allowed_tools": [
            "list_files",
            "read_file",
            "read_file_chunk",
            "search_code",
            "replace_in_file",
            "insert_after",
            "insert_before",
            "append_to_file",
            "write_file",
            "run_tests",
            "get_test_failures",
            "diff_working_tree",
            "submit",
        ],
        "done": False,
        "reward": None,
        "metadata": {},
    }
    payload.update(overrides)
    return Repo2EnvObservation.model_validate(payload)


class FakeInferenceClient:
    def reset(self, **kwargs):
        assert kwargs["repo_source"] == "/tmp/repo"
        return StepResult(observation=_observation(), reward=None, done=False)

    def step(self, action, **kwargs):
        del kwargs
        if action.tool == "list_files":
            return StepResult(
                observation=_observation(step_count=1, tool_result={"count": 2}),
                reward=-0.2,
                done=False,
            )
        if action.tool == "read_file":
            return StepResult(
                observation=_observation(
                    step_count=2,
                    tool_result={
                        "path": action.args["path"],
                        "content": "def demo(x):\n    return x + 1\n",
                    },
                ),
                reward=-0.2,
                done=False,
            )
        if action.tool == "run_tests":
            return StepResult(
                observation=_observation(
                    step_count=3,
                    tool_result={"passed": 3},
                    current_metrics={"passing_tests": 3, "failed_tests": 0, "invalid_tests": 0},
                ),
                reward=0.8,
                done=False,
            )
        return StepResult(
            observation=_observation(
                step_count=4,
                tool_result={"submitted": True},
                current_metrics={"passing_tests": 3, "failed_tests": 0, "invalid_tests": 0},
                done=True,
            ),
            reward=-0.2,
            done=True,
        )

    def state(self):
        return Repo2EnvState.model_validate(
            {
                "episode_id": "episode-1",
                "step_count": 4,
                "repo_source": "/tmp/repo",
                "repo_name": "demo_repo",
                "task_mode": "repo_editing",
                "current_metrics": {"passing_tests": 3, "failed_tests": 0, "invalid_tests": 0},
                "latest_tool_result": {"submitted": True},
                "latest_reward_breakdown": {},
                "setup_summary": {},
                "allowed_tools": ["list_files", "read_file", "write_file", "run_tests", "submit"],
            }
        )


def test_extract_json_object_handles_fenced_json() -> None:
    parsed = _extract_json_object("```json\n{\"tool\":\"submit\",\"args\":{}}\n```")
    assert parsed["tool"] == "submit"


def test_sanitize_action_falls_back_for_invalid_tool() -> None:
    action = _sanitize_action({"tool": "rm_repo", "args": {}}, ["list_files", "submit"])
    assert action.tool == "submit"


def test_sanitize_action_normalizes_common_aliases() -> None:
    action = _sanitize_action(
        {"tool": "search_code", "args": {"search_term": "take_nth"}},
        ["search_code", "submit"],
    )
    assert action.tool == "search_code"
    assert action.args == {"query": "take_nth"}


def test_render_observation_prompt_includes_tool_schema_and_workflow() -> None:
    prompt = _render_observation_prompt(
        _observation(
            tool_result={"path": "src/demo.py", "replacements": 1},
            reward_breakdown={"delta_passing_tests": 0, "step_penalty": 0.2},
        )
    )
    assert "tool_schemas" in prompt
    assert "replace_in_file" in prompt
    assert "get_test_failures" in prompt
    assert "diff_working_tree" in prompt
    assert "\"replacements\": 1" in prompt
    assert "\"step_penalty\": 0.2" in prompt
    assert "run tests immediately after editing" in prompt.lower()


def test_openai_policy_defaults_temperature_to_one(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_responses_create(
        *,
        api_base_url,
        api_key,
        model_name,
        instructions,
        input_text,
        previous_response_id,
        temperature,
        timeout_s,
    ):
        captured["api_base_url"] = api_base_url
        captured["api_key"] = api_key
        captured["model_name"] = model_name
        captured["instructions"] = instructions
        captured["input_text"] = input_text
        captured["previous_response_id"] = previous_response_id
        captured["temperature"] = temperature
        captured["timeout_s"] = timeout_s
        return '{"tool":"submit","args":{}}', "resp_123"

    monkeypatch.setattr("repo2env.app.inference._openai_responses_create", _fake_responses_create)

    policy = OpenAICompatiblePolicySession(
        model_name="gpt-5",
        api_base_url="https://api.openai.com/v1",
        api_key="secret",
    )

    action = policy.next_action(_observation())

    assert action.tool == "submit"
    assert captured["temperature"] == 1.0
    assert captured["previous_response_id"] is None
    assert policy.previous_response_id == "resp_123"


def test_auto_submit_after_green_test_run() -> None:
    observation = _observation(
        current_metrics={"passing_tests": 3, "failed_tests": 0, "invalid_tests": 0},
        done=False,
    )
    assert should_auto_submit_after_green_tests(
        Repo2EnvAction(tool="run_tests", args={}),
        observation,
    )
    assert not should_auto_submit_after_green_tests(
        Repo2EnvAction(tool="read_file", args={"path": "src/demo.py"}),
        observation,
    )


def test_safe_next_action_penalizes_invalid_model_output(monkeypatch) -> None:
    def _fake_responses_create(
        *,
        api_base_url,
        api_key,
        model_name,
        instructions,
        input_text,
        previous_response_id,
        temperature,
        timeout_s,
    ):
        del api_base_url, api_key, model_name, instructions, input_text, previous_response_id, temperature, timeout_s
        return "definitely not json", "resp_bad"

    monkeypatch.setattr("repo2env.app.inference._openai_responses_create", _fake_responses_create)
    policy = OpenAICompatiblePolicySession(
        model_name="gpt-5",
        api_base_url="https://api.openai.com/v1",
        api_key="secret",
    )

    action, reward, error = safe_next_action(policy, _observation())

    assert action.tool == "get_test_failures"
    assert reward == -INVALID_ACTION_PENALTY
    assert "Invalid model action" in error


def test_action_json_schema_closes_args_object() -> None:
    schema = ACTION_JSON_SCHEMA["schema"]

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["tool", "args"]
    assert schema["properties"]["args"]["type"] == "string"
    assert "read_file" in schema["properties"]["tool"]["enum"]


def test_openai_policy_reuses_previous_response_id(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_responses_create(
        *,
        api_base_url,
        api_key,
        model_name,
        instructions,
        input_text,
        previous_response_id,
        temperature,
        timeout_s,
    ):
        del api_base_url, api_key, model_name, instructions, input_text, temperature, timeout_s
        calls.append({"previous_response_id": previous_response_id})
        return '{"tool":"submit","args":{}}', f"resp_{len(calls)}"

    monkeypatch.setattr("repo2env.app.inference._openai_responses_create", _fake_responses_create)
    policy = OpenAICompatiblePolicySession(
        model_name="gpt-5",
        api_base_url="https://api.openai.com/v1",
        api_key="secret",
    )

    policy.next_action(_observation())
    policy.next_action(_observation(step_count=1))

    assert calls[0]["previous_response_id"] is None
    assert calls[1]["previous_response_id"] == "resp_1"


class SubmitFirstPolicy:
    name = "submit-first"
    model_name = "submit-first"

    def next_action(self, observation: Repo2EnvObservation) -> Repo2EnvAction:
        del observation
        return Repo2EnvAction(tool="submit", args={})


class SubmitValidationClient:
    def reset(self, **kwargs):
        del kwargs
        return StepResult(observation=_observation(), reward=None, done=False)

    def step(self, action, **kwargs):
        del kwargs
        if action.tool == "run_tests":
            return StepResult(
                observation=_observation(
                    step_count=1,
                    tool_result={"passed": 3},
                    current_metrics={"passing_tests": 3, "failed_tests": 0, "invalid_tests": 0},
                    latest_pytest_summary={"passed": 3, "failed": 0, "errors": 0},
                ),
                reward=0.8,
                done=False,
            )
        if action.tool == "submit":
            return StepResult(
                observation=_observation(
                    step_count=2,
                    tool_result={"submitted": True},
                    current_metrics={"passing_tests": 3, "failed_tests": 0, "invalid_tests": 0},
                    latest_pytest_summary={"passed": 3, "failed": 0, "errors": 0},
                    done=True,
                ),
                reward=-0.2,
                done=True,
            )
        raise AssertionError(f"unexpected tool: {action.tool}")

    def state(self):
        return Repo2EnvState.model_validate(
            {
                "episode_id": "episode-submit",
                "step_count": 2,
                "repo_source": "/tmp/repo",
                "repo_name": "demo_repo",
                "task_mode": "repo_editing",
                "current_metrics": {"passing_tests": 3, "failed_tests": 0, "invalid_tests": 0},
                "latest_tool_result": {"submitted": True},
                "latest_reward_breakdown": {},
                "setup_summary": {},
                "allowed_tools": ["run_tests", "submit"],
            }
        )


def test_run_inference_episode_with_heuristic_policy() -> None:
    result = run_inference_episode(
        FakeInferenceClient(),
        HeuristicPolicySession(),
        base_url="http://127.0.0.1:8000",
        repo_source="/tmp/repo",
    )

    assert result.policy_name == "heuristic"
    assert result.repo_name == "demo_repo"
    assert result.final_metrics["passing_tests"] == 3
    assert [step["tool"] for step in result.action_trace] == ["list_files", "read_file", "read_file", "run_tests", "submit"]


def test_run_inference_episode_forces_test_run_before_submit() -> None:
    result = run_inference_episode(
        SubmitValidationClient(),
        SubmitFirstPolicy(),
        base_url="http://127.0.0.1:8000",
    )

    assert [step["tool"] for step in result.action_trace] == ["run_tests", "submit"]
