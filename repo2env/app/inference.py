from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib import error, request

from openenv.core.client_types import StepResult

from repo2env.app.tooling import REPO2ENV_TOOL_SCHEMAS, normalize_tool_args
from repo2env.openenv import Repo2EnvAction, Repo2EnvClient, Repo2EnvObservation

DEFAULT_OUTPUT = Path("outputs/benchmarks") / "inference_results.json"
INVALID_ACTION_PENALTY = 1.0
def _build_action_json_schema() -> dict[str, Any]:
    return {
        "name": "repo2env_action",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "tool": {"type": "string", "enum": list(REPO2ENV_TOOL_SCHEMAS)},
                "args": {
                    "type": "string",
                    "description": "A JSON object string containing the tool arguments. Use '{}' when the tool takes no arguments.",
                },
            },
            "required": ["tool", "args"],
        },
        "strict": True,
    }


ACTION_JSON_SCHEMA = _build_action_json_schema()
MODEL_SYSTEM_PROMPT = (
    "You are operating a Repo2Env coding environment over safe tools. "
    "Return exactly one JSON object with keys `tool` and `args`. "
    "Only use the allowed tools shown in the observation, and follow the exact argument names in `tool_schemas`. "
    "Prefer targeted edit tools like `replace_in_file`, `insert_after`, `insert_before`, or `append_to_file` before `write_file`. "
    "Use `read_file_chunk` for large files, `get_test_failures` after `run_tests` when tests fail, and `diff_working_tree` to inspect your edits. "
    "After any successful edit, call `run_tests` next. If the latest tool_result says an edit succeeded, do not repeat the same edit before running tests. "
    "If the latest tool_result says an edit target was not found, inspect the file or diff instead of retrying the same edit blindly. "
    "Avoid rereading the same file unless the file changed. "
    "If you are done or blocked, return "
    '{"tool":"submit","args":{}}.'
)


@dataclass(slots=True)
class BenchmarkEpisodeResult:
    policy_name: str
    model_name: str | None
    base_url: str
    repo_source: str | None
    task_mode: str | None
    repo_name: str
    total_reward: float
    step_count: int
    final_metrics: dict[str, Any]
    final_state: dict[str, Any]
    action_trace: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Repo2EnvClientLike(Protocol):
    def reset(self, **kwargs: Any) -> StepResult[Repo2EnvObservation]: ...

    def step(self, action: Repo2EnvAction, **kwargs: Any) -> StepResult[Repo2EnvObservation]: ...

    def state(self) -> Any: ...


class PolicySession(Protocol):
    name: str
    model_name: str | None

    def next_action(self, observation: Repo2EnvObservation) -> Repo2EnvAction: ...


class HeuristicPolicySession:
    name = "heuristic"
    model_name = None

    def __init__(self) -> None:
        self._listed = False
        self._read_paths: set[str] = set()
        self._ran_tests = False

    def next_action(self, observation: Repo2EnvObservation) -> Repo2EnvAction:
        if not self._listed:
            self._listed = True
            return Repo2EnvAction(tool="list_files", args={"limit": 50})

        next_read = self._next_read_path(observation)
        if next_read:
            self._read_paths.add(next_read)
            return Repo2EnvAction(tool="read_file", args={"path": next_read})

        if not self._ran_tests:
            self._ran_tests = True
            return Repo2EnvAction(tool="run_tests", args={})

        return Repo2EnvAction(tool="submit", args={})

    def _next_read_path(self, observation: Repo2EnvObservation) -> str | None:
        candidates: list[str] = []
        for path in observation.task_summary.get("starter_paths", []):
            candidates.append(str(path))
        for bucket in (observation.selected_source_files, observation.selected_test_files):
            for item in bucket:
                path = item.get("path")
                if path:
                    candidates.append(str(path))
        for path in candidates:
            if path not in self._read_paths:
                return path
        return None


class OpenAICompatiblePolicySession:
    name = "openai-compatible"

    def __init__(
        self,
        *,
        model_name: str,
        api_base_url: str,
        api_key: str,
        temperature: float = 1.0,
        timeout_s: float = 60.0,
    ) -> None:
        self.model_name = model_name
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.previous_response_id: str | None = None

    def next_action(self, observation: Repo2EnvObservation) -> Repo2EnvAction:
        raw_text, response_id = _openai_responses_create(
            api_base_url=self.api_base_url,
            api_key=self.api_key,
            model_name=self.model_name,
            instructions=MODEL_SYSTEM_PROMPT,
            input_text=_render_observation_prompt(observation),
            previous_response_id=self.previous_response_id,
            temperature=self.temperature,
            timeout_s=self.timeout_s,
        )
        self.previous_response_id = response_id
        parsed = _extract_json_object(raw_text)
        return _sanitize_action(parsed, observation.allowed_tools)


def should_auto_submit_after_green_tests(action: Repo2EnvAction, observation: Repo2EnvObservation) -> bool:
    if action.tool != "run_tests" or observation.done:
        return False
    metrics = observation.current_metrics or {}
    return metrics.get("failed_tests", 0) == 0 and metrics.get("invalid_tests", 0) == 0


def safe_next_action(
    policy: PolicySession,
    observation: Repo2EnvObservation,
) -> tuple[Repo2EnvAction, float, str | None]:
    try:
        return policy.next_action(observation), 0.0, None
    except Exception as exc:
        fallback = _invalid_action_fallback_tool(observation.allowed_tools)
        return (
            Repo2EnvAction(tool=fallback, args={}),
            -INVALID_ACTION_PENALTY,
            f"Invalid model action: {exc}",
        )


def _invalid_action_fallback_tool(allowed_tools: list[str]) -> str:
    preferred_order = (
        "get_test_failures",
        "diff_working_tree",
        "list_files",
        "run_tests",
        "submit",
    )
    for tool in preferred_order:
        if tool in allowed_tools and not REPO2ENV_TOOL_SCHEMAS[tool]["required_args"]:
            return tool
    return allowed_tools[0]


def run_inference_episode(
    client: Repo2EnvClientLike,
    policy: PolicySession,
    *,
    base_url: str,
    repo_source: str | None = None,
    max_steps: int | None = None,
    coverage_target: float | None = None,
    task_spec_path: str | None = None,
) -> BenchmarkEpisodeResult:
    reset_kwargs: dict[str, Any] = {}
    if repo_source:
        reset_kwargs["repo_source"] = repo_source
    if max_steps is not None:
        reset_kwargs["max_steps"] = max_steps
    if coverage_target is not None:
        reset_kwargs["coverage_target"] = coverage_target
    if task_spec_path:
        reset_kwargs["task_spec_path"] = task_spec_path

    reset_result = client.reset(**reset_kwargs)
    observation = reset_result.observation
    total_reward = 0.0
    action_trace: list[dict[str, Any]] = []

    while not observation.done:
        action, action_penalty, action_error = safe_next_action(policy, observation)
        if action_error is not None:
            total_reward += action_penalty
            action_trace.append(
                {
                    "tool": "invalid_action",
                    "args": {},
                    "reward": action_penalty,
                    "done": False,
                    "tool_result": {"error": action_error, "fallback_tool": action.tool},
                    "metrics": observation.current_metrics,
                }
            )
        if action.tool == "submit":
            validation_result = client.step(Repo2EnvAction(tool="run_tests", args={}))
            observation = validation_result.observation
            total_reward += float(validation_result.reward or 0.0)
            action_trace.append(
                {
                    "tool": "run_tests",
                    "args": {},
                    "reward": validation_result.reward,
                    "done": validation_result.done,
                    "tool_result": observation.tool_result,
                    "metrics": observation.current_metrics,
                }
            )
            if validation_result.done:
                break
            if should_auto_submit_after_green_tests(Repo2EnvAction(tool="run_tests", args={}), observation):
                submit_result = client.step(Repo2EnvAction(tool="submit", args={}))
                observation = submit_result.observation
                total_reward += float(submit_result.reward or 0.0)
                action_trace.append(
                    {
                        "tool": "submit",
                        "args": {},
                        "reward": submit_result.reward,
                        "done": submit_result.done,
                        "tool_result": observation.tool_result,
                        "metrics": observation.current_metrics,
                    }
                )
                break
            continue
        step_result = client.step(action)
        observation = step_result.observation
        total_reward += float(step_result.reward or 0.0)
        action_trace.append(
            {
                "tool": action.tool,
                "args": action.args,
                "reward": step_result.reward,
                "done": step_result.done,
                "tool_result": observation.tool_result,
                "metrics": observation.current_metrics,
            }
        )
        if step_result.done:
            break
        if should_auto_submit_after_green_tests(action, observation):
            submit_action = Repo2EnvAction(tool="submit", args={})
            submit_result = client.step(submit_action)
            observation = submit_result.observation
            total_reward += float(submit_result.reward or 0.0)
            action_trace.append(
                {
                    "tool": submit_action.tool,
                    "args": submit_action.args,
                    "reward": submit_result.reward,
                    "done": submit_result.done,
                    "tool_result": observation.tool_result,
                    "metrics": observation.current_metrics,
                }
            )
            break

    state = client.state()
    return BenchmarkEpisodeResult(
        policy_name=policy.name,
        model_name=policy.model_name,
        base_url=base_url,
        repo_source=repo_source,
        task_mode=observation.task_summary.get("task_mode"),
        repo_name=observation.repo_name,
        total_reward=round(total_reward, 4),
        step_count=observation.step_count,
        final_metrics=observation.current_metrics,
        final_state=state.model_dump(),
        action_trace=action_trace,
    )


def benchmark_targets(
    *,
    base_urls: list[str],
    repo_sources: list[str] | None,
    policy_factories: list[Callable[[], PolicySession]],
    episodes: int,
    max_steps: int | None,
    coverage_target: float | None,
    task_spec_path: str | None,
) -> list[BenchmarkEpisodeResult]:
    results: list[BenchmarkEpisodeResult] = []
    repo_variants = repo_sources or [None]
    for base_url in base_urls:
        for repo_source in repo_variants:
            for policy_factory in policy_factories:
                for _ in range(episodes):
                    with Repo2EnvClient(base_url=base_url) as client:
                        results.append(
                            run_inference_episode(
                                client,
                                policy_factory(),
                                base_url=base_url,
                                repo_source=repo_source,
                                max_steps=max_steps,
                                coverage_target=coverage_target,
                                task_spec_path=task_spec_path,
                            )
                        )
    return results


def _render_observation_prompt(observation: Repo2EnvObservation) -> str:
    payload = {
        "repo_name": observation.repo_name,
        "task_summary": observation.task_summary,
        "setup_summary": observation.setup_summary,
        "selected_source_files": observation.selected_source_files,
        "selected_test_files": observation.selected_test_files,
        "latest_pytest_summary": observation.latest_pytest_summary,
        "tool_result": observation.tool_result,
        "reward_breakdown": observation.reward_breakdown,
        "recent_action_history": observation.recent_action_history,
        "current_metrics": observation.current_metrics,
        "tool_schemas": observation.tool_schemas,
        "allowed_tools": observation.allowed_tools,
    }
    return (
        "Choose the next safe Repo2Env action.\n"
        "Return only JSON.\n"
        "Workflow: inspect the relevant file, make the smallest valid edit, run tests immediately after editing, inspect structured failures or diffs if needed, then submit when no further progress is likely.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )


def _openai_responses_create(
    *,
    api_base_url: str,
    api_key: str,
    model_name: str,
    instructions: str,
    input_text: str,
    previous_response_id: str | None,
    temperature: float,
    timeout_s: float,
) -> tuple[str, str | None]:
    url = f"{api_base_url}/responses"
    payload: dict[str, Any] = {
        "model": model_name,
        "instructions": instructions,
        "input": input_text,
        "temperature": temperature,
        "store": True,
        "truncation": "auto",
        "text": {
            "format": {
                "type": "json_schema",
                **ACTION_JSON_SCHEMA,
            }
        },
    }
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Model API request failed: {exc.code} {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Model API request failed: {exc.reason}") from exc

    text = _extract_responses_output_text(body)
    return text, body.get("id")


def _extract_responses_output_text(body: dict[str, Any]) -> str:
    output = body.get("output")
    if not isinstance(output, list):
        raise RuntimeError(f"Unexpected Responses API output: {body}")

    text_parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                text_parts.append(str(content.get("text", "")))

    text = "\n".join(part for part in text_parts if part)
    if not text:
        raise RuntimeError(f"Responses API returned no output_text content: {body}")
    return text


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Could not parse JSON action from model output: {text}")


def _sanitize_action(candidate: dict[str, Any], allowed_tools: list[str]) -> Repo2EnvAction:
    tool = candidate.get("tool")
    args = candidate.get("args", {})
    if tool not in allowed_tools:
        fallback = "submit" if "submit" in allowed_tools else allowed_tools[0]
        return Repo2EnvAction(tool=fallback, args={})
    if isinstance(args, str):
        try:
            parsed_args = json.loads(args)
            args = parsed_args if isinstance(parsed_args, dict) else {}
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return Repo2EnvAction(tool=str(tool), args=normalize_tool_args(str(tool), args))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference and benchmark models on Repo2Env environments.")
    parser.add_argument(
        "--base-url",
        action="append",
        dest="base_urls",
        help="Repeatable OpenEnv base URL. Defaults to http://127.0.0.1:8000 if omitted.",
    )
    parser.add_argument("--repo-source", action="append", help="Optional repeatable repo override passed to reset().")
    parser.add_argument("--task-spec-path", help="Optional task spec override passed to reset().")
    parser.add_argument("--policy", action="append", choices=["heuristic"], help="Repeatable built-in policy.")
    parser.add_argument("--model", action="append", help="Repeatable OpenAI-compatible model name.")
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="Base URL for the OpenAI-compatible model API.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the model API key.",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for model policies.")
    parser.add_argument("--timeout-s", type=float, default=60.0, help="Per-request timeout for model API calls.")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes to run per env/policy combination.")
    parser.add_argument("--max-steps", type=int, help="Optional reset max_steps override.")
    parser.add_argument("--coverage-target", type=float, help="Ignored legacy argument.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Where to write the benchmark JSON report.",
    )
    args = parser.parse_args()

    base_urls = args.base_urls or ["http://127.0.0.1:8000"]
    policy_factories: list[Callable[[], PolicySession]] = []

    selected_policies = args.policy or []
    if not selected_policies and not args.model:
        selected_policies = ["heuristic"]
    if "heuristic" in selected_policies:
        policy_factories.append(HeuristicPolicySession)

    if args.model:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing model API key. Set the `{args.api_key_env}` environment variable."
            )
        for model_name in args.model:
            policy_factories.append(
                lambda model_name=model_name: OpenAICompatiblePolicySession(
                    model_name=model_name,
                    api_base_url=args.api_base_url,
                    api_key=api_key,
                    temperature=args.temperature,
                    timeout_s=args.timeout_s,
                )
            )

    results = benchmark_targets(
        base_urls=base_urls,
        repo_sources=args.repo_source,
        policy_factories=policy_factories,
        episodes=args.episodes,
        max_steps=args.max_steps,
        coverage_target=args.coverage_target,
        task_spec_path=args.task_spec_path,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([result.to_dict() for result in results], indent=2),
        encoding="utf-8",
    )
    print(f"wrote={output_path}")
    for result in results:
        print(
            " | ".join(
                [
                    f"policy={result.policy_name}",
                    f"model={result.model_name or '-'}",
                    f"base_url={result.base_url}",
                    f"repo={result.repo_name}",
                    f"return={result.total_reward}",
                    f"passing={result.final_metrics.get('passing_tests', 0)}",
                    f"failed={result.final_metrics.get('failed_tests', 0)}",
                ]
            )
        )


__all__ = [
    "BenchmarkEpisodeResult",
    "HeuristicPolicySession",
    "OpenAICompatiblePolicySession",
    "benchmark_targets",
    "run_inference_episode",
]


if __name__ == "__main__":
    main()
