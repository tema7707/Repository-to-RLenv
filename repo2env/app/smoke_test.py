from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from openenv.core.client_types import StepResult

from repo2env.openenv import Repo2EnvAction, Repo2EnvClient, Repo2EnvObservation, Repo2EnvState


class Repo2EnvClientLike(Protocol):
    def reset(self, **kwargs: Any) -> StepResult[Repo2EnvObservation]: ...

    def step(self, action: Repo2EnvAction, **kwargs: Any) -> StepResult[Repo2EnvObservation]: ...

    def state(self) -> Repo2EnvState: ...


@dataclass(slots=True)
class SmokeTestResult:
    base_url: str
    repo_source: str | None
    repo_name: str
    task_mode: str | None
    action_trace: list[dict[str, Any]]
    final_metrics: dict[str, Any]
    final_state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_smoke_test(
    client: Repo2EnvClientLike,
    *,
    base_url: str,
    repo_source: str | None = None,
    max_steps: int | None = None,
    coverage_target: float | None = None,
    task_spec_path: str | None = None,
) -> SmokeTestResult:
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
    action_trace: list[dict[str, Any]] = [
        {
            "event": "reset",
            "repo_name": observation.repo_name,
            "task_mode": observation.task_summary.get("task_mode"),
            "initial_metrics": observation.current_metrics,
        }
    ]

    actions = [Repo2EnvAction(tool="list_files", args={"limit": 25})]
    read_path = _pick_preview_path(observation)
    if read_path:
        actions.append(Repo2EnvAction(tool="read_file", args={"path": read_path}))
    actions.append(Repo2EnvAction(tool="run_tests", args={}))

    for action in actions:
        step_result = client.step(action)
        observation = step_result.observation
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

    state = client.state()
    return SmokeTestResult(
        base_url=base_url,
        repo_source=repo_source,
        repo_name=observation.repo_name,
        task_mode=observation.task_summary.get("task_mode"),
        action_trace=action_trace,
        final_metrics=observation.current_metrics,
        final_state=state.model_dump(),
    )


def _pick_preview_path(observation: Repo2EnvObservation) -> str | None:
    for bucket in (observation.selected_source_files, observation.selected_test_files):
        if bucket:
            return str(bucket[0].get("path"))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Repo2Env OpenEnv smoke test.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="OpenEnv server base URL.")
    parser.add_argument("--repo-source", help="Optional repo override passed to reset().")
    parser.add_argument("--task-spec-path", help="Optional task spec override passed to reset().")
    parser.add_argument("--max-steps", type=int, help="Optional reset max_steps override.")
    parser.add_argument("--coverage-target", type=float, help="Ignored legacy argument.")
    parser.add_argument(
        "--output",
        default=str(Path("outputs/benchmarks") / "smoke_test.json"),
        help="Where to write the smoke-test JSON report.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Repo2EnvClient(base_url=args.base_url) as client:
        result = run_smoke_test(
            client,
            base_url=args.base_url,
            repo_source=args.repo_source,
            max_steps=args.max_steps,
            coverage_target=args.coverage_target,
            task_spec_path=args.task_spec_path,
        )

    output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"wrote={output_path}")
    print(f"repo_name={result.repo_name}")
    print(f"task_mode={result.task_mode}")
    print(f"final_metrics={json.dumps(result.final_metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
