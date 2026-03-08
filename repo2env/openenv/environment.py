from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from repo2env.app.env import Repo2EnvEnvironment
from repo2env.openenv.models import (
    REPO2ENV_TOOL_NAMES,
    Repo2EnvAction,
    Repo2EnvObservation,
    Repo2EnvState,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXAMPLE_REPO = PROJECT_ROOT / "examples" / "repo_a"


class Repo2EnvOpenEnvEnvironment(
    Environment[Repo2EnvAction, Repo2EnvObservation, Repo2EnvState]
):
    """OpenEnv 0.2.1 wrapper around the core Repo2Env environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        *,
        default_repo_source: str | None = None,
        max_steps: int = 12,
        coverage_target: float = 75.0,
    ) -> None:
        super().__init__()
        self.default_repo_source = self._normalize_repo_source(
            default_repo_source or os.environ.get("REPO2ENV_DEFAULT_REPO")
        )
        self.default_max_steps = max_steps
        self.default_coverage_target = coverage_target
        self._inner: Repo2EnvEnvironment | None = None
        self._state = Repo2EnvState(
            episode_id=str(uuid4()),
            step_count=0,
            repo_source=self.default_repo_source,
            allowed_tools=list(REPO2ENV_TOOL_NAMES),
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        repo_source: str | None = None,
        max_steps: int | None = None,
        coverage_target: float | None = None,
        task_spec_path: str | None = None,
        **_: Any,
    ) -> Repo2EnvObservation:
        del seed
        self.close()

        selected_repo_source = self._normalize_repo_source(repo_source or self.default_repo_source)
        selected_max_steps = max_steps or self.default_max_steps
        selected_coverage_target = coverage_target or self.default_coverage_target

        self._inner = Repo2EnvEnvironment(
            selected_repo_source,
            max_steps=selected_max_steps,
            coverage_target=selected_coverage_target,
            task_spec_path=task_spec_path,
        )
        observation_dict = self._inner.reset()
        allowed_tools = observation_dict.get("allowed_tools", list(REPO2ENV_TOOL_NAMES))
        self._state = Repo2EnvState(
            episode_id=episode_id or str(uuid4()),
            step_count=observation_dict["step_count"],
            repo_source=selected_repo_source,
            repo_name=observation_dict["repo_summary"].get("repo_name"),
            task_mode=observation_dict.get("task_summary", {}).get("task_mode"),
            current_metrics=self._inner.state.metrics.to_dict() if self._inner.state else {},
            latest_tool_result={},
            latest_reward_breakdown={},
            setup_summary=observation_dict.get("setup_summary", {}),
            allowed_tools=allowed_tools,
        )
        return self._build_observation(
            observation_dict,
            reward=None,
            done=False,
            tool_result={},
            reward_breakdown={},
            current_metrics=self._state.current_metrics,
        )

    def step(
        self,
        action: Repo2EnvAction,
        timeout_s: float | None = None,
        **_: Any,
    ) -> Repo2EnvObservation:
        del timeout_s
        if self._inner is None:
            self.reset()
        assert self._inner is not None

        observation_dict, reward, done, info = self._inner.step(
            {"tool": action.tool, "args": action.args}
        )
        self._state.step_count = observation_dict["step_count"]
        self._state.repo_source = self._inner.source
        self._state.repo_name = observation_dict["repo_summary"].get("repo_name")
        self._state.task_mode = observation_dict.get("task_summary", {}).get("task_mode")
        self._state.current_metrics = info["metrics"]
        self._state.latest_tool_result = info["tool_result"]
        self._state.latest_reward_breakdown = info["reward_breakdown"]
        self._state.setup_summary = observation_dict.get("setup_summary", {})
        self._state.allowed_tools = observation_dict.get("allowed_tools", self._state.allowed_tools)
        return self._build_observation(
            observation_dict,
            reward=reward,
            done=done,
            tool_result=info["tool_result"],
            reward_breakdown=info["reward_breakdown"],
            current_metrics=info["metrics"],
        )

    @property
    def state(self) -> Repo2EnvState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="repo2env",
            description=(
                "Turn a Python repository into a rewardable OpenEnv environment "
                "for repository-editing coding agents."
            ),
            version="0.1.0",
            author="Repo2Env",
        )

    def close(self) -> None:
        if self._inner is not None:
            self._inner.close()
            self._inner = None

    def _build_observation(
        self,
        observation_dict: dict[str, Any],
        *,
        reward: float | None,
        done: bool,
        tool_result: dict[str, Any],
        reward_breakdown: dict[str, Any],
        current_metrics: dict[str, Any],
    ) -> Repo2EnvObservation:
        return self._apply_transform(
            Repo2EnvObservation(
                repo_source=self._state.repo_source or "",
                repo_name=observation_dict["repo_summary"].get("repo_name", ""),
                repo_summary=observation_dict["repo_summary"],
                task_summary=observation_dict.get("task_summary", {}),
                setup_summary=observation_dict.get("setup_summary", {}),
                selected_source_files=observation_dict["selected_source_files"],
                selected_test_files=observation_dict["selected_test_files"],
                latest_pytest_summary=observation_dict["latest_pytest_summary"],
                latest_coverage_summary=observation_dict["latest_coverage_summary"],
                recent_action_history=observation_dict["recent_action_history"],
                step_count=observation_dict["step_count"],
                max_steps=observation_dict["max_steps"],
                tool_result=tool_result,
                reward_breakdown=reward_breakdown,
                current_metrics=current_metrics,
                tool_schemas=observation_dict.get("tool_schemas", {}),
                allowed_tools=observation_dict.get("allowed_tools", self._state.allowed_tools),
                done=done,
                reward=reward,
                metadata={
                    "episode_id": self._state.episode_id,
                    "repo_source": self._state.repo_source,
                    "task_mode": self._state.task_mode,
                },
            )
        )

    def _normalize_repo_source(self, repo_source: str | None) -> str:
        if not repo_source:
            return str(DEFAULT_EXAMPLE_REPO.resolve())
        if repo_source.startswith(("http://", "https://", "git@")):
            return repo_source
        return str(Path(repo_source).expanduser().resolve())
