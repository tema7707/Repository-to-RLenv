from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State
from repo2env.app.tooling import REPO2ENV_TOOL_NAMES, REPO2ENV_TOOL_SCHEMAS

Repo2EnvToolName = Literal[
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
]


class Repo2EnvAction(Action):
    """Typed OpenEnv action for Repo2Env's safe tool interface."""

    tool: Repo2EnvToolName = Field(..., description="Safe tool name to execute.")
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool arguments. Shape depends on the selected tool.",
    )


class Repo2EnvObservation(Observation):
    """Observation returned after reset or step."""

    repo_source: str = Field(default="", description="Original repo path or URL.")
    repo_name: str = Field(default="", description="Repository name.")
    repo_summary: dict[str, Any] = Field(default_factory=dict)
    task_summary: dict[str, Any] = Field(default_factory=dict)
    setup_summary: dict[str, Any] = Field(default_factory=dict)
    selected_source_files: list[dict[str, Any]] = Field(default_factory=list)
    selected_test_files: list[dict[str, Any]] = Field(default_factory=list)
    latest_pytest_summary: dict[str, Any] | None = Field(default=None)
    latest_coverage_summary: dict[str, Any] | None = Field(default=None)
    recent_action_history: list[dict[str, Any]] = Field(default_factory=list)
    step_count: int = Field(default=0, ge=0)
    max_steps: int = Field(default=0, ge=0)
    tool_result: dict[str, Any] = Field(
        default_factory=dict,
        description="Serialized result from the most recent tool action.",
    )
    reward_breakdown: dict[str, Any] = Field(
        default_factory=dict,
        description="Interpretable reward components for the last step.",
    )
    current_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Current passing-test metrics.",
    )
    tool_schemas: dict[str, Any] = Field(
        default_factory=lambda: dict(REPO2ENV_TOOL_SCHEMAS),
        description="Exact tool argument schema available to the agent.",
    )
    allowed_tools: list[str] = Field(
        default_factory=lambda: list(REPO2ENV_TOOL_NAMES),
        description="Tools available to the agent.",
    )


class Repo2EnvState(State):
    """Server-side state surfaced by the OpenEnv state endpoint."""

    repo_source: str | None = Field(default=None)
    repo_name: str | None = Field(default=None)
    task_mode: str | None = Field(default=None)
    current_metrics: dict[str, Any] = Field(default_factory=dict)
    latest_tool_result: dict[str, Any] = Field(default_factory=dict)
    latest_reward_breakdown: dict[str, Any] = Field(default_factory=dict)
    setup_summary: dict[str, Any] = Field(default_factory=dict)
    allowed_tools: list[str] = Field(default_factory=lambda: list(REPO2ENV_TOOL_NAMES))
