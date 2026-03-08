from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from typing import Any, Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from repo2env.app.inference import (
    INVALID_ACTION_PENALTY,
    OpenAICompatiblePolicySession,
    safe_next_action,
    should_auto_submit_after_green_tests,
)
from repo2env.openenv.environment import Repo2EnvOpenEnvEnvironment
from repo2env.openenv.models import Repo2EnvAction

MAX_EVENT_CHARS = 1200


class InferenceSessionRequest(BaseModel):
    policy_kind: Literal["model", "baseline_test_submit"] = "model"
    model_name: str | None = Field(default=None, min_length=1)
    api_key: str | None = Field(default=None, min_length=1)
    api_base_url: str = Field(default="https://api.openai.com/v1", min_length=1)
    max_steps: int | None = Field(default=None, ge=1, le=100)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    timeout_s: float = Field(default=60.0, gt=0.0, le=600.0)


@dataclass(slots=True)
class UISessionRecord:
    session_id: str
    policy_kind: str
    model_name: str
    api_base_url: str
    max_steps: int | None
    revision: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "queued"
    status_message: str = "Waiting to start"
    repo_name: str | None = None
    task_mode: str | None = None
    current_metrics: dict[str, Any] = field(default_factory=dict)
    latest_pytest_summary: dict[str, Any] | None = None
    latest_reward_breakdown: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    final_state: dict[str, Any] | None = None
    error: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "policy_kind": self.policy_kind,
            "model_name": self.model_name,
            "api_base_url": self.api_base_url,
            "max_steps": self.max_steps,
            "revision": self.revision,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "status_message": self.status_message,
            "repo_name": self.repo_name,
            "task_mode": self.task_mode,
            "current_metrics": self.current_metrics,
            "latest_pytest_summary": self.latest_pytest_summary,
            "latest_reward_breakdown": self.latest_reward_breakdown,
            "events": list(self.events),
            "final_state": self.final_state,
            "error": self.error,
        }


class UISessionManager:
    def __init__(
        self,
        environment_factory: Callable[[], Repo2EnvOpenEnvEnvironment],
    ) -> None:
        self._environment_factory = environment_factory
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._sessions: dict[str, UISessionRecord] = {}

    def start_session(self, request: InferenceSessionRequest) -> UISessionRecord:
        if request.policy_kind == "model":
            if not request.model_name:
                raise ValueError("model_name is required for model runs.")
            if not request.api_key:
                raise ValueError("api_key is required for model runs.")
            model_name = request.model_name
        else:
            model_name = "baseline:test-submit"
        session = UISessionRecord(
            session_id=str(uuid4()),
            policy_kind=request.policy_kind,
            model_name=model_name,
            api_base_url=request.api_base_url,
            max_steps=request.max_steps,
        )
        with self._lock:
            self._sessions[session.session_id] = session
        worker = threading.Thread(
            target=self._run_session,
            args=(session.session_id, request),
            name=f"repo2env-ui-{session.session_id[:8]}",
            daemon=True,
        )
        worker.start()
        return session

    def get_session(self, session_id: str) -> UISessionRecord | None:
        with self._lock:
            return self._sessions.get(session_id)

    def wait_for_event(
        self,
        session_id: str,
        after_revision: int,
        timeout_s: float = 15.0,
    ) -> tuple[str, dict[str, Any] | None]:
        with self._condition:
            session = self._sessions.get(session_id)
            if session is None:
                return "missing", None
            if session.revision > after_revision:
                return "update", session.to_public_dict()
            notified = self._condition.wait(timeout=timeout_s)
            session = self._sessions.get(session_id)
            if session is None:
                return "missing", None
            if session.revision > after_revision:
                return "update", session.to_public_dict()
            if not notified:
                return "timeout", None
            return "timeout", None

    def _run_session(self, session_id: str, request: InferenceSessionRequest) -> None:
        env = self._environment_factory()
        try:
            policy = _create_policy(request)
            self._update_session(
                session_id,
                status="running",
                status_message="Resetting environment",
            )
            observation = env.reset(
                max_steps=request.max_steps,
            )
            self._record_reset(session_id, observation)

            while not observation.done:
                self._update_session(
                    session_id,
                    status="running",
                    status_message="Requesting next action from model",
                )
                action, action_penalty, action_error = safe_next_action(policy, observation)
                if action_error is not None:
                    self._append_event(
                        session_id,
                        {
                            "type": "invalid_action",
                            "step_count": observation.step_count,
                            "reward": action_penalty,
                            "error": _truncate_text(action_error, limit=MAX_EVENT_CHARS),
                            "fallback_tool": action.tool,
                            "current_metrics": observation.current_metrics,
                        },
                        status_message="Model returned an invalid action; applying penalty",
                    )
                if action.tool == "submit":
                    self._update_session(
                        session_id,
                        status="running",
                        status_message="Running final tests before submit",
                    )
                    observation = env.step(Repo2EnvAction(tool="run_tests", args={}))
                    self._record_step(session_id, Repo2EnvAction(tool="run_tests", args={}), observation)
                    if observation.done:
                        break
                    if should_auto_submit_after_green_tests(Repo2EnvAction(tool="run_tests", args={}), observation):
                        submit_action = Repo2EnvAction(tool="submit", args={})
                        self._update_session(
                            session_id,
                            status="running",
                            status_message="Final tests passed, submitting episode",
                        )
                        observation = env.step(submit_action)
                        self._record_step(session_id, submit_action, observation)
                        break
                    continue
                self._update_session(
                    session_id,
                    status="running",
                    status_message=f"Executing `{action.tool}`",
                )
                observation = env.step(action)
                self._record_step(session_id, action, observation)
                if observation.done:
                    break
                if should_auto_submit_after_green_tests(action, observation):
                    submit_action = Repo2EnvAction(tool="submit", args={})
                    self._update_session(
                        session_id,
                        status="running",
                        status_message="All tests passed, submitting episode",
                    )
                    observation = env.step(submit_action)
                    self._record_step(session_id, submit_action, observation)
                    break

            self._update_session(
                session_id,
                status="completed",
                status_message="Episode finished",
                final_state=env.state.model_dump(),
            )
        except Exception as exc:
            self._append_event(
                session_id,
                {
                    "type": "error",
                    "step_count": 0,
                    "error": _truncate_text(str(exc), limit=MAX_EVENT_CHARS),
                },
            )
            self._update_session(
                session_id,
                status="failed",
                status_message="Episode failed",
                error=str(exc),
            )
        finally:
            env.close()

    def _record_reset(self, session_id: str, observation: Any) -> None:
        event = {
            "type": "reset",
            "step_count": observation.step_count,
            "repo_name": observation.repo_name,
            "task_mode": observation.task_summary.get("task_mode"),
            "current_metrics": observation.current_metrics,
            "repo_summary": observation.repo_summary,
            "latest_pytest_summary": _summarize_pytest(observation.latest_pytest_summary),
        }
        self._append_event(
            session_id,
            event,
            repo_name=observation.repo_name,
            task_mode=observation.task_summary.get("task_mode"),
            current_metrics=observation.current_metrics,
            latest_pytest_summary=_summarize_pytest(observation.latest_pytest_summary),
            latest_reward_breakdown=observation.reward_breakdown,
            status_message="Running episode",
        )

    def _record_step(self, session_id: str, action: Repo2EnvAction, observation: Any) -> None:
        event = {
            "type": "step",
            "step_count": observation.step_count,
            "tool": action.tool,
            "args": _truncate_value(action.args),
            "reward": observation.reward,
            "done": observation.done,
            "tool_result": _truncate_value(observation.tool_result),
            "current_metrics": observation.current_metrics,
            "latest_pytest_summary": _summarize_pytest(observation.latest_pytest_summary),
            "reward_breakdown": observation.reward_breakdown,
        }
        self._append_event(
            session_id,
            event,
            repo_name=observation.repo_name,
            task_mode=observation.task_summary.get("task_mode"),
            current_metrics=observation.current_metrics,
            latest_pytest_summary=_summarize_pytest(observation.latest_pytest_summary),
            latest_reward_breakdown=observation.reward_breakdown,
            status_message=f"Completed `{action.tool}`",
            final_state=observation.metadata if observation.done else None,
        )

    def _append_event(self, session_id: str, event: dict[str, Any], **updates: Any) -> None:
        with self._condition:
            session = self._sessions[session_id]
            session.events.append(event)
            self._apply_updates(session, **updates)
            self._condition.notify_all()

    def _update_session(self, session_id: str, **updates: Any) -> None:
        with self._condition:
            session = self._sessions[session_id]
            self._apply_updates(session, **updates)
            self._condition.notify_all()

    @staticmethod
    def _apply_updates(session: UISessionRecord, **updates: Any) -> None:
        for key, value in updates.items():
            if value is not None:
                setattr(session, key, value)
        session.revision += 1
        session.updated_at = datetime.now(timezone.utc).isoformat()


def _summarize_pytest(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not summary:
        return summary
    result = dict(summary)
    output = result.get("output")
    if isinstance(output, str):
        result["output"] = _truncate_text(output, limit=MAX_EVENT_CHARS * 2)
    return result


def _truncate_value(value: Any) -> Any:
    if isinstance(value, str):
        return _truncate_text(value)
    if isinstance(value, dict):
        return {str(key): _truncate_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_truncate_value(item) for item in value]
    return value


def _truncate_text(value: str, *, limit: int = MAX_EVENT_CHARS) -> str:
    if len(value) <= limit:
        return value
    remainder = len(value) - limit
    return f"{value[:limit]}\n... [{remainder} chars truncated]"


class BaselineTestSubmitPolicy:
    def __init__(self) -> None:
        self._ran_tests = False

    def next_action(self, observation: Any) -> Repo2EnvAction:
        del observation
        if not self._ran_tests:
            self._ran_tests = True
            return Repo2EnvAction(tool="run_tests", args={})
        return Repo2EnvAction(tool="submit", args={})


def _create_policy(request: InferenceSessionRequest) -> Any:
    if request.policy_kind == "baseline_test_submit":
        return BaselineTestSubmitPolicy()
    return OpenAICompatiblePolicySession(
        model_name=request.model_name or "",
        api_base_url=request.api_base_url,
        api_key=request.api_key or "",
        temperature=request.temperature,
        timeout_s=request.timeout_s,
    )
