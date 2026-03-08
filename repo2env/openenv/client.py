from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from repo2env.openenv.models import Repo2EnvAction, Repo2EnvObservation, Repo2EnvState


class Repo2EnvClient(EnvClient[Repo2EnvAction, Repo2EnvObservation, Repo2EnvState]):
    """Typed OpenEnv client for Repo2Env's websocket environment API."""

    def _step_payload(self, action: Repo2EnvAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[Repo2EnvObservation]:
        observation_data = payload.get("observation", {})
        observation = Repo2EnvObservation.model_validate(
            {
                **observation_data,
                "done": payload.get("done", observation_data.get("done", False)),
                "reward": payload.get("reward", observation_data.get("reward")),
                "metadata": observation_data.get("metadata", {}),
            }
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> Repo2EnvState:
        return Repo2EnvState.model_validate(payload)

