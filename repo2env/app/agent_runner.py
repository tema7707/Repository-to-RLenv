from __future__ import annotations

from pathlib import Path
from typing import Any

from repo2env.app.env import DEFAULT_TRAJECTORY_DIR, Repo2EnvEnvironment
from repo2env.app.logger import TrajectoryLogger


class HeuristicAgentRunner:
    """Simple runner for the simplified Repo2Env benchmark env."""

    def __init__(self, output_dir: str | Path = DEFAULT_TRAJECTORY_DIR) -> None:
        self.logger = TrajectoryLogger(output_dir)

    def run_episode(self, env: Repo2EnvEnvironment) -> dict[str, Any]:
        observation = env.reset()
        latest_observation = observation
        steps: list[dict[str, Any]] = []
        episode_done = False

        def take(action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
            nonlocal latest_observation, episode_done
            next_observation, reward, done, info = env.step(action)
            latest_observation = next_observation
            episode_done = done
            steps.append(
                {
                    "observation": next_observation,
                    "action": action,
                    "tool_result": info["tool_result"],
                    "reward": reward,
                    "reward_breakdown": info["reward_breakdown"],
                }
            )
            return next_observation, reward, done, info

        take({"tool": "list_files", "args": {"limit": 50}})
        for bucket in (observation.get("selected_source_files", []), observation.get("selected_test_files", [])):
            if not bucket:
                continue
            path = bucket[0].get("path")
            if path:
                _, _, done, _ = take({"tool": "read_file", "args": {"path": path}})
                if done:
                    return self._finalize_episode(env, steps, latest_observation)

        if not episode_done:
            take({"tool": "run_tests", "args": {}})
        if not episode_done:
            take({"tool": "submit", "args": {}})

        return self._finalize_episode(env, steps, latest_observation)

    def _finalize_episode(
        self,
        env: Repo2EnvEnvironment,
        steps: list[dict[str, Any]],
        latest_observation: dict[str, Any],
    ) -> dict[str, Any]:
        episode = env.export_episode()
        episode["steps"] = steps
        episode["final_observation"] = latest_observation
        episode["episode_return"] = round(sum(step["reward"] for step in steps), 2)
        trajectory_path = self.logger.write(episode["repo"], episode)
        episode["trajectory_path"] = str(trajectory_path)
        return episode
