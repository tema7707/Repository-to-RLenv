from __future__ import annotations

from dataclasses import asdict, dataclass

STEP_PENALTY = 0.2


@dataclass(slots=True)
class EpisodeMetrics:
    passing_tests: int = 0
    failed_tests: int = 0
    invalid_tests: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_reward(previous: EpisodeMetrics, current: EpisodeMetrics) -> tuple[float, dict]:
    delta_passing_tests = current.passing_tests - previous.passing_tests
    reward = float(delta_passing_tests) - STEP_PENALTY
    breakdown = {
        "delta_passing_tests": delta_passing_tests,
        "step_penalty": STEP_PENALTY,
    }
    return round(reward, 2), breakdown
