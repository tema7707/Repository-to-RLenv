from __future__ import annotations

import argparse
from pathlib import Path

from repo2env.app.agent_runner import HeuristicAgentRunner
from repo2env.app.env import Repo2EnvEnvironment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Repo2Env demo episode.")
    parser.add_argument("--repo", required=True, help="Local path or GitHub URL for the target repository.")
    parser.add_argument("--max-steps", type=int, default=12, help="Maximum steps per episode.")
    parser.add_argument("--coverage-target", type=float, default=75.0, help="Coverage target for early stopping.")
    parser.add_argument(
        "--output-dir",
        default=str(Path("outputs/trajectories")),
        help="Directory for trajectory JSON files.",
    )
    args = parser.parse_args()

    env = Repo2EnvEnvironment(
        args.repo,
        max_steps=args.max_steps,
        coverage_target=args.coverage_target,
    )
    runner = HeuristicAgentRunner(args.output_dir)

    try:
        episode = runner.run_episode(env)
    finally:
        env.close()

    print(f"repo={episode['repo']}")
    print(f"task_mode={episode['task']['task_mode']}")
    print(f"episode_return={episode['episode_return']}")
    print(f"final_passing_tests={episode['final_metrics']['passing_tests']}")
    print(f"final_fail_to_pass_passed={episode['final_metrics']['fail_to_pass_passed']}")
    print(f"final_coverage_percent={episode['final_metrics']['coverage_percent']}")
    print(f"trajectory_path={episode['trajectory_path']}")


if __name__ == "__main__":
    main()
