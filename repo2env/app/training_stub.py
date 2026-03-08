from __future__ import annotations

import argparse
import json
from pathlib import Path

from repo2env.app.agent_runner import HeuristicAgentRunner
from repo2env.app.env import Repo2EnvEnvironment


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect reward trajectories across Repo2Env episodes.")
    parser.add_argument("--repo", action="append", required=True, help="Repeatable repo path or GitHub URL.")
    parser.add_argument("--episodes", type=int, default=2, help="Episodes to run per repo.")
    parser.add_argument(
        "--output",
        default=str(Path("outputs/trajectories") / "reward_progression.json"),
        help="Path for reward progression JSON.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runner = HeuristicAgentRunner(output_path.parent)

    results = []
    for repo in args.repo:
        for episode_index in range(args.episodes):
            env = Repo2EnvEnvironment(repo)
            try:
                episode = runner.run_episode(env)
            finally:
                env.close()
            results.append(
                {
                    "repo": repo,
                    "episode_index": episode_index,
                    "episode_return": episode["episode_return"],
                    "final_metrics": episode["final_metrics"],
                    "trajectory_path": episode["trajectory_path"],
                }
            )

    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote={output_path}")


if __name__ == "__main__":
    main()

