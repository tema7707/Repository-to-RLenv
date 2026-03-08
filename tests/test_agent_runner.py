from pathlib import Path

from repo2env.app.agent_runner import HeuristicAgentRunner
from repo2env.app.env import Repo2EnvEnvironment


def test_agent_runner_records_repo_a_episode(tmp_path: Path) -> None:
    env = Repo2EnvEnvironment(Path("examples/repo_a").resolve())
    runner = HeuristicAgentRunner(tmp_path)

    try:
        episode = runner.run_episode(env)
    finally:
        env.close()

    assert episode["task"]["task_mode"] == "repo_editing"
    assert episode["final_metrics"]["passing_tests"] >= episode["initial_metrics"]["passing_tests"]
    assert any(step["action"]["tool"] == "run_tests" for step in episode["steps"])
    assert Path(episode["trajectory_path"]).exists()


def test_agent_runner_records_repo_b_episode(tmp_path: Path) -> None:
    env = Repo2EnvEnvironment(Path("examples/repo_b").resolve())
    runner = HeuristicAgentRunner(tmp_path)

    try:
        episode = runner.run_episode(env)
    finally:
        env.close()

    assert episode["task"]["task_mode"] == "repo_editing"
    assert episode["initial_metrics"]["failed_tests"] >= 0
    assert any(step["action"]["tool"] == "run_tests" for step in episode["steps"])
    assert Path(episode["trajectory_path"]).exists()
