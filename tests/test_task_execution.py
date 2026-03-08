from pathlib import Path

from repo2env.app.env import Repo2EnvEnvironment
from repo2env.app.validator import run_coverage, run_tests
from repo2env.conversion import analyze_repository


def _build_scoped_repo(repo: Path) -> None:
    tests_dir = repo / "tests"
    repo.mkdir()
    tests_dir.mkdir()
    (repo / "sample.py").write_text(
        "def add(a, b):\n    return a + b\n\n\ndef broken() -> int:\n    return 1\n",
        encoding="utf-8",
    )
    (tests_dir / "test_ok.py").write_text(
        "from sample import add\n\n\ndef test_add() -> None:\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    (tests_dir / "test_fail.py").write_text(
        "from sample import broken\n\n\ndef test_broken() -> None:\n    assert broken() == 2\n",
        encoding="utf-8",
    )


def test_validator_honors_task_test_command(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _build_scoped_repo(repo)

    test_result = run_tests(repo, test_command="python -m pytest tests/test_ok.py -q")
    coverage_result = run_coverage(repo, test_command="python -m pytest tests/test_ok.py -q")

    assert test_result.passed == 1
    assert test_result.failed == 0
    assert coverage_result.exit_code == 0
    assert coverage_result.coverage_percent > 0


def test_env_and_analysis_honor_task_test_targets(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _build_scoped_repo(repo)
    (repo / "task_spec.json").write_text(
        """
{
  "spec_version": 1,
  "instance_id": "scoped-test-generation",
  "task_mode": "test_generation",
  "title": "Run only selected passing tests",
  "problem_statement": "Focus on the passing subset while ignoring the unrelated failing test.",
  "test_targets": ["tests/test_ok.py"],
  "starter_paths": ["sample.py", "tests/test_ok.py"],
  "allow_source_edits": false
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    analysis = analyze_repository(repo)
    env = Repo2EnvEnvironment(repo)
    try:
        observation = env.reset()
    finally:
        env.close()

    assert analysis.spec.validation.passing_tests == 1
    assert analysis.spec.validation.failed_tests == 0
    assert observation["latest_pytest_summary"]["passed"] == 1
    assert observation["latest_pytest_summary"]["failed"] == 0


def test_env_replace_in_file_supports_small_source_edits(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _build_scoped_repo(repo)
    (repo / "task_spec.json").write_text(
        """
{
  "spec_version": 1,
  "instance_id": "repo-editing-fix",
  "task_mode": "repo_editing",
  "title": "Fix the broken function",
  "problem_statement": "Edit sample.py so the scoped test passes.",
  "test_targets": ["tests/test_fail.py"],
  "starter_paths": ["sample.py", "tests/test_fail.py"],
  "allow_source_edits": true
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    env = Repo2EnvEnvironment(repo)
    try:
        observation = env.reset()
        assert "replace_in_file" in observation["allowed_tools"]

        observation, reward, done, info = env.step(
            {
                "tool": "replace_in_file",
                "args": {
                    "path": "sample.py",
                    "old_text": "return 1",
                    "new_text": "return 2",
                },
            }
        )
        assert done is False
        assert reward < 0
        assert info["tool_result"]["replacements"] == 1

        observation, reward, done, info = env.step({"tool": "run_tests", "args": {}})
        assert done is False
        assert reward > 0
        assert observation["latest_pytest_summary"]["failed"] == 0
        assert info["metrics"]["passing_tests"] == 1
    finally:
        env.close()


def test_repo_editing_support_tools_help_debug_and_modify_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _build_scoped_repo(repo)
    (repo / "task_spec.json").write_text(
        """
{
  "spec_version": 1,
  "instance_id": "repo-editing-tools",
  "task_mode": "repo_editing",
  "title": "Use repo editing support tools",
  "problem_statement": "Inspect the failing test, edit sample.py, and inspect the resulting diff.",
  "test_targets": ["tests/test_fail.py"],
  "starter_paths": ["sample.py", "tests/test_fail.py"],
  "allow_source_edits": true
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    env = Repo2EnvEnvironment(repo)
    try:
        observation = env.reset()
        assert observation["latest_pytest_summary"]["failed"] == 1

        observation, _, _, info = env.step(
            {"tool": "read_file_chunk", "args": {"path": "sample.py", "start_line": 5, "end_line": 6}}
        )
        assert info["tool_result"]["content"] == "def broken() -> int:\n    return 1"

        observation, _, _, info = env.step({"tool": "get_test_failures", "args": {}})
        failures = info["tool_result"]["failures"]
        assert failures[0]["node_id"] == "tests/test_fail.py::test_broken"
        assert "assert broken() == 2" in failures[0]["assertion_line"]

        observation, _, _, info = env.step(
            {
                "tool": "insert_after",
                "args": {
                    "path": "sample.py",
                    "anchor_text": "return 1",
                    "new_text": "\n# patched",
                },
            }
        )
        assert info["tool_result"]["insertions"] == 1

        observation, _, _, info = env.step(
            {
                "tool": "insert_before",
                "args": {
                    "path": "sample.py",
                    "anchor_text": "# patched",
                    "new_text": "# note\n",
                },
            }
        )
        assert info["tool_result"]["insertions"] == 1

        observation, _, _, info = env.step(
            {"tool": "append_to_file", "args": {"path": "sample.py", "content": "\n# trailer\n"}}
        )
        assert info["tool_result"]["bytes_appended"] > 0

        observation, _, _, info = env.step(
            {
                "tool": "replace_in_file",
                "args": {
                    "path": "sample.py",
                    "old_text": "return 1",
                    "new_text": "return 2",
                },
            }
        )
        assert info["tool_result"]["replacements"] == 1

        observation, reward, _, info = env.step({"tool": "run_tests", "args": {}})
        assert reward > 0
        assert observation["latest_pytest_summary"]["failed"] == 0

        observation, _, _, info = env.step({"tool": "diff_working_tree", "args": {}})
        diff_payload = info["tool_result"]
        assert diff_payload["count"] >= 1
        assert any(entry["path"] == "sample.py" for entry in diff_payload["changed_files"])
        sample_diff = next(entry["diff"] for entry in diff_payload["changed_files"] if entry["path"] == "sample.py")
        assert "return 2" in sample_diff
    finally:
        env.close()
