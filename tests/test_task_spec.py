from pathlib import Path

import pytest

from repo2env.app.install import InstallRecipe
from repo2env.app.manifest import build_manifest
from repo2env.app.task_spec import (
    TaskSpecValidationError,
    build_task_spec,
    load_task_spec,
)


def test_load_task_spec_rejects_unknown_fields(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    tests_dir = repo / "tests"
    repo.mkdir()
    tests_dir.mkdir()
    (repo / "sample.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (tests_dir / "test_sample.py").write_text(
        "from sample import add\n\n\ndef test_add() -> None:\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    (repo / "task_spec.json").write_text(
        '{"task_type":"test_generation","description":"invalid legacy schema"}',
        encoding="utf-8",
    )

    manifest = build_manifest(repo)

    with pytest.raises(TaskSpecValidationError, match="Unknown task_spec.json fields"):
        load_task_spec(repo, manifest, InstallRecipe(strategy="none"))


def test_build_task_spec_normalizes_to_repo_editing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    tests_dir = repo / "tests"
    repo.mkdir()
    tests_dir.mkdir()
    (repo / "sample.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (tests_dir / "test_sample.py").write_text(
        "from sample import add\n\n\ndef test_add() -> None:\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )

    manifest = build_manifest(repo)
    install_recipe = InstallRecipe(strategy="none")

    spec = build_task_spec(
        manifest,
        install_recipe,
        task_mode="pr_fix",
        title="Fix sample repo",
        problem_statement="Edit files and improve the number of passing tests.",
        starter_paths=["sample.py", "tests/test_sample.py"],
        allow_source_edits=True,
    )

    assert spec.task_mode == "repo_editing"
    assert spec.allow_source_edits is True
    assert spec.starter_paths == ["sample.py", "tests/test_sample.py"]


def test_build_task_spec_allows_custom_install_commands(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    tests_dir = repo / "tests"
    repo.mkdir()
    tests_dir.mkdir()
    (repo / "sample.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (tests_dir / "test_sample.py").write_text(
        "from sample import add\n\n\ndef test_add() -> None:\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )

    manifest = build_manifest(repo)
    install_recipe = InstallRecipe(strategy="editable", commands=["pip install -e {repo_dir}"])

    spec = build_task_spec(
        manifest,
        install_recipe,
        title="Target sample helpers",
        problem_statement="Edit files and raise the passing test count.",
        starter_paths=["sample.py", "tests/test_sample.py"],
        install_commands=["pip install --no-cache-dir 'setuptools>=77'", "pip install -e {repo_dir}"],
        metadata={"demo_role": "credibility"},
    )

    assert spec.install_config.commands[0] == "pip install --no-cache-dir 'setuptools>=77'"
    assert spec.starter_paths == ["sample.py", "tests/test_sample.py"]
    assert spec.metadata["demo_role"] == "credibility"


def test_load_task_spec_strips_stop_on_first_failure_flags(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    tests_dir = repo / "tests"
    repo.mkdir()
    tests_dir.mkdir()
    (repo / "sample.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (tests_dir / "test_sample.py").write_text(
        "from sample import add\n\n\ndef test_add() -> None:\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    (repo / "task_spec.json").write_text(
        """{
  "spec_version": 1,
  "instance_id": "demo",
  "task_mode": "repo_editing",
  "title": "Demo",
  "problem_statement": "Demo",
  "test_command": "python -m pytest tests/test_sample.py -x --maxfail=1 -q"
}""",
        encoding="utf-8",
    )

    manifest = build_manifest(repo)
    spec = load_task_spec(repo, manifest, InstallRecipe(strategy="none"))

    assert spec.test_command == "python -m pytest tests/test_sample.py -q"


def test_load_task_spec_strips_exitfirst_from_combined_short_flags(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    tests_dir = repo / "tests"
    repo.mkdir()
    tests_dir.mkdir()
    (repo / "sample.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (tests_dir / "test_sample.py").write_text(
        "from sample import add\n\n\ndef test_add() -> None:\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    (repo / "task_spec.json").write_text(
        """{
  "spec_version": 1,
  "instance_id": "demo",
  "task_mode": "repo_editing",
  "title": "Demo",
  "problem_statement": "Demo",
  "test_command": "python -m pytest tests/test_sample.py::test_add -xvs"
}""",
        encoding="utf-8",
    )

    manifest = build_manifest(repo)
    spec = load_task_spec(repo, manifest, InstallRecipe(strategy="none"))

    assert spec.test_command == "python -m pytest tests/test_sample.py::test_add -vs"
