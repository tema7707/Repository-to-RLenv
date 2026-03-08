from __future__ import annotations

from pathlib import Path

from repo2env.app.env import Repo2EnvEnvironment
from repo2env.app.install import detect_install_recipe
from repo2env.app.manifest import build_manifest
from repo2env.conversion import analyze_repository


SETUP_PY = """
from setuptools import setup

setup(name='samplepkg', version='0.1.0', packages=['samplepkg'])
""".strip() + "\n"


def test_analyze_repository_runs_detected_install_recipe(tmp_path: Path) -> None:
    repo = tmp_path / "samplepkg_repo"
    package_dir = repo / "samplepkg"
    tests_dir = repo / "tests"
    package_dir.mkdir(parents=True)
    tests_dir.mkdir()
    (repo / "setup.py").write_text(SETUP_PY, encoding="utf-8")
    (package_dir / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tests_dir / "test_package.py").write_text(
        "from importlib.metadata import version\n\n"
        "def test_package_version() -> None:\n"
        "    assert version('samplepkg') == '0.1.0'\n",
        encoding="utf-8",
    )

    result = analyze_repository(repo)

    assert result.spec.runtime["install_recipe"]["strategy"] == "editable"
    assert result.spec.validation.setup_succeeded is True
    assert result.spec.validation.failed_tests == 0


def test_detect_install_recipe_installs_pyproject_build_requirements_first(tmp_path: Path) -> None:
    repo = tmp_path / "poetry_repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        "[build-system]\n"
        'requires = ["poetry-core>=1.9.0"]\n'
        'build-backend = "poetry.core.masonry.api"\n',
        encoding="utf-8",
    )
    (repo / "pkg.py").write_text("VALUE = 1\n", encoding="utf-8")

    manifest = build_manifest(repo)
    recipe = detect_install_recipe(repo, manifest)

    assert recipe.commands[0] == "pip install --no-cache-dir 'poetry-core>=1.9.0'"
    assert recipe.commands[1] == "pip install --no-cache-dir --no-build-isolation -e {repo_dir}"


def test_env_reset_loads_repo_editing_tools() -> None:
    env = Repo2EnvEnvironment(Path("examples/repo_b").resolve())
    try:
        observation = env.reset()
    finally:
        env.close()

    assert observation["task_summary"]["task_mode"] == "repo_editing"
    assert observation["task_summary"]["allow_source_edits"] is True
    assert "replace_in_file" in observation["allowed_tools"]
    assert "append_to_file" in observation["allowed_tools"]
    assert "insert_after" in observation["allowed_tools"]
    assert "insert_before" in observation["allowed_tools"]
    assert "read_file_chunk" in observation["allowed_tools"]
    assert "get_test_failures" in observation["allowed_tools"]
    assert "diff_working_tree" in observation["allowed_tools"]
    assert "write_file" in observation["allowed_tools"]
    assert "replace_in_file" in observation["tool_schemas"]
    assert "get_test_failures" in observation["tool_schemas"]
    assert observation["latest_pytest_summary"]["passed"] >= 0
