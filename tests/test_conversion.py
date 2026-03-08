from pathlib import Path
import tempfile

from openenv.cli._cli_utils import validate_env_structure
from openenv.cli.commands.push import _prepare_staging_directory

from repo2env.conversion import analyze_repository, convert_repository, push_export, validate_export
from repo2env.conversion.converter import CommandResult
from repo2env.app.benchmark_prep import BenchmarkPrepResult


EXAMPLE_REPO = Path("examples/repo_a").resolve()
EXAMPLE_PR_REPO = Path("examples/repo_b").resolve()


def test_analyze_repository_returns_supported_spec() -> None:
    result = analyze_repository(EXAMPLE_REPO)

    assert result.spec.support.supported is True
    assert result.spec.repo_name == "repo_a"
    assert result.spec.runtime["allowed_tools"][0] == "list_files"
    assert result.spec.task["task_mode"] == "repo_editing"
    assert result.spec.validation.passing_tests >= 2
    assert result.spec.export["openenv_app"] == "repo2env.openenv.server:app"


def test_analyze_repository_preserves_pr_task_mode() -> None:
    result = analyze_repository(EXAMPLE_PR_REPO)

    assert result.spec.task["task_mode"] == "repo_editing"
    assert result.spec.runtime["allow_source_edits"] is True
    assert "write_file" in result.spec.runtime["allowed_tools"]


def test_convert_repository_exports_openenv_package(tmp_path: Path) -> None:
    export_dir = tmp_path / "repo_a-export"

    result = convert_repository(
        EXAMPLE_REPO,
        output_dir=export_dir,
        write_spec_to_repo=False,
        overwrite=False,
    )

    assert Path(result.export_dir, "openenv.yaml").exists()
    assert Path(result.export_dir, "Dockerfile").exists()
    assert Path(result.export_dir, ".repo2env", "rlenv_spec.json").exists()
    assert Path(result.export_dir, ".repo2env", "task_spec.json").exists()
    assert Path(result.export_dir, "server", "app.py").exists()
    assert Path(result.bundled_repo_dir, "billing.py").exists()
    dockerfile_text = Path(result.export_dir, "Dockerfile").read_text(encoding="utf-8")
    assert "RUN pip install --no-cache-dir -e ." in dockerfile_text
    assert "RUN pip install --no-cache-dir --no-build-isolation -e /app/examples/" not in dockerfile_text
    assert "RUN pip install --no-cache-dir -r /app/examples/" not in dockerfile_text


def test_validate_export_passes_for_generated_package(tmp_path: Path) -> None:
    export_dir = tmp_path / "repo_a-export"
    result = convert_repository(
        EXAMPLE_REPO,
        output_dir=export_dir,
        write_spec_to_repo=False,
        overwrite=False,
    )

    validation = validate_export(result.export_dir)

    assert validation.success is True
    assert "Ready for multi-mode deployment" in validation.stdout


def test_fresh_export_matches_openenv_push_preflight(tmp_path: Path) -> None:
    export_dir = tmp_path / "repo_a-export"
    result = convert_repository(
        EXAMPLE_REPO,
        output_dir=export_dir,
        write_spec_to_repo=False,
        overwrite=False,
    )
    export_path = Path(result.export_dir)

    warnings = validate_env_structure(export_path)

    assert warnings == []
    assert (export_path / "__init__.py").exists()
    assert (export_path / "client.py").exists()
    assert (export_path / "models.py").exists()
    assert (export_path / "server" / "Dockerfile").exists()
    export_readme = (export_path / "README.md").read_text(encoding="utf-8")
    assert export_readme.startswith("---\n")
    assert "base_path: /web" in export_readme
    assert "colorFrom: blue" in export_readme
    assert "colorTo: green" in export_readme

    with tempfile.TemporaryDirectory() as tmpdir:
        staging_dir = Path(tmpdir) / "staging"
        _prepare_staging_directory(
            export_path,
            str(result.export_name),
            staging_dir,
            enable_interface=True,
        )
        staged_readme = (staging_dir / "README.md").read_text(encoding="utf-8")

        assert (staging_dir / "Dockerfile").exists()
        assert not (staging_dir / "server" / "Dockerfile").exists()
        assert "base_path: /web" in staged_readme
        assert "colorFrom: blue" in staged_readme
        assert "colorTo: green" in staged_readme
        assert "#00C9FF" not in staged_readme


def test_push_export_defaults_to_headless(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: list[str], cwd: Path) -> CommandResult:
        captured["command"] = command
        captured["cwd"] = cwd
        return CommandResult(
            success=True,
            command=command,
            cwd=str(cwd),
            stdout="ok",
            stderr="",
            exit_code=0,
        )

    monkeypatch.setattr("repo2env.conversion.converter._run_openenv_command", fake_run)

    result = push_export(tmp_path / "export", repo_id="demo/demo-space")

    assert result.success is True
    assert captured["command"][1:] == [
        "push",
        str((tmp_path / "export").resolve()),
        "--repo-id",
        "demo/demo-space",
        "--no-interface",
    ]


def test_convert_repository_can_prepare_benchmark_for_repo_without_tests(tmp_path: Path) -> None:
    repo_dir = tmp_path / "demo_repo"
    repo_dir.mkdir()
    (repo_dir / "mathlib.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n",
        encoding="utf-8",
    )

    result = convert_repository(
        repo_dir,
        output_dir=tmp_path / "demo-export",
        write_spec_to_repo=False,
        overwrite=False,
        prepare_benchmark=True,
        benchmark_variant=0,
    )

    spec = result.spec.to_dict()

    assert result.export_name.endswith("-bug-0")
    assert spec["task"]["metadata"]["generated_tests"] == ["tests/test_repo2env_generated.py"]
    assert spec["task"]["metadata"]["injected_bug"]["path"] == "mathlib.py"
    assert spec["validation"]["failed_tests"] >= 1
    assert "Fix injected bug" in spec["task"]["title"]


def test_convert_repository_scopes_away_from_unrelated_failing_tests(tmp_path: Path) -> None:
    repo_dir = tmp_path / "scoped_repo"
    tests_dir = repo_dir / "tests"
    repo_dir.mkdir()
    tests_dir.mkdir()
    (repo_dir / "mathlib.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    (tests_dir / "test_mathlib.py").write_text(
        "from mathlib import add\n\n"
        "def test_add() -> None:\n"
        "    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    (tests_dir / "test_package.py").write_text(
        "def test_unrelated_packaging_failure() -> None:\n"
        "    assert False\n",
        encoding="utf-8",
    )

    result = convert_repository(
        repo_dir,
        output_dir=tmp_path / "scoped-export",
        write_spec_to_repo=False,
        overwrite=False,
        prepare_benchmark=True,
        benchmark_variant=0,
    )

    spec = result.spec.to_dict()

    assert spec["task"]["metadata"]["injected_bug"]["path"] == "mathlib.py"
    assert spec["task"]["test_targets"] == ["tests/test_mathlib.py"]
    assert spec["task"]["test_command"] == "python -m pytest tests/test_mathlib.py -q"
    assert spec["validation"]["failed_tests"] >= 1


def test_analyze_repository_rejects_clean_benchmark_when_injected_bug_required(monkeypatch) -> None:
    def fake_prepare_benchmark_repository(repo_root, manifest, install_recipe, task_spec, *, variant_index=0):
        return BenchmarkPrepResult(
            manifest=manifest,
            task_spec=task_spec,
            generated_test_files=[],
            injected_bug=None,
            warnings=["No deterministic bug candidate produced a failing benchmark. Exporting a clean task."],
        )

    monkeypatch.setattr(
        "repo2env.conversion.converter.prepare_benchmark_repository",
        fake_prepare_benchmark_repository,
    )

    try:
        analyze_repository(
            EXAMPLE_REPO,
            prepare_benchmark=True,
            require_injected_bug=True,
        )
    except ValueError as exc:
        assert "did not inject a source-code bug" in str(exc)
    else:
        raise AssertionError("Expected analyze_repository to reject a clean benchmark export")


def test_analyze_repository_rejects_top_level_shadowing_module_in_src_layout(tmp_path: Path) -> None:
    repo_dir = tmp_path / "shadow_repo"
    src_pkg = repo_dir / "src" / "humanize"
    tests_dir = repo_dir / "tests"
    src_pkg.mkdir(parents=True)
    tests_dir.mkdir()
    (repo_dir / "pyproject.toml").write_text(
        "[build-system]\n"
        'requires = ["setuptools>=61"]\n'
        'build-backend = "setuptools.build_meta"\n'
        "\n"
        "[project]\n"
        'name = "humanize-shadow"\n'
        'version = "0.1.0"\n',
        encoding="utf-8",
    )
    (src_pkg / "__init__.py").write_text("from .filesize import naturalsize\n", encoding="utf-8")
    (src_pkg / "filesize.py").write_text("def naturalsize(value):\n    return str(value)\n", encoding="utf-8")
    (repo_dir / "humanize.py").write_text("print('shadow')\n", encoding="utf-8")
    (tests_dir / "test_basic.py").write_text(
        "import humanize\n\n"
        "def test_smoke() -> None:\n"
        "    assert humanize.naturalsize(1) == '1'\n",
        encoding="utf-8",
    )

    try:
        analyze_repository(repo_dir, run_validation=False)
    except ValueError as exc:
        assert "shadows the `src/humanize` package" in str(exc)
    else:
        raise AssertionError("Expected analyze_repository to reject top-level import shadowing")
