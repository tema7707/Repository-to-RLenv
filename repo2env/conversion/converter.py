from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from repo2env.app.ingest import StagedRepository, ingest_repository
from repo2env.app.benchmark_prep import prepare_benchmark_repository
from repo2env.app.install import InstallRecipe, detect_install_recipe, prepare_runtime
from repo2env.app.manifest import RepoManifest
from repo2env.app.task_spec import TaskSpec, load_task_spec, write_task_spec
from repo2env.app.validator import run_coverage, run_tests
from repo2env.openenv.models import REPO2ENV_TOOL_NAMES

from .spec import Repo2EnvSpec, SupportAssessment, ValidationSummary, write_spec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REWARD_WEIGHTS = {
    "delta_passing_tests": 1.0,
    "step_penalty": 0.2,
}
SERVICE_MARKERS = (
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
    "manage.py",
    "alembic.ini",
)
EXPORT_REPO_SPEC_PATH = Path(".repo2env") / "rlenv_spec.json"
EXPORT_TASK_SPEC_PATH = Path(".repo2env") / "task_spec.json"
EXPORT_REPO_DIR = Path("examples")
RUNTIME_COPY_DIRS = ("repo2env", "server")
RUNTIME_COPY_FILES = ("pyproject.toml", "uv.lock")


@dataclass(slots=True)
class AnalysisResult:
    repo_source: str
    local_repo_path: str | None
    spec: Repo2EnvSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_source": self.repo_source,
            "local_repo_path": self.local_repo_path,
            "spec": self.spec.to_dict(),
        }


@dataclass(slots=True)
class ConversionResult:
    repo_source: str
    export_name: str
    export_dir: str
    bundled_repo_dir: str
    export_spec_path: str
    export_task_spec_path: str
    repo_spec_path: str | None
    repo_task_spec_path: str | None
    validate_command: str
    push_command: str
    spec: Repo2EnvSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_source": self.repo_source,
            "export_name": self.export_name,
            "export_dir": self.export_dir,
            "bundled_repo_dir": self.bundled_repo_dir,
            "export_spec_path": self.export_spec_path,
            "export_task_spec_path": self.export_task_spec_path,
            "repo_spec_path": self.repo_spec_path,
            "repo_task_spec_path": self.repo_task_spec_path,
            "validate_command": self.validate_command,
            "push_command": self.push_command,
            "spec": self.spec.to_dict(),
        }


@dataclass(slots=True)
class CommandResult:
    success: bool
    command: list[str]
    cwd: str
    stdout: str
    stderr: str
    exit_code: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_repository(
    source: str | Path,
    *,
    run_validation: bool = True,
    max_steps: int = 12,
    coverage_target: float = 75.0,
    prepare_benchmark: bool = False,
    benchmark_variant: int = 0,
    require_injected_bug: bool = False,
) -> AnalysisResult:
    source_text = str(source)
    local_repo_path = _local_repo_path(source_text)
    staged = ingest_repository(source_text)
    try:
        install_recipe = detect_install_recipe(staged.staged_path, staged.manifest)
        task_spec = load_task_spec(staged.staged_path, staged.manifest, install_recipe)
        shadowing_errors = _shadowing_module_errors(staged.staged_path, staged.manifest)
        if shadowing_errors:
            raise ValueError(" ".join(shadowing_errors))
        benchmark_warnings: list[str] = []
        if prepare_benchmark:
            prep = prepare_benchmark_repository(
                staged.staged_path,
                staged.manifest,
                install_recipe,
                task_spec,
                variant_index=benchmark_variant,
            )
            staged.manifest = prep.manifest
            task_spec = prep.task_spec
            benchmark_warnings = list(prep.warnings)
            if require_injected_bug and prep.injected_bug is None:
                raise ValueError(
                    "Benchmark preparation did not inject a source-code bug. "
                    "Refusing to export a clean benchmark task."
                )
        validation = (
            _collect_validation(staged.staged_path, staged.manifest, task_spec)
            if run_validation
            else ValidationSummary(ran=False)
        )
        support = assess_support(
            staged.staged_path,
            staged.manifest,
            task_spec.install_config,
            validation,
            extra_warnings=benchmark_warnings,
        )
        export_name = build_export_name(
            staged.manifest.repo_name,
            benchmark_variant=benchmark_variant if prepare_benchmark else None,
        )
        bundled_repo_name = build_bundle_name(staged.manifest.repo_name)
        repo_summary = build_repo_summary(staged.staged_path, staged.manifest)
        allowed_tools = list(REPO2ENV_TOOL_NAMES)
        spec = Repo2EnvSpec(
            spec_version=1,
            repo_name=staged.manifest.repo_name,
            repo_source=source_text,
            source_kind="local_path" if local_repo_path else "git_url",
            support=support,
            manifest=staged.manifest.to_dict(),
            task=task_spec.to_dict(),
            runtime={
                "task_mode": task_spec.task_mode,
                "allowed_tools": allowed_tools,
                "default_reset_parameters": {
                    "max_steps": max_steps,
                },
                "test_command": task_spec.test_command,
                "install_recipe": task_spec.install_config.to_dict(),
                "allow_source_edits": task_spec.allow_source_edits,
            },
            reward=REWARD_WEIGHTS,
            observation={
                "includes": [
                    "repo_summary",
                    "task_summary",
                    "setup_summary",
                    "selected_source_files",
                    "selected_test_files",
                    "latest_pytest_summary",
                    "recent_action_history",
                    "step_count",
                ],
            },
            hints={
                "repo_summary": repo_summary,
                "readme_excerpt": readme_excerpt(staged.staged_path),
                "candidate_source_files": staged.manifest.source_files[:8],
                "candidate_test_files": staged.manifest.test_files[:8],
                "pytest_config_files": staged.manifest.pytest_config_files,
                "starter_paths": task_spec.starter_paths,
                "benchmark_mode": "auto_bug" if prepare_benchmark else "snapshot",
                "injected_bug": task_spec.metadata.get("injected_bug"),
                "generated_tests": task_spec.metadata.get("generated_tests", []),
            },
            validation=validation,
            export={
                "openenv_name": export_name,
                "bundled_repo_dir": f"examples/{bundled_repo_name}",
                "openenv_app": "repo2env.openenv.server:app",
                "hf_space_repo_id_suggestion": f"<hf-username>/{export_name}",
            },
        )
        return AnalysisResult(
            repo_source=source_text,
            local_repo_path=str(local_repo_path) if local_repo_path else None,
            spec=spec,
        )
    finally:
        cleanup_staged_repository(staged)


def convert_repository(
    source: str | Path,
    *,
    output_dir: str | Path | None = None,
    write_spec_to_repo: bool = True,
    overwrite: bool = False,
    run_validation: bool = True,
    max_steps: int = 12,
    coverage_target: float = 75.0,
    prepare_benchmark: bool = False,
    benchmark_variant: int = 0,
    require_injected_bug: bool = False,
) -> ConversionResult:
    analysis = analyze_repository(
        source,
        run_validation=run_validation,
        max_steps=max_steps,
        coverage_target=coverage_target,
        prepare_benchmark=prepare_benchmark,
        benchmark_variant=benchmark_variant,
        require_injected_bug=require_injected_bug,
    )
    source_text = str(source)
    local_repo_path = _local_repo_path(source_text)
    export_name = str(analysis.spec.export["openenv_name"])
    export_dir_path = resolve_output_dir(source_text, export_name, output_dir)

    if export_dir_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Export directory already exists: {export_dir_path}. Pass overwrite=True to replace it."
            )
        shutil.rmtree(export_dir_path)

    staged = ingest_repository(source_text)
    try:
        bundled_repo_name = str(analysis.spec.export["bundled_repo_dir"]).split("/", maxsplit=1)[-1]
        bundled_repo_dir = export_dir_path / EXPORT_REPO_DIR / bundled_repo_name
        _write_export_package(
            export_dir=export_dir_path,
            bundled_repo_dir=bundled_repo_dir,
            staged=staged,
            spec=analysis.spec,
        )
    finally:
        cleanup_staged_repository(staged)

    repo_spec_path: Path | None = None
    repo_task_spec_path: Path | None = None
    task_spec = TaskSpec.from_dict(analysis.spec.task)
    if write_spec_to_repo and local_repo_path is not None:
        repo_spec_path = write_spec(analysis.spec, local_repo_path / EXPORT_REPO_SPEC_PATH)
        repo_task_spec_path = write_task_spec(task_spec, local_repo_path / EXPORT_TASK_SPEC_PATH)

    validate_command = f"cd {export_dir_path} && {resolve_openenv_cli()} validate"
    push_command = f"cd {export_dir_path} && {resolve_openenv_cli()} push"
    export_spec_path = write_spec(analysis.spec, export_dir_path / EXPORT_REPO_SPEC_PATH)
    export_task_spec_path = write_task_spec(task_spec, export_dir_path / EXPORT_TASK_SPEC_PATH)

    return ConversionResult(
        repo_source=source_text,
        export_name=export_name,
        export_dir=str(export_dir_path),
        bundled_repo_dir=str(bundled_repo_dir),
        export_spec_path=str(export_spec_path),
        export_task_spec_path=str(export_task_spec_path),
        repo_spec_path=str(repo_spec_path) if repo_spec_path else None,
        repo_task_spec_path=str(repo_task_spec_path) if repo_task_spec_path else None,
        validate_command=validate_command,
        push_command=push_command,
        spec=analysis.spec,
    )


def validate_export(export_dir: str | Path) -> CommandResult:
    export_path = Path(export_dir).expanduser().resolve()
    return _run_openenv_command([resolve_openenv_cli(), "validate"], export_path)


def push_export(
    export_dir: str | Path,
    *,
    repo_id: str | None = None,
    private: bool = False,
    base_image: str | None = None,
    enable_interface: bool = False,
) -> CommandResult:
    export_path = Path(export_dir).expanduser().resolve()
    command = [resolve_openenv_cli(), "push", str(export_path)]
    if repo_id:
        command.extend(["--repo-id", repo_id])
    if private:
        command.append("--private")
    if base_image:
        command.extend(["--base-image", base_image])
    if not enable_interface:
        command.append("--no-interface")
    return _run_openenv_command(command, export_path)


def convert_and_push_repository(
    source: str | Path,
    *,
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
    private: bool = False,
    overwrite: bool = False,
    run_validation: bool = True,
    write_spec_to_repo: bool = True,
    max_steps: int = 12,
    coverage_target: float = 75.0,
    base_image: str | None = None,
    enable_interface: bool = False,
    prepare_benchmark: bool = False,
    benchmark_variant: int = 0,
    require_injected_bug: bool = False,
) -> dict[str, Any]:
    conversion = convert_repository(
        source,
        output_dir=output_dir,
        write_spec_to_repo=write_spec_to_repo,
        overwrite=overwrite,
        run_validation=run_validation,
        max_steps=max_steps,
        coverage_target=coverage_target,
        prepare_benchmark=prepare_benchmark,
        benchmark_variant=benchmark_variant,
        require_injected_bug=require_injected_bug,
    )
    push_result = push_export(
        conversion.export_dir,
        repo_id=repo_id,
        private=private,
        base_image=base_image,
        enable_interface=enable_interface,
    )
    return {
        "conversion": conversion.to_dict(),
        "push": push_result.to_dict(),
    }


def assess_support(
    repo_root: Path,
    manifest: RepoManifest,
    install_recipe: InstallRecipe,
    validation: ValidationSummary,
    *,
    extra_warnings: list[str] | None = None,
) -> SupportAssessment:
    reasons: list[str] = []
    warnings: list[str] = []

    if not manifest.python_files:
        reasons.append("No Python files detected.")
    if len(manifest.python_files) > 400:
        warnings.append("Large Python repo detected. Repo2Env is tuned for small deterministic repos.")
    if not manifest.test_files:
        warnings.append("No tests detected. The environment can still generate tests, but baseline signals will be weak.")
    if install_recipe.strategy == "none":
        warnings.append("No dependency install recipe detected for the exported HF environment.")
    warnings.extend(_service_marker_warnings(repo_root))
    warnings.extend(_shadowing_module_errors(repo_root, manifest))
    warnings.extend(install_recipe.notes)
    if validation.setup_error:
        warnings.append(f"Setup failed during analysis: {validation.setup_error}")
    if validation.validation_error:
        warnings.append(f"Validation failed during analysis: {validation.validation_error}")
    elif validation.ran and validation.failed_tests > 0:
        warnings.append("Baseline pytest run already has failing or erroring tests.")
    if extra_warnings:
        warnings.extend(extra_warnings)
    return SupportAssessment(supported=not reasons, reasons=reasons, warnings=sorted(set(warnings)))


def build_export_name(repo_name: str, *, benchmark_variant: int | None = None) -> str:
    slug = build_bundle_name(repo_name)
    if benchmark_variant is None:
        return f"repo2env-{slug}"
    return f"repo2env-{slug}-bug-{benchmark_variant}"


def _shadowing_module_errors(repo_root: Path, manifest: RepoManifest) -> list[str]:
    if manifest.layout != "src":
        return []

    package_names = {
        Path(package_path).parts[1]
        for package_path in manifest.packages
        if len(Path(package_path).parts) >= 2 and Path(package_path).parts[0] == "src"
    }
    errors: list[str] = []
    for package_name in sorted(package_names):
        shadow_path = repo_root / f"{package_name}.py"
        if shadow_path.is_file():
            errors.append(
                f"Top-level module `{package_name}.py` shadows the `src/{package_name}` package. "
                "Benchmark exports must edit existing package files instead of creating import-shadowing modules."
            )
    return errors


def build_bundle_name(repo_name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in repo_name).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned or "repo"


def resolve_output_dir(source: str, export_name: str, output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()

    local_repo_path = _local_repo_path(source)
    if local_repo_path is not None:
        return (local_repo_path / ".repo2env" / "exports" / export_name).resolve()
    return (Path.cwd() / "outputs" / export_name).resolve()


def resolve_openenv_cli() -> str:
    candidate = Path(sys.executable).with_name("openenv")
    if candidate.exists():
        return str(candidate)
    return "openenv"


def cleanup_staged_repository(staged: StagedRepository) -> None:
    staged_root = staged.staged_path.parent
    if staged_root.exists():
        shutil.rmtree(staged_root, ignore_errors=True)


def build_repo_summary(repo_root: Path, manifest: RepoManifest) -> str:
    readme = readme_excerpt(repo_root)
    summary = (
        f"{manifest.repo_name} is a {manifest.layout} Python repo with "
        f"{len(manifest.source_files)} source files and {len(manifest.test_files)} test files."
    )
    if readme:
        return f"{summary} README: {readme}"
    return summary


def readme_excerpt(repo_root: Path, limit: int = 280) -> str:
    for candidate in ("README.md", "README.rst", "README.txt", "readme.md"):
        path = repo_root / candidate
        if path.is_file():
            text = " ".join(path.read_text(encoding="utf-8", errors="ignore").split())
            if len(text) <= limit:
                return text
            return f"{text[:limit].rstrip()}..."
    return ""


def _collect_validation(repo_root: Path, manifest: RepoManifest, task_spec: TaskSpec) -> ValidationSummary:
    try:
        runtime_setup = prepare_runtime(repo_root, task_spec.install_config)
        test_result = run_tests(
            repo_root,
            python_executable=runtime_setup.python_executable,
            targets=task_spec.test_targets,
            test_command=task_spec.test_command,
        )
        coverage_result = run_coverage(
            repo_root,
            manifest,
            python_executable=runtime_setup.python_executable,
            targets=task_spec.test_targets,
            test_command=task_spec.test_command,
        )
    except Exception as exc:
        return ValidationSummary(ran=False, validation_error=str(exc))
    return ValidationSummary(
        ran=True,
        setup_succeeded=runtime_setup.success,
        setup_error=runtime_setup.error,
        test_exit_code=test_result.exit_code,
        coverage_exit_code=coverage_result.exit_code,
        passing_tests=test_result.passed,
        failed_tests=test_result.failed + test_result.errors,
        invalid_tests=test_result.invalid_test_count,
        coverage_percent=coverage_result.coverage_percent,
        failing_locations=test_result.failing_locations,
    )


def _write_export_package(
    *,
    export_dir: Path,
    bundled_repo_dir: Path,
    staged: StagedRepository,
    spec: Repo2EnvSpec,
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    for directory_name in RUNTIME_COPY_DIRS:
        shutil.copytree(
            PROJECT_ROOT / directory_name,
            export_dir / directory_name,
            ignore=_copy_ignore,
        )
    for filename in RUNTIME_COPY_FILES:
        source_path = PROJECT_ROOT / filename
        if source_path.exists():
            shutil.copy2(source_path, export_dir / filename)

    shutil.copytree(staged.staged_path, bundled_repo_dir, ignore=_copy_ignore)
    (export_dir / "outputs" / "trajectories").mkdir(parents=True, exist_ok=True)

    write_spec(spec, export_dir / EXPORT_REPO_SPEC_PATH)
    write_task_spec(TaskSpec.from_dict(spec.task), export_dir / EXPORT_TASK_SPEC_PATH)
    write_openenv_root_shims(export_dir)
    (export_dir / "openenv.yaml").write_text(
        render_openenv_yaml(str(spec.export["openenv_name"])),
        encoding="utf-8",
    )
    dockerfile_content = render_dockerfile(
        bundled_repo_dir=f"/app/{spec.export['bundled_repo_dir']}",
        install_recipe=spec.runtime["install_recipe"],
    )
    (export_dir / "Dockerfile").write_text(dockerfile_content, encoding="utf-8")
    (export_dir / "server" / "Dockerfile").write_text(dockerfile_content, encoding="utf-8")
    (export_dir / "README.md").write_text(render_export_readme(spec), encoding="utf-8")


def render_openenv_yaml(export_name: str) -> str:
    return (
        "spec_version: 1\n"
        f"name: {export_name}\n"
        "type: space\n"
        "runtime: fastapi\n"
        "app: repo2env.openenv.server:app\n"
        "port: 8000\n"
    )


def render_dockerfile(*, bundled_repo_dir: str, install_recipe: dict[str, Any]) -> str:
    del install_recipe
    runtime_note = (
        "# Repo-specific installs happen inside Repo2Env's per-episode runtime venv.\n"
        "# Do not install the bundled repo into the server environment here, or target repo\n"
        "# dependencies may downgrade server dependencies and break the OpenEnv app.\n\n"
    )

    return (
        "FROM python:3.11-slim\n\n"
        "WORKDIR /app\n\n"
        "RUN apt-get update \\\n"
        "    && apt-get install -y --no-install-recommends git \\\n"
        "    && rm -rf /var/lib/apt/lists/*\n\n"
        "COPY . /app\n\n"
        "RUN pip install --no-cache-dir -e .\n"
        f"{runtime_note}"
        f"ENV REPO2ENV_DEFAULT_REPO={bundled_repo_dir}\n\n"
        "EXPOSE 8000\n\n"
        "CMD [\"repo2env-openenv-server\", \"--port\", \"8000\"]\n"
    )


def render_export_readme(spec: Repo2EnvSpec) -> str:
    repo_dir = spec.export["bundled_repo_dir"]
    supported = "yes" if spec.support.supported else "no"
    warnings = "\n".join(f"- {warning}" for warning in spec.support.warnings) or "- none"
    frontmatter = (
        "---\n"
        f"title: {spec.export['openenv_name']}\n"
        "emoji: 🔧\n"
        "colorFrom: blue\n"
        "colorTo: green\n"
        "sdk: docker\n"
        "pinned: false\n"
        "app_port: 8000\n"
        "base_path: /web\n"
        "tags:\n"
        "  - openenv\n"
        "---\n\n"
    )
    return frontmatter + f"# {spec.export['openenv_name']}\n\n" \
        f"Generated by Repo2Env from `{spec.repo_source}`.\n\n" \
        f"This package bundles the source repo under `{repo_dir}` and serves it through the generic Repo2Env OpenEnv runtime.\n\n" \
        f"## Task Mode\n\n" \
        f"- mode: {spec.task['task_mode']}\n" \
        f"- title: {spec.task['title']}\n\n" \
        f"## Support Assessment\n\n" \
        f"- supported: {supported}\n" \
        f"- repo name: {spec.repo_name}\n\n" \
        f"## Warnings\n\n" \
        f"{warnings}\n\n" \
        f"## Local Commands\n\n" \
        f"```bash\n" \
        f"openenv validate\n" \
        f"openenv push .\n" \
        f"```\n"


def write_openenv_root_shims(export_dir: Path) -> None:
    (export_dir / "__init__.py").write_text(
        '"""Repo2Env OpenEnv package compatibility exports."""\n\n'
        "from .client import Repo2EnvClient\n"
        "from .models import (\n"
        "    REPO2ENV_TOOL_NAMES,\n"
        "    Repo2EnvAction,\n"
        "    Repo2EnvObservation,\n"
        "    Repo2EnvState,\n"
        ")\n"
        "from repo2env.openenv.environment import Repo2EnvOpenEnvEnvironment\n\n"
        "__all__ = [\n"
        '    "REPO2ENV_TOOL_NAMES",\n'
        '    "Repo2EnvAction",\n'
        '    "Repo2EnvClient",\n'
        '    "Repo2EnvObservation",\n'
        '    "Repo2EnvOpenEnvEnvironment",\n'
        '    "Repo2EnvState",\n'
        "]\n",
        encoding="utf-8",
    )
    (export_dir / "client.py").write_text(
        '"""Repo2Env OpenEnv client compatibility shim."""\n\n'
        "from repo2env.openenv.client import Repo2EnvClient\n\n"
        '__all__ = ["Repo2EnvClient"]\n',
        encoding="utf-8",
    )
    (export_dir / "models.py").write_text(
        '"""Repo2Env OpenEnv model compatibility shim."""\n\n'
        "from repo2env.openenv.models import (\n"
        "    REPO2ENV_TOOL_NAMES,\n"
        "    Repo2EnvAction,\n"
        "    Repo2EnvObservation,\n"
        "    Repo2EnvState,\n"
        ")\n\n"
        "__all__ = [\n"
        '    "REPO2ENV_TOOL_NAMES",\n'
        '    "Repo2EnvAction",\n'
        '    "Repo2EnvObservation",\n'
        '    "Repo2EnvState",\n'
        "]\n",
        encoding="utf-8",
    )

def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored = {".git", ".hg", ".venv", ".pytest_cache", ".mypy_cache", "__pycache__", ".repo2env", ".uv-cache"}
    return {name for name in names if name in ignored}


def _service_marker_warnings(repo_root: Path) -> list[str]:
    warnings = []
    for marker in SERVICE_MARKERS:
        if (repo_root / marker).exists():
            warnings.append(
                f"Detected {marker}, which usually means the repo expects external services or framework setup."
            )
    return warnings


def _local_repo_path(source: str) -> Path | None:
    if source.startswith(("http://", "https://", "git@")):
        return None
    return Path(source).expanduser().resolve()


def _run_openenv_command(command: list[str], cwd: Path) -> CommandResult:
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = str(cwd / ".uv-cache")
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )
    return CommandResult(
        success=completed.returncode == 0,
        command=command,
        cwd=str(cwd),
        stdout=completed.stdout,
        stderr=completed.stderr,
        exit_code=completed.returncode,
    )
