from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

from repo2env.app.manifest import RepoManifest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

RECOGNIZED_REQUIREMENTS = (
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-test.txt",
    "requirements_ci.txt",
)
INSTALL_TIMEOUT_SECONDS = 300


@dataclass(slots=True)
class InstallRecipe:
    strategy: str
    commands: list[str] = field(default_factory=list)
    detected_files: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> InstallRecipe:
        data = payload or {}
        return cls(
            strategy=str(data.get("strategy", "none")),
            commands=[str(command) for command in data.get("commands", [])],
            detected_files=[str(path) for path in data.get("detected_files", [])],
            notes=[str(note) for note in data.get("notes", [])],
        )


@dataclass(slots=True)
class SetupCommandResult:
    command: list[str]
    rendered_command: str
    exit_code: int
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RuntimeSetup:
    success: bool
    python_executable: str
    install_recipe: InstallRecipe
    command_results: list[SetupCommandResult] = field(default_factory=list)
    error: str | None = None
    runtime_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "python_executable": self.python_executable,
            "install_recipe": self.install_recipe.to_dict(),
            "command_results": [result.to_dict() for result in self.command_results],
            "error": self.error,
            "runtime_dir": self.runtime_dir,
        }


def detect_install_recipe(repo_root: str | Path, manifest: RepoManifest) -> InstallRecipe:
    root = Path(repo_root).resolve()
    commands: list[str] = []
    detected_files: list[str] = []
    notes: list[str] = []
    strategy_parts: list[str] = []

    for filename in RECOGNIZED_REQUIREMENTS:
        if (root / filename).is_file():
            detected_files.append(filename)
            commands.append(f"pip install --no-cache-dir -r {{repo_dir}}/{filename}")
    if detected_files:
        strategy_parts.append("requirements")

    if (root / "pyproject.toml").is_file() or (root / "setup.py").is_file():
        metadata_file = "pyproject.toml" if (root / "pyproject.toml").is_file() else "setup.py"
        if metadata_file not in detected_files:
            detected_files.append(metadata_file)
        pyproject_build_requires = _detect_pyproject_build_requires(root / "pyproject.toml")
        if pyproject_build_requires:
            quoted_requirements = " ".join(shlex.quote(requirement) for requirement in pyproject_build_requires)
            commands.append(f"pip install --no-cache-dir {quoted_requirements}")
        commands.append("pip install --no-cache-dir --no-build-isolation -e {repo_dir}")
        strategy_parts.append("editable")
    elif manifest.layout == "src":
        notes.append("Detected src layout without package metadata; repo may need a manual install step.")

    if not commands:
        notes.append("No dependency manifest detected. Export assumes stdlib-only imports.")

    strategy = "+".join(strategy_parts) if strategy_parts else "none"
    return InstallRecipe(
        strategy=strategy,
        commands=commands,
        detected_files=detected_files,
        notes=notes,
    )


def prepare_runtime(
    repo_root: str | Path,
    install_recipe: InstallRecipe,
    *,
    timeout_seconds: int = INSTALL_TIMEOUT_SECONDS,
) -> RuntimeSetup:
    root = Path(repo_root).resolve()
    if not install_recipe.commands:
        return RuntimeSetup(
            success=True,
            python_executable=sys.executable,
            install_recipe=install_recipe,
        )

    runtime_dir = root / ".repo2env" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    python_executable = _create_virtualenv(runtime_dir)
    env = _runtime_env(runtime_dir)

    command_results: list[SetupCommandResult] = []
    for command in install_recipe.commands:
        rendered_command = command.format(repo_dir=str(root))
        argv = _normalize_command(rendered_command, python_executable)
        completed = subprocess.run(
            argv,
            cwd=root,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_seconds,
        )
        result = SetupCommandResult(
            command=argv,
            rendered_command=rendered_command,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        command_results.append(result)
        if completed.returncode != 0:
            return RuntimeSetup(
                success=False,
                python_executable=python_executable,
                install_recipe=install_recipe,
                command_results=command_results,
                error=_command_error_message(result),
                runtime_dir=str(runtime_dir),
            )

    return RuntimeSetup(
        success=True,
        python_executable=python_executable,
        install_recipe=install_recipe,
        command_results=command_results,
        runtime_dir=str(runtime_dir),
    )


def _create_virtualenv(runtime_dir: Path) -> str:
    venv_dir = runtime_dir / ".venv"
    if not venv_dir.exists():
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
            cwd=runtime_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    return str(_venv_python(venv_dir))


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _runtime_env(runtime_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PIP_CACHE_DIR"] = str(runtime_dir / "pip-cache")
    env["UV_CACHE_DIR"] = str(runtime_dir / "uv-cache")
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return env


def _normalize_command(command: str, python_executable: str) -> list[str]:
    tokens = shlex.split(command)
    if not tokens:
        raise ValueError("Install command cannot be empty.")
    if tokens[0] == "pip":
        return [python_executable, "-m", "pip", *tokens[1:]]
    if tokens[0] == "python":
        return [python_executable, *tokens[1:]]
    return tokens


def _command_error_message(result: SetupCommandResult) -> str:
    detail = result.stderr.strip() or result.stdout.strip() or "install command failed"
    return f"{result.rendered_command}: {detail}"


def _detect_pyproject_build_requires(pyproject_path: Path) -> list[str]:
    if not pyproject_path.is_file():
        return []
    try:
        payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return []
    build_system = payload.get("build-system")
    if not isinstance(build_system, dict):
        return []
    requires = build_system.get("requires")
    if not isinstance(requires, list):
        return []
    return [str(requirement) for requirement in requires if str(requirement).strip()]
