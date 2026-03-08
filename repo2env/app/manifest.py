from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

IGNORED_PARTS = {".git", ".hg", ".venv", ".mypy_cache", ".pytest_cache", "__pycache__", ".repo2env"}
PYTEST_CONFIG_NAMES = {"pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml"}


@dataclass(slots=True)
class RepoManifest:
    repo_name: str
    repo_root: str
    python_files: list[str]
    source_files: list[str]
    test_files: list[str]
    packages: list[str]
    pytest_config_files: list[str]
    layout: str
    default_test_dir: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_manifest(repo_root: str | Path) -> RepoManifest:
    root = Path(repo_root).resolve()
    python_files = []
    source_files = []
    test_files = []

    for path in sorted(root.rglob("*.py")):
        if _is_ignored(path, root):
            continue
        rel = path.relative_to(root).as_posix()
        python_files.append(rel)
        if _is_test_file(path.relative_to(root)):
            test_files.append(rel)
        else:
            source_files.append(rel)

    packages = sorted(
        {
            package.relative_to(root).as_posix()
            for package in root.rglob("__init__.py")
            if not _is_ignored(package, root)
        }
    )
    packages = [str(Path(package).parent).replace("\\", "/") for package in packages]

    pytest_configs = [
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name in PYTEST_CONFIG_NAMES and not _is_ignored(path, root)
    ]

    layout = "src" if any(path.startswith("src/") for path in source_files) else "flat"
    default_test_dir = _detect_test_dir(test_files)

    return RepoManifest(
        repo_name=root.name,
        repo_root=str(root),
        python_files=python_files,
        source_files=source_files,
        test_files=test_files,
        packages=packages,
        pytest_config_files=pytest_configs,
        layout=layout,
        default_test_dir=default_test_dir,
    )


def write_manifest(manifest: RepoManifest, destination: str | Path) -> Path:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return output_path


def _detect_test_dir(test_files: list[str]) -> str:
    for path in test_files:
        path_obj = Path(path)
        if len(path_obj.parts) > 1:
            return path_obj.parts[0]
    return "tests"


def _is_ignored(path: Path, root: Path) -> bool:
    relative_parts = path.relative_to(root).parts
    return any(part in IGNORED_PARTS for part in relative_parts)


def _is_test_file(relative_path: Path) -> bool:
    return "tests" in relative_path.parts or relative_path.name.startswith("test_")

