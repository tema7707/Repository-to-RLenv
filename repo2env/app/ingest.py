from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess
import tempfile
from pathlib import Path

from repo2env.app.manifest import RepoManifest, build_manifest, write_manifest


@dataclass(slots=True)
class StagedRepository:
    source: str
    staged_path: Path
    manifest: RepoManifest
    manifest_path: Path


def ingest_repository(source: str | Path, staging_root: str | Path | None = None) -> StagedRepository:
    source_text = str(source)
    staging_base = Path(staging_root) if staging_root else None
    stage_root = Path(tempfile.mkdtemp(prefix="repo2env-stage-", dir=str(staging_base) if staging_base else None))

    if _is_github_source(source_text):
        repo_path = _clone_repository(source_text, stage_root)
    else:
        repo_path = _copy_local_repository(source_text, stage_root)

    manifest = build_manifest(repo_path)
    manifest_path = write_manifest(manifest, repo_path / "repo2env_manifest.json")
    return StagedRepository(source=source_text, staged_path=repo_path, manifest=manifest, manifest_path=manifest_path)


def _copy_local_repository(source: str, stage_root: Path) -> Path:
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {source_path}")
    destination = stage_root / source_path.name
    shutil.copytree(source_path, destination)
    return destination


def _clone_repository(source: str, stage_root: Path) -> Path:
    destination = stage_root / _repo_name_from_url(source)
    subprocess.run(
        ["git", "clone", "--depth", "1", source, str(destination)],
        check=True,
        capture_output=True,
        text=True,
    )
    return destination


def _repo_name_from_url(source: str) -> str:
    return source.rstrip("/").rsplit("/", maxsplit=1)[-1].removesuffix(".git")


def _is_github_source(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://") or source.startswith("git@")

