from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


class SandboxWorkspace:
    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else None

    def create_clean_copy(self, source_repo: str | Path, repo_name: str) -> Path:
        source_path = Path(source_repo).resolve()
        sandbox_root = Path(
            tempfile.mkdtemp(prefix=f"repo2env-{repo_name}-", dir=str(self.base_dir) if self.base_dir else None)
        )
        destination = sandbox_root / source_path.name
        shutil.copytree(source_path, destination)
        return destination

    def cleanup(self, repo_path: str | Path | None) -> None:
        if not repo_path:
            return
        path = Path(repo_path).resolve()
        sandbox_root = path.parent
        if sandbox_root.exists():
            shutil.rmtree(sandbox_root, ignore_errors=True)

