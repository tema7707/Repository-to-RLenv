from __future__ import annotations

import json
from pathlib import Path
from time import time


class TrajectoryLogger:
    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def write(self, repo_name: str, payload: dict) -> Path:
        timestamp = int(time() * 1000)
        output_path = self.output_root / f"{repo_name}-{timestamp}.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path

