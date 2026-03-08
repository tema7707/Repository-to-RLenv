from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SupportAssessment:
    supported: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ValidationSummary:
    ran: bool = False
    setup_succeeded: bool = True
    setup_error: str | None = None
    test_exit_code: int | None = None
    coverage_exit_code: int | None = None
    passing_tests: int = 0
    failed_tests: int = 0
    invalid_tests: int = 0
    coverage_percent: float | None = None
    failing_locations: list[str] = field(default_factory=list)
    validation_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Repo2EnvSpec:
    spec_version: int
    repo_name: str
    repo_source: str
    source_kind: str
    support: SupportAssessment
    manifest: dict[str, Any]
    task: dict[str, Any]
    runtime: dict[str, Any]
    reward: dict[str, float]
    observation: dict[str, Any]
    hints: dict[str, Any]
    validation: ValidationSummary
    export: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["support"] = self.support.to_dict()
        payload["validation"] = self.validation.to_dict()
        return payload


def write_spec(spec: Repo2EnvSpec, destination: str | Path) -> Path:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
    return output_path
