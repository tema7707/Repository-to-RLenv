from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import shlex
from typing import Any

from repo2env.app.install import InstallRecipe
from repo2env.app.manifest import RepoManifest

TASK_SPEC_LOCATIONS = (
    Path("task_spec.json"),
    Path(".repo2env") / "task_spec.json",
)
ALLOWED_TASK_MODES = {"repo_editing", "test_generation", "pr_fix"}
TASK_SPEC_ALLOWED_KEYS = {
    "spec_version",
    "instance_id",
    "task_id",
    "task_mode",
    "title",
    "problem_statement",
    "hints_text",
    "install_config",
    "install_recipe",
    "test_command",
    "test_targets",
    "fail_to_pass",
    "pass_to_pass",
    "test_patch",
    "starter_paths",
    "allow_source_edits",
    "metadata",
}


class TaskSpecValidationError(ValueError):
    """Raised when a repo task_spec.json does not match the Repo2Env schema."""


@dataclass(slots=True)
class TaskSpec:
    spec_version: int
    instance_id: str
    task_mode: str
    title: str
    problem_statement: str
    hints_text: str = ""
    install_config: InstallRecipe = field(default_factory=lambda: InstallRecipe(strategy="none"))
    test_command: str = "pytest -q --disable-warnings"
    test_targets: list[str] = field(default_factory=list)
    starter_paths: list[str] = field(default_factory=list)
    allow_source_edits: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["install_config"] = self.install_config.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any], default_install: InstallRecipe | None = None) -> TaskSpec:
        validate_task_spec_payload(payload)
        install_payload = payload.get("install_config") or payload.get("install_recipe")
        install_config = (
            InstallRecipe.from_dict(install_payload)
            if install_payload
            else (default_install or InstallRecipe(strategy="none"))
        )
        task_mode = _normalize_task_mode(str(payload.get("task_mode", "repo_editing")))
        return cls(
            spec_version=int(payload.get("spec_version", 1)),
            instance_id=str(payload.get("instance_id") or payload.get("task_id") or "repo-task"),
            task_mode=task_mode,
            title=str(payload.get("title", "Improve repository test outcomes")),
            problem_statement=str(
                payload.get(
                    "problem_statement",
                    "Modify repository files to increase the number of passing tests.",
                )
            ),
            hints_text=str(payload.get("hints_text", "Prefer focused edits and verify changes with pytest.")),
            install_config=install_config,
            test_command=_normalize_test_command(str(payload.get("test_command", "pytest -q --disable-warnings"))),
            test_targets=[str(item) for item in payload.get("test_targets", [])],
            starter_paths=[str(item) for item in payload.get("starter_paths", [])],
            allow_source_edits=bool(payload.get("allow_source_edits", True)),
            metadata=dict(payload.get("metadata", {})),
        )


def default_task_spec(manifest: RepoManifest, install_recipe: InstallRecipe) -> TaskSpec:
    starter_paths = (manifest.source_files[:3] + manifest.test_files[:2])[:5]
    return TaskSpec(
        spec_version=1,
        instance_id=f"{manifest.repo_name}-repo-editing",
        task_mode="repo_editing",
        title=f"Increase passing tests for {manifest.repo_name}",
        problem_statement=(
            "Inspect the repository, edit files, run pytest, and improve the number of passing tests."
        ),
        hints_text="Use small edits, rerun tests, and stop when further changes are not helping.",
        install_config=install_recipe,
        starter_paths=starter_paths,
        allow_source_edits=True,
        metadata={"repo_name": manifest.repo_name},
    )


def build_task_spec(
    manifest: RepoManifest,
    install_recipe: InstallRecipe,
    *,
    task_mode: str = "repo_editing",
    instance_id: str | None = None,
    title: str | None = None,
    problem_statement: str | None = None,
    hints_text: str | None = None,
    test_command: str | None = None,
    test_targets: list[str] | None = None,
    fail_to_pass: list[str] | None = None,
    pass_to_pass: list[str] | None = None,
    test_patch: str | None = None,
    starter_paths: list[str] | None = None,
    allow_source_edits: bool | None = None,
    install_commands: list[str] | None = None,
    install_strategy: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TaskSpec:
    del fail_to_pass, pass_to_pass, test_patch

    if task_mode not in ALLOWED_TASK_MODES:
        raise TaskSpecValidationError(
            f"Unsupported task_mode {task_mode!r}. Expected one of {sorted(ALLOWED_TASK_MODES)}."
        )

    base = default_task_spec(manifest, install_recipe).to_dict()
    base["task_mode"] = "repo_editing"
    base["instance_id"] = instance_id or f"{manifest.repo_name}-repo-editing"
    if title is not None:
        base["title"] = title
    if problem_statement is not None:
        base["problem_statement"] = problem_statement
    if hints_text is not None:
        base["hints_text"] = hints_text
    if test_command is not None:
        base["test_command"] = test_command
    if test_targets is not None:
        base["test_targets"] = test_targets
    if starter_paths is not None:
        base["starter_paths"] = starter_paths
    if allow_source_edits is not None:
        base["allow_source_edits"] = allow_source_edits
    if metadata is not None:
        merged_metadata = dict(base.get("metadata", {}))
        merged_metadata.update(metadata)
        base["metadata"] = merged_metadata

    if install_commands is not None:
        resolved_strategy = install_strategy or (
            install_recipe.strategy if install_recipe.strategy != "none" else "custom"
        )
        base["install_config"] = InstallRecipe(
            strategy=resolved_strategy,
            commands=[str(command) for command in install_commands],
            detected_files=list(install_recipe.detected_files),
            notes=list(install_recipe.notes),
        ).to_dict()
    elif install_strategy is not None:
        base["install_config"] = InstallRecipe(
            strategy=install_strategy,
            commands=list(install_recipe.commands),
            detected_files=list(install_recipe.detected_files),
            notes=list(install_recipe.notes),
        ).to_dict()

    return TaskSpec.from_dict(base, default_install=install_recipe)


def load_task_spec(
    repo_root: str | Path,
    manifest: RepoManifest,
    install_recipe: InstallRecipe,
    *,
    override_path: str | Path | None = None,
) -> TaskSpec:
    root = Path(repo_root).resolve()
    candidates = [Path(override_path).expanduser().resolve()] if override_path else [root / relative for relative in TASK_SPEC_LOCATIONS]
    for candidate in candidates:
        if candidate.is_file():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            spec = TaskSpec.from_dict(payload, default_install=install_recipe)
            if spec.install_config.strategy == "none" and install_recipe.strategy != "none":
                spec.install_config = install_recipe
            return spec
    return default_task_spec(manifest, install_recipe)


def write_task_spec(spec: TaskSpec, destination: str | Path) -> Path:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
    return output_path


def validate_task_spec_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise TaskSpecValidationError(f"Task spec must be a JSON object, got {type(payload).__name__}.")

    unknown_keys = sorted(set(payload) - TASK_SPEC_ALLOWED_KEYS)
    if unknown_keys:
        raise TaskSpecValidationError(
            "Unknown task_spec.json fields: "
            + ", ".join(unknown_keys)
            + ". Use the Repo2Env schema instead."
        )

    task_mode = str(payload.get("task_mode", "repo_editing"))
    if task_mode not in ALLOWED_TASK_MODES:
        raise TaskSpecValidationError(
            f"Unsupported task_mode {task_mode!r}. Expected one of {sorted(ALLOWED_TASK_MODES)}."
        )

    for key in ("test_targets", "starter_paths"):
        value = payload.get(key, [])
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise TaskSpecValidationError(f"{key} must be a list of strings.")

    metadata = payload.get("metadata", {})
    if metadata is not None and not isinstance(metadata, dict):
        raise TaskSpecValidationError("metadata must be a JSON object.")

    install_payload = payload.get("install_config") or payload.get("install_recipe")
    if install_payload is not None and not isinstance(install_payload, dict):
        raise TaskSpecValidationError("install_config must be a JSON object.")


def _normalize_task_mode(task_mode: str) -> str:
    if task_mode in ALLOWED_TASK_MODES:
        return "repo_editing"
    return "repo_editing"


def _normalize_test_command(test_command: str) -> str:
    command_text = test_command.strip()
    if not command_text:
        return "pytest -q --disable-warnings"

    tokens = shlex.split(command_text)
    filtered_tokens: list[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in {"-x", "--exitfirst"}:
            continue
        if token.startswith("-") and not token.startswith("--") and len(token) > 2 and "x" in token[1:]:
            short_flags = token[1:].replace("x", "")
            if short_flags:
                filtered_tokens.append(f"-{short_flags}")
            continue
        if token == "--maxfail":
            skip_next = True
            continue
        if token.startswith("--maxfail="):
            continue
        filtered_tokens.append(token)
    return shlex.join(filtered_tokens) if filtered_tokens else "pytest -q --disable-warnings"
