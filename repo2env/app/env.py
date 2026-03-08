from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import difflib
import os
from pathlib import Path
import re
import shutil
from typing import Any

from repo2env.app.ingest import StagedRepository, ingest_repository
from repo2env.app.install import InstallRecipe, RuntimeSetup, detect_install_recipe, prepare_runtime
from repo2env.app.manifest import RepoManifest, build_manifest, write_manifest
from repo2env.app.rewards import EpisodeMetrics, calculate_reward
from repo2env.app.sandbox import SandboxWorkspace
from repo2env.app.task_spec import TaskSpec, load_task_spec
from repo2env.app.tooling import REPO2ENV_TOOL_NAMES, REPO2ENV_TOOL_SCHEMAS
from repo2env.app.validator import TestRunResult, run_tests

DEFAULT_TRAJECTORY_DIR = Path("outputs/trajectories")
MAX_PREVIEW_CHARS = 1200
INTERNAL_PARTS = {".git", ".repo2env", ".pytest_cache", "__pycache__"}


@dataclass(slots=True)
class EpisodeStepRecord:
    observation: dict[str, Any]
    action: dict[str, Any]
    tool_result: dict[str, Any]
    reward: float
    reward_breakdown: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EpisodeState:
    repo_path: Path
    manifest: RepoManifest
    task_spec: TaskSpec
    install_recipe: InstallRecipe
    runtime_setup: RuntimeSetup
    allowed_tools: list[str]
    step_count: int = 0
    metrics: EpisodeMetrics = field(default_factory=EpisodeMetrics)
    latest_test_result: TestRunResult | None = None
    focus_files: list[str] = field(default_factory=list)
    history: list[EpisodeStepRecord] = field(default_factory=list)
    no_progress_evals: int = 0
    done: bool = False


class Repo2EnvEnvironment:
    """Sandboxed repository editing environment with passing-test rewards."""

    def __init__(
        self,
        source: str | Path,
        *,
        max_steps: int = 12,
        coverage_target: float = 0.0,
        staging_root: str | Path | None = None,
        workspace_root: str | Path | None = None,
        task_spec_path: str | Path | None = None,
    ) -> None:
        del coverage_target
        self.source = str(source)
        self.max_steps = max_steps
        self.task_spec_path = str(task_spec_path) if task_spec_path else None
        self.staged: StagedRepository = ingest_repository(self.source, staging_root=staging_root)
        self.sandbox = SandboxWorkspace(workspace_root)
        self.state: EpisodeState | None = None
        self._baseline_metrics = EpisodeMetrics()

    def reset(self) -> dict[str, Any]:
        if self.state is not None:
            self.sandbox.cleanup(self.state.repo_path)

        repo_path = self.sandbox.create_clean_copy(self.staged.staged_path, self.staged.manifest.repo_name)
        manifest = build_manifest(repo_path)
        write_manifest(manifest, repo_path / "repo2env_manifest.json")

        install_recipe = detect_install_recipe(repo_path, manifest)
        task_spec = load_task_spec(
            repo_path,
            manifest,
            install_recipe,
            override_path=self.task_spec_path,
        )
        runtime_setup = prepare_runtime(repo_path, task_spec.install_config)
        latest_test_result = run_tests(
            repo_path,
            python_executable=runtime_setup.python_executable,
            targets=task_spec.test_targets,
            test_command=task_spec.test_command,
        )

        metrics = EpisodeMetrics(
            passing_tests=latest_test_result.passed,
            failed_tests=latest_test_result.failed + latest_test_result.errors,
            invalid_tests=latest_test_result.invalid_test_count,
        )

        focus_files = [
            path for path in task_spec.starter_paths if self._is_visible_repo_relative(Path(path))
        ] or [
            path
            for path in (manifest.source_files[:3] + manifest.test_files[:2])
            if self._is_visible_repo_relative(Path(path))
        ]
        self.state = EpisodeState(
            repo_path=repo_path,
            manifest=manifest,
            task_spec=task_spec,
            install_recipe=install_recipe,
            runtime_setup=runtime_setup,
            allowed_tools=list(REPO2ENV_TOOL_NAMES),
            metrics=metrics,
            latest_test_result=latest_test_result,
            focus_files=focus_files[:6],
        )
        self._baseline_metrics = deepcopy(metrics)
        return self._build_observation()

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Environment must be reset before calling step().")
        if self.state.done:
            raise RuntimeError("Episode is already complete.")

        tool = action.get("tool")
        args = action.get("args") or {}
        previous_metrics = deepcopy(self.state.metrics)

        try:
            tool_result = self._dispatch(tool, args)
            reward, reward_breakdown = calculate_reward(previous_metrics, self.state.metrics)
        except Exception as exc:
            tool_result = {"error": str(exc)}
            reward = -1.0
            reward_breakdown = {
                "delta_passing_tests": 0,
                "step_penalty": 1.0,
            }

        self.state.step_count += 1
        self._update_stagnation(tool, reward_breakdown)
        self.state.done = self._is_done(tool)

        record = EpisodeStepRecord(
            observation={},
            action=deepcopy(action),
            tool_result=deepcopy(tool_result),
            reward=reward,
            reward_breakdown=deepcopy(reward_breakdown),
        )
        self.state.history.append(record)
        observation = self._build_observation()
        record.observation = deepcopy(observation)
        info = {
            "tool_result": tool_result,
            "reward_breakdown": reward_breakdown,
            "metrics": self.state.metrics.to_dict(),
        }
        return observation, reward, self.state.done, info

    def export_episode(self) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("No episode state available.")
        return {
            "repo": self.state.manifest.repo_name,
            "source": self.source,
            "task": self.state.task_spec.to_dict(),
            "setup_summary": self.state.runtime_setup.to_dict(),
            "initial_metrics": self._baseline_metrics.to_dict(),
            "steps": [record.to_dict() for record in self.state.history],
            "final_metrics": self.state.metrics.to_dict(),
        }

    def close(self) -> None:
        if self.state is not None:
            self.sandbox.cleanup(self.state.repo_path)
            self.state = None
        staged_root = self.staged.staged_path.parent
        if staged_root.exists():
            shutil.rmtree(staged_root, ignore_errors=True)

    def _dispatch(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool not in self.state.allowed_tools:  # type: ignore[union-attr]
            raise ValueError(f"Unsupported tool: {tool}")
        if tool == "list_files":
            return self._list_files(args)
        if tool == "read_file":
            return self._read_file(args)
        if tool == "read_file_chunk":
            return self._read_file_chunk(args)
        if tool == "search_code":
            return self._search_code(args)
        if tool == "replace_in_file":
            return self._replace_in_file(args)
        if tool == "insert_after":
            return self._insert_after(args)
        if tool == "insert_before":
            return self._insert_before(args)
        if tool == "append_to_file":
            return self._append_to_file(args)
        if tool == "write_file":
            return self._write_file(args)
        if tool == "run_tests":
            return self._run_tests().to_dict()
        if tool == "get_test_failures":
            return self._get_test_failures(args)
        if tool == "diff_working_tree":
            return self._diff_working_tree(args)
        if tool == "submit":
            return {"submitted": True}
        raise ValueError(f"Unsupported tool: {tool}")

    def _list_files(self, args: dict[str, Any]) -> dict[str, Any]:
        assert self.state is not None
        limit = int(args.get("limit", 200))
        files = [
            path.relative_to(self.state.repo_path).as_posix()
            for path in sorted(self.state.repo_path.rglob("*"))
            if path.is_file() and self._is_visible_repo_path(path)
        ]
        return {"files": files[:limit], "count": len(files)}

    def _read_file(self, args: dict[str, Any]) -> dict[str, Any]:
        assert self.state is not None
        relative_path = str(args["path"])
        path = self._resolve_repo_path(relative_path)
        content = path.read_text(encoding="utf-8")
        self._focus(relative_path)
        return {"path": relative_path, "content": content}

    def _read_file_chunk(self, args: dict[str, Any]) -> dict[str, Any]:
        relative_path = str(args["path"])
        path = self._resolve_repo_path(relative_path)
        start_line = int(args["start_line"])
        end_line = int(args.get("end_line", start_line + 199))
        if start_line <= 0 or end_line < start_line:
            raise ValueError("start_line must be >= 1 and end_line must be >= start_line.")
        lines = path.read_text(encoding="utf-8").splitlines()
        selected = lines[start_line - 1 : end_line]
        self._focus(relative_path)
        return {
            "path": relative_path,
            "start_line": start_line,
            "end_line": start_line + len(selected) - 1 if selected else start_line - 1,
            "content": "\n".join(selected),
        }

    def _search_code(self, args: dict[str, Any]) -> dict[str, Any]:
        assert self.state is not None
        query = str(args["query"]).lower()
        limit = int(args.get("limit", 20))
        matches = []
        for relative_path in self.state.manifest.python_files:
            if not self._is_visible_repo_relative(Path(relative_path)):
                continue
            path = self._resolve_repo_path(relative_path)
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if query in line.lower():
                    matches.append(
                        {"path": relative_path, "line_number": line_number, "line": line.strip()}
                    )
                if len(matches) >= limit:
                    return {"query": args["query"], "matches": matches}
        return {"query": args["query"], "matches": matches}

    def _write_file(self, args: dict[str, Any]) -> dict[str, Any]:
        assert self.state is not None
        relative_path = str(args["path"])
        content = str(args["content"])
        target = self._resolve_repo_path(relative_path, must_exist=False)
        self._validate_write_path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self._invalidate_python_caches(target)
        normalized = self._normalize_path(target)
        self._track_manifest_path(normalized)
        self._focus(normalized)
        return {"path": normalized, "bytes_written": len(content.encode("utf-8"))}

    def _replace_in_file(self, args: dict[str, Any]) -> dict[str, Any]:
        relative_path = str(args["path"])
        old_text = str(args["old_text"])
        new_text = str(args["new_text"])
        count = int(args.get("count", 1))
        if not old_text:
            raise ValueError("old_text must be non-empty.")
        if count <= 0:
            raise ValueError("count must be a positive integer.")

        target = self._resolve_repo_path(relative_path)
        self._validate_write_path(target)
        content = target.read_text(encoding="utf-8")
        occurrences = content.count(old_text)
        if occurrences == 0:
            raise ValueError(f"old_text was not found in {relative_path}.")

        replacements = min(occurrences, count)
        updated = content.replace(old_text, new_text, count)
        target.write_text(updated, encoding="utf-8")
        self._invalidate_python_caches(target)
        normalized = self._normalize_path(target)
        self._focus(normalized)
        return {"path": normalized, "replacements": replacements}

    def _insert_after(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._insert_relative(args, before=False)

    def _insert_before(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._insert_relative(args, before=True)

    def _append_to_file(self, args: dict[str, Any]) -> dict[str, Any]:
        relative_path = str(args["path"])
        content = str(args["content"])
        target = self._resolve_repo_path(relative_path)
        self._validate_write_path(target)
        existing = target.read_text(encoding="utf-8")
        updated = f"{existing}{content}"
        target.write_text(updated, encoding="utf-8")
        self._invalidate_python_caches(target)
        normalized = self._normalize_path(target)
        self._focus(normalized)
        return {"path": normalized, "bytes_appended": len(content.encode('utf-8'))}

    def _run_tests(self) -> TestRunResult:
        assert self.state is not None
        result = run_tests(
            self.state.repo_path,
            python_executable=self.state.runtime_setup.python_executable,
            targets=self.state.task_spec.test_targets,
            test_command=self.state.task_spec.test_command,
        )
        self.state.latest_test_result = result
        self.state.metrics.passing_tests = result.passed
        self.state.metrics.failed_tests = result.failed + result.errors
        self.state.metrics.invalid_tests = result.invalid_test_count
        return result

    def _get_test_failures(self, args: dict[str, Any]) -> dict[str, Any]:
        assert self.state is not None
        result = self.state.latest_test_result
        if result is None:
            return {"failures": [], "count": 0}
        limit = int(args.get("limit", 20))
        failures = self._structured_failures(result.output, result.failing_locations)[:limit]
        return {"failures": failures, "count": len(failures)}

    def _diff_working_tree(self, args: dict[str, Any]) -> dict[str, Any]:
        assert self.state is not None
        requested_path = str(args["path"]) if "path" in args else None
        context_lines = int(args.get("context_lines", 3))
        max_diff_chars = int(args.get("max_diff_chars", 12000))
        changed_files: list[dict[str, Any]] = []
        total_chars = 0

        candidate_paths = self._diff_candidate_paths(requested_path)
        for relative_path in candidate_paths:
            if requested_path and relative_path != requested_path:
                continue
            original_text = self._read_snapshot_file(relative_path)
            current_text = self._resolve_repo_path(relative_path).read_text(encoding="utf-8")
            if original_text == current_text:
                continue
            diff_text = "".join(
                difflib.unified_diff(
                    original_text.splitlines(keepends=True),
                    current_text.splitlines(keepends=True),
                    fromfile=f"a/{relative_path}",
                    tofile=f"b/{relative_path}",
                    n=context_lines,
                )
            )
            truncated = False
            if total_chars + len(diff_text) > max_diff_chars:
                remaining = max_diff_chars - total_chars
                if remaining <= 0:
                    break
                diff_text = diff_text[:remaining]
                truncated = True
            total_chars += len(diff_text)
            changed_files.append({"path": relative_path, "diff": diff_text, "truncated": truncated})
            if total_chars >= max_diff_chars:
                break

        return {"changed_files": changed_files, "count": len(changed_files)}

    def _update_stagnation(self, tool: str, reward_breakdown: dict[str, Any]) -> None:
        assert self.state is not None
        if tool != "run_tests":
            return
        if reward_breakdown.get("delta_passing_tests", 0) > 0:
            self.state.no_progress_evals = 0
        else:
            self.state.no_progress_evals += 1

    def _is_done(self, tool: str) -> bool:
        assert self.state is not None
        if tool == "submit":
            return True
        if self.state.step_count >= self.max_steps:
            return True
        if self.state.no_progress_evals >= 3:
            return True
        return False

    def _build_observation(self) -> dict[str, Any]:
        assert self.state is not None
        selected_sources = self._selected_previews(self.state.manifest.source_files)
        selected_tests = self._selected_previews(self.state.manifest.test_files)
        recent_actions = [
            {
                "tool": record.action.get("tool"),
                "reward": record.reward,
                "reward_breakdown": record.reward_breakdown,
            }
            for record in self.state.history[-5:]
        ]
        return {
            "repo_summary": {
                "repo_name": self.state.manifest.repo_name,
                "layout": self.state.manifest.layout,
                "python_file_count": len(self.state.manifest.python_files),
                "source_file_count": len(self.state.manifest.source_files),
                "test_file_count": len(self.state.manifest.test_files),
                "source_files": [
                    path for path in self.state.manifest.source_files[:10] if self._is_visible_repo_relative(Path(path))
                ],
                "test_files": [
                    path for path in self.state.manifest.test_files[:10] if self._is_visible_repo_relative(Path(path))
                ],
                "pytest_config_files": self.state.manifest.pytest_config_files,
            },
            "task_summary": {
                "instance_id": self.state.task_spec.instance_id,
                "task_mode": self.state.task_spec.task_mode,
                "title": self.state.task_spec.title,
                "problem_statement": self.state.task_spec.problem_statement,
                "hints_text": self.state.task_spec.hints_text,
                "starter_paths": self.state.task_spec.starter_paths,
                "allow_source_edits": self.state.task_spec.allow_source_edits,
            },
            "setup_summary": self.state.runtime_setup.to_dict(),
            "selected_source_files": selected_sources,
            "selected_test_files": selected_tests,
            "latest_pytest_summary": self._pytest_summary(),
            "latest_coverage_summary": None,
            "recent_action_history": recent_actions,
            "step_count": self.state.step_count,
            "max_steps": self.max_steps,
            "allowed_tools": self.state.allowed_tools,
            "tool_schemas": REPO2ENV_TOOL_SCHEMAS,
        }

    def _pytest_summary(self) -> dict[str, Any] | None:
        assert self.state is not None
        result = self.state.latest_test_result
        if result is None:
            return None
        return {
            "exit_code": result.exit_code,
            "passed": result.passed,
            "failed": result.failed,
            "errors": result.errors,
            "skipped": result.skipped,
            "invalid_test_count": result.invalid_test_count,
            "failing_locations": result.failing_locations,
            "duration_seconds": result.duration_seconds,
            "output": result.output,
        }

    def _selected_previews(self, candidates: list[str]) -> list[dict[str, str]]:
        assert self.state is not None
        visible_candidates = [path for path in candidates if self._is_visible_repo_relative(Path(path))]
        selected = [path for path in self.state.focus_files if path in visible_candidates]
        if not selected:
            selected = visible_candidates[:2]
        previews = []
        for relative_path in selected[:2]:
            path = self._resolve_repo_path(relative_path)
            content = path.read_text(encoding="utf-8")
            previews.append(
                {
                    "path": relative_path,
                    "preview": content[:MAX_PREVIEW_CHARS],
                }
            )
        return previews

    def _focus(self, relative_path: str) -> None:
        assert self.state is not None
        if relative_path in self.state.focus_files:
            self.state.focus_files.remove(relative_path)
        self.state.focus_files.insert(0, relative_path)
        self.state.focus_files = self.state.focus_files[:6]

    def _resolve_repo_path(self, relative_path: str, *, must_exist: bool = True) -> Path:
        assert self.state is not None
        repo_root = self.state.repo_path.resolve()
        path = (repo_root / relative_path).resolve()
        if repo_root not in path.parents and path != repo_root:
            raise ValueError(f"Path escapes repository root: {relative_path}")
        if self._is_internal_path(path):
            raise ValueError(f"Path is not accessible: {relative_path}")
        if must_exist and not path.exists():
            raise FileNotFoundError(f"Path not found: {relative_path}")
        return path

    def _validate_write_path(self, path: Path) -> None:
        assert self.state is not None
        if not self.state.task_spec.allow_source_edits:
            raise ValueError("File edits are disabled for this task.")
        if self._is_internal_path(path):
            raise ValueError("Cannot modify internal Repo2Env files.")
        if path.name == "repo2env_manifest.json":
            raise ValueError("Cannot modify Repo2Env manifest files.")

    def _normalize_path(self, target: Path) -> str:
        assert self.state is not None
        return target.relative_to(self.state.repo_path.resolve()).as_posix()

    def _track_manifest_path(self, relative_path: str) -> None:
        assert self.state is not None
        if relative_path.endswith(".py"):
            if relative_path not in self.state.manifest.python_files:
                self.state.manifest.python_files.append(relative_path)
                self.state.manifest.python_files.sort()
            is_test = relative_path.startswith("tests/") or "/tests/" in relative_path or Path(relative_path).name.startswith("test_")
            target_bucket = self.state.manifest.test_files if is_test else self.state.manifest.source_files
            if relative_path not in target_bucket:
                target_bucket.append(relative_path)
                target_bucket.sort()

    def _insert_relative(self, args: dict[str, Any], *, before: bool) -> dict[str, Any]:
        relative_path = str(args["path"])
        anchor_text = str(args["anchor_text"])
        new_text = str(args["new_text"])
        count = int(args.get("count", 1))
        if not anchor_text:
            raise ValueError("anchor_text must be non-empty.")
        if count <= 0:
            raise ValueError("count must be a positive integer.")

        target = self._resolve_repo_path(relative_path)
        self._validate_write_path(target)
        content = target.read_text(encoding="utf-8")
        occurrences = content.count(anchor_text)
        if occurrences == 0:
            raise ValueError(f"anchor_text was not found in {relative_path}.")

        replacement = f"{new_text}{anchor_text}" if before else f"{anchor_text}{new_text}"
        insertions = min(occurrences, count)
        updated = content.replace(anchor_text, replacement, count)
        target.write_text(updated, encoding="utf-8")
        self._invalidate_python_caches(target)
        normalized = self._normalize_path(target)
        self._focus(normalized)
        return {"path": normalized, "insertions": insertions}

    def _structured_failures(self, output: str, failing_locations: list[str]) -> list[dict[str, Any]]:
        failures: list[dict[str, Any]] = []
        location_set = set(failing_locations)
        node_pattern = re.compile(r"([^\s]+\.py(?:::[^\s]+)+)")
        chunks = re.split(r"\n_{10,}\s+", output)
        for chunk in chunks:
            lines = [line for line in chunk.splitlines() if line.strip()]
            if not lines:
                continue
            node_id = None
            for line in lines:
                match = node_pattern.search(line)
                if match:
                    node_id = match.group(1)
                    break
            if node_id is None:
                for location in location_set:
                    if location in chunk:
                        node_id = location
                        break
            if node_id is None:
                continue
            error_line = next((line.strip() for line in lines if line.strip().startswith("E ")), "")
            assertion_line = next(
                (line.strip() for line in lines if "assert " in line and not line.strip().startswith("E ")),
                "",
            )
            failures.append(
                {
                    "node_id": node_id,
                    "assertion_line": assertion_line,
                    "error_message": error_line,
                }
            )
        if failures:
            return failures
        return [
            {"node_id": location, "assertion_line": "", "error_message": ""}
            for location in failing_locations
        ]

    def _diff_candidate_paths(self, requested_path: str | None) -> list[str]:
        assert self.state is not None
        if requested_path is not None:
            self._resolve_repo_path(requested_path)
            return [requested_path]
        visible_paths = {
            path.relative_to(self.state.repo_path).as_posix()
            for path in self.state.repo_path.rglob("*")
            if path.is_file() and self._is_visible_repo_path(path)
        }
        snapshot_paths = {
            path.relative_to(self.staged.staged_path).as_posix()
            for path in self.staged.staged_path.rglob("*")
            if path.is_file() and self._is_visible_repo_relative(path.relative_to(self.staged.staged_path))
        }
        return sorted(visible_paths | snapshot_paths)

    def _read_snapshot_file(self, relative_path: str) -> str:
        snapshot_path = self.staged.staged_path / relative_path
        if not snapshot_path.exists():
            return ""
        return snapshot_path.read_text(encoding="utf-8")

    def _invalidate_python_caches(self, target: Path) -> None:
        if target.suffix == ".py":
            pycache_dir = target.parent / "__pycache__"
            if pycache_dir.exists():
                for compiled in pycache_dir.glob(f"{target.stem}.*.pyc"):
                    compiled.unlink(missing_ok=True)
        timestamp = max(target.stat().st_mtime + 1.0, os.path.getmtime(target))
        os.utime(target, (timestamp, timestamp))

    def _is_visible_repo_path(self, path: Path) -> bool:
        assert self.state is not None
        try:
            relative = path.relative_to(self.state.repo_path)
        except ValueError:
            return False
        return self._is_visible_repo_relative(relative)

    def _is_visible_repo_relative(self, relative: Path) -> bool:
        return not any(part in INTERNAL_PARTS for part in relative.parts)

    def _is_internal_path(self, path: Path) -> bool:
        assert self.state is not None
        try:
            relative = path.relative_to(self.state.repo_path.resolve())
        except ValueError:
            return True
        return any(part in INTERNAL_PARTS for part in relative.parts)
