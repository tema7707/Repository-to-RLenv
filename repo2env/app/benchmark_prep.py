from __future__ import annotations

from dataclasses import dataclass
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from repo2env.app.install import InstallRecipe, prepare_runtime
from repo2env.app.manifest import RepoManifest, build_manifest
from repo2env.app.task_spec import TaskSpec
from repo2env.app.validator import TestRunResult, run_tests


MUTATION_RULES: tuple[tuple[str, str, str], ...] = (
    ("return True", "return False", "flip boolean return"),
    ("return False", "return True", "flip boolean return"),
    (">=", ">", "tighten comparison"),
    ("<=", "<", "tighten comparison"),
    ("==", "!=", "invert equality check"),
    ("!=", "==", "invert inequality check"),
    (" is None", " is not None", "invert None handling"),
    (" is not None", " is None", "invert None handling"),
    (" and ", " or ", "weaken boolean conjunction"),
    (" or ", " and ", "tighten boolean disjunction"),
    (" + ", " - ", "change arithmetic operator"),
    (" - ", " + ", "change arithmetic operator"),
    (" * ", " + ", "change arithmetic operator"),
    (" / ", " * ", "change arithmetic operator"),
    ("+ 1", "+ 2", "introduce off-by-one increment"),
    ("- 1", "+ 1", "invert decrement"),
    ("* 2", "+ 2", "change arithmetic operator"),
    ("/ 2", "* 2", "change arithmetic operator"),
    (", 0, None,", ", 1, None,", "shift sequence starting offset"),
)


@dataclass(slots=True)
class BugCandidate:
    path: str
    line_number: int
    old_text: str
    new_text: str
    reason: str
    function_name: str | None


@dataclass(slots=True)
class InjectedBug:
    path: str
    line_number: int
    old_text: str
    new_text: str
    reason: str
    function_name: str | None
    description: str
    failing_locations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "line_number": self.line_number,
            "old_text": self.old_text,
            "new_text": self.new_text,
            "reason": self.reason,
            "function_name": self.function_name,
            "description": self.description,
            "failing_locations": self.failing_locations,
        }


@dataclass(slots=True)
class BenchmarkPrepResult:
    manifest: RepoManifest
    task_spec: TaskSpec
    generated_test_files: list[str]
    injected_bug: InjectedBug | None
    warnings: list[str]


@dataclass(slots=True)
class GeneratedRegressionCandidate:
    source_path: str
    function_name: str
    args: list[Any]
    expected_repr: str


def prepare_benchmark_repository(
    repo_root: str | Path,
    manifest: RepoManifest,
    install_recipe: InstallRecipe,
    task_spec: TaskSpec,
    *,
    variant_index: int = 0,
) -> BenchmarkPrepResult:
    root = Path(repo_root).resolve()
    warnings: list[str] = []
    generated_test_files: list[str] = []
    runtime_setup = prepare_runtime(root, task_spec.install_config)

    regression_candidate: GeneratedRegressionCandidate | None = None
    if not manifest.test_files:
        regression_candidate = _find_regression_candidate(root, manifest, runtime_setup.python_executable)
        if regression_candidate is not None:
            generated_path = _write_generated_regression_test(root, regression_candidate, manifest.layout)
            generated_test_files.append(generated_path)
            manifest = build_manifest(root)
            task_spec.test_targets = [generated_path]
            task_spec.test_command = f"python -m pytest {generated_path} -q"
            task_spec.starter_paths = [regression_candidate.source_path, generated_path]
            task_spec.title = f"Fix injected bug in {regression_candidate.function_name}"
            task_spec.problem_statement = (
                f"Fix the injected bug in `{regression_candidate.function_name}` so the generated regression test passes."
            )
            task_spec.metadata = {
                **task_spec.metadata,
                "generated_tests": generated_test_files,
                "regression_function": regression_candidate.function_name,
            }
        else:
            generated_path = _write_generated_smoke_test(root, manifest)
            generated_test_files.append(generated_path)
            manifest = build_manifest(root)
            task_spec.test_targets = [generated_path]
            task_spec.test_command = f"python -m pytest {generated_path} -q"
            task_spec.starter_paths = [manifest.source_files[0], generated_path] if manifest.source_files else [generated_path]
            warnings.append("Generated smoke tests only because no simple deterministic function candidate was found.")

    baseline = run_tests(
        root,
        python_executable=runtime_setup.python_executable,
        targets=task_spec.test_targets,
        test_command=task_spec.test_command,
    )
    if baseline.failed or baseline.errors or baseline.invalid_test_count:
        scoped_baseline = _find_green_benchmark_scope(
            root,
            manifest,
            task_spec,
            python_executable=runtime_setup.python_executable,
            failing_locations=baseline.failing_locations,
        )
        if scoped_baseline is None:
            warnings.append("Baseline tests were not fully green, so no bug was injected.")
            return BenchmarkPrepResult(
                manifest=manifest,
                task_spec=task_spec,
                generated_test_files=generated_test_files,
                injected_bug=None,
                warnings=warnings,
            )
        task_spec, baseline = scoped_baseline
        warnings.append(
            "Scoped the benchmark to a clean test target because the full baseline already had unrelated failing tests."
        )

    injected_bug = _inject_bug_variant(
        root,
        manifest,
        task_spec,
        python_executable=runtime_setup.python_executable,
        variant_index=variant_index,
        preferred_source_path=regression_candidate.source_path if regression_candidate else None,
    )
    if injected_bug is None:
        warnings.append("No deterministic bug candidate produced a failing benchmark. Exporting a clean task.")
        return BenchmarkPrepResult(
            manifest=manifest,
            task_spec=task_spec,
            generated_test_files=generated_test_files,
            injected_bug=None,
            warnings=warnings,
        )

    task_spec.title = f"Fix injected bug in {injected_bug.function_name or Path(injected_bug.path).name}"
    task_spec.problem_statement = injected_bug.description
    task_spec.hints_text = "Inspect the changed source file, rerun tests, and restore the original behavior."
    task_spec.metadata = {
        **task_spec.metadata,
        "generated_tests": generated_test_files,
        "injected_bug": injected_bug.to_dict(),
        "benchmark_variant": variant_index,
    }
    failing_test_files = [location.split("::", 1)[0] for location in injected_bug.failing_locations]
    task_spec.starter_paths = _unique_preserve_order(
        [injected_bug.path, *failing_test_files, *task_spec.starter_paths]
    )[:5]
    return BenchmarkPrepResult(
        manifest=manifest,
        task_spec=task_spec,
        generated_test_files=generated_test_files,
        injected_bug=injected_bug,
        warnings=warnings,
    )


def _find_regression_candidate(
    repo_root: Path,
    manifest: RepoManifest,
    python_executable: str,
) -> GeneratedRegressionCandidate | None:
    for source_path in manifest.source_files:
        file_path = repo_root / source_path
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name.startswith("_") or node.decorator_list:
                continue
            if node.args.vararg or node.args.kwarg or node.args.kwonlyargs:
                continue
            arg_count = len(node.args.args)
            if arg_count == 0 or arg_count > 3:
                continue
            args = [index + 2 for index in range(arg_count)]
            module_name = _module_name_from_path(source_path, manifest.layout)
            payload = _call_function_for_expected_repr(repo_root, python_executable, module_name, node.name, args)
            if payload is None:
                continue
            return GeneratedRegressionCandidate(
                source_path=source_path,
                function_name=node.name,
                args=args,
                expected_repr=payload,
            )
    return None


def _call_function_for_expected_repr(
    repo_root: Path,
    python_executable: str,
    module_name: str,
    function_name: str,
    args: list[Any],
) -> str | None:
    script = """
import importlib
import json
from pathlib import Path
import sys

repo_root = Path(sys.argv[1]).resolve()
module_name = sys.argv[2]
function_name = sys.argv[3]
args = json.loads(sys.argv[4])
sys.path.insert(0, str(repo_root))
src_dir = repo_root / "src"
if src_dir.is_dir():
    sys.path.insert(0, str(src_dir))
module = importlib.import_module(module_name)
result = getattr(module, function_name)(*args)
result_repr = repr(result)
if "<" in result_repr and ">" in result_repr:
    raise SystemExit(3)
print(result_repr)
"""
    completed = subprocess.run(
        [python_executable, "-c", script, str(repo_root), module_name, function_name, json.dumps(args)],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _write_generated_regression_test(
    repo_root: Path,
    candidate: GeneratedRegressionCandidate,
    layout: str,
) -> str:
    tests_dir = repo_root / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_path = tests_dir / "test_repo2env_generated.py"
    module_name = _module_name_from_path(candidate.source_path, layout)
    args_literal = ", ".join(repr(value) for value in candidate.args)
    test_path.write_text(
        "from "
        + module_name
        + f" import {candidate.function_name}\n\n\n"
        + f"def test_repo2env_generated_{candidate.function_name}():\n"
        + f"    assert {candidate.function_name}({args_literal}) == {candidate.expected_repr}\n",
        encoding="utf-8",
    )
    return test_path.relative_to(repo_root).as_posix()


def _write_generated_smoke_test(repo_root: Path, manifest: RepoManifest) -> str:
    tests_dir = repo_root / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    source_path = manifest.source_files[0]
    module_name = _module_name_from_path(source_path, manifest.layout)
    test_path = tests_dir / "test_repo2env_smoke.py"
    test_path.write_text(
        f"import {module_name}\n\n\ndef test_repo2env_smoke_import():\n    assert {module_name} is not None\n",
        encoding="utf-8",
    )
    return test_path.relative_to(repo_root).as_posix()


def _inject_bug_variant(
    repo_root: Path,
    manifest: RepoManifest,
    task_spec: TaskSpec,
    *,
    python_executable: str,
    variant_index: int,
    preferred_source_path: str | None,
) -> InjectedBug | None:
    successful_variant = -1
    candidate_files = _candidate_source_files(manifest, task_spec, preferred_source_path)
    for relative_path in candidate_files:
        path = repo_root / relative_path
        original_text = path.read_text(encoding="utf-8")
        candidates = _mutation_candidates_for_file(repo_root, path, original_text)
        for candidate in candidates:
            _apply_line_mutation(path, original_text, candidate)
            result = run_tests(
                repo_root,
                python_executable=python_executable,
                targets=task_spec.test_targets,
                test_command=task_spec.test_command,
            )
            if result.failed > 0 and result.invalid_test_count == 0:
                successful_variant += 1
                if successful_variant == variant_index:
                    description = _describe_bug(candidate, result)
                    return InjectedBug(
                        path=candidate.path,
                        line_number=candidate.line_number,
                        old_text=candidate.old_text,
                        new_text=candidate.new_text,
                        reason=candidate.reason,
                        function_name=candidate.function_name,
                        description=description,
                        failing_locations=result.failing_locations,
                    )
            path.write_text(original_text, encoding="utf-8")
            _invalidate_python_caches(path)
    return None


def _candidate_source_files(
    manifest: RepoManifest,
    task_spec: TaskSpec,
    preferred_source_path: str | None,
) -> list[str]:
    preferred: list[str] = []
    if preferred_source_path:
        preferred.append(preferred_source_path)
    preferred.extend(path for path in task_spec.starter_paths if path in manifest.source_files)
    test_based_names = {
        Path(test_file).name.removeprefix("test_").removesuffix(".py")
        for test_file in manifest.test_files
    }
    for source_file in manifest.source_files:
        if Path(source_file).stem in test_based_names:
            preferred.append(source_file)
    preferred.extend(manifest.source_files[:10])
    return _unique_preserve_order(preferred)


def _find_green_benchmark_scope(
    repo_root: Path,
    manifest: RepoManifest,
    task_spec: TaskSpec,
    *,
    python_executable: str,
    failing_locations: list[str],
) -> tuple[TaskSpec, TestRunResult] | None:
    failing_files = {location.split("::", 1)[0] for location in failing_locations}
    ordered_candidates = _candidate_test_files(manifest, task_spec, failing_files)
    for test_target in ordered_candidates:
        scoped_command = f"python -m pytest {test_target} -q"
        scoped_result = run_tests(
            repo_root,
            python_executable=python_executable,
            targets=[test_target],
            test_command=scoped_command,
        )
        if scoped_result.failed or scoped_result.errors or scoped_result.invalid_test_count:
            continue
        scoped_spec = TaskSpec.from_dict(task_spec.to_dict(), default_install=task_spec.install_config)
        scoped_spec.test_targets = [test_target]
        scoped_spec.test_command = scoped_command
        scoped_spec.starter_paths = _unique_preserve_order(
            [
                *_matching_source_files_for_test(manifest, test_target),
                test_target,
                *task_spec.starter_paths,
            ]
        )[:5]
        return scoped_spec, scoped_result
    return None


def _candidate_test_files(
    manifest: RepoManifest,
    task_spec: TaskSpec,
    failing_files: set[str],
) -> list[str]:
    preferred: list[str] = []
    for starter in task_spec.starter_paths:
        if starter in manifest.test_files and starter not in failing_files:
            preferred.append(starter)
    for source_file in task_spec.starter_paths:
        if source_file not in manifest.source_files:
            continue
        preferred.extend(_tests_for_source_file(manifest, source_file))
    preferred.extend(test_file for test_file in manifest.test_files if test_file not in failing_files)
    return _unique_preserve_order(preferred)


def _tests_for_source_file(manifest: RepoManifest, source_file: str) -> list[str]:
    source_stem = Path(source_file).stem
    candidates = []
    for test_file in manifest.test_files:
        test_stem = Path(test_file).stem.removeprefix("test_")
        if test_stem == source_stem:
            candidates.append(test_file)
    return candidates


def _matching_source_files_for_test(manifest: RepoManifest, test_file: str) -> list[str]:
    test_stem = Path(test_file).stem.removeprefix("test_")
    matches = []
    for source_file in manifest.source_files:
        if Path(source_file).stem == test_stem:
            matches.append(source_file)
    return matches


def _mutation_candidates_for_file(repo_root: Path, path: Path, content: str) -> list[BugCandidate]:
    function_map = _function_name_by_line(path, content)
    candidates: list[BugCandidate] = []
    lines = content.splitlines()
    for line_number, line in enumerate(lines, start=1):
        for old, new, reason in MUTATION_RULES:
            if old not in line:
                continue
            new_line = line.replace(old, new, 1)
            if new_line == line:
                continue
            candidates.append(
                BugCandidate(
                    path=path.relative_to(repo_root).as_posix(),
                    line_number=line_number,
                    old_text=line.rstrip("\n"),
                    new_text=new_line.rstrip("\n"),
                    reason=reason,
                    function_name=function_map.get(line_number),
                )
            )
    return candidates


def _apply_line_mutation(path: Path, original_text: str, candidate: BugCandidate) -> None:
    lines = original_text.splitlines()
    lines[candidate.line_number - 1] = candidate.new_text
    path.write_text("\n".join(lines) + ("\n" if original_text.endswith("\n") else ""), encoding="utf-8")
    _invalidate_python_caches(path)


def _function_name_by_line(path: Path, content: str) -> dict[int, str]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return {}
    function_lines: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", start)
            if start is None or end is None:
                continue
            for line_number in range(start, end + 1):
                function_lines[line_number] = node.name
    return function_lines


def _describe_bug(candidate: BugCandidate, result: TestRunResult) -> str:
    target = f"`{candidate.path}`"
    if candidate.function_name:
        target = f"`{candidate.path}` in `{candidate.function_name}`"
    failing = result.failing_locations[0] if result.failing_locations else "the scoped tests"
    return (
        f"A small benchmark bug was injected into {target} by changing `{candidate.old_text.strip()}` "
        f"to `{candidate.new_text.strip()}`. Fix the implementation so `{failing}` passes again."
    )


def _module_name_from_path(relative_path: str, layout: str) -> str:
    path = Path(relative_path)
    parts = list(path.with_suffix("").parts)
    if layout == "src" and parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _invalidate_python_caches(target: Path) -> None:
    if target.suffix == ".py":
        pycache_dir = target.parent / "__pycache__"
        if pycache_dir.exists():
            for compiled in pycache_dir.glob(f"{target.stem}.*.pyc"):
                compiled.unlink(missing_ok=True)
    timestamp = max(target.stat().st_mtime + 1.0, os.path.getmtime(target))
    os.utime(target, (timestamp, timestamp))
