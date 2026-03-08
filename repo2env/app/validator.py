from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from repo2env.app.manifest import RepoManifest


SUMMARY_PATTERNS = {
    "passed": re.compile(r"(\d+)\s+passed"),
    "failed": re.compile(r"(\d+)\s+failed"),
    "errors": re.compile(r"(\d+)\s+error[s]?\b"),
    "skipped": re.compile(r"(\d+)\s+skipped"),
    "xfailed": re.compile(r"(\d+)\s+xfailed"),
    "xpassed": re.compile(r"(\d+)\s+xpassed"),
}
FAILED_LOCATION_RE = re.compile(r"^(FAILED|ERROR)\s+([^\s]+?\.py(?:::[^\s]+)?)", re.MULTILINE)
DURATION_RE = re.compile(r"in\s+([0-9.]+)s")
DEFAULT_PYTEST_ARGS = ["-q", "--disable-warnings"]
DISALLOWED_COMMAND_SNIPPETS = ("&&", "||", "|", ";", ">", "<", "$(")

TRACE_SCRIPT = r"""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import trace

import pytest

IGNORED_PARTS = {".git", ".hg", ".venv", ".mypy_cache", ".pytest_cache", "__pycache__", ".repo2env"}
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"


def is_source_file(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part in IGNORED_PARTS for part in relative.parts):
        return False
    if "tests" in relative.parts or path.name.startswith("test_"):
        return False
    return path.suffix == ".py"


def statement_lines(path: Path) -> set[int]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    return {node.lineno for node in ast.walk(tree) if isinstance(node, ast.stmt) and hasattr(node, "lineno")}


repo_root = Path(sys.argv[1]).resolve()
pytest_args = sys.argv[2:]
sys.path.insert(0, str(repo_root))

stream = io.StringIO()
runner = trace.Trace(count=1, trace=0)
with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
    exit_code = runner.runfunc(pytest.main, pytest_args)

results = runner.results()
executed = {}
for (filename, lineno), _count in results.counts.items():
    file_path = Path(filename)
    try:
        relative = file_path.resolve().relative_to(repo_root)
    except Exception:
        continue
    executed.setdefault(relative.as_posix(), set()).add(lineno)

source_files = []
for candidate in sorted(repo_root.rglob("*.py")):
    if is_source_file(candidate, repo_root):
        source_files.append(candidate)

per_file = {}
covered_total = 0
statement_total = 0
for source_file in source_files:
    relative = source_file.relative_to(repo_root).as_posix()
    statements = statement_lines(source_file)
    covered = statements.intersection(executed.get(relative, set()))
    statement_total += len(statements)
    covered_total += len(covered)
    coverage_percent = round((len(covered) / len(statements) * 100.0), 2) if statements else 100.0
    per_file[relative] = {
        "covered_lines": len(covered),
        "statement_lines": len(statements),
        "coverage_percent": coverage_percent,
    }

overall = round((covered_total / statement_total * 100.0), 2) if statement_total else 100.0

print(
    json.dumps(
        {
            "exit_code": int(exit_code),
            "output": stream.getvalue(),
            "coverage_percent": overall,
            "covered_lines": covered_total,
            "statement_lines": statement_total,
            "per_file": per_file,
        }
    )
)
"""


@dataclass(slots=True)
class TestRunResult:
    exit_code: int
    passed: int
    failed: int
    errors: int
    skipped: int
    xfailed: int
    xpassed: int
    invalid_test_count: int
    failing_locations: list[str]
    duration_seconds: float | None
    output: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class CoverageRunResult:
    exit_code: int
    coverage_percent: float
    covered_lines: int
    statement_lines: int
    per_file: dict[str, dict]
    output: str

    def to_dict(self) -> dict:
        return asdict(self)


def run_tests(
    repo_root: str | Path,
    *,
    python_executable: str | Path | None = None,
    targets: Sequence[str] | None = None,
    test_command: str | None = None,
) -> TestRunResult:
    path = Path(repo_root).resolve()
    command = [
        str(python_executable or sys.executable),
        "-m",
        "pytest",
        *resolve_pytest_args(test_command=test_command, targets=targets),
    ]
    completed = subprocess.run(
        command,
        cwd=path,
        capture_output=True,
        text=True,
        env=_runtime_env(),
    )
    output = _merge_output(completed.stdout, completed.stderr)
    return TestRunResult(
        exit_code=completed.returncode,
        passed=_extract_count("passed", output),
        failed=_extract_count("failed", output),
        errors=_extract_count("errors", output),
        skipped=_extract_count("skipped", output),
        xfailed=_extract_count("xfailed", output),
        xpassed=_extract_count("xpassed", output),
        invalid_test_count=_count_invalid_tests(output, completed.returncode),
        failing_locations=sorted({match.group(2) for match in FAILED_LOCATION_RE.finditer(output)}),
        duration_seconds=_extract_duration(output),
        output=output,
    )


def run_coverage(
    repo_root: str | Path,
    manifest: RepoManifest | None = None,
    *,
    python_executable: str | Path | None = None,
    targets: Sequence[str] | None = None,
    test_command: str | None = None,
) -> CoverageRunResult:
    path = Path(repo_root).resolve()
    pytest_args = resolve_pytest_args(test_command=test_command, targets=targets)
    completed = subprocess.run(
        [str(python_executable or sys.executable), "-c", TRACE_SCRIPT, str(path), *pytest_args],
        cwd=path,
        capture_output=True,
        text=True,
        env=_runtime_env(),
    )
    if completed.returncode not in {0, 1, 2, 5}:
        raise RuntimeError(completed.stderr or completed.stdout or "Coverage run failed")
    payload = json.loads(completed.stdout)
    return CoverageRunResult(
        exit_code=payload["exit_code"],
        coverage_percent=payload["coverage_percent"],
        covered_lines=payload["covered_lines"],
        statement_lines=payload["statement_lines"],
        per_file=payload["per_file"],
        output=payload["output"],
    )


def _runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return env


def resolve_pytest_args(
    *,
    test_command: str | None = None,
    targets: Sequence[str] | None = None,
) -> list[str]:
    normalized_targets = [str(target) for target in (targets or [])]
    if test_command and test_command.strip():
        command_text = test_command.strip()
        if any(snippet in command_text for snippet in DISALLOWED_COMMAND_SNIPPETS):
            raise ValueError(
                "Unsupported test_command: shell operators and redirection are not allowed. "
                "Provide plain pytest arguments or split setup into install_commands."
            )
        tokens = shlex.split(command_text)
        pytest_args = _extract_pytest_args(tokens)
        for target in normalized_targets:
            if target not in pytest_args:
                pytest_args.append(target)
        return pytest_args
    if normalized_targets:
        return [*DEFAULT_PYTEST_ARGS, *normalized_targets]
    return list(DEFAULT_PYTEST_ARGS)


def _extract_count(label: str, output: str) -> int:
    pattern = SUMMARY_PATTERNS[label]
    match = pattern.search(output)
    return int(match.group(1)) if match else 0


def _extract_duration(output: str) -> float | None:
    match = DURATION_RE.search(output)
    return float(match.group(1)) if match else None


def _count_invalid_tests(output: str, exit_code: int) -> int:
    collect_errors = len(re.findall(r"ERROR collecting", output))
    syntax_errors = len(re.findall(r"SyntaxError", output))
    broken_imports = len(re.findall(r"ImportError while importing test module", output))
    count = collect_errors + syntax_errors + broken_imports
    if count == 0 and exit_code in {2, 4}:
        return 1
    return count


def _merge_output(stdout: str, stderr: str) -> str:
    joined = stdout
    if stderr:
        joined = f"{joined}\n{stderr}" if joined else stderr
    return joined.strip()


def _extract_pytest_args(tokens: list[str]) -> list[str]:
    if not tokens:
        raise ValueError("test_command cannot be empty.")
    executable = Path(tokens[0]).name
    if executable in {"pytest", "py.test"}:
        return tokens[1:]
    if executable.startswith("python"):
        if len(tokens) < 3 or tokens[1:3] != ["-m", "pytest"]:
            raise ValueError(
                "Unsupported test_command. Use `pytest ...` or `python -m pytest ...`."
            )
        return tokens[3:]
    raise ValueError(
        "Unsupported test_command. Use `pytest ...` or `python -m pytest ...`."
    )
