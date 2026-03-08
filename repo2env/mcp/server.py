from __future__ import annotations

from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.prompts import PromptResult

from repo2env.app.install import detect_install_recipe
from repo2env.app.manifest import build_manifest
from repo2env.app.task_spec import build_task_spec, write_task_spec
from repo2env.conversion import (
    analyze_repository,
    convert_and_push_repository,
    convert_repository,
    push_export,
    validate_export,
)

INSTRUCTIONS = """Use these tools to convert the currently opened Python repo into a standardized Repo2Env/OpenEnv package.
Inspect the repo first, but do not edit package metadata, build files, or source files in the user's repo just to make conversion succeed. Prefer benchmark-prep exports that mutate only the staged/exported copy, then call convert_repo to snapshot a deployable HF/OpenEnv environment."""


def build_mcp_server() -> FastMCP:
    mcp = FastMCP(
        name="Repo2Env",
        instructions=INSTRUCTIONS,
        version="0.1.0",
    )

    @mcp.tool(
        title="Analyze Repo",
        description=(
            "Analyze a Python repo and emit a standardized rlenv spec without modifying the repo. "
            "Defaults to the current working directory."
        ),
    )
    def analyze_repo(
        repo_path: str = ".",
        run_validation: bool = True,
        max_steps: int = 12,
        prepare_benchmark: bool = False,
        benchmark_variant: int = 0,
        require_injected_bug: bool = True,
    ) -> dict[str, Any]:
        result = analyze_repository(
            repo_path,
            run_validation=run_validation,
            max_steps=max_steps,
            prepare_benchmark=prepare_benchmark,
            benchmark_variant=benchmark_variant,
            require_injected_bug=require_injected_bug,
        )
        return result.to_dict()

    @mcp.tool(
        title="Write Task Spec",
        description=(
            "Write a validated Repo2Env task_spec.json for the target repo. "
            "Use this instead of manually authoring task_spec.json."
        ),
    )
    def write_repo_task_spec(
        repo_path: str = ".",
        task_mode: str = "repo_editing",
        title: str | None = None,
        problem_statement: str | None = None,
        hints_text: str | None = None,
        test_command: str | None = None,
        test_targets: list[str] | None = None,
        fail_to_pass: list[str] | None = None,
        pass_to_pass: list[str] | None = None,
        starter_paths: list[str] | None = None,
        allow_source_edits: bool | None = None,
        install_commands: list[str] | None = None,
        install_strategy: str | None = None,
        destination_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        repo_root = Path(repo_path).expanduser().resolve()
        manifest = build_manifest(repo_root)
        install_recipe = detect_install_recipe(repo_root, manifest)
        spec = build_task_spec(
            manifest,
            install_recipe,
            task_mode=task_mode,
            title=title,
            problem_statement=problem_statement,
            hints_text=hints_text,
            test_command=test_command,
            test_targets=test_targets,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            starter_paths=starter_paths,
            allow_source_edits=allow_source_edits,
            install_commands=install_commands,
            install_strategy=install_strategy,
            metadata=metadata,
        )
        destination = Path(destination_path) if destination_path else Path("task_spec.json")
        output_path = write_task_spec(
            spec,
            destination if destination.is_absolute() else repo_root / destination,
        )
        return {
            "repo_path": str(repo_root),
            "task_spec_path": str(output_path),
            "task_spec": spec.to_dict(),
        }

    @mcp.tool(
        title="Convert Repo",
        description=(
            "Snapshot the repo, write rlenv_spec.json, and export a deployable OpenEnv/HF package. "
            "Claude should inspect or modify the repo before calling this tool if needed."
        ),
    )
    def convert_repo(
        repo_path: str = ".",
        output_dir: str | None = None,
        write_spec_to_repo: bool = True,
        overwrite: bool = False,
        run_validation: bool = True,
        max_steps: int = 12,
        prepare_benchmark: bool = False,
        benchmark_variant: int = 0,
        require_injected_bug: bool = True,
    ) -> dict[str, Any]:
        result = convert_repository(
            repo_path,
            output_dir=output_dir,
            write_spec_to_repo=write_spec_to_repo,
            overwrite=overwrite,
            run_validation=run_validation,
            max_steps=max_steps,
            prepare_benchmark=prepare_benchmark,
            benchmark_variant=benchmark_variant,
            require_injected_bug=require_injected_bug,
        )
        return result.to_dict()

    @mcp.tool(
        title="Validate Export",
        description="Run `openenv validate` against an exported Repo2Env package.",
    )
    def validate_openenv_export(export_dir: str) -> dict[str, Any]:
        return validate_export(export_dir).to_dict()

    @mcp.tool(
        title="Push Export",
        description=(
            "Run `openenv push` for an exported Repo2Env package. Requires local OpenEnv auth and network access."
        ),
    )
    def push_openenv_export(
        export_dir: str,
        repo_id: str | None = None,
        private: bool = False,
        base_image: str | None = None,
        enable_interface: bool = False,
    ) -> dict[str, Any]:
        return push_export(
            export_dir,
            repo_id=repo_id,
            private=private,
            base_image=base_image,
            enable_interface=enable_interface,
        ).to_dict()

    @mcp.tool(
        title="Convert And Push Repo",
        description=(
            "Convert the repo into a deployable OpenEnv package and immediately run `openenv push`. "
            "Use this when the repo is already ready and you want a one-shot export to HF."
        ),
    )
    def convert_and_push_repo(
        repo_path: str = ".",
        output_dir: str | None = None,
        repo_id: str | None = None,
        private: bool = False,
        overwrite: bool = False,
        run_validation: bool = True,
        write_spec_to_repo: bool = True,
        max_steps: int = 12,
        base_image: str | None = None,
        enable_interface: bool = False,
        prepare_benchmark: bool = False,
        benchmark_variant: int = 0,
        require_injected_bug: bool = True,
    ) -> dict[str, Any]:
        return convert_and_push_repository(
            repo_path,
            output_dir=output_dir,
            repo_id=repo_id,
            private=private,
            overwrite=overwrite,
            run_validation=run_validation,
            write_spec_to_repo=write_spec_to_repo,
            max_steps=max_steps,
            base_image=base_image,
            enable_interface=enable_interface,
            prepare_benchmark=prepare_benchmark,
            benchmark_variant=benchmark_variant,
            require_injected_bug=require_injected_bug,
        )

    @mcp.prompt(
        name="convert_to_rl_env",
        title="Convert To RL Env",
        description=(
            "Inspect the current repo, create or update task_spec.json if needed, "
            "then use Repo2Env MCP tools to analyze, convert, validate, and optionally push."
        ),
    )
    def convert_to_rl_env(hf_space_repo_id: str = "") -> PromptResult:
        push_step = (
            "8. Call the Repo2Env MCP tool `push_openenv_export` with:\n"
            "   - `export_dir: <returned export_dir>`\n"
            f"   - `repo_id: \"{hf_space_repo_id}\"`\n"
            "   - `enable_interface: false`\n"
            "9. Include the push result and resulting Hugging Face Space URL in the final summary.\n"
            if hf_space_repo_id.strip()
            else "8. Call the Repo2Env MCP tool `push_openenv_export` with:\n"
            "   - `export_dir: <returned export_dir>`\n"
            "   - omit `repo_id` unless the platform requires one\n"
            "   - `enable_interface: false`\n"
            "9. Include the push result and resulting Hugging Face Space URL in the final summary.\n"
        )
        return PromptResult(
            f"""Convert the current repository into a Repo2Env/OpenEnv RL environment.

Workflow:

1. Inspect the current repo and identify the files and tests that define the benchmark.
   - Do not modify `pyproject.toml`, `setup.py`, `requirements*`, or source files in the user's repo just to make Repo2Env conversion succeed.
   - All bug injection and benchmark shaping should happen in a staged copy, not in the source repo.
2. Use Claude Code to author the benchmark bug.
   - Create a staged working copy in a repo-specific temp directory shaped like `/tmp/repo2env-<repo-name>-bug-<variant>-staged`
   - Keep the source repo untouched
   - Analyze the tests first and choose a clean passing test scope
   - Inject one small source-code bug yourself in the staged copy
   - Mutate existing source files only; do not create new package shims or top-level module files
   - For `src/`-layout repos, prefer editing files under `src/<package>/...`
   - Never create a top-level file that shadows the package import name (for example `humanize.py` in a repo whose package is `humanize`)
   - Do not inject bugs into test files
   - After the edit, run the chosen tests in the staged copy and confirm the new bug makes them fail
   - Use a different staged directory or different bug choice when creating multiple variants for the same repo
3. Choose an explicit `max_steps` budget for the environment.
   - Use `12` for very small one-edit tasks.
   - Use `16` for tasks that likely need one extra inspect/edit/test cycle.
   - Do not jump to very large budgets like `100` unless there is a concrete reason.
4. After you inject the bug in the staged copy, call `write_repo_task_spec` on that staged copy to create a validated `task_spec.json` with fields such as:
   - `task_mode`
   - `title`
   - `problem_statement`
   - `test_command`
   - `test_targets`
   - `starter_paths`
   - `allow_source_edits`
   - Describe the specific bug Claude inserted so the UI explains the task clearly.
5. Call the Repo2Env MCP tool `analyze_repo` with:
   - `repo_path: "<staged repo path>"`
   - `run_validation: true`
   - the explicit `max_steps` value you chose
   - `prepare_benchmark: false`
   - If analysis reports setup or packaging issues, do not patch the source repo to work around them unless the user explicitly asked for permanent repo edits.
6. Summarize:
   - support status
   - detected install recipe
   - baseline pytest state
   - selected benchmark scope and why Claude chose it
   - the exact source bug Claude inserted
   - chosen step budget and why
   - any conversion risks
7. Choose a repo-specific export directory and avoid generic shared temp paths.
   - Use a path shaped like `/tmp/repo2env-<repo-name>-export`
   - Do not reuse `/tmp/repo2env-export` across different repos
   - If an earlier export directory looks stale or belongs to another repo, do not inspect or patch it; rerun conversion with a fresh repo-specific output directory instead
8. Call the Repo2Env MCP tool `convert_repo` with:
   - `repo_path: "<staged repo path>"`
   - `output_dir: "/tmp/repo2env-<repo-name>-export"`
   - `overwrite: true`
   - `run_validation: true`
   - the same explicit `max_steps` value
   - `prepare_benchmark: false`
   - `write_spec_to_repo: false`
   - Do not edit the source repo between analysis and conversion unless the user explicitly asked for permanent repo changes.
9. Call the Repo2Env MCP tool `validate_openenv_export` with the returned `export_dir`.
{push_step}

End with a concise result block containing:
- `export_dir`
- `export_spec_path`
- `export_task_spec_path`
- validation outcome
- push outcome
- the exact next command or URL the user should use

Do not stop at advice. Use the MCP tools and perform the conversion."""
        )

    return mcp


mcp = build_mcp_server()


def main() -> None:
    mcp.run(transport="stdio")


__all__ = ["build_mcp_server", "mcp", "main"]
