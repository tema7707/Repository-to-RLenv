from pathlib import Path
import shutil

import anyio
from fastmcp.client import Client

from repo2env.mcp.server import build_mcp_server


EXAMPLE_REPO = Path("examples/repo_b").resolve()


async def _call_analyze_tool() -> dict:
    server = build_mcp_server()
    async with Client(server) as client:
        tools = await client.list_tools()
        assert {tool.name for tool in tools} >= {
            "analyze_repo",
            "write_repo_task_spec",
            "convert_repo",
            "validate_openenv_export",
            "push_openenv_export",
            "convert_and_push_repo",
        }
        result = await client.call_tool(
            "analyze_repo",
            {"repo_path": str(EXAMPLE_REPO), "run_validation": False, "max_steps": 16},
        )
        return result.structured_content or result.data


async def _get_convert_prompt() -> str:
    server = build_mcp_server()
    async with Client(server) as client:
        prompts = await client.list_prompts()
        assert {prompt.name for prompt in prompts} >= {"convert_to_rl_env"}
        result = await client.get_prompt(
            "convert_to_rl_env",
            {"hf_space_repo_id": "demo-user/demo-space"},
        )
        return result.messages[0].content.text


async def _get_convert_prompt_without_repo_id() -> str:
    server = build_mcp_server()
    async with Client(server) as client:
        result = await client.get_prompt("convert_to_rl_env", {"hf_space_repo_id": ""})
        return result.messages[0].content.text


async def _write_task_spec(repo_path: str) -> dict:
    server = build_mcp_server()
    async with Client(server) as client:
        result = await client.call_tool(
            "write_repo_task_spec",
            {
                "repo_path": repo_path,
                "task_mode": "repo_editing",
                "title": "Increase passing tests",
                "problem_statement": "Edit repository files and improve the number of passing tests.",
                "starter_paths": ["payouts.py", "tests/test_payouts_pr.py"],
                "allow_source_edits": True,
            },
        )
        return result.structured_content or result.data


def test_mcp_server_exposes_analyze_repo_tool() -> None:
    payload = anyio.run(_call_analyze_tool)

    assert payload["spec"]["repo_name"] == "repo_b"
    assert payload["spec"]["support"]["supported"] is True
    assert payload["spec"]["runtime"]["default_reset_parameters"]["max_steps"] == 16


def test_mcp_server_exposes_convert_prompt() -> None:
    text = anyio.run(_get_convert_prompt)

    assert "convert_repo" in text
    assert "Use Claude Code to author the benchmark bug." in text
    assert "Create a staged working copy in a repo-specific temp directory" in text
    assert "Inject one small source-code bug yourself in the staged copy" in text
    assert "Mutate existing source files only; do not create new package shims or top-level module files" in text
    assert "For `src/`-layout repos, prefer editing files under `src/<package>/...`" in text
    assert "Never create a top-level file that shadows the package import name" in text
    assert "After the edit, run the chosen tests in the staged copy and confirm the new bug makes them fail" in text
    assert "push_openenv_export" in text
    assert "demo-user/demo-space" in text
    assert "max_steps" in text
    assert "Do not jump to very large budgets like `100`" in text
    assert 'repo_path: "<staged repo path>"' in text
    assert "prepare_benchmark: false" in text
    assert "Analyze the tests first and choose a clean passing test scope" in text
    assert "Do not inject bugs into test files" in text
    assert "write_spec_to_repo: false" in text
    assert "Do not modify `pyproject.toml`, `setup.py`, `requirements*`, or source files" in text
    assert "do not patch the source repo to work around them" in text
    assert 'output_dir: "/tmp/repo2env-<repo-name>-export"' in text
    assert "Do not reuse `/tmp/repo2env-export` across different repos" in text
    assert "If an earlier export directory looks stale or belongs to another repo" in text


def test_mcp_server_convert_prompt_pushes_without_repo_id() -> None:
    text = anyio.run(_get_convert_prompt_without_repo_id)

    assert "push_openenv_export" in text
    assert "Do not push yet" not in text
    assert "omit `repo_id` unless the platform requires one" in text
    assert "`enable_interface: false`" in text


def test_mcp_server_write_task_spec_tool_writes_valid_schema(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo_b_copy"
    shutil.copytree(EXAMPLE_REPO, repo_copy)

    payload = anyio.run(_write_task_spec, str(repo_copy))
    task_spec_path = Path(payload["task_spec_path"])
    task_spec_text = task_spec_path.read_text(encoding="utf-8")

    assert task_spec_path == repo_copy / "task_spec.json"
    assert '"task_mode": "repo_editing"' in task_spec_text
    assert '"problem_statement": "Edit repository files and improve the number of passing tests."' in task_spec_text
    assert '"allow_source_edits": true' in task_spec_text
