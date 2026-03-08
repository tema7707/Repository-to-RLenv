---
description: Convert the current repo into a Repo2Env/OpenEnv package and optionally push it to Hugging Face
argument-hint: [hf-space-repo-id optional]
---

Convert the current repository into a Repo2Env/OpenEnv RL environment.

Workflow:

1. Inspect the current repo and decide whether the best demo task is `test_generation` or `pr_fix`.
2. Do not write `task_spec.json` by hand. Call the Repo2Env MCP tool `write_repo_task_spec` to create or overwrite a validated root-level `task_spec.json`.
   - Prefer `test_generation` unless there is a small, well-scoped fix with clear `FAIL_TO_PASS` and `PASS_TO_PASS` tests.
   - Keep the task narrow and demoable.
   - If setup needs a custom bootstrap sequence, pass explicit `install_commands`.
3. When calling `write_repo_task_spec`, use Repo2Env schema fields such as:
   - `task_mode`
   - `title`
   - `problem_statement`
   - `test_command`
   - `test_targets`
   - `fail_to_pass`
   - `pass_to_pass`
   - `starter_paths`
   - `allow_source_edits`
4. Call the Repo2Env MCP tool `analyze_repo` with:
   - `repo_path: "."`
   - `run_validation: true`
5. Summarize:
   - support status
   - detected install recipe
   - baseline pytest/coverage state
   - selected task mode
   - any conversion risks
6. Call the Repo2Env MCP tool `convert_repo` with:
   - `repo_path: "."`
   - `output_dir: "/tmp/repo2env-export"`
   - `overwrite: true`
   - `run_validation: true`
7. Call the Repo2Env MCP tool `validate_openenv_export` with the returned `export_dir`.
8. If `$ARGUMENTS` is non-empty, treat it as the Hugging Face repo id and call the Repo2Env MCP tool `push_openenv_export` with:
   - `export_dir: <returned export_dir>`
   - `repo_id: $ARGUMENTS`
9. End with a concise result block containing:
   - `export_dir`
   - `export_spec_path`
   - `export_task_spec_path`
   - validation outcome
   - push outcome if attempted
   - the exact next command or URL the user should use

Do not stop at advice. Use the MCP tools and perform the conversion.
