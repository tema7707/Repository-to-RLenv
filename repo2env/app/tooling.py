from __future__ import annotations

from typing import Any


REPO2ENV_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "list_files": {
        "description": "List visible repository files.",
        "required_args": [],
        "optional_args": {
            "limit": "Maximum number of file paths to return.",
        },
    },
    "read_file": {
        "description": "Read a repository file.",
        "required_args": {
            "path": "Repository-relative file path.",
        },
        "optional_args": {},
    },
    "read_file_chunk": {
        "description": "Read a line range from a repository file.",
        "required_args": {
            "path": "Repository-relative file path.",
            "start_line": "1-based starting line number.",
        },
        "optional_args": {
            "end_line": "Inclusive ending line number. Defaults to start_line + 199.",
        },
    },
    "search_code": {
        "description": "Search visible Python files for a text query.",
        "required_args": {
            "query": "Case-insensitive text query to search for.",
        },
        "optional_args": {
            "limit": "Maximum number of matches to return.",
        },
    },
    "replace_in_file": {
        "description": "Replace a specific snippet inside an existing file.",
        "required_args": {
            "path": "Repository-relative file path.",
            "old_text": "Exact existing text to replace.",
            "new_text": "Replacement text.",
        },
        "optional_args": {
            "count": "Maximum number of replacements. Defaults to 1.",
        },
    },
    "insert_after": {
        "description": "Insert text immediately after a matching snippet in an existing file.",
        "required_args": {
            "path": "Repository-relative file path.",
            "anchor_text": "Exact existing text to insert after.",
            "new_text": "Text to insert.",
        },
        "optional_args": {
            "count": "Maximum number of insertions. Defaults to 1.",
        },
    },
    "insert_before": {
        "description": "Insert text immediately before a matching snippet in an existing file.",
        "required_args": {
            "path": "Repository-relative file path.",
            "anchor_text": "Exact existing text to insert before.",
            "new_text": "Text to insert.",
        },
        "optional_args": {
            "count": "Maximum number of insertions. Defaults to 1.",
        },
    },
    "append_to_file": {
        "description": "Append text to the end of an existing file.",
        "required_args": {
            "path": "Repository-relative file path.",
            "content": "Text to append.",
        },
        "optional_args": {},
    },
    "write_file": {
        "description": "Overwrite or create a whole file. Prefer replace_in_file for small edits.",
        "required_args": {
            "path": "Repository-relative file path.",
            "content": "Complete new file content.",
        },
        "optional_args": {},
    },
    "run_tests": {
        "description": "Run the task-scoped pytest command and refresh metrics.",
        "required_args": [],
        "optional_args": {},
    },
    "get_test_failures": {
        "description": "Return structured details for the most recent failing pytest cases.",
        "required_args": [],
        "optional_args": {
            "limit": "Maximum number of failures to return.",
        },
    },
    "diff_working_tree": {
        "description": "Show diffs between the current sandbox and the initial repo snapshot.",
        "required_args": [],
        "optional_args": {
            "path": "Optional repository-relative path to restrict the diff.",
            "context_lines": "Unified diff context lines. Defaults to 3.",
            "max_diff_chars": "Maximum diff characters to return. Defaults to 12000.",
        },
    },
    "submit": {
        "description": "End the episode.",
        "required_args": [],
        "optional_args": {},
    },
}

REPO2ENV_TOOL_NAMES = tuple(REPO2ENV_TOOL_SCHEMAS.keys())

REPO2ENV_TOOL_ARG_ALIASES: dict[str, dict[str, str]] = {
    "read_file": {
        "file_path": "path",
    },
    "read_file_chunk": {
        "file_path": "path",
        "start": "start_line",
        "end": "end_line",
    },
    "search_code": {
        "search_term": "query",
    },
    "replace_in_file": {
        "file_path": "path",
        "old": "old_text",
        "new": "new_text",
    },
    "insert_after": {
        "file_path": "path",
        "anchor": "anchor_text",
        "text": "new_text",
        "insert_text": "new_text",
    },
    "insert_before": {
        "file_path": "path",
        "anchor": "anchor_text",
        "text": "new_text",
        "insert_text": "new_text",
    },
    "append_to_file": {
        "file_path": "path",
        "text": "content",
    },
    "write_file": {
        "file_path": "path",
        "text": "content",
    },
    "diff_working_tree": {
        "file_path": "path",
    },
}


def normalize_tool_args(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(args)
    for alias, canonical in REPO2ENV_TOOL_ARG_ALIASES.get(tool, {}).items():
        if alias in normalized and canonical not in normalized:
            normalized[canonical] = normalized.pop(alias)
    return normalized
