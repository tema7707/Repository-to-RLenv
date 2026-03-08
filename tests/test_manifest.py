from pathlib import Path

from repo2env.app.manifest import build_manifest


def test_manifest_detects_source_and_test_files() -> None:
    repo_path = Path("examples/repo_a").resolve()
    manifest = build_manifest(repo_path)

    assert manifest.repo_name == "repo_a"
    assert "billing.py" in manifest.source_files
    assert "tests/test_billing.py" in manifest.test_files
    assert manifest.default_test_dir == "tests"

