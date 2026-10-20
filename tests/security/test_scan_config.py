"""The CI security scans: the Makefile scan-secrets and audit-deps targets and the gitleaks allow-lists.

The unit tests read the committed Makefile and .gitleaks.toml; they download nothing and run no scanner. The tests
marked integration run a real gitleaks binary (the GITLEAKS_BIN environment variable, or `gitleaks` on PATH; skipped
when there is none) over a throwaway git repository. Every key in that repository is a SYNTHETIC value built at run
time from a hash, so this file holds nothing key-shaped for the repository's own secret scan to flag.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"
GITLEAKS_CONFIG = REPO_ROOT / ".gitleaks.toml"


def makefile_variable(name: str) -> str:
    """The value of a simple `NAME := value` line in the Makefile."""
    for line in MAKEFILE.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name} :="):
            return line.split(":=", 1)[1].strip()
    raise AssertionError(f"{name} is not set in the Makefile")


def makefile_recipe(target: str) -> str:
    """The recipe lines (tab-indented) under `target:` in the Makefile, joined with newlines."""
    lines = MAKEFILE.read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith(f"{target}:"))
    recipe = []
    for line in lines[start + 1 :]:
        if not line.startswith("\t"):
            break
        recipe.append(line)
    return "\n".join(recipe)


def test_scan_secrets_also_scans_merge_commit_diffs() -> None:
    """Plain `git log -p` shows no diff for a merge commit; -m makes gitleaks scan what a merge resolution added."""
    log_options = makefile_variable("GITLEAKS_LOG_OPTS").split()
    assert "-m" in log_options
    assert "--all" in log_options and "--full-history" in log_options
    recipe = makefile_recipe("scan-secrets")
    assert re.search(r'gitleaks" detect .*--log-opts="\$\(GITLEAKS_LOG_OPTS\)"', recipe)


# ---------------------------------------------------------------------------------------------------------------
# Integration: a real gitleaks binary over a throwaway repository
# ---------------------------------------------------------------------------------------------------------------


def synthetic_key(label: str) -> str:
    """A SYNTHETIC, key-shaped value (32 lowercase hex characters). It is a hash of a label, never a credential."""
    return hashlib.sha256(f"SYNTHETIC test value: {label}".encode()).hexdigest()[:32]


def gitleaks_binary() -> str:
    """The gitleaks binary to run, or skip the test when there is none."""
    binary = os.environ.get("GITLEAKS_BIN") or shutil.which("gitleaks")
    if not binary:
        pytest.skip("no gitleaks binary: set GITLEAKS_BIN or put gitleaks on PATH")
    return binary


def git(repo: Path, *arguments: str, check: bool = True) -> str:
    """Run git in repo with a fixed identity and no signing; return its standard output."""
    command = ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false"]
    result = subprocess.run([*command, *arguments], cwd=repo, check=check, capture_output=True, text=True)
    return result.stdout.strip()


def commit_files(repo: Path, files: dict[str, str], message: str) -> None:
    """Write the files (relative paths) and commit them."""
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        git(repo, "add", name)
    git(repo, "commit", "-q", "-m", message)


def run_gitleaks(repo: Path, report: Path) -> list[dict]:
    """Scan repo's history the way `make scan-secrets` does (same config and git log options); return the findings."""
    command = [
        gitleaks_binary(),
        "detect",
        "--source",
        str(repo),
        "--config",
        str(GITLEAKS_CONFIG),
        f"--log-opts={makefile_variable('GITLEAKS_LOG_OPTS')}",
        "--report-format",
        "json",
        "--report-path",
        str(report),
        "--no-banner",
        "--exit-code",
        "0",
    ]
    subprocess.run(command, check=True, capture_output=True)
    return json.loads(report.read_text(encoding="utf-8"))


@pytest.mark.integration
def test_gitleaks_reports_a_key_added_while_resolving_a_merge_conflict(tmp_path: Path) -> None:
    """The key exists only in the merge commit, so a scan without -m never sees it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    commit_files(repo, {"settings.txt": "mode = base\n"}, "base")
    git(repo, "checkout", "-q", "-b", "side")
    commit_files(repo, {"settings.txt": "mode = side\n"}, "side")
    git(repo, "checkout", "-q", "main")
    commit_files(repo, {"settings.txt": "mode = main\n"}, "main")
    git(repo, "merge", "-q", "side", check=False)  # conflicts; resolved below, with a SYNTHETIC key typed in
    key = synthetic_key("merge resolution")
    commit_files(repo, {"settings.txt": f'mode = main\nservice_api_key = "{key}"\n'}, "merge side")
    merge_commit = git(repo, "rev-parse", "HEAD")
    assert len(git(repo, "rev-list", "--parents", "-n", "1", "HEAD").split()) == 3  # it is a merge commit

    findings = run_gitleaks(repo, tmp_path / "report.json")

    assert findings, "the key typed in during the merge resolution was not reported"
    assert {finding["File"] for finding in findings} == {"settings.txt"}
    assert {finding["Commit"] for finding in findings} == {merge_commit}
    assert {finding["Secret"] for finding in findings} == {key}
