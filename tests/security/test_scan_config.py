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
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"
GITLEAKS_CONFIG = REPO_ROOT / ".gitleaks.toml"


def synthetic_key(label: str) -> str:
    """A SYNTHETIC, key-shaped value (32 lowercase hex characters). It is a hash of a label, never a credential."""
    return hashlib.sha256(f"SYNTHETIC test value: {label}".encode()).hexdigest()[:32]


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


def locked_requirements(text: str) -> dict[str, list[str]]:
    """Map each `name==version` in a `uv pip compile --generate-hashes` file to its hashes."""
    hashes: dict[str, list[str]] = {}
    current = ""
    for raw_line in text.splitlines():
        line = raw_line.strip().removesuffix("\\").strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("--hash="):
            hashes[current].append(line.removeprefix("--hash="))
        else:
            current = line.split(";")[0].strip()  # drop an environment marker such as `; python_full_version < '3.13'`
            hashes[current] = []
    return hashes


def test_audit_deps_installs_pip_audit_only_from_the_hashed_lock() -> None:
    """pip-audit's own dependencies are pinned and hashed, not resolved afresh from PyPI on every run."""
    lock_path = REPO_ROOT / makefile_variable("PIP_AUDIT_LOCK")
    recipe = makefile_recipe("audit-deps")
    assert "uv pip install" in recipe and "--require-hashes --requirement $(PIP_AUDIT_LOCK)" in recipe
    assert "uvx" not in recipe  # uvx would resolve pip-audit's dependencies to their newest releases

    lock_text = lock_path.read_text(encoding="utf-8")
    assert "--generate-hashes" in lock_text and "--exclude-newer" in lock_text  # in the command at its top
    requirements = locked_requirements(lock_text)
    lock_input = (lock_path.parent / "requirements.in").read_text(encoding="utf-8").splitlines()
    wanted = [line for line in lock_input if line and not line.startswith("#")]
    assert len(wanted) == 1 and wanted[0].startswith("pip-audit==") and wanted[0] in requirements
    for requirement, hashes in requirements.items():
        assert re.fullmatch(r"[a-z0-9][a-z0-9._-]*==[0-9][0-9A-Za-z.+!-]*", requirement), requirement
        assert hashes and all(re.fullmatch(r"sha256:[0-9a-f]{64}", value) for value in hashes), requirement


# ---------------------------------------------------------------------------------------------------------------
# .gitleaks.toml allow-lists
# ---------------------------------------------------------------------------------------------------------------


def allowlists() -> list[dict]:
    """The [[allowlists]] tables of .gitleaks.toml."""
    with GITLEAKS_CONFIG.open("rb") as handle:
        return tomllib.load(handle)["allowlists"]


def skipping_allowlists(rule: str, path: str, secret: str, match: str, line: str) -> list[str]:
    """Descriptions of the allow-lists that would skip this finding.

    A simplified copy of how gitleaks 8 applies an allow-list: it must name the rule; with condition AND, a path
    regex must be found in the file path AND a regex must be found in the target (`regexTarget`: the captured
    secret by default, or the whole match, or the whole line). The regexes here use only syntax that Go's RE2 and
    Python's `re` read the same way.
    """
    skipped_by = []
    for allowlist in allowlists():
        if rule not in allowlist["targetRules"]:
            continue
        target = {"secret": secret, "match": match, "line": line}[allowlist.get("regexTarget", "secret")]
        path_found = any(re.search(pattern, path) for pattern in allowlist["paths"])
        regex_found = any(re.search(pattern, target) for pattern in allowlist["regexes"])
        assert allowlist["condition"] == "AND"  # every allow-list in this config; OR is not modelled here
        if path_found and regex_found:
            skipped_by.append(allowlist["description"])
    return skipped_by


DIGEST = hashlib.sha256(b"SYNTHETIC key-facts file").hexdigest()
OUTCOME_FIELD = "p_stressed_dscr_at_least_min_year2"

# The findings the two narrowed allow-lists were written for: (path, secret, match, line). The match and line are
# templates filled in at run time, so this file does not itself hold the key-shaped text the scan flags.
NARROWED_ALLOWLIST_FINDINGS = [
    ("results/injection/planned_requests.json", DIGEST, 'key_facts_sha256": "{}"', '  "key_facts_sha256": "{}",'),
    ("results/smoke.json", DIGEST, 'rag_key_facts.jsonl": "{}"', '    "eval/rag_key_facts.jsonl": "{}"'),
    ("frontend/components/studio/project/path-panel.tsx", OUTCOME_FIELD, 'key: "{}"', '  {{ key: "{}" }},'),
]


@pytest.mark.parametrize(("path", "secret", "match", "line"), NARROWED_ALLOWLIST_FINDINGS)
def test_the_known_non_secrets_are_still_skipped(path: str, secret: str, match: str, line: str) -> None:
    skipped_by = skipping_allowlists("generic-api-key", path, secret, match.format(secret), line.format(secret))
    assert len(skipped_by) == 1


def test_a_key_on_the_same_line_as_a_key_facts_digest_is_still_reported() -> None:
    """A results JSON written on one line: the digest's allow-list must not hide the key next to it."""
    key = synthetic_key("one-line results JSON")
    line = json.dumps({"key_facts_sha256": DIGEST, "service_api_key": key})
    match = f'service_api_key": "{key}"'
    assert skipping_allowlists("generic-api-key", "results/one_line.json", key, match, line) == []


def test_another_p_prefixed_key_in_the_path_panel_is_still_reported() -> None:
    key = "p_live_" + synthetic_key("path panel")
    line = f'const token = "{key}";'
    path = "frontend/components/studio/project/path-panel.tsx"
    assert skipping_allowlists("generic-api-key", path, key, f'token = "{key}"', line) == []


def test_every_allowlist_regex_is_anchored_and_none_reads_the_whole_line() -> None:
    """The header promises each allow-list matches the exact shape of the flagged value."""
    for allowlist in allowlists():
        assert allowlist.get("regexTarget", "secret") in {"secret", "match"}, allowlist["description"]
        for pattern in [*allowlist["regexes"], *allowlist["paths"]]:
            assert pattern.startswith("^") and pattern.endswith("$"), pattern


# ---------------------------------------------------------------------------------------------------------------
# Integration: a real gitleaks binary over a throwaway repository
# ---------------------------------------------------------------------------------------------------------------


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


@pytest.mark.integration
def test_gitleaks_reports_keys_next_to_allowlisted_values_and_skips_the_values(tmp_path: Path) -> None:
    """A key on a key_facts digest's line, or a p_ key in the path panel, is reported; the known values are not."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    one_line_key = synthetic_key("one-line results JSON")
    panel_key = "p_live_" + synthetic_key("path panel")
    panel = "frontend/components/studio/project/path-panel.tsx"
    files = {
        "results/digests.json": json.dumps({"key_facts_sha256": DIGEST}, indent=2) + "\n",
        "results/one_line.json": json.dumps({"key_facts_sha256": DIGEST, "service_api_key": one_line_key}) + "\n",
        panel: f'const rows = [{{ key: "{OUTCOME_FIELD}" }}];\nconst token = "{panel_key}";\n',
    }
    commit_files(repo, files, "SYNTHETIC files")

    findings = run_gitleaks(repo, tmp_path / "report.json")

    reported = {(finding["File"], finding["Secret"]) for finding in findings}
    assert reported == {("results/one_line.json", one_line_key), (panel, panel_key)}
