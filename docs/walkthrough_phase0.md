# Phase 0 walkthrough: repository setup

Phase 0 builds the skeleton the later phases plug into: folder layout, ground rules, data download, lint, tests,
and CI. It has no models and no metrics yet.

## Files

| File | What it does | Why it is built this way |
|---|---|---|
| `CLAUDE.md` | The project's ground rules, word for word. | Every working session reads it first, so the rules (honest metrics, public or synthetic data only, reproducibility, no secrets) apply to everyone who works on the repo. |
| `pyproject.toml` | Package metadata, dependencies, and ruff/pytest settings. | One file for all project configuration. `requires-python` is pinned to 3.11 so every machine runs the same interpreter. |
| `uv.lock` | Exact versions of every dependency. | `uv sync --locked` installs the same versions on a laptop and in CI, so a result does not change because a library updated. |
| `.python-version` | Tells uv to use Python 3.11. | Saves passing the version on every command. |
| `Makefile` | One target per step: `setup`, `lint`, `format`, `test`, `data`, `all`, `clean`. | A reviewer can run any step without reading the code first. `make help` lists the targets. |
| `.gitignore` | Keeps `data/`, `.env`, virtualenvs, caches, and MLflow runs out of git. | Raw datasets are large and some have license terms, so they are downloaded by script and never committed. `.env` holds secrets. |
| `.env.example` | Template for the secrets file (Kaggle key, LLM key, cost cap). | Shows which variables exist without exposing real values. The Kaggle lines are commented out because the kaggle package lets even an empty `KAGGLE_*` variable override a `kaggle.json` file. |
| `src/uwcopilot/` | The Python package, with one subpackage per component: `risk`, `docintel`, `kyc`, `evidence`, `agent`, `governance`, `common`. | The `src/` layout means code can only import `uwcopilot` once it is installed (`uv sync` installs it in editable mode), so tests run against the package as installed and a missing package file shows up as an import error. |
| `src/uwcopilot/common/paths.py` | Constants for the repo root, `data/`, `data/raw/`, `results/`, `docs/`. The `UWCOPILOT_ROOT` environment variable overrides the root. | Every script finds folders the same way and does not depend on the working directory. The override is for a container where the package is installed without the repo around it. |
| `src/uwcopilot/common/files.py` | `sha256_of_file`, `human_size`, `append_checksum_log`. | Small helpers with one job each, reused by every download step. Each log line records whether the file was `downloaded` or `verified_existing` in that run, and when it was hashed. |
| `scripts/download_data.py` | Downloads Lending Club (Kaggle), freMTPL2 (OpenML), German Credit (UCI), OFAC SDN with aliases, and OpenSanctions. Prints each file's size and appends its SHA-256 to `data/raw/checksums.jsonl`. | One function per dataset, plus a dictionary mapping names to functions, so `--datasets` can pick a subset. Existing files are re-hashed, not re-downloaded, unless `--force` is passed. Downloads go to a `.part` file that is renamed only when complete, so an interrupted download never looks finished; for Lending Club the zip must also open as a valid zip with no Kaggle resume marker next to it. If Kaggle credentials are missing or invalid, Lending Club is skipped with instructions and the script exits non-zero so the gap is visible. For freMTPL2 both the original OpenML file and the CSV made from it are hashed, because the CSV's hash can change with a pandas upgrade even when the data did not. |
| `tests/` | `test_smoke.py` (the package imports), `test_common_files.py` (hashing, sizes, checksum log), `test_common_paths.py` (folder locations, root override), `test_download_data.py` (argument parsing, the Kaggle skip, empty Kaggle variables, partial-zip detection, skip-if-present). | None of the tests touch the network, so CI is fast and does not fail when a data host is down. |
| `.github/workflows/ci.yml` | On every push and pull request: install uv and Python 3.11, install dependencies from the lockfile, then ruff lint, ruff format check, and pytest. | `uv sync --locked` (and `UV_LOCKED=1` for every later `uv run`) makes CI fail if `uv.lock` is out of date with `pyproject.toml`, so the lockfile cannot silently drift. `--frozen` would not catch that: it installs from the lockfile without checking it. The workflow only has read access to the repo. |
| `README.md` | Problem, architecture diagram, datasets and licenses, how to run, results, limitations, synthetic-data notice. | The Results table sits between `RESULTS:START` and `RESULTS:END` markers so a script can rewrite it from `results/*.json`. Until then every row says "not measured". |
| `results/`, `notebooks/`, `data/` | Empty folders, kept in git with `.gitkeep`. | `results/` will hold the JSON files that every reported number comes from. `notebooks/` is for exploration only, never for reported numbers. |

## Design choices

- **uv instead of pip-tools.** Both give a lockfile. uv is much faster and manages the Python version too.
- **Checksums for downloads.** If a public dataset changes upstream, the checksum log shows it, which explains a
  metric that moved for no other reason.
- **Fail visibly, keep going.** If one dataset fails, the script downloads the others and lists what is missing
  at the end, with a non-zero exit code.
- **The Kaggle import happens inside the function.** Importing `kaggle` runs an authentication attempt as a side
  effect: with no key it prints a long login banner and creates a config folder. Importing it only when Lending
  Club is requested keeps those side effects away from people who only want the other datasets. The script
  captures the banner and prints one short message of its own.

## Likely interview questions

1. **Why commit a lockfile?**
   Without one, `pip install` picks up whatever versions are newest that day, and a model can give different
   numbers a month later. The lockfile pins every package, including indirect dependencies.

2. **Why not commit the datasets?**
   They are large (OpenSanctions alone is several hundred MB), some licenses restrict redistribution, and git is
   bad at large binary files. A download script with checksums reproduces the data just as well.

3. **What does the SHA-256 checksum give you?**
   Proof that the file you trained on is byte-for-byte the file in the log. If a rerun gives different numbers,
   comparing checksums tells you whether the data changed.

4. **Why the `src/` layout?**
   The package is not importable from the repo folder by accident; it has to be installed. Tests therefore run
   against the installed package, so a packaging mistake (a missing file, a wrong package path) fails in CI
   instead of in production.

5. **What does CI check, and why on every pull request?**
   Lint (ruff), formatting, and tests. Running them on every PR means a broken change is caught before it reaches
   `main`, and reviewers only look at code that already passes.

6. **Why don't the tests download data?**
   Network tests are slow and fail whenever a third-party server is down, which teaches people to ignore red
   CI. The download logic is tested with temporary files instead.

7. **How are secrets handled?**
   They live in `.env`, which is gitignored. `.env.example` lists the variable names with empty values. Code reads
   them with `python-dotenv`.

8. **What is the OpenSanctions license, and what does it mean here?**
   CC BY-NC 4.0: free for non-commercial use with attribution. This is a non-commercial portfolio project, and the
   README credits OpenSanctions. A commercial product would need a commercial license.

9. **How do you know the lockfile matches `pyproject.toml`?**
   CI runs `uv sync --locked`, which fails if the lockfile is stale. `--frozen` would quietly install the old
   lockfile, so it was replaced after a review caught the difference.

10. **Why does the README say "not measured" instead of leaving the table out?**
   It shows the planned evaluation up front, and it makes it obvious which numbers are real. A number only
   appears once a script has produced it.
