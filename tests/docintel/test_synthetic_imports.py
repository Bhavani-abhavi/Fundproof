"""The extraction pipeline never imports the SYNTHETIC generator or the ground truth (Phase 2 rule).

Only scripts/make_packets.py, scripts/eval_*.py and tests may import uwcopilot.docintel.synthetic or
uwcopilot.docintel.truth. Imports are read from the syntax tree, so a mention in a docstring does not count.
"""

import ast
from pathlib import Path

from uwcopilot.common.paths import REPO_ROOT

DOCINTEL = REPO_ROOT / "src" / "uwcopilot" / "docintel"
FORBIDDEN = ("uwcopilot.docintel.synthetic", "uwcopilot.docintel.truth")


def imported_modules(path: Path) -> set[str]:
    """Every module a file imports (absolute names; "from uwcopilot.docintel import truth" counts as the module)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def pipeline_files() -> list[Path]:
    """docintel modules outside synthetic/ and truth.py."""
    files = [path for path in DOCINTEL.rglob("*.py") if "synthetic" not in path.relative_to(DOCINTEL).parts]
    return [path for path in files if path.name != "truth.py"]


def test_pipeline_modules_do_not_import_generator_or_truth() -> None:
    files = pipeline_files()
    assert any(path.name == "levers.py" for path in files)
    offenders = []
    for path in files:
        for module in imported_modules(path):
            if module.startswith(FORBIDDEN):
                offenders.append(f"{path.relative_to(REPO_ROOT)} imports {module}")
    assert offenders == []


def test_the_scanner_sees_a_forbidden_import(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text('"""uwcopilot.docintel.truth in a docstring is fine."""\nfrom uwcopilot.docintel import truth\n')
    assert "uwcopilot.docintel.truth" in imported_modules(sample)
