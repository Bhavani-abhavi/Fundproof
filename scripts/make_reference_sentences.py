"""Write eval/provenance/reference_sentences.jsonl: every numeric guideline sentence and table row, cited.

Usage: uv run python scripts/make_reference_sentences.py
The guidelines are SYNTHETIC (corpus/guidelines/). See uwcopilot.provenance.reference for the rules.
"""

import argparse
import json
from pathlib import Path

from uwcopilot.common.paths import REPO_ROOT
from uwcopilot.provenance.reference import build_reference_sentences

DEFAULT_OUT = REPO_ROOT / "eval" / "provenance" / "reference_sentences.jsonl"


def main() -> None:
    """Build the reference sentences and write them as JSON lines."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    sentences = build_reference_sentences()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for sentence in sentences:
            handle.write(json.dumps(sentence.to_json(), ensure_ascii=False) + "\n")
    print(f"wrote {len(sentences)} reference sentences to {args.out}")


if __name__ == "__main__":
    main()
