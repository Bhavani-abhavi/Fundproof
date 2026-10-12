"""Write eval/provenance/mutations.jsonl: seeded mutations and controls of the reference sentences.

Usage: uv run python scripts/make_mutations.py [--seed 42]
Reads eval/provenance/reference_sentences.jsonl (make it first with scripts/make_reference_sentences.py).
"""

import argparse
import json
from pathlib import Path

from uwcopilot.common.paths import REPO_ROOT
from uwcopilot.provenance.mutations import build_mutations
from uwcopilot.provenance.reference import ReferenceSentence, section_texts

EVAL_DIR = REPO_ROOT / "eval" / "provenance"


def load_references(path: Path) -> list[ReferenceSentence]:
    """Read reference sentences from JSON lines."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ReferenceSentence(**json.loads(line)) for line in lines if line.strip()]


def main() -> None:
    """Build the mutations from the committed reference sentences and write them."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--references", type=Path, default=EVAL_DIR / "reference_sentences.jsonl")
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "mutations.jsonl")
    args = parser.parse_args()
    mutations = build_mutations(load_references(args.references), section_texts(), seed=args.seed)
    with args.out.open("w", encoding="utf-8") as handle:
        for mutation in mutations:
            handle.write(json.dumps(mutation.to_json(), ensure_ascii=False) + "\n")
    print(f"wrote {len(mutations)} mutations and controls to {args.out} (seed {args.seed})")


if __name__ == "__main__":
    main()
