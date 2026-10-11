"""Make the SYNTHETIC KYC screening customers and their ground truth (Phase 3).

Needs the OFAC and OpenSanctions files in data/raw/ (run `make data` first). Writes, under data/processed/kyc/
(gitignored, never committed):
    customers.csv           one row per SYNTHETIC customer: name, entity type, country, label
                            (clean / true_match / hard_negative), split (train / validation / test), and for
                            true matches the list entry the name was derived from
    homonyms.csv            the dropped exact-name collisions, one row per distinct name: SYNTHETIC customers who
                            share a listed name (label exact_name_homonym, no split); scored as a separate stratum
    generation_report.json  counts, dropped collisions, seed, and the checksums of the list files used

Usage:
    uv run python scripts/make_kyc_customers.py            # seed 42
    uv run python scripts/make_kyc_customers.py --seed 7
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from uwcopilot.common.files import sha256_of_file
from uwcopilot.kyc.synthetic import (
    CUSTOMERS_FILE,
    GENERATION_REPORT_FILE,
    HOMONYMS_FILE,
    LABELS,
    SPLITS,
    GenerationConfig,
    generate_customers,
    list_collision_keys,
)
from uwcopilot.kyc.watchlist import DEFAULT_OFAC_SDN, DEFAULT_OPENSANCTIONS, load_watchlist


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options: the seed and where to write."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    parser.add_argument("--customers-file", type=Path, default=CUSTOMERS_FILE)
    parser.add_argument("--report-file", type=Path, default=GENERATION_REPORT_FILE)
    parser.add_argument("--homonyms-file", type=Path, default=HOMONYMS_FILE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load the watchlist, generate the customers, write the CSV and the report."""
    args = parse_args(argv)
    for required in (DEFAULT_OFAC_SDN, DEFAULT_OPENSANCTIONS):
        if not required.exists():
            print(f"Missing {required}. Run `make data` first.", file=sys.stderr)
            return 1
    start = time.perf_counter()
    watchlist = load_watchlist()
    print(f"Loaded {len(watchlist.entries):,} list entries in {time.perf_counter() - start:.0f}s")
    list_keys = list_collision_keys(watchlist.entries)
    print(f"Built {len(list_keys):,} normalized list-name keys for the collision check")

    homonyms: list[dict] = []
    customers, report = generate_customers(
        watchlist.entries, list_keys, GenerationConfig(seed=args.seed), homonyms_out=homonyms
    )
    args.customers_file.parent.mkdir(parents=True, exist_ok=True)
    customers.to_csv(args.customers_file, index=False)
    pd.DataFrame(homonyms).to_csv(args.homonyms_file, index=False)
    report["list_snapshots"] = [snapshot.model_dump() for snapshot in watchlist.snapshots]
    report["watchlist_load_report"] = watchlist.load_report
    report["customers_file"] = {"path": str(args.customers_file.name), "sha256": sha256_of_file(args.customers_file)}
    report["homonyms_file"] = {
        "path": str(args.homonyms_file.name),
        "sha256": sha256_of_file(args.homonyms_file),
        "rows": len(homonyms),
    }
    args.report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(customers):,} SYNTHETIC customers to {args.customers_file}")
    for label in LABELS:
        by_split = ", ".join(f"{split} {report['counts_by_label_and_split'][label][split]}" for split in SPLITS)
        print(f"  {label}: {by_split}")
    clean = report["clean"]
    print(
        f"  clean names generated {clean['generated']:,}; dropped {clean['dropped_list_collisions']:,} exact list "
        f"collisions and {clean['dropped_duplicates']:,} duplicates"
    )
    print(f"  distinct exact-name homonyms kept apart in {args.homonyms_file.name}: {len(homonyms):,}")
    print(f"  hard negatives dropped for list collisions: {report['hard_negative']['dropped_list_collisions']}")
    print(f"Report: {args.report_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
