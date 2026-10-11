"""Generate the SYNTHETIC agent evaluation applications into eval/applications/ (`make applications`).

Needs data/raw/ofac/sdn.csv and alt.csv and the OpenSanctions file (`make data`):
- a few applications get an organization owner whose name is a lightly varied copy of a real OFAC SDN organization
  entry, so sanctions screening has true matches to find. OFAC data is US government data in the public domain;
  only organization entries are used, never individuals;
- every made-up applicant and owner name is checked against all OFAC and OpenSanctions names and drawn again if it
  is exactly a listed name (as the KYC customer generator does). OpenSanctions data is CC BY-NC 4.0; it is only
  read here, nothing from it is written to the applications.

Writes one APP-*.json per application (fixed seed 42, so the files are the same on every run) and manifest.json
with counts, the OFAC file checksum and a checksum of every application file.

--keep-exact-name-homonyms switches the redraw off and writes to data/processed/agent/applications_homonyms_kept/
(gitignored) unless --out-dir is given. That set keeps made-up owners who share a listed person's or
organization's exact name, still labelled clean. It is the set the first full agent evaluation ran on, before the
redraw was added, and `make eval-agent-homonyms` evaluates it so the effect of the redraw is reported, not hidden.
"""

import argparse
import hashlib
import json
from pathlib import Path

from uwcopilot.agent.synthetic_apps import (
    AS_OF_DATE,
    DEFAULT_SEED,
    GENERATOR_VERSION,
    generate_applications,
    watchlist_pool,
)
from uwcopilot.common.files import sha256_of_file
from uwcopilot.common.paths import RAW_DATA_DIR, REPO_ROOT
from uwcopilot.kyc.synthetic import list_collision_keys
from uwcopilot.kyc.watchlist import OPENSANCTIONS_ATTRIBUTION, load_ofac, load_opensanctions

DEFAULT_OUT_DIR = REPO_ROOT / "eval" / "applications"
HOMONYMS_KEPT_OUT_DIR = REPO_ROOT / "data" / "processed" / "agent" / "applications_homonyms_kept"
DEFAULT_SDN = RAW_DATA_DIR / "ofac" / "sdn.csv"
DEFAULT_ALT = RAW_DATA_DIR / "ofac" / "alt.csv"
DEFAULT_OPENSANCTIONS = RAW_DATA_DIR / "opensanctions" / "targets.simple.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=None, help="default eval/applications/")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sdn", type=Path, default=DEFAULT_SDN)
    parser.add_argument("--alt", type=Path, default=DEFAULT_ALT)
    parser.add_argument("--opensanctions", type=Path, default=DEFAULT_OPENSANCTIONS)
    parser.add_argument(
        "--keep-exact-name-homonyms",
        action="store_true",
        help="do not redraw made-up names that exactly equal a listed name (sensitivity set; see the module doc)",
    )
    args = parser.parse_args(argv)
    if args.out_dir is None:
        args.out_dir = HOMONYMS_KEPT_OUT_DIR if args.keep_exact_name_homonyms else DEFAULT_OUT_DIR
    return args


def application_text(application: dict) -> str:
    """The JSON text written for one application (stable formatting)."""
    return json.dumps(application, indent=2, ensure_ascii=False) + "\n"


def build_manifest(
    applications: list[dict], seed: int, args: argparse.Namespace, file_hashes: dict[str, str], redrawn: int
) -> dict:
    """Counts and checksums describing one generated set."""
    truths = [application["ground_truth"] for application in applications]
    combined = hashlib.sha256("".join(f"{name}:{digest}\n" for name, digest in sorted(file_hashes.items())).encode())
    return {
        "data_label": "SYNTHETIC",
        "description": "SYNTHETIC loan and insurance applications with designed ground truth for the agent evaluation. "
        "They stand in for document extraction (Phase 2).",
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "as_of_date": AS_OF_DATE.isoformat(),
        "count": len(applications),
        "credit_count": sum(1 for application in applications if application["line"] == "credit"),
        "insurance_count": sum(1 for application in applications if application["line"] == "insurance"),
        "injection_count": sum(1 for truth in truths if truth["contains_injection"]),
        "watchlist_owner_count": sum(len(truth["watchlist_derived_owners"]) for truth in truths),
        "expected_recommendations": {
            name: sum(1 for truth in truths if truth["expected_recommendation"] == name)
            for name in ("approve", "refer", "decline_for_review", "request_more_information")
        },
        "watchlist_source": {
            "file": "data/raw/ofac/sdn.csv",
            "sha256": sha256_of_file(args.sdn),
            "entries_used": "OFAC SDN organization entries only (no individuals)",
            "attribution": "OFAC Specially Designated Nationals list, US Department of the Treasury (public domain)",
        },
        "clean_name_check": {
            "rule": "not applied: made-up names that exactly equal a listed name are kept and still labelled clean "
            "(sensitivity set, --keep-exact-name-homonyms)"
            if args.keep_exact_name_homonyms
            else "made-up applicant and owner names that exactly equal any OFAC or OpenSanctions name (after "
            "screening normalization) are drawn again",
            "names_redrawn": redrawn,
            "opensanctions_sha256": sha256_of_file(args.opensanctions),
            "opensanctions_attribution": OPENSANCTIONS_ATTRIBUTION,
        },
        "files": dict(sorted(file_hashes.items())),
        "applications_sha256": combined.hexdigest(),
    }


def write_applications(applications: list[dict], out_dir: Path) -> dict[str, str]:
    """Write every application, remove stale APP-*.json files, and return {file name: sha256}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {f"{application['application_id']}.json" for application in applications}
    for stale in out_dir.glob("APP-*.json"):
        if stale.name not in wanted:
            stale.unlink()
    hashes = {}
    for application in applications:
        text = application_text(application)
        name = f"{application['application_id']}.json"
        (out_dir / name).write_text(text, encoding="utf-8")
        hashes[name] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return hashes


def main(argv: list[str] | None = None) -> int:
    """Generate, write and summarize the application set."""
    args = parse_args(argv)
    entries, report = load_ofac(args.sdn, args.alt)
    other, _ = load_opensanctions(args.opensanctions)
    reserved = frozenset(list_collision_keys(entries) | list_collision_keys(other))
    if args.keep_exact_name_homonyms:
        reserved = frozenset()
    pool = watchlist_pool(entries)
    generation: dict = {}
    applications = generate_applications(pool, seed=args.seed, reserved_keys=reserved, report=generation)
    hashes = write_applications(applications, args.out_dir)
    redrawn = generation["clean_name_collisions_redrawn"]
    manifest = build_manifest(applications, args.seed, args, hashes, redrawn)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"OFAC entries read: {report['entries_kept']}; organization source names usable: {len(pool)}")
    print(f"Watchlist name keys checked: {len(reserved)}; made-up names drawn again after a collision: {redrawn}")
    print(
        f"Wrote {manifest['count']} SYNTHETIC applications to {args.out_dir} "
        f"({manifest['credit_count']} credit, {manifest['insurance_count']} insurance; "
        f"{manifest['injection_count']} with injected instructions, "
        f"{manifest['watchlist_owner_count']} watchlist-derived owners)."
    )
    print(f"Set checksum: {manifest['applications_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
