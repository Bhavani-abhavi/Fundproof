"""Load the OFAC SDN list and OpenSanctions into one normalized watchlist table.

Sources (downloaded by scripts/download_data.py into data/raw/, never committed):
- OFAC SDN: sdn.csv (one row per listed entry, no header) and alt.csv (aliases, no header). US government data.
- OpenSanctions "default" collection, simplified CSV (targets.simple.csv). Data from OpenSanctions.org,
  licensed CC BY-NC 4.0: free for non-commercial use with attribution. This project is a non-commercial
  demonstration; anyone reusing it commercially needs an OpenSanctions data license.

The output table has one row per list entry with these columns:
    list_entry_id  "OFAC-<ent_num>" or "OS-<id>", unique across sources
    name           the primary name as the list writes it
    aliases        tuple of other names (OFAC alt.csv, OpenSanctions aliases column)
    entity_type    person / organization / vessel / aircraft / other
    countries      tuple of normalized country codes (see countries.py); empty when unknown
    programs       tuple of sanction program ids (OFAC Program column, OpenSanctions program_ids)
    source         "OFAC" or "OpenSanctions"
    source_id      the id inside the source file (OFAC ent_num, OpenSanctions id)
    list_name      a readable list label for analysts, e.g. "OFAC SDN (SDGT)"

Every screening result carries a ListSnapshot per file: its SHA-256 checksum and the date it was first
recorded in data/raw/checksums.jsonl, so an alert can always be traced to the exact list version.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from uwcopilot.common.files import sha256_of_file
from uwcopilot.common.paths import RAW_DATA_DIR
from uwcopilot.kyc.countries import normalize_countries
from uwcopilot.tools.schemas import ListSnapshot

OFAC_SDN_COLUMNS: list[str] = [
    "ent_num",
    "SDN_Name",
    "SDN_Type",
    "Program",
    "Title",
    "Call_Sign",
    "Vess_type",
    "Tonnage",
    "GRT",
    "Vess_flag",
    "Vess_owner",
    "Remarks",
]
OFAC_ALT_COLUMNS: list[str] = ["ent_num", "alt_num", "alt_type", "alt_name", "alt_remarks"]
OFAC_NULL = "-0-"
# Only these OpenSanctions columns are read; the file has more (addresses, identifiers, ...).
OPENSANCTIONS_COLUMNS: list[str] = ["id", "schema", "name", "aliases", "countries", "program_ids", "dataset"]
OPENSANCTIONS_ATTRIBUTION = (
    "Data from OpenSanctions.org, licensed CC BY-NC 4.0 (https://www.opensanctions.org/licensing/). "
    "Non-commercial use only without a separate data license."
)
WATCHLIST_COLUMNS: list[str] = [
    "list_entry_id",
    "name",
    "aliases",
    "entity_type",
    "countries",
    "programs",
    "source",
    "source_id",
    "list_name",
]
ENTITY_TYPES: tuple[str, ...] = ("person", "organization", "vessel", "aircraft", "other")
# OFAC leaves SDN_Type empty ("-0-") for entities, which are organizations.
OFAC_TYPE_MAP: dict[str, str] = {"individual": "person", "vessel": "vessel", "aircraft": "aircraft", "": "organization"}
# FollowTheMoney schemas used by OpenSanctions. LegalEntity can be a person or a company, so it maps to "other"
# (type unknown); Security, CryptoWallet, Asset, Address and Position are not people or companies either.
OPENSANCTIONS_TYPE_MAP: dict[str, str] = {
    "Person": "person",
    "Organization": "organization",
    "Company": "organization",
    "PublicBody": "organization",
    "Vessel": "vessel",
    "Airplane": "aircraft",
}
# "nationality Iran", "citizen Iraq", "Nationality of Registration Panama" inside OFAC Remarks.
OFAC_COUNTRY_PATTERN = re.compile(r"\b(?:nationality|citizen|Nationality of Registration)\s+([^;]+)")
MAX_DATASETS_IN_LIST_NAME = 3

DEFAULT_OFAC_SDN = RAW_DATA_DIR / "ofac" / "sdn.csv"
DEFAULT_OFAC_ALT = RAW_DATA_DIR / "ofac" / "alt.csv"
DEFAULT_OPENSANCTIONS = RAW_DATA_DIR / "opensanctions" / "targets.simple.csv"
DEFAULT_CHECKSUM_LOG = RAW_DATA_DIR / "checksums.jsonl"


@dataclass
class Watchlist:
    """The combined watchlist table, the file versions it came from, and what was dropped while loading."""

    entries: pd.DataFrame
    snapshots: list[ListSnapshot]
    load_report: dict = field(default_factory=dict)


def clean_ofac_value(value: str) -> str:
    """Strip whitespace and turn the OFAC null marker "-0-" into an empty string."""
    text = (value or "").strip()
    return "" if text == OFAC_NULL else text


def split_ofac_programs(program: str) -> tuple[str, ...]:
    """OFAC writes several programs as "SDGT] [IRGC"; return ("SDGT", "IRGC")."""
    parts = [part.strip(" []") for part in clean_ofac_value(program).split("] [")]
    return tuple(part for part in parts if part)


def ofac_countries(remarks: str, vessel_flag: str) -> tuple[str, ...]:
    """Countries named in OFAC Remarks (nationality, citizen, registration) plus the vessel flag."""
    names = [match.strip().rstrip(".").strip() for match in OFAC_COUNTRY_PATTERN.findall(clean_ofac_value(remarks))]
    flag = clean_ofac_value(vessel_flag)
    if flag:
        names.append(flag)
    return normalize_countries(names)


def split_semicolon(value: str) -> tuple[str, ...]:
    """Split an OpenSanctions multi-value cell ("a;b;c"), dropping empty parts and duplicates."""
    parts: list[str] = []
    for part in (value or "").split(";"):
        part = part.strip()
        if part and part not in parts:
            parts.append(part)
    return tuple(parts)


def opensanctions_list_name(datasets: tuple[str, ...]) -> str:
    """A short label such as "OpenSanctions (EU Sanctions Map; US OFAC SDN, +2 more)"."""
    if not datasets:
        return "OpenSanctions"
    shown = "; ".join(datasets[:MAX_DATASETS_IN_LIST_NAME])
    extra = len(datasets) - MAX_DATASETS_IN_LIST_NAME
    suffix = f", +{extra} more" if extra > 0 else ""
    return f"OpenSanctions ({shown}{suffix})"


def read_ofac_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a header-less OFAC CSV as strings. Rows whose first column is not a number are dropped."""
    frame = pd.read_csv(
        path,
        header=None,
        names=columns,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
        encoding_errors="replace",
    )
    frame["ent_num"] = frame["ent_num"].str.strip()
    return frame[frame["ent_num"].str.fullmatch(r"\d+")].reset_index(drop=True)


def load_ofac(sdn_path: Path, alt_path: Path) -> tuple[pd.DataFrame, dict]:
    """Load sdn.csv and alt.csv into the watchlist table format. Returns (table, load counts)."""
    sdn = read_ofac_csv(sdn_path, OFAC_SDN_COLUMNS)
    alt = read_ofac_csv(alt_path, OFAC_ALT_COLUMNS)
    aliases_by_entry: dict[str, list[str]] = {}
    for ent_num, alt_name in zip(alt["ent_num"], alt["alt_name"], strict=True):
        alias = clean_ofac_value(alt_name)
        if alias:
            aliases_by_entry.setdefault(ent_num, []).append(alias)

    rows: list[dict] = []
    dropped_empty_name = 0
    for record in sdn.itertuples(index=False):
        name = clean_ofac_value(record.SDN_Name)
        if not name:
            dropped_empty_name += 1
            continue
        programs = split_ofac_programs(record.Program)
        aliases = tuple(alias for alias in aliases_by_entry.get(record.ent_num, []) if alias != name)
        rows.append(
            {
                "list_entry_id": f"OFAC-{record.ent_num}",
                "name": name,
                "aliases": aliases,
                "entity_type": OFAC_TYPE_MAP.get(clean_ofac_value(record.SDN_Type).lower(), "other"),
                "countries": ofac_countries(record.Remarks, record.Vess_flag),
                "programs": programs,
                "source": "OFAC",
                "source_id": record.ent_num,
                "list_name": f"OFAC SDN ({', '.join(programs)})" if programs else "OFAC SDN",
            }
        )
    report = {
        "sdn_rows_read": int(len(sdn)),
        "alt_rows_read": int(len(alt)),
        "entries_kept": len(rows),
        "entries_dropped_empty_name": dropped_empty_name,
        "aliases_kept": int(sum(len(row["aliases"]) for row in rows)),
    }
    return pd.DataFrame(rows, columns=WATCHLIST_COLUMNS), report


def load_opensanctions(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load the OpenSanctions simplified CSV (only the needed columns) into the watchlist table format."""
    frame = pd.read_csv(path, usecols=OPENSANCTIONS_COLUMNS, dtype=str, keep_default_na=False)
    rows: list[dict] = []
    dropped_empty_name = 0
    schema_counts: dict[str, int] = {}
    for entity_id, schema, name, aliases, countries, program_ids, datasets in zip(
        frame["id"],
        frame["schema"],
        frame["name"],
        frame["aliases"],
        frame["countries"],
        frame["program_ids"],
        frame["dataset"],
        strict=True,
    ):
        name = name.strip()
        if not name:
            dropped_empty_name += 1
            continue
        schema_counts[schema] = schema_counts.get(schema, 0) + 1
        dataset_titles = split_semicolon(datasets)
        rows.append(
            {
                "list_entry_id": f"OS-{entity_id}",
                "name": name,
                "aliases": tuple(alias for alias in split_semicolon(aliases) if alias != name),
                "entity_type": OPENSANCTIONS_TYPE_MAP.get(schema, "other"),
                "countries": normalize_countries(split_semicolon(countries)),
                "programs": split_semicolon(program_ids),
                "source": "OpenSanctions",
                "source_id": entity_id,
                "list_name": opensanctions_list_name(dataset_titles),
            }
        )
    report = {
        "rows_read": int(len(frame)),
        "entries_kept": len(rows),
        "entries_dropped_empty_name": dropped_empty_name,
        "aliases_kept": int(sum(len(row["aliases"]) for row in rows)),
        "schema_counts": dict(sorted(schema_counts.items(), key=lambda item: -item[1])),
        "attribution": OPENSANCTIONS_ATTRIBUTION,
    }
    return pd.DataFrame(rows, columns=WATCHLIST_COLUMNS), report


def first_recorded_date(checksum_log: Path, file_name: str, sha256: str) -> str | None:
    """Date (YYYY-MM-DD) of the earliest checksum-log line for this file name and checksum, or None."""
    if not checksum_log.exists():
        return None
    dates: list[str] = []
    for line in checksum_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("file") == file_name and record.get("sha256") == sha256:
            dates.append(str(record.get("hashed_at_utc", ""))[:10])
    dates = [date for date in dates if date]
    return min(dates) if dates else None


def snapshot_for(path: Path, source: str, checksum_log: Path) -> ListSnapshot:
    """Hash a list file and look up when that exact version was first recorded."""
    sha256 = sha256_of_file(path)
    return ListSnapshot(
        source=source,
        file_name=path.name,
        sha256=sha256,
        snapshot_date=first_recorded_date(checksum_log, path.name, sha256),
        attribution=OPENSANCTIONS_ATTRIBUTION if source == "OpenSanctions" else None,
    )


def load_watchlist(
    ofac_sdn: Path | None = DEFAULT_OFAC_SDN,
    ofac_alt: Path | None = DEFAULT_OFAC_ALT,
    opensanctions: Path | None = DEFAULT_OPENSANCTIONS,
    checksum_log: Path = DEFAULT_CHECKSUM_LOG,
) -> Watchlist:
    """Load every available source into one table. Pass None for a source to leave it out.

    The load report says which sources were used and how many rows each contributed or dropped, so a result
    file can state its scope exactly.
    """
    tables: list[pd.DataFrame] = []
    snapshots: list[ListSnapshot] = []
    report: dict = {"sources_used": [], "sources_skipped": []}
    if ofac_sdn is not None and ofac_alt is not None:
        table, ofac_report = load_ofac(ofac_sdn, ofac_alt)
        tables.append(table)
        snapshots.append(snapshot_for(ofac_sdn, "OFAC", checksum_log))
        snapshots.append(snapshot_for(ofac_alt, "OFAC", checksum_log))
        report["sources_used"].append("OFAC")
        report["OFAC"] = ofac_report
    else:
        report["sources_skipped"].append("OFAC")
    if opensanctions is not None:
        table, os_report = load_opensanctions(opensanctions)
        tables.append(table)
        snapshots.append(snapshot_for(opensanctions, "OpenSanctions", checksum_log))
        report["sources_used"].append("OpenSanctions")
        report["OpenSanctions"] = os_report
    else:
        report["sources_skipped"].append("OpenSanctions")
    if not tables:
        raise ValueError("load_watchlist needs at least one source.")
    entries = pd.concat(tables, ignore_index=True)
    if entries["list_entry_id"].duplicated().any():
        raise ValueError("list_entry_id values must be unique across sources.")
    report["total_entries"] = int(len(entries))
    report["entity_type_counts"] = {key: int(value) for key, value in entries["entity_type"].value_counts().items()}
    return Watchlist(entries=entries, snapshots=snapshots, load_report=report)


def watchlist_from_records(records: list[dict], snapshots: list[ListSnapshot]) -> Watchlist:
    """Build a Watchlist from plain dictionaries (used by tests and small demos; no files needed)."""
    rows = []
    for record in records:
        row = {column: record.get(column) for column in WATCHLIST_COLUMNS}
        row["aliases"] = tuple(record.get("aliases", ()))
        row["countries"] = normalize_countries(list(record.get("countries", ())))
        row["programs"] = tuple(record.get("programs", ()))
        row["source_id"] = record.get("source_id", row["list_entry_id"])
        row["list_name"] = record.get("list_name", row["source"])
        rows.append(row)
    entries = pd.DataFrame(rows, columns=WATCHLIST_COLUMNS)
    return Watchlist(entries=entries, snapshots=snapshots, load_report={"total_entries": len(rows)})
