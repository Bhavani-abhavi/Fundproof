"""Tests for uwcopilot.kyc.watchlist with tiny CSV files written in tmp_path (no data/raw)."""

import hashlib
import json
from pathlib import Path

import pytest

from uwcopilot.kyc.watchlist import (
    OPENSANCTIONS_ATTRIBUTION,
    first_recorded_date,
    load_ofac,
    load_opensanctions,
    load_watchlist,
    ofac_countries,
    split_ofac_programs,
)

SDN_ROWS = (
    '36,"AEROTEST AIRLINES",-0- ,"CUBA",-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- \n'
    '7001,"HASSAN, Mohammed Ali","individual","SDGT] [IRGC",-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,'
    '"DOB 1970; nationality Iraq; citizen Korea, North."\n'
    '7002,"SEA TESTER","vessel","IRAN",-0- ,-0- ,"Crude Oil Tanker",-0- ,-0- ,"Panama",-0- ,-0- \n'
)
ALT_ROWS = '36,1,"aka","AERO-TEST",-0- \n7001,2,"aka","Muhammad Ali HASAN",-0- \n'
OS_HEADER = (
    '"id","schema","name","aliases","birth_date","countries","addresses","identifiers","sanctions","phones",'
    '"emails","program_ids","dataset","first_seen","last_seen","last_change"\n'
)
OS_ROWS = (
    '"NK-1","Person","Ivan Testov","Иван Тестов;Ivan Testov","","ru;by","","","","","","EU-RUS",'
    '"EU Sanctions Map;Ukraine War and Sanctions","","",""\n'
    '"NK-2","LegalEntity","Test Holding","","","","","","","","","","Some Registry","","",""\n'
    '"NK-3","CryptoWallet","0xabc","","","","","","","","","","Wallets","","",""\n'
)


@pytest.fixture
def list_files(tmp_path: Path) -> dict[str, Path]:
    """Write tiny OFAC and OpenSanctions files plus a checksum log that knows the SDN file."""
    sdn = tmp_path / "sdn.csv"
    alt = tmp_path / "alt.csv"
    opensanctions = tmp_path / "targets.simple.csv"
    sdn.write_text(SDN_ROWS, encoding="utf-8")
    alt.write_text(ALT_ROWS, encoding="utf-8")
    opensanctions.write_text(OS_HEADER + OS_ROWS, encoding="utf-8")
    log = tmp_path / "checksums.jsonl"
    sdn_hash = hashlib.sha256(sdn.read_bytes()).hexdigest()
    lines = [
        {"file": "sdn.csv", "sha256": sdn_hash, "hashed_at_utc": "2026-10-11T08:00:00+00:00"},
        {"file": "sdn.csv", "sha256": sdn_hash, "hashed_at_utc": "2026-10-10T07:00:00+00:00"},
        {"file": "sdn.csv", "sha256": "0" * 64, "hashed_at_utc": "2020-01-01T00:00:00+00:00"},
    ]
    log.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return {"sdn": sdn, "alt": alt, "opensanctions": opensanctions, "log": log}


def test_ofac_helpers() -> None:
    assert split_ofac_programs("SDGT] [IRGC") == ("SDGT", "IRGC")
    assert split_ofac_programs("-0- ") == ()
    assert ofac_countries("nationality Iraq; citizen Korea, North.", "-0-") == ("iq", "kp")
    assert ofac_countries("-0-", "Panama") == ("pa",)


def test_load_ofac_joins_aliases_and_maps_types(list_files: dict[str, Path]) -> None:
    table, report = load_ofac(list_files["sdn"], list_files["alt"])
    rows = table.set_index("list_entry_id")
    assert list(rows.index) == ["OFAC-36", "OFAC-7001", "OFAC-7002"]
    assert rows.loc["OFAC-36", "entity_type"] == "organization"  # OFAC leaves entity type empty
    assert rows.loc["OFAC-36", "aliases"] == ("AERO-TEST",)
    assert rows.loc["OFAC-7001", "entity_type"] == "person"
    assert rows.loc["OFAC-7001", "programs"] == ("SDGT", "IRGC")
    assert rows.loc["OFAC-7001", "countries"] == ("iq", "kp")
    assert rows.loc["OFAC-7002", "entity_type"] == "vessel"
    assert rows.loc["OFAC-7002", "list_name"] == "OFAC SDN (IRAN)"
    assert report["entries_kept"] == 3 and report["aliases_kept"] == 2


def test_load_opensanctions_reads_only_needed_columns(list_files: dict[str, Path]) -> None:
    table, report = load_opensanctions(list_files["opensanctions"])
    rows = table.set_index("list_entry_id")
    assert rows.loc["OS-NK-1", "entity_type"] == "person"
    assert rows.loc["OS-NK-1", "aliases"] == ("Иван Тестов",)  # the alias equal to the name is dropped
    assert rows.loc["OS-NK-1", "countries"] == ("ru", "by")
    assert rows.loc["OS-NK-1", "list_name"] == "OpenSanctions (EU Sanctions Map; Ukraine War and Sanctions)"
    assert rows.loc["OS-NK-2", "entity_type"] == "other"  # LegalEntity: person or company, unknown
    assert rows.loc["OS-NK-3", "entity_type"] == "other"
    assert report["attribution"] == OPENSANCTIONS_ATTRIBUTION
    assert report["schema_counts"] == {"Person": 1, "LegalEntity": 1, "CryptoWallet": 1}


def test_snapshot_date_is_the_first_log_line_for_that_exact_checksum(list_files: dict[str, Path]) -> None:
    sdn_hash = hashlib.sha256(list_files["sdn"].read_bytes()).hexdigest()
    assert first_recorded_date(list_files["log"], "sdn.csv", sdn_hash) == "2026-10-10"
    assert first_recorded_date(list_files["log"], "alt.csv", sdn_hash) is None
    assert first_recorded_date(list_files["log"].with_name("missing.jsonl"), "sdn.csv", sdn_hash) is None


def test_load_watchlist_combines_sources_and_records_snapshots(list_files: dict[str, Path]) -> None:
    watchlist = load_watchlist(list_files["sdn"], list_files["alt"], list_files["opensanctions"], list_files["log"])
    assert len(watchlist.entries) == 6
    assert watchlist.entries["list_entry_id"].is_unique
    assert [snapshot.file_name for snapshot in watchlist.snapshots] == ["sdn.csv", "alt.csv", "targets.simple.csv"]
    assert watchlist.snapshots[0].snapshot_date == "2026-10-10"
    assert watchlist.snapshots[1].snapshot_date is None  # alt.csv is not in the log: date unknown, not invented
    assert watchlist.load_report["sources_used"] == ["OFAC", "OpenSanctions"]


def test_a_source_can_be_left_out_but_not_all(list_files: dict[str, Path]) -> None:
    watchlist = load_watchlist(None, None, list_files["opensanctions"], list_files["log"])
    assert watchlist.load_report["sources_skipped"] == ["OFAC"]
    with pytest.raises(ValueError):
        load_watchlist(None, None, None, list_files["log"])
