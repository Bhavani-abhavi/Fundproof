"""Tests for uwcopilot.kyc.countries."""

from uwcopilot.kyc.countries import normalize_countries, normalize_country


def test_codes_and_names_map_to_lowercase_iso_codes() -> None:
    assert normalize_country("RU") == "ru"
    assert normalize_country("Russia") == "ru"
    assert normalize_country("Korea, North") == "kp"
    assert normalize_country("UK") == "gb"


def test_unknown_values() -> None:
    assert normalize_country(None) is None
    assert normalize_country("  ") is None
    assert normalize_country("-0-") is None
    assert normalize_country("Atlantis") == "atlantis"  # kept as text, never guessed


def test_normalize_countries_drops_duplicates_and_blanks() -> None:
    assert normalize_countries(["Iran", "IR", "", "Iraq"]) == ("ir", "iq")
