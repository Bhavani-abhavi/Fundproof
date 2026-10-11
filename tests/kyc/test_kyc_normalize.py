"""Tests for uwcopilot.kyc.normalize: Unicode folding, punctuation, honorifics and legal forms."""

from uwcopilot.kyc.normalize import basic_tokens, is_initial, is_latin_token, normalize_name, strip_accents


def test_accents_case_and_apostrophes_are_folded() -> None:
    assert strip_accents("José Müller") == "Jose Muller"
    assert basic_tokens("O'Brien-SMITH, José") == ["obrien", "smith", "jose"]
    assert basic_tokens("Straße") == ["strasse"]  # casefold, not lower()


def test_other_scripts_are_kept_not_dropped() -> None:
    assert basic_tokens("Иван Сидоренко") == ["иван", "сидоренко"]


def test_person_honorifics_and_suffixes_are_stripped_from_the_ends_only() -> None:
    result = normalize_name("Dr. John A. Smith Jr.", "person")
    assert result.tokens == ("john", "a", "smith")
    assert result.removed == ("dr", "jr")
    # A title in the middle is part of the name and stays.
    assert normalize_name("Smith Major Jones", "person").tokens == ("smith", "major", "jones")


def test_dotted_initials_stay_separate_for_people() -> None:
    assert normalize_name("J.R. Smith", "person").tokens == ("j", "r", "smith")


def test_organization_legal_forms_are_stripped() -> None:
    assert normalize_name("ANGLO-CARIBBEAN CO., LTD.", "organization").tokens == ("anglo", "caribbean")
    assert normalize_name("Acme Trading S.A.", "organization").tokens == ("acme", "trading")
    assert normalize_name("Smith Holdings L.L.C.", "organization").tokens == ("smith", "holdings")
    assert normalize_name("OOO Romashka", "organization").tokens == ("romashka",)
    assert normalize_name("Public Joint Stock Company Priboy", "organization").tokens == ("priboy",)


def test_honorifics_are_not_stripped_from_companies() -> None:
    assert normalize_name("General Dynamics Corp", "organization").tokens == ("general", "dynamics")


def test_a_name_is_never_stripped_to_nothing() -> None:
    assert normalize_name("Ltd", "organization").tokens == ("ltd",)
    assert normalize_name("Dr.", "person").tokens == ("dr",)


def test_sorted_key_ignores_token_order() -> None:
    assert normalize_name("Hassan Ali", "person").sorted_key == normalize_name("Ali Hassan", "person").sorted_key


def test_token_helpers() -> None:
    assert is_initial("j") and not is_initial("jo")
    assert is_latin_token("abc12") and not is_latin_token("иван")
