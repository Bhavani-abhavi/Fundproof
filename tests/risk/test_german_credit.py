"""Tests for uwcopilot.risk.data.german_credit, on SYNTHETIC rows in the german.data format (no real data)."""

from pathlib import Path

import pandas as pd
import pytest

from uwcopilot.risk.data.german_credit import (
    ATTRIBUTE_NAMES,
    CATEGORICAL_ATTRIBUTES,
    CATEGORY_CODES,
    EXCLUDED_FROM_FEATURES,
    MODEL_FEATURES,
    NUMERIC_ATTRIBUTES,
    SEX_BY_CODE_CORRECTED,
    SEX_BY_CODE_GERMAN_DOC,
    add_derived_columns,
    age_group,
    load_german_credit,
    parse_german_data,
    synthetic_german_data_text,
)

# The first row of german.data as published by UCI (codes and numbers only, no personal data).
FIRST_ROW = "A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 A192 A201 1"


def _row_with(personal_status: str = "A93", age: int = 67, target: int = 1) -> str:
    """FIRST_ROW with the personal status code, age and target replaced."""
    fields = FIRST_ROW.split()
    fields[8] = personal_status
    fields[12] = str(age)
    fields[20] = str(target)
    return " ".join(fields)


def test_attribute_lists_match_german_doc_counts() -> None:
    assert len(ATTRIBUTE_NAMES) == 20
    assert len(NUMERIC_ATTRIBUTES) == 7
    assert len(CATEGORICAL_ATTRIBUTES) == 13
    assert set(CATEGORY_CODES) == set(CATEGORICAL_ATTRIBUTES)


def test_parse_first_row_types_and_values() -> None:
    frame = parse_german_data(FIRST_ROW + "\n")
    assert frame.shape == (1, 21)
    row = frame.iloc[0]
    assert row["checking_account_status"] == "A11"
    assert row["duration_months"] == 6
    assert row["credit_amount"] == 1169
    assert row["age_years"] == 67
    assert row["target"] == 1
    assert frame["duration_months"].dtype == "int64"
    assert isinstance(frame["purpose"].dtype, pd.CategoricalDtype)
    # Categories are the full german.doc list, even codes absent from the data (so train/test agree).
    assert list(frame["purpose"].cat.categories) == list(CATEGORY_CODES["purpose"])


def test_parse_skips_blank_lines() -> None:
    frame = parse_german_data("\n" + FIRST_ROW + "\n\n" + FIRST_ROW + "\n")
    assert len(frame) == 2


def test_parse_rejects_wrong_field_count() -> None:
    with pytest.raises(ValueError, match="expected 21 fields"):
        parse_german_data(FIRST_ROW + " extra\n")


def test_parse_rejects_unknown_code() -> None:
    with pytest.raises(ValueError, match="unknown codes"):
        parse_german_data(FIRST_ROW.replace("A43", "A99") + "\n")


def test_parse_rejects_non_numeric_value() -> None:
    with pytest.raises(ValueError, match="non-numeric"):
        parse_german_data(FIRST_ROW.replace(" 1169 ", " abc ") + "\n")


def test_parse_rejects_bad_target() -> None:
    with pytest.raises(ValueError, match="target"):
        parse_german_data(_row_with(target=3) + "\n")


def test_parse_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="no data rows"):
        parse_german_data("\n\n")


def test_labels_bad_and_good() -> None:
    frame = add_derived_columns(parse_german_data(_row_with(target=1) + "\n" + _row_with(target=2) + "\n"))
    assert frame["bad"].tolist() == [0, 1]
    assert frame["good"].tolist() == [1, 0]


@pytest.mark.parametrize(
    ("code", "sex_doc", "sex_corrected"),
    [
        ("A91", "male", "male"),
        ("A92", "female", "undetermined"),
        ("A93", "male", "male"),
        ("A94", "male", "female"),
        ("A95", "female", "undetermined"),
    ],
)
def test_sex_mappings(code: str, sex_doc: str, sex_corrected: str) -> None:
    frame = add_derived_columns(parse_german_data(_row_with(personal_status=code) + "\n"))
    assert frame.loc[0, "sex"] == sex_doc
    assert frame.loc[0, "sex_corrected"] == sex_corrected


def test_sex_mappings_cover_every_code() -> None:
    codes = set(CATEGORY_CODES["personal_status_sex"])
    assert set(SEX_BY_CODE_GERMAN_DOC) == codes
    assert set(SEX_BY_CODE_CORRECTED) == codes


@pytest.mark.parametrize(
    ("age", "group_25", "group_30"),
    [(19, "<25", "<=30"), (24, "<25", "<=30"), (25, ">=25", "<=30"), (30, ">=25", "<=30"), (31, ">=25", ">30")],
)
def test_age_groups_at_the_boundaries(age: int, group_25: str, group_30: str) -> None:
    frame = add_derived_columns(parse_german_data(_row_with(age=age) + "\n"))
    assert frame.loc[0, "age_group_25"] == group_25
    assert frame.loc[0, "age_group_30"] == group_30


def test_age_group_helper_keeps_index() -> None:
    ages = pd.Series([20, 40], index=[7, 9])
    assert age_group(ages, 25, cutoff_is_upper_bound_of_young=False).to_dict() == {7: "<25", 9: ">=25"}


def test_protected_attributes_are_not_model_features() -> None:
    for column in ("personal_status_sex", "age_years", "foreign_worker"):
        assert column in EXCLUDED_FROM_FEATURES
        assert column not in MODEL_FEATURES
    assert set(MODEL_FEATURES) | set(EXCLUDED_FROM_FEATURES) == set(ATTRIBUTE_NAMES)
    for derived in ("sex", "sex_corrected", "age_group_25", "age_group_30", "bad", "good", "target"):
        assert derived not in MODEL_FEATURES


def test_synthetic_text_parses_and_is_reproducible() -> None:
    text = synthetic_german_data_text(n_rows=200, seed=3)
    assert text == synthetic_german_data_text(n_rows=200, seed=3)
    assert text != synthetic_german_data_text(n_rows=200, seed=4)
    frame = add_derived_columns(parse_german_data(text))
    assert len(frame) == 200
    assert 0 < frame["bad"].mean() < 1
    assert set(frame["age_group_25"]) == {"<25", ">=25"}


def test_load_from_file(tmp_path: Path) -> None:
    path = tmp_path / "german.data"
    path.write_text(synthetic_german_data_text(n_rows=50, seed=1), encoding="utf-8")
    frame = load_german_credit(path)
    assert len(frame) == 50
    assert {"bad", "good", "sex", "sex_corrected", "age_group_25", "age_group_30"} <= set(frame.columns)


def test_load_missing_file_says_how_to_get_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make data"):
        load_german_credit(tmp_path / "missing.data")
