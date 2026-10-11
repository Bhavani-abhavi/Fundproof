"""Tests for the Lending Club loader, on SYNTHETIC Lending Club-shaped data (no real data needed)."""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uwcopilot.risk.data.lending_club import (
    BAD_STATUSES,
    FEATURES,
    GOOD_STATUSES,
    MISSING_CATEGORY,
    PREPARED_COLUMNS,
    REQUIRED_RAW_COLUMNS,
    UNFINISHED_STATUSES,
    finished_loans,
    load_lending_club,
    map_loan_status,
    months_between,
    parse_emp_length,
    parse_month,
    parse_percent,
    parse_term_months,
    prepare_loans,
    synthetic_accepted_loans,
    vintage_table,
    write_lending_club_zip,
    zip_status,
)
from uwcopilot.risk.leakage import BANNED_POST_ORIGINATION_COLUMNS, LENDER_OUTPUT_COLUMNS, TARGET_COLUMN


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    """SYNTHETIC raw frame with every documented column, unfinished loans, joint apps and footer rows."""
    return synthetic_accepted_loans(n_rows=800, seed=5)


@pytest.fixture(scope="module")
def loans(raw: pd.DataFrame) -> pd.DataFrame:
    return prepare_loans(raw)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def test_parse_term_months() -> None:
    parsed = parse_term_months(pd.Series([" 36 months", " 60 months", None]))
    assert parsed.iloc[0] == 36.0 and parsed.iloc[1] == 60.0 and np.isnan(parsed.iloc[2])


def test_parse_emp_length() -> None:
    parsed = parse_emp_length(pd.Series(["< 1 year", "1 year", "7 years", "10+ years", "n/a", None]))
    assert parsed.iloc[:4].tolist() == [0.0, 1.0, 7.0, 10.0]
    assert parsed.iloc[4:].isna().all()


def test_parse_percent_accepts_strings_and_numbers() -> None:
    parsed = parse_percent(pd.Series(["13.5%", " 7.07% ", 12.0, None, "bad"]))
    assert parsed.iloc[:3].tolist() == [13.5, 7.07, 12.0]
    assert parsed.iloc[3:].isna().all()


def test_parse_month_and_months_between() -> None:
    start = parse_month(pd.Series(["Aug-2003", "Dec-2015", "garbage"]))
    end = parse_month(pd.Series(["Dec-2015", "Jan-2016", "Dec-2015"]))
    months = months_between(start, end)
    assert months.iloc[0] == (2015 - 2003) * 12 + (12 - 8)
    assert months.iloc[1] == 1
    assert np.isnan(months.iloc[2])


# ---------------------------------------------------------------------------
# Target mapping and finished-loan filter
# ---------------------------------------------------------------------------


def test_target_mapping_good_bad_unfinished() -> None:
    statuses = pd.Series([*GOOD_STATUSES, *BAD_STATUSES, *UNFINISHED_STATUSES])
    is_bad, is_finished = map_loan_status(statuses)
    n_good, n_bad = len(GOOD_STATUSES), len(BAD_STATUSES)
    assert is_bad.iloc[:n_good].tolist() == [0.0] * n_good
    assert is_bad.iloc[n_good : n_good + n_bad].tolist() == [1.0] * n_bad
    assert is_bad.iloc[n_good + n_bad :].isna().all()
    assert is_finished.tolist() == [True] * (n_good + n_bad) + [False] * len(UNFINISHED_STATUSES)


def test_old_credit_policy_statuses_map_like_plain_status() -> None:
    is_bad, _ = map_loan_status(
        pd.Series(
            [
                "Does not meet the credit policy. Status:Fully Paid",
                "Does not meet the credit policy. Status:Charged Off",
            ]
        )
    )
    assert is_bad.tolist() == [0.0, 1.0]


def test_unknown_status_raises() -> None:
    with pytest.raises(ValueError, match="unknown loan_status"):
        map_loan_status(pd.Series(["Fully Paid", "Paid in kind"]))


def test_finished_filter_keeps_only_final_outcomes(loans: pd.DataFrame) -> None:
    finished = finished_loans(loans)
    assert finished["is_finished"].all()
    assert set(finished[TARGET_COLUMN].unique()) == {0, 1}
    assert not finished["loan_status"].isin(UNFINISHED_STATUSES).any()
    only_finished = prepare_loans(synthetic_accepted_loans(n_rows=800, seed=5), keep_unfinished=False)
    assert len(only_finished) == len(finished)


def test_unfinished_loans_have_no_target(loans: pd.DataFrame) -> None:
    unfinished = loans.loc[~loans["is_finished"]]
    assert len(unfinished) > 0
    assert unfinished[TARGET_COLUMN].isna().all()
    assert unfinished["loan_status"].isin(UNFINISHED_STATUSES).all()


def test_footer_rows_and_joint_applications_are_removed(raw: pd.DataFrame, loans: pd.DataFrame) -> None:
    n_footer = int(raw["loan_status"].isna().sum())
    n_joint = int((raw["application_type"] == "Joint App").sum())
    assert n_footer == 2 and n_joint > 0
    assert len(loans) == len(raw) - n_footer - n_joint


def test_old_credit_policy_flag(loans: pd.DataFrame) -> None:
    flagged = loans.loc[loans["old_credit_policy"]]
    assert len(flagged) > 0
    assert flagged["loan_status"].str.startswith("Does not meet the credit policy.").all()


# ---------------------------------------------------------------------------
# Columns and features
# ---------------------------------------------------------------------------


def test_prepared_frame_has_only_allow_listed_columns(loans: pd.DataFrame) -> None:
    assert loans.columns.tolist() == PREPARED_COLUMNS
    # loan_status and is_bad stay as metadata (target and vintage tables); every other banned column is dropped.
    assert set(loans.columns) & BANNED_POST_ORIGINATION_COLUMNS == {"loan_status", TARGET_COLUMN}
    assert "loan_status" not in FEATURES and TARGET_COLUMN not in FEATURES


def test_features_contain_no_banned_or_lender_columns() -> None:
    assert not set(FEATURES) & BANNED_POST_ORIGINATION_COLUMNS
    assert not set(FEATURES) & LENDER_OUTPUT_COLUMNS


def test_derived_features(loans: pd.DataFrame) -> None:
    assert set(loans["term_months"].unique()) == {36.0, 60.0}
    expected_ratio = loans["loan_amnt"] / loans["annual_inc"]
    np.testing.assert_allclose(loans["loan_to_income"], expected_ratio)
    assert (loans["credit_history_months"].dropna() >= 0).all()
    assert loans["revol_util"].dtype == float  # "52.3%" strings were converted
    assert loans["int_rate"].between(0, 40).all()


def test_home_ownership_grouping_and_missing_categories(raw: pd.DataFrame) -> None:
    changed = raw.copy()
    changed.loc[0, "home_ownership"] = "NONE"
    changed.loc[1, "purpose"] = np.nan
    changed.loc[[0, 1], "application_type"] = "Individual"
    changed.loc[[0, 1], "loan_status"] = "Fully Paid"
    prepared = prepare_loans(changed.iloc[:2])
    assert prepared.loc[0, "home_ownership"] == "OTHER"
    assert prepared.loc[1, "purpose"] == MISSING_CATEGORY


def test_bad_values_become_missing_not_errors(raw: pd.DataFrame) -> None:
    changed = raw.iloc[:3].copy()
    changed["application_type"] = "Individual"
    changed["loan_status"] = "Fully Paid"
    changed.loc[0, "annual_inc"] = 0.0
    changed.loc[1, "earliest_cr_line"] = "Jan-2100"  # after the issue date
    prepared = prepare_loans(changed)
    assert np.isnan(prepared.loc[0, "loan_to_income"])
    assert np.isnan(prepared.loc[1, "credit_history_months"])


def test_missing_required_column_raises(raw: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        prepare_loans(raw.drop(columns=["dti"]))


def test_unparseable_issue_date_raises(raw: pd.DataFrame) -> None:
    changed = raw.iloc[:3].copy()
    changed["application_type"] = "Individual"
    changed["loan_status"] = "Fully Paid"
    changed.loc[0, "issue_d"] = "2015-12"
    with pytest.raises(ValueError, match="issue_d could not be parsed"):
        prepare_loans(changed)


# ---------------------------------------------------------------------------
# Vintage table
# ---------------------------------------------------------------------------


def test_vintage_table_counts(loans: pd.DataFrame) -> None:
    table = vintage_table(loans)
    assert [row["issue_year"] for row in table] == sorted(loans["issue_year"].unique().tolist())
    assert sum(row["n_loans"] for row in table) == len(loans)
    assert sum(row["n_finished"] for row in table) == int(loans["is_finished"].sum())
    for row in table:
        unfinished = sum(row["n_unfinished_by_status"].values())
        assert row["n_finished"] + unfinished == row["n_loans"]


# ---------------------------------------------------------------------------
# Reading the Kaggle zip lazily
# ---------------------------------------------------------------------------


def test_load_from_zip_in_chunks_matches_in_memory(tmp_path: Path, raw: pd.DataFrame, loans: pd.DataFrame) -> None:
    zip_path = write_lending_club_zip(raw, tmp_path / "lending-club.zip")
    data = load_lending_club(zip_path, chunk_rows=97)  # many small chunks, including a footer-only one
    pd.testing.assert_frame_equal(data.loans, loans, check_dtype=False)


def test_quality_report_counts(tmp_path: Path, raw: pd.DataFrame, loans: pd.DataFrame) -> None:
    zip_path = write_lending_club_zip(raw, tmp_path / "lending-club.zip")
    quality = load_lending_club(zip_path, chunk_rows=250).quality
    assert quality["n_rows_in_file"] == len(raw)
    assert quality["n_rows_without_status_or_issue_date"] == 2
    assert quality["n_joint_applications_excluded"] == int((raw["application_type"] == "Joint App").sum())
    assert sum(quality["joint_applications_by_issue_year"].values()) == quality["n_joint_applications_excluded"]
    assert min(int(year) for year in quality["joint_applications_by_issue_year"]) >= 2016  # as generated
    assert quality["n_individual_loans"] == len(loans)
    assert quality["n_finished"] == int(loans["is_finished"].sum())
    assert sum(quality["status_counts_individual"].values()) == len(loans)
    assert quality["undocumented_columns_in_file"] == []
    assert quality["documented_columns_missing_from_file"] == []
    # The late-introduced columns are empty in the synthetic data, and the audit sees that.
    assert set(quality["missing_share_by_issue_year"]["il_util"].values()) == {1.0}
    assert quality["missing_share_by_issue_year"]["num_il_tl"]["2013"] == 1.0  # empty in the first synthetic year


def test_zip_with_only_required_columns_loads(tmp_path: Path, raw: pd.DataFrame) -> None:
    zip_path = write_lending_club_zip(raw[REQUIRED_RAW_COLUMNS], tmp_path / "small.zip")
    data = load_lending_club(zip_path)
    assert len(data.loans) > 0
    assert "il_util" in data.quality["documented_columns_missing_from_file"]


def test_zip_status(tmp_path: Path, raw: pd.DataFrame) -> None:
    assert zip_status(tmp_path / "missing.zip")[0] is False
    not_a_zip = tmp_path / "broken.zip"
    not_a_zip.write_text("not a zip")
    assert zip_status(not_a_zip) == (False, f"{not_a_zip} is not a readable zip file")
    empty_zip = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty_zip, "w") as archive:
        archive.writestr("rejected_2007_to_2018Q4.csv.gz", b"")
    assert zip_status(empty_zip)[0] is False
    good = write_lending_club_zip(raw.iloc[:10], tmp_path / "lending-club.zip")
    assert zip_status(good)[0] is True
    (tmp_path / "lending-club.zip.kaggle-partial").write_text("")
    assert "partial" in zip_status(good)[1]
