"""Tests for the leakage guard and the column decisions behind it."""

from collections import Counter

import pytest

from uwcopilot.common.paths import DOCS_DIR
from uwcopilot.risk.credit import CREDIT_FEATURES
from uwcopilot.risk.leakage import (
    BANNED_POST_ORIGINATION_COLUMNS,
    CATEGORY_DESCRIPTIONS,
    COLUMN_DECISIONS,
    DOCUMENTED_COLUMNS,
    FEATURE,
    LENDER_OUTPUT_COLUMNS,
    POST_ORIGINATION,
    TARGET_COLUMN,
    LeakageError,
    assert_no_leakage,
    base_feature_name,
    decision_for,
    leakage_problems,
    undocumented_columns,
)

# Columns the build plan names explicitly as post-origination (docs/build_prompts.md, Prompt 1).
REQUIRED_BANNED = [
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "out_prncp",
    "out_prncp_inv",
    "last_credit_pull_d",
    "last_fico_range_high",
    "last_fico_range_low",
    "hardship_flag",
    "hardship_status",
    "settlement_status",
    "settlement_amount",
    "debt_settlement_flag",
    "debt_settlement_flag_date",
    "pymnt_plan",
    "loan_status",
]


def test_every_column_of_the_accepted_file_has_one_decision() -> None:
    assert len(DOCUMENTED_COLUMNS) == 151
    duplicates = [column for column, count in Counter(DOCUMENTED_COLUMNS).items() if count > 1]
    assert duplicates == []
    assert {decision.category for decision in COLUMN_DECISIONS} <= set(CATEGORY_DESCRIPTIONS)
    assert all(decision.reason.strip() for decision in COLUMN_DECISIONS)


def test_required_post_origination_columns_are_banned() -> None:
    missing = [column for column in REQUIRED_BANNED if column not in BANNED_POST_ORIGINATION_COLUMNS]
    assert missing == []
    assert TARGET_COLUMN in BANNED_POST_ORIGINATION_COLUMNS


def test_lending_club_risk_outputs_are_listed_separately() -> None:
    assert {"grade", "sub_grade", "int_rate", "installment"} <= LENDER_OUTPUT_COLUMNS
    assert not LENDER_OUTPUT_COLUMNS & BANNED_POST_ORIGINATION_COLUMNS


def test_model_allow_list_passes_the_guard() -> None:
    assert_no_leakage(CREDIT_FEATURES)
    for feature in CREDIT_FEATURES:
        decision = decision_for(feature)
        derived = feature in {"term_months", "emp_length_years", "credit_history_months", "loan_to_income"}
        assert derived or (decision is not None and decision.category == FEATURE), feature


@pytest.mark.parametrize("column", sorted(BANNED_POST_ORIGINATION_COLUMNS))
def test_guard_rejects_every_banned_column(column: str) -> None:
    with pytest.raises(LeakageError, match=column):
        assert_no_leakage([*CREDIT_FEATURES, column])


@pytest.mark.parametrize("column", sorted(LENDER_OUTPUT_COLUMNS))
def test_guard_rejects_lender_outputs_unless_allowed(column: str) -> None:
    with pytest.raises(LeakageError, match="underwriting/pricing output"):
        assert_no_leakage([column])
    assert_no_leakage([column], allow_lender_outputs=True)


@pytest.mark.parametrize(
    "encoded_name",
    [
        "cat__pymnt_plan_y",  # one-hot of a banned column inside a ColumnTransformer
        "num__missingindicator_recoveries",  # missing indicator of a banned column
        "hardship_payment_new_field",  # an unknown member of a banned family
        "settlement_new_field",
        "last_pymnt_d_month",
        "cat__grade_B",
    ],
)
def test_guard_catches_encoded_and_family_names(encoded_name: str) -> None:
    with pytest.raises(LeakageError):
        assert_no_leakage([encoded_name])


@pytest.mark.parametrize(
    "encoded_name",
    ["cat__purpose_car", "cat__home_ownership_RENT", "num__missingindicator_mths_since_last_delinq", "log__loan_amnt"],
)
def test_guard_accepts_encoded_allowed_names(encoded_name: str) -> None:
    assert leakage_problems([encoded_name]) == []


def test_error_lists_every_problem() -> None:
    with pytest.raises(LeakageError) as error:
        assert_no_leakage(["dti", "recoveries", "int_rate", "total_pymnt"])
    message = str(error.value)
    assert "recoveries" in message and "int_rate" in message and "total_pymnt" in message
    assert "'dti'" not in message


def test_base_feature_name() -> None:
    assert base_feature_name("num__missingindicator_dti") == "dti"
    assert base_feature_name("cat__purpose_car") == "purpose_car"
    assert base_feature_name("dti") == "dti"


def test_undocumented_columns() -> None:
    assert undocumented_columns(["dti", "brand_new_column"]) == ["brand_new_column"]


def test_docs_leakage_md_lists_every_column_with_its_decision() -> None:
    text = (DOCS_DIR / "leakage.md").read_text(encoding="utf-8")
    missing = [column for column in DOCUMENTED_COLUMNS if f"`{column}`" not in text]
    assert missing == []
    for decision in COLUMN_DECISIONS:
        if decision.category == POST_ORIGINATION:
            assert f"| `{decision.column}` | {decision.reason} |" in text
