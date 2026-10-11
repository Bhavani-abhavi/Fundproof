"""Tests for uwcopilot.risk.scenario: ranges are learned from training data and overrides are never clipped."""

import numpy as np
import pandas as pd
import pytest

from uwcopilot.risk.scenario import (
    FeatureRange,
    InputFeaturesError,
    OverrideError,
    apply_overrides,
    build_feature_ranges,
    monotonic_check,
    validate_input_features,
    validate_overrides,
)


@pytest.fixture
def training_data() -> pd.DataFrame:
    """Small synthetic training table with numeric, categorical and boolean columns."""
    return pd.DataFrame(
        {
            "dti": [5.0, 12.5, 30.0, np.nan],
            "loan_amount": [1000, 5000, 20000, 7000],
            "grade": pd.Categorical(["A", "B", "C", "B"]),
            "home_ownership": ["RENT", "OWN", "MORTGAGE", "RENT"],
            "is_joint": [True, False, False, True],
            "term_months": [36, 60, 36, 60],
        }
    )


@pytest.fixture
def ranges(training_data: pd.DataFrame) -> dict[str, FeatureRange]:
    return build_feature_ranges(training_data, categorical_columns=["term_months"])


def test_build_feature_ranges(ranges: dict[str, FeatureRange]) -> None:
    assert ranges["dti"] == FeatureRange(
        name="dti", kind="numeric", min_value=5.0, max_value=30.0, integer_only=False, missing_allowed=True
    )
    assert ranges["loan_amount"].integer_only and not ranges["loan_amount"].missing_allowed
    assert ranges["loan_amount"].min_value == 1000.0
    assert ranges["loan_amount"].max_value == 20000.0
    assert ranges["grade"].kind == "categorical"
    assert ranges["grade"].allowed_values == ("A", "B", "C")
    assert ranges["home_ownership"].allowed_values == ("MORTGAGE", "OWN", "RENT")
    assert ranges["is_joint"].kind == "categorical"
    assert ranges["term_months"].allowed_values == (36, 60)


def test_feature_range_to_dict(ranges: dict[str, FeatureRange]) -> None:
    assert ranges["grade"].to_dict() == {
        "name": "grade",
        "kind": "categorical",
        "min_value": None,
        "max_value": None,
        "allowed_values": ["A", "B", "C"],
        "integer_only": False,
        "missing_allowed": False,
    }


def test_feature_range_round_trips_through_a_dict(ranges: dict[str, FeatureRange]) -> None:
    for feature_range in ranges.values():
        assert FeatureRange.from_dict(feature_range.to_dict()) == feature_range
    # A file written before integer_only and missing_allowed existed still loads (both False).
    old = {"name": "x", "kind": "numeric", "min_value": 0.0, "max_value": 1.0, "allowed_values": None}
    assert FeatureRange.from_dict(old) == FeatureRange(name="x", kind="numeric", min_value=0.0, max_value=1.0)


def test_fraction_of_a_whole_number_feature_is_rejected(ranges: dict[str, FeatureRange]) -> None:
    with pytest.raises(OverrideError, match="'loan_amount' = 5000.5 is not a whole number"):
        validate_overrides({"loan_amount": 5000.5}, ranges)
    validate_overrides({"loan_amount": 5000.0, "dti": 12.75}, ranges)  # a whole float, and a fractional feature


def test_validate_input_features_accepts_a_complete_row(ranges: dict[str, FeatureRange]) -> None:
    row = {
        "dti": None,
        "loan_amount": 5000,
        "grade": "B",
        "home_ownership": "OWN",
        "is_joint": False,
        "term_months": 36,
    }
    validate_input_features(row, ranges)  # dti was missing in training, so null is a value the model has seen
    validate_input_features({**row, "dti": float("nan")}, ranges)


def test_validate_input_features_lists_every_problem(ranges: dict[str, FeatureRange]) -> None:
    row = {"dti": 99.0, "loan_amount": None, "grade": "Z", "is_joint": False, "term_months": 36, "income": 1}
    with pytest.raises(InputFeaturesError) as error:
        validate_input_features(row, ranges)
    message = str(error.value)
    assert "unknown features ['income']" in message
    assert "missing features ['home_ownership']" in message
    assert "'loan_amount' cannot be missing" in message
    assert "'dti' = 99.0 is outside the training range" in message
    assert "'grade' = 'Z' is not an allowed value" in message
    assert not isinstance(error.value, OverrideError)  # the API reports it as invalid input, not an override
    with pytest.raises(InputFeaturesError, match="missing features"):
        validate_input_features({}, ranges)


def test_valid_overrides_pass(ranges: dict[str, FeatureRange]) -> None:
    validate_overrides({"dti": 30.0, "grade": "C", "term_months": 60, "loan_amount": np.int64(1000)}, ranges)
    validate_overrides(None, ranges)
    validate_overrides({}, ranges)


def test_unknown_feature_is_rejected(ranges: dict[str, FeatureRange]) -> None:
    with pytest.raises(OverrideError, match="unknown feature 'income'"):
        validate_overrides({"income": 50000}, ranges)


def test_out_of_range_numeric_is_rejected_not_clipped(ranges: dict[str, FeatureRange]) -> None:
    row = {"dti": 12.5, "grade": "B"}
    with pytest.raises(OverrideError, match=r"'dti' = 45.0 is outside the training range \[5.0, 30.0\]"):
        apply_overrides(row, {"dti": 45.0}, ranges)
    assert row["dti"] == 12.5  # input untouched


def test_bad_numeric_values_are_rejected(ranges: dict[str, FeatureRange]) -> None:
    with pytest.raises(OverrideError, match="numeric"):
        validate_overrides({"dti": "high"}, ranges)
    with pytest.raises(OverrideError, match="numeric"):
        validate_overrides({"dti": True}, ranges)
    with pytest.raises(OverrideError, match="NaN"):
        validate_overrides({"dti": float("nan")}, ranges)


def test_unknown_category_is_rejected(ranges: dict[str, FeatureRange]) -> None:
    with pytest.raises(OverrideError, match="'grade' = 'Z' is not an allowed value"):
        validate_overrides({"grade": "Z"}, ranges)


def test_all_problems_reported_together(ranges: dict[str, FeatureRange]) -> None:
    with pytest.raises(OverrideError) as error:
        validate_overrides({"dti": 99.0, "grade": "Z", "income": 1}, ranges)
    message = str(error.value)
    assert "'dti'" in message and "'grade'" in message and "'income'" in message


def test_empty_override_changes_nothing(ranges: dict[str, FeatureRange]) -> None:
    row = {"dti": 12.5, "grade": "B", "loan_amount": 5000}
    assert apply_overrides(row, {}, ranges) == row
    assert apply_overrides(row, None, ranges) == row
    assert apply_overrides(row, {}, ranges) is not row


def test_apply_overrides_returns_updated_copy(ranges: dict[str, FeatureRange]) -> None:
    row = {"dti": 12.5, "grade": "B", "loan_amount": 5000}
    updated = apply_overrides(row, {"dti": 25.0, "grade": "C"}, ranges)
    assert updated == {"dti": 25.0, "grade": "C", "loan_amount": 5000}
    assert row["dti"] == 12.5


def test_column_with_only_missing_values_cannot_be_ranged() -> None:
    with pytest.raises(ValueError, match="no non-missing values"):
        build_feature_ranges(pd.DataFrame({"empty": [np.nan, np.nan]}))


# ---------------------------------------------------------------------------
# Monotonic sanity check
# ---------------------------------------------------------------------------


def _linear_predict(frame: pd.DataFrame) -> np.ndarray:
    """A toy model whose prediction rises with x: 1 + x."""
    return 1.0 + frame["x"].to_numpy(dtype=float)


def test_monotonic_check_counts_rows_and_never_clips() -> None:
    rows = pd.DataFrame({"x": [0.0, 1.0, 2.0, 9.5, np.nan]})
    feature_range = FeatureRange(name="x", kind="numeric", min_value=0.0, max_value=10.0)
    result = monotonic_check(_linear_predict, rows, "x", 1.0, feature_range, expected="increase")
    assert result["n_rows_checked"] == 3  # 9.5 + 1 leaves the range and is skipped, not clipped
    assert result["n_rows_skipped_outside_training_range"] == 1
    assert result["n_rows_skipped_missing_value"] == 1
    assert result["share_violations"] == 0.0
    assert result["share_expected_direction"] == 1.0


def test_monotonic_check_reports_violations_instead_of_failing() -> None:
    rows = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    feature_range = FeatureRange(name="x", kind="numeric", min_value=0.0, max_value=10.0)
    result = monotonic_check(_linear_predict, rows, "x", 1.0, feature_range, expected="decrease")
    assert result["share_violations"] == 1.0  # every row moved up although "decrease" was expected


def test_monotonic_check_flat_model_is_unchanged() -> None:
    rows = pd.DataFrame({"x": [0.0, 1.0]})
    feature_range = FeatureRange(name="x", kind="numeric", min_value=0.0, max_value=10.0)
    result = monotonic_check(lambda frame: np.ones(len(frame)), rows, "x", 2.0, feature_range)
    assert result["share_unchanged"] == 1.0 and result["share_violations"] == 0.0


def test_monotonic_check_rejects_bad_arguments() -> None:
    rows = pd.DataFrame({"x": [0.0]})
    numeric = FeatureRange(name="x", kind="numeric", min_value=0.0, max_value=1.0)
    with pytest.raises(ValueError, match="numeric"):
        monotonic_check(_linear_predict, rows, "x", 1.0, FeatureRange(name="x", kind="categorical"))
    with pytest.raises(ValueError, match="zero"):
        monotonic_check(_linear_predict, rows, "x", 0.0, numeric)
    with pytest.raises(ValueError, match="expected"):
        monotonic_check(_linear_predict, rows, "x", 1.0, numeric, expected="sideways")  # type: ignore[arg-type]


def test_validate_overrides_names_what_was_checked() -> None:
    ranges = {"x": FeatureRange(name="x", kind="numeric", min_value=0.0, max_value=1.0)}
    with pytest.raises(OverrideError, match="Rejected input features"):
        validate_overrides({"x": 5.0}, ranges, what="input features")
