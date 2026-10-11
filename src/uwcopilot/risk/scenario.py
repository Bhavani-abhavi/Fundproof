"""Scenario overrides for stress tests: change some feature values and re-score, safely.

The allowed range of every feature is learned from the training data (build_feature_ranges). An override
must name a known feature and stay inside that feature's training range (numeric) or allowed values
(categorical), and be a whole number where every training value was one. Anything else is rejected with an
OverrideError. Values are never clipped silently, because a clipped value would make a stress test report a
scenario that was not the one requested.

validate_input_features applies the same rules to a complete applicant or policy before it is scored, so a
model is never asked to extrapolate from a base row either.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

import numpy as np
import pandas as pd

FeatureKind = Literal["numeric", "categorical"]


class OverrideError(ValueError):
    """Raised when a scenario override names an unknown feature or uses a value outside the allowed range."""


class InputFeaturesError(ValueError):
    """Raised when the features of an applicant or policy to score are incomplete or outside the training data."""


@dataclass(frozen=True)
class FeatureRange:
    """The values a feature may take in a scenario.

    Numeric features use min_value and max_value (inclusive, from the training data); integer_only is True
    when every training value was a whole number (VehPower, DrivAge, ...), so 6.5 is rejected.
    Categorical features use allowed_values (every category seen in the training data).
    missing_allowed is True when some training rows had no value, so a missing value is something the model
    has seen (for example months since the last delinquency, for an applicant with none).
    """

    name: str
    kind: FeatureKind
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: tuple[Any, ...] | None = None
    integer_only: bool = False
    missing_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for JSON output (for example in an API response or a results file)."""
        return {
            "name": self.name,
            "kind": self.kind,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "allowed_values": list(self.allowed_values) if self.allowed_values is not None else None,
            "integer_only": self.integer_only,
            "missing_allowed": self.missing_allowed,
        }

    @classmethod
    def from_dict(cls, entry: Mapping[str, Any]) -> "FeatureRange":
        """Rebuild a FeatureRange from to_dict() output (files written before integer_only and missing_allowed
        existed load with both set to False)."""
        allowed = entry.get("allowed_values")
        return cls(
            name=entry["name"],
            kind=entry["kind"],
            min_value=entry.get("min_value"),
            max_value=entry.get("max_value"),
            allowed_values=tuple(allowed) if allowed is not None else None,
            integer_only=bool(entry.get("integer_only", False)),
            missing_allowed=bool(entry.get("missing_allowed", False)),
        )


def _python_value(value: Any) -> Any:
    """Turn numpy scalars (np.int64, np.str_, ...) into plain Python values so they compare and print cleanly."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_feature_ranges(df: pd.DataFrame, categorical_columns: Iterable[str] | None = None) -> dict[str, FeatureRange]:
    """Learn a FeatureRange for every column of the training data.

    Numeric columns (except bool) become numeric ranges [min, max] of their non-missing values, marked
    integer_only when every value is a whole number. Bool, object, string and category columns, plus any
    column named in categorical_columns, become categorical with their non-missing values sorted as
    allowed_values. missing_allowed records whether the column had any missing value.
    """
    forced_categorical = set(categorical_columns or [])
    unknown = forced_categorical - set(df.columns)
    if unknown:
        raise ValueError(f"categorical_columns not in the data: {sorted(unknown)}")

    ranges: dict[str, FeatureRange] = {}
    for column in df.columns:
        values = df[column].dropna()
        if values.empty:
            raise ValueError(f"column '{column}' has no non-missing values, so its range cannot be learned")
        has_missing = bool(len(values) < len(df[column]))
        is_numeric = pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_bool_dtype(df[column])
        if is_numeric and column not in forced_categorical:
            numbers = values.to_numpy(dtype=float)
            ranges[column] = FeatureRange(
                name=column,
                kind="numeric",
                min_value=float(numbers.min()),
                max_value=float(numbers.max()),
                integer_only=bool(np.all(numbers == np.round(numbers))),
                missing_allowed=has_missing,
            )
        else:
            allowed = sorted({_python_value(value) for value in values.unique()}, key=str)
            ranges[column] = FeatureRange(
                name=column, kind="categorical", allowed_values=tuple(allowed), missing_allowed=has_missing
            )
    return ranges


def _problem_with_value(value: Any, feature_range: FeatureRange) -> str | None:
    """Return a description of what is wrong with one override value, or None if it is allowed."""
    name = feature_range.name
    if feature_range.kind == "numeric":
        if isinstance(value, bool) or not isinstance(value, Real):
            return f"'{name}' is numeric, got {value!r} of type {type(value).__name__}"
        if np.isnan(float(value)):
            return f"'{name}' cannot be set to NaN"
        if not feature_range.min_value <= float(value) <= feature_range.max_value:
            return (
                f"'{name}' = {value!r} is outside the training range "
                f"[{feature_range.min_value}, {feature_range.max_value}]"
            )
        if feature_range.integer_only and not float(value).is_integer():
            return f"'{name}' = {value!r} is not a whole number (every training value of '{name}' is one)"
        return None
    allowed = feature_range.allowed_values or ()
    if _python_value(value) not in allowed:
        shown = list(allowed[:20])
        more = f" (and {len(allowed) - 20} more)" if len(allowed) > 20 else ""
        return f"'{name}' = {value!r} is not an allowed value; allowed: {shown}{more}"
    return None


def validate_overrides(
    overrides: Mapping[str, Any] | None, ranges: Mapping[str, FeatureRange], what: str = "scenario overrides"
) -> None:
    """Check every override and raise one OverrideError listing all problems. Returns None if all are valid.

    Problems: the feature is unknown, a numeric feature gets a non-number, NaN, a value outside its
    training range or a fraction where training had only whole numbers, or a categorical feature gets a value
    not seen in training. Nothing is clipped. what names the values in the error message (the scoring API
    also uses this to check input features).
    """
    problems = _value_problems(overrides or {}, ranges)
    if problems:
        raise OverrideError(f"Rejected {what} (values are never clipped): " + "; ".join(problems))


def _value_problems(values: Mapping[str, Any], ranges: Mapping[str, FeatureRange]) -> list[str]:
    """One description per value that is not allowed (unknown feature or a value outside its training data)."""
    problems = []
    for name, value in values.items():
        if name not in ranges:
            problems.append(f"unknown feature '{name}'")
            continue
        problem = _problem_with_value(value, ranges[name])
        if problem is not None:
            problems.append(problem)
    return problems


def _is_missing(value: Any) -> bool:
    """True for None and for a float NaN (the two ways a missing value arrives)."""
    return value is None or (isinstance(value, float) and np.isnan(value))


def validate_input_features(features: Mapping[str, Any], ranges: Mapping[str, FeatureRange]) -> None:
    """Check the complete feature set of one applicant or policy before scoring it. Returns None if valid.

    Every feature in ranges must be present, and no other. A missing value (None or NaN) is accepted only for
    a feature that was missing in some training rows (FeatureRange.missing_allowed); send it explicitly as
    null, never by leaving the key out. Every other value gets the same checks as a scenario override (inside
    the training range, a category seen in training, a whole number where training had only whole numbers).
    Raises one InputFeaturesError listing every problem; nothing is filled in or clipped.
    """
    problems = []
    unknown = sorted(set(features) - set(ranges))
    missing_keys = sorted(set(ranges) - set(features))
    if unknown:
        problems.append(f"unknown features {unknown}")
    if missing_keys:
        problems.append(f"missing features {missing_keys} (send null for a value that is genuinely unknown)")
    present = {}
    for name in sorted(set(features) & set(ranges)):
        value = features[name]
        if _is_missing(value):
            if not ranges[name].missing_allowed:
                problems.append(f"'{name}' cannot be missing (it was never missing in the training data)")
        else:
            present[name] = value
    problems += _value_problems(present, ranges)
    if problems:
        message = "Rejected input features (values are never filled in or clipped): " + "; ".join(problems)
        raise InputFeaturesError(message)


def apply_overrides(
    row: Mapping[str, Any], overrides: Mapping[str, Any] | None, ranges: Mapping[str, FeatureRange]
) -> dict[str, Any]:
    """Return a copy of row with the overrides applied, after validating them. The input row is not changed.

    With no overrides (None or {}), the copy is identical to the input.
    """
    validate_overrides(overrides, ranges)
    updated = dict(row)
    for name, value in (overrides or {}).items():
        updated[name] = _python_value(value)
    return updated


Direction = Literal["increase", "decrease"]
RELATIVE_TOLERANCE = 1e-9  # changes smaller than this share of the baseline count as "unchanged"


def monotonic_check(
    predict: Callable[[pd.DataFrame], np.ndarray],
    rows: pd.DataFrame,
    feature: str,
    step: float,
    feature_range: FeatureRange,
    expected: Direction = "increase",
) -> dict[str, Any]:
    """Sanity check for a scenario: does adding step to one feature move predictions the expected way?

    Every row whose shifted value stays inside the feature's training range is scored twice (as is, and with
    feature + step); rows that would leave the range are skipped and counted, never clipped. A violation is
    a row whose prediction moves against the expected direction. The share of violations is reported rather
    than asserted to be zero: a flexible model can legitimately be non-monotonic for a few rows, and the
    point is to measure how often.
    """
    if feature_range.kind != "numeric":
        raise ValueError(f"monotonic_check needs a numeric feature, '{feature}' is {feature_range.kind}")
    if expected not in ("increase", "decrease"):
        raise ValueError("expected must be 'increase' or 'decrease'")
    if step == 0:
        raise ValueError("step must not be zero")
    original = rows[feature].astype(float)
    shifted_values = original + step
    missing = original.isna().to_numpy()
    inside = shifted_values.between(feature_range.min_value, feature_range.max_value).to_numpy()
    checked = rows.loc[inside]
    result: dict[str, Any] = {
        "feature": feature,
        "step": float(step),
        "expected_direction": expected,
        "n_rows_checked": int(inside.sum()),
        "n_rows_skipped_outside_training_range": int((~inside & ~missing).sum()),
        "n_rows_skipped_missing_value": int(missing.sum()),
    }
    if checked.empty:
        result["status"] = "not measured"
        result["reason"] = "no row stays inside the training range after the shift"
        return result
    shifted = checked.copy()
    shifted[feature] = shifted_values[inside].to_numpy()
    baseline = np.asarray(predict(checked), dtype=float)
    scenario = np.asarray(predict(shifted), dtype=float)
    change = scenario - baseline
    unchanged = np.abs(change) <= RELATIVE_TOLERANCE * np.abs(baseline)
    moved_up = (change > 0) & ~unchanged
    moved_down = (change < 0) & ~unchanged
    violations = moved_down if expected == "increase" else moved_up
    result.update(
        {
            "share_violations": float(violations.mean()),
            "share_unchanged": float(unchanged.mean()),
            "share_expected_direction": float((moved_up if expected == "increase" else moved_down).mean()),
            "mean_relative_change": float(np.mean(change / baseline)),
        }
    )
    return result
