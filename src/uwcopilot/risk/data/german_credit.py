"""Load the UCI Statlog German Credit data (german.data) into a labeled pandas DataFrame.

The file has 1,000 rows and 21 space-separated fields per row: 20 applicant attributes followed by the
target (1 = good credit risk, 2 = bad credit risk). Categorical attributes are stored as codes such as
"A11"; their meaning is given in german.doc (downloaded next to german.data by scripts/download_data.py).

This module is used only for the fairness demonstration (Phase 1, part C). Lending Club, the main credit
dataset, has no protected attributes, so group fairness cannot be measured on it.

Protected attributes and a known documentation error
----------------------------------------------------
- Age comes from attribute 13 (age in years). It is reported as "<25" vs ">=25" and, as a sensitivity
  check, "<=30" vs ">30".
- Sex comes from attribute 9, "personal status and sex". This attribute mixes sex with marital status, so
  it was never a clean sex variable.
- Worse, german.doc labels this attribute wrongly. Groemping (2019) checked the data against the original
  German sources and published a corrected code table with the "South German Credit" data (UCI
  Machine Learning Repository, 2020, https://doi.org/10.24432/C5QG88, file codetable.txt). The row counts
  per code match one to one (A91=50, A92=310, A93=548, A94=92 rows; corrected codes 1-4 have the same
  counts), and the corrected meanings are:
      A91 male: divorced/separated            (german.doc agrees)
      A92 female non-single OR male single    (german.doc says "female: divorced/separated/married")
      A93 male: married/widowed               (german.doc says "male: single")
      A94 female: single                      (german.doc says "male: married/widowed")
  So the german.doc mapping, which most fairness papers and libraries use, puts single women in the
  "male" group and some single men in the "female" group. We keep it as the primary "sex" column because
  it is the convention the task and the literature use, and add a "sex_corrected" column where A92 is
  "undetermined" (its rows mix both sexes). Results on "sex" must be read with this warning.
- The same report notes that the foreign-worker coding in german.doc is reversed. foreign_worker is not
  used as a model feature or as a fairness group here, so this does not affect the demo.

Reference: Groemping, U. (2019). South German Credit Data: Correcting a Widely Used Data Set.
Report 4/2019, Reports in Mathematics, Physics and Chemistry, Department II, Beuth University of
Applied Sciences Berlin.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from uwcopilot.common.paths import RAW_DATA_DIR

GERMAN_DATA_PATH = RAW_DATA_DIR / "german_credit" / "german.data"
TARGET_COLUMN = "target"  # raw target: 1 = good, 2 = bad
BAD_COLUMN = "bad"  # 1 = bad credit risk (the event the model predicts)
GOOD_COLUMN = "good"  # 1 = good credit risk (the favorable real outcome used for equal opportunity)

# Attributes in file order, with the type german.doc gives them (7 numerical, 13 categorical).
ATTRIBUTES: tuple[tuple[str, str], ...] = (
    ("checking_account_status", "categorical"),  # attribute 1
    ("duration_months", "numeric"),  # attribute 2
    ("credit_history", "categorical"),  # attribute 3
    ("purpose", "categorical"),  # attribute 4
    ("credit_amount", "numeric"),  # attribute 5
    ("savings", "categorical"),  # attribute 6
    ("employment_since", "categorical"),  # attribute 7
    ("installment_rate_pct", "numeric"),  # attribute 8
    ("personal_status_sex", "categorical"),  # attribute 9
    ("other_debtors", "categorical"),  # attribute 10
    ("residence_since", "numeric"),  # attribute 11
    ("property", "categorical"),  # attribute 12
    ("age_years", "numeric"),  # attribute 13
    ("other_installment_plans", "categorical"),  # attribute 14
    ("housing", "categorical"),  # attribute 15
    ("existing_credits", "numeric"),  # attribute 16
    ("job", "categorical"),  # attribute 17
    ("people_liable", "numeric"),  # attribute 18
    ("telephone", "categorical"),  # attribute 19
    ("foreign_worker", "categorical"),  # attribute 20
)
ATTRIBUTE_NAMES: tuple[str, ...] = tuple(name for name, _ in ATTRIBUTES)
NUMERIC_ATTRIBUTES: tuple[str, ...] = tuple(name for name, kind in ATTRIBUTES if kind == "numeric")
CATEGORICAL_ATTRIBUTES: tuple[str, ...] = tuple(name for name, kind in ATTRIBUTES if kind == "categorical")

# Code meanings copied from german.doc (see the module docstring for the attribute 9 correction).
CATEGORY_CODES: dict[str, dict[str, str]] = {
    "checking_account_status": {
        "A11": "< 0 DM",
        "A12": "0 <= ... < 200 DM",
        "A13": ">= 200 DM / salary assignments for at least 1 year",
        "A14": "no checking account",
    },
    "credit_history": {
        "A30": "no credits taken / all credits paid back duly",
        "A31": "all credits at this bank paid back duly",
        "A32": "existing credits paid back duly till now",
        "A33": "delay in paying off in the past",
        "A34": "critical account / other credits existing (not at this bank)",
    },
    "purpose": {
        "A40": "car (new)",
        "A41": "car (used)",
        "A42": "furniture/equipment",
        "A43": "radio/television",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation (german.doc: does not exist?)",
        "A48": "retraining",
        "A49": "business",
        "A410": "others",
    },
    "savings": {
        "A61": "< 100 DM",
        "A62": "100 <= ... < 500 DM",
        "A63": "500 <= ... < 1000 DM",
        "A64": ">= 1000 DM",
        "A65": "unknown / no savings account",
    },
    "employment_since": {
        "A71": "unemployed",
        "A72": "< 1 year",
        "A73": "1 <= ... < 4 years",
        "A74": "4 <= ... < 7 years",
        "A75": ">= 7 years",
    },
    "personal_status_sex": {
        "A91": "male: divorced/separated",
        "A92": "female: divorced/separated/married",
        "A93": "male: single",
        "A94": "male: married/widowed",
        "A95": "female: single",
    },
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {
        "A121": "real estate",
        "A122": "building society savings agreement / life insurance",
        "A123": "car or other, not in attribute 6",
        "A124": "unknown / no property",
    },
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {
        "A171": "unemployed / unskilled - non-resident",
        "A172": "unskilled - resident",
        "A173": "skilled employee / official",
        "A174": "management / self-employed / highly qualified employee / officer",
    },
    "telephone": {"A191": "none", "A192": "yes, registered under the customer's name"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}

# Sex as german.doc labels attribute 9 (the convention most fairness work uses).
SEX_BY_CODE_GERMAN_DOC: dict[str, str] = {
    "A91": "male",
    "A92": "female",
    "A93": "male",
    "A94": "male",
    "A95": "female",
}

# Sex from the corrected code table (Groemping 2019; UCI South German Credit 2020, codetable.txt).
# A92 mixes non-single women and single men, so its sex cannot be determined. A95 does not exist in the
# corrected table (and has no rows in german.data), so it is also "undetermined".
SEX_BY_CODE_CORRECTED: dict[str, str] = {
    "A91": "male",
    "A92": "undetermined",
    "A93": "male",
    "A94": "female",
    "A95": "undetermined",
}
CORRECTED_PERSONAL_STATUS_MEANING: dict[str, str] = {
    "A91": "male: divorced/separated",
    "A92": "female: non-single or male: single",
    "A93": "male: married/widowed",
    "A94": "female: single",
}

AGE_CUTOFF_PRIMARY = 25  # "<25" vs ">=25"
AGE_CUTOFF_SENSITIVITY = 30  # "<=30" vs ">30"

# Columns kept out of the model features (fairness through unawareness). Leaving them out does NOT remove
# proxy effects: other features (employment length, housing, ...) can still carry the same information.
EXCLUDED_FROM_FEATURES: dict[str, str] = {
    "personal_status_sex": "protected: encodes sex (and marital status)",
    "age_years": "protected: age",
    "foreign_worker": "relates to national origin; german.doc coding is also reported reversed (Groemping 2019)",
}
MODEL_NUMERIC_FEATURES: tuple[str, ...] = tuple(c for c in NUMERIC_ATTRIBUTES if c not in EXCLUDED_FROM_FEATURES)
MODEL_CATEGORICAL_FEATURES: tuple[str, ...] = tuple(
    c for c in CATEGORICAL_ATTRIBUTES if c not in EXCLUDED_FROM_FEATURES
)
MODEL_FEATURES: tuple[str, ...] = MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES


def parse_german_data(text: str) -> pd.DataFrame:
    """Parse the contents of german.data into a DataFrame with one named column per attribute plus target.

    Numeric attributes become integers, categorical ones become pandas categoricals whose categories are
    every code listed in german.doc (so train and test always share the same categories). Raises
    ValueError for a row with the wrong number of fields, a non-integer numeric value, an unknown code, or
    a target other than 1 or 2.
    """
    rows: list[list[str]] = []
    expected_fields = len(ATTRIBUTES) + 1
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != expected_fields:
            raise ValueError(f"line {line_number}: expected {expected_fields} fields, found {len(fields)}")
        rows.append(fields)
    if not rows:
        raise ValueError("no data rows found")

    frame = pd.DataFrame(rows, columns=[*ATTRIBUTE_NAMES, TARGET_COLUMN])
    for column in (*NUMERIC_ATTRIBUTES, TARGET_COLUMN):
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.isna().any():
            bad_value = frame.loc[converted.isna(), column].iloc[0]
            raise ValueError(f"column {column}: non-numeric value {bad_value!r}")
        if not (converted == converted.round()).all():
            raise ValueError(f"column {column}: expected whole numbers")
        frame[column] = converted.astype("int64")

    for column in CATEGORICAL_ATTRIBUTES:
        allowed = list(CATEGORY_CODES[column])
        unknown = sorted(set(frame[column]) - set(allowed))
        if unknown:
            raise ValueError(f"column {column}: unknown codes {unknown}; german.doc allows {allowed}")
        frame[column] = pd.Categorical(frame[column], categories=allowed)

    if not frame[TARGET_COLUMN].isin([1, 2]).all():
        raise ValueError("target must be 1 (good) or 2 (bad)")
    return frame


def age_group(age_years: pd.Series, cutoff: int, cutoff_is_upper_bound_of_young: bool) -> pd.Series:
    """Label each age as young or older around cutoff.

    cutoff_is_upper_bound_of_young=False gives "<cutoff" / ">=cutoff" (for example "<25" / ">=25");
    True gives "<=cutoff" / ">cutoff" (for example "<=30" / ">30").
    """
    if cutoff_is_upper_bound_of_young:
        young = age_years <= cutoff
        return pd.Series(np.where(young, f"<={cutoff}", f">{cutoff}"), index=age_years.index)
    young = age_years < cutoff
    return pd.Series(np.where(young, f"<{cutoff}", f">={cutoff}"), index=age_years.index)


def add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the label columns (bad, good) and the protected-group columns added.

    Added columns: bad, good, sex (german.doc labels), sex_corrected (corrected code table; A92 is
    "undetermined"), age_group_25 ("<25" / ">=25") and age_group_30 ("<=30" / ">30").
    """
    result = frame.copy()
    result[BAD_COLUMN] = (result[TARGET_COLUMN] == 2).astype("int64")
    result[GOOD_COLUMN] = 1 - result[BAD_COLUMN]
    codes = result["personal_status_sex"].astype(str)
    result["sex"] = codes.map(SEX_BY_CODE_GERMAN_DOC)
    result["sex_corrected"] = codes.map(SEX_BY_CODE_CORRECTED)
    result["age_group_25"] = age_group(result["age_years"], AGE_CUTOFF_PRIMARY, cutoff_is_upper_bound_of_young=False)
    result["age_group_30"] = age_group(result["age_years"], AGE_CUTOFF_SENSITIVITY, cutoff_is_upper_bound_of_young=True)
    return result


def load_german_credit(path: Path = GERMAN_DATA_PATH) -> pd.DataFrame:
    """Read german.data from disk and return the parsed frame with label and protected-group columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `make data` (scripts/download_data.py) first")
    return add_derived_columns(parse_german_data(path.read_text(encoding="utf-8")))


def synthetic_german_data_text(n_rows: int = 300, seed: int = 0) -> str:
    """SYNTHETIC rows in the german.data format, for tests only (no real applicants).

    Codes are drawn uniformly from german.doc's lists (A95 included, A47 included) and the target depends
    on the checking account, duration and age, so a model has something to learn and the age groups
    differ in their bad rate.
    """
    rng = np.random.default_rng(seed)
    numeric_ranges = {
        "duration_months": (4, 72),
        "credit_amount": (250, 18_500),
        "installment_rate_pct": (1, 4),
        "residence_since": (1, 4),
        "age_years": (19, 75),
        "existing_credits": (1, 4),
        "people_liable": (1, 2),
    }
    lines: list[str] = []
    for _ in range(n_rows):
        values: dict[str, str] = {}
        for name, kind in ATTRIBUTES:
            if kind == "numeric":
                low, high = numeric_ranges[name]
                values[name] = str(int(rng.integers(low, high + 1)))
            else:
                values[name] = str(rng.choice(list(CATEGORY_CODES[name])))
        logit = -1.0
        logit += 1.2 if values["checking_account_status"] == "A11" else 0.0
        logit -= 1.0 if values["checking_account_status"] == "A14" else 0.0
        logit += 0.03 * (int(values["duration_months"]) - 20)
        logit += 0.6 if int(values["age_years"]) < 25 else 0.0
        probability_bad = 1.0 / (1.0 + np.exp(-logit))
        target = 2 if rng.random() < probability_bad else 1
        lines.append(" ".join([*values.values(), str(target)]))
    return "\n".join(lines) + "\n"
