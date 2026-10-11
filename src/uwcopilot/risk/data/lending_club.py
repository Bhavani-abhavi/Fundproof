"""Load and prepare the Lending Club accepted-loans file for the credit-default model.

Source: Kaggle dataset "wordsforthewise/lending-club", downloaded by scripts/download_data.py to
data/raw/lending_club/lending-club.zip. Inside the zip, accepted_2007_to_2018Q4.csv.gz has one row per
issued loan (about 2.2 million rows and 151 columns, about 1.6 GB uncompressed). The file is read lazily:
straight from the zip, in chunks, and only the columns the model needs (with fixed dtypes).

Target (finished loans only):
    Fully Paid                                            -> 0 (good)
    Does not meet the credit policy. Status:Fully Paid    -> 0 (good)
    Charged Off                                           -> 1 (bad)
    Default                                               -> 1 (bad: 121+ days past due, not yet charged off)
    Does not meet the credit policy. Status:Charged Off   -> 1 (bad)
    Current, In Grace Period, Late (16-30 days), Late (31-120 days) -> unfinished, no target
Any other status raises an error, so a new label is never mapped silently. The "Does not meet the credit
policy" loans were issued under an older credit policy (2007-2010); their outcome is mapped like the
plain status, and the loader counts them. That flag lives in loan_status, so it is never a feature.

Population: individual applications only. Joint applications have co-borrower fields that do not exist for
most training years, so they are counted and left out.

Unfinished loans are kept in the prepared frame (is_finished = False, is_bad = NaN) because the stability
checks (PSI) compare populations and do not need outcomes. The models only train and test on finished loans.

All columns used as features are known at application time; see uwcopilot.risk.leakage for every column's
decision. This module also builds SYNTHETIC Lending Club-shaped data for tests (synthetic_accepted_loans).
"""

import gzip
import io
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from uwcopilot.common.paths import RAW_DATA_DIR
from uwcopilot.risk.leakage import (
    DOCUMENTED_COLUMNS,
    LATE_INTRODUCED_COLUMNS,
    TARGET_COLUMN,
    assert_no_leakage,
    undocumented_columns,
)

DEFAULT_ZIP_PATH = RAW_DATA_DIR / "lending_club" / "lending-club.zip"
ACCEPTED_FILE_NAMES = ("accepted_2007_to_2018q4.csv.gz", "accepted_2007_to_2018q4.csv")
DEFAULT_CHUNK_ROWS = 200_000

GOOD_STATUSES = ("Fully Paid", "Does not meet the credit policy. Status:Fully Paid")
BAD_STATUSES = ("Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off")
UNFINISHED_STATUSES = ("Current", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)")
OLD_POLICY_PREFIX = "Does not meet the credit policy."
INDIVIDUAL_APPLICATION = "Individual"
MISSING_CATEGORY = "missing"
HOME_OWNERSHIP_GROUPS = {"ANY": "OTHER", "NONE": "OTHER", "OTHER": "OTHER"}
MONTH_FORMAT = "%b-%Y"  # issue_d and earliest_cr_line look like "Dec-2015"

# Application-time features used by the models (see uwcopilot.risk.leakage for why each one is allowed).
DERIVED_NUMERIC_FEATURES = ["term_months", "emp_length_years", "credit_history_months", "loan_to_income"]
BUREAU_NUMERIC_FEATURES = [
    "delinq_2yrs",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "collections_12_mths_ex_med",
    "mths_since_last_major_derog",
    "acc_now_delinq",
    "tot_coll_amt",
    "tot_cur_bal",
    "total_rev_hi_lim",
    "acc_open_past_24mths",
    "avg_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "chargeoff_within_12_mths",
    "delinq_amnt",
    "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl",
    "mort_acc",
    "mths_since_recent_bc",
    "mths_since_recent_bc_dlq",
    "mths_since_recent_inq",
    "mths_since_recent_revol_delinq",
    "num_accts_ever_120_pd",
    "num_actv_bc_tl",
    "num_actv_rev_tl",
    "num_bc_sats",
    "num_bc_tl",
    "num_il_tl",
    "num_op_rev_tl",
    "num_rev_accts",
    "num_rev_tl_bal_gt_0",
    "num_sats",
    "num_tl_120dpd_2m",
    "num_tl_30dpd",
    "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m",
    "pct_tl_nvr_dlq",
    "percent_bc_gt_75",
    "pub_rec_bankruptcies",
    "tax_liens",
    "tot_hi_cred_lim",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_il_high_credit_limit",
]
NUMERIC_FEATURES = [
    "loan_amnt",
    "term_months",
    "annual_inc",
    "loan_to_income",
    "dti",
    "emp_length_years",
    "fico_range_low",
    "credit_history_months",
    *BUREAU_NUMERIC_FEATURES,
]
CATEGORICAL_FEATURES = ["home_ownership", "verification_status", "purpose"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Kept next to the features for the benchmark and for reporting only (never features; the guard rejects them).
BENCHMARK_COLUMNS = ["grade", "sub_grade", "int_rate"]
META_COLUMNS = ["issue_date", "issue_year", "loan_status", "is_finished", TARGET_COLUMN, "old_credit_policy"]
PREPARED_COLUMNS = META_COLUMNS + BENCHMARK_COLUMNS + FEATURES

# Raw columns to read, with the dtype each one is read as.
_RAW_NUMERIC = ["loan_amnt", "annual_inc", "dti", "fico_range_low"] + [
    column for column in BUREAU_NUMERIC_FEATURES if column != "revol_util"
]
_RAW_PERCENT = ["revol_util", "int_rate"]  # may be "13.5%" strings in some versions of the file
_RAW_TEXT = [
    "loan_status",
    "issue_d",
    "application_type",
    "term",
    "emp_length",
    "earliest_cr_line",
    "home_ownership",
    "verification_status",
    "purpose",
    "grade",
    "sub_grade",
]
REQUIRED_RAW_COLUMNS = _RAW_TEXT + _RAW_PERCENT + _RAW_NUMERIC
# Read only to measure their missing share per issue year (they are never features).
AUDIT_ONLY_COLUMNS = list(LATE_INTRODUCED_COLUMNS)
# Raw feature columns whose missing share per issue year is reported (alongside AUDIT_ONLY_COLUMNS).
FEATURE_AUDIT_COLUMNS = ["emp_length", "revol_util", *_RAW_NUMERIC]


@dataclass
class LendingClubData:
    """Prepared Lending Club loans plus a data-quality report.

    loans: one row per individual-application loan (finished and unfinished), columns PREPARED_COLUMNS.
    quality: counts for every filtering step, per-vintage outcome counts and missing shares (JSON-ready).
    """

    loans: pd.DataFrame
    quality: dict[str, Any]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_month(values: pd.Series) -> pd.Series:
    """Parse "Dec-2015" style strings to timestamps (first day of the month). Unparseable values become NaT."""
    return pd.to_datetime(values, format=MONTH_FORMAT, errors="coerce")


def parse_term_months(values: pd.Series) -> pd.Series:
    """ " 36 months" -> 36.0. Missing or unparseable values become NaN."""
    digits = values.astype("str").str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(digits, errors="coerce").astype(float)


def parse_emp_length(values: pd.Series) -> pd.Series:
    """Employment length in years: "< 1 year" -> 0, "1 year" -> 1, "10+ years" -> 10, missing or "n/a" -> NaN."""
    text = values.astype("str").str.strip()
    years = pd.to_numeric(text.str.extract(r"(\d+)", expand=False), errors="coerce").astype(float)
    years = years.where(~text.str.startswith("<"), 0.0)
    return years


def parse_percent(values: pd.Series) -> pd.Series:
    """Numbers that may be written as "13.5%" or 13.5 -> 13.5. Blank or unparseable values become NaN."""
    text = values.astype("str").str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(text, errors="coerce").astype(float)


def months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Whole calendar months from start to end (both datetime). NaN if either is missing."""
    months = (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)
    return months.astype(float)


def map_loan_status(status: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (is_bad, is_finished) for each loan_status value.

    is_bad is 0.0 for good, 1.0 for bad and NaN for unfinished loans. Unknown statuses raise ValueError.
    """
    known = set(GOOD_STATUSES) | set(BAD_STATUSES) | set(UNFINISHED_STATUSES)
    unknown = sorted(set(status.dropna().unique()) - known)
    if unknown or status.isna().any():
        problems = unknown + (["<missing>"] if status.isna().any() else [])
        raise ValueError(f"unknown loan_status values (add them to the documented mapping first): {problems}")
    is_bad = pd.Series(np.nan, index=status.index)
    is_bad[status.isin(GOOD_STATUSES)] = 0.0
    is_bad[status.isin(BAD_STATUSES)] = 1.0
    is_finished = ~status.isin(UNFINISHED_STATUSES)
    return is_bad, is_finished


# ---------------------------------------------------------------------------
# Preparing one raw frame (or one chunk of the file)
# ---------------------------------------------------------------------------


def check_required_columns(columns: list[str] | pd.Index) -> None:
    """Raise ValueError listing every required raw column that is missing."""
    missing = [column for column in REQUIRED_RAW_COLUMNS if column not in set(columns)]
    if missing:
        raise ValueError(f"Lending Club file is missing required columns: {missing}")


def _has_status_and_issue_date(raw: pd.DataFrame) -> pd.Series:
    """Rows with both loan_status and issue_d. The Kaggle file ends with a few summary rows that have neither."""
    return raw["loan_status"].notna() & raw["issue_d"].notna()


def _is_individual(raw: pd.DataFrame) -> pd.Series:
    """True for individual applications (the population this model covers)."""
    return raw["application_type"].fillna(INDIVIDUAL_APPLICATION).astype("str").str.strip() == INDIVIDUAL_APPLICATION


def prepare_loans(raw: pd.DataFrame, keep_unfinished: bool = True) -> pd.DataFrame:
    """Turn raw Lending Club rows into the model table (columns PREPARED_COLUMNS).

    Steps: drop rows without loan_status or issue_d, keep individual applications, map the target, parse
    dates and text fields, derive features, and keep only the allow-listed columns. Every other raw column,
    including all post-origination ones, is dropped here. With keep_unfinished=False only finished loans
    are returned.
    """
    check_required_columns(raw.columns)
    rows = raw.loc[_has_status_and_issue_date(raw) & _is_individual(raw)]
    status = rows["loan_status"].astype("str").str.strip()
    is_bad, is_finished = map_loan_status(status)

    issue_date = parse_month(rows["issue_d"])
    if issue_date.isna().any():
        bad_values = rows.loc[issue_date.isna(), "issue_d"].unique()[:5].tolist()
        raise ValueError(f"issue_d could not be parsed for {int(issue_date.isna().sum())} rows, e.g. {bad_values}")

    out = pd.DataFrame(index=rows.index)
    out["issue_date"] = issue_date
    out["issue_year"] = issue_date.dt.year.astype("int64")
    out["loan_status"] = status
    out["is_finished"] = is_finished
    out[TARGET_COLUMN] = is_bad
    out["old_credit_policy"] = status.str.startswith(OLD_POLICY_PREFIX)
    out["grade"] = rows["grade"].astype("str")
    out["sub_grade"] = rows["sub_grade"].astype("str")
    out["int_rate"] = parse_percent(rows["int_rate"])

    for column in ["loan_amnt", "annual_inc", "dti", "fico_range_low"]:
        out[column] = rows[column].astype(float)
    out["term_months"] = parse_term_months(rows["term"])
    out["emp_length_years"] = parse_emp_length(rows["emp_length"])
    history = months_between(parse_month(rows["earliest_cr_line"]), issue_date)
    out["credit_history_months"] = history.where(history >= 0)
    income = out["annual_inc"].where(out["annual_inc"] > 0)
    out["loan_to_income"] = out["loan_amnt"] / income
    for column in BUREAU_NUMERIC_FEATURES:
        if column == "revol_util":
            out[column] = parse_percent(rows[column])
        else:
            out[column] = rows[column].astype(float)

    home = rows["home_ownership"].astype("str").str.strip().str.upper()
    out["home_ownership"] = home.replace(HOME_OWNERSHIP_GROUPS)
    out["verification_status"] = rows["verification_status"].astype("str").str.strip()
    out["purpose"] = rows["purpose"].astype("str").str.strip()
    for column in CATEGORICAL_FEATURES:
        out[column] = out[column].where(rows[column].notna(), MISSING_CATEGORY)

    assert_no_leakage(FEATURES)  # self-check: the allow-list itself must never contain a banned column
    out = out[PREPARED_COLUMNS]
    if not keep_unfinished:
        out = out.loc[out["is_finished"]]
    return out.reset_index(drop=True)


def finished_loans(loans: pd.DataFrame) -> pd.DataFrame:
    """Only loans with a final outcome (Fully Paid, Charged Off, Default), with is_bad as an integer 0/1."""
    finished = loans.loc[loans["is_finished"]].copy()
    finished[TARGET_COLUMN] = finished[TARGET_COLUMN].astype("int64")
    return finished.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Data-quality counts (computed per chunk, then added up)
# ---------------------------------------------------------------------------


def chunk_counts(raw: pd.DataFrame, audit_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Counts used by the data-quality report, for one raw chunk.

    Returns (status_counts, missing_counts):
    - status_counts: rows per issue_year, loan_status and application_type (missing values shown as "<missing>"
      and issue_year -1 when issue_d is missing or unparseable).
    - missing_counts: for individual applications, rows and missing values per issue_year and audited column.
    """
    issue_year = parse_month(raw["issue_d"]).dt.year.fillna(-1).astype("int64")
    labels = pd.DataFrame(
        {
            "issue_year": issue_year,
            "loan_status": raw["loan_status"].astype("str").str.strip().fillna("<missing>"),
            "application_type": raw["application_type"].astype("str").str.strip().fillna("<missing>"),
        }
    )
    status_counts = labels.value_counts().rename("n_rows").reset_index()
    keep = _has_status_and_issue_date(raw) & _is_individual(raw)
    present = [column for column in audit_columns if column in raw.columns]
    missing = raw.loc[keep, present].isna()
    missing["issue_year"] = issue_year[keep]
    grouped = missing.groupby("issue_year")
    missing_counts = grouped.sum().stack().rename("n_missing").reset_index()
    missing_counts.columns = ["issue_year", "column", "n_missing"]
    sizes = grouped.size().rename("n_rows").reset_index()
    missing_counts = missing_counts.merge(sizes, on="issue_year")
    return status_counts, missing_counts


def vintage_table(loans: pd.DataFrame) -> list[dict[str, Any]]:
    """Per issue year: loans, finished share, bad rate among finished loans, and unfinished loans by status.

    Recent vintages have had little time to finish, so their finished loans are mostly early payoffs and
    early charge-offs. This table makes that visible instead of hiding it in one overall bad rate.
    """
    rows = []
    for year, group in loans.groupby("issue_year"):
        finished = group.loc[group["is_finished"]]
        n_finished = int(len(finished))
        n_bad = int(finished[TARGET_COLUMN].sum()) if n_finished else 0
        unfinished = group.loc[~group["is_finished"], "loan_status"].value_counts().sort_index()
        rows.append(
            {
                "issue_year": int(year),
                "n_loans": int(len(group)),
                "n_finished": n_finished,
                "share_finished": n_finished / len(group),
                "n_bad": n_bad,
                "bad_rate_finished": (n_bad / n_finished) if n_finished else None,
                "n_old_credit_policy": int(group["old_credit_policy"].sum()),
                "n_unfinished_by_status": {str(status): int(count) for status, count in unfinished.items()},
            }
        )
    return rows


def missing_share_by_year(missing_counts: pd.DataFrame) -> dict[str, dict[str, float]]:
    """{column: {issue_year: share missing}} from the summed missing counts."""
    totals = missing_counts.groupby(["column", "issue_year"])[["n_missing", "n_rows"]].sum()
    shares: dict[str, dict[str, float]] = {}
    for (column, year), row in totals.iterrows():
        shares.setdefault(str(column), {})[str(int(year))] = float(row["n_missing"] / row["n_rows"])
    return shares


def quality_report(
    status_counts: pd.DataFrame,
    missing_counts: pd.DataFrame,
    loans: pd.DataFrame,
    file_columns: list[str],
    source: dict[str, Any],
) -> dict[str, Any]:
    """Data-quality report: every count behind the filtering steps, computed from the data."""
    totals = status_counts.groupby(["loan_status", "application_type", "issue_year"])["n_rows"].sum().reset_index()
    no_status_or_date = (totals["loan_status"] == "<missing>") | (totals["issue_year"] == -1)
    with_status = totals.loc[~no_status_or_date]
    # A missing application_type is treated as individual, the same rule prepare_loans uses.
    individual = with_status["application_type"].isin([INDIVIDUAL_APPLICATION, "<missing>"])
    status_of_individuals = with_status.loc[individual].groupby("loan_status")["n_rows"].sum().sort_index()
    finished = loans.loc[loans["is_finished"]]
    n_finished = int(len(finished))
    return {
        "source": source,
        "n_columns_in_file": len(file_columns),
        "undocumented_columns_in_file": undocumented_columns(file_columns),
        "documented_columns_missing_from_file": [c for c in DOCUMENTED_COLUMNS if c not in set(file_columns)],
        "n_rows_in_file": int(totals["n_rows"].sum()),
        "n_rows_without_status_or_issue_date": int(totals.loc[no_status_or_date, "n_rows"].sum()),
        "n_joint_applications_excluded": int(with_status.loc[~individual, "n_rows"].sum()),
        "joint_applications_by_issue_year": {
            str(int(year)): int(count)
            for year, count in with_status.loc[~individual].groupby("issue_year")["n_rows"].sum().items()
        },
        "n_individual_loans": int(len(loans)),
        "status_counts_individual": {str(k): int(v) for k, v in status_of_individuals.items()},
        "n_finished": n_finished,
        "n_unfinished_excluded_from_modelling": int(len(loans) - n_finished),
        "n_bad": int(finished[TARGET_COLUMN].sum()),
        "bad_rate_finished": float(finished[TARGET_COLUMN].mean()) if n_finished else None,
        "n_old_credit_policy_finished": int(finished["old_credit_policy"].sum()),
        "first_issue_month": str(loans["issue_date"].min().date()) if len(loans) else None,
        "last_issue_month": str(loans["issue_date"].max().date()) if len(loans) else None,
        "vintages": vintage_table(loans),
        "missing_share_by_issue_year": missing_share_by_year(missing_counts),
        "target_mapping": {"good_0": list(GOOD_STATUSES), "bad_1": list(BAD_STATUSES)},
        "excluded_unfinished_statuses": list(UNFINISHED_STATUSES),
        "home_ownership_grouping": HOME_OWNERSHIP_GROUPS,
    }


# ---------------------------------------------------------------------------
# Reading the zip lazily
# ---------------------------------------------------------------------------


def zip_status(zip_path: Path = DEFAULT_ZIP_PATH) -> tuple[bool, str]:
    """(ready, message): is the Kaggle zip present, complete and does it contain the accepted-loans file?"""
    zip_path = Path(zip_path)
    if not zip_path.exists():
        return False, f"{zip_path} does not exist"
    if zip_path.with_name(zip_path.name + ".kaggle-partial").exists():
        return False, f"{zip_path} is a partial Kaggle download"
    if not zipfile.is_zipfile(zip_path):
        return False, f"{zip_path} is not a readable zip file"
    try:
        member = find_accepted_member(zip_path)
    except FileNotFoundError as error:
        return False, str(error)
    return True, f"found {member} in {zip_path}"


def find_accepted_member(zip_path: Path) -> str:
    """Name of the accepted-loans CSV (gzipped or not) inside the Kaggle zip. Case-insensitive."""
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if name.lower().endswith(ACCEPTED_FILE_NAMES):
                return name
    raise FileNotFoundError(f"no accepted_2007_to_2018Q4 CSV inside {zip_path}")


def _read_options(member: str) -> dict[str, Any]:
    """Compression setting for pandas: the Kaggle zip stores the CSV gzipped."""
    return {"compression": "gzip" if member.lower().endswith(".gz") else None}


def read_file_columns(zip_path: Path, member: str) -> list[str]:
    """Header of the accepted-loans CSV (reads one line, not the whole file)."""
    with zipfile.ZipFile(zip_path) as archive, archive.open(member) as handle:
        return pd.read_csv(handle, nrows=0, **_read_options(member)).columns.tolist()


def raw_dtypes(columns: list[str]) -> dict[str, str]:
    """dtype for every raw column that is read: numbers as float64, text and percent fields as strings."""
    dtypes = {}
    for column in columns:
        dtypes[column] = "float64" if column in _RAW_NUMERIC or column in AUDIT_ONLY_COLUMNS else "str"
    return dtypes


def iter_raw_chunks(zip_path: Path, member: str, columns: list[str], chunk_rows: int) -> Iterator[pd.DataFrame]:
    """Yield the selected columns of the accepted-loans CSV, chunk_rows rows at a time, straight from the zip."""
    with zipfile.ZipFile(zip_path) as archive, archive.open(member) as handle:
        reader = pd.read_csv(
            handle,
            usecols=columns,
            dtype=raw_dtypes(columns),
            chunksize=chunk_rows,
            **_read_options(member),
        )
        yield from reader


def load_lending_club(zip_path: Path = DEFAULT_ZIP_PATH, chunk_rows: int = DEFAULT_CHUNK_ROWS) -> LendingClubData:
    """Read the accepted-loans file from the Kaggle zip and prepare it (see the module docstring)."""
    zip_path = Path(zip_path)
    member = find_accepted_member(zip_path)
    file_columns = read_file_columns(zip_path, member)
    check_required_columns(file_columns)
    audit_columns = [column for column in AUDIT_ONLY_COLUMNS if column in file_columns]
    columns = REQUIRED_RAW_COLUMNS + audit_columns

    prepared, status_parts, missing_parts = [], [], []
    for chunk in iter_raw_chunks(zip_path, member, columns, chunk_rows):
        status_counts, missing_counts = chunk_counts(chunk, FEATURE_AUDIT_COLUMNS + audit_columns)
        status_parts.append(status_counts)
        missing_parts.append(missing_counts)
        prepared.append(prepare_loans(chunk, keep_unfinished=True))

    loans = pd.concat(prepared, ignore_index=True)
    source = {"zip": zip_path.name, "member": member, "chunk_rows": chunk_rows}
    quality = quality_report(
        pd.concat(status_parts, ignore_index=True),
        pd.concat(missing_parts, ignore_index=True),
        loans,
        file_columns,
        source,
    )
    return LendingClubData(loans=loans, quality=quality)


# ---------------------------------------------------------------------------
# SYNTHETIC Lending Club-shaped data (tests only; not real loans)
# ---------------------------------------------------------------------------

SYNTHETIC_NOTE = "SYNTHETIC data shaped like the Lending Club accepted-loans file; for tests only."
_SYNTHETIC_UNFINISHED_SHARE = {2013: 0.03, 2014: 0.08, 2015: 0.2, 2016: 0.35, 2017: 0.55, 2018: 0.75}
_PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase", "car"]
_SYNTHETIC_EMPTY_IN_FIRST_YEAR = ["mo_sin_old_il_acct", "num_il_tl", "total_il_high_credit_limit"]


def _month_strings(rng: np.random.Generator, years: np.ndarray) -> np.ndarray:
    """Random "Mon-YYYY" strings in the given years."""
    months = rng.integers(1, 13, size=years.shape[0])
    stamps = pd.to_datetime({"year": years, "month": months, "day": 1})
    return stamps.dt.strftime(MONTH_FORMAT).to_numpy()


def synthetic_accepted_loans(
    n_rows: int = 1200, seed: int = 42, years: tuple[int, ...] = (2013, 2014, 2015, 2016, 2017, 2018)
) -> pd.DataFrame:
    """SYNTHETIC raw frame with every documented Lending Club column, plus two footer rows like the real file.

    Default risk depends on FICO, DTI, term, inquiries, utilisation and loan-to-income, so models can learn
    something. Post-origination columns (recoveries, total_pymnt, last_fico_range_low, ...) are built from
    the outcome, so they would reveal it almost perfectly if they ever reached a model. The frame includes
    unfinished statuses, Default, "Does not meet the credit policy" statuses, joint applications and percent
    strings, to exercise every branch of the loader.
    """
    rng = np.random.default_rng(seed)
    year = rng.choice(np.array(years), size=n_rows)
    fico_low = np.clip(np.round(rng.normal(695, 30, n_rows) / 5) * 5, 660, 845)
    dti = np.clip(rng.normal(18, 8, n_rows), 0, 45).round(2)
    term = rng.choice([36, 60], size=n_rows, p=[0.72, 0.28])
    inquiries = rng.poisson(0.7, n_rows).astype(float)
    revol_util = np.clip(rng.normal(52, 22, n_rows), 0, 120).round(1)
    annual_inc = np.round(np.exp(rng.normal(11.1, 0.5, n_rows)), -2)
    loan_amnt = np.round(rng.uniform(1000, 40000, n_rows), -2)
    logit = (
        -1.6
        - 0.03 * (fico_low - 695)
        + 0.05 * (dti - 18)
        + 0.6 * (term == 60)
        + 0.25 * inquiries
        + 0.01 * (revol_util - 52)
        + 1.5 * (loan_amnt / annual_inc - 0.2)
    )
    bad = rng.random(n_rows) < 1 / (1 + np.exp(-logit))
    unfinished_share = np.array([_SYNTHETIC_UNFINISHED_SHARE.get(int(y), 0.1) for y in year])
    unfinished = rng.random(n_rows) < unfinished_share

    status = np.where(bad, "Charged Off", "Fully Paid").astype(object)
    status[bad & (rng.random(n_rows) < 0.05)] = "Default"
    old_policy = (year == min(years)) & (rng.random(n_rows) < 0.1)
    status[old_policy & ~bad] = OLD_POLICY_PREFIX + " Status:Fully Paid"
    status[old_policy & bad] = OLD_POLICY_PREFIX + " Status:Charged Off"
    status[unfinished] = rng.choice(list(UNFINISHED_STATUSES), size=int(unfinished.sum()), p=[0.85, 0.05, 0.04, 0.06])

    grade_index = np.clip(((logit + 3.5) * 1.3).astype(int), 0, 6)
    grade = np.array(list("ABCDEFG"))[grade_index]
    int_rate = np.round(6 + 3 * grade_index + rng.uniform(0, 2, n_rows), 2)
    joint = (year >= 2016) & (rng.random(n_rows) < 0.06)

    def count(mean: float) -> np.ndarray:
        """Poisson counts as floats."""
        return rng.poisson(mean, n_rows).astype(float)

    def sometimes_missing(values: np.ndarray, share: float) -> np.ndarray:
        """Copy of values with about `share` of them set to NaN."""
        values = values.astype(float)
        values[rng.random(n_rows) < share] = np.nan
        return values

    values: dict[str, Any] = {column: np.nan for column in DOCUMENTED_COLUMNS}
    for column in BUREAU_NUMERIC_FEATURES:
        values[column] = sometimes_missing(count(5), 0.05)
    # A few bureau fields are empty for the whole earliest year, so tests cover a vintage in which a feature
    # has no values at all (stability checks and missing-value handling must cope with that).
    for column in _SYNTHETIC_EMPTY_IN_FIRST_YEAR:
        values[column][year == min(years)] = np.nan
    values.update(
        {
            "id": np.arange(1, n_rows + 1).astype(str),
            "loan_amnt": loan_amnt,
            "funded_amnt": loan_amnt,
            "funded_amnt_inv": loan_amnt - 25 * rng.integers(0, 3, n_rows),
            "term": [f" {months} months" for months in term],
            "int_rate": [f"{rate}%" for rate in int_rate],
            "installment": np.round(loan_amnt / term * 1.15, 2),
            "grade": grade,
            "sub_grade": [
                f"{letter}{number}" for letter, number in zip(grade, rng.integers(1, 6, n_rows), strict=True)
            ],
            "emp_title": rng.choice(["Teacher", "Nurse", "Manager", "Driver"], size=n_rows),
            "emp_length": rng.choice(["< 1 year", "1 year", "3 years", "7 years", "10+ years", "n/a"], size=n_rows),
            "home_ownership": rng.choice(["MORTGAGE", "RENT", "OWN", "ANY"], size=n_rows, p=[0.48, 0.4, 0.11, 0.01]),
            "annual_inc": annual_inc,
            "verification_status": rng.choice(["Not Verified", "Source Verified", "Verified"], size=n_rows),
            "issue_d": _month_strings(rng, year),
            "loan_status": status,
            "pymnt_plan": "n",
            "purpose": rng.choice(_PURPOSES, size=n_rows),
            "title": "Debt consolidation",
            "zip_code": rng.choice(["100xx", "945xx", "606xx"], size=n_rows),
            "addr_state": rng.choice(["NY", "CA", "IL"], size=n_rows),
            "dti": dti,
            "earliest_cr_line": _month_strings(rng, year - rng.integers(3, 30, n_rows)),
            "fico_range_low": fico_low,
            "fico_range_high": fico_low + 4,
            "inq_last_6mths": inquiries,
            "mths_since_last_delinq": sometimes_missing(rng.integers(1, 120, n_rows), 0.5),
            "mths_since_last_record": sometimes_missing(rng.integers(1, 120, n_rows), 0.85),
            "open_acc": count(11),
            "pub_rec": count(0.2),
            "revol_bal": np.round(rng.gamma(2, 8000, n_rows)),
            "revol_util": [f"{value}%" for value in revol_util],
            "total_acc": count(24),
            "initial_list_status": rng.choice(["w", "f"], size=n_rows),
            "application_type": np.where(joint, "Joint App", INDIVIDUAL_APPLICATION),
            "annual_inc_joint": np.where(joint, annual_inc * 1.6, np.nan),
            "policy_code": 1.0,
            "disbursement_method": "Cash",
            # Outcome-revealing post-origination columns: a model that saw these would look near-perfect.
            "total_pymnt": np.round(np.where(bad, 0.45, 1.12) * loan_amnt, 2),
            "total_pymnt_inv": np.round(np.where(bad, 0.45, 1.12) * loan_amnt, 2),
            "total_rec_prncp": np.round(np.where(bad, 0.35, 1.0) * loan_amnt, 2),
            "total_rec_int": np.round(0.1 * loan_amnt, 2),
            "total_rec_late_fee": np.where(bad, 15.0, 0.0),
            "recoveries": np.where(bad, np.round(0.08 * loan_amnt, 2), 0.0),
            "collection_recovery_fee": np.where(bad, np.round(0.01 * loan_amnt, 2), 0.0),
            "out_prncp": np.where(unfinished, np.round(0.5 * loan_amnt, 2), 0.0),
            "last_pymnt_d": _month_strings(rng, year + np.where(bad, 1, 3)),
            "last_pymnt_amnt": np.where(bad, 50.0, 900.0),
            "last_fico_range_low": np.where(bad, 540.0, fico_low + 10),
            "last_fico_range_high": np.where(bad, 544.0, fico_low + 14),
            "hardship_flag": np.where(bad & (rng.random(n_rows) < 0.2), "Y", "N"),
            "debt_settlement_flag": np.where(bad & (rng.random(n_rows) < 0.3), "Y", "N"),
        }
    )
    frame = pd.DataFrame(values, index=range(n_rows))[list(DOCUMENTED_COLUMNS)]
    footer = pd.DataFrame(
        {"id": ["Total amount funded in policy code 1: 123", "Total amount funded in policy code 2: 0"]}
    )
    return pd.concat([frame, footer], ignore_index=True)


def write_lending_club_zip(
    raw: pd.DataFrame, zip_path: Path, member: str = "accepted_2007_to_2018Q4.csv/accepted_2007_to_2018Q4.csv.gz"
) -> Path:
    """Write a raw frame as a gzipped CSV inside a zip, laid out like the Kaggle download (for tests)."""
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    csv_bytes = raw.to_csv(index=False).encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gz:
        gz.write(csv_bytes)
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(member, buffer.getvalue())
    return zip_path
