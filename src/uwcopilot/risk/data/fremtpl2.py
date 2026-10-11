"""Load, clean and split the French motor third-party liability data (freMTPL2, OpenML 41214 and 41215).

Two files, both downloaded by scripts/download_data.py into data/raw/fremtpl2/:

- freMTPL2freq.csv: one row per policy (IDpol) with the number of claims (ClaimNb), the exposure in years
  (Exposure) and nine rating features.
- freMTPL2sev.csv: one row per claim with its amount (IDpol, ClaimAmount).

Known quirks, and what this module does about each one (every count is computed and returned in a
data-quality report, never typed by hand):

- IDpol is a float in the frequency file ("1.0") but an integer in the severity file. It is converted to
  int64 after checking that every value is a whole number, so the two files join correctly.
- VehGas values carry literal single quotes ("'Regular'"). Quotes and spaces are stripped from every
  text column.
- Exposure above 1 year cannot happen within one policy-year record, so it is capped at 1.
- A few policies report 5 to 16 claims, some within a few weeks of exposure. Following the scikit-learn
  example "Tweedie regression on insurance claims", which treats these as likely data errors, ClaimNb is
  capped at 4. The recorded values are kept in ClaimNbRecorded and ExposureRecorded.
- Some policies with ClaimNb > 0 have no rows in the severity file. Almost all of them sit in one block of
  low IDpol values in which every policy has a claim, which a random portfolio sample cannot produce. As in
  the scikit-learn example, a claim with no recorded amount is not counted as a claim for modelling
  (ClaimNb is set to 0; the capped count is kept in ClaimNbCapped and the file value in ClaimNbRecorded),
  so the frequency and severity models describe the same claims, except on the few policies whose number
  of claim amounts differs from the capped count (counted in the report). The report counts every policy
  affected and shows where they sit.
- Some severity rows belong to an IDpol that is not in the frequency file. They have no exposure and no
  features, so they are dropped and counted.

This module also generates SYNTHETIC data shaped like freMTPL2 (synthetic_fremtpl2_frames), used by the
tests so CI never needs the real files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from uwcopilot.common.paths import RAW_DATA_DIR

DEFAULT_DATA_DIR = RAW_DATA_DIR / "fremtpl2"
FREQUENCY_FILE = "freMTPL2freq.csv"
SEVERITY_FILE = "freMTPL2sev.csv"

CLAIM_COUNT_CAP = 4
EXPOSURE_CAP = 1.0
SEVERITY_CAP_QUANTILE = 0.995
SPLIT_SEED = 42
SPLIT_FRACTIONS = {"train": 0.6, "validation": 0.2, "test": 0.2}

NUMERIC_FEATURES = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
CATEGORICAL_FEATURES = ["Area", "VehBrand", "VehGas", "Region"]
RATING_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
FREQUENCY_COLUMNS = ["IDpol", "ClaimNb", "Exposure", *RATING_FEATURES]
SEVERITY_COLUMNS = ["IDpol", "ClaimAmount"]
MAX_LISTED_ROWS = 20  # longest list of individual policies copied into the data-quality report

# Fixed IDpol bands used only to describe where policies without claim amounts sit (not used by any model).
IDPOL_BAND_EDGES = [0, 10_000, 30_000, 100_000, 1_000_000, np.inf]
IDPOL_BAND_LABELS = ["1-10,000", "10,001-30,000", "30,001-100,000", "100,001-1,000,000", "above 1,000,000"]


@dataclass
class Fremtpl2Data:
    """Cleaned freMTPL2 data.

    policies: one row per policy. ClaimNb is the modelling target: capped at 4 and set to 0 when the policy has
        no claim amounts. ClaimNbCapped is the capped count before that step; ClaimNbRecorded and
        ExposureRecorded keep the file's values; Exposure is capped at 1. n_severity_rows and
        claim_amount_total come from the severity file (0 when none).
    claims: one row per claim amount, only for IDpol values present in policies.
    quality: counts describing every cleaning step (written to results/risk_metrics.json).
    """

    policies: pd.DataFrame
    claims: pd.DataFrame
    quality: dict[str, Any]


def _check_columns(df: pd.DataFrame, expected: list[str], name: str) -> None:
    """Raise a clear error if a required column is missing."""
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _check_no_missing_values(df: pd.DataFrame, columns: list[str], name: str) -> None:
    """Raise if any required value is missing. Missing values are never filled in silently."""
    counts = df[columns].isna().sum()
    if counts.any():
        raise ValueError(f"{name} has missing values: {counts[counts > 0].to_dict()}")


def _to_integer_ids(values: pd.Series, name: str) -> pd.Series:
    """Convert IDpol to int64, refusing values that are not whole numbers (for example 1.5)."""
    as_float = values.astype(float)
    if not np.all(as_float == np.round(as_float)):
        raise ValueError(f"{name}: IDpol has non-integer values")
    return as_float.astype("int64")


def clean_frequency(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the policy table and return it with a report of what was changed.

    Steps: IDpol to int, strip quotes from text columns, drop rows with non-positive exposure (none in the
    real file, but a zero exposure would break the log offset), cap Exposure at 1 and ClaimNb at 4.
    """
    _check_columns(raw, FREQUENCY_COLUMNS, FREQUENCY_FILE)
    _check_no_missing_values(raw, FREQUENCY_COLUMNS, FREQUENCY_FILE)
    df = raw[FREQUENCY_COLUMNS].copy()
    report: dict[str, Any] = {"n_rows_in_file": int(len(df))}

    report["idpol_dtype_in_file"] = str(raw["IDpol"].dtype)
    df["IDpol"] = _to_integer_ids(df["IDpol"], FREQUENCY_FILE)
    n_duplicate_ids = int(df["IDpol"].duplicated().sum())
    if n_duplicate_ids:
        raise ValueError(f"{FREQUENCY_FILE}: {n_duplicate_ids} duplicate IDpol values")

    values_with_quotes = {}
    for column in CATEGORICAL_FEATURES:
        original = df[column].astype(str)
        cleaned = original.str.strip().str.strip("'").str.strip()
        values_with_quotes[column] = int((cleaned != original).sum())
        df[column] = cleaned
    report["text_values_with_quotes_or_spaces_stripped"] = values_with_quotes
    report["vehgas_values_after_cleaning"] = sorted(df["VehGas"].unique().tolist())

    for column in ["ClaimNb", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]:
        df[column] = df[column].astype("int64")
    df["Exposure"] = df["Exposure"].astype(float)

    non_positive_exposure = df["Exposure"] <= 0
    report["n_rows_dropped_non_positive_exposure"] = int(non_positive_exposure.sum())
    df = df.loc[~non_positive_exposure].copy()

    df["ClaimNbRecorded"] = df["ClaimNb"]
    df["ExposureRecorded"] = df["Exposure"]
    above_claim_cap = df["ClaimNb"] > CLAIM_COUNT_CAP
    above_exposure_cap = df["Exposure"] > EXPOSURE_CAP
    report["claim_count_cap"] = CLAIM_COUNT_CAP
    report["n_policies_claimnb_above_cap"] = int(above_claim_cap.sum())
    report["max_claimnb_recorded"] = int(df["ClaimNb"].max())
    report["n_claims_removed_by_cap"] = int((df["ClaimNb"] - df["ClaimNb"].clip(upper=CLAIM_COUNT_CAP)).sum())
    report["policies_above_claim_cap"] = (
        df.loc[above_claim_cap, ["IDpol", "ClaimNb", "Exposure"]].head(MAX_LISTED_ROWS).to_dict(orient="records")
    )
    report["exposure_cap_years"] = EXPOSURE_CAP
    report["n_policies_exposure_above_cap"] = int(above_exposure_cap.sum())
    report["max_exposure_recorded"] = float(df["Exposure"].max())
    df["ClaimNb"] = df["ClaimNb"].clip(upper=CLAIM_COUNT_CAP)
    df["Exposure"] = df["Exposure"].clip(upper=EXPOSURE_CAP)

    report["n_policies"] = int(len(df))
    report["n_policies_with_claims"] = int((df["ClaimNb"] > 0).sum())
    report["n_claims_after_cap"] = int(df["ClaimNb"].sum())
    report["total_exposure_years"] = float(df["Exposure"].sum())
    return df.reset_index(drop=True), report


def clean_severity(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the claim-amount table: IDpol to int, amounts to float, drop non-positive amounts (counted).

    A Gamma severity model needs strictly positive amounts; the real file has none at or below zero.
    """
    _check_columns(raw, SEVERITY_COLUMNS, SEVERITY_FILE)
    _check_no_missing_values(raw, SEVERITY_COLUMNS, SEVERITY_FILE)
    df = raw[SEVERITY_COLUMNS].copy()
    df["IDpol"] = _to_integer_ids(df["IDpol"], SEVERITY_FILE)
    df["ClaimAmount"] = df["ClaimAmount"].astype(float)
    non_positive = df["ClaimAmount"] <= 0
    report = {
        "n_rows_in_file": int(len(df)),
        "n_rows_dropped_non_positive_amount": int(non_positive.sum()),
    }
    return df.loc[~non_positive].reset_index(drop=True), report


def link_severity(policies: pd.DataFrame, claims: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Join claim amounts to policies and count every way the two files disagree.

    Returns (policies with n_severity_rows and claim_amount_total added, claims kept, report).
    claim_amount_total is the uncapped sum; the large-loss cap is applied later, from training data only.

    The number of amount rows is compared twice: with the file's ClaimNb (ClaimNbRecorded), and with the
    capped ClaimNb that the frequency model uses for every policy that has amounts. A policy above the
    claim-count cap can match the file but not the model (5 claims, 5 amounts, modelling count 4), so the
    second comparison says exactly where frequency and severity do not describe the same claims.
    """
    known_ids = set(policies["IDpol"])
    in_frequency_file = claims["IDpol"].isin(known_ids)
    orphan_claims = claims.loc[~in_frequency_file]
    kept_claims = claims.loc[in_frequency_file].reset_index(drop=True)

    per_policy = kept_claims.groupby("IDpol").agg(
        n_severity_rows=("ClaimAmount", "size"), claim_amount_total=("ClaimAmount", "sum")
    )
    linked = policies.merge(per_policy, left_on="IDpol", right_index=True, how="left")
    linked["n_severity_rows"] = linked["n_severity_rows"].fillna(0).astype("int64")
    linked["claim_amount_total"] = linked["claim_amount_total"].fillna(0.0)

    has_claims = linked["ClaimNbRecorded"] > 0
    has_amounts = linked["n_severity_rows"] > 0
    without_amounts = has_claims & ~has_amounts
    amounts_without_claims = ~has_claims & has_amounts
    count_mismatch = has_amounts & (linked["n_severity_rows"] != linked["ClaimNbRecorded"])
    modelling_mismatch = has_amounts & (linked["n_severity_rows"] != linked["ClaimNb"])
    rows_above_model = (linked["n_severity_rows"] - linked["ClaimNb"]).clip(lower=0)
    model_claims_above_rows = (linked["ClaimNb"] - linked["n_severity_rows"]).clip(lower=0)

    zero_claim_ids = linked.loc[~has_claims, "IDpol"]
    lowest_zero_claim_id = int(zero_claim_ids.min()) if len(zero_claim_ids) else None
    below_lowest = claimant_only_block_mask(linked, lowest_zero_claim_id)

    report: dict[str, Any] = {
        "n_severity_rows_with_unknown_idpol": int(len(orphan_claims)),
        "n_unknown_idpol_values_in_severity_file": int(orphan_claims["IDpol"].nunique()),
        "claim_amount_dropped_with_unknown_idpol": float(orphan_claims["ClaimAmount"].sum()),
        "n_severity_rows_linked": int(len(kept_claims)),
        "n_policies_with_claim_amounts": int(has_amounts.sum()),
        "n_policies_with_claims_but_no_amounts": int(without_amounts.sum()),
        "n_claims_on_policies_without_amounts": int(linked.loc[without_amounts, "ClaimNb"].sum()),
        "share_of_claim_policies_without_amounts": float(without_amounts.sum() / max(has_claims.sum(), 1)),
        "share_of_claims_without_amounts": float(
            linked.loc[without_amounts, "ClaimNb"].sum() / max(linked.loc[has_claims, "ClaimNb"].sum(), 1)
        ),
        "n_policies_with_amounts_but_zero_claims": int(amounts_without_claims.sum()),
        "n_policies_where_severity_rows_differ_from_claimnb": int(count_mismatch.sum()),
        "policies_where_severity_rows_differ_from_claimnb": linked.loc[
            count_mismatch, ["IDpol", "ClaimNbRecorded", "n_severity_rows"]
        ]
        .head(MAX_LISTED_ROWS)
        .to_dict(orient="records"),
        "n_policies_where_severity_rows_differ_from_modelling_claimnb": int(modelling_mismatch.sum()),
        "n_amount_rows_above_modelling_claimnb": int(rows_above_model[modelling_mismatch].sum()),
        "n_modelling_claims_without_an_amount_row": int(model_claims_above_rows[modelling_mismatch].sum()),
        "policies_where_severity_rows_differ_from_modelling_claimnb": linked.loc[
            modelling_mismatch, ["IDpol", "ClaimNbRecorded", "ClaimNb", "n_severity_rows"]
        ]
        .rename(columns={"ClaimNb": "ClaimNbModelling"})
        .head(MAX_LISTED_ROWS)
        .to_dict(orient="records"),
        "lowest_idpol_with_zero_claims": lowest_zero_claim_id,
        "n_policies_below_that_idpol": int(below_lowest.sum()),
        "n_policies_below_that_idpol_without_amounts": int((below_lowest & without_amounts).sum()),
        "share_of_policies_below_that_idpol_with_amounts": float(
            (below_lowest & has_amounts).sum() / below_lowest.sum()
        )
        if below_lowest.any()
        else None,
        "n_claims_below_that_idpol": int(linked.loc[below_lowest, "ClaimNb"].sum()),
        "share_of_all_claims_below_that_idpol": float(
            linked.loc[below_lowest, "ClaimNb"].sum() / max(linked["ClaimNb"].sum(), 1)
        ),
        "share_of_policies_with_claims_from_that_idpol_on": float(has_claims[~below_lowest].mean())
        if (~below_lowest).any()
        else None,
        "claim_policies_by_idpol_band": _claim_policies_by_idpol_band(linked, has_claims, has_amounts),
    }
    return linked, kept_claims, report


def claimant_only_block_mask(policies: pd.DataFrame, lowest_idpol_with_zero_claims: int | None) -> pd.Series:
    """True for policies whose IDpol is below the first IDpol with zero claims (the claimant-only block).

    In the real file every one of these policies has a claim, which a random portfolio sample with a claim
    rate of about 5% would essentially never produce, and most of them have no claim amounts. Used only for a
    sensitivity check of the test metrics, never to drop data from training.
    """
    if lowest_idpol_with_zero_claims is None:
        return pd.Series(True, index=policies.index)
    return policies["IDpol"] < lowest_idpol_with_zero_claims


def _claim_policies_by_idpol_band(
    linked: pd.DataFrame, has_claims: pd.Series, has_amounts: pd.Series
) -> list[dict[str, Any]]:
    """Count policies with claims, with and without claim amounts, in fixed IDpol bands."""
    bands = pd.cut(linked["IDpol"], bins=IDPOL_BAND_EDGES, labels=IDPOL_BAND_LABELS, right=True)
    rows = []
    for label in IDPOL_BAND_LABELS:
        in_band = bands == label
        rows.append(
            {
                "idpol_band": label,
                "n_policies": int(in_band.sum()),
                "n_policies_with_claims": int((in_band & has_claims).sum()),
                "n_claim_policies_with_amounts": int((in_band & has_claims & has_amounts).sum()),
                "n_claim_policies_without_amounts": int((in_band & has_claims & ~has_amounts).sum()),
            }
        )
    return rows


def load_fremtpl2(data_dir: Path = DEFAULT_DATA_DIR) -> Fremtpl2Data:
    """Read both CSV files from data_dir, clean them, and link claim amounts to policies."""
    data_dir = Path(data_dir)
    frequency_path = data_dir / FREQUENCY_FILE
    severity_path = data_dir / SEVERITY_FILE
    for path in (frequency_path, severity_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `make data` (scripts/download_data.py) first.")
    return prepare_fremtpl2(pd.read_csv(frequency_path), pd.read_csv(severity_path))


def prepare_fremtpl2(frequency_raw: pd.DataFrame, severity_raw: pd.DataFrame) -> Fremtpl2Data:
    """Clean and link already-loaded frequency and severity tables (used by load_fremtpl2 and the tests)."""
    policies, frequency_report = clean_frequency(frequency_raw)
    claims, severity_report = clean_severity(severity_raw)
    linked, kept_claims, link_report = link_severity(policies, claims)
    modelled, target_report = count_only_claims_with_amounts(linked)
    quality = {
        "frequency_file": frequency_report,
        "severity_file": severity_report,
        "linkage": link_report,
        "modelling_target": target_report,
    }
    return Fremtpl2Data(policies=modelled, claims=kept_claims, quality=quality)


def count_only_claims_with_amounts(policies: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Set ClaimNb to 0 for policies whose claims have no recorded amount (the modelling target).

    Same rule as the scikit-learn example "Tweedie regression on insurance claims": a claim without an amount
    is not counted, so frequency x severity describes the same claims as the observed losses. The capped
    count before this step is kept in ClaimNbCapped (used by the sensitivity check that keeps these claims).
    """
    result = policies.copy()
    result["ClaimNbCapped"] = result["ClaimNb"]
    without_amounts = (result["ClaimNb"] > 0) & (result["n_severity_rows"] == 0)
    result.loc[without_amounts, "ClaimNb"] = 0
    report = {
        "rule": "claims with no recorded amount are not counted (ClaimNb set to 0); ClaimNbCapped keeps the count",
        "n_policies_set_to_zero_claims": int(without_amounts.sum()),
        "n_claims_not_counted": int(result.loc[without_amounts, "ClaimNbCapped"].sum()),
        "n_claims_for_modelling": int(result["ClaimNb"].sum()),
        "n_policies_with_claims_for_modelling": int((result["ClaimNb"] > 0).sum()),
        "claims_per_policy_year_for_modelling": float(result["ClaimNb"].sum() / result["Exposure"].sum()),
        "claims_per_policy_year_as_recorded_after_cap": float(result["ClaimNbCapped"].sum() / result["Exposure"].sum()),
    }
    return result, report


def split_policies(
    policies: pd.DataFrame, seed: int = SPLIT_SEED, fractions: dict[str, float] | None = None
) -> dict[str, pd.DataFrame]:
    """Seeded random split into train / validation / test, stratified on "has at least one claim".

    freMTPL2 has no date field, so an out-of-time split is impossible; this is a random split and the test
    set comes from the same period as the training data. Stratifying keeps the share of policies with
    claims (about 4%) the same in every split.
    """
    fractions = fractions or SPLIT_FRACTIONS
    if set(fractions) != {"train", "validation", "test"} or not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError("fractions must have keys train, validation, test and sum to 1")
    has_claim = (policies["ClaimNb"] > 0).astype(int)
    rest, test = train_test_split(policies, test_size=fractions["test"], stratify=has_claim, random_state=seed)
    validation_share_of_rest = fractions["validation"] / (fractions["train"] + fractions["validation"])
    train, validation = train_test_split(
        rest,
        test_size=validation_share_of_rest,
        stratify=(rest["ClaimNb"] > 0).astype(int),
        random_state=seed,
    )
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def split_policies_grouped(
    policies: pd.DataFrame,
    seed: int = SPLIT_SEED,
    fractions: dict[str, float] | None = None,
    group_columns: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Seeded split in which policies with identical rating features always land in the same split.

    Used only for a sensitivity check of the headline (random) split. freMTPL2 has no customer or vehicle
    identifier, so the same insured risk can appear in train and test; grouping on the full rating-feature
    vector makes that impossible. GroupShuffleSplit draws whole groups, so the fractions apply to groups
    and the split sizes in policies are close to, not exactly, the requested fractions; the split is not
    stratified on claims.
    """
    fractions = fractions or SPLIT_FRACTIONS
    if set(fractions) != {"train", "validation", "test"} or not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError("fractions must have keys train, validation, test and sum to 1")
    columns = list(group_columns or RATING_FEATURES)
    groups = policies.groupby(columns, sort=False).ngroup().to_numpy()
    outer = GroupShuffleSplit(n_splits=1, test_size=fractions["test"], random_state=seed)
    rest_index, test_index = next(outer.split(policies, groups=groups))
    rest = policies.iloc[rest_index]
    validation_share_of_rest = fractions["validation"] / (fractions["train"] + fractions["validation"])
    inner = GroupShuffleSplit(n_splits=1, test_size=validation_share_of_rest, random_state=seed)
    train_index, validation_index = next(inner.split(rest, groups=groups[rest_index]))
    return {
        "train": rest.iloc[train_index].reset_index(drop=True),
        "validation": rest.iloc[validation_index].reset_index(drop=True),
        "test": policies.iloc[test_index].reset_index(drop=True),
    }


def share_with_features_seen_in(rows: pd.DataFrame, reference: pd.DataFrame, columns: list[str] | None = None) -> float:
    """Share of rows whose full rating-feature vector (every column equal) also appears in reference."""
    columns = list(columns or RATING_FEATURES)
    if rows.empty:
        raise ValueError("rows is empty")
    seen = reference[columns].drop_duplicates()
    matched = rows[columns].merge(seen, on=columns, how="left", indicator=True)["_merge"] == "both"
    return float(matched.mean())


def split_summary(splits: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    """Size of each split: policies, exposure, claims and the share of policies with a claim."""
    summary = {}
    for name, df in splits.items():
        summary[name] = {
            "n_policies": int(len(df)),
            "total_exposure_years": float(df["Exposure"].sum()),
            "n_claims": int(df["ClaimNb"].sum()),
            "n_claims_recorded_after_cap": int(df["ClaimNbCapped"].sum()) if "ClaimNbCapped" in df else None,
            "n_policies_with_claims": int((df["ClaimNb"] > 0).sum()),
            "share_policies_with_claims": float((df["ClaimNb"] > 0).mean()),
            "n_policies_with_claim_amounts": int((df["n_severity_rows"] > 0).sum()),
        }
    return summary


def severity_cap(claim_amounts: pd.Series | np.ndarray, quantile: float = SEVERITY_CAP_QUANTILE) -> float:
    """Large-loss cap: the given quantile of individual claim amounts (use training claims only)."""
    amounts = np.asarray(claim_amounts, dtype=float)
    if amounts.size == 0:
        raise ValueError("no claim amounts to compute a cap from")
    return float(np.quantile(amounts, quantile))


def cap_summary(claim_amounts: pd.Series | np.ndarray, cap: float) -> dict[str, float | int]:
    """How much a cap changes: number and share of claims above it, and share of the total amount removed."""
    amounts = np.asarray(claim_amounts, dtype=float)
    above = amounts > cap
    total = float(amounts.sum())
    removed = float((amounts[above] - cap).sum())
    return {
        "n_claims": int(amounts.size),
        "n_claims_capped": int(above.sum()),
        "share_of_claims_capped": float(above.mean()) if amounts.size else 0.0,
        "total_amount_uncapped": total,
        "amount_removed_by_cap": removed,
        "share_of_amount_removed_by_cap": removed / total if total > 0 else 0.0,
    }


def severity_table(policies: pd.DataFrame, claims: pd.DataFrame, cap: float) -> pd.DataFrame:
    """One row per policy that has claim amounts, with the capped average amount per claim.

    Each claim is capped at `cap` before averaging. n_claims is the number of claim amounts on the policy
    (the weight for the severity models). Policies without amounts are not included.
    """
    with_amounts = policies.loc[policies["n_severity_rows"] > 0]
    policy_claims = claims.loc[claims["IDpol"].isin(with_amounts["IDpol"])].copy()
    policy_claims["ClaimAmountCapped"] = policy_claims["ClaimAmount"].clip(upper=cap)
    per_policy = policy_claims.groupby("IDpol").agg(
        n_claims=("ClaimAmount", "size"),
        total_amount_uncapped=("ClaimAmount", "sum"),
        total_amount_capped=("ClaimAmountCapped", "sum"),
    )
    table = with_amounts.merge(per_policy, left_on="IDpol", right_index=True, how="inner")
    table["average_amount_capped"] = table["total_amount_capped"] / table["n_claims"]
    return table.reset_index(drop=True)


def observed_capped_loss(policies: pd.DataFrame, claims: pd.DataFrame, cap: float) -> np.ndarray:
    """Total capped claim amount per policy, in the order of `policies` (0 when a policy has no amounts)."""
    capped = claims.assign(ClaimAmountCapped=claims["ClaimAmount"].clip(upper=cap))
    per_policy = capped.groupby("IDpol")["ClaimAmountCapped"].sum()
    return policies["IDpol"].map(per_policy).fillna(0.0).to_numpy(dtype=float)


def synthetic_fremtpl2_frames(n_policies: int = 3000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SYNTHETIC frequency and severity tables shaped like freMTPL2, including its quirks. For tests only.

    Claim counts follow a Poisson process whose rate rises with BonusMalus, falls with driver age, and is
    higher for Area F, so a model has something real to learn. Quirks reproduced on purpose:
    float IDpol, quoted VehGas, exposure above 1, claim counts above 4, policies with claims but no claim
    amounts, claim amounts for an IDpol missing from the policy table, and one policy whose number of claim
    amounts differs from its ClaimNb.
    """
    rng = np.random.default_rng(seed)
    exposure = np.round(rng.uniform(0.05, 1.0, n_policies), 2)
    exposure[:5] = [1.2, 1.5, 2.0, 1.1, 1.3]
    driver_age = rng.integers(18, 90, n_policies)
    bonus_malus = np.clip(np.round(rng.gamma(2.0, 8.0, n_policies)) + 50, 50, 230).astype(int)
    area = rng.choice(list("ABCDEF"), n_policies, p=[0.15, 0.11, 0.28, 0.22, 0.2, 0.04])
    density = np.clip(np.round(rng.lognormal(6.5, 1.5, n_policies)), 1, 27000).astype(int)
    frequency = np.exp(
        -2.4 + 0.015 * (bonus_malus - 50) - 0.01 * (driver_age - 45) + 0.4 * (area == "F") - 0.2 * (area == "A")
    )
    claim_count = rng.poisson(frequency * exposure)
    claim_count[5:8] = [5, 8, 16]

    frequency_raw = pd.DataFrame(
        {
            "IDpol": np.arange(1, n_policies + 1, dtype=float),
            "ClaimNb": claim_count,
            "Exposure": exposure,
            "Area": area,
            "VehPower": rng.integers(4, 16, n_policies),
            "VehAge": np.clip(rng.poisson(7, n_policies), 0, 100),
            "DrivAge": driver_age,
            "BonusMalus": bonus_malus,
            "VehBrand": rng.choice(["B1", "B2", "B3", "B5", "B12"], n_policies),
            "VehGas": rng.choice(["'Regular'", "'Diesel'"], n_policies),
            "Density": density,
            "Region": rng.choice(["R11", "R24", "R52", "R82", "R93"], n_policies),
        }
    )

    claim_ids = np.repeat(frequency_raw["IDpol"].astype(int).to_numpy(), claim_count)
    amounts = np.round(rng.gamma(2.0, 900.0, claim_ids.size) + 1.0, 2)
    severity_raw = pd.DataFrame({"IDpol": claim_ids, "ClaimAmount": amounts})

    policies_with_claims = frequency_raw.loc[frequency_raw["ClaimNb"] > 0, "IDpol"].astype(int).to_numpy()
    no_amount_ids = policies_with_claims[3:6]
    severity_raw = severity_raw.loc[~severity_raw["IDpol"].isin(no_amount_ids)]
    mismatch_id = policies_with_claims[-1]
    mismatch_rows = severity_raw.index[severity_raw["IDpol"] == mismatch_id]
    severity_raw = severity_raw.drop(mismatch_rows[1:]) if len(mismatch_rows) > 1 else severity_raw
    if len(mismatch_rows) == 1:
        extra = pd.DataFrame({"IDpol": [mismatch_id], "ClaimAmount": [500.0]})
        severity_raw = pd.concat([severity_raw, extra], ignore_index=True)
    orphans = pd.DataFrame({"IDpol": [n_policies + 10, n_policies + 10], "ClaimAmount": [1200.0, 800.0]})
    severity_raw = pd.concat([severity_raw, orphans], ignore_index=True)
    return frequency_raw, severity_raw.reset_index(drop=True)
