"""Tests for uwcopilot.risk.data.fremtpl2 on small hand-made and SYNTHETIC tables (no real data needed)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uwcopilot.risk.data.fremtpl2 import (
    CLAIM_COUNT_CAP,
    FREQUENCY_FILE,
    SEVERITY_FILE,
    cap_summary,
    claimant_only_block_mask,
    clean_frequency,
    clean_severity,
    count_only_claims_with_amounts,
    link_severity,
    load_fremtpl2,
    observed_capped_loss,
    prepare_fremtpl2,
    severity_cap,
    severity_table,
    share_with_features_seen_in,
    split_policies,
    split_policies_grouped,
    split_summary,
    synthetic_fremtpl2_frames,
)


def _policy_rows(ids: list[float], claims: list[int], exposure: list[float] | None = None) -> pd.DataFrame:
    """A tiny frequency table with the freMTPL2 columns and quirks (float IDpol, quoted VehGas)."""
    n_rows = len(ids)
    return pd.DataFrame(
        {
            "IDpol": ids,
            "ClaimNb": claims,
            "Exposure": exposure or [0.5] * n_rows,
            "Area": ["C"] * n_rows,
            "VehPower": [6] * n_rows,
            "VehAge": [3] * n_rows,
            "DrivAge": [40] * n_rows,
            "BonusMalus": [50] * n_rows,
            "VehBrand": ["B12"] * n_rows,
            "VehGas": ["'Regular'"] * n_rows,
            "Density": [1000] * n_rows,
            "Region": ["R82"] * n_rows,
        }
    )


def test_clean_frequency_fixes_known_quirks() -> None:
    raw = _policy_rows([1.0, 2.0, 3.0], [0, 6, 1], exposure=[0.4, 1.5, 2.01])
    cleaned, report = clean_frequency(raw)

    assert cleaned["IDpol"].dtype == "int64"
    assert cleaned["IDpol"].tolist() == [1, 2, 3]
    assert cleaned["VehGas"].unique().tolist() == ["Regular"]
    assert cleaned["ClaimNb"].tolist() == [0, CLAIM_COUNT_CAP, 1]
    assert cleaned["ClaimNbRecorded"].tolist() == [0, 6, 1]
    assert cleaned["Exposure"].tolist() == [0.4, 1.0, 1.0]
    assert cleaned["ExposureRecorded"].tolist() == [0.4, 1.5, 2.01]

    assert report["idpol_dtype_in_file"] == "float64"
    assert report["text_values_with_quotes_or_spaces_stripped"]["VehGas"] == 3
    assert report["n_policies_claimnb_above_cap"] == 1
    assert report["n_claims_removed_by_cap"] == 2
    assert report["max_claimnb_recorded"] == 6
    assert report["policies_above_claim_cap"] == [{"IDpol": 2, "ClaimNb": 6, "Exposure": 1.5}]
    assert report["n_policies_exposure_above_cap"] == 2
    assert report["max_exposure_recorded"] == pytest.approx(2.01)
    assert report["n_claims_after_cap"] == 5
    assert report["n_policies_with_claims"] == 2


def test_clean_frequency_drops_non_positive_exposure() -> None:
    cleaned, report = clean_frequency(_policy_rows([1.0, 2.0], [0, 0], exposure=[0.0, 0.5]))
    assert cleaned["IDpol"].tolist() == [2]
    assert report["n_rows_dropped_non_positive_exposure"] == 1


def test_clean_frequency_rejects_bad_ids_and_missing_values() -> None:
    with pytest.raises(ValueError, match="non-integer"):
        clean_frequency(_policy_rows([1.5, 2.0], [0, 0]))
    with pytest.raises(ValueError, match="duplicate"):
        clean_frequency(_policy_rows([1.0, 1.0], [0, 0]))
    with_missing = _policy_rows([1.0, 2.0], [0, 0])
    with_missing.loc[0, "DrivAge"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        clean_frequency(with_missing)
    with pytest.raises(ValueError, match="missing columns"):
        clean_frequency(_policy_rows([1.0], [0]).drop(columns=["Region"]))


def test_clean_severity_drops_non_positive_amounts() -> None:
    raw = pd.DataFrame({"IDpol": [1, 2, 3], "ClaimAmount": [100.0, 0.0, -5.0]})
    cleaned, report = clean_severity(raw)
    assert cleaned["ClaimAmount"].tolist() == [100.0]
    assert report == {"n_rows_in_file": 3, "n_rows_dropped_non_positive_amount": 2}


def test_link_severity_counts_every_disagreement() -> None:
    # IDpol 1-3 all have claims (a claimant-only block); IDpol 4 is the first with zero claims.
    policies, _ = clean_frequency(_policy_rows([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 1, 2, 0, 1, 0]))
    claims = pd.DataFrame(
        {
            "IDpol": [3, 5, 5, 99],  # policy 3 has 1 amount for 2 claims; 5 has 2 amounts for 1 claim; 99 unknown
            "ClaimAmount": [300.0, 50.0, 70.0, 1000.0],
        }
    )
    linked, kept, report = link_severity(policies, claims)

    assert kept["IDpol"].tolist() == [3, 5, 5]
    assert linked.set_index("IDpol")["n_severity_rows"].to_dict() == {1: 0, 2: 0, 3: 1, 4: 0, 5: 2, 6: 0}
    assert linked.set_index("IDpol").loc[5, "claim_amount_total"] == pytest.approx(120.0)
    assert report["n_severity_rows_with_unknown_idpol"] == 1
    assert report["n_unknown_idpol_values_in_severity_file"] == 1
    assert report["claim_amount_dropped_with_unknown_idpol"] == pytest.approx(1000.0)
    assert report["n_policies_with_claims_but_no_amounts"] == 2  # policies 1 and 2
    assert report["n_claims_on_policies_without_amounts"] == 2
    assert report["share_of_claims_without_amounts"] == pytest.approx(2 / 5)
    assert report["n_policies_where_severity_rows_differ_from_claimnb"] == 2
    assert report["n_policies_where_severity_rows_differ_from_modelling_claimnb"] == 2
    assert report["n_amount_rows_above_modelling_claimnb"] == 1  # policy 5: 2 amounts, 1 claim
    assert report["n_modelling_claims_without_an_amount_row"] == 1  # policy 3: 2 claims, 1 amount
    assert report["lowest_idpol_with_zero_claims"] == 4
    assert report["n_policies_below_that_idpol"] == 3
    assert report["n_policies_below_that_idpol_without_amounts"] == 2
    assert report["n_claims_below_that_idpol"] == 4
    assert report["share_of_policies_below_that_idpol_with_amounts"] == pytest.approx(1 / 3)
    assert report["share_of_policies_with_claims_from_that_idpol_on"] == pytest.approx(1 / 3)
    first_band = report["claim_policies_by_idpol_band"][0]
    assert first_band == {
        "idpol_band": "1-10,000",
        "n_policies": 6,
        "n_policies_with_claims": 4,
        "n_claim_policies_with_amounts": 2,
        "n_claim_policies_without_amounts": 2,
    }


def test_link_severity_compares_amounts_with_the_capped_claim_count() -> None:
    # Policy 1 records 6 claims with 6 amounts: it matches the file, but the model counts only 4 claims.
    policies, _ = clean_frequency(_policy_rows([1.0, 2.0, 3.0], [6, 1, 0]))
    claims = pd.DataFrame({"IDpol": [1] * 6 + [2], "ClaimAmount": [100.0] * 6 + [50.0]})
    _, _, report = link_severity(policies, claims)
    assert report["n_policies_where_severity_rows_differ_from_claimnb"] == 0
    assert report["n_policies_where_severity_rows_differ_from_modelling_claimnb"] == 1
    assert report["n_amount_rows_above_modelling_claimnb"] == 2
    assert report["policies_where_severity_rows_differ_from_modelling_claimnb"] == [
        {"IDpol": 1, "ClaimNbRecorded": 6, "ClaimNbModelling": CLAIM_COUNT_CAP, "n_severity_rows": 6}
    ]


def test_claimant_only_block_mask() -> None:
    policies = pd.DataFrame({"IDpol": [1, 2, 3, 10]})
    assert claimant_only_block_mask(policies, 3).tolist() == [True, True, False, False]
    assert claimant_only_block_mask(policies, None).tolist() == [True, True, True, True]


def test_split_is_seeded_stratified_and_disjoint() -> None:
    data = prepare_fremtpl2(*synthetic_fremtpl2_frames(n_policies=5000, seed=3))
    splits = split_policies(data.policies, seed=42)

    sizes = {name: len(df) for name, df in splits.items()}
    assert sizes == {"train": 3000, "validation": 1000, "test": 1000}
    all_ids = pd.concat([df["IDpol"] for df in splits.values()])
    assert all_ids.is_unique
    assert set(all_ids) == set(data.policies["IDpol"])

    overall_share = (data.policies["ClaimNb"] > 0).mean()
    for df in splits.values():
        assert (df["ClaimNb"] > 0).mean() == pytest.approx(overall_share, abs=0.002)

    again = split_policies(data.policies, seed=42)
    assert again["test"]["IDpol"].tolist() == splits["test"]["IDpol"].tolist()
    other_seed = split_policies(data.policies, seed=7)
    assert other_seed["test"]["IDpol"].tolist() != splits["test"]["IDpol"].tolist()

    summary = split_summary(splits)
    assert summary["test"]["n_policies"] == 1000
    assert summary["train"]["n_claims"] == int(splits["train"]["ClaimNb"].sum())


def test_split_rejects_bad_fractions() -> None:
    data = prepare_fremtpl2(*synthetic_fremtpl2_frames(n_policies=200, seed=0))
    with pytest.raises(ValueError, match="fractions"):
        split_policies(data.policies, fractions={"train": 0.5, "validation": 0.2, "test": 0.2})


def test_severity_cap_and_cap_summary_by_hand() -> None:
    amounts = np.arange(1.0, 201.0)  # 1, 2, ..., 200
    cap = severity_cap(amounts, quantile=0.995)
    assert cap == pytest.approx(np.quantile(amounts, 0.995))

    summary = cap_summary([100.0, 200.0, 1000.0], cap=300.0)
    assert summary["n_claims"] == 3
    assert summary["n_claims_capped"] == 1
    assert summary["share_of_claims_capped"] == pytest.approx(1 / 3)
    assert summary["amount_removed_by_cap"] == pytest.approx(700.0)
    assert summary["share_of_amount_removed_by_cap"] == pytest.approx(700.0 / 1300.0)
    with pytest.raises(ValueError):
        severity_cap([])


def test_severity_table_caps_each_claim_before_averaging() -> None:
    policies, _ = clean_frequency(_policy_rows([1.0, 2.0, 3.0], [2, 1, 1]))
    claims = pd.DataFrame({"IDpol": [1, 1, 2], "ClaimAmount": [100.0, 5000.0, 400.0]})
    linked, kept, _ = link_severity(policies, claims)
    table = severity_table(linked, kept, cap=1000.0).set_index("IDpol")

    assert list(table.index) == [1, 2]  # policy 3 has a claim but no amount, so it is not in the table
    assert table.loc[1, "n_claims"] == 2
    assert table.loc[1, "total_amount_uncapped"] == pytest.approx(5100.0)
    assert table.loc[1, "total_amount_capped"] == pytest.approx(1100.0)
    assert table.loc[1, "average_amount_capped"] == pytest.approx(550.0)
    assert table.loc[2, "average_amount_capped"] == pytest.approx(400.0)

    observed = observed_capped_loss(linked, kept, cap=1000.0)
    assert observed.tolist() == pytest.approx([1100.0, 400.0, 0.0])


def test_load_fremtpl2_reads_csv_files(tmp_path: Path) -> None:
    frequency_raw, severity_raw = synthetic_fremtpl2_frames(n_policies=800, seed=2)
    frequency_raw.to_csv(tmp_path / FREQUENCY_FILE, index=False)
    severity_raw.to_csv(tmp_path / SEVERITY_FILE, index=False)

    data = load_fremtpl2(tmp_path)

    assert len(data.policies) == 800
    assert data.policies["VehGas"].isin(["Regular", "Diesel"]).all()
    assert data.policies["ClaimNb"].max() <= CLAIM_COUNT_CAP
    assert data.policies["Exposure"].max() <= 1.0
    assert set(data.quality) == {"frequency_file", "severity_file", "linkage", "modelling_target"}
    assert data.claims["IDpol"].isin(data.policies["IDpol"]).all()


def test_load_fremtpl2_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make data"):
        load_fremtpl2(tmp_path)


def test_synthetic_frames_reproduce_the_quirks() -> None:
    data = prepare_fremtpl2(*synthetic_fremtpl2_frames(n_policies=3000, seed=0))
    frequency = data.quality["frequency_file"]
    linkage = data.quality["linkage"]
    assert frequency["text_values_with_quotes_or_spaces_stripped"]["VehGas"] == 3000
    assert frequency["n_policies_exposure_above_cap"] > 0
    assert frequency["n_policies_claimnb_above_cap"] > 0
    assert linkage["n_policies_with_claims_but_no_amounts"] > 0
    assert linkage["n_severity_rows_with_unknown_idpol"] > 0
    assert linkage["n_policies_where_severity_rows_differ_from_claimnb"] >= 1


def test_claims_without_amounts_are_not_counted_for_modelling() -> None:
    policies, _ = clean_frequency(_policy_rows([1.0, 2.0, 3.0, 4.0], [2, 1, 0, 6], exposure=[1.0, 0.5, 0.5, 1.0]))
    claims = pd.DataFrame({"IDpol": [1, 1, 4], "ClaimAmount": [100.0, 200.0, 50.0]})
    linked, _, _ = link_severity(policies, claims)
    modelled, report = count_only_claims_with_amounts(linked)

    by_id = modelled.set_index("IDpol")
    assert by_id["ClaimNb"].to_dict() == {1: 2, 2: 0, 3: 0, 4: CLAIM_COUNT_CAP}  # policy 2 has no amount
    assert by_id["ClaimNbCapped"].to_dict() == {1: 2, 2: 1, 3: 0, 4: CLAIM_COUNT_CAP}
    assert by_id["ClaimNbRecorded"].to_dict() == {1: 2, 2: 1, 3: 0, 4: 6}
    assert report["n_policies_set_to_zero_claims"] == 1
    assert report["n_claims_not_counted"] == 1
    assert report["n_claims_for_modelling"] == 6
    assert report["claims_per_policy_year_for_modelling"] == pytest.approx(6 / 3.0)
    assert report["claims_per_policy_year_as_recorded_after_cap"] == pytest.approx(7 / 3.0)


def test_prepare_fremtpl2_reports_the_modelling_target() -> None:
    data = prepare_fremtpl2(*synthetic_fremtpl2_frames(n_policies=2000, seed=6))
    target = data.quality["modelling_target"]
    no_amounts = (data.policies["ClaimNbCapped"] > 0) & (data.policies["n_severity_rows"] == 0)
    assert (data.policies.loc[no_amounts, "ClaimNb"] == 0).all()
    assert target["n_policies_set_to_zero_claims"] == int(no_amounts.sum()) > 0
    assert target["n_claims_for_modelling"] == int(data.policies["ClaimNb"].sum())


def test_grouped_split_keeps_identical_risks_together() -> None:
    data = prepare_fremtpl2(*synthetic_fremtpl2_frames(n_policies=4000, seed=3))
    policies = data.policies.copy()
    # Make identical rating vectors common, as in the real file: few distinct values per feature.
    policies["DrivAge"] = policies["DrivAge"] // 20 * 20
    policies["Density"] = 100
    policies["BonusMalus"] = 50
    splits = split_policies_grouped(policies, seed=42)
    assert sum(len(df) for df in splits.values()) == len(policies)
    ids = [set(df["IDpol"]) for df in splits.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert share_with_features_seen_in(splits["test"], splits["train"]) == 0.0
    assert share_with_features_seen_in(splits["validation"], splits["train"]) == 0.0
    random_splits = split_policies(policies, seed=42)
    assert share_with_features_seen_in(random_splits["test"], random_splits["train"]) > 0.0
    again = split_policies_grouped(policies, seed=42)
    assert again["test"]["IDpol"].tolist() == splits["test"]["IDpol"].tolist()  # seeded


def test_share_with_features_seen_in_counts_exact_matches() -> None:
    reference = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "x"]})
    rows = pd.DataFrame({"a": [1, 2, 2, 3], "b": ["x", "x", "y", "x"]})
    assert share_with_features_seen_in(rows, reference, ["a", "b"]) == pytest.approx(0.5)
