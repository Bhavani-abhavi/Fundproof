"""Tests for the helper functions in scripts/eval_kyc.py and scripts/make_kyc_customers.py (no data/raw)."""

from pathlib import Path

import pandas as pd
import pytest

import eval_kyc
import make_kyc_customers


def score_table() -> pd.DataFrame:
    """A small fake screening-score table shaped like the one eval_kyc writes."""
    rows = []
    for index in range(20):
        label = "true_match" if index < 4 else "clean"
        rows.append(
            {
                "customer_id": f"SYN-KYC-{index:05d}",
                "split": "train",
                "label": label,
                "generation": "typo+honorific" if index < 2 else ("initials" if index < 4 else "faker_person"),
                "score_rules": 0.9 if index < 3 else 0.5,
                "top_rules_entry_source": "OpenSanctions",
                "top_rules_entry_list": "OpenSanctions (US SAM Procurement Exclusions; Wikidata)",
            }
        )
        for weight in eval_kyc.RERANKER_WEIGHT_GRID:
            rows[-1][eval_kyc.weight_column(weight)] = rows[-1]["score_rules"] - weight / 10
    return pd.DataFrame(rows)


def test_weight_column_names() -> None:
    assert eval_kyc.weight_column(0.3) == "score_rerank_w0.3"


def test_reranker_weight_ties_go_to_the_smallest_weight() -> None:
    weight, table = eval_kyc.choose_reranker_weight(score_table())
    assert weight == min(eval_kyc.RERANKER_WEIGHT_GRID)
    assert len(table) == len(eval_kyc.RERANKER_WEIGHT_GRID)


def test_recall_change_points_keep_only_rises() -> None:
    curve = [
        {"threshold": 0.9, "alerts_per_1000": 10.0, "recall": 0.5},
        {"threshold": 0.8, "alerts_per_1000": 20.0, "recall": 0.5},
        {"threshold": 0.7, "alerts_per_1000": 30.0, "recall": 1.0},
    ]
    assert [point["threshold"] for point in eval_kyc.recall_change_points(curve)] == [0.9, 0.7]


def test_recall_by_noise_counts_each_operation() -> None:
    table = eval_kyc.recall_by_noise(score_table(), "score_rules", 0.8)
    assert table["typo"] == {"true_matches": 2, "alerted": 2}
    assert table["initials"] == {"true_matches": 2, "alerted": 1}


def test_false_alert_sources_use_the_first_dataset_title() -> None:
    table = score_table()
    table.loc[10, "score_rules"] = 0.95
    result = eval_kyc.false_alert_sources(table, "score_rules", 0.8)
    assert result["clean_customers_alerted"] == 1
    assert result["top_first_listed_dataset"] == {"US SAM Procurement Exclusions": 1}


def test_scripts_stop_cleanly_without_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert eval_kyc.main(["--customers-file", str(tmp_path / "missing.csv")]) == 1
    monkeypatch.setattr(make_kyc_customers, "DEFAULT_OFAC_SDN", tmp_path / "missing_sdn.csv")
    assert make_kyc_customers.main(["--customers-file", str(tmp_path / "c.csv")]) == 1


def test_homonym_stratum_reports_alert_rates_per_operating_point() -> None:
    homonyms = pd.DataFrame(
        {
            "score_rules": [1.0, 1.0, 0.6, 0.4],
            "score_rerank_w0.1": [0.9, 0.9, 0.5, 0.3],
            "entity_type": ["person", "person", "organization", "organization"],
        }
    )
    columns = {eval_kyc.RULES: "score_rules", eval_kyc.RERANK: "score_rerank_w0.1"}
    points = {
        "100": {
            eval_kyc.RULES: {"threshold_chosen_on_validation": 0.95},
            eval_kyc.RERANK: {"threshold_chosen_on_validation": 0.45},
        }
    }
    selected = {"method": eval_kyc.RULES, "threshold": 0.95}
    report = {"clean": {"generated": 100, "dropped_list_collisions": 12}}
    stratum = eval_kyc.homonym_stratum(homonyms, points, columns, selected, report)
    assert stratum["customers"] == 4
    assert stratum["share_of_generated_clean_names_that_were_exact_list_names"] == 0.12
    assert stratum["alert_rate_at_selected_point"]["value"] == 0.5
    assert stratum["alerted_at_selected_point_by_entity_type"] == {
        "organization": {"customers": 2, "alerted": 0},
        "person": {"customers": 2, "alerted": 2},
    }
    assert stratum["alert_rate_by_budget"]["100"][eval_kyc.RERANK]["alerted"] == 3
