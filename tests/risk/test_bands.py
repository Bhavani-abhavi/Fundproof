"""Tests for uwcopilot.risk.bands: risk-band cut-offs from training-score quantiles."""

import json

import numpy as np
import pytest

from uwcopilot.risk.bands import BAND_LABELS, BAND_QUANTILES, assign_band, band_cutoffs, describe_bands


def test_cutoffs_are_quantiles_of_the_scores() -> None:
    scores = np.random.default_rng(0).gamma(2.0, 0.05, size=5000)
    bands = band_cutoffs(scores, "test score", "SYNTHETIC rows")
    assert bands["cutoffs"] == pytest.approx([np.quantile(scores, q) for q in BAND_QUANTILES])
    assert bands["labels"] == list(BAND_LABELS)
    assert bands["n_scores"] == 5000
    json.dumps(bands, allow_nan=False)  # stored in metadata.json and results JSON


def test_band_shares_match_the_quantile_levels() -> None:
    scores = np.random.default_rng(1).uniform(size=20000)
    bands = band_cutoffs(scores, "test score", "SYNTHETIC rows")
    labels = [assign_band(value, bands) for value in scores]
    shares = {label: labels.count(label) / len(labels) for label in BAND_LABELS}
    assert shares["low"] == pytest.approx(0.50, abs=0.01)
    assert shares["medium"] == pytest.approx(0.30, abs=0.01)
    assert shares["high"] == pytest.approx(0.15, abs=0.01)
    assert shares["very high"] == pytest.approx(0.05, abs=0.01)


def test_assign_band_edges() -> None:
    bands = {"cutoffs": [0.1, 0.2, 0.3], "labels": ["low", "medium", "high", "very high"]}
    assert assign_band(0.0, bands) == "low"
    assert assign_band(0.1, bands) == "medium"  # equal to a cut-off goes up
    assert assign_band(0.25, bands) == "high"
    assert assign_band(5.0, bands) == "very high"
    with pytest.raises(ValueError, match="NaN"):
        assign_band(float("nan"), bands)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        band_cutoffs([], "s", "p")
    with pytest.raises(ValueError, match="NaN"):
        band_cutoffs([0.1, float("nan")], "s", "p")
    with pytest.raises(ValueError, match="one more label"):
        band_cutoffs([0.1, 0.2], "s", "p", quantiles=(0.5,), labels=("a", "b", "c"))
    with pytest.raises(ValueError, match="increasing"):
        band_cutoffs([0.1, 0.2], "s", "p", quantiles=(0.8, 0.5), labels=("a", "b", "c"))


def test_describe_bands_has_open_ends() -> None:
    bands = band_cutoffs(np.arange(100.0), "s", "p")
    rows = describe_bands(bands)
    assert [row["band"] for row in rows] == list(BAND_LABELS)
    assert rows[0]["lower_cutoff"] is None and rows[-1]["upper_cutoff"] is None
    assert rows[1]["lower_cutoff"] == rows[0]["upper_cutoff"] == bands["cutoffs"][0]
