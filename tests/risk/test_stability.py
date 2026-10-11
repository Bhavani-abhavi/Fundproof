"""Tests for uwcopilot.risk.stability."""

import math

import numpy as np
import pytest

from uwcopilot.risk.stability import (
    EMPTY_BIN_SHARE,
    categorical_population_stability_index,
    population_stability_index,
    psi_band,
    psi_flag,
    quantile_bin_edges,
)


def test_identical_samples_have_zero_psi() -> None:
    values = np.arange(100, dtype=float)
    assert population_stability_index(values, values) == pytest.approx(0.0)


def test_psi_hand_example_two_bins() -> None:
    # Expected 0..99 splits at the median 49.5 into shares [0.5, 0.5].
    # Actual: 25 values below 49.5 and 75 above -> shares [0.25, 0.75].
    # PSI = (0.25 - 0.5) ln(0.25 / 0.5) + (0.75 - 0.5) ln(0.75 / 0.5) = 0.25 ln 2 + 0.25 ln 1.5
    expected = np.arange(100, dtype=float)
    actual = np.r_[np.zeros(25), np.full(75, 80.0)]
    hand_value = 0.25 * math.log(2.0) + 0.25 * math.log(1.5)
    assert population_stability_index(expected, actual, n_bins=2) == pytest.approx(hand_value)


def test_empty_bin_uses_epsilon_share() -> None:
    # All actual values fall in the upper bin; the lower bin is empty and gets EMPTY_BIN_SHARE.
    expected = np.arange(100, dtype=float)
    actual = np.full(50, 90.0)
    e_shares = np.array([0.5, 0.5])
    a_shares = np.array([EMPTY_BIN_SHARE, 1.0])
    hand_value = float(np.sum((a_shares - e_shares) * np.log(a_shares / e_shares)))
    assert population_stability_index(expected, actual, n_bins=2) == pytest.approx(hand_value)


def test_values_outside_expected_range_go_to_end_bins() -> None:
    expected = np.arange(100, dtype=float)
    shifted = expected + 1000.0  # all above the expected range -> all in the last bin
    assert population_stability_index(expected, shifted) > 0.25


def test_quantile_bin_edges_drop_duplicates() -> None:
    heavily_tied = np.r_[np.zeros(90), np.arange(10, dtype=float)]
    edges = quantile_bin_edges(heavily_tied, n_bins=10)
    assert len(edges) == len(np.unique(edges))
    assert len(edges) < 9


def test_missing_values_get_their_own_bin() -> None:
    expected = np.arange(100, dtype=float)
    actual_with_missing = np.r_[np.arange(50, dtype=float) * 2, np.full(50, np.nan)]
    assert population_stability_index(expected, actual_with_missing) > 0.25


def test_categorical_psi_hand_example() -> None:
    expected = ["A"] * 50 + ["B"] * 50
    actual = ["A"] * 25 + ["B"] * 75
    hand_value = 0.25 * math.log(2.0) + 0.25 * math.log(1.5)
    assert categorical_population_stability_index(expected, actual) == pytest.approx(hand_value)
    assert categorical_population_stability_index(expected, expected) == pytest.approx(0.0)


def test_psi_flag_and_band() -> None:
    assert psi_flag(0.3) is True
    assert psi_flag(0.25) is False
    assert psi_flag(0.2, threshold=0.1) is True
    assert psi_band(0.05) == "stable"
    assert psi_band(0.15) == "moderate shift"
    assert psi_band(0.4) == "significant shift"
