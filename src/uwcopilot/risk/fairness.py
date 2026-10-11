"""Group fairness checks: selection rates, adverse impact ratio (4/5ths rule) and equal opportunity gap.

Terms used here:
- "selected" is 1 when the model gives the favorable decision (for credit: approved / predicted good) and 0
  otherwise.
- "group" is the value of a protected or demographic attribute for each person (for example an age band).
- "y_true" is 1 when the person's real outcome was the favorable one (for credit: the loan was repaid).

These are demonstration checks on public data (UCI German Credit in this project). Passing or failing a
threshold here is not a legal finding; which tests apply depends on the institution and jurisdiction.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

FOUR_FIFTHS_THRESHOLD = 0.8


def _to_frame(
    selected: Sequence[Any] | np.ndarray | pd.Series, groups: Sequence[Any] | np.ndarray | pd.Series
) -> pd.DataFrame:
    """Put selected (as 0/1 ints) and group (as strings) side by side and validate them."""
    frame = pd.DataFrame({"selected": np.asarray(selected), "group": np.asarray(groups)})
    if frame["group"].isna().any():
        raise ValueError("groups contains missing values; label them explicitly (for example 'unknown')")
    if not frame["selected"].isin([0, 1, True, False]).all():
        raise ValueError("selected must contain only 0/1 or True/False")
    frame["selected"] = frame["selected"].astype(int)
    frame["group"] = frame["group"].astype(str)
    return frame


def selection_rates(
    selected: Sequence[Any] | np.ndarray | pd.Series, groups: Sequence[Any] | np.ndarray | pd.Series
) -> dict[str, float]:
    """Share of people in each group who received the favorable decision. Keys are group labels as strings."""
    frame = _to_frame(selected, groups)
    rates = frame.groupby("group")["selected"].mean()
    return {str(group): float(rate) for group, rate in rates.items()}


def adverse_impact_ratio(
    selected: Sequence[Any] | np.ndarray | pd.Series,
    groups: Sequence[Any] | np.ndarray | pd.Series,
    reference_group: str | None = None,
    threshold: float = FOUR_FIFTHS_THRESHOLD,
) -> dict[str, Any]:
    """Adverse impact ratio of each group: its selection rate divided by the reference group's rate.

    The reference group defaults to the group with the highest selection rate, as in the 4/5ths rule of the
    US Uniform Guidelines on Employee Selection Procedures (29 CFR 1607.4(D), adopted 1978). The rule comes
    from employment selection; it is used here as a simple screening check, not as a fair-lending test.
    A ratio below the threshold (0.8) is flagged.

    Returns {"reference_group", "threshold", "groups": {group: {"n", "selection_rate", "impact_ratio",
    "below_threshold"}}}. impact_ratio is NaN if the reference group's selection rate is 0.
    """
    frame = _to_frame(selected, groups)
    rates = selection_rates(frame["selected"], frame["group"])
    counts = frame["group"].value_counts()
    if reference_group is None:
        reference_group = max(rates, key=lambda group: rates[group])
    reference_group = str(reference_group)
    if reference_group not in rates:
        raise ValueError(f"reference_group '{reference_group}' not found; groups are {sorted(rates)}")

    reference_rate = rates[reference_group]
    result_groups: dict[str, dict[str, Any]] = {}
    for group, rate in rates.items():
        ratio = rate / reference_rate if reference_rate > 0 else float("nan")
        result_groups[group] = {
            "n": int(counts[group]),
            "selection_rate": rate,
            "impact_ratio": ratio,
            "below_threshold": bool(ratio < threshold),
        }
    return {"reference_group": reference_group, "threshold": threshold, "groups": result_groups}


def equal_opportunity_gap(
    y_true: Sequence[Any] | np.ndarray | pd.Series,
    selected: Sequence[Any] | np.ndarray | pd.Series,
    groups: Sequence[Any] | np.ndarray | pd.Series,
    reference_group: str | None = None,
) -> dict[str, Any]:
    """Difference in true positive rate between groups, among people whose real outcome was favorable.

    TPR of a group = share of its truly-favorable people (y_true == 1) who were selected. Equal opportunity
    asks that people who would in fact repay have the same chance of approval whatever their group.

    The reference group defaults to the group with the highest TPR. Returns {"reference_group",
    "max_gap" (highest TPR minus lowest TPR), "groups": {group: {"n_favorable_outcome", "tpr",
    "gap_vs_reference" (group TPR minus reference TPR)}}}. A group with no favorable outcomes has TPR NaN and
    is left out of max_gap.
    """
    labels = np.asarray(y_true)
    if not np.isin(labels, [0, 1, True, False]).all():
        raise ValueError("y_true must contain only 0/1 or True/False")
    frame = _to_frame(selected, groups)
    frame["y_true"] = labels.astype(int)

    tpr_by_group: dict[str, float] = {}
    n_favorable_by_group: dict[str, int] = {}
    for group, rows in frame.groupby("group"):
        favorable = rows[rows["y_true"] == 1]
        n_favorable_by_group[str(group)] = int(favorable.shape[0])
        tpr_by_group[str(group)] = float(favorable["selected"].mean()) if favorable.shape[0] > 0 else float("nan")

    measured = {group: tpr for group, tpr in tpr_by_group.items() if not np.isnan(tpr)}
    if not measured:
        raise ValueError("no group has any favorable outcomes (y_true == 1)")
    if reference_group is None:
        reference_group = max(measured, key=lambda group: measured[group])
    reference_group = str(reference_group)
    if reference_group not in measured:
        raise ValueError(f"reference_group '{reference_group}' has no measured TPR; groups are {sorted(measured)}")

    reference_tpr = measured[reference_group]
    result_groups = {
        group: {
            "n_favorable_outcome": n_favorable_by_group[group],
            "tpr": tpr,
            "gap_vs_reference": tpr - reference_tpr,
        }
        for group, tpr in tpr_by_group.items()
    }
    max_gap = max(measured.values()) - min(measured.values())
    return {"reference_group": reference_group, "max_gap": float(max_gap), "groups": result_groups}
