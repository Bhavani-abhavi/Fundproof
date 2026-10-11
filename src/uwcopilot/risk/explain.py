"""SHAP explanations for tree models and plain-language reason codes (adverse-action style).

SHAP values for LightGBM come from shap.TreeExplainer and are in the model's raw output units:
log-odds for a binary classifier (credit PD), log of the expected value for Poisson or Gamma objectives
(claim frequency or severity). In every case a positive SHAP value pushes the prediction up, which means
higher risk. For each row, base_value + sum(SHAP values) equals the model's raw prediction.

Caveat for the model cards: SHAP describes how the model uses a feature, not a causal effect, and
correlated features can share or swap credit between them.
"""

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import shap

Direction = Literal["increase_risk", "decrease_risk"]


@dataclass
class ShapResult:
    """SHAP values for a set of rows.

    values has shape (n_rows, n_features) and is in the model's raw output units (see module docstring).
    data holds the explained rows, in the same order as values.
    """

    values: np.ndarray
    base_value: float
    feature_names: list[str]
    data: pd.DataFrame


def _plain_value(value: Any) -> Any:
    """Convert numpy scalars to plain Python values so the result can be written to JSON."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def reason_codes(
    shap_values_row: Sequence[float] | np.ndarray,
    feature_names: Sequence[str],
    feature_values: Sequence[Any] | np.ndarray | pd.Series,
    reason_map: Mapping[str, str],
    top_k: int = 4,
    direction: Direction = "increase_risk",
) -> list[dict[str, Any]]:
    """Top features behind one prediction, each with a plain-language reason.

    direction="increase_risk" keeps features with a positive SHAP value, largest first (the reasons a
    credit applicant's risk is higher, as in an adverse-action notice). direction="decrease_risk" keeps
    features with a negative SHAP value, most negative first. Features with a zero contribution are never
    listed, so fewer than top_k reasons can be returned.

    reason_map maps a feature name to its plain-language reason. Features missing from the map get a
    generic reason that names the feature, so a missing mapping is visible rather than hidden.

    Returns a list of {"feature", "value", "contribution", "reason"} dicts.
    """
    contributions = np.asarray(shap_values_row, dtype=float).ravel()
    values = list(feature_values)
    names = list(feature_names)
    if not len(contributions) == len(names) == len(values):
        raise ValueError("shap_values_row, feature_names and feature_values must have the same length")
    if direction not in ("increase_risk", "decrease_risk"):
        raise ValueError("direction must be 'increase_risk' or 'decrease_risk'")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if direction == "increase_risk":
        candidates = [i for i in range(len(names)) if contributions[i] > 0]
        candidates.sort(key=lambda i: contributions[i], reverse=True)
        fallback = "{feature} increased the estimated risk"
    else:
        candidates = [i for i in range(len(names)) if contributions[i] < 0]
        candidates.sort(key=lambda i: contributions[i])
        fallback = "{feature} decreased the estimated risk"

    reasons = []
    for i in candidates[:top_k]:
        feature = names[i]
        reasons.append(
            {
                "feature": feature,
                "value": _plain_value(values[i]),
                "contribution": float(contributions[i]),
                "reason": reason_map.get(feature, fallback.format(feature=feature)),
            }
        )
    return reasons


def _positive_class_values(raw_values: Any) -> np.ndarray:
    """Return SHAP values for the positive class as an (n_rows, n_features) array.

    Depending on the shap version and model type, TreeExplainer returns either one array, a list with one
    array per class, or an array of shape (n_rows, n_features, n_classes). For a binary classifier we keep
    the positive class (the last one).
    """
    if isinstance(raw_values, list):
        return np.asarray(raw_values[-1])
    values = np.asarray(raw_values)
    if values.ndim == 3:
        return values[:, :, -1]
    return values


class TreeShapExplainer:
    """shap.TreeExplainer for one tree model, built once and reused.

    Building the explainer parses every tree, so a service that explains one row per request keeps one of
    these per model instead of rebuilding it each time. explain() gives the same result as compute_tree_shap.
    """

    def __init__(self, model: Any) -> None:
        self._explainer = shap.TreeExplainer(model)

    def explain(self, rows: pd.DataFrame) -> ShapResult:
        """Exact SHAP values for every row (in the model's raw output units)."""
        with warnings.catch_warnings():
            # shap warns that its output format for LightGBM binary classifiers changed between versions.
            # _positive_class_values handles every format, so the warning is not useful here.
            warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer")
            raw_values = self._explainer.shap_values(rows)
        values = _positive_class_values(raw_values)
        base_value = np.asarray(self._explainer.expected_value, dtype=float).ravel()[-1]
        return ShapResult(
            values=values,
            base_value=float(base_value),
            feature_names=[str(name) for name in rows.columns],
            data=rows,
        )


def compute_tree_shap(model: Any, features: pd.DataFrame, sample_size: int | None = 2000, seed: int = 42) -> ShapResult:
    """Compute SHAP values for a LightGBM (or other tree) model on a random sample of rows.

    sample_size rows are drawn without replacement using seed (all rows if sample_size is None or larger than
    the data). Uses shap.TreeExplainer, which is exact for tree ensembles and needs no background sample.
    """
    if sample_size is not None and sample_size < features.shape[0]:
        sample = features.sample(n=sample_size, random_state=seed)
    else:
        sample = features
    return TreeShapExplainer(model).explain(sample)


def global_importance(shap_values: np.ndarray, feature_names: Sequence[str]) -> list[dict[str, Any]]:
    """Global importance: mean absolute SHAP value per feature, largest first.

    Returns a list of {"feature", "mean_abs_shap"} dicts.
    """
    values = np.asarray(shap_values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("shap_values must have shape (n_rows, n_features) matching feature_names")
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(-mean_abs, kind="mergesort")
    return [{"feature": str(feature_names[i]), "mean_abs_shap": float(mean_abs[i])} for i in order]
