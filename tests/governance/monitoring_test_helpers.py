"""Builders shared by the monitoring tests (tests/governance/test_monitoring*.py). All data is SYNTHETIC.

- synthetic_policies / toy_frequency: cleaned freMTPL2-shaped policies and a stand-in "model" (no training).
"""

import numpy as np
import pandas as pd

from uwcopilot.risk.data.fremtpl2 import prepare_fremtpl2, synthetic_fremtpl2_frames


def synthetic_policies(n_policies: int, seed: int) -> pd.DataFrame:
    """SYNTHETIC cleaned policies shaped like freMTPL2."""
    frequency_raw, severity_raw = synthetic_fremtpl2_frames(n_policies=n_policies, seed=seed)
    return prepare_fremtpl2(frequency_raw, severity_raw).policies.reset_index(drop=True)


def toy_frequency(frame: pd.DataFrame) -> np.ndarray:
    """A stand-in for a model: claims per year rising with BonusMalus (no training needed)."""
    return 0.08 * np.exp(0.01 * (frame["BonusMalus"].to_numpy(dtype=float) - 50))
