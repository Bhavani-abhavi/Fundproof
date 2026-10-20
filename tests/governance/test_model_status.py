"""The model status registry: only a model validator promotes or closes a review; monitoring only flags (SYNTHETIC)."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from uwcopilot.governance.model_status import ModelEntry, ModelStatusStore
from uwcopilot.security.rbac import Principal

FAMILY = "fremtpl2_frequency"
VALIDATOR = Principal("validator-1", "model_validator", "claimed")
UNDERWRITER = Principal("underwriter-1", "underwriter", "claimed")
ADMIN = Principal("admin-1", "admin", "claimed")
AT = datetime(2026, 10, 20, tzinfo=UTC)


def store_with_champion_and_challenger(tmp_path: Path) -> ModelStatusStore:
    """A registry with champion v4 and challenger v5 of one family."""
    store = ModelStatusStore(tmp_path / "model_status.json")
    store.register(ModelEntry(model_id="freq-v4", family=FAMILY, version="4", status="champion"), at=AT)
    store.register(ModelEntry(model_id="freq-v5", family=FAMILY, version="5", status="challenger"), at=AT)
    return store


def test_a_validator_promotes_a_challenger_and_the_old_champion_is_retired(tmp_path: Path) -> None:
    store = store_with_champion_and_challenger(tmp_path)
    promoted = store.promote("freq-v5", VALIDATOR, reason="better on the same window (SYNTHETIC)", at=AT)
    assert promoted.status == "champion" and store.status("freq-v4") == "retired"
    last = promoted.history[-1]
    assert (last.from_status, last.to_status, last.actor, last.role) == (
        "challenger",
        "champion",
        "validator-1",
        "model_validator",
    )


@pytest.mark.parametrize("principal", [UNDERWRITER, ADMIN])
def test_nobody_else_can_promote_and_nothing_is_written(tmp_path: Path, principal: Principal) -> None:
    store = store_with_champion_and_challenger(tmp_path)
    before = store.path.read_bytes()
    with pytest.raises(PermissionError):
        store.promote("freq-v5", principal, reason="(SYNTHETIC)")
    assert store.path.read_bytes() == before


def test_monitoring_marks_under_review_and_only_a_validator_closes_it(tmp_path: Path) -> None:
    store = store_with_champion_and_challenger(tmp_path)
    flagged = store.mark_under_review("freq-v4", reason="PSI above threshold (SYNTHETIC)", alert_id="ALERT-1", at=AT)
    assert flagged.status == "under_review" and flagged.history[-1].actor == "monitor"
    assert store.mark_under_review("freq-v4", reason="again", alert_id="ALERT-2").history == flagged.history
    with pytest.raises(PermissionError):
        store.close_review("freq-v4", UNDERWRITER, reason="(SYNTHETIC)")
    assert store.close_review("freq-v4", VALIDATOR, reason="explained (SYNTHETIC)", at=AT).status == "champion"


def test_only_a_challenger_can_be_promoted_and_a_family_has_one_model_in_use(tmp_path: Path) -> None:
    store = store_with_champion_and_challenger(tmp_path)
    with pytest.raises(ValueError):
        store.promote("freq-v4", VALIDATOR, reason="(SYNTHETIC)")
    with pytest.raises(ValueError):
        store.register(ModelEntry(model_id="freq-v6", family=FAMILY, version="6", status="champion"))
    with pytest.raises(ValueError):
        store.register(ModelEntry(model_id="freq-v5", family=FAMILY, version="5", status="challenger"))
    with pytest.raises(KeyError):
        store.status("unknown")
