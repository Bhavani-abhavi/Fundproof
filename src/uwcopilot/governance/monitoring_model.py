"""The model the monitor watches: a cached Poisson GLM of claim frequency, and its registry entry (Phase 6 B).

The GLM is trained by scripts/monitor.py on the seeded training split of freMTPL2 (risk/insurance.py
fit_frequency_glm). It is cached under data/governance/monitoring/ as plain JSON (coefficients and design levels),
not as a pickle, so loading the cache can never run code. The cache is reused only when its fingerprint (the data
file checksums, the split seed, the number of training rows and CACHE_VERSION) equals the current one; otherwise the
model is trained again and the cache replaced.

ensure_registered() adds the model to the model status registry (governance/model_status.py) as the champion of its
family the first time, so a monitoring breach has a registered model to mark under review.
"""

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from uwcopilot.governance.model_status import ModelEntry, ModelStatus, ModelStatusStore
from uwcopilot.governance.monitoring_config import MonitoredModel
from uwcopilot.risk.insurance import GlmDesign, GlmModel, fit_frequency_glm

CACHE_VERSION = "monitoring-frequency-glm-v1"
MODEL_FILE = "frequency_glm.json"
MODEL_DATA_LABEL = "PUBLIC freMTPL2 training split; monitored on a SIMULATED stream"


def file_checksums(paths: list[Path]) -> dict[str, str]:
    """SHA-256 of each file, keyed by file name."""
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def glm_to_dict(model: GlmModel) -> dict[str, Any]:
    """Everything a GlmModel holds, as plain JSON types."""
    return asdict(model)


def glm_from_dict(data: dict[str, Any]) -> GlmModel:
    """Rebuild a GlmModel written by glm_to_dict."""
    fields = dict(data)
    fields["design"] = GlmDesign(**fields["design"])
    return GlmModel(**fields)


def load_or_train_frequency_glm(
    train: pd.DataFrame, cache_dir: Path, fingerprint: dict[str, Any]
) -> tuple[GlmModel, dict[str, Any]]:
    """The cached GLM when its fingerprint matches, otherwise a newly trained one (written to the cache).

    Returns the model and a description of what happened: cache path, whether the cache was used, and the
    SHA-256 of the cached file.
    """
    full_fingerprint = {**fingerprint, "cache_version": CACHE_VERSION, "n_train_rows": int(len(train))}
    path = cache_dir / MODEL_FILE
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == full_fingerprint:
            return glm_from_dict(cached["model"]), _cache_info(path, used_cache=True, fingerprint=full_fingerprint)
    model = fit_frequency_glm(train)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    document = {"fingerprint": full_fingerprint, "data_label": MODEL_DATA_LABEL, "model": glm_to_dict(model)}
    temporary.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return model, _cache_info(path, used_cache=False, fingerprint=full_fingerprint)


def _cache_info(path: Path, used_cache: bool, fingerprint: dict[str, Any]) -> dict[str, Any]:
    """Where the model file is, whether it was reused, and its checksum."""
    return {
        "cache_file": path,
        "loaded_from_cache": used_cache,
        "model_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fingerprint": fingerprint,
    }


def ensure_registered(store: ModelStatusStore, model: MonitoredModel) -> tuple[ModelStatus | None, str]:
    """Register the monitored model as champion when it is not registered yet.

    Returns (status before this run or None when it was just registered, a note). Raises ValueError when another
    model of the same family is already in use, because the registry allows one model in use per family.
    """
    try:
        return store.status(model.model_id), "already registered"
    except KeyError:
        pass
    store.register(
        ModelEntry(
            model_id=model.model_id,
            family=model.family,
            version=model.version,
            status="champion",
            data_label=MODEL_DATA_LABEL,
        ),
        reason="registered by scripts/monitor.py as the model the monitoring demonstration watches",
    )
    return None, "registered as champion by this run"
