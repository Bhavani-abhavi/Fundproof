"""scripts/eval_studio_feasibility.py writes a labeled results file with a provenance block (tiny settings)."""

import json

import eval_studio_feasibility as script
from uwcopilot.agent.policy_credit import DSCR_MINIMUM


def test_main_writes_results_with_provenance(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(script, "TIMING_PATHS", 50)
    monkeypatch.setattr(script, "BOOTSTRAP_RESAMPLES", 20)
    output = tmp_path / "feasibility.json"
    args = ["--worlds", "2", "--paths", "50", "--timing-repeats", "1", "--output", str(output)]
    assert script.main(args) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["data_label"] == "SYNTHETIC"
    assert report["simulation_label"] == "simulated, based on stated assumptions"
    assert report["real_metros"].startswith("not measured")
    assert set(report["conditions"]) == {"well_specified", "misspecified"}
    per_type = report["conditions"]["well_specified"]["per_business_type"]
    assert len(per_type) == 6 and all(block["n_worlds"] == 2 for block in per_type.values())
    assert report["settings"]["dscr_target"]["value"] == DSCR_MINIMUM["term_loan"]
    provenance = report["provenance"]
    assert provenance["generated_by"] == "scripts/eval_studio_feasibility.py"
    assert "config/studio/feasibility/assumptions.yaml" in provenance["eval_sets"]
    assert provenance["data_snapshots"]["studio_synthetic_metro"]["data_label"] == "SYNTHETIC"
    assert set(report["timing"]["per_business_type"]) == set(per_type)
    assert "well_specified" in capsys.readouterr().out
