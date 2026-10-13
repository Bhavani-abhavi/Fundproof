"""`make studio-data`: every step runs, failures name the host or key, and the snapshot keeps what succeeded."""

import io
import zipfile
from datetime import date

import httpx
import pytest
from studio_helpers import FakeNetwork, fixture_body, make_client

from uwcopilot.studio.data import census_geography, rates, traffic_hpms, wages
from uwcopilot.studio.data.fabric import SnapshotFabric, latest_snapshot
from uwcopilot.studio.data.osm import OVERPASS_URL
from uwcopilot.studio.data.snapshot import build_snapshot, main, write_snapshot
from uwcopilot.studio.metros import load_metro
from uwcopilot.studio.profiles import load_all_profiles


@pytest.fixture(autouse=True)
def no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("uwcopilot.studio.data.keys.ENV_FILE", None)
    for name in ("CENSUS_API_KEY", "BLS_API_KEY", "FRED_API_KEY", "HUD_USER_TOKEN", "ORS_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def houston_network() -> FakeNetwork:
    """Fixtures for every keyless host; Overpass resets the connection like it did from the build environment."""
    network = FakeNetwork()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("2024_Gaz_tracts_national.txt", fixture_body("census_geography/gazetteer_excerpt.json"))
    body = buffer.getvalue()
    network.add(census_geography.gazetteer_url(), lambda request: httpx.Response(200, content=body))
    network.add_fixture(wages.AREA_TITLES_URL, "qcew/area_titles.json")
    network.add_fixture(wages.area_url(2025, 2, "C2642"), "qcew/area_C2642_2025q2.json")
    network.add(f"{wages.QCEW_BASE}/", lambda request: httpx.Response(404))
    network.add_fixture(rates.CSV_URL, "fred/fredgraph_dprime.json")
    network.add_fixture(traffic_hpms.layer_url("TX"), "hpms/county_count.json")

    def reset(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection reset by peer", request=request)

    network.add(OVERPASS_URL.replace("/interpreter", "/status"), reset)
    return network


def test_houston_snapshot_with_keyless_sources_only(tmp_path) -> None:
    network = houston_network()
    client, _ = make_client(tmp_path, network)
    build = build_snapshot(load_metro("houston_tx"), client, load_all_profiles(), date(2025, 9, 1))
    status = {step.name: step.status for step in build.steps}
    assert status["tract_geography"] == "ok" and len(build.tracts) == 6
    assert status["acs_tracts"] == "failed" and status["business_patterns"] == "failed"
    assert status["qcew_wages"] == "ok" and "restaurant_cafe" in build.metro_inputs["local_wages"]
    assert status["prime_rate"] == "ok" and build.metro_inputs["prime_rate"]["status"] == "DATA"
    assert status["hpms_roads"] == "ok" and build.layers["roads"]["counties_with_aadt"] == 10
    assert status["osm_pois"] == "failed"
    assert status["enhancement:traffic_state:TX"] == "not_available"
    failures = "\n".join(build.failures())
    assert "CENSUS_API_KEY" in failures and "overpass-api.de" in failures
    wage = build.metro_inputs["local_wages"]["restaurant_cafe"]
    assert wage["assumptions"] and "hours per week" in wage["assumptions"][0]

    write_snapshot(build, tmp_path / "snapshots")
    fabric = SnapshotFabric(
        metro=load_metro("houston_tx"), snapshot=latest_snapshot("houston_tx", tmp_path / "snapshots")
    )
    assert len(fabric.tract_ids()) == 6
    assert fabric.metro_inputs(load_all_profiles()["restaurant_cafe"])["local_wages"].status == "DATA"
    assert fabric.mode()[0] == "limited"


def test_synthetic_metro_has_nothing_to_download(capsys) -> None:
    assert main(["--metro", "synthetic_metro"]) == 0
    assert "nothing to download" in capsys.readouterr().out


def test_unverified_metro_is_refused(tmp_path) -> None:
    metro = load_metro("houston_tx").model_copy(update={"verified": False, "counties": [], "delineation": None})
    client, _ = make_client(tmp_path, FakeNetwork())
    with pytest.raises(ValueError, match="studio_fetch_delineation"):
        build_snapshot(metro, client, load_all_profiles(), date(2026, 10, 12))
