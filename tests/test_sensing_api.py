"""Sensing API and CAPIF: unit tests with TestClient and an end-to-end run over real HTTP."""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from sensing_api import capif_core
from sensing_api.capif_client import CapifInvoker
from sensing_api.config import CapifSettings, SensingApiSettings, Settings
from sensing_api.label_map import LabelMap
from sensing_api.producer import ReportPublisher, build_report
from sensing_api.server import create_app

SENSOR = {"sensor_id": "5g-ssb-rx-01", "sensor_type": "5g_ssb", "technology": "5G-NR",
          "model_name": "ssb_fisher_logreg"}
LABEL_MAP = LabelMap.from_dict({"labels": {
    "empty": [],
    "zone_1": {"object_type": "person", "position": {"x": 1.5, "y": 2.0}},
}})


def _settings(capif_url="http://127.0.0.1:1/capif/v1", port=8090, register=False, token=None) -> Settings:
    return Settings(
        capif=CapifSettings(base_url=capif_url),
        sensing_api=SensingApiSettings(port=port, advertised_host="127.0.0.1", history_size=5,
                                       register_with_capif=register, producer_token=token),
    )


def _report(label="zone_1", confidence=0.93, valid=True, sequence=0, timestamp=None, sensor=SENSOR):
    prediction = {"label": label, "confidence": confidence, "probabilities": {label: confidence}}
    return build_report(sensor=sensor, sequence=sequence, valid=valid, prediction=prediction,
                        label_map=LABEL_MAP, timestamp=timestamp,
                        invalid_reason=None if valid else "ibar_mismatch: detected 4, model expects 3")


# ─────────────────────────────────────────────
# Unit tests (in-process)
# ─────────────────────────────────────────────

@pytest.fixture
def client():
    with TestClient(create_app(_settings())) as c:
        yield c


def test_publish_and_read_current_detections(client):
    ack = client.post("/sensing/v1/reports", json=_report())
    assert ack.status_code == 201, ack.text
    body = client.get("/sensing/v1/detections").json()
    assert body["count"] == 1
    det = body["detections"][0]
    assert det["object_type"] == "person"
    assert det["position"] == {"x": 1.5, "y": 2.0, "z": 0.0}
    assert det["confidence"] == pytest.approx(0.93)
    assert det["sensor_id"] == "5g-ssb-rx-01" and det["sensor_type"] == "5g_ssb"
    assert det["report_id"] == ack.json()["report_id"]


def test_invalid_or_empty_latest_report_clears_detections(client):
    client.post("/sensing/v1/reports", json=_report())
    client.post("/sensing/v1/reports", json=_report(valid=False))
    assert client.get("/sensing/v1/detections").json()["count"] == 0
    client.post("/sensing/v1/reports", json=_report(label="empty"))
    assert client.get("/sensing/v1/detections").json()["count"] == 0
    latest = client.get("/sensing/v1/sensors/5g-ssb-rx-01/latest").json()
    assert latest["label"] == "empty" and latest["valid"] is True


def test_filters_and_staleness(client):
    old = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()
    client.post("/sensing/v1/reports", json=_report(timestamp=old))
    other = {**SENSOR, "sensor_id": "rx-02"}
    client.post("/sensing/v1/reports", json=_report(confidence=0.4, sensor=other))

    assert client.get("/sensing/v1/detections").json()["count"] == 2
    assert client.get("/sensing/v1/detections", params={"max_age_s": 5}).json()["count"] == 1
    assert client.get("/sensing/v1/detections", params={"min_confidence": 0.5}).json()["count"] == 1
    assert client.get("/sensing/v1/detections", params={"sensor_id": "rx-02"}).json()["count"] == 1
    assert client.get("/sensing/v1/detections", params={"object_type": "robot"}).json()["count"] == 0
    assert client.get("/sensing/v1/sensors").json()["count"] == 2


def test_history_is_bounded_and_filterable(client):
    for i in range(8):
        client.post("/sensing/v1/reports", json=_report(sequence=i))
    reports = client.get("/sensing/v1/reports").json()["reports"]
    assert [r["sequence"] for r in reports] == [3, 4, 5, 6, 7]  # history_size=5
    assert client.get("/sensing/v1/reports", params={"limit": 2}).json()["count"] == 2
    assert client.get("/sensing/v1/reports", params={"since": "not-a-date"}).status_code == 422
    assert client.get("/sensing/v1/reports", params={"since": "2020-01-01T00:00:00"}).status_code == 422
    assert client.get("/sensing/v1/reports", params={"since": "2020-01-01T00:00:00Z"}).json()["count"] == 5


@pytest.mark.parametrize("mutate", [
    lambda r: r["detections"][0].update(confidence=1.5),
    lambda r: r["sensor"].pop("sensor_id"),
    lambda r: r.update(timestamp="2026-01-01T00:00:00"),  # no UTC offset
    lambda r: r["detections"][0]["position"].pop("x"),
])
def test_rejects_malformed_reports(client, mutate):
    report = _report()
    mutate(report)
    assert client.post("/sensing/v1/reports", json=report).status_code == 422


def test_unknown_sensor_is_404(client):
    assert client.get("/sensing/v1/sensors/nope/latest").status_code == 404


def test_producer_token():
    with TestClient(create_app(_settings(token="s3cret"))) as c:
        assert c.post("/sensing/v1/reports", json=_report()).status_code == 401
        ok = c.post("/sensing/v1/reports", json=_report(), headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 201
        assert c.get("/sensing/v1/detections").status_code == 200  # reading stays open


def test_capif_core_requires_enrolled_provider():
    with TestClient(capif_core.create_app()) as c:
        desc = {"apiName": "sensing-api", "aefProfiles": [{"aefId": "x", "versions": [{"apiVersion": "v1"}]}]}
        assert c.post("/capif/v1/apf-publish-service-apis/APF-unknown/service-apis", json=desc).status_code == 404


# ─────────────────────────────────────────────
# End-to-end over real HTTP: CCF + Sensing API + producer + consumer
# ─────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Server:
    def __init__(self, app, port):
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("server did not start")
            time.sleep(0.02)
        return self

    def __exit__(self, *_):
        self.server.should_exit = True
        self.thread.join(timeout=5)


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_end_to_end_capif_discovery_publish_and_consume():
    ccf_port, api_port = _free_port(), _free_port()
    ccf_url = f"http://127.0.0.1:{ccf_port}/capif/v1"
    settings = _settings(capif_url=ccf_url, port=api_port, register=True)

    # The Sensing API starts first: it must keep retrying until the CCF is up.
    with _Server(create_app(settings), api_port) as api:
        time.sleep(0.3)
        with _Server(capif_core.create_app(), ccf_port):
            app = api.server.config.app
            assert _wait_for(lambda: app.state.capif_provider.api_id is not None, timeout=15), "not registered"

            # Consumer: enrol and discover through CAPIF
            with CapifInvoker(ccf_url) as capif:
                capif.enrol("e2e-consumer")
                base_url = capif.resolve_base_url()
            assert base_url == f"http://127.0.0.1:{api_port}"

            # Consumer subscribes to the SSE stream before anything is published
            events: list[dict] = []

            def listen():
                with httpx.stream("GET", f"{base_url}/sensing/v1/stream", timeout=httpx.Timeout(5, read=10)) as r:
                    for line in r.iter_lines():
                        if line.startswith("data: "):
                            events.append(json.loads(line[6:]))
                            return

            listener = threading.Thread(target=listen, daemon=True)
            listener.start()
            assert _wait_for(lambda: httpx.get(f"{base_url}/sensing/v1/health").json()["stream_subscribers"] == 1)

            # Producer: discovers the API through CAPIF as well
            publisher = ReportPublisher(capif_url=ccf_url, app_name="e2e-producer")
            publisher.submit(_report(sequence=7))
            publisher.close()
            assert publisher.stats.sent == 1 and publisher.stats.failed == 0, publisher.stats

            listener.join(timeout=10)
            assert events and events[0]["sequence"] == 7
            assert events[0]["detections"][0]["position"]["x"] == 1.5

            detections = httpx.get(f"{base_url}/sensing/v1/detections").json()["detections"]
            assert [(d["object_type"], d["sensor_id"]) for d in detections] == [("person", "5g-ssb-rx-01")]

        # CCF gone: health still answers, publication still works locally
        assert httpx.get(f"http://127.0.0.1:{api_port}/sensing/v1/health").json()["capif_registered"] is True


def test_publisher_survives_api_down_and_keeps_only_latest():
    port = _free_port()  # nothing listening
    publisher = ReportPublisher(base_url=f"http://127.0.0.1:{port}", timeout_s=0.5)
    for i in range(20):
        publisher.submit(_report(sequence=i))
    publisher.close(flush_timeout_s=1.0)
    assert publisher.stats.sent == 0
    assert publisher.stats.failed >= 1
    assert publisher.stats.failed + publisher.stats.dropped <= 20


def test_publisher_requires_a_destination():
    with pytest.raises(ValueError):
        ReportPublisher()


def test_settings_base_url():
    settings = _settings(port=9999)
    assert replace(settings.sensing_api, advertised_host="10.0.0.5").base_url == "http://10.0.0.5:9999"


def test_reregisters_after_capif_restart(monkeypatch):
    from sensing_api import server as server_module

    monkeypatch.setattr(server_module, "CAPIF_CHECK_S", 0.2)
    monkeypatch.setattr(server_module, "CAPIF_RETRY_S", 0.2)
    ccf_port, api_port = _free_port(), _free_port()
    ccf_url = f"http://127.0.0.1:{ccf_port}/capif/v1"
    registry = capif_core.CapifRegistry()

    with _Server(capif_core.create_app(registry), ccf_port):
        with _Server(create_app(_settings(capif_url=ccf_url, port=api_port, register=True)), api_port) as api:
            provider = api.server.config.app.state.capif_provider
            assert _wait_for(lambda: provider.api_id is not None)
            first_id = provider.api_id

            registry.clear()  # same effect as restarting the CCF: in-memory registry lost

            assert _wait_for(lambda: provider.api_id not in (None, first_id))
            with CapifInvoker(ccf_url) as capif:
                assert capif.resolve_base_url() == f"http://127.0.0.1:{api_port}"


def test_publisher_rate_limits_capif_discovery():
    """CCF reachable but sensing-api not published: no discovery storm, one invoker only."""
    ccf_port = _free_port()
    registry = capif_core.CapifRegistry()
    with _Server(capif_core.create_app(registry), ccf_port):
        publisher = ReportPublisher(capif_url=f"http://127.0.0.1:{ccf_port}/capif/v1")
        for i in range(200):
            publisher.submit(_report(sequence=i))
            time.sleep(0.002)
        publisher.close(flush_timeout_s=0.5)
        assert publisher.stats.errors_by_type["capif_discovery"] <= 2
        assert len(registry.invokers) == 0  # enrolled once, offboarded on close
    assert publisher.stats.sent == 0


def test_capif_discovery_is_exact_name_match():
    with TestClient(capif_core.create_app()) as c:
        apf = c.post("/capif/v1/api-provider-enrolments", json={}).json()["apfId"]
        for name in ("sensing-api", "sensing-api-test"):
            desc = {"apiName": name, "aefProfiles": [{"aefId": "x", "versions": [{"apiVersion": "v1"}]}]}
            assert c.post(f"/capif/v1/apf-publish-service-apis/{apf}/service-apis", json=desc).status_code == 201
        found = c.get("/capif/v1/service-apis", params={"api_name": "sensing-api"}).json()
        assert [a["apiName"] for a in found["serviceAPIDescriptions"]] == ["sensing-api"]
