"""Run the real online script main() with a simulated radio, against a live Sensing API."""

from __future__ import annotations

import csv
import json
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import online_5g_ssb_sensing as online
from conftest import IBAR, PCI, make_grids
from sensing_api.server import create_app
from ssb_fisher_torch_inference import (
    build_feature_vector,
    export_sklearn_pipeline_as_torch,
    normalised_channel_magnitude_db,
)
from test_sensing_api import _free_port, _Server, _settings

LABELS = ["empty", "zone_1"]


@pytest.fixture(scope="module")
def setup(tmp_path_factory, rng):
    tmp = tmp_path_factory.mktemp("online")
    grids = {label: make_grids(i, 40, rng) for i, label in enumerate(LABELS)}
    selected = [(k, s) for k in range(0, 240, 20) for s in (1, 3)]
    x = np.vstack([
        np.stack([build_feature_vector(normalised_channel_magnitude_db(g, PCI, IBAR), selected)
                  for g in np.moveaxis(grids[label], 2, 0)])
        for label in LABELS
    ])
    y = np.repeat([0, 1], 40)
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=5000))]).fit(x, y)
    model_path = tmp / "model_fisher_logreg.pt"
    export_sklearn_pipeline_as_torch(pipeline, model_path, LABELS, selected, {"pci": PCI, "ibar": IBAR})

    label_map = tmp / "label_map.json"
    label_map.write_text(json.dumps({"labels": {
        "empty": [], "zone_1": [{"object_type": "person", "position": {"x": 4.0, "y": 1.0}}],
    }}))
    return {"tmp": tmp, "grids": grids, "model": model_path, "label_map": label_map}


def _fake_radio(monkeypatch, sequence):
    """Replace USRP + DSP: each capture yields the next (grid, metric) of ``sequence``."""
    feed = iter(sequence)

    class Usrp:
        def get_rx_rate(self, _ch):
            return 15.36e6

    class Streamer:
        def get_max_num_samps(self):
            return 2000

    current = {}

    def capture_one_block(**_kw):
        current["item"] = next(feed, current.get("item"))
        return np.zeros(8, dtype=np.complex64)

    def extract(waveform, args):
        grid, metric = current["item"]
        info = {"nid2": args.force_nid2, "metric": metric, "timing_offset_samples": 100,
                "n_symbols_extracted": args.num_symbols}
        return None, grid, info, {"pss_time_ms": 1.0, "ofdm_time_ms": 1.0, "total_dsp_time_ms": 2.0}

    monkeypatch.setattr(online, "configure_usrp", lambda args: Usrp())
    monkeypatch.setattr(online, "make_rx_streamer", lambda usrp, ch: Streamer())
    monkeypatch.setattr(online, "capture_one_block", capture_one_block)
    monkeypatch.setattr(online, "extract_rxgrid_from_waveform", extract)


def test_online_main_publishes_to_sensing_api(setup, monkeypatch, rng):
    grids = setup["grids"]
    other_beam = make_grids(1, 1, rng, ibar=(IBAR + 1) % 8)[:, :, 0]
    sequence = [
        (grids["zone_1"][:, :, 0], 0.9),   # valid, person
        (grids["empty"][:, :, 0], 0.9),    # valid, empty
        (other_beam, 0.9),                 # other SSB beam -> rejected
        (grids["zone_1"][:, :, 1], 0.1),   # weak PSS -> rejected
        (grids["zone_1"][:, :, 2], 0.9),   # valid, person (last state)
    ]
    _fake_radio(monkeypatch, sequence)
    tmp = setup["tmp"]
    port = _free_port()

    with _Server(create_app(_settings(port=port)), port) as server:
        monkeypatch.setattr(sys, "argv", [
            "online_5g_ssb_sensing.py",
            "--fisher-model", str(setup["model"]), "--label-map", str(setup["label_map"]),
            "--force-nid2", str(PCI % 3), "--num-iters", str(len(sequence)), "--warmup-iters", "0",
            "--sensing-api-url", f"http://127.0.0.1:{port}",
            "--local-json", str(tmp / "state.json"), "--log-csv", str(tmp / "log.csv"),
            "--enable-mitsuba-export", "--progress-every", "0",
        ])
        online.main()
        store = server.server.config.app.state.store
        history = store.history(limit=100)
        latest = store.current_detections()

    rows = list(csv.DictReader((tmp / "log.csv").open()))
    assert [r["valid"] for r in rows] == ["1", "1", "0", "0", "1"]
    assert rows[2]["error"].startswith("ibar_mismatch")
    assert rows[3]["error"].startswith("pss_metric_below_threshold")
    assert all(r["rx_mean_abs"] not in ("", "nan") for r in rows)

    assert [r.sequence for r in history][-1] == 4          # last report delivered
    assert [(d.object_type, d.position.x, d.position.y) for d in latest] == [("person", 4.0, 1.0)]

    state = json.loads((tmp / "state.json").read_text())
    assert state["schema_version"] == "python_5g_ssb_online_v2"
    assert state["detections"][0]["position"] == {"x": 4.0, "y": 1.0, "z": 0.0}

    shapes = ET.parse(tmp / "live_person_sionna_scene.xml").getroot().findall("shape")
    assert len(shapes) == 2  # body + head of one person


def test_online_refuses_label_map_without_model_classes(setup, monkeypatch, tmp_path):
    bad_map = tmp_path / "map.json"
    bad_map.write_text(json.dumps({"labels": {"empty": []}}))
    monkeypatch.setattr(sys, "argv", [
        "online_5g_ssb_sensing.py", "--fisher-model", str(setup["model"]), "--label-map", str(bad_map),
        "--disable-api",
    ])
    with pytest.raises(Exception, match="zone_1"):
        online.main()


def test_online_requires_api_destination(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["online_5g_ssb_sensing.py"])
    with pytest.raises(SystemExit):
        online.parse_args()
