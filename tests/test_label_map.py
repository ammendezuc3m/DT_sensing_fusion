from __future__ import annotations

import json
from pathlib import Path

import pytest

from sensing_api.label_map import LabelMap, LabelMapError
from sensing_api.producer import build_detections, build_report

REPO = Path(__file__).resolve().parents[1]


def _map(labels: dict, **extra) -> LabelMap:
    return LabelMap.from_dict({"schema_version": "sensing_label_map_v1", "labels": labels, **extra})


def test_repository_template_is_valid():
    label_map = LabelMap.load(REPO / "config" / "sensing_label_map.json")
    assert label_map.empty_labels() == {"empty"}
    assert label_map.frame_id == "world"


def test_parses_lists_single_objects_and_defaults():
    label_map = _map({
        "empty": [],
        "a": [{"object_type": "person", "position": {"x": 1, "y": 2, "z": 0.5}, "yaw_deg": 90}],
        "b": {"position": {"x": 3, "y": 4}},
    }, default_object_type="robot")
    (a,) = label_map.objects_for("a")
    (b,) = label_map.objects_for("b")
    assert (a.object_type, a.x, a.y, a.z, a.yaw_deg) == ("person", 1.0, 2.0, 0.5, 90.0)
    assert (b.object_type, b.z, b.yaw_deg) == ("robot", 0.0, 0.0)


def test_missing_model_labels_fail_fast():
    label_map = _map({"empty": [], "zone_1": {"position": {"x": 0, "y": 0}}})
    with pytest.raises(LabelMapError, match="zone_2"):
        label_map.require_labels(["empty", "zone_1", "zone_2"])


@pytest.mark.parametrize("bad", [
    {"position": {"x": 1}},
    {"position": {"x": "1", "y": 2}},
    {"position": {"x": float("nan"), "y": 2}},
    {"object_type": "", "position": {"x": 1, "y": 2}},
    "zone",
])
def test_rejects_malformed_objects(bad):
    with pytest.raises(LabelMapError):
        _map({"zone": [bad]})


def test_rejects_unknown_schema_version():
    with pytest.raises(LabelMapError, match="schema_version"):
        LabelMap.from_dict({"schema_version": "v999", "labels": {"empty": []}})


def test_invalid_json_file(tmp_path):
    path = tmp_path / "map.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(LabelMapError, match="Invalid JSON"):
        LabelMap.load(path)


def test_detections_keep_stable_ids_per_type():
    label_map = _map({"two": [
        {"position": {"x": 0, "y": 0}}, {"position": {"x": 1, "y": 0}},
        {"object_type": "robot", "position": {"x": 2, "y": 0}},
    ]})
    ids = [d["object_id"] for d in build_detections("two", 0.8, label_map)]
    assert ids == ["person_0", "person_1", "robot_0"]


def test_report_for_valid_and_invalid_iterations():
    label_map = _map({"empty": [], "zone": {"position": {"x": 2, "y": 3}}})
    sensor = {"sensor_id": "rx", "sensor_type": "5g_ssb"}
    prediction = {"label": "zone", "confidence": 0.9, "probabilities": {"empty": 0.1, "zone": 0.9}}

    valid = build_report(sensor=sensor, sequence=1, valid=True, prediction=prediction, label_map=label_map)
    assert valid["detections"][0]["position"] == {"x": 2.0, "y": 3.0, "z": 0.0}
    assert valid["detections"][0]["confidence"] == 0.9

    empty = build_report(sensor=sensor, sequence=2, valid=True,
                         prediction={**prediction, "label": "empty", "confidence": 0.7}, label_map=label_map)
    assert empty["label"] == "empty" and empty["detections"] == []

    invalid = build_report(sensor=sensor, sequence=3, valid=False, prediction=prediction, label_map=label_map,
                           invalid_reason="ibar_mismatch")
    assert invalid["detections"] == [] and invalid["label"] is None
    assert invalid["invalid_reason"] == "ibar_mismatch"
    json.dumps(invalid)  # serialisable
