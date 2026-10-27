from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from sensing_api.mitsuba import make_mitsuba_scene_xml

PERSON = {"object_id": "person_0", "object_type": "person", "position": {"x": 4.0, "y": 2.0, "z": 0.0},
          "yaw_deg": 30.0, "confidence": 0.9}


def test_one_person_gives_body_and_head():
    root = ET.fromstring(make_mitsuba_scene_xml([PERSON]))
    ids = [s.get("id") for s in root.findall("shape")]
    assert ids == ["person_0_body", "person_0_head"]
    ops = [child.tag for child in root.find("shape").find("transform")]
    assert ops == ["scale", "rotate", "translate"]  # scale/rotate about the origin, then place


def test_no_person_gives_empty_scene():
    root = ET.fromstring(make_mitsuba_scene_xml([]))
    assert root.findall("shape") == []
    assert root.find("bsdf").get("id") == "human_body_standard"


def test_mitsuba_places_person_at_published_coordinates():
    """With Mitsuba installed, the loaded body must be centred on (x, y)."""
    mi = pytest.importorskip("mitsuba")
    mi.set_variant("scalar_rgb")
    scene = mi.load_string(make_mitsuba_scene_xml([PERSON]))
    body = next(s for s in scene.shapes() if s.id() == "person_0_body")
    center = (np.array(body.bbox().min) + np.array(body.bbox().max)) / 2
    np.testing.assert_allclose(center[:2], [4.0, 2.0], atol=1e-4)
    assert center[2] == pytest.approx(0.92, abs=1e-4)


def test_two_sensors_reporting_person_0_get_distinct_shape_ids():
    a = {**PERSON, "sensor_id": "rx-a"}
    b = {**PERSON, "sensor_id": "rx-b", "position": {"x": 1.0, "y": 1.0, "z": 0.0}}
    ids = [s.get("id") for s in ET.fromstring(make_mitsuba_scene_xml([a, b])).findall("shape")]
    assert len(ids) == len(set(ids)) == 4
