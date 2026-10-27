"""Tests for the API-to-ROS 2 adapter that do not require a ROS 2 install."""

from __future__ import annotations

import pytest

from sensing_api.ros2_dynamic_obstacle_bridge import detection_to_circle, stable_obstacle_id


DETECTION = {
    "object_id": "person_0",
    "object_type": "person",
    "position": {"x": 1.0, "y": 2.0, "z": 0.0},
    "sensor_id": "5g-ssb-rx-01",
    "frame_id": "world",
}


def test_obstacle_id_is_stable_and_sensor_specific():
    assert stable_obstacle_id("sensor-a", "person_0") == stable_obstacle_id("sensor-a", "person_0")
    assert stable_obstacle_id("sensor-a", "person_0") != stable_obstacle_id("sensor-b", "person_0")


def test_detection_becomes_circle_in_nav2_coordinates():
    obstacle = detection_to_circle(DETECTION, radius_m=0.4)
    assert obstacle.x == pytest.approx(1.0)
    assert obstacle.y == pytest.approx(2.0)
    assert obstacle.radius == pytest.approx(0.4)
    assert obstacle.obstacle_id == stable_obstacle_id("5g-ssb-rx-01", "person_0")


def test_static_planar_transform_is_applied():
    obstacle = detection_to_circle(
        DETECTION, radius_m=0.5, rotate_deg=90.0, translate_x_m=10.0, translate_y_m=-3.0
    )
    assert obstacle.x == pytest.approx(8.0)
    assert obstacle.y == pytest.approx(-2.0)
    assert obstacle.radius == pytest.approx(0.5)
