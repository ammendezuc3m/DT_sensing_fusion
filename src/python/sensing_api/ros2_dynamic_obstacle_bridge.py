"""Bridge active Sensing API detections into ROS 2 Nav2 dynamic obstacles.

The bridge is a consumer: it never accesses the USRP or changes the online
5G sensing producer.  It polls ``/sensing/v1/detections`` and publishes one
``dynamic_obstacle_msgs/msg/DynamicObstacle`` per active detection.

The Nav2 dynamic-obstacle workspace must be sourced before this module is run,
so that ``rclpy`` and ``dynamic_obstacle_msgs`` are available.

Example (from ``src/python``)::

    python -m sensing_api.ros2_dynamic_obstacle_bridge \\
      --capif-url http://<capif-host>:8080/capif/v1 \\
      --topic /global_costmap/dynamic_obstacles --object-type person
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import zlib
from dataclasses import dataclass

import httpx

from .capif_client import CapifInvoker
from .config import load_settings
from .schemas import API_NAME, API_PREFIX


@dataclass(frozen=True)
class ObstacleFields:
    """ROS-message values independent of ROS 2, kept separately for testing."""

    obstacle_id: int
    x: float
    y: float
    radius: float


def stable_obstacle_id(sensor_id: str, object_id: str) -> int:
    """Return a repeatable positive 31-bit id for one sensor/object pair."""
    value = zlib.crc32(f"{sensor_id}\0{object_id}".encode("utf-8")) & 0x7FFF_FFFF
    return value or 1


def detection_to_circle(
    detection: dict,
    *,
    radius_m: float,
    translate_x_m: float = 0.0,
    translate_y_m: float = 0.0,
    rotate_deg: float = 0.0,
) -> ObstacleFields:
    """Convert an API detection into a Nav2 circle, with an optional 2-D transform."""
    position = detection["position"]
    x, y = float(position["x"]), float(position["y"])
    theta = math.radians(rotate_deg)
    transformed_x = math.cos(theta) * x - math.sin(theta) * y + translate_x_m
    transformed_y = math.sin(theta) * x + math.cos(theta) * y + translate_y_m
    return ObstacleFields(
        obstacle_id=stable_obstacle_id(str(detection["sensor_id"]), str(detection["object_id"])),
        x=transformed_x,
        y=transformed_y,
        radius=radius_m,
    )


def parse_args() -> argparse.Namespace:
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--capif-url", default=settings.capif.base_url, help="Discover the Sensing API through CAPIF")
    source.add_argument("--api-url", help="Call this Sensing API base URL directly")
    parser.add_argument("--app-name", default="sensing-api-ros2-bridge")
    parser.add_argument("--topic", default="/global_costmap/dynamic_obstacles")
    parser.add_argument("--poll-interval", type=float, default=0.5, help="Seconds between API reads")
    parser.add_argument("--object-type", default="person", help="Only bridge this object type; empty string accepts all")
    parser.add_argument("--sensor-id", default=None, help="Only bridge this sensor")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-age-s", type=float, default=2.0, help="Ignore silent sensors after this many seconds")
    parser.add_argument("--expected-frame-id", default=None,
                        help="Only accept detections in this frame, e.g. world")
    parser.add_argument("--person-radius-m", type=float, default=0.40,
                        help="Circle radius used for each person detection")
    parser.add_argument("--translate-x-m", type=float, default=0.0,
                        help="Static X translation from sensing coordinates to Nav2 coordinates")
    parser.add_argument("--translate-y-m", type=float, default=0.0,
                        help="Static Y translation from sensing coordinates to Nav2 coordinates")
    parser.add_argument("--rotate-deg", type=float, default=0.0,
                        help="Counter-clockwise rotation from sensing coordinates to Nav2 coordinates")
    parser.add_argument("--once", action="store_true", help="Publish the current state once and exit")
    args = parser.parse_args()
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be greater than zero")
    if args.max_age_s is not None and args.max_age_s <= 0:
        parser.error("--max-age-s must be greater than zero")
    if not 0.0 <= args.min_confidence <= 1.0:
        parser.error("--min-confidence must be in [0, 1]")
    if args.person_radius_m <= 0:
        parser.error("--person-radius-m must be greater than zero")
    return args


def resolve_base_url(args: argparse.Namespace) -> str:
    if args.api_url:
        return args.api_url.rstrip("/")
    with CapifInvoker(args.capif_url) as capif:
        invoker_id = capif.enrol(args.app_name)
        base_url = capif.resolve_base_url(API_NAME)
    print(f"CAPIF invoker {invoker_id}: {API_NAME} at {base_url}", file=sys.stderr)
    return base_url


class DynamicObstacleBridge:
    """Polling adapter from the API data contract to Florian's ROS 2 message."""

    def __init__(self, node, message_type, base_url: str, args: argparse.Namespace):
        self.node = node
        self.message_type = message_type
        self.base_url = base_url.rstrip("/")
        self.args = args
        self.publisher = node.create_publisher(message_type, args.topic, 10)
        self.client = httpx.Client(timeout=5.0)

    def close(self) -> None:
        self.client.close()

    def _query_params(self) -> dict:
        values = {
            "object_type": self.args.object_type or None,
            "sensor_id": self.args.sensor_id,
            "min_confidence": self.args.min_confidence,
            "max_age_s": self.args.max_age_s,
        }
        return {key: value for key, value in values.items() if value is not None}

    def publish_current(self) -> int:
        try:
            response = self.client.get(f"{self.base_url}{API_PREFIX}/detections", params=self._query_params())
            response.raise_for_status()
            detections = response.json()["detections"]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            self.node.get_logger().error(f"Could not read Sensing API: {exc}")
            return 0

        published = 0
        for detection in detections:
            if self.args.expected_frame_id and detection.get("frame_id") != self.args.expected_frame_id:
                self.node.get_logger().warning(
                    f"Skipping {detection.get('object_id', '?')}: frame_id={detection.get('frame_id')!r}, "
                    f"expected {self.args.expected_frame_id!r}"
                )
                continue
            try:
                fields = detection_to_circle(
                    detection,
                    radius_m=self.args.person_radius_m,
                    translate_x_m=self.args.translate_x_m,
                    translate_y_m=self.args.translate_y_m,
                    rotate_deg=self.args.rotate_deg,
                )
            except (KeyError, TypeError, ValueError) as exc:
                self.node.get_logger().warning(f"Skipping malformed detection: {exc}")
                continue

            message = self.message_type()
            message.obstacle_id = fields.obstacle_id
            message.shape = 0  # CIRCLE, defined by dynamic_obstacle_msgs/DynamicObstacle.msg
            message.x_points = [fields.x]
            message.y_points = [fields.y]
            message.radius = fields.radius
            message.thickness = 0.0
            self.publisher.publish(message)
            published += 1

        self.node.get_logger().info(f"Published {published} active obstacle(s) from {len(detections)} API detection(s)")
        return published


def main() -> None:
    args = parse_args()
    base_url = resolve_base_url(args)
    try:
        import rclpy
        from dynamic_obstacle_msgs.msg import DynamicObstacle
        from rclpy.node import Node
    except ImportError as exc:
        raise SystemExit(
            "ROS 2 or dynamic_obstacle_msgs is unavailable. Source the ROS 2 installation and the "
            f"Nav2 dynamic-obstacle workspace before running this bridge ({exc})."
        ) from exc

    rclpy.init()
    node = Node("sensing_api_dynamic_obstacle_bridge")
    bridge = DynamicObstacleBridge(node, DynamicObstacle, base_url, args)
    try:
        if args.once:
            bridge.publish_current()
            return
        node.create_timer(args.poll_interval, bridge.publish_current)
        node.get_logger().info(
            f"Bridging {base_url}{API_PREFIX}/detections to {args.topic} every {args.poll_interval:.2f} s"
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
