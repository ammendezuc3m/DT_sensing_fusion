"""Example Sensing API consumer (CAPIF invoker).

1. Enrols as API invoker in the CAPIF Core Function.
2. Discovers ``sensing-api`` and resolves its base URL from the routing info.
3. Reads detections, either by polling ``GET /sensing/v1/detections`` or by
   subscribing to the ``GET /sensing/v1/stream`` Server-Sent Events.
4. Optionally converts them to the consumer's own format: a JSON file and/or a
   Mitsuba XML scene for Sionna.

Examples (from src/python):
    python -m sensing_api.consume --capif-url http://127.0.0.1:8080/capif/v1 --once
    python -m sensing_api.consume --capif-url http://127.0.0.1:8080/capif/v1 --stream \
        --mitsuba-xml /tmp/live_scene.xml
    python -m sensing_api.consume --api-url http://127.0.0.1:8090 --poll-interval 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .capif_client import CapifInvoker
from .config import load_settings
from .mitsuba import make_mitsuba_scene_xml
from .schemas import API_NAME, API_PREFIX


def parse_args() -> argparse.Namespace:
    settings = load_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--capif-url", default=settings.capif.base_url, help="Discover the API through CAPIF")
    source.add_argument("--api-url", help="Skip CAPIF and call this Sensing API base URL directly")
    p.add_argument("--app-name", default="sensing-consumer-example")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Read the current detections once and exit")
    mode.add_argument("--stream", action="store_true", help="Subscribe to the SSE report stream")
    p.add_argument("--poll-interval", type=float, default=1.0, help="Seconds between polls")
    p.add_argument("--object-type", default=None, help="Filter, e.g. person")
    p.add_argument("--sensor-id", default=None)
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--max-age-s", type=float, default=5.0, help="Ignore sensors silent for longer")
    p.add_argument("--json-out", type=Path, help="Write the latest detections to this JSON file")
    p.add_argument("--mitsuba-xml", type=Path, help="Write the latest detections as a Mitsuba XML scene")
    return p.parse_args()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def handle(detections: list[dict], args: argparse.Namespace, timestamp: str) -> None:
    summary = ", ".join(
        f"{d['object_type']}@({d['position']['x']:.2f},{d['position']['y']:.2f}) "
        f"p={d['confidence']:.2f} [{d.get('sensor_id', '?')}]"
        for d in detections
    ) or "no objects"
    print(f"{timestamp}  {summary}", flush=True)
    if args.json_out:
        _atomic_write(args.json_out, json.dumps({"timestamp": timestamp, "detections": detections}, indent=2))
    if args.mitsuba_xml:
        _atomic_write(args.mitsuba_xml, make_mitsuba_scene_xml(detections, timestamp_utc=timestamp))


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def resolve_base_url(args: argparse.Namespace) -> str:
    if args.api_url:
        return args.api_url.rstrip("/")
    with CapifInvoker(args.capif_url) as capif:
        invoker_id = capif.enrol(args.app_name)
        base_url = capif.resolve_base_url(API_NAME)
    print(f"CAPIF invoker {invoker_id}: {API_NAME} at {base_url}", file=sys.stderr)
    return base_url


def poll(base_url: str, args: argparse.Namespace) -> None:
    params = {k: v for k, v in {
        "object_type": args.object_type,
        "sensor_id": args.sensor_id,
        "min_confidence": args.min_confidence,
        "max_age_s": args.max_age_s,
    }.items() if v is not None}
    with httpx.Client(timeout=5.0) as client:
        while True:
            resp = client.get(f"{base_url}{API_PREFIX}/detections", params=params)
            resp.raise_for_status()
            body = resp.json()
            handle(body["detections"], args, body["generated_at"])
            if args.once:
                return
            time.sleep(args.poll_interval)


def stream(base_url: str, args: argparse.Namespace) -> None:
    """Keep the latest detections of every sensor and emit the merged set on each event."""
    per_sensor: dict[str, tuple[datetime, list[dict]]] = {}
    with httpx.Client(timeout=httpx.Timeout(5.0, read=None)) as client:
        with client.stream("GET", f"{base_url}{API_PREFIX}/stream") as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                report = json.loads(line[len("data: "):])
                sensor = report["sensor"]
                if args.sensor_id and sensor["sensor_id"] != args.sensor_id:
                    continue
                detections = [
                    {**d, "sensor_id": sensor["sensor_id"], "sensor_type": sensor["sensor_type"],
                     "timestamp": report["timestamp"]}
                    for d in report["detections"]
                    if (not args.object_type or d["object_type"] == args.object_type)
                    and d["confidence"] >= args.min_confidence
                ] if report["valid"] else []
                per_sensor[sensor["sensor_id"]] = (_parse_time(report["timestamp"]), detections)

                now = datetime.now(timezone.utc)
                merged = [
                    d for ts, dets in per_sensor.values()
                    if args.max_age_s is None or (now - ts).total_seconds() <= args.max_age_s
                    for d in dets
                ]
                handle(merged, args, report["timestamp"])


def main() -> None:
    args = parse_args()
    base_url = resolve_base_url(args)
    try:
        (stream if args.stream else poll)(base_url, args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
