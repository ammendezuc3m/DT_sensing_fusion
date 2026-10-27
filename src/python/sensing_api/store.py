"""Thread-safe in-memory store of sensing reports."""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone

from .schemas import PublishedDetection, SensingReport, SensorStatus


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp {value!r} must include a UTC offset")
    return parsed


class ReportStore:
    def __init__(self, history_size: int = 1000):
        self._lock = threading.Lock()
        self._history: deque[SensingReport] = deque(maxlen=history_size)
        self._latest: dict[str, SensingReport] = {}
        self._counts: dict[str, int] = {}

    def add(self, report: SensingReport) -> None:
        sensor_id = report.sensor.sensor_id
        with self._lock:
            self._history.append(report)
            self._latest[sensor_id] = report
            self._counts[sensor_id] = self._counts.get(sensor_id, 0) + 1

    def clear(self) -> None:
        with self._lock:
            self._history.clear()
            self._latest.clear()
            self._counts.clear()

    def latest(self, sensor_id: str) -> SensingReport | None:
        with self._lock:
            return self._latest.get(sensor_id)

    def history(self, sensor_id: str | None = None, since: str | None = None, limit: int = 100) -> list[SensingReport]:
        since_dt = _parse_time(since) if since else None
        with self._lock:
            reports = list(self._history)
        if sensor_id:
            reports = [r for r in reports if r.sensor.sensor_id == sensor_id]
        if since_dt:
            reports = [r for r in reports if _parse_time(r.timestamp) > since_dt]
        return reports[-limit:]

    def sensors(self) -> list[SensorStatus]:
        with self._lock:
            return [
                SensorStatus(
                    sensor=report.sensor,
                    last_report_id=report.report_id,
                    last_timestamp=report.timestamp,
                    last_received_at=report.received_at,
                    last_valid=report.valid,
                    reports_received=self._counts[sensor_id],
                )
                for sensor_id, report in self._latest.items()
            ]

    def current_detections(
        self,
        object_type: str | None = None,
        sensor_id: str | None = None,
        min_confidence: float = 0.0,
        max_age_s: float | None = None,
    ) -> list[PublishedDetection]:
        """Detections of the latest report of each sensor.

        A sensor whose latest report is invalid contributes nothing: the
        previous valid state is not repeated, because it may be outdated.
        """
        now = datetime.now(timezone.utc)
        with self._lock:
            latest = list(self._latest.values())
        result: list[PublishedDetection] = []
        for report in latest:
            if not report.valid or (sensor_id and report.sensor.sensor_id != sensor_id):
                continue
            if max_age_s is not None and (now - _parse_time(report.timestamp)).total_seconds() > max_age_s:
                continue
            for det in report.detections:
                if object_type and det.object_type != object_type:
                    continue
                if det.confidence < min_confidence:
                    continue
                result.append(PublishedDetection(
                    **det.model_dump(),
                    sensor_id=report.sensor.sensor_id,
                    sensor_type=report.sensor.sensor_type,
                    technology=report.sensor.technology,
                    model_name=report.sensor.model_name,
                    timestamp=report.timestamp,
                    report_id=report.report_id,
                ))
        return result
