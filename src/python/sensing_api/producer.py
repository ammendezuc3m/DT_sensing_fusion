"""Producer side: build sensing reports and publish them without blocking the DSP loop.

``build_report`` converts one classifier output into the Sensing API contract
using the label map (label → objects with coordinates).

``ReportPublisher`` POSTs reports from a background thread. It keeps only the
newest pending report ("latest wins"): if the API is slow or down, old states
are dropped instead of queueing up, so the online loop never waits for the
network and consumers never receive stale backlogs.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

import httpx

from .capif_client import CapifError, CapifInvoker
from .label_map import LabelMap
from .schemas import API_NAME, API_PREFIX, utc_now_iso

log = logging.getLogger("sensing_api.producer")


def build_detections(label: str, confidence: float, label_map: LabelMap) -> list[dict]:
    """Objects associated with ``label``; every object gets the class confidence.

    ``object_id`` is ``<object_type>_<n>`` so the same physical object keeps its
    id when the predicted position changes.
    """
    per_type: dict[str, int] = defaultdict(int)
    detections = []
    for obj in label_map.objects_for(label):
        index = per_type[obj.object_type]
        per_type[obj.object_type] += 1
        detections.append({
            "object_id": f"{obj.object_type}_{index}",
            "object_type": obj.object_type,
            "position": {"x": obj.x, "y": obj.y, "z": obj.z},
            "yaw_deg": obj.yaw_deg,
            "frame_id": label_map.frame_id,
            "confidence": float(confidence),
            "label": label,
        })
    return detections


def build_report(
    *,
    sensor: dict,
    sequence: int,
    valid: bool,
    prediction: dict | None,
    label_map: LabelMap,
    invalid_reason: str | None = None,
    extra: dict | None = None,
    timestamp: str | None = None,
) -> dict:
    """Sensing API report for one inference. Invalid iterations carry no detections."""
    report = {
        "sensor": copy.deepcopy(sensor),
        "timestamp": timestamp or utc_now_iso(),
        "sequence": int(sequence),
        "valid": bool(valid),
        "invalid_reason": invalid_reason or None,
        "label": None,
        "confidence": None,
        "probabilities": {},
        "detections": [],
        "extra": extra or {},
    }
    if valid and prediction:
        label = str(prediction["label"])
        confidence = float(prediction["confidence"])
        report.update(
            label=label,
            confidence=confidence,
            probabilities={str(k): float(v) for k, v in prediction.get("probabilities", {}).items()},
            detections=build_detections(label, confidence, label_map),
        )
    return report


@dataclass
class PublisherStats:
    sent: int = 0
    failed: int = 0
    dropped: int = 0
    last_error: str = ""
    base_url: str = ""
    errors_by_type: dict = field(default_factory=lambda: defaultdict(int))


class ReportPublisher:
    """Background, non-blocking POST of reports to the Sensing API."""

    RESOLVE_RETRY_S = 5.0
    LOG_EVERY_S = 10.0

    def __init__(
        self,
        base_url: str | None = None,
        capif_url: str | None = None,
        token: str | None = None,
        timeout_s: float = 2.0,
        app_name: str = "sensing-producer",
    ):
        if not base_url and not capif_url:
            raise ValueError("Give the Sensing API base URL or a CAPIF URL to discover it.")
        self._base_url = base_url.rstrip("/") if base_url else None
        self._fixed_url = self._base_url is not None
        self._capif_url = capif_url
        self._invoker: CapifInvoker | None = None
        self._next_resolve_at = 0.0
        self._app_name = app_name
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._timeout = timeout_s
        self._pending: dict | None = None
        self._cond = threading.Condition()
        self._stop = False
        self._last_log = 0.0
        self.stats = PublisherStats(base_url=self._base_url or "")
        self._thread = threading.Thread(target=self._run, name="sensing-publisher", daemon=True)
        self._thread.start()

    def submit(self, report: dict) -> None:
        with self._cond:
            if self._pending is not None:
                self.stats.dropped += 1
            self._pending = report
            self._cond.notify()

    def close(self, flush_timeout_s: float = 2.0) -> None:
        """Stop the worker after trying to deliver the last pending report."""
        deadline = time.monotonic() + flush_timeout_s
        with self._cond:
            while self._pending is not None and time.monotonic() < deadline:
                self._cond.wait(timeout=0.05)
            self._stop = True
            self._cond.notify()
        self._thread.join(timeout=flush_timeout_s)
        if self._invoker is not None:
            try:
                self._invoker.offboard()
            except httpx.HTTPError:
                pass
            self._invoker.close()

    def _resolve(self) -> str | None:
        """Base URL of the Sensing API; CAPIF discovery is rate-limited to RESOLVE_RETRY_S."""
        if self._base_url:
            return self._base_url
        now = time.monotonic()
        if now < self._next_resolve_at:
            return None
        self._next_resolve_at = now + self.RESOLVE_RETRY_S
        try:
            if self._invoker is None:
                self._invoker = CapifInvoker(self._capif_url, timeout=self._timeout)
            if self._invoker.invoker_id is None:
                self._invoker.enrol(self._app_name)  # enrol once, not on every retry
            self._base_url = self._invoker.resolve_base_url(API_NAME)
            self.stats.base_url = self._base_url
            log.info("Sensing API discovered through CAPIF: %s", self._base_url)
        except (CapifError, httpx.HTTPError, KeyError, ValueError) as exc:
            self._record_error(f"capif_discovery: {exc}")
        return self._base_url

    def _record_error(self, message: str) -> None:
        self.stats.failed += 1
        self.stats.last_error = message
        self.stats.errors_by_type[message.split(":")[0]] += 1
        now = time.monotonic()
        if now - self._last_log > self.LOG_EVERY_S:
            self._last_log = now
            log.warning("Sensing API publish failed (%d so far): %s", self.stats.failed, message)

    def _run(self) -> None:
        with httpx.Client(timeout=self._timeout) as client:
            while True:
                with self._cond:
                    while self._pending is None and not self._stop:
                        self._cond.wait()
                    if self._stop and self._pending is None:
                        return
                    report = self._pending

                base_url = self._resolve()
                if base_url is None:
                    # Not discoverable yet: drop this report (a newer one will follow)
                    # instead of retrying discovery at the sensing loop rate.
                    with self._cond:
                        if self._pending is report:
                            self._pending = None
                            self.stats.dropped += 1
                        self._cond.notify_all()
                        if self._stop:
                            return
                    continue
                try:
                    resp = client.post(f"{base_url}{API_PREFIX}/reports", json=report, headers=self._headers)
                    if resp.status_code == 201:
                        self.stats.sent += 1
                    else:
                        self._record_error(f"http_{resp.status_code}: {resp.text[:200]}")
                except httpx.HTTPError as exc:
                    self._record_error(f"{type(exc).__name__}: {exc}")
                    if not self._fixed_url and isinstance(exc, httpx.TransportError):
                        self._base_url = None  # the API may have moved: discover it again
                with self._cond:
                    if self._pending is report:
                        self._pending = None
                    self._cond.notify_all()
