"""Data contract of the Sensing API (``/sensing/v1``).

Conventions (same as the PRIME-6G Digital Twin APIs):

* JSON, UTF-8, snake_case field names;
* positions in metres in the frame named by ``frame_id`` (right-handed, Z up);
* angles in degrees;
* RFC 3339 / ISO 8601 UTC timestamps.

A *report* is what a sensing pipeline publishes once per inference. Each
report carries zero or more *detections* (object type + coordinates +
confidence) and identifies the *sensor* that produced it.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator

API_NAME = "sensing-api"
API_VERSION = "v1"
API_PREFIX = f"/sensing/{API_VERSION}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Position3D(BaseModel):
    x: float
    y: float
    z: float = 0.0


class SensorInfo(BaseModel):
    """Who produced the measurement."""
    model_config = ConfigDict(protected_namespaces=())

    sensor_id: str = Field(..., min_length=1, description="Unique id of the physical sensor, e.g. 5g-ssb-rx-01")
    sensor_type: str = Field(..., min_length=1, description="e.g. 5g_ssb, wifi_csi")
    technology: str | None = Field(None, description="e.g. 5G-NR, WiFi")
    method: str | None = Field(None, description="e.g. passive_ssb_fingerprinting")
    model_name: str | None = Field(None, description="Inference model, e.g. ssb_fisher_logreg")
    model_version: str | None = None
    position: Position3D | None = Field(None, description="Sensor location, if known")
    metadata: dict = Field(default_factory=dict)


class Detection(BaseModel):
    object_id: str = Field(..., description="Stable id of the object within the sensor, e.g. person_0")
    object_type: str = Field(..., min_length=1, description="e.g. person")
    position: Position3D
    yaw_deg: float = 0.0
    frame_id: str = "world"
    confidence: float = Field(..., ge=0.0, le=1.0)
    label: str | None = Field(None, description="Model class that produced this detection")


class SensingReportIn(BaseModel):
    """POST /sensing/v1/reports body."""
    sensor: SensorInfo
    timestamp: str = Field(default_factory=utc_now_iso, description="Measurement time (UTC)")
    sequence: int | None = Field(None, ge=0, description="Producer iteration counter")
    valid: bool = True
    invalid_reason: str | None = None
    label: str | None = None
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(default_factory=dict)
    detections: list[Detection] = Field(default_factory=list)
    extra: dict = Field(default_factory=dict, description="Producer diagnostics (timings, DSP)")

    @field_validator("timestamp")
    @classmethod
    def _check_timestamp(cls, value: str) -> str:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timestamp must include a UTC offset")
        return value


class SensingReport(SensingReportIn):
    report_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    received_at: str = Field(default_factory=utc_now_iso)


class PublishedDetection(Detection):
    """Detection as returned to consumers, with its source and time."""
    model_config = ConfigDict(protected_namespaces=())

    sensor_id: str
    sensor_type: str
    technology: str | None = None
    model_name: str | None = None
    timestamp: str
    report_id: str


class DetectionList(BaseModel):
    detections: list[PublishedDetection]
    count: int
    generated_at: str = Field(default_factory=utc_now_iso)


class ReportList(BaseModel):
    reports: list[SensingReport]
    count: int


class SensorStatus(BaseModel):
    sensor: SensorInfo
    last_report_id: str
    last_timestamp: str
    last_received_at: str
    last_valid: bool
    reports_received: int


class SensorList(BaseModel):
    sensors: list[SensorStatus]
    count: int


class PublishAck(BaseModel):
    report_id: str
    received_at: str
    accepted: bool = True
