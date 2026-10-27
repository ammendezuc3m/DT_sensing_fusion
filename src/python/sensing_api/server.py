"""Sensing API — CAPIF API Exposing Function (AEF) for passive sensing results.

Producers (the online 5G SSB pipeline, later WiFi CSI, ...) publish one report
per inference; consumers (Digital Twin, Sionna, dashboards) read the current
detections or subscribe to a Server-Sent Events stream. On startup the server
enrols in the CAPIF Core Function and publishes itself as ``sensing-api`` so
invokers can discover it.

  POST /sensing/v1/reports                   publish a report (producer)
  GET  /sensing/v1/reports                   report history (sensor_id, since, limit)
  GET  /sensing/v1/detections                current detections of every sensor
  GET  /sensing/v1/sensors                   sensors that have published
  GET  /sensing/v1/sensors/{sensor_id}/latest latest report of a sensor
  GET  /sensing/v1/stream                    SSE stream of new reports
  GET  /sensing/v1/health                    status and CAPIF registration

Run:
    python -m sensing_api.server
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import threading
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from .capif_client import CapifError, CapifProvider
from .config import Settings, load_settings
from .schemas import (
    API_NAME,
    API_PREFIX,
    DetectionList,
    PublishAck,
    ReportList,
    SensingReport,
    SensingReportIn,
    SensorList,
    utc_now_iso,
)
from .store import ReportStore

log = logging.getLogger("sensing_api.server")

SSE_QUEUE_SIZE = 100
SSE_HEARTBEAT_S = 15.0
CAPIF_RETRY_S = 5.0
CAPIF_CHECK_S = 30.0


class _Subscribers:
    """Fan-out of new reports to SSE clients (event-loop side only)."""

    def __init__(self) -> None:
        self._queues: set[asyncio.Queue] = set()

    def add(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=SSE_QUEUE_SIZE)
        self._queues.add(queue)
        return queue

    def remove(self, queue: asyncio.Queue) -> None:
        self._queues.discard(queue)

    def publish(self, payload: str) -> None:
        for queue in self._queues:
            if queue.full():  # slow consumer: drop its oldest event, keep the newest
                queue.get_nowait()
            queue.put_nowait(payload)

    def __len__(self) -> int:
        return len(self._queues)


def _capif_registration_loop(provider: CapifProvider, settings: Settings, stop: threading.Event) -> None:
    """Register in CAPIF and keep the registration alive.

    The CCF may start after this server or restart and lose its in-memory
    registry, so registration is retried and periodically verified. A failed
    verification (CCF unreachable) does not re-register: only a confirmed
    "not published" does, so duplicate entries are not created.
    """
    api = settings.sensing_api
    while not stop.is_set():
        try:
            if provider.api_id is None or not provider.is_published():
                provider.register(api.advertised_host, api.port)
            delay = CAPIF_CHECK_S
        except (CapifError, httpx.HTTPError, KeyError, ValueError) as exc:
            log.warning("CAPIF registration/check failed (%s); retrying in %.0fs", exc, CAPIF_RETRY_S)
            delay = CAPIF_RETRY_S
        stop.wait(delay)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    store = ReportStore(settings.sensing_api.history_size)
    subscribers = _Subscribers()
    provider = CapifProvider(settings.capif.base_url)
    stop_event = threading.Event()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        thread = None
        if settings.sensing_api.register_with_capif:
            thread = threading.Thread(
                target=_capif_registration_loop, args=(provider, settings, stop_event), daemon=True
            )
            thread.start()
        yield
        stop_event.set()
        if thread:
            # let an in-flight registration finish so it can be unpublished below
            await asyncio.to_thread(thread.join, 2 * provider.timeout + 1.0)
        await asyncio.to_thread(provider.unregister)

    app = FastAPI(
        title="Sensing API",
        description=(
            "CAPIF-exposed (3GPP TS 29.222) API publishing passive RF sensing results: "
            "object type, coordinates, confidence and source sensor."
        ),
        version="1.0.0",
        docs_url=f"{API_PREFIX}/ui",
        openapi_url=f"{API_PREFIX}/openapi.json",
        lifespan=lifespan,
    )
    app.state.store = store
    app.state.capif_provider = provider

    def _check_token(authorization: str | None) -> None:
        token = settings.sensing_api.producer_token
        if token and authorization != f"Bearer {token}":
            raise HTTPException(401, "Invalid or missing producer token")

    @app.post(f"{API_PREFIX}/reports", status_code=201, response_model=PublishAck, tags=["Producers"])
    async def publish_report(report_in: SensingReportIn, authorization: str | None = Header(None)):
        _check_token(authorization)
        report = SensingReport(**report_in.model_dump())
        store.add(report)
        subscribers.publish(report.model_dump_json())
        return PublishAck(report_id=report.report_id, received_at=report.received_at)

    @app.get(f"{API_PREFIX}/reports", response_model=ReportList, tags=["Consumers"])
    def list_reports(
        sensor_id: str | None = None,
        since: str | None = Query(None, description="Only reports with timestamp after this UTC time"),
        limit: int = Query(100, ge=1, le=10000),
    ):
        try:
            reports = store.history(sensor_id=sensor_id, since=since, limit=limit)
        except ValueError as exc:
            raise HTTPException(422, f"Invalid 'since': {exc}") from exc
        return ReportList(reports=reports, count=len(reports))

    @app.get(f"{API_PREFIX}/detections", response_model=DetectionList, tags=["Consumers"])
    def list_detections(
        object_type: str | None = None,
        sensor_id: str | None = None,
        min_confidence: float = Query(0.0, ge=0.0, le=1.0),
        max_age_s: float | None = Query(None, gt=0, description="Ignore sensors whose last report is older"),
    ):
        detections = store.current_detections(object_type, sensor_id, min_confidence, max_age_s)
        return DetectionList(detections=detections, count=len(detections))

    @app.get(f"{API_PREFIX}/sensors", response_model=SensorList, tags=["Consumers"])
    def list_sensors():
        sensors = store.sensors()
        return SensorList(sensors=sensors, count=len(sensors))

    @app.get(f"{API_PREFIX}/sensors/{{sensor_id}}/latest", response_model=SensingReport, tags=["Consumers"])
    def latest_report(sensor_id: str):
        report = store.latest(sensor_id)
        if report is None:
            raise HTTPException(404, f"No report from sensor {sensor_id!r}")
        return report

    @app.get(f"{API_PREFIX}/stream", tags=["Consumers"],
             summary="Server-Sent Events: one 'report' event per published report")
    async def stream(request: Request):
        queue = subscribers.add()

        async def events():
            try:
                yield ": connected\n\n"
                while not await request.is_disconnected():
                    try:
                        payload = await asyncio.wait_for(queue.get(), timeout=SSE_HEARTBEAT_S)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    yield f"event: report\ndata: {payload}\n\n"
            finally:
                subscribers.remove(queue)

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get(f"{API_PREFIX}/health", tags=["Health"])
    def health():
        return {
            "status": "ok",
            "component": f"{API_NAME} (CAPIF AEF)",
            "spec": "3GPP TS 29.222 CAPIF",
            "capif_ccf": settings.capif.base_url,
            "capif_registered": provider.api_id is not None,
            "apf_id": provider.apf_id,
            "aef_id": provider.aef_id,
            "service_api_id": provider.api_id,
            "advertised_base_url": settings.sensing_api.base_url,
            "sensors": len(store.sensors()),
            "stream_subscribers": len(subscribers),
            "timestamp": utc_now_iso(),
        }

    return app


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Run the Sensing API (CAPIF AEF).")
    parser.add_argument("--host", default=settings.sensing_api.host)
    parser.add_argument("--port", type=int, default=settings.sensing_api.port)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [SENSING-API] %(levelname)s %(message)s")
    if args.port != settings.sensing_api.port:
        from dataclasses import replace
        settings = replace(settings, sensing_api=replace(settings.sensing_api, port=args.port))
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
