"""CAPIF Core Function (CCF) for the sensing deployment.

Subset of 3GPP TS 29.222 / ETSI TS 123 222, adapted from the PRIME-6G Digital
Twin repository (``prime6g_dt/capif/ccf_server.py``) so this repository can run
its own CAPIF instance:

  Provider enrolment     POST   /capif/v1/api-provider-enrolments
  Invoker enrolment      POST   /capif/v1/api-invoker-enrolments
                         DELETE /capif/v1/api-invoker-enrolments/{apiInvokerId}
  Publish service API    POST   /capif/v1/apf-publish-service-apis/{apfId}/service-apis
                         PUT/DELETE .../service-apis/{serviceApiId}
  Discover service APIs  GET    /capif/v1/service-apis
  Routing info           GET    /capif/v1/routing-info/{serviceApiId}

The registry is in memory: providers re-register when the CCF restarts (the
Sensing API retries automatically). PKI/TLS fields are declared metadata
only; production deployments must terminate TLS and enforce authorisation
outside this process.

Run:
    python -m sensing_api.capif_core            # uses config/sensing_api_config.json
    python -m sensing_api.capif_core --port 8080
"""

from __future__ import annotations

import argparse
import logging
import threading
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import load_settings
from .schemas import utc_now_iso

log = logging.getLogger("capif.ccf")


# ─────────────────────────────────────────────
# CAPIF models (subset of 3GPP TS 29.222)
# ─────────────────────────────────────────────

class OnboardingInfo(BaseModel):
    apiInvokerPublicKey: str = "placeholder-key"
    onboardingSecret: str | None = None
    apiInvokerCertificate: str | None = None


class ApiInvokerEnrolmentDetails(BaseModel):
    apiInvokerId: str | None = None
    apiList: list[str] | None = None
    onboardingInformation: OnboardingInfo | None = None
    notificationDestination: str | None = None
    requestTestNotification: bool | None = False


class Resource(BaseModel):
    resourceName: str
    uri: str
    commType: str = "REQUEST_RESPONSE"
    operations: list[str] | None = None
    description: str | None = None


class ApiVersion(BaseModel):
    apiVersion: str = "v1"
    expiry: str | None = None
    resources: list[Resource] | None = None


class AefProfile(BaseModel):
    aefId: str
    versions: list[ApiVersion]
    protocol: str = "HTTP_1_1"
    securityMethods: list[str] = ["PKI"]
    domainName: str | None = None
    interfaceDescriptions: list[dict] | None = None


class ServiceAPIDescription(BaseModel):
    apiId: str | None = None
    apiName: str
    apiVersion: str = "v1"
    description: str | None = None
    aefProfiles: list[AefProfile]
    serviceAPICategory: str | None = None
    apiSuppFeats: str | None = None
    pubDate: str | None = None
    apiStatus: dict | None = None


class ProviderEnrolmentDetails(BaseModel):
    apiProvDomId: str | None = None
    apiProvDomInfo: str | None = None
    apiProvFuncs: list[dict] | None = None
    regSec: str = "placeholder-reg-sec"


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class CapifRegistry:
    """Thread-safe in-memory CAPIF registry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.providers: dict[str, dict] = {}
        self.invokers: dict[str, dict] = {}
        self.service_apis: dict[str, dict] = {}

    def clear(self) -> None:
        with self._lock:
            self.providers.clear()
            self.invokers.clear()
            self.service_apis.clear()


def create_app(registry: CapifRegistry | None = None) -> FastAPI:
    registry = registry or CapifRegistry()
    app = FastAPI(
        title="CAPIF Core Function",
        description="3GPP TS 29.222 CAPIF Core Function for the passive sensing deployment",
        version="1.0.0",
        docs_url="/capif/v1/ui",
    )
    app.state.registry = registry

    # ── Provider enrolment (APF / AEF registration) ─────────────────────────

    @app.post("/capif/v1/api-provider-enrolments", status_code=201)
    def enrol_provider(details: ProviderEnrolmentDetails):
        record = {
            "apiProvDomId": _new_id("PROV"),
            "apfId": _new_id("APF"),
            "aefId": _new_id("AEF"),
            "apiProvDomInfo": details.apiProvDomInfo or "Sensing provider",
            "registeredAt": utc_now_iso(),
        }
        with registry._lock:
            registry.providers[record["apfId"]] = record
        log.info("Provider enrolled: apfId=%s aefId=%s", record["apfId"], record["aefId"])
        return record

    @app.get("/capif/v1/api-provider-enrolments")
    def list_providers():
        with registry._lock:
            return list(registry.providers.values())

    # ── Invoker enrolment ───────────────────────────────────────────────────

    @app.post("/capif/v1/api-invoker-enrolments", status_code=201)
    def enrol_invoker(details: ApiInvokerEnrolmentDetails):
        record = {
            "apiInvokerId": _new_id("INV"),
            "apiList": details.apiList or [],
            "onboardingInformation": (
                details.onboardingInformation.model_dump() if details.onboardingInformation else {}
            ),
            "notificationDestination": details.notificationDestination,
            "registeredAt": utc_now_iso(),
        }
        with registry._lock:
            registry.invokers[record["apiInvokerId"]] = record
        log.info("Invoker enrolled: id=%s", record["apiInvokerId"])
        return record

    @app.get("/capif/v1/api-invoker-enrolments")
    def list_invokers():
        with registry._lock:
            return list(registry.invokers.values())

    @app.delete("/capif/v1/api-invoker-enrolments/{invoker_id}", status_code=204)
    def offboard_invoker(invoker_id: str):
        with registry._lock:
            if registry.invokers.pop(invoker_id, None) is None:
                raise HTTPException(404, "Invoker not found")

    # ── Publish service API (APF → CCF) ─────────────────────────────────────

    def _require_provider(apf_id: str) -> None:
        if apf_id not in registry.providers:
            raise HTTPException(404, f"Unknown apfId {apf_id!r}; enrol the provider first")

    @app.post("/capif/v1/apf-publish-service-apis/{apfId}/service-apis", status_code=201)
    def publish_service_api(apfId: str, api_desc: ServiceAPIDescription):
        with registry._lock:
            _require_provider(apfId)
            api_desc.apiId = _new_id("API")
            api_desc.pubDate = utc_now_iso()
            record = api_desc.model_dump()
            record["apfId"] = apfId
            registry.service_apis[api_desc.apiId] = record
        log.info("API published: apiId=%s name=%s apfId=%s", api_desc.apiId, api_desc.apiName, apfId)
        return record

    @app.get("/capif/v1/apf-publish-service-apis/{apfId}/service-apis")
    def list_published_apis(apfId: str):
        with registry._lock:
            return [v for v in registry.service_apis.values() if v.get("apfId") == apfId]

    @app.put("/capif/v1/apf-publish-service-apis/{apfId}/service-apis/{serviceApiId}")
    def update_published_api(apfId: str, serviceApiId: str, api_desc: ServiceAPIDescription):
        with registry._lock:
            existing = registry.service_apis.get(serviceApiId)
            if existing is None or existing.get("apfId") != apfId:
                raise HTTPException(404, "Service API not found")
            record = api_desc.model_dump()
            record.update({"apfId": apfId, "apiId": serviceApiId, "pubDate": utc_now_iso()})
            registry.service_apis[serviceApiId] = record
        return record

    @app.delete("/capif/v1/apf-publish-service-apis/{apfId}/service-apis/{serviceApiId}", status_code=204)
    def unpublish_api(apfId: str, serviceApiId: str):
        with registry._lock:
            existing = registry.service_apis.get(serviceApiId)
            if existing is None or existing.get("apfId") != apfId:
                raise HTTPException(404, "Service API not found")
            del registry.service_apis[serviceApiId]

    # ── Discover service APIs (invoker → CCF) ───────────────────────────────

    @app.get("/capif/v1/service-apis")
    def discover_service_apis(
        api_name: str | None = None,
        api_version: str | None = None,
        api_category: str | None = None,
        aef_id: str | None = None,
    ):
        with registry._lock:
            results = list(registry.service_apis.values())
        if api_name:
            results = [a for a in results if a["apiName"].lower() == api_name.lower()]
        if api_version:
            results = [a for a in results if a.get("apiVersion") == api_version]
        if api_category:
            results = [a for a in results if a.get("serviceAPICategory") == api_category]
        if aef_id:
            results = [a for a in results if any(p.get("aefId") == aef_id for p in a.get("aefProfiles") or [])]
        return {"serviceAPIDescriptions": results, "count": len(results)}

    @app.get("/capif/v1/service-apis/{serviceApiId}")
    def get_service_api(serviceApiId: str):
        with registry._lock:
            if serviceApiId not in registry.service_apis:
                raise HTTPException(404, "Service API not found")
            return registry.service_apis[serviceApiId]

    # ── Routing info (invoker → CCF: where to call) ─────────────────────────

    @app.get("/capif/v1/routing-info/{serviceApiId}")
    def get_routing_info(serviceApiId: str):
        with registry._lock:
            api = registry.service_apis.get(serviceApiId)
        if api is None:
            raise HTTPException(404, "Service API not found")
        routing = [
            {
                "aefId": profile.get("aefId"),
                "aefLocation": (profile.get("interfaceDescriptions") or [{}])[0],
                "versions": profile.get("versions", []),
                "protocol": profile.get("protocol", "HTTP_1_1"),
                "securityMethods": profile.get("securityMethods", []),
            }
            for profile in api.get("aefProfiles", [])
        ]
        return {"apiId": serviceApiId, "apiName": api.get("apiName"), "routingProfiles": routing}

    # ── Health ──────────────────────────────────────────────────────────────

    @app.get("/capif/v1/health")
    def health():
        with registry._lock:
            counts = len(registry.providers), len(registry.invokers), len(registry.service_apis)
        return {
            "status": "ok",
            "component": "CAPIF Core Function",
            "spec": "3GPP TS 29.222 / ETSI TS 123 222",
            "providers": counts[0],
            "invokers": counts[1],
            "service_apis": counts[2],
            "timestamp": utc_now_iso(),
        }

    return app


app = create_app()


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Run the CAPIF Core Function.")
    parser.add_argument("--host", default=settings.capif.host)
    parser.add_argument("--port", type=int, default=settings.capif.port)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [CCF] %(levelname)s %(message)s")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
