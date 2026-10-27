"""CAPIF helpers: provider registration (AEF side) and API discovery (invoker side).

Adapted from ``prime6g_dt/clients/capif_client.py`` and the self-registration
in ``prime6g_dt/dt_server/main.py``.

Invoker usage::

    from sensing_api.capif_client import CapifInvoker
    with CapifInvoker("http://ccf-host:8080/capif/v1") as capif:
        capif.enrol("my-digital-twin")
        base_url = capif.resolve_base_url("sensing-api")   # http://host:8090
"""

from __future__ import annotations

import logging

import httpx

from .schemas import API_NAME, API_PREFIX, API_VERSION

log = logging.getLogger("sensing_api.capif")


class CapifError(RuntimeError):
    pass


def sensing_service_api_description(aef_id: str, host: str, port: int) -> dict:
    """ServiceAPIDescription published to CAPIF for the Sensing API."""
    def resource(name: str, uri: str, ops: list[str], description: str, comm: str = "REQUEST_RESPONSE") -> dict:
        return {"resourceName": name, "uri": f"{API_PREFIX}{uri}", "commType": comm,
                "operations": ops, "description": description}

    return {
        "apiName": API_NAME,
        "apiVersion": API_VERSION,
        "description": "Passive RF sensing results: detected objects with type, coordinates, "
                       "confidence and source sensor (5G SSB, WiFi CSI, ...)",
        "serviceAPICategory": "sensing",
        "aefProfiles": [{
            "aefId": aef_id,
            "protocol": "HTTP_1_1",
            "securityMethods": ["PKI"],
            "interfaceDescriptions": [{"ipv4Addr": host, "port": port, "securityMethods": ["PKI"]}],
            "versions": [{
                "apiVersion": API_VERSION,
                "resources": [
                    resource("detections", "/detections", ["GET"],
                             "Current detections of every sensor (filters: object_type, sensor_id, "
                             "min_confidence, max_age_s)"),
                    resource("reports", "/reports", ["GET", "POST"],
                             "Publish a sensing report (producers) or read the report history"),
                    resource("sensor-collection", "/sensors", ["GET"], "Sensors that have published"),
                    resource("sensor-latest", "/sensors/{sensorId}/latest", ["GET"],
                             "Latest report of one sensor"),
                    resource("report-stream", "/stream", ["GET"],
                             "Server-Sent Events stream of new reports", comm="SUBSCRIBE_NOTIFY"),
                ],
            }],
        }],
    }


class CapifProvider:
    """Enrols the Sensing API as provider and publishes its description."""

    def __init__(self, ccf_base_url: str, timeout: float = 5.0):
        self.ccf = ccf_base_url.rstrip("/")
        self.timeout = timeout
        self.apf_id: str | None = None
        self.aef_id: str | None = None
        self.api_id: str | None = None

    def register(self, host: str, port: int, provider_info: str = "Passive sensing provider") -> str:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.ccf}/api-provider-enrolments",
                               json={"apiProvDomInfo": provider_info, "regSec": "sensing-reg-sec"})
            if resp.status_code != 201:
                raise CapifError(f"Provider enrolment failed: HTTP {resp.status_code} {resp.text}")
            data = resp.json()
            self.apf_id, self.aef_id = data["apfId"], data["aefId"]
            self.api_id = None

            desc = sensing_service_api_description(self.aef_id, host, port)
            resp = client.post(f"{self.ccf}/apf-publish-service-apis/{self.apf_id}/service-apis", json=desc)
            if resp.status_code != 201:
                raise CapifError(f"Service API publication failed: HTTP {resp.status_code} {resp.text}")
            self.api_id = resp.json()["apiId"]
        log.info("Published %s to CAPIF: apfId=%s apiId=%s", API_NAME, self.apf_id, self.api_id)
        return self.api_id

    def is_published(self) -> bool:
        """True if the CCF still knows our service API (False after a CCF restart)."""
        if not self.api_id:
            return False
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.ccf}/service-apis/{self.api_id}")
        if resp.status_code == 404:
            log.warning("Service API %s no longer in CAPIF; registering again", self.api_id)
            self.api_id = None
            return False
        resp.raise_for_status()
        return True

    def unregister(self) -> None:
        if not (self.apf_id and self.api_id):
            return
        try:
            with httpx.Client(timeout=self.timeout) as client:
                client.delete(f"{self.ccf}/apf-publish-service-apis/{self.apf_id}/service-apis/{self.api_id}")
        except httpx.HTTPError as exc:
            log.warning("Could not unpublish %s: %s", self.api_id, exc)
        self.api_id = None


class CapifInvoker:
    """Thin invoker client: enrol, discover and resolve the AEF base URL."""

    def __init__(self, ccf_base_url: str, timeout: float = 5.0):
        self.ccf = ccf_base_url.rstrip("/")
        self.invoker_id: str | None = None
        self._http = httpx.Client(timeout=timeout)

    def enrol(self, app_name: str) -> str:
        resp = self._http.post(f"{self.ccf}/api-invoker-enrolments", json={
            "onboardingInformation": {"apiInvokerPublicKey": f"pubkey-{app_name}"},
            "apiList": [API_NAME],
        })
        resp.raise_for_status()
        self.invoker_id = resp.json()["apiInvokerId"]
        return self.invoker_id

    def offboard(self) -> None:
        if self.invoker_id:
            self._http.delete(f"{self.ccf}/api-invoker-enrolments/{self.invoker_id}")
            self.invoker_id = None

    def discover(self, api_name: str = API_NAME) -> list[dict]:
        resp = self._http.get(f"{self.ccf}/service-apis", params={"api_name": api_name})
        resp.raise_for_status()
        return [a for a in resp.json().get("serviceAPIDescriptions", []) if a.get("apiName") == api_name]

    def routing_info(self, api_id: str) -> dict:
        resp = self._http.get(f"{self.ccf}/routing-info/{api_id}")
        resp.raise_for_status()
        return resp.json()

    def resolve_base_url(self, api_name: str = API_NAME) -> str:
        """Return ``http(s)://host:port`` of the most recently published instance."""
        apis = sorted(self.discover(api_name), key=lambda a: a.get("pubDate") or "")
        if not apis:
            raise CapifError(f"No {api_name!r} published in CAPIF at {self.ccf}")
        for profile in self.routing_info(apis[-1]["apiId"]).get("routingProfiles", []):
            location = profile.get("aefLocation") or {}
            host = location.get("ipv4Addr") or location.get("fqdn") or location.get("ipv6Addr")
            port = location.get("port")
            if host and port:
                scheme = "https" if "TLS" in str(location.get("securityMethods", [])) else "http"
                return f"{scheme}://{host}:{port}"
        raise CapifError(f"{api_name!r} has no usable interface description in CAPIF")

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "CapifInvoker":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
