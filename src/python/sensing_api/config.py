"""Configuration for the CAPIF Core Function and the Sensing API.

Values come from ``config/sensing_api_config.json`` (or the file named by
``SENSING_API_CONFIG``) and can be overridden by environment variables:

=============================  =========================================
``SENSING_CAPIF_URL``          CAPIF base URL, e.g. http://host:8080/capif/v1
``SENSING_API_HOST``           bind address of the Sensing API
``SENSING_API_PORT``           port of the Sensing API
``SENSING_API_ADVERTISED_HOST`` host published in CAPIF for invokers
``SENSING_API_PRODUCER_TOKEN`` bearer token required to POST reports
=============================  =========================================
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "sensing_api_config.json"


@dataclass(frozen=True)
class CapifSettings:
    host: str = "0.0.0.0"
    port: int = 8080
    base_url: str = "http://127.0.0.1:8080/capif/v1"


@dataclass(frozen=True)
class SensingApiSettings:
    host: str = "0.0.0.0"
    port: int = 8090
    advertised_host: str = "127.0.0.1"
    history_size: int = 1000
    register_with_capif: bool = True
    producer_token: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.advertised_host}:{self.port}"


@dataclass(frozen=True)
class Settings:
    capif: CapifSettings
    sensing_api: SensingApiSettings


def load_settings(path: str | Path | None = None) -> Settings:
    path = Path(path or os.environ.get("SENSING_API_CONFIG", DEFAULT_CONFIG_PATH))
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    capif_raw = data.get("capif", {})
    api_raw = data.get("sensing_api", {})
    env = os.environ

    capif = CapifSettings(
        host=str(capif_raw.get("host", CapifSettings.host)),
        port=int(capif_raw.get("port", CapifSettings.port)),
        base_url=env.get("SENSING_CAPIF_URL", capif_raw.get("base_url", CapifSettings.base_url)).rstrip("/"),
    )
    sensing_api = SensingApiSettings(
        host=env.get("SENSING_API_HOST", api_raw.get("host", SensingApiSettings.host)),
        port=int(env.get("SENSING_API_PORT", api_raw.get("port", SensingApiSettings.port))),
        advertised_host=env.get(
            "SENSING_API_ADVERTISED_HOST", api_raw.get("advertised_host", SensingApiSettings.advertised_host)
        ),
        history_size=int(api_raw.get("history_size", SensingApiSettings.history_size)),
        register_with_capif=bool(api_raw.get("register_with_capif", True)),
        producer_token=env.get("SENSING_API_PRODUCER_TOKEN") or api_raw.get("producer_token") or None,
    )
    if sensing_api.history_size < 1:
        raise ValueError("sensing_api.history_size must be >= 1")
    return Settings(capif=capif, sensing_api=sensing_api)
