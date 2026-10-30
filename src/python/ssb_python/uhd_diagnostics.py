"""Per-run native UHD logs and Python radio diagnostics (UTC)."""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def start_diagnostics(args):
    directory = Path(args.uhd_log_dir) if args.uhd_log_dir else Path(args.log_csv).parent / "uhd_logs"
    directory.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + f"_{os.getpid()}"
    native = (directory / f"{run_id}_uhd.log").resolve()
    diagnostics = directory / f"{run_id}_receiver.log"
    # Set both thresholds before importing UHD; do not change console verbosity.
    os.environ["UHD_LOG_FILE"] = str(native)
    os.environ["UHD_LOG_LEVEL"] = args.uhd_log_level
    os.environ["UHD_LOG_FILE_LEVEL"] = args.uhd_log_level
    logger = logging.getLogger("ssb.uhd")
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(diagnostics, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Keep the handler reference so repeated main() calls do not leak files.
    logger._ssb_handler = handler
    logger.info("Session start: Python=%s platform=%s", sys.version, platform.platform())
    # Never log API tokens or all environment variables.
    radio_keys = ("serial", "freq", "rate", "gain", "duration_ms", "channel", "antenna",
                  "nfft", "force_nid2", "enable_cfo_correction", "manual_cfo_hz",
                  "cfo_warmup_iters", "cfo_correction_sign", "fisher_pci", "fisher_ibar")
    logger.info("Requested radio configuration: %s", json.dumps(
        {key: getattr(args, key, None) for key in radio_keys}, sort_keys=True))
    logger.info("Native UHD log: %s (level=%s)", native, args.uhd_log_level)
    if "uhd" in sys.modules:
        logger.warning("UHD already imported: native file settings may require a fresh process")
    print(f"UHD native log:     {native}")
    print(f"Receiver log:       {diagnostics.resolve()}")
    return logger


def stop_diagnostics(logger):
    logger.info("Session end")
    handler = logger._ssb_handler
    logger.removeHandler(handler)
    handler.close()
