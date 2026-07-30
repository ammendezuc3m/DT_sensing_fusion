#!/usr/bin/env python3
"""
Scan supported 5G NR FR1 bands for Synchronization Signal Blocks.

Current supported signal configuration:
    - SSB subcarrier spacing: 30 kHz
    - Sample rate: 15.36 Msps
    - FFT size: 512
    - Global synchronization raster spacing above 3 GHz: 1.44 MHz

Supported bands:
    - n77: 3300-4200 MHz
    - n78: 3300-3800 MHz
    - n79: 4400-5000 MHz

The scanner tunes directly to every candidate SSREF frequency in the selected
band and runs the repository's existing PSS detector.

This detects:
    - candidate SSB center frequency;
    - PSS NID2;
    - timing offset;
    - normalized PSS correlation metric;
    - repeatability across captures.

Important:
    PSS and NID2 alone do not uniquely identify a cell. Definitive cell
    identification requires SSS decoding to obtain NID1 and the full PCI,
    and optionally PBCH/MIB decoding.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import uhd


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from extract_datassb_offline import detect_best_pss_timing  # noqa: E402
from profile_online_datassb_pipeline import (  # noqa: E402
    capture_one_block,
    make_rx_streamer,
)


# Frequency limits in MHz.
BAND_RANGES_MHZ: dict[str, tuple[float, float]] = {
    "n77": (3300.0, 4200.0),
    "n78": (3300.0, 3800.0),
    "n79": (4400.0, 5000.0),
}

# Global synchronization raster used from 3 GHz to 24.25008 GHz:
#
#     SSREF = 3000 MHz + N * 1.44 MHz
#
SSREF_BASE_MHZ = 3000.0
SSREF_STEP_MHZ = 1.44


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan n77/n78/n79 synchronization-raster frequencies and detect "
            "5G NR PSS/SSB candidates."
        )
    )

    parser.add_argument(
        "--band",
        "--channel",
        dest="band",
        choices=["n77", "n78", "n79", "all"],
        default="all",
        help=(
            "5G NR band to scan. If omitted, scans all currently supported "
            "bands. '--channel' is accepted as an alias."
        ),
    )

    parser.add_argument(
        "--serial",
        default="",
        help=(
            "USRP serial number. If omitted, UHD selects the available device."
        ),
    )
    parser.add_argument(
        "--rx-channel",
        type=int,
        default=0,
        help="Physical USRP RX channel.",
    )
    parser.add_argument(
        "--antenna",
        default="",
        help="Optional USRP RX antenna name.",
    )

    parser.add_argument(
        "--rate",
        type=float,
        default=15.36e6,
        help="USRP sample rate in samples/s.",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=60.0,
        help="USRP RX gain in dB.",
    )
    parser.add_argument(
        "--duration-ms",
        type=float,
        default=40.0,
        help="Duration of every IQ capture block in milliseconds.",
    )
    parser.add_argument(
        "--settle-sec",
        type=float,
        default=0.04,
        help="Wait time after every frequency retune.",
    )
    parser.add_argument(
        "--captures-per-frequency",
        type=int,
        default=2,
        help="Number of independent captures at every raster frequency.",
    )

    parser.add_argument(
        "--nfft",
        type=int,
        default=512,
        help="FFT size used by the existing 30 kHz PSS detector.",
    )
    parser.add_argument(
        "--nrb-ssb",
        type=int,
        default=20,
        help="SSB reference bandwidth in resource blocks.",
    )
    parser.add_argument(
        "--force-nid2",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help=(
            "Only test one known PSS NID2. If omitted, test NID2 0, 1 and 2."
        ),
    )

    parser.add_argument(
        "--min-pss-metric",
        type=float,
        default=0.50,
        help="Minimum PSS correlation metric for one capture to count as a hit.",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=2,
        help=(
            "Minimum successful captures at one frequency for a confirmed "
            "candidate."
        ),
    )

    parser.add_argument(
        "--expected-frequency-mhz",
        type=float,
        default=None,
        help="Optional expected SSB center frequency used to mark likely matches.",
    )
    parser.add_argument(
        "--expected-nid2",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="Optional expected NID2 used to mark likely matches.",
    )
    parser.add_argument(
        "--frequency-tolerance-khz",
        type=float,
        default=100.0,
        help="Tolerance for --expected-frequency-mhz matching.",
    )

    parser.add_argument(
        "--start-mhz",
        type=float,
        default=None,
        help="Optional lower frequency limit inside the selected band.",
    )
    parser.add_argument(
        "--stop-mhz",
        type=float,
        default=None,
        help="Optional upper frequency limit inside the selected band.",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of strongest scan positions printed at the end.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/ssb_scan",
        help="Output directory.",
    )
    parser.add_argument(
        "--prefix",
        default="ssb_scan",
        help="Output filename prefix.",
    )

    args = parser.parse_args()

    if args.captures_per_frequency < 1:
        parser.error("--captures-per-frequency must be at least 1")

    if args.min_hits < 1:
        parser.error("--min-hits must be at least 1")

    if args.min_hits > args.captures_per_frequency:
        parser.error(
            "--min-hits cannot exceed --captures-per-frequency"
        )

    if args.duration_ms <= 0:
        parser.error("--duration-ms must be positive")

    if args.start_mhz is not None and args.stop_mhz is not None:
        if args.start_mhz > args.stop_mhz:
            parser.error("--start-mhz cannot exceed --stop-mhz")

    return args


def ssref_frequencies_for_range(
    low_mhz: float,
    high_mhz: float,
) -> list[float]:
    """Return all 1.44 MHz SSREF raster frequencies inside a range."""

    first_n = math.ceil(
        (low_mhz - SSREF_BASE_MHZ) / SSREF_STEP_MHZ - 1e-12
    )
    last_n = math.floor(
        (high_mhz - SSREF_BASE_MHZ) / SSREF_STEP_MHZ + 1e-12
    )

    return [
        round(SSREF_BASE_MHZ + n * SSREF_STEP_MHZ, 6)
        for n in range(first_n, last_n + 1)
    ]


def selected_bands(band: str) -> list[str]:
    if band == "all":
        return list(BAND_RANGES_MHZ)

    return [band]


def bands_containing_frequency(
    frequency_mhz: float,
    bands: list[str],
) -> list[str]:
    matches = []

    for band in bands:
        low_mhz, high_mhz = BAND_RANGES_MHZ[band]

        if low_mhz <= frequency_mhz <= high_mhz:
            matches.append(band)

    return matches


def build_scan_frequencies(args: argparse.Namespace) -> list[dict[str, Any]]:
    bands = selected_bands(args.band)
    frequencies: dict[float, set[str]] = {}

    for band in bands:
        band_low_mhz, band_high_mhz = BAND_RANGES_MHZ[band]

        low_mhz = band_low_mhz
        high_mhz = band_high_mhz

        if args.start_mhz is not None:
            low_mhz = max(low_mhz, args.start_mhz)

        if args.stop_mhz is not None:
            high_mhz = min(high_mhz, args.stop_mhz)

        if low_mhz > high_mhz:
            continue

        for frequency_mhz in ssref_frequencies_for_range(
            low_mhz,
            high_mhz,
        ):
            frequencies.setdefault(frequency_mhz, set()).add(band)

    return [
        {
            "frequency_mhz": frequency_mhz,
            "bands": sorted(frequency_bands),
        }
        for frequency_mhz, frequency_bands in sorted(frequencies.items())
    ]


def configure_usrp(args: argparse.Namespace):
    device_args = f"serial={args.serial}" if args.serial else ""

    print("=== USRP configuration ===")
    usrp = uhd.usrp.MultiUSRP(device_args)

    channel = args.rx_channel

    usrp.set_rx_rate(args.rate, channel)
    usrp.set_rx_gain(args.gain, channel)

    if args.antenna:
        usrp.set_rx_antenna(args.antenna, channel)

    actual_rate = float(usrp.get_rx_rate(channel))

    print(f"Motherboard:       {usrp.get_mboard_name()}")
    print(f"RX channel:        {channel}")
    print(f"Requested rate:    {args.rate / 1e6:.6f} Msps")
    print(f"Actual rate:       {actual_rate / 1e6:.6f} Msps")
    print(f"Requested gain:    {args.gain:.2f} dB")
    print(f"Actual gain:       {usrp.get_rx_gain(channel):.2f} dB")

    if args.antenna:
        print(f"RX antenna:        {usrp.get_rx_antenna(channel)}")

    return usrp, actual_rate


def calculate_power_dbfs(waveform: np.ndarray) -> float:
    power = float(np.mean(np.abs(waveform) ** 2))
    return float(10.0 * np.log10(max(power, 1e-15)))


def expected_match(
    frequency_mhz: float,
    nid2: int,
    args: argparse.Namespace,
) -> bool | None:
    checks: list[bool] = []

    if args.expected_frequency_mhz is not None:
        error_khz = abs(
            frequency_mhz - args.expected_frequency_mhz
        ) * 1000.0

        checks.append(error_khz <= args.frequency_tolerance_khz)

    if args.expected_nid2 is not None:
        checks.append(nid2 == args.expected_nid2)

    if not checks:
        return None

    return all(checks)


def scan_frequency(
    usrp,
    rx_streamer,
    max_samps: int,
    actual_rate: float,
    frequency_mhz: float,
    bands: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    requested_frequency_hz = frequency_mhz * 1e6

    usrp.set_rx_freq(
        uhd.types.TuneRequest(requested_frequency_hz),
        args.rx_channel,
    )

    time.sleep(args.settle_sec)

    actual_frequency_hz = float(
        usrp.get_rx_freq(args.rx_channel)
    )
    actual_frequency_mhz = actual_frequency_hz / 1e6

    samples_per_block = int(
        round(actual_rate * args.duration_ms * 1e-3)
    )

    capture_rows: list[dict[str, Any]] = []

    for capture_index in range(args.captures_per_frequency):
        row: dict[str, Any] = {
            "capture_index": capture_index,
            "valid": False,
            "nid2": -1,
            "metric": float("nan"),
            "timing_offset_samples": -1,
            "timing_offset_ms": float("nan"),
            "power_dbfs": float("nan"),
            "error": "",
        }

        try:
            waveform = capture_one_block(
                rx_streamer=rx_streamer,
                total_samples=samples_per_block,
                max_samps=max_samps,
            )

            power_dbfs = calculate_power_dbfs(waveform)

            timing_info = detect_best_pss_timing(
                waveform=waveform,
                nfft=args.nfft,
                nrb_ssb=args.nrb_ssb,
                force_nid2=args.force_nid2,
                sample_rate=actual_rate,
            )

            metric = float(timing_info["metric"])
            valid = metric >= args.min_pss_metric

            row.update(
                {
                    "valid": valid,
                    "nid2": int(timing_info["nid2"]),
                    "metric": metric,
                    "timing_offset_samples": int(
                        timing_info["timing_offset_samples"]
                    ),
                    "timing_offset_ms": float(
                        timing_info["timing_offset_ms"]
                    ),
                    "power_dbfs": power_dbfs,
                }
            )

        except Exception as exc:
            row["error"] = str(exc)

        capture_rows.append(row)

    valid_rows = [
        row for row in capture_rows
        if row["valid"]
    ]

    hits = len(valid_rows)

    all_metrics = np.asarray(
        [
            row["metric"]
            for row in capture_rows
            if np.isfinite(row["metric"])
        ],
        dtype=np.float64,
    )

    all_powers = np.asarray(
        [
            row["power_dbfs"]
            for row in capture_rows
            if np.isfinite(row["power_dbfs"])
        ],
        dtype=np.float64,
    )

    if valid_rows:
        nid2_values = [int(row["nid2"]) for row in valid_rows]
        nid2 = max(set(nid2_values), key=nid2_values.count)
    else:
        nid2 = -1

    confirmed = hits >= args.min_hits

    result = {
        "requested_frequency_mhz": frequency_mhz,
        "actual_frequency_mhz": actual_frequency_mhz,
        "bands": bands,
        "hits": hits,
        "captures": args.captures_per_frequency,
        "confirmed": confirmed,
        "nid2": nid2,
        "metric_max": (
            float(np.max(all_metrics))
            if all_metrics.size
            else float("nan")
        ),
        "metric_median": (
            float(np.median(all_metrics))
            if all_metrics.size
            else float("nan")
        ),
        "power_dbfs_median": (
            float(np.median(all_powers))
            if all_powers.size
            else float("nan")
        ),
        "expected_match": (
            expected_match(
                frequency_mhz=frequency_mhz,
                nid2=nid2,
                args=args,
            )
            if confirmed
            else False
        ),
        "capture_results": capture_rows,
    }

    return result


def csv_safe(value: Any) -> Any:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)

    if value is None:
        return ""

    return value


def main() -> None:
    args = parse_args()

    scan_frequencies = build_scan_frequencies(args)

    if not scan_frequencies:
        raise RuntimeError(
            "No synchronization-raster frequencies fall inside the "
            "selected scan range."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"{args.prefix}_{timestamp}.csv"
    out_json = out_dir / f"{args.prefix}_{timestamp}.json"

    print()
    print("=== 5G NR SSB scan ===")
    print(f"Selected band:          {args.band}")
    print(
        "Supported bands:       "
        + ", ".join(selected_bands(args.band))
    )
    print(f"Raster frequencies:     {len(scan_frequencies)}")
    print(f"Captures/frequency:     {args.captures_per_frequency}")
    print(f"Minimum hits:           {args.min_hits}")
    print(f"Minimum PSS metric:     {args.min_pss_metric:.3f}")
    print(f"Capture duration:       {args.duration_ms:.1f} ms")
    print(f"Output CSV:             {out_csv}")
    print(f"Output JSON:            {out_json}")
    print()

    usrp, actual_rate = configure_usrp(args)
    rx_streamer = make_rx_streamer(
        usrp,
        args.rx_channel,
    )
    max_samps = rx_streamer.get_max_num_samps()

    results: list[dict[str, Any]] = []

    scan_start = time.perf_counter()

    for index, item in enumerate(scan_frequencies, start=1):
        frequency_mhz = float(item["frequency_mhz"])
        bands = list(item["bands"])

        result = scan_frequency(
            usrp=usrp,
            rx_streamer=rx_streamer,
            max_samps=max_samps,
            actual_rate=actual_rate,
            frequency_mhz=frequency_mhz,
            bands=bands,
            args=args,
        )

        results.append(result)

        status = "SSB" if result["confirmed"] else "---"

        expected_text = ""
        if result["expected_match"] is True:
            expected_text = " EXPECTED_MATCH"
        elif result["expected_match"] is False and (
            args.expected_frequency_mhz is not None
            or args.expected_nid2 is not None
        ):
            expected_text = " OTHER"

        print(
            f"[{index:04d}/{len(scan_frequencies):04d}] "
            f"{frequency_mhz:10.3f} MHz "
            f"{'/'.join(bands):8s} "
            f"{status} "
            f"hits={result['hits']}/{result['captures']} "
            f"NID2={result['nid2']:2d} "
            f"metric={result['metric_max']:.3f} "
            f"power={result['power_dbfs_median']:.1f} dBFS"
            f"{expected_text}"
        )

    elapsed_sec = time.perf_counter() - scan_start

    candidates = [
        result for result in results
        if result["confirmed"]
    ]

    candidates.sort(
        key=lambda row: (
            row["hits"],
            row["metric_median"],
            row["metric_max"],
        ),
        reverse=True,
    )

    ranked_positions = sorted(
        results,
        key=lambda row: row["metric_max"],
        reverse=True,
    )

    csv_fields = [
        "requested_frequency_mhz",
        "actual_frequency_mhz",
        "bands",
        "hits",
        "captures",
        "confirmed",
        "nid2",
        "metric_max",
        "metric_median",
        "power_dbfs_median",
        "expected_match",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    field: csv_safe(result.get(field))
                    for field in csv_fields
                }
            )

    summary = {
        "created_at": timestamp,
        "configuration": vars(args),
        "actual_sample_rate_hz": actual_rate,
        "raster_base_mhz": SSREF_BASE_MHZ,
        "raster_step_mhz": SSREF_STEP_MHZ,
        "num_frequencies_scanned": len(results),
        "num_confirmed_candidates": len(candidates),
        "elapsed_seconds": elapsed_sec,
        "confirmed_candidates": candidates,
        "all_results": results,
    }

    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            allow_nan=True,
        )

    print()
    print("=== Confirmed SSB candidates ===")

    if not candidates:
        print("No confirmed SSB candidates were found.")
    else:
        for rank, result in enumerate(candidates, start=1):
            expected_text = (
                " [EXPECTED MATCH]"
                if result["expected_match"] is True
                else ""
            )

            print(
                f"{rank:2d}. "
                f"{result['requested_frequency_mhz']:.3f} MHz "
                f"bands={','.join(result['bands'])} "
                f"NID2={result['nid2']} "
                f"hits={result['hits']}/{result['captures']} "
                f"metric_max={result['metric_max']:.3f} "
                f"metric_median={result['metric_median']:.3f}"
                f"{expected_text}"
            )

    print()
    print(f"=== Top {min(args.top, len(ranked_positions))} scan positions ===")

    for rank, result in enumerate(
        ranked_positions[: args.top],
        start=1,
    ):
        print(
            f"{rank:2d}. "
            f"{result['requested_frequency_mhz']:.3f} MHz "
            f"bands={','.join(result['bands'])} "
            f"NID2={result['nid2']} "
            f"hits={result['hits']}/{result['captures']} "
            f"metric_max={result['metric_max']:.3f} "
            f"power={result['power_dbfs_median']:.1f} dBFS"
        )

    print()
    print(f"Elapsed: {elapsed_sec:.1f} seconds")
    print(f"CSV:     {out_csv}")
    print(f"JSON:    {out_json}")


if __name__ == "__main__":
    main()
