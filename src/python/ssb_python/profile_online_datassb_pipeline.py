#!/usr/bin/env python3
"""
Profile the live Python UHD -> PSS -> OFDM -> dataSSB/rxGridSSB pipeline.

Goal:
    Measure how often we can produce one valid rxGridSSB sample.

Measured times:
    capture_time_ms
    pss_time_ms
    ofdm_time_ms
    total_dsp_time_ms
    total_loop_time_ms
    samples_per_second

This script does not write raw IQ blocks to disk. It only captures, processes,
prints timing information, and writes a lightweight CSV summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import uhd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from extract_datassb_offline import (  # noqa: E402
    detect_best_pss_timing,
    ofdm_demodulate_centered,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile live Python UHD capture + PSS + OFDM dataSSB extraction."
    )

    parser.add_argument("--serial", default="34B73C3", help="USRP serial.")
    parser.add_argument("--freq", type=float, default=3541.44e6, help="Center frequency in Hz.")
    parser.add_argument("--rate", type=float, default=15.36e6, help="Sample rate in Hz.")
    parser.add_argument("--gain", type=float, default=60.0, help="RX gain in dB.")
    parser.add_argument("--duration-ms", type=float, default=20.0, help="Capture duration per block.")
    parser.add_argument("--channel", type=int, default=0, help="RX channel.")
    parser.add_argument("--antenna", default="", help="Optional RX antenna name.")
    parser.add_argument("--settle-sec", type=float, default=0.5, help="Initial settle time.")

    parser.add_argument("--num-iters", type=int, default=100, help="Number of live iterations.")
    parser.add_argument("--warmup-iters", type=int, default=5, help="Iterations excluded from summary.")
    parser.add_argument("--progress-every", type=int, default=1, help="Print every N iterations.")

    parser.add_argument("--nfft", type=int, default=512, help="OFDM FFT size.")
    parser.add_argument("--demod-rb", type=int, default=30, help="30 RB = 360 subcarriers.")
    parser.add_argument("--nrb-ssb", type=int, default=20, help="20 RB = PSS timing grid.")
    parser.add_argument("--num-symbols", type=int, default=6, help="OFDM symbols to extract.")
    parser.add_argument("--force-nid2", type=int, default=0, choices=[0, 1, 2], help="Force NID2.")
    parser.add_argument("--min-pss-metric", type=float, default=0.50, help="Minimum valid PSS metric.")

    parser.add_argument(
        "--out-dir",
        default="results/python_online_profile",
        help="Output directory for CSV and summary.",
    )
    parser.add_argument(
        "--prefix",
        default="online_profile",
        help="Output file prefix.",
    )

    return parser.parse_args()


def configure_usrp(args: argparse.Namespace) -> uhd.usrp.MultiUSRP:
    print("=== USRP setup ===")

    usrp = uhd.usrp.MultiUSRP(f"serial={args.serial}")
    ch = args.channel

    usrp.set_rx_rate(args.rate, ch)
    usrp.set_rx_freq(uhd.types.TuneRequest(args.freq), ch)
    usrp.set_rx_gain(args.gain, ch)

    if args.antenna:
        usrp.set_rx_antenna(args.antenna, ch)

    time.sleep(args.settle_sec)

    print(f"Motherboard:      {usrp.get_mboard_name()}")
    print(f"RX channels:      {usrp.get_rx_num_channels()}")
    print(f"Requested rate:   {args.rate:.3f} S/s")
    print(f"Actual rate:      {usrp.get_rx_rate(ch):.3f} S/s")
    print(f"Requested freq:   {args.freq/1e6:.6f} MHz")
    print(f"Actual freq:      {usrp.get_rx_freq(ch)/1e6:.6f} MHz")
    print(f"Requested gain:   {args.gain:.2f} dB")
    print(f"Actual gain:      {usrp.get_rx_gain(ch):.2f} dB")

    return usrp


def make_rx_streamer(usrp: uhd.usrp.MultiUSRP, channel: int):
    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = [channel]
    rx_streamer = usrp.get_rx_stream(stream_args)
    return rx_streamer


def capture_one_block(
    rx_streamer,
    total_samples: int,
    max_samps: int,
    timeout: float = 3.0,
) -> np.ndarray:
    metadata = uhd.types.RXMetadata()
    output = np.empty(total_samples, dtype=np.complex64)
    buffer = np.empty(max_samps, dtype=np.complex64)

    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    cmd.num_samps = total_samples
    cmd.stream_now = True
    rx_streamer.issue_stream_cmd(cmd)

    received = 0

    while received < total_samples:
        n_to_recv = min(max_samps, total_samples - received)
        num_rx = rx_streamer.recv(buffer[:n_to_recv], metadata, timeout)

        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
            raise RuntimeError(f"UHD RX error: {metadata.strerror()}")

        if num_rx > 0:
            output[received : received + num_rx] = buffer[:num_rx]
            received += num_rx

    return output


def extract_rxgrid_from_waveform(
    waveform: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Returns:
        dataSSB, rxGridSSB, timing_info, timing_breakdown
    """
    t0 = time.perf_counter()

    timing_info = detect_best_pss_timing(
        waveform=waveform,
        nfft=args.nfft,
        nrb_ssb=args.nrb_ssb,
        force_nid2=args.force_nid2,
        sample_rate=args.rate,
    )

    t1 = time.perf_counter()

    timing = int(timing_info["timing_offset_samples"])
    waveform_aligned = waveform[timing:]

    rx_grid_save = ofdm_demodulate_centered(
        waveform_aligned=waveform_aligned,
        nfft=args.nfft,
        demod_rb=args.demod_rb,
        num_symbols=args.num_symbols,
    )

    data_ssb = np.zeros((args.demod_rb * 12, args.num_symbols), dtype=np.complex64)
    n_keep = min(args.num_symbols, rx_grid_save.shape[1])
    data_ssb[:, :n_keep] = rx_grid_save[:, :n_keep]

    rx_grid_ssb = data_ssb[60:300, 1:5]

    t2 = time.perf_counter()

    timing_info["n_symbols_extracted"] = int(n_keep)

    timing_breakdown = {
        "pss_time_ms": 1000.0 * (t1 - t0),
        "ofdm_time_ms": 1000.0 * (t2 - t1),
        "total_dsp_time_ms": 1000.0 * (t2 - t0),
    }

    return data_ssb, rx_grid_ssb, timing_info, timing_breakdown


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p95": None,
        }

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
    }


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"{args.prefix}_{timestamp}.csv"
    summary_path = out_dir / f"{args.prefix}_{timestamp}_summary.json"

    usrp = configure_usrp(args)
    ch = args.channel
    actual_rate = float(usrp.get_rx_rate(ch))

    samples_per_block = int(round(actual_rate * args.duration_ms * 1e-3))

    rx_streamer = make_rx_streamer(usrp, ch)
    max_samps = rx_streamer.get_max_num_samps()

    print("\n=== Live profile setup ===")
    print(f"iterations:         {args.num_iters}")
    print(f"warmup iterations:  {args.warmup_iters}")
    print(f"duration/block:     {args.duration_ms:.3f} ms")
    print(f"samples/block:      {samples_per_block}")
    print(f"max chunk:          {max_samps}")
    print(f"output CSV:         {csv_path}")

    rows = []

    fieldnames = [
        "iter",
        "used_for_summary",
        "valid",
        "capture_time_ms",
        "pss_time_ms",
        "ofdm_time_ms",
        "total_dsp_time_ms",
        "total_loop_time_ms",
        "samples_per_second",
        "nid2",
        "timing_samples",
        "timing_ms",
        "pss_metric",
        "n_symbols_extracted",
        "rxGridSSB_mean_abs",
        "rxGridSSB_max_abs",
        "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(args.num_iters):
            loop_t0 = time.perf_counter()

            row = {
                "iter": i,
                "used_for_summary": int(i >= args.warmup_iters),
                "valid": 0,
                "capture_time_ms": np.nan,
                "pss_time_ms": np.nan,
                "ofdm_time_ms": np.nan,
                "total_dsp_time_ms": np.nan,
                "total_loop_time_ms": np.nan,
                "samples_per_second": np.nan,
                "nid2": -1,
                "timing_samples": -1,
                "timing_ms": np.nan,
                "pss_metric": np.nan,
                "n_symbols_extracted": 0,
                "rxGridSSB_mean_abs": np.nan,
                "rxGridSSB_max_abs": np.nan,
                "error": "",
            }

            try:
                cap_t0 = time.perf_counter()
                waveform = capture_one_block(
                    rx_streamer=rx_streamer,
                    total_samples=samples_per_block,
                    max_samps=max_samps,
                )
                cap_t1 = time.perf_counter()

                _data_ssb, rx_grid_ssb, timing_info, tb = extract_rxgrid_from_waveform(
                    waveform=waveform,
                    args=args,
                )

                loop_t1 = time.perf_counter()

                capture_time_ms = 1000.0 * (cap_t1 - cap_t0)
                total_loop_time_ms = 1000.0 * (loop_t1 - loop_t0)
                samples_per_second = 1000.0 / total_loop_time_ms

                metric = float(timing_info["metric"])
                n_symbols = int(timing_info["n_symbols_extracted"])

                valid = bool(
                    metric >= args.min_pss_metric
                    and n_symbols == args.num_symbols
                    and rx_grid_ssb.shape == (240, 4)
                )

                row.update(
                    {
                        "valid": int(valid),
                        "capture_time_ms": capture_time_ms,
                        "pss_time_ms": tb["pss_time_ms"],
                        "ofdm_time_ms": tb["ofdm_time_ms"],
                        "total_dsp_time_ms": tb["total_dsp_time_ms"],
                        "total_loop_time_ms": total_loop_time_ms,
                        "samples_per_second": samples_per_second,
                        "nid2": int(timing_info["nid2"]),
                        "timing_samples": int(timing_info["timing_offset_samples"]),
                        "timing_ms": float(1000.0 * int(timing_info["timing_offset_samples"]) / actual_rate),
                        "pss_metric": metric,
                        "n_symbols_extracted": n_symbols,
                        "rxGridSSB_mean_abs": float(np.mean(np.abs(rx_grid_ssb))),
                        "rxGridSSB_max_abs": float(np.max(np.abs(rx_grid_ssb))),
                    }
                )

            except Exception as exc:
                loop_t1 = time.perf_counter()
                row["total_loop_time_ms"] = 1000.0 * (loop_t1 - loop_t0)
                row["samples_per_second"] = 1000.0 / row["total_loop_time_ms"]
                row["error"] = str(exc)

            rows.append(row)
            writer.writerow(row)
            f.flush()

            if args.progress_every > 0 and (i % args.progress_every == 0 or i == args.num_iters - 1):
                print(
                    f"[{i + 1:04d}/{args.num_iters:04d}] "
                    f"valid={row['valid']} "
                    f"cap={row['capture_time_ms']:.2f} ms "
                    f"pss={row['pss_time_ms']:.2f} ms "
                    f"ofdm={row['ofdm_time_ms']:.2f} ms "
                    f"dsp={row['total_dsp_time_ms']:.2f} ms "
                    f"loop={row['total_loop_time_ms']:.2f} ms "
                    f"rate={row['samples_per_second']:.2f} samples/s "
                    f"metric={row['pss_metric']:.3f} "
                    f"err={row['error']}"
                )

    used = [r for r in rows if r["used_for_summary"] == 1]
    valid_used = [r for r in used if r["valid"] == 1]

    def col(name: str, source: list[dict]) -> np.ndarray:
        return np.array([float(r[name]) for r in source], dtype=np.float64)

    summary = {
        "config": {
            "serial": args.serial,
            "freq_hz": args.freq,
            "rate_hz_requested": args.rate,
            "rate_hz_actual": actual_rate,
            "gain_db": args.gain,
            "duration_ms": args.duration_ms,
            "samples_per_block": samples_per_block,
            "num_iters": args.num_iters,
            "warmup_iters": args.warmup_iters,
            "nfft": args.nfft,
            "demod_rb": args.demod_rb,
            "nrb_ssb": args.nrb_ssb,
            "num_symbols": args.num_symbols,
            "force_nid2": args.force_nid2,
            "min_pss_metric": args.min_pss_metric,
        },
        "counts": {
            "total_iters": len(rows),
            "summary_iters": len(used),
            "valid_summary_iters": len(valid_used),
            "valid_ratio_summary": float(len(valid_used) / len(used)) if used else None,
        },
        "all_summary_iters": {
            "capture_time_ms": summarize(col("capture_time_ms", used)),
            "pss_time_ms": summarize(col("pss_time_ms", used)),
            "ofdm_time_ms": summarize(col("ofdm_time_ms", used)),
            "total_dsp_time_ms": summarize(col("total_dsp_time_ms", used)),
            "total_loop_time_ms": summarize(col("total_loop_time_ms", used)),
            "samples_per_second": summarize(col("samples_per_second", used)),
            "pss_metric": summarize(col("pss_metric", used)),
        },
        "valid_summary_iters": {
            "capture_time_ms": summarize(col("capture_time_ms", valid_used)),
            "pss_time_ms": summarize(col("pss_time_ms", valid_used)),
            "ofdm_time_ms": summarize(col("ofdm_time_ms", valid_used)),
            "total_dsp_time_ms": summarize(col("total_dsp_time_ms", valid_used)),
            "total_loop_time_ms": summarize(col("total_loop_time_ms", valid_used)),
            "samples_per_second": summarize(col("samples_per_second", valid_used)),
            "pss_metric": summarize(col("pss_metric", valid_used)),
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Profile summary ===")
    print(f"summary iters:       {len(used)}")
    print(f"valid summary iters: {len(valid_used)}/{len(used)}")

    valid_stats = summary["valid_summary_iters"]

    print("\nValid iterations timing:")
    print(f"capture mean:        {valid_stats['capture_time_ms']['mean']:.3f} ms")
    print(f"PSS mean:            {valid_stats['pss_time_ms']['mean']:.3f} ms")
    print(f"OFDM mean:           {valid_stats['ofdm_time_ms']['mean']:.3f} ms")
    print(f"DSP mean:            {valid_stats['total_dsp_time_ms']['mean']:.3f} ms")
    print(f"loop mean:           {valid_stats['total_loop_time_ms']['mean']:.3f} ms")
    print(f"sample rate mean:    {valid_stats['samples_per_second']['mean']:.3f} rxGridSSB/s")

    print(f"\nCSV:                 {csv_path}")
    print(f"summary:             {summary_path}")


if __name__ == "__main__":
    main()
