#!/usr/bin/env python3
"""
Batch offline dataSSB/rxGridSSB extractor from raw Python UHD IQ captures.

Input:
    .h5 from capture_iq_blocks_uhd.py
    or .npz from test_capture_iq_uhd.py

Output:
    One batch .h5 file containing:
        dataSSB        = 360 x 6 x N complex64
        rxGridSSB      = 240 x 4 x N complex64
        timing_offsets = N int64
        pss_metrics    = N float32
        nid2           = N int16
        valid_mask     = N bool

This script wraps the single-block prototype:
    extract_datassb_offline.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from detect_pss_offline import load_waveform  # noqa: E402
from extract_datassb_offline import (  # noqa: E402
    detect_best_pss_timing,
    ofdm_demodulate_centered,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch extract dataSSB/rxGridSSB from raw IQ capture."
    )

    parser.add_argument("--input", required=True, help="Input .h5 or .npz IQ capture.")
    parser.add_argument(
        "--out-dir",
        default="results/python_datassb_batch",
        help="Output directory.",
    )
    parser.add_argument("--sample-rate", type=float, default=15.36e6, help="Sample rate in Hz.")
    parser.add_argument("--nfft", type=int, default=512, help="OFDM FFT size.")
    parser.add_argument("--demod-rb", type=int, default=30, help="30 RB = 360 subcarriers.")
    parser.add_argument("--nrb-ssb", type=int, default=20, help="20 RB = 240 subcarriers.")
    parser.add_argument("--num-symbols", type=int, default=6, help="Number of OFDM symbols.")
    parser.add_argument("--force-nid2", type=int, default=None, choices=[0, 1, 2])
    parser.add_argument("--min-pss-metric", type=float, default=0.50)
    parser.add_argument("--block-start", type=int, default=0)
    parser.add_argument("--block-end", type=int, default=None)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--plot", action="store_true")

    return parser.parse_args()


def count_blocks(input_path: Path) -> int:
    if input_path.suffix.lower() == ".npz":
        return 1

    if input_path.suffix.lower() in [".h5", ".hdf5"]:
        with h5py.File(input_path, "r") as f:
            ds = f["waveform"]
            if ds.ndim == 1:
                return 1
            return int(ds.shape[0])

    raise ValueError(f"Unsupported input extension: {input_path.suffix}")


def extract_one_block(
    input_path: Path,
    block: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict]:
    waveform, _cfg = load_waveform(input_path, block)

    if waveform.ndim != 1:
        waveform = np.asarray(waveform).reshape(-1)

    timing_info = detect_best_pss_timing(
        waveform=waveform,
        nfft=args.nfft,
        nrb_ssb=args.nrb_ssb,
        force_nid2=args.force_nid2,
        sample_rate=args.sample_rate,
    )

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

    timing_info["timing_offset_ms"] = float(1000.0 * timing / args.sample_rate)
    timing_info["n_symbols_extracted"] = int(n_keep)
    timing_info["dataSSB_mean_abs"] = float(np.mean(np.abs(data_ssb)))
    timing_info["dataSSB_max_abs"] = float(np.max(np.abs(data_ssb)))
    timing_info["rxGridSSB_mean_abs"] = float(np.mean(np.abs(rx_grid_ssb)))
    timing_info["rxGridSSB_max_abs"] = float(np.max(np.abs(rx_grid_ssb)))

    return data_ssb, rx_grid_ssb, timing_info


def make_plots(
    out_dir: Path,
    stem: str,
    data_all: np.ndarray,
    rx_all: np.ndarray,
    timing_offsets: np.ndarray,
    pss_metrics: np.ndarray,
    valid_mask: np.ndarray,
    sample_rate: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    block_idx = np.arange(len(pss_metrics))

    plt.figure(figsize=(10, 4))
    plt.plot(block_idx, pss_metrics, marker="o")
    plt.xlabel("Block index")
    plt.ylabel("Best PSS metric")
    plt.title("PSS metric per block")
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_pss_metric_per_block.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(block_idx, 1000.0 * timing_offsets / sample_rate, marker="o")
    plt.xlabel("Block index")
    plt.ylabel("Timing offset [ms]")
    plt.title("Detected timing offset per block")
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_timing_ms_per_block.png", dpi=160)
    plt.close()

    rx_mean = np.array(
        [
            float(np.mean(np.abs(rx_all[:, :, i]))) if valid_mask[i] else np.nan
            for i in range(rx_all.shape[2])
        ],
        dtype=np.float32,
    )

    plt.figure(figsize=(10, 4))
    plt.plot(block_idx, rx_mean, marker="o")
    plt.xlabel("Block index")
    plt.ylabel("Mean |rxGridSSB|")
    plt.title("Mean rxGridSSB amplitude per block")
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_rxGridSSB_mean_abs_per_block.png", dpi=160)
    plt.close()

    if np.any(valid_mask):
        rx_avg = np.mean(np.abs(rx_all[:, :, valid_mask]), axis=2)
        data_avg = np.mean(np.abs(data_all[:, :, valid_mask]), axis=2)

        plt.figure(figsize=(9, 4))
        plt.imshow(20.0 * np.log10(data_avg + 1e-9).T, aspect="auto", origin="lower")
        plt.colorbar(label="Magnitude [dB]")
        plt.xlabel("Subcarrier index, 360")
        plt.ylabel("OFDM symbol index")
        plt.title("Average dataSSB magnitude over valid blocks")
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_average_dataSSB_heatmap.png", dpi=160)
        plt.close()

        plt.figure(figsize=(9, 4))
        plt.imshow(20.0 * np.log10(rx_avg + 1e-9).T, aspect="auto", origin="lower")
        plt.colorbar(label="Magnitude [dB]")
        plt.xlabel("Subcarrier index, 240")
        plt.ylabel("OFDM symbol index")
        plt.title("Average rxGridSSB magnitude over valid blocks")
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_average_rxGridSSB_heatmap.png", dpi=160)
        plt.close()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_blocks = count_blocks(input_path)

    block_end = args.block_end
    if block_end is None:
        block_end = total_blocks

    block_indices = list(range(args.block_start, min(block_end, total_blocks)))

    if args.max_blocks is not None:
        block_indices = block_indices[: args.max_blocks]

    if not block_indices:
        raise RuntimeError("No blocks selected.")

    n_blocks = len(block_indices)

    data_shape = (args.demod_rb * 12, args.num_symbols, n_blocks)
    rx_shape = (240, 4, n_blocks)

    data_all = np.zeros(data_shape, dtype=np.complex64)
    rx_all = np.zeros(rx_shape, dtype=np.complex64)

    timing_offsets = np.full(n_blocks, -1, dtype=np.int64)
    pss_metrics = np.full(n_blocks, np.nan, dtype=np.float32)
    nid2_values = np.full(n_blocks, -1, dtype=np.int16)
    valid_mask = np.zeros(n_blocks, dtype=bool)

    rows = []
    errors = []

    print("=== Batch dataSSB extraction ===")
    print(f"input:        {input_path}")
    print(f"total blocks: {total_blocks}")
    print(f"selected:     {n_blocks}")
    print(f"out dir:      {out_dir}")

    for out_idx, block in enumerate(block_indices):
        try:
            data_ssb, rx_grid_ssb, info = extract_one_block(input_path, block, args)

            metric = float(info["metric"])
            nid2 = int(info["nid2"])
            timing = int(info["timing_offset_samples"])
            n_symbols_extracted = int(info["n_symbols_extracted"])

            is_valid = bool(
                np.isfinite(metric)
                and metric >= args.min_pss_metric
                and n_symbols_extracted == args.num_symbols
                and data_ssb.shape == (args.demod_rb * 12, args.num_symbols)
                and rx_grid_ssb.shape == (240, 4)
            )

            data_all[:, :, out_idx] = data_ssb
            rx_all[:, :, out_idx] = rx_grid_ssb

            timing_offsets[out_idx] = timing
            pss_metrics[out_idx] = metric
            nid2_values[out_idx] = nid2
            valid_mask[out_idx] = is_valid

            row = {
                "out_index": out_idx,
                "source_block": block,
                "valid": int(is_valid),
                "nid2": nid2,
                "timing_samples": timing,
                "timing_ms": float(1000.0 * timing / args.sample_rate),
                "pss_metric": metric,
                "n_symbols_extracted": n_symbols_extracted,
                "dataSSB_mean_abs": float(info["dataSSB_mean_abs"]),
                "dataSSB_max_abs": float(info["dataSSB_max_abs"]),
                "rxGridSSB_mean_abs": float(info["rxGridSSB_mean_abs"]),
                "rxGridSSB_max_abs": float(info["rxGridSSB_max_abs"]),
                "error": "",
            }

            rows.append(row)

            print(
                f"[{out_idx + 1:04d}/{n_blocks:04d}] "
                f"block={block:04d} valid={int(is_valid)} "
                f"nid2={nid2} timing={timing:7d} "
                f"metric={metric:.6f} "
                f"rx_mean={row['rxGridSSB_mean_abs']:.3f}"
            )

        except Exception as exc:
            err = str(exc)
            errors.append({"out_index": out_idx, "source_block": block, "error": err})

            rows.append(
                {
                    "out_index": out_idx,
                    "source_block": block,
                    "valid": 0,
                    "nid2": -1,
                    "timing_samples": -1,
                    "timing_ms": np.nan,
                    "pss_metric": np.nan,
                    "n_symbols_extracted": 0,
                    "dataSSB_mean_abs": np.nan,
                    "dataSSB_max_abs": np.nan,
                    "rxGridSSB_mean_abs": np.nan,
                    "rxGridSSB_max_abs": np.nan,
                    "error": err,
                }
            )

            print(
                f"[{out_idx + 1:04d}/{n_blocks:04d}] "
                f"block={block:04d} ERROR: {err}"
            )

    stem = input_path.stem
    out_h5 = out_dir / f"{stem}_datassb_batch.h5"
    out_csv = out_dir / f"{stem}_metadata.csv"
    out_summary = out_dir / f"{stem}_summary.json"

    summary = {
        "input": str(input_path),
        "total_blocks_in_input": int(total_blocks),
        "selected_source_blocks": block_indices,
        "n_selected_blocks": int(n_blocks),
        "n_valid": int(np.sum(valid_mask)),
        "valid_ratio": float(np.mean(valid_mask)),
        "sample_rate": float(args.sample_rate),
        "nfft": int(args.nfft),
        "demod_rb": int(args.demod_rb),
        "nrb_ssb": int(args.nrb_ssb),
        "num_symbols": int(args.num_symbols),
        "min_pss_metric": float(args.min_pss_metric),
        "dataSSB_shape": list(data_all.shape),
        "rxGridSSB_shape": list(rx_all.shape),
        "pss_metric_mean_valid": float(np.nanmean(pss_metrics[valid_mask])) if np.any(valid_mask) else None,
        "pss_metric_min_valid": float(np.nanmin(pss_metrics[valid_mask])) if np.any(valid_mask) else None,
        "pss_metric_max_valid": float(np.nanmax(pss_metrics[valid_mask])) if np.any(valid_mask) else None,
        "errors": errors,
    }

    with h5py.File(out_h5, "w") as f:
        f.create_dataset("dataSSB", data=data_all)
        f.create_dataset("rxGridSSB", data=rx_all)
        f.create_dataset("timing_offsets", data=timing_offsets)
        f.create_dataset("pss_metrics", data=pss_metrics)
        f.create_dataset("nid2", data=nid2_values)
        f.create_dataset("valid_mask", data=valid_mask)
        f.create_dataset("source_blocks", data=np.array(block_indices, dtype=np.int64))
        f.attrs["summary_json"] = json.dumps(summary, indent=2)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.plot:
        make_plots(
            out_dir=out_dir,
            stem=stem,
            data_all=data_all,
            rx_all=rx_all,
            timing_offsets=timing_offsets,
            pss_metrics=pss_metrics,
            valid_mask=valid_mask,
            sample_rate=args.sample_rate,
        )

    print("\n=== Batch summary ===")
    print(f"dataSSB shape:       {data_all.shape}")
    print(f"rxGridSSB shape:     {rx_all.shape}")
    print(f"valid blocks:        {int(np.sum(valid_mask))}/{n_blocks}")
    print(f"valid ratio:         {float(np.mean(valid_mask)):.3f}")
    print(f"saved h5:            {out_h5}")
    print(f"saved csv:           {out_csv}")
    print(f"saved summary:       {out_summary}")


if __name__ == "__main__":
    main()
