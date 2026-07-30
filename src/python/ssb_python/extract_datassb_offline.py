#!/usr/bin/env python3
"""
Offline dataSSB extractor from raw Python UHD IQ captures.

Input:
    .h5 from capture_iq_blocks_uhd.py
    or .npz from test_capture_iq_uhd.py

Current processing:
    1. Load raw IQ waveform.
    2. Detect PSS/NID2/timing using detect_pss_offline helpers.
    3. Align waveform at timing offset.
    4. OFDM demodulate 30 RB with NFFT=512.
    5. Save:
        dataSSB    = 360 x 6 complex64
        rxGridSSB  = 240 x 4 complex64

MATLAB equivalent target:
    rxGridSave = nrOFDMDemodulate(...)
    dataSSB(:, 1:nKeep) = single(rxGridSave(:, 1:nKeep, 1))
    rxGridSSB = dataSSB(61:300, 2:5)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Make imports work when this script is executed directly.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from detect_pss_offline import (  # noqa: E402
    build_pss_reference_grid,
    correlate_reference,
    load_waveform,
    ofdm_modulate_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract dataSSB/rxGridSSB from a raw IQ capture."
    )

    parser.add_argument("--input", required=True, help="Input .h5 or .npz IQ capture.")
    parser.add_argument("--block", type=int, default=0, help="Block index for HDF5 input.")
    parser.add_argument("--sample-rate", type=float, default=15.36e6, help="Sample rate in Hz.")
    parser.add_argument("--nfft", type=int, default=512, help="OFDM FFT size.")
    parser.add_argument(
        "--demod-rb",
        type=int,
        default=30,
        help="Demodulation bandwidth in RB. 30 RB = 360 subcarriers.",
    )
    parser.add_argument(
        "--nrb-ssb",
        type=int,
        default=20,
        help="PSS timing reference bandwidth in RB. 20 RB = 240 subcarriers.",
    )
    parser.add_argument(
        "--num-symbols",
        type=int,
        default=6,
        help="Number of OFDM symbols to save into dataSSB.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/python_datassb_extract",
        help="Output directory.",
    )
    parser.add_argument(
        "--force-nid2",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="Force a specific NID2. If omitted, the best PSS correlation is used.",
    )
    parser.add_argument(
        "--timing-offset",
        type=int,
        default=None,
        help="Manual timing offset in samples. If omitted, PSS detection is used.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate diagnostic plots.",
    )

    return parser.parse_args()


def cp_lengths_for_30khz(num_symbols: int) -> list[int]:
    """
    Approximate normal CP lengths for Fs=15.36 Msps, SCS=30 kHz, NFFT=512.

    For the current prototype:
        symbol 0 CP = 40 samples
        other symbols CP = 36 samples

    This matches the simple reference used in detect_pss_offline.py.
    """
    return [40 if i == 0 else 36 for i in range(num_symbols)]


def detect_best_pss_timing(
    waveform: np.ndarray,
    nfft: int,
    nrb_ssb: int,
    force_nid2: int | None,
    sample_rate: float,
) -> dict:
    candidates: list[dict] = []

    nid2_values = [force_nid2] if force_nid2 is not None else [0, 1, 2]

    for nid2 in nid2_values:
        grid = build_pss_reference_grid(nid2=int(nid2), nrb_ssb=nrb_ssb)
        reference = ofdm_modulate_grid(grid, nfft=nfft)
        metric = correlate_reference(waveform, reference)

        idx = int(np.argmax(metric))
        val = float(metric[idx])

        candidates.append(
            {
                "nid2": int(nid2),
                "timing_offset_samples": idx,
                "timing_offset_ms": float(1000.0 * idx / sample_rate),
                "metric": val,
                "reference_len_samples": int(len(reference)),
            }
        )

    return max(candidates, key=lambda item: item["metric"])


def ofdm_demodulate_centered(
    waveform_aligned: np.ndarray,
    nfft: int,
    demod_rb: int,
    num_symbols: int,
) -> np.ndarray:
    """
    Simple centered OFDM demodulator.

    Input:
        waveform_aligned: time-domain IQ aligned to the start of the timing reference.

    Output:
        grid with shape [demod_rb*12, n_symbols_available]
    """
    n_sc = demod_rb * 12
    cp_lengths = cp_lengths_for_30khz(num_symbols)

    start_sc = nfft // 2 - n_sc // 2
    end_sc = start_sc + n_sc

    pos = 0
    symbols: list[np.ndarray] = []

    for sym_idx in range(num_symbols):
        cp = cp_lengths[sym_idx]

        sym_start = pos + cp
        sym_end = sym_start + nfft

        if sym_end > len(waveform_aligned):
            break

        time_symbol = waveform_aligned[sym_start:sym_end]

        # Time -> frequency.
        freq_symbol = np.fft.fftshift(np.fft.fft(time_symbol, n=nfft))

        # Keep centered demod_rb*12 subcarriers.
        sc = freq_symbol[start_sc:end_sc]
        symbols.append(sc.astype(np.complex64))

        pos = sym_end

    if not symbols:
        raise RuntimeError("No complete OFDM symbols available after timing alignment.")

    return np.stack(symbols, axis=1).astype(np.complex64)


def save_outputs(
    out_dir: Path,
    input_path: Path,
    block: int,
    timing_info: dict,
    data_ssb: np.ndarray,
    rx_grid_ssb: np.ndarray,
    args: argparse.Namespace,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{input_path.stem}_block{block}_datassb.h5"

    summary = {
        "input": str(input_path),
        "block": int(block),
        "sample_rate": float(args.sample_rate),
        "nfft": int(args.nfft),
        "demod_rb": int(args.demod_rb),
        "nrb_ssb": int(args.nrb_ssb),
        "num_symbols_requested": int(args.num_symbols),
        "timing_info": timing_info,
        "dataSSB_shape": list(data_ssb.shape),
        "rxGridSSB_shape": list(rx_grid_ssb.shape),
        "dataSSB_mean_abs": float(np.mean(np.abs(data_ssb))),
        "dataSSB_max_abs": float(np.max(np.abs(data_ssb))),
        "rxGridSSB_mean_abs": float(np.mean(np.abs(rx_grid_ssb))),
        "rxGridSSB_max_abs": float(np.max(np.abs(rx_grid_ssb))),
    }

    with h5py.File(out_path, "w") as f:
        f.create_dataset("dataSSB", data=data_ssb.astype(np.complex64))
        f.create_dataset("rxGridSSB", data=rx_grid_ssb.astype(np.complex64))
        f.attrs["summary_json"] = json.dumps(summary, indent=2)

    summary_path = out_dir / f"{input_path.stem}_block{block}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return out_path


def make_plots(
    out_dir: Path,
    input_path: Path,
    block: int,
    data_ssb: np.ndarray,
    rx_grid_ssb: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{input_path.stem}_block{block}"

    data_db = 20 * np.log10(np.abs(data_ssb) + 1e-9)
    rx_db = 20 * np.log10(np.abs(rx_grid_ssb) + 1e-9)

    plt.figure(figsize=(9, 4))
    plt.imshow(data_db.T, aspect="auto", origin="lower")
    plt.colorbar(label="Magnitude [dB]")
    plt.xlabel("Subcarrier index, 360")
    plt.ylabel("OFDM symbol index")
    plt.title("dataSSB magnitude, 360 x 6")
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_dataSSB_heatmap.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.imshow(rx_db.T, aspect="auto", origin="lower")
    plt.colorbar(label="Magnitude [dB]")
    plt.xlabel("Subcarrier index, 240")
    plt.ylabel("OFDM symbol index")
    plt.title("rxGridSSB magnitude, 240 x 4")
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_rxGridSSB_heatmap.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(20 * np.log10(np.mean(np.abs(data_ssb), axis=1) + 1e-9))
    plt.xlabel("Subcarrier index, 360")
    plt.ylabel("Mean magnitude [dB]")
    plt.title("Mean dataSSB magnitude by subcarrier")
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_dataSSB_mean_by_subcarrier.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(20 * np.log10(np.mean(np.abs(rx_grid_ssb), axis=1) + 1e-9))
    plt.xlabel("Subcarrier index, 240")
    plt.ylabel("Mean magnitude [dB]")
    plt.title("Mean rxGridSSB magnitude by subcarrier")
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_rxGridSSB_mean_by_subcarrier.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    waveform, _cfg = load_waveform(input_path, args.block)

    if waveform.ndim != 1:
        waveform = np.asarray(waveform).reshape(-1)

    if args.timing_offset is None:
        timing_info = detect_best_pss_timing(
            waveform=waveform,
            nfft=args.nfft,
            nrb_ssb=args.nrb_ssb,
            force_nid2=args.force_nid2,
            sample_rate=args.sample_rate,
        )
    else:
        timing_info = {
            "nid2": int(args.force_nid2) if args.force_nid2 is not None else -1,
            "timing_offset_samples": int(args.timing_offset),
            "timing_offset_ms": float(1000.0 * args.timing_offset / args.sample_rate),
            "metric": None,
            "manual": True,
        }

    timing = int(timing_info["timing_offset_samples"])

    if timing < 0 or timing >= len(waveform):
        raise RuntimeError(f"Invalid timing offset: {timing}")

    waveform_aligned = waveform[timing:]

    rx_grid_save = ofdm_demodulate_centered(
        waveform_aligned=waveform_aligned,
        nfft=args.nfft,
        demod_rb=args.demod_rb,
        num_symbols=args.num_symbols,
    )

    # Ensure MATLAB-like dataSSB shape: 360 x 6.
    data_ssb = np.zeros((args.demod_rb * 12, args.num_symbols), dtype=np.complex64)
    n_keep = min(args.num_symbols, rx_grid_save.shape[1])
    data_ssb[:, :n_keep] = rx_grid_save[:, :n_keep]

    # MATLAB:
    #   rxGridSSB = dataSSB(61:300, 2:5)
    #
    # Python 0-based:
    #   rows 60:300, cols 1:5
    rx_grid_ssb = data_ssb[60:300, 1:5]

    out_dir = Path(args.out_dir)

    out_path = save_outputs(
        out_dir=out_dir,
        input_path=input_path,
        block=args.block,
        timing_info=timing_info,
        data_ssb=data_ssb,
        rx_grid_ssb=rx_grid_ssb,
        args=args,
    )

    if args.plot:
        make_plots(out_dir, input_path, args.block, data_ssb, rx_grid_ssb)

    print("=== dataSSB extraction ===")
    print(f"input:              {input_path}")
    print(f"block:              {args.block}")
    print(f"NID2:               {timing_info.get('nid2')}")
    print(f"timing samples:     {timing}")
    print(f"timing ms:          {1000.0 * timing / args.sample_rate:.6f}")
    print(f"PSS metric:         {timing_info.get('metric')}")
    print(f"dataSSB shape:      {data_ssb.shape}")
    print(f"rxGridSSB shape:    {rx_grid_ssb.shape}")
    print(f"dataSSB mean abs:   {np.mean(np.abs(data_ssb)):.6f}")
    print(f"dataSSB max abs:    {np.max(np.abs(data_ssb)):.6f}")
    print(f"rxGridSSB mean abs: {np.mean(np.abs(rx_grid_ssb)):.6f}")
    print(f"rxGridSSB max abs:  {np.max(np.abs(rx_grid_ssb)):.6f}")
    print(f"saved:              {out_path}")


if __name__ == "__main__":
    main()
