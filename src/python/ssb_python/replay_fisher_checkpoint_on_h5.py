#!/usr/bin/env python3
"""Replay a Fisher SSB checkpoint on a saved ``session_data.h5``.

Runs the exact online inference path (radio check + feature extraction +
PyTorch model) on recorded ``rxGridSSB`` samples and prints a summary.

Example:
    python src/python/ssb_python/replay_fisher_checkpoint_on_h5.py \
        --model-pt results/ssb_sensing_analysis/model_fisher_logreg.pt \
        --input-h5 data/python_ssb_datasets/P1/session_xxx/session_data.h5 \
        --max-samples 30
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from ssb_fisher_torch_inference import SSBFisherTorchModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-pt", required=True, help="model_fisher_logreg.pt from run_ssb_sensing_analysis.py")
    p.add_argument("--input-h5", required=True, help="session_data.h5 containing rxGridSSB [240,4,N]")
    p.add_argument("--max-samples", type=int, default=30, help="0 replays every sample")
    p.add_argument("--quiet", action="store_true", help="Only print the summary")
    return p.parse_args()


def load_session(path: Path) -> tuple[np.ndarray, int | None]:
    with h5py.File(path, "r") as f:
        if "rxGridSSB" not in f:
            raise KeyError(f"rxGridSSB not found in {path}")
        rx = f["rxGridSSB"][()].astype(np.complex64)
        nid2 = int(f.attrs["force_nid2"]) if "force_nid2" in f.attrs else None
    if rx.ndim != 3 or rx.shape[:2] != (240, 4):
        raise ValueError(f"Expected rxGridSSB [240,4,N], got {list(rx.shape)}")
    return rx, nid2


def main() -> None:
    args = parse_args()
    model = SSBFisherTorchModel(args.model_pt)
    rx, nid2 = load_session(Path(args.input_h5))
    n = rx.shape[2] if args.max_samples <= 0 else min(args.max_samples, rx.shape[2])

    print(f"checkpoint: {args.model_pt}")
    print(f"classes:    {model.classes}")
    print(f"radio:      PCI={model.pci} i_bar={model.ibar}  (H5 NID2={nid2})")
    print(f"samples:    {n} of {rx.shape[2]}")

    labels: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    probabilities = []
    for i in range(n):
        grid = rx[:, :, i]
        check = model.check_radio(grid, nid2=nid2)
        if not check.ok:
            rejections[check.reason.split(":")[0]] += 1
            if not args.quiet:
                print(f"[{i:04d}] INVALID {check.reason}")
            continue
        pred = model.predict_proba(grid)
        labels[pred["label"]] += 1
        probabilities.append([pred["probabilities"][c] for c in model.classes])
        if not args.quiet:
            probs = " ".join(f"{c}={pred['probabilities'][c]:.3f}" for c in model.classes)
            print(f"[{i:04d}] label={pred['label']} conf={pred['confidence']:.3f} {probs}")

    print("\nPredicted labels:")
    for label, count in labels.most_common():
        print(f"  {label}: {count}")
    if rejections:
        print("Rejected by radio check:")
        for reason, count in rejections.most_common():
            print(f"  {reason}: {count}")
    if probabilities:
        mean = np.mean(np.asarray(probabilities), axis=0)
        print("Mean probability by class:")
        for name, value in zip(model.classes, mean):
            print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
