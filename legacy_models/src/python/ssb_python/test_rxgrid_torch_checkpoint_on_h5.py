#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from rxgrid_torch_inference import RxGridTorchBinaryModel
from ssb_fisher_torch_inference import SSBFisherTorchModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-pt", default="results/binary_empty_vs_P5_rx/model_rxGridSSB/model.pt")
    p.add_argument("--input-h5", required=True)
    p.add_argument("--max-samples", type=int, default=30)
    return p.parse_args()


def load_rxgrid_h5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        x = f["rxGridSSB"][:].astype(np.complex64)

        if "valid_mask" in f:
            mask = f["valid_mask"][:].astype(bool)
            if x.ndim == 3 and len(mask) == x.shape[2]:
                x = x[:, :, mask]

    if x.shape[0] != 240 or x.shape[1] != 4:
        raise RuntimeError(f"Expected rxGridSSB [240,4,N], got {x.shape}")

    return x


def main():
    args = parse_args()

    checkpoint = torch.load(args.model_pt, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") == "torch_fisher_logreg":
        model = SSBFisherTorchModel(args.model_pt, device="cpu")
    else:
        model = RxGridTorchBinaryModel(args.model_pt, device="cpu")
    rx = load_rxgrid_h5(Path(args.input_h5))

    if args.max_samples > 0:
        rx = rx[:, :, : args.max_samples]

    print("=== Torch checkpoint rxGridSSB test ===")
    print("model:", args.model_pt)
    print("input:", args.input_h5)
    print("rxGridSSB:", rx.shape)
    print("classes:", model.classes)

    labels = []
    prediction_rows = []

    for i in range(rx.shape[2]):
        pred = model.predict_proba(rx[:, :, i])
        labels.append(pred["label"])
        prediction_rows.append(pred["probabilities"])

        print(
            f"[{i:04d}] "
            f"label={pred['label']} "
            f"conf={pred['confidence']:.4f} "
            f"probabilities={pred['probabilities']}"
        )

    unique, counts = np.unique(labels, return_counts=True)
    print("\nCounts:")
    for u, c in zip(unique, counts):
        print(f"  {u}: {c}")

    print("\nMean probability by class:")
    probabilities = np.asarray(
        [[row[name] for name in model.classes] for row in prediction_rows],
        dtype=float,
    )
    for index, name in enumerate(model.classes):
        print(f"  {name}: {float(np.mean(probabilities[:, index])):.4f}")


if __name__ == "__main__":
    main()
