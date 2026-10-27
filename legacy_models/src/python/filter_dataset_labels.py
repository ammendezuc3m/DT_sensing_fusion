#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def filter_dataset(input_npz, input_meta, output_npz, output_meta, keep_labels):
    input_npz = Path(input_npz)
    input_meta = Path(input_meta)
    output_npz = Path(output_npz)
    output_meta = Path(output_meta)

    data = np.load(input_npz, allow_pickle=True)
    meta = pd.read_csv(input_meta)

    X = data["X"]
    y = data["y"].astype(str)
    groups = data["groups"].astype(str)
    folds = data["folds"].astype(int)

    keep_labels = list(keep_labels)

    mask = np.isin(y, keep_labels)

    X_f = X[mask]
    y_f = y[mask]
    groups_f = groups[mask]
    folds_f = folds[mask]
    meta_f = meta[mask].copy()

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_meta.parent.mkdir(parents=True, exist_ok=True)

    # Guardamos campos principales.
    out_dict = {
        "X": X_f.astype(np.float32),
        "y": y_f,
        "groups": groups_f,
        "folds": folds_f,
    }

    # Conservamos campos extra si existen: window_size, stride, feature_type...
    for key in data.files:
        if key not in out_dict:
            out_dict[key] = data[key]

    np.savez_compressed(output_npz, **out_dict)
    meta_f.to_csv(output_meta, index=False)

    print("\n=== Dataset filtrado ===")
    print(f"Input:  {input_npz}")
    print(f"Output: {output_npz}")
    print(f"Labels conservadas: {keep_labels}")
    print(f"X original: {X.shape}")
    print(f"X filtrado: {X_f.shape}")
    print("\nConteo por label:")
    print(pd.Series(y_f).value_counts())
    print("\nConteo por fold:")
    print(pd.Series(folds_f).value_counts().sort_index())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--input-meta", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-meta", required=True)
    parser.add_argument("--labels", nargs="+", required=True)

    args = parser.parse_args()

    filter_dataset(
        input_npz=args.input_npz,
        input_meta=args.input_meta,
        output_npz=args.output_npz,
        output_meta=args.output_meta,
        keep_labels=args.labels,
    )


if __name__ == "__main__":
    main()
