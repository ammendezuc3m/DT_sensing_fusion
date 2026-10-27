#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


EPS = 1e-9


def read_mat_variable(path: Path, name: str):
    try:
        mat = scipy.io.loadmat(path)
        if name in mat:
            return mat[name]
    except NotImplementedError:
        pass
    except Exception:
        pass

    with h5py.File(path, "r") as f:
        if name not in f:
            raise KeyError(f"{name} not found in {path}")
        x = f[name][()]

    if hasattr(x, "dtype") and x.dtype.fields is not None:
        if "real" in x.dtype.fields and "imag" in x.dtype.fields:
            x = x["real"] + 1j * x["imag"]

    return np.asarray(x)


def normalize_datassb_shape(data):
    data = np.asarray(data)

    if data.ndim != 3:
        raise RuntimeError(f"Expected 3D dataSSB, got shape {data.shape}")

    shape = data.shape

    if data.shape[0] == 360 and data.shape[1] == 6:
        return data

    if 360 in shape and 6 in shape:
        ax_sc = shape.index(360)
        ax_sym = shape.index(6)
        ax_n = [i for i in range(3) if i not in (ax_sc, ax_sym)][0]
        return np.transpose(data, (ax_sc, ax_sym, ax_n))

    raise RuntimeError(f"Cannot infer dataSSB orientation from shape {data.shape}")


def valid_mask(path: Path, n: int):
    try:
        v = read_mat_variable(path, "validMask").reshape(-1).astype(bool)
        return v[:n]
    except Exception:
        return np.ones(n, dtype=bool)


def block_index(path: Path):
    for part in reversed(path.parts):
        m = re.match(r"block[_-]?(\d+)", part, re.IGNORECASE)
        if m:
            return int(m.group(1))

    m = re.search(r"block[_-]?(\d+)", path.name, re.IGNORECASE)
    return int(m.group(1)) if m else -1


def split_from_block(b: int):
    if b == 5:
        return "test"
    if b == 4:
        return "val"
    return "train"


def region_mean(power_sc: np.ndarray, a: int, b: int):
    a0 = max(0, a - 1)
    b0 = min(240, b)
    return power_sc[:, a0:b0].mean(axis=1)


def robust_profile(power_sc: np.ndarray):
    med = np.median(power_sc, axis=1, keepdims=True)
    q25 = np.percentile(power_sc, 25, axis=1, keepdims=True)
    q75 = np.percentile(power_sc, 75, axis=1, keepdims=True)
    iqr = np.maximum(q75 - q25, 1e-3)
    return (power_sc - med) / iqr


def build_p5_signature_features(power_sc: np.ndarray, template: np.ndarray):
    low = region_mean(power_sc, 1, 45)
    valley = region_mean(power_sc, 50, 140)
    plateau = region_mean(power_sc, 145, 180)
    high = region_mean(power_sc, 185, 240)
    tail = region_mean(power_sc, 220, 240)

    ref = 0.5 * (low + plateau)

    valley_depth = ref - valley
    high_minus_valley = high - valley
    tail_minus_valley = tail - valley
    high_minus_low = high - low
    plateau_minus_valley = plateau - valley

    prof = robust_profile(power_sc)

    t = template.astype(np.float32)
    t = t - t.mean()
    t = t / (np.linalg.norm(t) + EPS)
    template_score = prof @ t

    F = np.stack([
        valley,
        high,
        tail,
        valley_depth,
        high_minus_valley,
        tail_minus_valley,
        high_minus_low,
        plateau_minus_valley,
        template_score,
    ], axis=1).astype(np.float32)

    names = [
        "valley_mean_50_140",
        "high_mean_185_240",
        "tail_mean_220_240",
        "valley_depth_ref_minus_valley",
        "high_minus_valley",
        "tail_minus_valley",
        "high_minus_low",
        "plateau_minus_valley",
        "p5_template_score",
    ]

    return F, names


def load_power_dataset(dataset: Path, labels: list[str]):
    label_to_id = {label: i for i, label in enumerate(labels)}

    power_parts = []
    y_parts = []
    rows = []

    for label in labels:
        files = sorted((dataset / label).rglob("*.mat"))
        if not files:
            raise FileNotFoundError(f"No files for {label}")

        print(f"\nLoading {label}: {len(files)} files")

        for p in files:
            b = block_index(p)
            split = split_from_block(b)

            data = normalize_datassb_shape(read_mat_variable(p, "dataSSB"))
            n = data.shape[2]
            v = valid_mask(p, n)
            idx = np.where(v)[0]

            rx = data[60:300, 1:5, idx]
            rx = np.transpose(rx, (2, 0, 1))  # [N, 240, 4]

            power_sc = 10.0 * np.log10(np.mean(np.abs(rx) ** 2, axis=2) + EPS).astype(np.float32)
            y = np.full(len(rx), label_to_id[label], dtype=np.int64)

            power_parts.append(power_sc)
            y_parts.append(y)

            for capture_idx in idx:
                rows.append({
                    "label": label,
                    "label_id": label_to_id[label],
                    "file": str(p),
                    "block": b,
                    "split": split,
                    "capture_idx": int(capture_idx + 1),
                })

            print(f"  {p} | valid={len(rx)} | block={b} | split={split}")

    return np.concatenate(power_parts), np.concatenate(y_parts), pd.DataFrame(rows), label_to_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/dataset_datassb/datassb_side_v1_6labels")
    parser.add_argument("--labels", nargs="+", default=["empty", "P5", "P3"])
    parser.add_argument("--out", default="results/multiclass_empty_P5_P3_rx/model_rxGridSSB/p5_signature_online_calibration.npz")
    parser.add_argument("--threshold", type=float, default=0.0100)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    power_sc, y, meta, label_to_id = load_power_dataset(dataset, args.labels)

    p5_id = label_to_id["P5"]

    train = meta["split"].to_numpy() == "train"
    val = meta["split"].to_numpy() == "val"
    test = meta["split"].to_numpy() == "test"

    prof_train = robust_profile(power_sc[train])
    med_p5 = np.median(prof_train[y[train] == p5_id], axis=0)
    med_nonp5 = np.median(prof_train[y[train] != p5_id], axis=0)
    p5_template = med_p5 - med_nonp5

    F, feature_names = build_p5_signature_features(power_sc, p5_template)
    y_bin = (y == p5_id).astype(int)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight={0: 1.0, 1: 1.5},
        solver="lbfgs",
    )
    clf.fit(F[train], y_bin[train])

    score = clf.predict_proba(F)[:, 1]
    pred_bin = (score >= args.threshold).astype(int)

    print("\nP5 signature binary test:")
    print(classification_report(y_bin[test], pred_bin[test], target_names=["non-P5", "P5"], zero_division=0))
    print(confusion_matrix(y_bin[test], pred_bin[test]))
    print("accuracy:", accuracy_score(y_bin[test], pred_bin[test]))

    np.savez(
        out,
        threshold=np.array(args.threshold, dtype=np.float32),
        coef=clf.coef_.astype(np.float32),
        intercept=clf.intercept_.astype(np.float32),
        p5_template=p5_template.astype(np.float32),
        feature_names=np.array(feature_names),
        labels=np.array(args.labels),
    )

    info = {
        "threshold": args.threshold,
        "labels": args.labels,
        "feature_names": feature_names,
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "out": str(out),
    }

    out.with_suffix(".json").write_text(json.dumps(info, indent=2))

    print("\nSaved:")
    print(out)
    print(out.with_suffix(".json"))


if __name__ == "__main__":
    main()
