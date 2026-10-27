#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


EPS = 1e-9


class RxGridSSBCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=(7, 2), padding=(3, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(16, 32, kernel_size=(5, 2), padding=(2, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(32, 64, kernel_size=(3, 2), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),

            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


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
    """
    power_sc: [N, 240]
    a,b: 1-based inclusive
    """
    a0 = max(0, a - 1)
    b0 = min(240, b)
    return power_sc[:, a0:b0].mean(axis=1)


def robust_profile(power_sc: np.ndarray):
    """
    Normalización por muestra para quitar ganancia global.
    Así miramos la forma, no la potencia absoluta.
    """
    med = np.median(power_sc, axis=1, keepdims=True)
    q25 = np.percentile(power_sc, 25, axis=1, keepdims=True)
    q75 = np.percentile(power_sc, 75, axis=1, keepdims=True)
    iqr = np.maximum(q75 - q25, 1e-3)
    return (power_sc - med) / iqr


def load_dataset(dataset: Path, labels: list[str]):
    label_to_id = {label: i for i, label in enumerate(labels)}

    X_parts = []
    y_parts = []
    power_parts = []
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

            rx = data[60:300, 1:5, idx]        # [240, 4, N]
            rx = np.transpose(rx, (2, 0, 1))   # [N, 240, 4]

            mag = np.abs(rx).astype(np.float32)
            phase = np.angle(rx).astype(np.float32)

            X = np.stack([mag, phase], axis=1).astype(np.float32)

            power_sc = 10.0 * np.log10(np.mean(np.abs(rx) ** 2, axis=2) + EPS).astype(np.float32)

            y = np.full(len(rx), label_to_id[label], dtype=np.int64)

            X_parts.append(X)
            y_parts.append(y)
            power_parts.append(power_sc)

            for local_i, capture_idx in enumerate(idx):
                rows.append({
                    "label": label,
                    "label_id": label_to_id[label],
                    "file": str(p),
                    "block": b,
                    "split": split,
                    "capture_idx": int(capture_idx + 1),
                    "local_i": int(local_i),
                })

            print(f"  {p} | valid={len(rx)} | block={b} | split={split}")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    power_sc = np.concatenate(power_parts, axis=0)
    meta = pd.DataFrame(rows)

    return X, y, power_sc, meta, label_to_id


def build_p5_signature_features(power_sc: np.ndarray, template: np.ndarray | None = None):
    """
    Features explícitas de firma P5.

    P5 visualmente:
      - valle fuerte en 50:140
      - zona alta 185:240 relativamente elevada
      - forma completa parecida a una plantilla P5-empty
    """
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

    if template is None:
        template_score = np.zeros(len(power_sc), dtype=np.float32)
    else:
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


def load_model(model_path: Path, labels: list[str], device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    class_names = ckpt.get("class_names", ckpt.get("classes", labels))
    model = RxGridSSBCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mean = ckpt.get("mean", ckpt.get("x_mean"))
    std = ckpt.get("std", ckpt.get("x_std"))

    if mean is None or std is None:
        raise RuntimeError("Checkpoint does not contain mean/std")

    mean = np.asarray(mean).astype(np.float32)
    std = np.asarray(std).astype(np.float32)

    return model, class_names, mean, std


def predict_model(model, Xn, device, batch_size=512):
    probs_all = []

    with torch.no_grad():
        for i in range(0, len(Xn), batch_size):
            xb = torch.from_numpy(Xn[i:i+batch_size]).to(device, dtype=torch.float32)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            probs_all.append(probs)

    probs = np.concatenate(probs_all, axis=0)
    pred = probs.argmax(axis=1)

    return pred, probs


def print_report(title, y_true, y_pred, labels):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print("accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))))


def plot_hist(scores, y, labels, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.8))

    for i, label in enumerate(labels):
        ax.hist(scores[y == i], bins=80, alpha=0.45, density=True, label=label)

    ax.set_title("Distribución del score físico P5")
    ax.set_xlabel("P5 signature score")
    ax.set_ylabel("Densidad")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/dataset_datassb/datassb_side_v1_6labels")
    parser.add_argument("--model", default="results/multiclass_empty_P5_P3_rx/model_rxGridSSB/model.pt")
    parser.add_argument("--out-dir", default="results/p5_signature_override_check")
    parser.add_argument("--labels", nargs="+", default=["empty", "P5", "P3"])
    parser.add_argument("--max-false-p5-rate", type=float, default=0.01)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, power_sc, meta, label_to_id = load_dataset(dataset, args.labels)

    p5_id = label_to_id["P5"]

    train = meta["split"].to_numpy() == "train"
    val = meta["split"].to_numpy() == "val"
    test = meta["split"].to_numpy() == "test"

    # Plantilla P5 basada en perfiles robustos de train.
    prof_train = robust_profile(power_sc[train])
    med_p5 = np.median(prof_train[y[train] == p5_id], axis=0)
    med_nonp5 = np.median(prof_train[y[train] != p5_id], axis=0)
    p5_template = med_p5 - med_nonp5

    F, feature_names = build_p5_signature_features(power_sc, p5_template)

    # Clasificador simple P5 vs no-P5 usando solo features físicas.
    y_bin = (y == p5_id).astype(int)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight={0: 1.0, 1: 1.5},
        solver="lbfgs",
    )
    clf.fit(F[train], y_bin[train])

    p5_score = clf.predict_proba(F)[:, 1]

    # Modelo base.
    model, class_names, mean, std = load_model(model_path, args.labels, device)
    Xn = ((X - mean) / (std + 1e-8)).astype(np.float32)
    base_pred, base_probs = predict_model(model, Xn, device)

    print_report("BASE MODEL | TEST", y[test], base_pred[test], args.labels)

    # Umbral 1: maximiza macro-F1 en validación después de override.
    thresholds = np.linspace(0.01, 0.99, 199)

    best = {
        "threshold": None,
        "macro_f1": -1,
        "p5_recall": None,
        "false_p5_rate_nonp5": None,
    }

    for th in thresholds:
        pred_val = base_pred[val].copy()
        pred_val[p5_score[val] >= th] = p5_id

        macro_f1 = f1_score(y[val], pred_val, average="macro", zero_division=0)

        nonp5_val = y[val] != p5_id
        false_p5_rate = np.mean(pred_val[nonp5_val] == p5_id)

        p5_val = y[val] == p5_id
        p5_recall = np.mean(pred_val[p5_val] == p5_id)

        if macro_f1 > best["macro_f1"]:
            best = {
                "threshold": float(th),
                "macro_f1": float(macro_f1),
                "p5_recall": float(p5_recall),
                "false_p5_rate_nonp5": float(false_p5_rate),
            }

    # Umbral 2: umbral conservador: máximo P5 recall con falso P5 <= max-false-p5-rate.
    best_conservative = {
        "threshold": None,
        "p5_recall": -1,
        "macro_f1": None,
        "false_p5_rate_nonp5": None,
    }

    for th in thresholds:
        pred_val = base_pred[val].copy()
        pred_val[p5_score[val] >= th] = p5_id

        nonp5_val = y[val] != p5_id
        false_p5_rate = np.mean(pred_val[nonp5_val] == p5_id)

        if false_p5_rate > args.max_false_p5_rate:
            continue

        p5_val = y[val] == p5_id
        p5_recall = np.mean(pred_val[p5_val] == p5_id)
        macro_f1 = f1_score(y[val], pred_val, average="macro", zero_division=0)

        if p5_recall > best_conservative["p5_recall"]:
            best_conservative = {
                "threshold": float(th),
                "p5_recall": float(p5_recall),
                "macro_f1": float(macro_f1),
                "false_p5_rate_nonp5": float(false_p5_rate),
            }

    print("\nBest threshold by validation macro-F1:")
    print(json.dumps(best, indent=2))

    print("\nBest conservative threshold:")
    print(json.dumps(best_conservative, indent=2))

    # Evaluar threshold macro-F1.
    th = best["threshold"]
    pred_override = base_pred.copy()
    pred_override[p5_score >= th] = p5_id

    print_report(f"BASE + P5 SIGNATURE OVERRIDE | TEST | th={th:.4f}", y[test], pred_override[test], args.labels)

    # Evaluar threshold conservador si existe.
    if best_conservative["threshold"] is not None:
        th2 = best_conservative["threshold"]
        pred_cons = base_pred.copy()
        pred_cons[p5_score >= th2] = p5_id

        print_report(f"BASE + CONSERVATIVE P5 OVERRIDE | TEST | th={th2:.4f}", y[test], pred_cons[test], args.labels)

    # P5 signature solo como binario P5/no-P5.
    p5_pred_bin = (p5_score >= th).astype(int)

    print("\n" + "=" * 100)
    print("P5 SIGNATURE ONLY | TEST | binary P5 vs non-P5")
    print("=" * 100)
    print(classification_report(y_bin[test], p5_pred_bin[test], target_names=["non-P5", "P5"], zero_division=0))
    print(confusion_matrix(y_bin[test], p5_pred_bin[test]))

    # Guardar resultados por muestra.
    out = meta.copy()
    out["y_true"] = y
    out["base_pred"] = base_pred
    out["p5_score"] = p5_score
    out["override_pred"] = pred_override
    out.to_csv(out_dir / "p5_signature_override_predictions.csv", index=False)

    metrics = {
        "labels": args.labels,
        "feature_names": feature_names,
        "best_threshold_macro_f1": best,
        "best_threshold_conservative": best_conservative,
        "test_base_accuracy": float(accuracy_score(y[test], base_pred[test])),
        "test_override_accuracy": float(accuracy_score(y[test], pred_override[test])),
        "test_base_cm": confusion_matrix(y[test], base_pred[test], labels=list(range(len(args.labels)))).tolist(),
        "test_override_cm": confusion_matrix(y[test], pred_override[test], labels=list(range(len(args.labels)))).tolist(),
        "logistic_coef": clf.coef_.tolist(),
        "logistic_intercept": clf.intercept_.tolist(),
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_hist(p5_score[train], y[train], args.labels, out_dir / "p5_signature_score_train.png")
    plot_hist(p5_score[val], y[val], args.labels, out_dir / "p5_signature_score_val.png")
    plot_hist(p5_score[test], y[test], args.labels, out_dir / "p5_signature_score_test.png")

    print("\nSaved:")
    print(out_dir / "metrics.json")
    print(out_dir / "p5_signature_override_predictions.csv")
    print(out_dir / "p5_signature_score_test.png")


if __name__ == "__main__":
    main()
