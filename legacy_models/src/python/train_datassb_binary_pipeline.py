#!/usr/bin/env python3
"""
Pipeline binario empty vs P5 para dataSSB SSB.

Salidas principales:
  results/binary_empty_vs_P5/
    figures/
    model_raw_dataSSB_group3/
    model_rxGridSSB/
    model_hSSB/

Notas:
  - Los .mat v7.3 requieren h5py.
  - El entrenamiento requiere torch. Las metricas ROC/PR y PCA usan sklearn
    cuando esta instalado.
  - hSSB usa, por defecto, un proxy determinista basado en rxGridSSB. Para una
    estimacion fisica con nrChannelEstimate, ejecutar primero el script MATLAB
    src/matlab/export_datassb_hssb_dataset.m y pasar --hssb-dir.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


EPS = 1e-9
LABEL_TO_ID = {"empty": 0, "P5": 1}
ID_TO_LABEL = ["empty", "P5"]


def require_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "No se puede leer MATLAB v7.3 sin h5py. Instala: "
            "python3 -m pip install h5py"
        ) from exc
    return h5py


def optional_sklearn():
    try:
        from sklearn.decomposition import PCA  # type: ignore
        from sklearn.metrics import (  # type: ignore
            auc,
            average_precision_score,
            classification_report,
            precision_recall_curve,
            roc_auc_score,
            roc_curve,
        )

        return {
            "PCA": PCA,
            "auc": auc,
            "average_precision_score": average_precision_score,
            "classification_report": classification_report,
            "precision_recall_curve": precision_recall_curve,
            "roc_auc_score": roc_auc_score,
            "roc_curve": roc_curve,
        }
    except ImportError:
        return None


def optional_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        from torch.utils.data import DataLoader, Dataset  # type: ignore

        return torch, nn, DataLoader, Dataset
    except ImportError:
        return None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def read_hdf5_mat_variable(mat_path: Path, name: str):
    h5py = require_h5py()
    with h5py.File(mat_path, "r") as f:
        if name not in f:
            raise KeyError(f"No existe {name} en {mat_path}")
        data = f[name][()]

    if getattr(data, "dtype", None) is not None and data.dtype.fields is not None:
        fields = data.dtype.fields.keys()
        if "real" in fields and "imag" in fields:
            data = data["real"] + 1j * data["imag"]
        else:
            raise ValueError(f"Compound dtype inesperado en {name}: {list(fields)}")

    return np.asarray(data)


def read_optional_string(mat_path: Path, name: str) -> str | None:
    try:
        h5py = require_h5py()
        with h5py.File(mat_path, "r") as f:
            if name not in f:
                return None
            obj = f[name]
            arr = obj[()]
            if np.issubdtype(arr.dtype, np.integer):
                flat = np.asarray(arr).reshape(-1)
                return "".join(chr(int(x)) for x in flat if int(x) > 0)
            if arr.dtype.kind in {"S", "U"}:
                return str(arr.squeeze())
    except Exception:
        return None
    return None


def force_data_ssb_to_n_360_6(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"dataSSB debe ser 3D, shape={data.shape}")
    shape = data.shape
    if 360 not in shape or 6 not in shape:
        raise ValueError(f"No encuentro dimensiones 360 y 6 en dataSSB shape={shape}")
    ax_sc = shape.index(360)
    ax_sym = shape.index(6)
    ax_n = [i for i in range(3) if i not in (ax_sc, ax_sym)][0]
    out = np.transpose(data, (ax_n, ax_sc, ax_sym))
    return clean_complex(out)


def force_hssb_to_n_240_4(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"hSSB debe ser 3D, shape={data.shape}")
    shape = data.shape
    if 240 not in shape or 4 not in shape:
        raise ValueError(f"No encuentro dimensiones 240 y 4 en hSSB shape={shape}")
    ax_sc = shape.index(240)
    ax_sym = shape.index(4)
    ax_n = [i for i in range(3) if i not in (ax_sc, ax_sym)][0]
    out = np.transpose(data, (ax_n, ax_sc, ax_sym))
    return clean_complex(out)


def clean_complex(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex64)
    real = np.nan_to_num(x.real, nan=0.0, posinf=0.0, neginf=0.0)
    imag = np.nan_to_num(x.imag, nan=0.0, posinf=0.0, neginf=0.0)
    return (real + 1j * imag).astype(np.complex64)


def scalar_from_mat(value: np.ndarray | Any, default: int | str | None = None):
    try:
        arr = np.asarray(value).squeeze()
        if arr.size == 0:
            return default
        return arr.reshape(-1)[0].item()
    except Exception:
        return default


def parse_block_index(path: Path) -> int:
    for part in reversed(path.parts):
        m = re.match(r"block[_-]?(\d+)", part, re.IGNORECASE)
        if m:
            return int(m.group(1))
    m = re.search(r"block[_-]?(\d+)", path.name, re.IGNORECASE)
    return int(m.group(1)) if m else -1


@dataclass
class Capture:
    label: str
    y: int
    block_index: int
    session_id: str
    file_path: str
    capture_index: int
    block_key: str
    data_full: np.ndarray
    rx_grid: np.ndarray
    h_ssb: np.ndarray


def proxy_hssb_from_rx(rx: np.ndarray) -> np.ndarray:
    """Proxy estable cuando no hay hSSB MATLAB: elimina fase/potencia global por SSB."""
    rx = clean_complex(rx)
    ref = np.median(rx, axis=0, keepdims=True)
    ref = np.where(np.abs(ref) < EPS, 1.0 + 0j, ref)
    h = rx / ref
    # Reinyecta un nivel global suave para no borrar toda la informacion de potencia.
    gain = np.sqrt(np.mean(np.abs(rx) ** 2, axis=(0, 1), keepdims=True) + EPS)
    return clean_complex(h * gain)


def load_hssb_sidecar(hssb_dir: Path | None, mat_path: Path) -> np.ndarray | None:
    if hssb_dir is None:
        return None
    candidates = [
        hssb_dir / f"{mat_path.stem}_hssb.npz",
        hssb_dir / f"{mat_path.stem}.npz",
        hssb_dir / f"{mat_path.stem}_hssb.mat",
        hssb_dir / f"{mat_path.stem}.mat",
        hssb_dir / mat_path.parent.parent.name / mat_path.parent.name / f"{mat_path.stem}_hssb.npz",
        hssb_dir / mat_path.parent.parent.name / mat_path.parent.name / f"{mat_path.stem}.npz",
        hssb_dir / mat_path.parent.parent.name / mat_path.parent.name / f"{mat_path.stem}_hssb.mat",
        hssb_dir / mat_path.parent.parent.name / mat_path.parent.name / f"{mat_path.stem}.mat",
    ]
    for candidate in candidates:
        if candidate.exists():
            if candidate.suffix == ".npz":
                data = np.load(candidate, allow_pickle=True)
                key = "hSSB" if "hSSB" in data else "h_ssb"
                return force_hssb_to_n_240_4(data[key])
            return force_hssb_to_n_240_4(read_hdf5_mat_variable(candidate, "hSSB"))
    return None


def find_mat_files(dataset_dir: Path, labels: list[str]) -> list[Path]:
    files: list[Path] = []
    for label in labels:
        files.extend(sorted((dataset_dir / label).rglob("*.mat")))
    return sorted(files)


def load_captures(dataset_dir: Path, hssb_dir: Path | None) -> tuple[list[Capture], list[dict[str, Any]]]:
    captures: list[Capture] = []
    file_rows: list[dict[str, Any]] = []

    mat_files = find_mat_files(dataset_dir, list(LABEL_TO_ID))
    for file_idx, mat_path in enumerate(mat_files, start=1):
        print(f"Cargando {file_idx}/{len(mat_files)}: {mat_path}", flush=True)
        label = mat_path.relative_to(dataset_dir).parts[0]
        y = LABEL_TO_ID[label]
        data = force_data_ssb_to_n_360_6(read_hdf5_mat_variable(mat_path, "dataSSB"))
        valid = np.asarray(read_hdf5_mat_variable(mat_path, "validMask")).reshape(-1).astype(bool)
        n = min(data.shape[0], valid.shape[0])
        data = data[:n]
        valid = valid[:n]

        block_index = parse_block_index(mat_path)
        try:
            block_index_mat = scalar_from_mat(read_hdf5_mat_variable(mat_path, "blockIndex"), None)
            if block_index_mat is not None:
                block_index = int(block_index_mat)
        except Exception:
            pass
        session_id = read_optional_string(mat_path, "sessionId") or mat_path.stem
        block_key = f"{label}:block_{block_index:02d}:{session_id}"
        h_side = load_hssb_sidecar(hssb_dir, mat_path)

        valid_count = int(valid.sum())
        file_rows.append(
            {
                "label": label,
                "blockIndex": block_index,
                "sessionId": session_id,
                "filePath": str(mat_path),
                "nCaptures": int(n),
                "nValid": valid_count,
                "blockKey": block_key,
                "hSSBSource": "sidecar" if h_side is not None else "proxy_rxGridSSB",
            }
        )

        for idx in np.flatnonzero(valid):
            rx = data[idx, 60:300, 1:5]
            if h_side is not None and idx < h_side.shape[0]:
                h = h_side[idx]
            else:
                h = proxy_hssb_from_rx(rx)
            captures.append(
                Capture(
                    label=label,
                    y=y,
                    block_index=block_index,
                    session_id=session_id,
                    file_path=str(mat_path),
                    capture_index=int(idx + 1),
                    block_key=block_key,
                    data_full=data[idx],
                    rx_grid=rx,
                    h_ssb=h,
                )
            )

    return captures, file_rows


def split_blocks(captures: list[Capture], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    by_label: dict[str, list[str]] = {}
    for c in captures:
        by_label.setdefault(c.label, [])
        if c.block_key not in by_label[c.label]:
            by_label[c.label].append(c.block_key)

    split: dict[str, str] = {}
    for label, keys in by_label.items():
        keys = sorted(keys)
        rng.shuffle(keys)
        n = len(keys)
        if n >= 5:
            n_train = max(1, int(round(0.70 * n)))
            n_val = max(1, int(round(0.15 * n)))
            if n_train + n_val >= n:
                n_train = n - 2
                n_val = 1
            train, val, test = keys[:n_train], keys[n_train:n_train + n_val], keys[n_train + n_val:]
        elif n >= 3:
            train, val, test = keys[:-2], keys[-2:-1], keys[-1:]
        elif n == 2:
            train, val, test = keys[:1], keys[:1], keys[1:]
        else:
            train, val, test = keys, keys, keys
        for key in train:
            split[key] = "train"
        for key in val:
            split[key] = "val"
        for key in test:
            split[key] = "test"
    return split


def complex_to_channels(x: np.ndarray, mode: str) -> np.ndarray:
    x = clean_complex(x)
    if mode == "abs_phase":
        out = np.stack([np.abs(x), np.angle(x)], axis=0)
    elif mode == "real_imag":
        out = np.stack([x.real, x.imag], axis=0)
    else:
        raise ValueError(f"complex_mode no soportado: {mode}")
    return np.nan_to_num(out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def build_single_dataset(captures: list[Capture], view: str, complex_mode: str):
    X, y, rows, groups = [], [], [], []
    for c in captures:
        arr = c.rx_grid if view == "rx" else c.h_ssb
        X.append(complex_to_channels(arr, complex_mode))
        y.append(c.y)
        groups.append(c.block_key)
        rows.append(meta_row(c, sample_id=len(rows), capture_indices=str(c.capture_index), group_size=1))
    return np.stack(X), np.asarray(y, dtype=np.int64), rows, groups


def build_raw_group_dataset(captures: list[Capture], complex_mode: str, group_size: int, stride: int):
    X, y, rows, groups = [], [], [], []
    by_block: dict[str, list[Capture]] = {}
    for c in captures:
        by_block.setdefault(c.block_key, []).append(c)
    for key, items in sorted(by_block.items()):
        items = sorted(items, key=lambda c: c.capture_index)
        for start in range(0, len(items) - group_size + 1, stride):
            win = items[start:start + group_size]
            arr = np.stack([c.data_full for c in win], axis=0)  # T,H,W
            ch = complex_to_channels(arr, complex_mode)  # C,T,H,W
            X.append(ch)
            y.append(win[0].y)
            groups.append(key)
            rows.append(
                meta_row(
                    win[0],
                    sample_id=len(rows),
                    capture_indices=";".join(str(c.capture_index) for c in win),
                    group_size=group_size,
                )
            )
    return np.stack(X), np.asarray(y, dtype=np.int64), rows, groups


def meta_row(c: Capture, sample_id: int, capture_indices: str, group_size: int) -> dict[str, Any]:
    return {
        "sampleId": sample_id,
        "label": c.label,
        "y": c.y,
        "blockIndex": c.block_index,
        "sessionId": c.session_id,
        "filePath": c.file_path,
        "captureIndex": capture_indices,
        "groupSize": group_size,
        "blockKey": c.block_key,
    }


def standardize_train_only(X: np.ndarray, train_mask: np.ndarray):
    axes = tuple(i for i in range(X.ndim) if i != 1)
    mean = X[train_mask].mean(axis=axes, keepdims=True)
    std = X[train_mask].std(axis=axes, keepdims=True)
    std = np.maximum(std, 1e-4)
    return ((X - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def write_csv(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def power_db(x: np.ndarray, axis=None) -> np.ndarray:
    p = np.abs(x) ** 2
    if axis is not None:
        p = np.mean(p, axis=axis)
    return (10.0 * np.log10(p + EPS)).astype(np.float32)


def robust_limits(values: np.ndarray, p_low=2, p_high=98, symmetric=False):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1.0, 1.0)
    if symmetric:
        lim = float(np.percentile(np.abs(values), p_high))
        return -max(lim, 1e-3), max(lim, 1e-3)
    lo = float(np.percentile(values, p_low))
    hi = float(np.percentile(values, p_high))
    return (lo, hi if hi > lo else lo + 1.0)


def generate_figures(captures: list[Capture], out_dir: Path):
    fig_dir = ensure_dir(out_dir / "figures")
    rows_report: list[str] = []
    by_label = {label: [c for c in captures if c.label == label] for label in LABEL_TO_ID}

    per_sc: dict[str, np.ndarray] = {}
    per_grid: dict[str, np.ndarray] = {}
    total_power: dict[str, np.ndarray] = {}
    for label, items in by_label.items():
        rx = np.stack([c.rx_grid for c in items])
        per_sc[label] = power_db(rx, axis=2)
        per_grid[label] = power_db(rx)
        total_power[label] = power_db(rx, axis=(1, 2))

    x = np.arange(1, 241)
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {"empty": "#1f77b4", "P5": "#d62728"}
    for label in ["empty", "P5"]:
        vals = per_sc[label]
        q25, q50, q75 = np.percentile(vals, [25, 50, 75], axis=0)
        ax.plot(x, q50, label=f"{label} mediana", color=colors[label])
        ax.fill_between(x, q25, q75, alpha=0.18, color=colors[label], label=f"{label} IQR")
    ax.set_xlabel("Subportadora SSB")
    ax.set_ylabel("Potencia media por simbolo [dB]")
    ax.set_title("Potencia por subportadora: empty vs P5")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "power_by_subcarrier_empty_vs_P5.png", dpi=180)
    plt.close(fig)

    med_empty = np.median(per_sc["empty"], axis=0)
    med_p5 = np.median(per_sc["P5"], axis=0)
    delta = med_p5 - med_empty
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, delta, color="#111111")
    ax.axhline(0, color="#777777", linewidth=1)
    ax.set_xlabel("Subportadora SSB")
    ax.set_ylabel("Delta potencia [dB]")
    ax.set_title("P5 - empty por subportadora")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "delta_power_by_subcarrier_P5_minus_empty.png", dpi=180)
    plt.close(fig)

    q25_e, q75_e = np.percentile(per_sc["empty"], [25, 75], axis=0)
    q25_p, q75_p = np.percentile(per_sc["P5"], [25, 75], axis=0)
    iqr_pooled = np.maximum(0.5 * ((q75_e - q25_e) + (q75_p - q25_p)), 1e-3)
    effect = delta / iqr_pooled
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, effect, color="#5b2c6f")
    ax.axhline(0, color="#777777", linewidth=1)
    ax.set_xlabel("Subportadora SSB")
    ax.set_ylabel("Efecto robusto [IQR pooled]")
    ax.set_title("Tamano de efecto robusto P5 vs empty")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "effect_size_by_subcarrier_P5_vs_empty.png", dpi=180)
    plt.close(fig)

    delta_grid = np.median(per_grid["P5"], axis=0) - np.median(per_grid["empty"], axis=0)
    vmin, vmax = robust_limits(delta_grid, symmetric=True)
    fig, ax = plt.subplots(figsize=(5.8, 7.0))
    im = ax.imshow(
        delta_grid,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        extent=[0.5, 4.5, 0.5, 240.5],
    )
    ax.set_xlabel("Simbolo OFDM SSB")
    ax.set_ylabel("Subportadora SSB")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_title("Delta potencia rxGridSSB: P5 - empty")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Delta [dB]")
    fig.tight_layout()
    fig.savefig(fig_dir / "delta_power_heatmap_rxGridSSB_P5_minus_empty.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    data = [total_power["empty"], total_power["P5"]]
    try:
        ax.boxplot(data, tick_labels=["empty", "P5"], showmeans=True)
    except TypeError:
        ax.boxplot(data, labels=["empty", "P5"], showmeans=True)
    ax.scatter(np.ones_like(data[0]) + np.random.default_rng(0).normal(0, 0.025, data[0].shape), data[0], s=7, alpha=0.25)
    ax.scatter(2 * np.ones_like(data[1]) + np.random.default_rng(1).normal(0, 0.025, data[1].shape), data[1], s=7, alpha=0.25)
    ax.set_ylabel("Potencia total media [dB]")
    ax.set_title("Distribucion de potencia total por SSB")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "total_power_distribution_empty_vs_P5.png", dpi=180)
    plt.close(fig)

    sk = optional_sklearn()
    if sk is not None:
        X = np.concatenate([per_sc["empty"], per_sc["P5"]], axis=0)
        labels = np.array(["empty"] * len(per_sc["empty"]) + ["P5"] * len(per_sc["P5"]))
        Xz = (X - X.mean(axis=0, keepdims=True)) / np.maximum(X.std(axis=0, keepdims=True), 1e-6)
        Z = sk["PCA"](n_components=2, random_state=0).fit_transform(Xz)
        fig, ax = plt.subplots(figsize=(6, 5))
        for label in ["empty", "P5"]:
            mask = labels == label
            ax.scatter(Z[mask, 0], Z[mask, 1], s=10, alpha=0.55, label=label)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA de potencia por subportadora")
        ax.legend()
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(fig_dir / "pca_power_features_empty_vs_P5.png", dpi=180)
        plt.close(fig)
    else:
        rows_report.append("PCA omitida: scikit-learn no esta instalado.")

    for label in ["empty", "P5"]:
        rows_report.append(
            f"{label}: total_power median={np.median(total_power[label]):.3f} dB, "
            f"mean={np.mean(total_power[label]):.3f} dB, n={len(total_power[label])}"
        )
    (fig_dir / "analysis_notes.txt").write_text("\n".join(rows_report) + "\n")


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm.tolist()


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None = None) -> dict[str, Any]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    out: dict[str, Any] = {
        "accuracy": acc,
        "precision_P5": precision,
        "recall_P5": recall,
        "f1_P5": f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
    sk = optional_sklearn()
    if sk is not None and scores is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(sk["roc_auc_score"](y_true, scores))
        out["pr_auc"] = float(sk["average_precision_score"](y_true, scores))
        out["classification_report"] = sk["classification_report"](
            y_true, y_pred, target_names=ID_TO_LABEL, digits=4, zero_division=0
        )
    return out


def plot_confusion(cm: list[list[int]], out_path: Path):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    arr = np.asarray(cm)
    im = ax.imshow(arr, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], ID_TO_LABEL)
    ax.set_yticks([0, 1], ID_TO_LABEL)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_curves(y_true: np.ndarray, scores: np.ndarray, out_dir: Path):
    sk = optional_sklearn()
    if sk is None or len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = sk["roc_curve"](y_true, scores)
    precision, recall, _ = sk["precision_recall_curve"](y_true, scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC={sk['auc'](fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], color="#777777", linewidth=1, linestyle="--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"AP={sk['average_precision_score'](y_true, scores):.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve.png", dpi=170)
    plt.close(fig)


def train_torch_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    split_by_group: dict[str, str],
    rows: list[dict[str, Any]],
    out_dir: Path,
    args: argparse.Namespace,
):
    torch_pack = optional_torch()
    if torch_pack is None:
        raise RuntimeError(
            "El entrenamiento requiere PyTorch. Instala una version adecuada, por ejemplo: "
            "python3 -m pip install torch scikit-learn pandas h5py"
        )

    torch, nn, DataLoader, Dataset = torch_pack
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(out_dir)

    split = np.array([split_by_group[g] for g in groups])
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    if not train_mask.any() or not test_mask.any():
        raise RuntimeError(f"Split invalido para {model_name}: train={train_mask.sum()} test={test_mask.sum()}")
    if not val_mask.any():
        val_mask = test_mask

    Xz, mean, std = standardize_train_only(X, train_mask)
    write_csv(out_dir / "metadata.csv", [{**r, "split": split_by_group[r["blockKey"]]} for r in rows])
    if not args.skip_dataset_cache:
        np.savez_compressed(
            out_dir / "dataset_cache.npz",
            X=X.astype(np.float32),
            X_standardized=Xz.astype(np.float32),
            y=y,
            split=split,
            groups=np.asarray(groups),
        )

    class ArrayDataset(Dataset):
        def __init__(self, x, yy):
            self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
            self.y = torch.from_numpy(np.asarray(yy, dtype=np.float32))

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    class CNN2D(nn.Module):
        def __init__(self, in_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 16, kernel_size=(7, 2), padding=(3, 1)),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 1)),
                nn.Conv2d(16, 32, kernel_size=(5, 2), padding=(2, 1)),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 1)),
                nn.Conv2d(32, 64, kernel_size=(3, 2), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(0.30),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Dropout(0.20),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(1)

    class CNN3D(nn.Module):
        def __init__(self, in_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(in_ch, 16, kernel_size=(2, 7, 3), padding=(0, 3, 1)),
                nn.BatchNorm3d(16),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(1, 2, 1)),
                nn.Conv3d(16, 32, kernel_size=(2, 5, 3), padding=(0, 2, 1)),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(1, 2, 1)),
                nn.Conv3d(32, 64, kernel_size=(1, 3, 2), padding=(0, 1, 1)),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Dropout(0.30),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Dropout(0.20),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(1)

    is_3d = Xz.ndim == 5
    model = (CNN3D if is_3d else CNN2D)(Xz.shape[1]).to(device)
    pos = max(1, int(y[train_mask].sum()))
    neg = max(1, int(train_mask.sum() - y[train_mask].sum()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(ArrayDataset(Xz[train_mask], y[train_mask]), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ArrayDataset(Xz[val_mask], y[val_mask]), batch_size=args.batch_size, shuffle=False)
    history = []
    best_state = None
    best_f1 = -1.0
    bad = 0

    torch.manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        probs, yt = predict_torch(model, val_loader, device)
        pred = (probs >= 0.5).astype(int)
        m = binary_metrics(yt.astype(int), pred, probs)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_accuracy": m["accuracy"],
            "val_precision_P5": m["precision_P5"],
            "val_recall_P5": m["recall_P5"],
            "val_f1_P5": m["f1_P5"],
        }
        history.append(row)
        print(
            f"{model_name} epoch {epoch:03d} "
            f"loss={row['train_loss']:.4f} val_f1={row['val_f1_P5']:.4f}",
            flush=True,
        )
        if row["val_f1_P5"] > best_f1:
            best_f1 = row["val_f1_P5"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    write_csv(out_dir / "train_history.csv", history)

    test_loader = DataLoader(ArrayDataset(Xz[test_mask], y[test_mask]), batch_size=args.batch_size, shuffle=False)
    t0 = time.perf_counter()
    probs, yt = predict_torch(model, test_loader, device)
    predict_s = time.perf_counter() - t0
    pred = (probs >= 0.5).astype(int)
    metrics = binary_metrics(yt.astype(int), pred, probs)
    metrics.update(
        {
            "model": model_name,
            "device": str(device),
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_test": int(test_mask.sum()),
            "predict_ms_per_sample": float(1000.0 * predict_s / max(1, test_mask.sum())),
            "config": vars(args),
        }
    )

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=json_default))
    if "classification_report" in metrics:
        (out_dir / "classification_report.txt").write_text(metrics["classification_report"])
    plot_confusion(metrics["confusion_matrix"], out_dir / "confusion_matrix.png")
    plot_curves(yt.astype(int), probs, out_dir)
    torch.save(
        {
            "model_state_dict": model.to("cpu").state_dict(),
            "mean": mean,
            "std": std,
            "model_name": model_name,
            "input_shape": list(X.shape[1:]),
            "complex_mode": args.complex_mode,
            "classes": ID_TO_LABEL,
            "config": vars(args),
        },
        out_dir / "model.pt",
    )
    return metrics


def predict_torch(model, loader, device):
    torch_pack = optional_torch()
    assert torch_pack is not None
    torch = torch_pack[0]
    probs, y_true = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            y_true.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(y_true)


def summarize(captures: list[Capture], file_rows: list[dict[str, Any]], split_by_group: dict[str, str], out_dir: Path):
    summary: dict[str, Any] = {
        "n_files": len(file_rows),
        "n_valid_captures": len(captures),
        "labels": {},
        "splits_by_block": {},
        "files": file_rows,
    }
    for label in LABEL_TO_ID:
        items = [c for c in captures if c.label == label]
        blocks = sorted({c.block_key for c in items})
        powers = np.array([float(power_db(c.rx_grid, axis=(0, 1))) for c in items], dtype=np.float32)
        summary["labels"][label] = {
            "n_valid_captures": len(items),
            "n_blocks": len(blocks),
            "blocks": blocks,
            "total_power_mean_db": float(np.mean(powers)) if len(powers) else None,
            "total_power_median_db": float(np.median(powers)) if len(powers) else None,
        }
    for key, part in sorted(split_by_group.items()):
        summary["splits_by_block"][key] = part
    (out_dir / "quick_report.json").write_text(json.dumps(summary, indent=2, default=json_default))

    lines = [
        "# Informe rapido dataSSB empty vs P5",
        "",
        f"Archivos encontrados: {summary['n_files']}",
        f"Capturas validas: {summary['n_valid_captures']}",
        "",
    ]
    for label, info in summary["labels"].items():
        lines.extend(
            [
                f"## {label}",
                f"- Capturas validas: {info['n_valid_captures']}",
                f"- Bloques: {info['n_blocks']}",
                f"- Potencia media: {info['total_power_mean_db']:.3f} dB",
                f"- Potencia mediana: {info['total_power_median_db']:.3f} dB",
                "",
            ]
        )
    lines.append("## Split por bloque")
    for key, part in sorted(split_by_group.items()):
        lines.append(f"- {part}: {key}")
    lines.extend(
        [
            "",
            "## Figuras",
            "- figures/power_by_subcarrier_empty_vs_P5.png",
            "- figures/delta_power_by_subcarrier_P5_minus_empty.png",
            "- figures/effect_size_by_subcarrier_P5_vs_empty.png",
            "- figures/delta_power_heatmap_rxGridSSB_P5_minus_empty.png",
            "- figures/total_power_distribution_empty_vs_P5.png",
            "- figures/pca_power_features_empty_vs_P5.png si scikit-learn esta instalado",
        ]
    )
    (out_dir / "quick_report.md").write_text("\n".join(lines) + "\n")
    write_csv(out_dir / "files_used.csv", file_rows)


def save_config(out_dir: Path, args: argparse.Namespace):
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=json_default))


def parse_model_selection(selection: str) -> set[str]:
    aliases = {
        "all": {"raw", "rx", "h"},
        "raw": {"raw"},
        "dataSSB": {"raw"},
        "rx": {"rx"},
        "rxGridSSB": {"rx"},
        "h": {"h"},
        "hSSB": {"h"},
    }
    selected: set[str] = set()
    for item in selection.split(","):
        key = item.strip()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"Modelo no soportado en --models: {key}")
        selected.update(aliases[key])
    return selected or aliases["all"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/dataset_datassb/datassb_side_v1_6labels")
    parser.add_argument("--out-dir", default="results/binary_empty_vs_P5")
    parser.add_argument("--hssb-dir", default=None, help="Directorio con NPZ generados por export_datassb_hssb_dataset.m")
    parser.add_argument("--complex-mode", choices=["abs_phase", "real_imag"], default="abs_phase")
    parser.add_argument("--raw-group-size", type=int, default=3)
    parser.add_argument("--raw-stride", type=int, default=3)
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-dataset-cache", action="store_true")
    parser.add_argument(
        "--models",
        default="all",
        help="Modelos a construir/entrenar: all, raw, rx, h o lista separada por comas.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    dataset_dir = Path(args.dataset_dir)
    out_dir = ensure_dir(Path(args.out_dir))
    hssb_dir = Path(args.hssb_dir) if args.hssb_dir else None
    save_config(out_dir, args)

    captures, file_rows = load_captures(dataset_dir, hssb_dir)
    if not captures:
        raise RuntimeError(f"No hay capturas validas en {dataset_dir}")
    split_by_group = split_blocks(captures, args.seed)
    summarize(captures, file_rows, split_by_group, out_dir)

    if not args.skip_figures:
        generate_figures(captures, out_dir)

    selected_models = parse_model_selection(args.models)
    dataset_builders_all = [
        (
            "raw",
            "model_raw_dataSSB_group3",
            lambda: build_raw_group_dataset(
                captures, args.complex_mode, args.raw_group_size, args.raw_stride
            ),
        ),
        ("rx", "model_rxGridSSB", lambda: build_single_dataset(captures, "rx", args.complex_mode)),
        ("h", "model_hSSB", lambda: build_single_dataset(captures, "h", args.complex_mode)),
    ]
    dataset_builders = [item for item in dataset_builders_all if item[0] in selected_models]

    dataset_summary = {}
    all_metrics = {}
    for _, name, build_dataset in dataset_builders:
        print(f"\nConstruyendo {name}", flush=True)
        X, y, rows, groups = build_dataset()
        dataset_summary[name] = {
            "shape": list(X.shape),
            "n_samples": int(X.shape[0]),
            "labels": {label: int((y == idx).sum()) for label, idx in LABEL_TO_ID.items()},
        }
        (out_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2))

        if args.analysis_only or args.skip_training:
            print(json.dumps({name: dataset_summary[name]}, indent=2), flush=True)
            del X, y, rows, groups
            gc.collect()
            continue

        model_dir = out_dir / name
        print(f"\nEntrenando {name}: X={X.shape}", flush=True)
        all_metrics[name] = train_torch_model(name, X, y, groups, split_by_group, rows, model_dir, args)
        del X, y, rows, groups
        gc.collect()

    if args.analysis_only or args.skip_training:
        print(json.dumps(dataset_summary, indent=2), flush=True)
        return

    (out_dir / "all_model_metrics.json").write_text(json.dumps(all_metrics, indent=2, default=json_default))


if __name__ == "__main__":
    main()
