#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn

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


def read_mat_variable(mat_path: Path, name: str):
    try:
        mat = scipy.io.loadmat(mat_path)
        if name in mat:
            return mat[name]
    except NotImplementedError:
        pass
    except Exception:
        pass

    with h5py.File(mat_path, "r") as f:
        if name not in f:
            raise KeyError(f"No existe {name} en {mat_path}")
        data = f[name][()]

    if getattr(data, "dtype", None) is not None and data.dtype.fields is not None:
        fields = data.dtype.fields.keys()
        if "real" in fields and "imag" in fields:
            data = data["real"] + 1j * data["imag"]
        else:
            raise ValueError(f"Compound dtype inesperado: {list(fields)}")

    return np.asarray(data)


def clean_complex(x):
    x = np.asarray(x, dtype=np.complex64)
    real = np.nan_to_num(x.real, nan=0.0, posinf=0.0, neginf=0.0)
    imag = np.nan_to_num(x.imag, nan=0.0, posinf=0.0, neginf=0.0)
    return (real + 1j * imag).astype(np.complex64)


def force_data_ssb_to_n_360_6(data):
    data = np.asarray(data)

    if data.ndim != 3:
        raise ValueError(f"dataSSB debe ser 3D, shape={data.shape}")

    shape = data.shape

    if 360 not in shape or 6 not in shape:
        raise ValueError(f"No encuentro 360 y 6 en dataSSB shape={shape}")

    ax_sc = shape.index(360)
    ax_sym = shape.index(6)
    ax_n = [i for i in range(3) if i not in (ax_sc, ax_sym)][0]

    out = np.transpose(data, (ax_n, ax_sc, ax_sym))
    return clean_complex(out)


def load_valid_mask(mat_path: Path, n: int):
    try:
        valid = np.asarray(read_mat_variable(mat_path, "validMask")).reshape(-1).astype(bool)
        return valid[:n]
    except Exception:
        return np.ones(n, dtype=bool)


def parse_block_index(path: Path):
    for part in reversed(path.parts):
        m = re.match(r"block[_-]?(\d+)", part, re.IGNORECASE)
        if m:
            return int(m.group(1))

    m = re.search(r"block[_-]?(\d+)", path.name, re.IGNORECASE)
    return int(m.group(1)) if m else -1


def load_label_x(dataset_dir: Path, label: str, only_block: int | None = None, max_samples: int | None = None, seed: int = 0):
    files = sorted((dataset_dir / label).rglob("*.mat"))

    if only_block is not None:
        files = [p for p in files if parse_block_index(p) == only_block]

    if not files:
        raise FileNotFoundError(f"No hay .mat para label={label}, block={only_block}")

    X_parts = []

    print(f"\nLoading {label} | block={only_block} | files={len(files)}")

    for p in files:
        print(f"  {p}")
        data = force_data_ssb_to_n_360_6(read_mat_variable(p, "dataSSB"))
        valid = load_valid_mask(p, data.shape[0])
        data = data[valid]

        # rxGridSSB validado:
        # MATLAB: dataSSB(61:300, 2:5, :)
        # Python: data[:, 60:300, 1:5]
        rx = data[:, 60:300, 1:5]  # [N, 240, 4]

        mag = np.abs(rx).astype(np.float32)
        phase = np.angle(rx).astype(np.float32)

        x = np.stack([mag, phase], axis=1).astype(np.float32)  # [N, 2, 240, 4]
        X_parts.append(x)

        print(f"    valid: {x.shape[0]}")

    X = np.concatenate(X_parts, axis=0)

    if max_samples is not None and X.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[idx]

    return X


def robust_effect(a, b):
    """
    Efecto robusto: (median(a)-median(b)) / IQR combinado.
    a y b: [N, 240, 4]
    """
    a_q25, a_med, a_q75 = np.percentile(a, [25, 50, 75], axis=0)
    b_q25, b_med, b_q75 = np.percentile(b, [25, 50, 75], axis=0)

    delta = a_med - b_med
    iqr_a = a_q75 - a_q25
    iqr_b = b_q75 - b_q25
    pooled_iqr = np.maximum(0.5 * (iqr_a + iqr_b), 1e-4)

    return delta / pooled_iqr, delta


def save_effect_maps(X_empty_n, X_p3_n, out_dir: Path):
    """
    X_*_n ya normalizado como entra al modelo.
    """
    # Canal 0: abs normalizado
    effect_abs, delta_abs = robust_effect(X_p3_n[:, 0], X_empty_n[:, 0])

    # Canal 1: phase normalizada tal como la ve el modelo
    effect_phase, delta_phase = robust_effect(X_p3_n[:, 1], X_empty_n[:, 1])

    combined = np.sqrt(effect_abs**2 + effect_phase**2)

    vmax_eff = np.percentile(np.abs(np.concatenate([
        effect_abs.reshape(-1),
        effect_phase.reshape(-1),
        combined.reshape(-1),
    ])), 98)
    vmax_eff = max(float(vmax_eff), 0.5)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    maps = [
        (effect_abs, "Efecto robusto en |rxGridSSB|"),
        (effect_phase, "Efecto robusto en phase(rxGridSSB)"),
        (combined, "Efecto combinado abs+phase"),
    ]

    for ax, (m, title) in zip(axes, maps):
        im = ax.imshow(
            m,
            origin="lower",
            aspect="auto",
            cmap="coolwarm" if "combinado" not in title else "viridis",
            vmin=-vmax_eff if "combinado" not in title else 0,
            vmax=vmax_eff,
            extent=[0.5, 4.5, 0.5, 240.5],
        )
        ax.set_title(title)
        ax.set_xlabel("Símbolo OFDM")
        ax.set_xticks([1, 2, 3, 4])
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Efecto robusto")

    axes[0].set_ylabel("Subportadora SSB")
    fig.suptitle("P3 vs empty en la entrada real del modelo normalizada", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "explain_model_input_effect_abs_phase_empty_vs_P3.png", dpi=200)
    plt.close(fig)

    # Resumen por subportadora: máximo efecto combinado entre los 4 símbolos
    combined_by_sc = np.max(combined, axis=1)
    abs_by_sc = np.max(np.abs(effect_abs), axis=1)
    phase_by_sc = np.max(np.abs(effect_phase), axis=1)

    x = np.arange(1, 241)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(x, abs_by_sc, label="abs max efecto")
    ax.plot(x, phase_by_sc, label="phase max efecto")
    ax.plot(x, combined_by_sc, label="combinado max efecto", linewidth=2.0)
    ax.axhline(0.5, linestyle="--", linewidth=1.0)
    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.set_title("P3 vs empty: efecto máximo por subportadora")
    ax.set_xlabel("Subportadora SSB")
    ax.set_ylabel("Efecto robusto")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "explain_model_input_effect_by_subcarrier_empty_vs_P3.png", dpi=200)
    plt.close(fig)

    notes = []
    notes.append("=== P3 vs empty | model input effect ===")
    notes.append(f"abs_effect median_abs={np.median(np.abs(effect_abs)):.4f} max_abs={np.max(np.abs(effect_abs)):.4f}")
    notes.append(f"phase_effect median_abs={np.median(np.abs(effect_phase)):.4f} max_abs={np.max(np.abs(effect_phase)):.4f}")
    notes.append(f"combined_effect median={np.median(combined):.4f} max={np.max(combined):.4f}")
    notes.append("")
    notes.append("Top subcarriers by combined effect:")
    top = np.argsort(combined_by_sc)[-15:][::-1]
    for sc in top:
        notes.append(
            f"  sc={sc+1:03d} combined={combined_by_sc[sc]:.4f} "
            f"abs={abs_by_sc[sc]:.4f} phase={phase_by_sc[sc]:.4f}"
        )

    (out_dir / "explain_model_input_effect_notes.txt").write_text("\n".join(notes) + "\n")


def save_pca(X_empty_n, X_p3_n, out_dir: Path, max_per_label: int = 8000):
    try:
        from sklearn.decomposition import PCA
    except Exception as e:
        print(f"PCA omitida: {e}")
        return

    rng = np.random.default_rng(0)

    if X_empty_n.shape[0] > max_per_label:
        idx = rng.choice(X_empty_n.shape[0], size=max_per_label, replace=False)
        X_empty_n = X_empty_n[idx]

    if X_p3_n.shape[0] > max_per_label:
        idx = rng.choice(X_p3_n.shape[0], size=max_per_label, replace=False)
        X_p3_n = X_p3_n[idx]

    X = np.concatenate([X_empty_n, X_p3_n], axis=0)
    y = np.array(["empty"] * len(X_empty_n) + ["P3"] * len(X_p3_n))

    Xf = X.reshape(X.shape[0], -1)
    Z = PCA(n_components=2, random_state=0).fit_transform(Xf)

    fig, ax = plt.subplots(figsize=(7, 5.8))
    for label in ["empty", "P3"]:
        mask = y == label
        ax.scatter(Z[mask, 0], Z[mask, 1], s=8, alpha=0.35, label=label)

    ax.set_title("PCA usando la entrada completa del modelo: abs + phase")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pca_model_input_abs_phase_empty_vs_P3.png", dpi=200)
    plt.close(fig)


def save_model_confidence_histogram(model_path: Path, X_empty_test_n, X_p3_test_n, out_dir: Path):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    class_names = checkpoint.get("class_names", checkpoint.get("classes", ["empty", "P5", "P3"]))
    if "empty" not in class_names or "P3" not in class_names:
        raise RuntimeError(f"El checkpoint no tiene clases empty/P3: {class_names}")

    empty_id = class_names.index("empty")
    p3_id = class_names.index("P3")

    model = RxGridSSBCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def predict_probs(Xn, batch_size=512):
        probs_all = []
        with torch.no_grad():
            for i in range(0, Xn.shape[0], batch_size):
                xb = torch.from_numpy(Xn[i:i+batch_size]).float()
                logits = model(xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_all.append(probs)
        return np.concatenate(probs_all, axis=0)

    probs_empty = predict_probs(X_empty_test_n)
    probs_p3 = predict_probs(X_p3_test_n)

    # Score binario interpretativo entre empty y P3:
    # ignora P5 y renormaliza entre esas dos clases.
    empty_score_for_p3 = probs_empty[:, p3_id] / (probs_empty[:, p3_id] + probs_empty[:, empty_id] + EPS)
    p3_score_for_p3 = probs_p3[:, p3_id] / (probs_p3[:, p3_id] + probs_p3[:, empty_id] + EPS)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bins = np.linspace(0, 1, 60)
    ax.hist(empty_score_for_p3, bins=bins, alpha=0.55, density=True, label="true empty")
    ax.hist(p3_score_for_p3, bins=bins, alpha=0.55, density=True, label="true P3")
    ax.set_title("Separación del modelo: score P3 frente a empty")
    ax.set_xlabel("score = P(P3) / [P(P3) + P(empty)]")
    ax.set_ylabel("Densidad")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "softmax_p3_score_empty_vs_P3_test.png", dpi=200)
    plt.close(fig)

    pred_empty = np.argmax(probs_empty, axis=1)
    pred_p3 = np.argmax(probs_p3, axis=1)

    notes = []
    notes.append("=== Model confidence | test block only ===")
    notes.append(f"class_names={class_names}")
    notes.append(f"empty test n={len(X_empty_test_n)}")
    notes.append(f"P3 test n={len(X_p3_test_n)}")
    notes.append("")
    notes.append(f"true empty: median score_P3_vs_empty={np.median(empty_score_for_p3):.6f}")
    notes.append(f"true P3:    median score_P3_vs_empty={np.median(p3_score_for_p3):.6f}")
    notes.append("")
    notes.append(f"true empty predicted empty={int(np.sum(pred_empty == empty_id))}")
    notes.append(f"true empty predicted P3={int(np.sum(pred_empty == p3_id))}")
    notes.append(f"true P3 predicted empty={int(np.sum(pred_p3 == empty_id))}")
    notes.append(f"true P3 predicted P3={int(np.sum(pred_p3 == p3_id))}")

    (out_dir / "softmax_p3_score_empty_vs_P3_test_notes.txt").write_text("\n".join(notes) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/dataset_datassb/datassb_side_v1_6labels")
    parser.add_argument("--model", default="results/multiclass_empty_P5_P3_rx/model_rxGridSSB/model.pt")
    parser.add_argument("--out-dir", default="results/multiclass_empty_P5_P3_rx/figures")
    parser.add_argument("--max-effect-samples-per-label", type=int, default=25000)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    mean = checkpoint["mean"].astype(np.float32)
    std = checkpoint["std"].astype(np.float32)

    print("Loaded checkpoint:")
    print(" ", model_path)
    print("mean shape:", mean.shape)
    print("std shape:", std.shape)

    # Para mapas de efecto usamos muestra representativa de todos los bloques.
    X_empty = load_label_x(
        dataset_dir,
        "empty",
        only_block=None,
        max_samples=args.max_effect_samples_per_label,
        seed=1,
    )
    X_p3 = load_label_x(
        dataset_dir,
        "P3",
        only_block=None,
        max_samples=args.max_effect_samples_per_label,
        seed=2,
    )

    X_empty_n = ((X_empty - mean) / (std + 1e-8)).astype(np.float32)
    X_p3_n = ((X_p3 - mean) / (std + 1e-8)).astype(np.float32)

    save_effect_maps(X_empty_n, X_p3_n, out_dir)
    save_pca(X_empty_n, X_p3_n, out_dir)

    # Para histograma de confianza usamos solo test: block_05.
    X_empty_test = load_label_x(dataset_dir, "empty", only_block=5, max_samples=None)
    X_p3_test = load_label_x(dataset_dir, "P3", only_block=5, max_samples=None)

    X_empty_test_n = ((X_empty_test - mean) / (std + 1e-8)).astype(np.float32)
    X_p3_test_n = ((X_p3_test - mean) / (std + 1e-8)).astype(np.float32)

    save_model_confidence_histogram(model_path, X_empty_test_n, X_p3_test_n, out_dir)

    print("\nSaved:")
    for p in [
        "explain_model_input_effect_abs_phase_empty_vs_P3.png",
        "explain_model_input_effect_by_subcarrier_empty_vs_P3.png",
        "pca_model_input_abs_phase_empty_vs_P3.png",
        "softmax_p3_score_empty_vs_P3_test.png",
        "explain_model_input_effect_notes.txt",
        "softmax_p3_score_empty_vs_P3_test_notes.txt",
    ]:
        print(" ", out_dir / p)


if __name__ == "__main__":
    main()
