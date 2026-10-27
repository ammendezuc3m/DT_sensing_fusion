#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

EPS = 1e-9


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
            raise ValueError(f"Compound dtype inesperado en {name}: {list(fields)}")

    return np.asarray(data)


def clean_complex(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex64)
    real = np.nan_to_num(x.real, nan=0.0, posinf=0.0, neginf=0.0)
    imag = np.nan_to_num(x.imag, nan=0.0, posinf=0.0, neginf=0.0)
    return (real + 1j * imag).astype(np.complex64)


def force_data_ssb_to_n_360_6(data: np.ndarray) -> np.ndarray:
    """
    Devuelve dataSSB como [N, 360, 6], igual que el script binario original.
    """
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


def load_valid_mask(mat_path: Path, n: int) -> np.ndarray:
    try:
        valid = np.asarray(read_mat_variable(mat_path, "validMask")).reshape(-1).astype(bool)
        if len(valid) >= n:
            return valid[:n]
    except Exception:
        pass

    return np.ones(n, dtype=bool)


def power_db(x: np.ndarray, axis=None) -> np.ndarray:
    p = np.abs(x) ** 2
    if axis is not None:
        p = np.mean(p, axis=axis)
    return 10.0 * np.log10(p + EPS)


def robust_limits(values: np.ndarray, p_low=2, p_high=98, symmetric=False):
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return -1.0, 1.0

    if symmetric:
        lim = float(np.percentile(np.abs(values), p_high))
        lim = max(lim, 1e-3)
        return -lim, lim

    lo = float(np.percentile(values, p_low))
    hi = float(np.percentile(values, p_high))

    if hi <= lo:
        hi = lo + 1.0

    return lo, hi


def load_label_power(dataset_dir: Path, label: str):
    label_dir = dataset_dir / label
    files = sorted(label_dir.rglob("*.mat"))

    if not files:
        raise FileNotFoundError(f"No hay .mat para {label} en {label_dir}")

    per_sc_all = []
    per_grid_all = []
    total_all = []

    print(f"\nLoading {label}: {len(files)} files")

    for mat_path in files:
        print(f"  {mat_path}")

        data = force_data_ssb_to_n_360_6(read_mat_variable(mat_path, "dataSSB"))
        valid = load_valid_mask(mat_path, data.shape[0])

        data = data[valid]

        # rxGridSSB validado:
        # MATLAB: dataSSB(61:300, 2:5, :)
        # Python: data[:, 60:300, 1:5]
        rx = data[:, 60:300, 1:5]

        per_sc = power_db(rx, axis=2)       # [N, 240]
        per_grid = power_db(rx)             # [N, 240, 4]
        total = power_db(rx, axis=(1, 2))   # [N]

        per_sc_all.append(per_sc.astype(np.float32))
        per_grid_all.append(per_grid.astype(np.float32))
        total_all.append(total.astype(np.float32))

        print(f"    valid samples: {rx.shape[0]}")

    return (
        np.concatenate(per_sc_all, axis=0),
        np.concatenate(per_grid_all, axis=0),
        np.concatenate(total_all, axis=0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/dataset_datassb/datassb_side_v1_6labels")
    parser.add_argument("--labels", nargs="+", default=["empty", "P5", "P3"])
    parser.add_argument("--out-dir", default="results/multiclass_empty_P5_P3_rx/figures")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_sc = {}
    per_grid = {}
    total_power = {}

    for label in args.labels:
        per_sc[label], per_grid[label], total_power[label] = load_label_power(dataset_dir, label)

    x = np.arange(1, 241)

    # ---------------------------------------------------------------------
    # Figura principal: potencia por subportadora con IQR
    # ---------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for label in args.labels:
        vals = per_sc[label]
        q25, q50, q75 = np.percentile(vals, [25, 50, 75], axis=0)

        line = ax.plot(x, q50, linewidth=1.8, label=f"{label} mediana")[0]
        ax.fill_between(
            x,
            q25,
            q75,
            alpha=0.15,
            color=line.get_color(),
            label=f"{label} IQR",
        )

    ax.set_xlabel("Subportadora SSB")
    ax.set_ylabel("Potencia media por símbolo [dB]")
    ax.set_title("Potencia por subportadora: empty vs P5 vs P3")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()

    fig.savefig(out_dir / "power_by_subcarrier_empty_P5_P3.png", dpi=180)
    plt.close(fig)

    # ---------------------------------------------------------------------
    # Deltas contra empty
    # ---------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 4.5))

    med_empty = np.median(per_sc["empty"], axis=0)

    for label in args.labels:
        if label == "empty":
            continue

        med_label = np.median(per_sc[label], axis=0)
        delta = med_label - med_empty

        ax.plot(x, delta, linewidth=1.8, label=f"{label} - empty")

    ax.axhline(0, linewidth=1.0)
    ax.set_xlabel("Subportadora SSB")
    ax.set_ylabel("Delta potencia [dB]")
    ax.set_title("Diferencia de potencia por subportadora frente a empty")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(out_dir / "delta_power_by_subcarrier_P5_P3_minus_empty.png", dpi=180)
    plt.close(fig)

    # ---------------------------------------------------------------------
    # Heatmaps delta frente a empty
    # ---------------------------------------------------------------------

    for label in args.labels:
        if label == "empty":
            continue

        delta_grid = np.median(per_grid[label], axis=0) - np.median(per_grid["empty"], axis=0)
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

        ax.set_xlabel("Símbolo OFDM SSB")
        ax.set_ylabel("Subportadora SSB")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_title(f"Delta potencia rxGridSSB: {label} - empty")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Delta [dB]")

        fig.tight_layout()
        fig.savefig(out_dir / f"delta_power_heatmap_rxGridSSB_{label}_minus_empty.png", dpi=180)
        plt.close(fig)

    # ---------------------------------------------------------------------
    # Boxplot potencia total
    # ---------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [total_power[label] for label in args.labels]

    try:
        ax.boxplot(data, tick_labels=args.labels, showmeans=True)
    except TypeError:
        ax.boxplot(data, labels=args.labels, showmeans=True)

    rng = np.random.default_rng(0)
    for i, label in enumerate(args.labels, start=1):
        vals = total_power[label]
        if len(vals) > 2500:
            idx = rng.choice(len(vals), size=2500, replace=False)
            vals_plot = vals[idx]
        else:
            vals_plot = vals

        ax.scatter(
            i + rng.normal(0, 0.025, size=len(vals_plot)),
            vals_plot,
            s=5,
            alpha=0.18,
        )

    ax.set_ylabel("Potencia total media [dB]")
    ax.set_title("Distribución de potencia total por SSB")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    fig.savefig(out_dir / "total_power_distribution_empty_P5_P3.png", dpi=180)
    plt.close(fig)

    # ---------------------------------------------------------------------
    # PCA opcional
    # ---------------------------------------------------------------------

    try:
        from sklearn.decomposition import PCA

        X = np.concatenate([per_sc[label] for label in args.labels], axis=0)
        y = np.concatenate([
            np.full(len(per_sc[label]), label)
            for label in args.labels
        ])

        Xz = (X - X.mean(axis=0, keepdims=True)) / np.maximum(X.std(axis=0, keepdims=True), 1e-6)
        Z = PCA(n_components=2, random_state=0).fit_transform(Xz)

        fig, ax = plt.subplots(figsize=(6.5, 5.5))

        for label in args.labels:
            mask = y == label
            ax.scatter(Z[mask, 0], Z[mask, 1], s=6, alpha=0.35, label=label)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA de potencia por subportadora: empty / P5 / P3")
        ax.grid(True, alpha=0.2)
        ax.legend()
        fig.tight_layout()

        fig.savefig(out_dir / "pca_power_features_empty_P5_P3.png", dpi=180)
        plt.close(fig)

    except Exception as e:
        print(f"PCA omitida: {e}")

    notes = []

    for label in args.labels:
        notes.append(
            f"{label}: n={len(total_power[label])}, "
            f"total_power_median={np.median(total_power[label]):.3f} dB, "
            f"total_power_mean={np.mean(total_power[label]):.3f} dB"
        )

    (out_dir / "analysis_notes.txt").write_text("\n".join(notes) + "\n")

    print("\nSaved figures in:")
    print(out_dir)

    for p in sorted(out_dir.glob("*.png")):
        print(" ", p)


if __name__ == "__main__":
    main()
