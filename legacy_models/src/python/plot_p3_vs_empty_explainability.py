#!/usr/bin/env python3
from pathlib import Path
import argparse
import h5py
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

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
        data = f[name][()]

    if getattr(data, "dtype", None) is not None and data.dtype.fields is not None:
        fields = data.dtype.fields.keys()
        if "real" in fields and "imag" in fields:
            data = data["real"] + 1j * data["imag"]

    return np.asarray(data)


def clean_complex(x):
    x = np.asarray(x, dtype=np.complex64)
    return np.nan_to_num(x.real) + 1j * np.nan_to_num(x.imag)


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


def power_db(x, axis=None):
    p = np.abs(x) ** 2
    if axis is not None:
        p = np.mean(p, axis=axis)
    return 10 * np.log10(p + EPS)


def load_label(dataset_dir: Path, label: str):
    files = sorted((dataset_dir / label).rglob("*.mat"))

    if not files:
        raise FileNotFoundError(f"No hay .mat para {label}")

    per_sc_all = []
    per_grid_all = []
    total_all = []

    print(f"\nLoading {label}: {len(files)} files")

    for p in files:
        print(f"  {p}")
        data = force_data_ssb_to_n_360_6(read_mat_variable(p, "dataSSB"))
        valid = load_valid_mask(p, data.shape[0])
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

        print(f"    valid: {rx.shape[0]}")

    return (
        np.concatenate(per_sc_all, axis=0),
        np.concatenate(per_grid_all, axis=0),
        np.concatenate(total_all, axis=0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/dataset_datassb/datassb_side_v1_6labels")
    parser.add_argument("--out-dir", default="results/multiclass_empty_P5_P3_rx/figures")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    empty_sc, empty_grid, empty_total = load_label(dataset_dir, "empty")
    p3_sc, p3_grid, p3_total = load_label(dataset_dir, "P3")

    x = np.arange(1, 241)

    empty_q25, empty_med, empty_q75 = np.percentile(empty_sc, [25, 50, 75], axis=0)
    p3_q25, p3_med, p3_q75 = np.percentile(p3_sc, [25, 50, 75], axis=0)

    delta = p3_med - empty_med

    empty_iqr = empty_q75 - empty_q25
    p3_iqr = p3_q75 - p3_q25
    pooled_iqr = np.maximum(0.5 * (empty_iqr + p3_iqr), 1e-3)
    robust_effect = delta / pooled_iqr

    # ------------------------------------------------------------------
    # Figura principal: P3 vs empty explicable
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    line_empty = ax.plot(x, empty_med, linewidth=1.8, label="empty mediana")[0]
    ax.fill_between(x, empty_q25, empty_q75, alpha=0.18, color=line_empty.get_color(), label="empty IQR")

    line_p3 = ax.plot(x, p3_med, linewidth=1.8, label="P3 mediana")[0]
    ax.fill_between(x, p3_q25, p3_q75, alpha=0.18, color=line_p3.get_color(), label="P3 IQR")

    ax.set_title("P3 vs empty: potencia por subportadora")
    ax.set_ylabel("Potencia [dB]")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)

    ax = axes[1]
    ax.plot(x, delta, linewidth=1.8)
    ax.axhline(0, linewidth=1.0)
    ax.set_title("Diferencia mediana: P3 - empty")
    ax.set_ylabel("Delta [dB]")
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    ax.plot(x, robust_effect, linewidth=1.8)
    ax.axhline(0, linewidth=1.0)
    ax.axhline(0.5, linestyle="--", linewidth=1.0)
    ax.axhline(-0.5, linestyle="--", linewidth=1.0)
    ax.set_title("Tamaño de efecto robusto: (P3 - empty) / IQR combinado")
    ax.set_xlabel("Subportadora SSB")
    ax.set_ylabel("Efecto robusto")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "explain_p3_vs_empty_profile_delta_effect.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Heatmap P3-empty por subportadora y símbolo OFDM
    # ------------------------------------------------------------------

    delta_grid = np.median(p3_grid, axis=0) - np.median(empty_grid, axis=0)

    lim = np.percentile(np.abs(delta_grid[np.isfinite(delta_grid)]), 98)
    lim = max(float(lim), 0.5)

    fig, ax = plt.subplots(figsize=(6.5, 8))
    im = ax.imshow(
        delta_grid,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=-lim,
        vmax=lim,
        extent=[0.5, 4.5, 0.5, 240.5],
    )

    ax.set_title("Heatmap de diferencia: P3 - empty")
    ax.set_xlabel("Símbolo OFDM dentro del SSB")
    ax.set_ylabel("Subportadora SSB")
    ax.set_xticks([1, 2, 3, 4])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Delta potencia [dB]")

    fig.tight_layout()
    fig.savefig(out_dir / "explain_p3_vs_empty_heatmap_240x4.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Histograma / boxplot de potencia total
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 4.5))

    try:
        ax.boxplot([empty_total, p3_total], tick_labels=["empty", "P3"], showmeans=True)
    except TypeError:
        ax.boxplot([empty_total, p3_total], labels=["empty", "P3"], showmeans=True)

    ax.set_title("Potencia total: empty vs P3")
    ax.set_ylabel("Potencia total media [dB]")
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "explain_p3_vs_empty_total_power_boxplot.png", dpi=200)
    plt.close(fig)

    notes = []
    notes.append(f"empty n={len(empty_total)} total_power_median={np.median(empty_total):.4f} dB")
    notes.append(f"P3    n={len(p3_total)} total_power_median={np.median(p3_total):.4f} dB")
    notes.append(f"P3-empty total_power_median_delta={np.median(p3_total)-np.median(empty_total):.4f} dB")
    notes.append("")
    notes.append(f"max_abs_delta_subcarrier={np.max(np.abs(delta)):.4f} dB")
    notes.append(f"median_abs_delta_subcarrier={np.median(np.abs(delta)):.4f} dB")
    notes.append(f"max_abs_robust_effect={np.max(np.abs(robust_effect)):.4f}")
    notes.append(f"median_abs_robust_effect={np.median(np.abs(robust_effect)):.4f}")

    (out_dir / "explain_p3_vs_empty_notes.txt").write_text("\n".join(notes) + "\n")

    print("\nSaved:")
    for p in [
        out_dir / "explain_p3_vs_empty_profile_delta_effect.png",
        out_dir / "explain_p3_vs_empty_heatmap_240x4.png",
        out_dir / "explain_p3_vs_empty_total_power_boxplot.png",
        out_dir / "explain_p3_vs_empty_notes.txt",
    ]:
        print(" ", p)


if __name__ == "__main__":
    main()
