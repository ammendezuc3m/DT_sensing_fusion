#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from build_sensing_dataset import (
    read_matlab_dataset,
    force_hssb_to_n_240_4,
    assign_fold,
)

EPS = 1e-6


def h_to_amp_db(h):
    return 20.0 * np.log10(np.abs(h) + EPS).astype(np.float32)


def robust_minmax(x, p_low=1, p_high=99):
    x = np.asarray(x, dtype=np.float32)
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)

    if hi <= lo + 1e-6:
        return np.zeros_like(x, dtype=np.float32)

    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    return y.astype(np.float32)


def build_empty_baselines(session_dirs):
    """
    Crea un baseline empty por fold:
      fold 1 -> primer empty
      fold 2 -> segundo empty
      fold 3 -> tercer empty
    """
    empty_dirs = []

    for sdir in session_dirs:
        meta_path = sdir / "metadata.json"
        mat_path = sdir / "session_data.mat"

        if not meta_path.exists() or not mat_path.exists():
            continue

        meta = json.loads(meta_path.read_text())

        if meta.get("label") == "empty":
            empty_dirs.append(sdir)

    empty_dirs = sorted(empty_dirs, key=lambda p: p.name)

    if len(empty_dirs) < 3:
        raise RuntimeError(f"Necesito 3 sesiones empty. Encontradas: {len(empty_dirs)}")

    empty_index_by_session = {
        sdir.name: i for i, sdir in enumerate(empty_dirs)
    }

    baselines = {}

    print("\n=== Baselines empty ===")

    for sdir in empty_dirs:
        meta = json.loads((sdir / "metadata.json").read_text())
        session_id = meta["session_id"]

        empty_idx = empty_index_by_session[sdir.name]
        fold_id = (empty_idx % 3) + 1

        h_raw = read_matlab_dataset(sdir / "session_data.mat", "hSSB")
        h = force_hssb_to_n_240_4(h_raw)  # N × 240 × 4

        A = h_to_amp_db(h)

        # Baseline robusto por subportadora y símbolo.
        mu = np.median(A, axis=0).astype(np.float32)  # 240 × 4

        baselines[fold_id] = {
            "mu": mu,
            "session_id": session_id,
        }

        print(f"fold {fold_id}: {session_id}, N={h.shape[0]}")

    return baselines, empty_index_by_session


def make_csi_image(h_win, empty_mu):
    """
    h_win:
      5 × 240 × 4 complejo

    empty_mu:
      240 × 4 amplitud dB del empty del fold

    Devuelve imagen:
      C × H × W = 3 × 240 × 20

    Canales:
      0: delta amplitud normalizada
      1: atenuación normalizada
      2: periodograma 2D normalizado
    """
    h_win = np.asarray(h_win, dtype=np.complex64)

    # Limpiar complejos.
    real = np.nan_to_num(h_win.real, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    imag = np.nan_to_num(h_win.imag, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    h_win = (real + 1j * imag).astype(np.complex64)

    A = h_to_amp_db(h_win)  # 5 × 240 × 4

    # Delta respecto a empty.
    delta = A - empty_mu[None, :, :]  # 5 × 240 × 4

    # Reordenamos a imagen subportadora × tiempo-símbolo.
    # Tenemos 5 capturas y 4 símbolos SSB -> ancho 20.
    delta_img = np.transpose(delta, (1, 0, 2)).reshape(240, 20).astype(np.float32)

    # Canal 0: delta dB escalado robusto.
    ch_delta = robust_minmax(delta_img, 1, 99)

    # Canal 1: atenuación positiva. Si delta < 0, hay caída respecto a empty.
    attenuation = np.maximum(-delta_img, 0.0)
    ch_att = robust_minmax(attenuation, 1, 99)

    # Canal 2: periodograma 2D.
    x = delta_img.copy()
    x = x - np.mean(x, axis=0, keepdims=True)
    x = x - np.mean(x, axis=1, keepdims=True)

    P = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2

    # Eliminar DC central y líneas centrales.
    r0 = P.shape[0] // 2
    c0 = P.shape[1] // 2
    P[max(0, r0-1):min(P.shape[0], r0+2), :] = 0
    P[:, max(0, c0-1):min(P.shape[1], c0+2)] = 0

    P = np.log1p(P).astype(np.float32)
    ch_periodogram = robust_minmax(P, 1, 99)

    img = np.stack([ch_delta, ch_att, ch_periodogram], axis=0).astype(np.float32)

    return img


def build_dataset(raw_dir, out_npz, out_meta, labels, window_size, stride):
    raw_dir = Path(raw_dir)
    out_npz = Path(out_npz)
    out_meta = Path(out_meta)

    session_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])

    baselines, empty_index_by_session = build_empty_baselines(session_dirs)

    X_list = []
    rows = []

    print("\n=== Construyendo CSI image dataset v3 ===")
    print(f"Labels usadas: {labels}")
    print(f"Window size: {window_size}")
    print(f"Stride: {stride}")

    for sdir in tqdm(session_dirs, desc="Sesiones"):
        meta_path = sdir / "metadata.json"
        mat_path = sdir / "session_data.mat"

        if not meta_path.exists() or not mat_path.exists():
            continue

        meta = json.loads(meta_path.read_text())
        label = meta["label"]

        if label not in labels:
            continue

        orientation = meta.get("orientation", "unknown")
        session_id = meta["session_id"]

        empty_idx = empty_index_by_session.get(sdir.name, None)
        fold_id = assign_fold(label, orientation, empty_idx)

        if fold_id not in baselines:
            raise RuntimeError(f"No baseline para fold {fold_id}")

        empty_mu = baselines[fold_id]["mu"]

        h_raw = read_matlab_dataset(mat_path, "hSSB")
        h = force_hssb_to_n_240_4(h_raw)

        n = h.shape[0]

        if n < window_size:
            continue

        n_windows = 0

        for start in range(0, n - window_size + 1, stride):
            end = start + window_size

            img = make_csi_image(h[start:end], empty_mu)

            X_list.append(img)

            rows.append({
                "session_dir": sdir.name,
                "session_id": session_id,
                "label": label,
                "orientation": orientation,
                "fold_id": fold_id,
                "window_start": start,
                "window_end": end - 1,
                "window_size": window_size,
                "stride": stride,
                "baseline_empty_session_id": baselines[fold_id]["session_id"],
            })

            n_windows += 1

        print(
            f"{session_id}: label={label}, orientation={orientation}, "
            f"N={n}, windows={n_windows}, fold={fold_id}"
        )

    if not X_list:
        raise RuntimeError("No se ha generado ninguna imagen.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    meta_df = pd.DataFrame(rows)

    y = meta_df["label"].to_numpy()
    groups = meta_df["session_id"].to_numpy()
    folds = meta_df["fold_id"].to_numpy(dtype=int)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_npz,
        X=X,
        y=y,
        groups=groups,
        folds=folds,
        labels=np.array(labels),
        image_shape=np.array(X.shape[1:], dtype=np.int32),
        feature_type=np.array(["csi_delta_attenuation_periodogram_v3"]),
    )

    meta_df.to_csv(out_meta, index=False)

    print("\n=== Dataset CSI image v3 construido ===")
    print(f"X shape: {X.shape}")
    print("Labels:")
    print(meta_df["label"].value_counts())
    print("Folds:")
    print(meta_df["fold_id"].value_counts().sort_index())
    print(f"Guardado: {out_npz}")
    print(f"Metadata: {out_meta}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="data/processed/ssb_csi_images_v3_empty_P1_P4.npz")
    parser.add_argument("--meta-out", default="data/processed/ssb_csi_images_metadata_v3_empty_P1_P4.csv")
    parser.add_argument("--labels", nargs="+", default=["empty", "P1", "P4"])
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)

    args = parser.parse_args()

    build_dataset(
        raw_dir=args.raw_dir,
        out_npz=args.out,
        out_meta=args.meta_out,
        labels=args.labels,
        window_size=args.window,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
