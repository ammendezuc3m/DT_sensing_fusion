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
    return (20.0 * np.log10(np.abs(h) + EPS)).astype(np.float32)


def fixed_clip_scale(x, lo, hi):
    """
    Escalado físico fijo.

    No hacemos min-max por imagen porque eso elimina diferencias globales de potencia.
    """
    x = np.asarray(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    return y.astype(np.float32)


def clean_complex(h):
    h = np.asarray(h, dtype=np.complex64)
    real = np.nan_to_num(h.real, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    imag = np.nan_to_num(h.imag, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return (real + 1j * imag).astype(np.complex64)


def build_empty_baselines(session_dirs):
    """
    Baseline empty por fold:
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
        raise RuntimeError(f"Necesito al menos 3 sesiones empty. Encontradas: {len(empty_dirs)}")

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
        h = force_hssb_to_n_240_4(h_raw)
        A = h_to_amp_db(h)

        mu = np.median(A, axis=0).astype(np.float32)  # 240 × 4

        q25 = np.percentile(A, 25, axis=0)
        q75 = np.percentile(A, 75, axis=0)
        sigma = ((q75 - q25) / 1.349).astype(np.float32)
        sigma = np.maximum(sigma, 0.5).astype(np.float32)

        baselines[fold_id] = {
            "mu": mu,
            "sigma": sigma,
            "session_id": session_id,
        }

        print(f"fold {fold_id}: {session_id}, N={h.shape[0]}")

    return baselines, empty_index_by_session


def make_image_v4(h_win, empty_mu, empty_sigma):
    """
    h_win:
      5 × 240 × 4 complejo

    Devuelve:
      4 × 240 × 20

    Canales:
      0: delta dB respecto a empty, escala fija [-12,+12] dB
      1: atenuación positiva, escala fija [0,12] dB
      2: z-score respecto a empty, escala fija [-4,+4]
      3: variación temporal de amplitud, escala fija [0,6] dB

    Importante:
      NO hay min-max por imagen.
    """
    h_win = clean_complex(h_win)

    A = h_to_amp_db(h_win)  # 5 × 240 × 4

    delta = A - empty_mu[None, :, :]       # 5 × 240 × 4
    zdelta = delta / empty_sigma[None, :, :]
    attenuation = np.maximum(-delta, 0.0)

    # Imagen 240 × 20: 5 capturas × 4 símbolos = 20 columnas.
    delta_img = np.transpose(delta, (1, 0, 2)).reshape(240, 20)
    zdelta_img = np.transpose(zdelta, (1, 0, 2)).reshape(240, 20)
    att_img = np.transpose(attenuation, (1, 0, 2)).reshape(240, 20)

    # Variación temporal: 4 diferencias temporales × 4 símbolos = 16 columnas.
    dA = np.abs(np.diff(A, axis=0))
    dA_img = np.transpose(dA, (1, 0, 2)).reshape(240, 16)

    # Pad a 20 columnas.
    dA_img = np.pad(dA_img, ((0, 0), (0, 4)), mode="constant", constant_values=0.0)

    ch0_delta = fixed_clip_scale(delta_img, -12.0, 12.0)
    ch1_attenuation = fixed_clip_scale(att_img, 0.0, 12.0)
    ch2_zdelta = fixed_clip_scale(zdelta_img, -4.0, 4.0)
    ch3_temporal = fixed_clip_scale(dA_img, 0.0, 6.0)

    img = np.stack(
        [ch0_delta, ch1_attenuation, ch2_zdelta, ch3_temporal],
        axis=0,
    ).astype(np.float32)

    return img


def get_output_label(original_label, mode, labels, presence_labels):
    if mode == "binary":
        if original_label == "empty":
            return "empty"
        if original_label in presence_labels:
            return "presence"
        return None

    if mode == "multiclass":
        if original_label in labels:
            return original_label
        return None

    raise ValueError(f"Modo desconocido: {mode}")


def build_dataset(raw_dir, out_npz, out_meta, mode, labels, presence_labels, window_size, stride):
    raw_dir = Path(raw_dir)
    out_npz = Path(out_npz)
    out_meta = Path(out_meta)

    session_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])

    baselines, empty_index_by_session = build_empty_baselines(session_dirs)

    X_list = []
    rows = []

    print("\n=== Construyendo dataset fixed-scale v4 ===")
    print(f"Mode: {mode}")
    print(f"Labels multiclass: {labels}")
    print(f"Presence labels binary: {presence_labels}")
    print(f"Window size: {window_size}")
    print(f"Stride: {stride}")

    for sdir in tqdm(session_dirs, desc="Sesiones"):
        meta_path = sdir / "metadata.json"
        mat_path = sdir / "session_data.mat"

        if not meta_path.exists() or not mat_path.exists():
            continue

        meta = json.loads(meta_path.read_text())
        original_label = meta["label"]

        out_label = get_output_label(
            original_label=original_label,
            mode=mode,
            labels=labels,
            presence_labels=presence_labels,
        )

        if out_label is None:
            continue

        orientation = meta.get("orientation", "unknown")
        session_id = meta["session_id"]

        empty_idx = empty_index_by_session.get(sdir.name, None)
        fold_id = assign_fold(original_label, orientation, empty_idx)

        if fold_id not in baselines:
            raise RuntimeError(f"No hay baseline para fold {fold_id}")

        empty_mu = baselines[fold_id]["mu"]
        empty_sigma = baselines[fold_id]["sigma"]

        h_raw = read_matlab_dataset(mat_path, "hSSB")
        h = force_hssb_to_n_240_4(h_raw)

        n = h.shape[0]

        if n < window_size:
            continue

        n_windows = 0

        for start in range(0, n - window_size + 1, stride):
            end = start + window_size

            img = make_image_v4(h[start:end], empty_mu, empty_sigma)

            X_list.append(img)

            rows.append({
                "session_dir": sdir.name,
                "session_id": session_id,
                "original_label": original_label,
                "label": out_label,
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
            f"{session_id}: original={original_label}, out={out_label}, "
            f"orientation={orientation}, N={n}, windows={n_windows}, fold={fold_id}"
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
        image_shape=np.array(X.shape[1:], dtype=np.int32),
        feature_type=np.array(["fixedscale_delta_attenuation_z_temporal_v4"]),
        mode=np.array([mode]),
    )

    meta_df.to_csv(out_meta, index=False)

    print("\n=== Dataset fixed-scale v4 construido ===")
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
    parser.add_argument("--out", required=True)
    parser.add_argument("--meta-out", required=True)
    parser.add_argument("--mode", choices=["binary", "multiclass"], required=True)
    parser.add_argument("--labels", nargs="+", default=["empty", "P1", "P4"])
    parser.add_argument("--presence-labels", nargs="+", default=["P1", "P2", "P3", "P4"])
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)

    args = parser.parse_args()

    build_dataset(
        raw_dir=args.raw_dir,
        out_npz=args.out,
        out_meta=args.meta_out,
        mode=args.mode,
        labels=args.labels,
        presence_labels=args.presence_labels,
        window_size=args.window,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()

