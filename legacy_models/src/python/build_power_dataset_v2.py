#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from build_sensing_dataset import (
    read_matlab_dataset,
    read_optional_1d,
    force_hssb_to_n_240_4,
    assign_fold,
)


EPS = 1e-6


def amp_db_from_h(h):
    """
    Convierte canal complejo H a potencia/amplitud en dB.
    Entrada:
        h: N × 240 × 4 complejo
    Salida:
        A: N × 240 × 4 real
    """
    return 20.0 * np.log10(np.abs(h) + EPS).astype(np.float32)


def robust_empty_stats(A_empty):
    """
    Calcula baseline robusto del escenario vacío.

    A_empty:
        N × 240 × 4

    Devuelve:
        mu: mediana empty por subportadora/símbolo
        sigma: escala robusta por subportadora/símbolo
    """
    mu = np.median(A_empty, axis=0).astype(np.float32)

    q25 = np.percentile(A_empty, 25, axis=0)
    q75 = np.percentile(A_empty, 75, axis=0)
    sigma = ((q75 - q25) / 1.349).astype(np.float32)

    # Evitamos divisiones explosivas si una subportadora tiene varianza casi cero.
    sigma = np.maximum(sigma, 0.5).astype(np.float32)

    return mu, sigma


def pool_subcarriers(M, n_blocks=24):
    """
    Agrupa las 240 subportadoras en bloques.

    M:
        240 × 4

    Con n_blocks=24, cada bloque contiene 10 subportadoras.
    Para cada bloque y símbolo sacamos:
        media, std, min, max

    Salida:
        vector de 24 × 4 × 4 = 384 features
    """
    if M.shape != (240, 4):
        raise ValueError(f"Shape esperado 240×4, recibido {M.shape}")

    block_size = 240 // n_blocks
    B = M.reshape(n_blocks, block_size, 4)

    feats = [
        B.mean(axis=1).reshape(-1),
        B.std(axis=1).reshape(-1),
        B.min(axis=1).reshape(-1),
        B.max(axis=1).reshape(-1),
    ]

    return np.concatenate(feats).astype(np.float32)


def map_stats(M, prefix_unused=None):
    """
    Estadísticos globales de un mapa 240×4.
    """
    M = np.asarray(M, dtype=np.float32)

    feats = []

    # Globales.
    feats.extend([
        np.mean(M),
        np.std(M),
        np.min(M),
        np.max(M),
        np.percentile(M, 5),
        np.percentile(M, 25),
        np.percentile(M, 50),
        np.percentile(M, 75),
        np.percentile(M, 95),
    ])

    # Por símbolo OFDM.
    for s in range(4):
        x = M[:, s]
        feats.extend([
            np.mean(x),
            np.std(x),
            np.min(x),
            np.max(x),
            np.percentile(x, 10),
            np.percentile(x, 90),
        ])

    return np.array(feats, dtype=np.float32)


def extract_power_features(h_win, empty_mu, empty_sigma, freq_win, noise_win, timing_win):
    """
    Features orientadas a potencia y atenuación.

    h_win:
        5 × 240 × 4 complejo

    empty_mu / empty_sigma:
        baseline vacío del fold correspondiente.

    La feature central es:
        delta = A_persona - A_empty

    Si delta es negativo en ciertas subportadoras, significa atenuación respecto a vacío.
    """
    h_win = np.asarray(h_win, dtype=np.complex64)

    h_real = np.nan_to_num(h_win.real, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    h_imag = np.nan_to_num(h_win.imag, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    h_win = (h_real + 1j * h_imag).astype(np.complex64)

    A = amp_db_from_h(h_win)  # 5 × 240 × 4

    # Usamos mediana temporal para ser robustos dentro de la ventana.
    A_med = np.median(A, axis=0).astype(np.float32)
    A_mean = np.mean(A, axis=0).astype(np.float32)
    A_std = np.std(A, axis=0).astype(np.float32)

    # Variación temporal de potencia dentro de la ventana.
    A_diff = np.abs(np.diff(A, axis=0))
    A_diff_mean = np.mean(A_diff, axis=0).astype(np.float32)

    # Diferencia respecto a empty.
    delta = (A_med - empty_mu).astype(np.float32)

    # Z-score respecto a empty.
    zdelta = (delta / empty_sigma).astype(np.float32)

    # Atenuación positiva: si delta < 0, attenuation > 0.
    attenuation = np.maximum(-delta, 0.0).astype(np.float32)

    # Features full-resolution.
    full_feats = [
        delta.reshape(-1),
        zdelta.reshape(-1),
        attenuation.reshape(-1),
        A_std.reshape(-1),
        A_diff_mean.reshape(-1),
    ]

    # Features agrupadas por bloques de subportadoras.
    pooled_feats = [
        pool_subcarriers(delta),
        pool_subcarriers(zdelta),
        pool_subcarriers(attenuation),
        pool_subcarriers(A_std),
        pool_subcarriers(A_diff_mean),
    ]

    # Estadísticos compactos.
    stats_feats = [
        map_stats(delta),
        map_stats(zdelta),
        map_stats(attenuation),
        map_stats(A_std),
        map_stats(A_diff_mean),
    ]

    # Calidad.
    quality = np.array([
        np.mean(freq_win),
        np.std(freq_win),
        np.mean(noise_win),
        np.std(noise_win),
        np.mean(timing_win),
        np.std(timing_win),
        np.mean(A_mean),
        np.std(A_mean),
        np.mean(delta),
        np.std(delta),
        np.mean(attenuation),
        np.max(attenuation),
    ], dtype=np.float32)

    feat = np.concatenate(full_feats + pooled_feats + stats_feats + [quality])
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return feat


def build_empty_baselines(session_dirs):
    """
    Construye un baseline empty por fold.

    fold 1 usa el primer empty.
    fold 2 usa el segundo empty.
    fold 3 usa el tercer empty.

    Esto simula una calibración empty por ronda/estado del entorno.
    """
    empty_sessions = []

    for sdir in session_dirs:
        meta_path = sdir / "metadata.json"
        mat_path = sdir / "session_data.mat"

        if not meta_path.exists() or not mat_path.exists():
            continue

        meta = json.loads(meta_path.read_text())

        if meta.get("label") == "empty":
            empty_sessions.append(sdir)

    empty_sessions = sorted(empty_sessions, key=lambda p: p.name)

    if len(empty_sessions) < 3:
        raise RuntimeError(f"Necesito al menos 3 sesiones empty, encontradas {len(empty_sessions)}")

    empty_index_by_session = {
        sdir.name: i for i, sdir in enumerate(empty_sessions)
    }

    baselines = {}

    print("\n=== Construyendo baselines empty ===")

    for sdir in empty_sessions:
        meta = json.loads((sdir / "metadata.json").read_text())
        session_id = meta["session_id"]

        empty_idx = empty_index_by_session[sdir.name]
        fold_id = (empty_idx % 3) + 1

        h_raw = read_matlab_dataset(sdir / "session_data.mat", "hSSB")
        h = force_hssb_to_n_240_4(h_raw)

        A_empty = amp_db_from_h(h)
        mu, sigma = robust_empty_stats(A_empty)

        baselines[fold_id] = {
            "mu": mu,
            "sigma": sigma,
            "session_id": session_id,
        }

        print(f"fold {fold_id}: baseline={session_id}, N={h.shape[0]}")

    return baselines, empty_index_by_session


def build_dataset(raw_dir, out_npz, out_meta, window_size, stride):
    raw_dir = Path(raw_dir)
    out_npz = Path(out_npz)
    out_meta = Path(out_meta)

    session_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])

    if not session_dirs:
        raise RuntimeError(f"No hay sesiones en {raw_dir}")

    baselines, empty_index_by_session = build_empty_baselines(session_dirs)

    X_list = []
    rows = []

    print("\n=== Construyendo dataset power v2 ===")
    print(f"Sesiones: {len(session_dirs)}")
    print(f"Window size: {window_size}")
    print(f"Stride: {stride}")

    for sdir in tqdm(session_dirs, desc="Procesando sesiones"):
        meta_path = sdir / "metadata.json"
        mat_path = sdir / "session_data.mat"

        if not meta_path.exists() or not mat_path.exists():
            print(f"Saltando {sdir.name}: falta metadata.json o session_data.mat")
            continue

        meta = json.loads(meta_path.read_text())

        label = meta["label"]
        orientation = meta.get("orientation", "unknown")
        session_id = meta["session_id"]

        empty_idx = empty_index_by_session.get(sdir.name, None)
        fold_id = assign_fold(label, orientation, empty_idx)

        if fold_id not in baselines:
            raise RuntimeError(f"No hay baseline empty para fold {fold_id}")

        empty_mu = baselines[fold_id]["mu"]
        empty_sigma = baselines[fold_id]["sigma"]

        h_raw = read_matlab_dataset(mat_path, "hSSB")
        h = force_hssb_to_n_240_4(h_raw)

        n = h.shape[0]

        freq = read_optional_1d(mat_path, "freqLog", n)
        noise = read_optional_1d(mat_path, "noiseLog", n)
        timing = read_optional_1d(mat_path, "timingLog", n)

        if n < window_size:
            print(f"Saltando {session_id}: solo {n} capturas")
            continue

        n_windows = 0

        for start in range(0, n - window_size + 1, stride):
            end = start + window_size

            feat = extract_power_features(
                h_win=h[start:end, :, :],
                empty_mu=empty_mu,
                empty_sigma=empty_sigma,
                freq_win=freq[start:end],
                noise_win=noise[start:end],
                timing_win=timing[start:end],
            )

            X_list.append(feat)

            rows.append({
                "session_dir": sdir.name,
                "session_id": session_id,
                "label": label,
                "orientation": orientation,
                "movement_state": meta.get("movement_state", "static"),
                "person_id": meta.get("person_id", "unknown"),
                "fold_id": fold_id,
                "baseline_empty_session_id": baselines[fold_id]["session_id"],
                "window_start": start,
                "window_end": end - 1,
                "window_size": window_size,
                "stride": stride,
                "n_captures_session": n,
                "accepted_rate_hz": meta.get("accepted_rate_hz", np.nan),
            })

            n_windows += 1

        print(
            f"{session_id}: label={label}, orientation={orientation}, "
            f"N={n}, windows={n_windows}, fold={fold_id}"
        )

    X = np.vstack(X_list).astype(np.float32)
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
        window_size=np.array([window_size], dtype=np.int32),
        stride=np.array([stride], dtype=np.int32),
        feature_type=np.array(["power_delta_empty_v2"]),
    )

    meta_df.to_csv(out_meta, index=False)

    print("\n=== Dataset power v2 construido ===")
    print(f"X shape: {X.shape}")
    print("Labels:")
    print(meta_df["label"].value_counts())
    print("Folds:")
    print(meta_df["fold_id"].value_counts().sort_index())
    print(f"Guardado NPZ: {out_npz}")
    print(f"Guardado CSV: {out_meta}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="data/processed/ssb_power_windows_v002.npz")
    parser.add_argument("--meta-out", default="data/processed/ssb_power_windows_metadata_v002.csv")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)

    args = parser.parse_args()

    build_dataset(
        raw_dir=args.raw_dir,
        out_npz=args.out,
        out_meta=args.meta_out,
        window_size=args.window,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
