#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def read_matlab_dataset(mat_path: Path, name: str):
    """
    Lee una variable de un .mat v7.3 generado por MATLAB.

    MATLAB guarda arrays complejos en HDF5 normalmente como compound dtype
    con campos 'real' e 'imag'. Esta función reconstruye el array complejo.
    """
    with h5py.File(mat_path, "r") as f:
        if name not in f:
            raise KeyError(f"No existe la variable '{name}' en {mat_path}")

        data = f[name][()]

    if data.dtype.fields is not None:
        fields = data.dtype.fields.keys()
        if "real" in fields and "imag" in fields:
            data = data["real"] + 1j * data["imag"]
        else:
            raise ValueError(f"Array compound inesperado en {name}: {fields}")

    return np.array(data)


def read_optional_1d(mat_path: Path, name: str, expected_len: int):
    """
    Lee logs 1D como freqLog, noiseLog, timingLog.
    Si no existe, devuelve ceros.
    """
    try:
        arr = read_matlab_dataset(mat_path, name)
        arr = np.asarray(arr).squeeze()
        arr = arr.reshape(-1)
        arr = arr.astype(np.float32)

        if len(arr) != expected_len:
            if len(arr) > expected_len:
                arr = arr[:expected_len]
            else:
                pad = np.full(expected_len - len(arr), np.nan, dtype=np.float32)
                arr = np.concatenate([arr, pad])

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    except Exception:
        return np.zeros(expected_len, dtype=np.float32)


def force_hssb_to_n_240_4(h):
    """
    Convierte hSSB a shape estándar:

        N × 240 × 4

    MATLAB suele guardar 240×4×N, pero al leer con h5py puede aparecer
    transpuesto como N×4×240 u otra permutación. Detectamos ejes por tamaño.
    """
    h = np.asarray(h)

    if h.ndim != 3:
        raise ValueError(f"hSSB debe tener 3 dimensiones, pero tiene shape {h.shape}")

    shape = h.shape

    if 240 not in shape:
        raise ValueError(f"No encuentro dimensión 240 en hSSB shape={shape}")

    if 4 not in shape:
        raise ValueError(f"No encuentro dimensión 4 en hSSB shape={shape}")

    ax_sc = shape.index(240)
    ax_sym = shape.index(4)

    axes = [0, 1, 2]
    ax_n_candidates = [a for a in axes if a not in [ax_sc, ax_sym]]
    if len(ax_n_candidates) != 1:
        raise ValueError(f"No puedo determinar eje temporal N en shape={shape}")

    ax_n = ax_n_candidates[0]

    h = np.transpose(h, (ax_n, ax_sc, ax_sym))

    return h.astype(np.complex64)


def assign_fold(label: str, orientation: str, empty_index: int | None):
    """
    Asignamos fold_id para validación por rondas.

    Fold 1: facing_rx + primer empty
    Fold 2: facing_dot + segundo empty
    Fold 3: sideways + tercer empty

    Esto evita un split aleatorio falso con ventanas consecutivas muy parecidas.
    """
    if label == "empty":
        if empty_index is None:
            return 0
        return (empty_index % 3) + 1

    orientation = str(orientation)

    if orientation == "facing_rx":
        return 1
    if orientation == "facing_dot":
        return 2
    if orientation == "sideways":
        return 3

    return 0


def extract_window_features(h_win, freq_win, noise_win, timing_win):
    """
    Extrae features de una ventana:

        h_win shape = 5 × 240 × 4 complejo

    Features:
      - amplitud en dB normalizada por captura
      - media temporal de amplitud
      - desviación temporal de amplitud
      - diferencia temporal de amplitud
      - rotación de fase temporal
      - fase diferencial frecuencial
      - calidad: freq/noise/timing/global_amp
    """
    eps = 1e-6

    h_win = np.asarray(h_win, dtype=np.complex64)
    h_real = np.nan_to_num(
        h_win.real,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    h_imag = np.nan_to_num(
        h_win.imag,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    h_win = (h_real + 1j * h_imag).astype(np.complex64)

    # Amplitud en dB.
    amp_db = 20.0 * np.log10(np.abs(h_win) + eps)

    # Normalización por captura: quitamos nivel global de potencia.
    # Esto hace que el modelo se centre más en la huella espectral/espacial
    # y menos en cambios globales de potencia.
    amp_center = np.median(amp_db, axis=(1, 2), keepdims=True)
    amp_norm = amp_db - amp_center

    amp_mean = np.mean(amp_norm, axis=0).reshape(-1)
    amp_std = np.std(amp_norm, axis=0).reshape(-1)

    amp_diff = np.abs(np.diff(amp_norm, axis=0))
    amp_diff_mean = np.mean(amp_diff, axis=0).reshape(-1)

    # Fase diferencial temporal:
    # angle(H(t) * conj(H(t-1)))
    phase_dt = np.angle(h_win[1:, :, :] * np.conj(h_win[:-1, :, :]))
    phase_dt_sin = np.mean(np.sin(phase_dt), axis=0).reshape(-1)
    phase_dt_cos = np.mean(np.cos(phase_dt), axis=0).reshape(-1)

    # Fase diferencial en frecuencia:
    # angle(H(k+1) * conj(H(k)))
    phase_df = np.angle(h_win[:, 1:, :] * np.conj(h_win[:, :-1, :]))
    phase_df_sin = np.mean(np.sin(phase_df), axis=0).reshape(-1)
    phase_df_cos = np.mean(np.cos(phase_df), axis=0).reshape(-1)

    # Calidad de la ventana.
    global_amp = amp_db.reshape(amp_db.shape[0], -1).mean(axis=1)

    quality = np.array([
        np.mean(freq_win),
        np.std(freq_win),
        np.mean(noise_win),
        np.std(noise_win),
        np.mean(timing_win),
        np.std(timing_win),
        np.mean(global_amp),
        np.std(global_amp),
    ], dtype=np.float32)

    feat = np.concatenate([
        amp_mean,
        amp_std,
        amp_diff_mean,
        phase_dt_sin,
        phase_dt_cos,
        phase_df_sin,
        phase_df_cos,
        quality,
    ]).astype(np.float32)

    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    return feat


def build_dataset(raw_dir: Path, out_npz: Path, out_meta: Path, window_size: int, stride: int):
    session_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])

    if not session_dirs:
        raise RuntimeError(f"No hay sesiones en {raw_dir}")

    empty_sessions_sorted = []
    for sdir in session_dirs:
        meta_path = sdir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("label") == "empty":
            empty_sessions_sorted.append(sdir.name)

    empty_index_by_session = {
        sid: i for i, sid in enumerate(sorted(empty_sessions_sorted))
    }

    X_list = []
    rows = []

    print(f"Sesiones encontradas: {len(session_dirs)}")
    print(f"Window size: {window_size}")
    print(f"Stride: {stride}")

    for sdir in tqdm(session_dirs, desc="Procesando sesiones"):
        mat_path = sdir / "session_data.mat"
        meta_path = sdir / "metadata.json"

        if not mat_path.exists() or not meta_path.exists():
            print(f"Saltando {sdir.name}: falta session_data.mat o metadata.json")
            continue

        meta = json.loads(meta_path.read_text())

        label = meta["label"]
        orientation = meta.get("orientation", "unknown")
        session_id = meta["session_id"]

        h_raw = read_matlab_dataset(mat_path, "hSSB")
        h = force_hssb_to_n_240_4(h_raw)

        n = h.shape[0]

        freq = read_optional_1d(mat_path, "freqLog", n)
        noise = read_optional_1d(mat_path, "noiseLog", n)
        timing = read_optional_1d(mat_path, "timingLog", n)

        if n < window_size:
            print(f"Saltando {session_id}: solo {n} capturas")
            continue

        empty_idx = empty_index_by_session.get(sdir.name, None)
        fold_id = assign_fold(label, orientation, empty_idx)

        n_windows = 0

        for start in range(0, n - window_size + 1, stride):
            end = start + window_size

            h_win = h[start:end, :, :]
            freq_win = freq[start:end]
            noise_win = noise[start:end]
            timing_win = timing[start:end]

            feat = extract_window_features(h_win, freq_win, noise_win, timing_win)

            X_list.append(feat)

            rows.append({
                "session_dir": sdir.name,
                "session_id": session_id,
                "label": label,
                "orientation": orientation,
                "movement_state": meta.get("movement_state", "static"),
                "person_id": meta.get("person_id", "unknown"),
                "fold_id": fold_id,
                "window_start": start,
                "window_end": end - 1,
                "window_size": window_size,
                "stride": stride,
                "n_captures_session": n,
                "accepted_rate_hz": meta.get("accepted_rate_hz", np.nan),
            })

            n_windows += 1

        print(f"{session_id}: label={label}, orientation={orientation}, N={n}, windows={n_windows}, fold={fold_id}")

    if not X_list:
        raise RuntimeError("No se ha generado ninguna ventana.")

    X = np.vstack(X_list).astype(np.float32)
    meta_df = pd.DataFrame(rows)

    y = meta_df["label"].to_numpy()
    groups = meta_df["session_id"].to_numpy()
    folds = meta_df["fold_id"].to_numpy()

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
    )

    meta_df.to_csv(out_meta, index=False)

    print("\n=== Dataset construido ===")
    print(f"X shape: {X.shape}")
    print(f"Metadata rows: {len(meta_df)}")
    print(f"Labels:")
    print(meta_df["label"].value_counts())
    print(f"Folds:")
    print(meta_df["fold_id"].value_counts().sort_index())
    print(f"Guardado NPZ: {out_npz}")
    print(f"Guardado CSV: {out_meta}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="data/processed/ssb_windows_v001.npz")
    parser.add_argument("--meta-out", default="data/processed/ssb_windows_metadata_v001.csv")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)

    args = parser.parse_args()

    build_dataset(
        raw_dir=Path(args.raw_dir),
        out_npz=Path(args.out),
        out_meta=Path(args.meta_out),
        window_size=args.window,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
