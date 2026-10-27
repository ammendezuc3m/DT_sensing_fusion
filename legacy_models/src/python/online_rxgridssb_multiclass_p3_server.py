#!/usr/bin/env python3
from __future__ import annotations

import argparse
import socket
import struct
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

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


def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n

    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Client disconnected.")
        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def normalize_display_label(label: str) -> str:
    if label.lower() == "empty":
        return "EMPTY"
    return label


def color_label(label: str, text: str) -> str:
    if label == "P5":
        return RED + text + RESET
    if label == "P3":
        return BLUE + text + RESET
    if label == "EMPTY":
        return GREEN + text + RESET
    if label == "UNKNOWN":
        return YELLOW + text + RESET
    return text


def load_multiclass_model(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    class_names = checkpoint.get("class_names", checkpoint.get("classes", ["empty", "P5", "P3"]))
    class_names = [str(x) for x in class_names]

    model = RxGridSSBCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = checkpoint.get("mean", None)
    std = checkpoint.get("std", None)

    if mean is None:
        mean = checkpoint.get("x_mean", None)
    if std is None:
        std = checkpoint.get("x_std", None)

    if mean is None or std is None:
        raise RuntimeError("Checkpoint does not contain mean/std normalization.")

    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    print(f"{GREEN}Loaded multiclass model:{RESET} {model_path}")
    print(f"Classes: {class_names}")
    print(f"mean shape: {mean.shape}")
    print(f"std shape:  {std.shape}")

    return model, checkpoint, class_names, mean, std


def robust_profile_one(power_sc: np.ndarray):
    """
    power_sc: [240]
    """
    med = np.median(power_sc)
    q25 = np.percentile(power_sc, 25)
    q75 = np.percentile(power_sc, 75)
    iqr = max(float(q75 - q25), 1e-3)
    return (power_sc - med) / iqr


def region_mean(power_sc: np.ndarray, a: int, b: int):
    """
    a,b: 1-based inclusive
    """
    a0 = max(0, a - 1)
    b0 = min(240, b)
    return float(np.mean(power_sc[a0:b0]))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class P5SignatureOverride:
    def __init__(self, calibration_path: Path, threshold_override: float | None = None):
        data = np.load(calibration_path, allow_pickle=True)

        self.threshold = float(data["threshold"])
        if threshold_override is not None:
            self.threshold = float(threshold_override)

        self.coef = np.asarray(data["coef"], dtype=np.float32).reshape(-1)
        self.intercept = float(np.asarray(data["intercept"]).reshape(-1)[0])
        self.template = np.asarray(data["p5_template"], dtype=np.float32).reshape(-1)

        if self.template.shape[0] != 240:
            raise RuntimeError(f"Expected p5_template length 240, got {self.template.shape}")

        t = self.template.astype(np.float32)
        t = t - np.mean(t)
        t = t / (np.linalg.norm(t) + EPS)
        self.template_norm = t

        print(f"{GREEN}Loaded P5 signature calibration:{RESET} {calibration_path}")
        print(f"P5 signature threshold: {self.threshold:.6f}")
        print(f"coef shape: {self.coef.shape}")

    def features_from_rx(self, rx_complex: np.ndarray):
        """
        rx_complex: [240, 4]
        """
        power_sc = 10.0 * np.log10(np.mean(np.abs(rx_complex) ** 2, axis=1) + EPS)

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

        prof = robust_profile_one(power_sc)
        template_score = float(np.dot(prof, self.template_norm))

        features = np.array([
            valley,
            high,
            tail,
            valley_depth,
            high_minus_valley,
            tail_minus_valley,
            high_minus_low,
            plateau_minus_valley,
            template_score,
        ], dtype=np.float32)

        return features

    def score(self, rx_complex: np.ndarray):
        f = self.features_from_rx(rx_complex)
        logit = float(np.dot(self.coef, f) + self.intercept)
        return float(sigmoid(logit))


def build_input_from_payload(payload: bytes):
    arr = np.frombuffer(payload, dtype="<f4")

    expected = 2 * 240 * 4
    if arr.size != expected:
        raise RuntimeError(f"Bad payload size: got {arr.size} float32, expected {expected}")

    # MATLAB manda column-major.
    x = arr.reshape((2, 240, 4), order="F")

    real = x[0]
    imag = x[1]
    rx_complex = real + 1j * imag

    mag = np.abs(rx_complex).astype(np.float32)
    phase = np.angle(rx_complex).astype(np.float32)

    model_input = np.stack([mag, phase], axis=0).astype(np.float32)  # [2,240,4]

    return rx_complex.astype(np.complex64), model_input


def apply_normalization(model_input: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """
    model_input: [2,240,4]
    checkpoint mean/std can be [1,2,1,1] or [2,1,1] or compatible.
    returns [1,2,240,4]
    """
    xb = model_input[None, :, :, :].astype(np.float32)

    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    try:
        xb = (xb - mean) / (std + 1e-8)
    except ValueError:
        mean2 = np.squeeze(mean)
        std2 = np.squeeze(std)

        if mean2.shape == (2,):
            xb = (xb - mean2[None, :, None, None]) / (std2[None, :, None, None] + 1e-8)
        else:
            raise

    return xb.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)

    parser.add_argument("--model", required=True)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--normalization", default="auto")
    parser.add_argument("--normalization-cache", default=None)

    parser.add_argument("--p5-signature-calibration", required=True)
    parser.add_argument("--p5-signature-threshold", type=float, default=None)

    parser.add_argument("--stable-window", type=int, default=5)
    parser.add_argument("--stable-count", type=int, default=4)
    parser.add_argument("--initial-state", default="EMPTY")

    parser.add_argument("--warn-ms", type=float, default=200.0)
    parser.add_argument("--log", default="online_rxgridssb_multiclass_log.csv")

    args = parser.parse_args()

    project_root = Path.cwd()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = project_root / model_path

    cal_path = Path(args.p5_signature_calibration)
    if not cal_path.is_absolute():
        cal_path = project_root / cal_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint, class_names, mean, std = load_multiclass_model(model_path, device)
    p5_override = P5SignatureOverride(cal_path, args.p5_signature_threshold)

    display_classes = [normalize_display_label(c) for c in class_names]

    if "P5" not in display_classes:
        raise RuntimeError(f"Class P5 not found in checkpoint classes: {display_classes}")
    if "P3" not in display_classes:
        raise RuntimeError(f"Class P3 not found in checkpoint classes: {display_classes}")

    p5_idx = display_classes.index("P5")
    p3_idx = display_classes.index("P3")
    empty_idx = display_classes.index("EMPTY") if "EMPTY" in display_classes else None

    print()
    print("============================================================")
    print("Online multiclass rxGridSSB server")
    print("============================================================")
    print(f"Device: {device}")
    print(f"Listening: {args.host}:{args.port}")
    print(f"Stable window/count: {args.stable_window}/{args.stable_count}")
    print(f"Initial state: {args.initial_state}")
    print("Protocol: same MATLAB request-response, no queue.")
    print("Labels: EMPTY / P5 / P3")
    print("P3 color: blue")
    print("============================================================")
    print()

    decision_window = deque(maxlen=max(1, args.stable_window))
    stable_label = normalize_display_label(args.initial_state)

    sample_idx = 0
    last_time = None

    log_path = Path(args.log)
    with log_path.open("w") as f:
        f.write(
            "sample_idx,unix_time,"
            "p_empty,p_p5,p_p3,p5_signature_score,"
            "raw_model_label,candidate_label,stable_label,"
            "votes_empty,votes_p5,votes_p3,"
            "inference_ms,total_ms,rate_hz\n"
        )

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)

    print(f"{GREEN}TCP server listening on {args.host}:{args.port}{RESET}")
    print("Waiting for MATLAB connection...")

    try:
        while True:
            conn, addr = server.accept()
            print(f"{GREEN}MATLAB connected from {addr}{RESET}")

            with conn:
                while True:
                    total_t0 = time.perf_counter()

                    try:
                        header = recv_exact(conn, 4)
                    except ConnectionError:
                        print(f"{YELLOW}MATLAB disconnected.{RESET}")
                        break

                    payload_len = struct.unpack("<I", header)[0]
                    payload = recv_exact(conn, payload_len)

                    sample_idx += 1

                    rx_complex, model_input = build_input_from_payload(payload)

                    infer_t0 = time.perf_counter()

                    xb = apply_normalization(model_input, mean, std)
                    xb_t = torch.from_numpy(xb).to(device, dtype=torch.float32)

                    with torch.no_grad():
                        logits = model(xb_t)
                        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

                    raw_model_idx = int(np.argmax(probs))
                    raw_model_label = display_classes[raw_model_idx]

                    p5_sig_score = p5_override.score(rx_complex)

                    if p5_sig_score >= p5_override.threshold:
                        candidate_label = "P5"
                    else:
                        candidate_label = raw_model_label

                    decision_window.append(candidate_label)

                    votes = {
                        "EMPTY": sum(1 for x in decision_window if x == "EMPTY"),
                        "P5": sum(1 for x in decision_window if x == "P5"),
                        "P3": sum(1 for x in decision_window if x == "P3"),
                    }

                    if len(decision_window) >= args.stable_window:
                        # Prioridad: si una clase consigue stable_count votos, se acepta.
                        # En empate, se mantiene el estado anterior salvo que P5 tenga override fuerte.
                        if votes["P5"] >= args.stable_count:
                            stable_label = "P5"
                        elif votes["P3"] >= args.stable_count:
                            stable_label = "P3"
                        elif votes["EMPTY"] >= args.stable_count:
                            stable_label = "EMPTY"

                    label = stable_label

                    inference_ms = 1000.0 * (time.perf_counter() - infer_t0)
                    total_ms = 1000.0 * (time.perf_counter() - total_t0)

                    now = time.time()
                    if last_time is None:
                        rate_hz = 0.0
                    else:
                        dt = now - last_time
                        rate_hz = 1.0 / dt if dt > 0 else 0.0
                    last_time = now

                    p_empty = float(probs[empty_idx]) if empty_idx is not None else 0.0
                    p_p5 = float(probs[p5_idx])
                    p_p3 = float(probs[p3_idx])

                    line = (
                        f"[{sample_idx:06d}] {label:<5s} | "
                        f"raw={candidate_label:<5s} | "
                        f"model={raw_model_label:<5s} | "
                        f"p_empty={p_empty:7.4f} | "
                        f"p_P5={p_p5:7.4f} | "
                        f"p_P3={p_p3:7.4f} | "
                        f"p5sig={p5_sig_score:8.5f} | "
                        f"votes=E:{votes['EMPTY']}/P5:{votes['P5']}/P3:{votes['P3']} | "
                        f"infer≈{inference_ms:.1f} ms | "
                        f"total≈{total_ms:.1f} ms | "
                        f"rate≈{rate_hz:.1f}/s"
                    )

                    if total_ms > args.warn_ms:
                        line += f" | {YELLOW}WARN slow{RESET}"

                    print(color_label(label, line), flush=True)

                    # Mantengo una respuesta compatible con el MATLAB anterior:
                    # RESULT sample_idx prob_p5 smooth_prob stable_mean label candidate inference_ms total_ms rate_hz
                    #
                    # smooth_prob y stable_mean se dejan como p_P5 para compatibilidad.
                    response = (
                        f"RESULT {sample_idx} {p_p5:.8f} {p_p5:.8f} "
                        f"{p_p5:.8f} {label} {candidate_label} "
                        f"{inference_ms:.3f} {total_ms:.3f} {rate_hz:.3f}\n"
                    )
                    conn.sendall(response.encode("utf-8"))

                    with log_path.open("a") as f:
                        f.write(
                            f"{sample_idx},{now:.6f},"
                            f"{p_empty:.8f},{p_p5:.8f},{p_p3:.8f},{p5_sig_score:.8f},"
                            f"{raw_model_label},{candidate_label},{label},"
                            f"{votes['EMPTY']},{votes['P5']},{votes['P3']},"
                            f"{inference_ms:.3f},{total_ms:.3f},{rate_hz:.3f}\n"
                        )

            print("Waiting for MATLAB connection...")

    finally:
        server.close()


if __name__ == "__main__":
    main()
