#!/usr/bin/env python3
"""
Online rxGridSSB inference server.

This server is designed for real-time inference with no accumulated delay.

MATLAB sends one rxGridSSB sample and waits for the answer.
Therefore:
    - there is no inference queue,
    - there is no growing buffer,
    - if Python is slower, MATLAB waits and physical SSBs are missed,
      but stale delayed predictions are never produced.

Protocol:
    MATLAB -> Python:
        uint32 little-endian payload length
        float32 payload, MATLAB column-major order, shape [2, 240, 4]
            channel 0 = real(rxGridSSB)
            channel 1 = imag(rxGridSSB)

    Python -> MATLAB:
        ASCII line:
        RESULT sample_idx prob_p5 smooth_prob label inference_ms total_ms

Python preprocessing:
    real/imag -> abs/phase
    optional normalization:
        - checkpoint if available,
        - metadata train split if available,
        - identity otherwise.
"""

import argparse
import gc
import importlib.util
import json
import shlex
import socket
import struct
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


# -------------------------------------------------------------------------
# Default 5G online export configuration
# -------------------------------------------------------------------------

DEFAULT_MODEL = "results/binary_empty_vs_P5_rx/model_rxGridSSB/model.pt"
DEFAULT_METADATA = "results/binary_empty_vs_P5_rx/model_rxGridSSB/metadata.csv"
DEFAULT_NORMALIZATION_CACHE = "results/binary_empty_vs_P5_rx/model_rxGridSSB/normalization_rx_abs_phase.npz"

DEFAULT_JSON_OUT = "results/online/live_inference_state_5G.json"
DEFAULT_JSON_NODE_ID = "ssb_receiver_5G"
DEFAULT_JSON_MODEL_ID = "binary_empty_P5_rxGridSSB"

DEFAULT_REMOTE_USER = "nextnet"
DEFAULT_REMOTE_IP = "163.117.140.146"
DEFAULT_REMOTE_JSON_PATH = "~/AlbertoDir/demo_5G/5G_inference/live_inference_state_5G.json"
DEFAULT_SCP_DESTINATION = f"{DEFAULT_REMOTE_USER}@{DEFAULT_REMOTE_IP}:{DEFAULT_REMOTE_JSON_PATH}"

# BatchMode=yes evita que SCP se quede pidiendo contraseña dentro del loop online.
DEFAULT_SCP_COMMAND = "scp -q -o BatchMode=yes -o ConnectTimeout=1"


# -------------------------------------------------------------------------
# Socket helpers
# -------------------------------------------------------------------------

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


# -------------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------------

def load_training_module(project_root: Path):
    """
    Dynamically imports the original training script.

    Important:
        sys.modules[module_name] = module is required for dataclasses.
    """

    train_script = project_root / "src" / "python" / "train_datassb_binary_pipeline.py"

    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")

    module_name = "train_datassb_binary_pipeline"
    spec = importlib.util.spec_from_file_location(module_name, train_script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, nn.Module):
        return None

    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model", "net", "network"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

        # Sometimes model.pt is directly a state_dict.
        tensor_values = [v for v in checkpoint.values() if torch.is_tensor(v)]
        if len(tensor_values) > 0:
            return checkpoint

    raise RuntimeError("Could not extract state_dict from checkpoint.")


def normalize_state_dict_keys(state_dict):
    out = {}

    for key, value in state_dict.items():
        new_key = key

        for prefix in ["module.", "model.", "net.", "network."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]

        out[new_key] = value

    return out


def try_build_model_from_training_script(module, state_dict):
    """
    Tries to reconstruct the model using builders/classes from the training script.
    This may fail if the Codex-generated training script does not expose the builder.
    """

    candidate_builders = [
        "build_model",
        "make_model",
        "create_model",
        "build_rx_model",
        "build_rxgridssb_model",
        "build_cnn_model",
        "make_cnn",
    ]

    candidate_kwargs = [
        {"model_name": "model_rxGridSSB", "input_shape": (2, 240, 4)},
        {"name": "model_rxGridSSB", "input_shape": (2, 240, 4)},
        {"model_key": "model_rxGridSSB", "input_shape": (2, 240, 4)},
        {"model": "rx", "input_shape": (2, 240, 4)},
        {"input_shape": (2, 240, 4)},
        {"in_channels": 2},
        {"input_channels": 2},
        {},
    ]

    for builder_name in candidate_builders:
        if not hasattr(module, builder_name):
            continue

        builder = getattr(module, builder_name)

        for kwargs in candidate_kwargs:
            try:
                model = builder(**kwargs)

                if not isinstance(model, nn.Module):
                    continue

                model.load_state_dict(state_dict, strict=True)
                print(f"{GREEN}Loaded model using {builder_name}({kwargs}).{RESET}")
                return model

            except Exception:
                continue

    for attr_name in dir(module):
        attr = getattr(module, attr_name)

        if not isinstance(attr, type):
            continue

        if not issubclass(attr, nn.Module) or attr is nn.Module:
            continue

        constructor_attempts = [
            {},
            {"input_shape": (2, 240, 4)},
            {"in_channels": 2},
            {"input_channels": 2},
            {"n_channels": 2},
            {"num_classes": 1},
            {"in_channels": 2, "num_classes": 1},
            {"input_channels": 2, "num_classes": 1},
            {"input_shape": (2, 240, 4), "num_classes": 1},
        ]

        for kwargs in constructor_attempts:
            try:
                model = attr(**kwargs)
                model.load_state_dict(state_dict, strict=True)
                print(f"{GREEN}Loaded model using class {attr_name}({kwargs}).{RESET}")
                return model

            except Exception:
                continue

    return None


class NumericSequentialRxGridSSBCNN(nn.Module):
    """
    Architecture compatible with the Codex-trained nn.Sequential checkpoint.

    Expected parameter keys:
        0.*   Conv2d
        1.*   BatchNorm2d
        4.*   Conv2d
        5.*   BatchNorm2d
        8.*   Conv2d
        9.*   BatchNorm2d
        14.*  Linear
        17.*  Linear

    The non-parametric layers are placed so that the state_dict names match exactly.
    """

    def __init__(self, state_dict):
        super().__init__()

        required = [
            "0.weight", "0.bias",
            "1.weight", "1.bias",
            "4.weight", "4.bias",
            "5.weight", "5.bias",
            "8.weight", "8.bias",
            "9.weight", "9.bias",
            "14.weight", "14.bias",
            "17.weight", "17.bias",
        ]

        missing = [k for k in required if k not in state_dict]
        if missing:
            raise RuntimeError(f"Missing keys for NumericSequentialRxGridSSBCNN: {missing}")

        w0 = state_dict["0.weight"]
        w4 = state_dict["4.weight"]
        w8 = state_dict["8.weight"]
        w14 = state_dict["14.weight"]
        w17 = state_dict["17.weight"]

        c0_out, c0_in, k0_h, k0_w = w0.shape
        c4_out, c4_in, k4_h, k4_w = w4.shape
        c8_out, c8_in, k8_h, k8_w = w8.shape

        hidden_out, linear_in = w14.shape
        final_out, hidden_in = w17.shape

        if linear_in % c8_out != 0:
            raise RuntimeError(
                f"Cannot infer adaptive pool shape: linear_in={linear_in}, conv3_out={c8_out}"
            )

        pool_area = linear_in // c8_out

        if pool_area == 1:
            pool_hw = (1, 1)
        elif pool_area == 2:
            pool_hw = (2, 1)
        elif pool_area == 4:
            pool_hw = (2, 2)
        elif pool_area == 8:
            pool_hw = (4, 2)
        elif pool_area == 16:
            pool_hw = (4, 4)
        else:
            pool_hw = (pool_area, 1)

        print(f"{BLUE}NumericSequential inferred architecture:{RESET}")
        print(f"  Conv0: in={c0_in}, out={c0_out}, kernel=({k0_h},{k0_w})")
        print(f"  Conv4: in={c4_in}, out={c4_out}, kernel=({k4_h},{k4_w})")
        print(f"  Conv8: in={c8_in}, out={c8_out}, kernel=({k8_h},{k8_w})")
        print(f"  AdaptiveAvgPool2d: {pool_hw}")
        print(f"  Linear14: in={linear_in}, out={hidden_out}")
        print(f"  Linear17: in={hidden_in}, out={final_out}")

        self.net = nn.Sequential(
            nn.Conv2d(c0_in, c0_out, kernel_size=(k0_h, k0_w), padding=(k0_h // 2, 0)),
            nn.BatchNorm2d(c0_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(c4_in, c4_out, kernel_size=(k4_h, k4_w), padding=(k4_h // 2, 0)),
            nn.BatchNorm2d(c4_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(c8_in, c8_out, kernel_size=(k8_h, k8_w), padding=(k8_h // 2, 0)),
            nn.BatchNorm2d(c8_out),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(pool_hw),
            nn.Flatten(),
            nn.Dropout(0.2),

            nn.Linear(linear_in, hidden_out),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(hidden_in, final_out),
        )

    def forward(self, x):
        return self.net(x).reshape(-1)


def try_load_numeric_sequential_model(state_dict):
    model = NumericSequentialRxGridSSBCNN(state_dict)

    # NumericSequentialRxGridSSBCNN stores the actual Sequential as self.net,
    # so its state_dict keys are net.0.weight, net.1.weight, ...
    # The training checkpoint was saved from a plain nn.Sequential, so its keys
    # are 0.weight, 1.weight, ...
    prefixed_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("net."):
            prefixed_state_dict[key] = value
        else:
            prefixed_state_dict["net." + key] = value

    model.load_state_dict(prefixed_state_dict, strict=True)
    print(f"{GREEN}Loaded model using NumericSequentialRxGridSSBCNN.{RESET}")
    return model


class FallbackRxGridSSBCNN(nn.Module):
    """
    Fallback architecture with named submodules.

    This is only used if the numeric Sequential loader fails.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=(7, 2), padding=(3, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(16, 32, kernel_size=(5, 2), padding=(2, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.reshape(-1)


def try_load_fallback_model(state_dict):
    model = FallbackRxGridSSBCNN()
    model.load_state_dict(state_dict, strict=True)
    print(f"{YELLOW}Loaded model using fallback CNN architecture.{RESET}")
    return model


def load_model(model_path: Path, project_root: Path, device: torch.device):
    print(f"Loading model:\n{model_path}")

    # PyTorch >= 2.6 defaults to weights_only=True.
    # Our checkpoint was created locally and contains numpy objects, so we explicitly use weights_only=False.
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
        print(f"{GREEN}Checkpoint contains full nn.Module.{RESET}")
    else:
        state_dict = extract_state_dict(checkpoint)
        state_dict = normalize_state_dict_keys(state_dict)

        module = None
        try:
            module = load_training_module(project_root)
        except Exception as exc:
            print(f"{YELLOW}Could not import training module. Will try direct architecture reconstruction.{RESET}")
            print(f"{YELLOW}{exc}{RESET}")

        model = None
        if module is not None:
            model = try_build_model_from_training_script(module, state_dict)

        if model is None:
            print(f"{YELLOW}Could not reconstruct from training script. Trying numeric Sequential architecture...{RESET}")

            try:
                model = try_load_numeric_sequential_model(state_dict)
            except Exception as exc_numeric:
                print(f"{YELLOW}Numeric Sequential loading failed. Trying fallback CNN...{RESET}")
                print(exc_numeric)

                try:
                    model = try_load_fallback_model(state_dict)
                except Exception as exc:
                    print(f"{RED}Could not load model.{RESET}")
                    print(exc)
                    print("")
                    print("Inspect checkpoint with:")
                    print(f"""
python3 - <<'PY'
import torch
ckpt = torch.load("{model_path}", map_location="cpu", weights_only=False)
print(type(ckpt))
if isinstance(ckpt, dict):
    print(ckpt.keys())
    sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    print(list(sd.keys())[:80])
    for k,v in sd.items():
        if hasattr(v, "shape"):
            print(k, tuple(v.shape))
PY
""")
                    sys.exit(1)

    model.to(device)
    model.eval()

    return model, checkpoint


# -------------------------------------------------------------------------
# Normalization
# -------------------------------------------------------------------------

def extract_checkpoint_normalization(checkpoint):
    if not isinstance(checkpoint, dict):
        return None

    candidates = []

    if "normalization" in checkpoint and isinstance(checkpoint["normalization"], dict):
        norm = checkpoint["normalization"]
        candidates.append((norm.get("mean"), norm.get("std")))

    candidates.append((checkpoint.get("mean"), checkpoint.get("std")))
    candidates.append((checkpoint.get("norm_mean"), checkpoint.get("norm_std")))
    candidates.append((checkpoint.get("x_mean"), checkpoint.get("x_std")))

    for mean, std in candidates:
        if mean is None or std is None:
            continue

        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        std = np.where(std < 1e-8, 1.0, std)

        print(f"{GREEN}Using normalization from checkpoint. mean={mean.shape}, std={std.shape}{RESET}")
        return mean, std

    return None


def load_mat_dataSSB(mat_path: Path):
    """
    Loads dataSSB from either MATLAB v7.3 HDF5 or older MAT formats.

    Returns:
        complex ndarray with shape [360, 6, N]
    """

    # First try scipy.
    try:
        import scipy.io

        mat = scipy.io.loadmat(mat_path)

        if "dataSSB" in mat:
            z = mat["dataSSB"]

            if z.ndim != 3:
                raise RuntimeError(f"Expected 3D dataSSB from scipy, got {z.shape}")

            if z.shape[0] == 360 and z.shape[1] == 6:
                return np.asarray(z)

            if z.shape[-1] == 360 and z.shape[-2] == 6:
                return np.transpose(z, (2, 1, 0))

            raise RuntimeError(f"Unexpected scipy dataSSB shape: {z.shape}")

    except NotImplementedError:
        pass
    except ValueError:
        pass
    except Exception:
        pass

    # Then try h5py for v7.3.
    import h5py

    with h5py.File(mat_path, "r") as f:
        if "dataSSB" not in f:
            raise KeyError(f"dataSSB not found in {mat_path}")

        dset = f["dataSSB"]
        raw = dset[()]

    if raw.dtype.fields is not None:
        if "real" in raw.dtype.fields and "imag" in raw.dtype.fields:
            z = raw["real"] + 1j * raw["imag"]
        else:
            raise RuntimeError(f"Unsupported complex dtype fields: {raw.dtype.fields}")
    else:
        z = raw

    z = np.asarray(z)

    if z.ndim != 3:
        raise RuntimeError(f"Expected 3D dataSSB from h5py, got {z.shape}")

    if z.shape[0] == 360 and z.shape[1] == 6:
        return z

    if z.shape[-1] == 360 and z.shape[-2] == 6:
        return np.transpose(z, (2, 1, 0))

    raise RuntimeError(f"Could not infer dataSSB orientation. Shape={z.shape}, file={mat_path}")


def resolve_dataset_file_path(file_path: str, project_root: Path, metadata_path: Path):
    path = Path(file_path)

    if path.is_absolute() and path.exists():
        return path

    candidates = [
        project_root / file_path,
        metadata_path.parent / file_path,
        metadata_path.parents[1] / file_path,
        metadata_path.parents[2] / file_path,
        metadata_path.parents[3] / file_path if len(metadata_path.parents) > 3 else None,
    ]

    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not resolve dataset file path: {file_path}")


def compute_normalization_from_metadata(metadata_path: Path, project_root: Path, cache_path: Path | None):
    """
    Computes feature-wise mean/std over train split.

    Model input:
        abs/phase rxGridSSB
        shape [2,240,4]
    """

    if cache_path is not None and cache_path.exists():
        data = np.load(cache_path)
        mean = data["mean"].astype(np.float32)
        std = data["std"].astype(np.float32)
        print(f"{GREEN}Loaded normalization cache:{RESET} {cache_path}")
        return mean, std

    if metadata_path is None or not metadata_path.exists():
        print(f"{YELLOW}No metadata.csv found. Using identity normalization.{RESET}")
        return None

    print(f"{BLUE}Computing normalization from train split metadata:{RESET}")
    print(metadata_path)

    df = pd.read_csv(metadata_path)

    if "split" not in df.columns:
        print(f"{YELLOW}metadata.csv has no split column. Using identity normalization.{RESET}")
        return None

    train_df = df[df["split"] == "train"].copy()

    if train_df.empty:
        print(f"{YELLOW}No train rows in metadata.csv. Using identity normalization.{RESET}")
        return None

    sum_x = np.zeros((2, 240, 4), dtype=np.float64)
    sum_x2 = np.zeros((2, 240, 4), dtype=np.float64)
    count = 0

    grouped = list(train_df.groupby("filePath"))

    for idx, (file_path, group) in enumerate(grouped, 1):
        path = resolve_dataset_file_path(file_path, project_root, metadata_path)
        print(f"  [{idx}/{len(grouped)}] stats from {path} ({len(group)} samples)")

        dataSSB = load_mat_dataSSB(path)

        capture_indices = group["captureIndex"].to_numpy(dtype=int) - 1
        capture_indices = capture_indices[(capture_indices >= 0) & (capture_indices < dataSSB.shape[2])]

        if len(capture_indices) == 0:
            continue

        rx = dataSSB[60:300, 1:5, capture_indices]  # [240,4,n]
        rx = np.transpose(rx, (2, 0, 1))             # [n,240,4]

        mag = np.abs(rx).astype(np.float32)
        phase = np.angle(rx).astype(np.float32)

        x = np.stack([mag, phase], axis=1)           # [n,2,240,4]

        sum_x += x.sum(axis=0)
        sum_x2 += (x.astype(np.float64) ** 2).sum(axis=0)
        count += x.shape[0]

        del dataSSB, rx, mag, phase, x
        gc.collect()

    if count == 0:
        print(f"{YELLOW}Could not compute normalization. Using identity normalization.{RESET}")
        return None

    mean = sum_x / count
    var = sum_x2 / count - mean ** 2
    var = np.maximum(var, 1e-8)
    std = np.sqrt(var)

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    print(f"{GREEN}Computed normalization from {count} train samples.{RESET}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, mean=mean, std=std)
        print(f"{GREEN}Saved normalization cache:{RESET} {cache_path}")

    return mean, std


def get_normalization(checkpoint, metadata_path: Path | None, project_root: Path, mode: str, cache_path: Path | None):
    if mode == "identity":
        print(f"{YELLOW}Using identity normalization.{RESET}")
        return None

    if mode in ["auto", "checkpoint"]:
        norm = extract_checkpoint_normalization(checkpoint)

        if norm is not None:
            return norm

        if mode == "checkpoint":
            print(f"{YELLOW}Checkpoint normalization not found. Falling back to identity.{RESET}")
            return None

    if mode in ["auto", "metadata"]:
        return compute_normalization_from_metadata(metadata_path, project_root, cache_path)

    raise ValueError(f"Unknown normalization mode: {mode}")


# -------------------------------------------------------------------------
# Preprocessing / inference
# -------------------------------------------------------------------------

def preprocess_payload(payload: bytes):
    arr = np.frombuffer(payload, dtype="<f4")

    if arr.size != 2 * 240 * 4:
        raise RuntimeError(f"Unexpected payload float count: {arr.size}")

    # MATLAB sends payload(:), column-major.
    arr = arr.reshape((2, 240, 4), order="F")

    real = arr[0]
    imag = arr[1]
    z = real + 1j * imag

    mag = np.abs(z).astype(np.float32)
    phase = np.angle(z).astype(np.float32)

    x = np.stack([mag, phase], axis=0).astype(np.float32)
    return x


def apply_normalization(x, norm):
    """
    Apply the same normalization used during training.

    Online x arrives as:
        x shape = [2, 240, 4]

    But the checkpoint may store mean/std as:
        [1, 2, 1, 1]
    because training used batched tensors.

    This function always returns:
        [2, 240, 4]
    """

    if norm is None:
        return x.astype(np.float32, copy=False)

    mean, std = norm

    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    # Remove leading singleton batch dimensions until mean/std can match x.
    while mean.ndim > x.ndim and mean.shape[0] == 1:
        mean = mean[0]

    while std.ndim > x.ndim and std.shape[0] == 1:
        std = std[0]

    # Typical case:
    #   x    = [2,240,4]
    #   mean = [2,1,1]
    #   std  = [2,1,1]
    y = (x - mean) / (std + 1e-8)

    # Safety: if broadcasting still produced a leading singleton dimension,
    # remove it.
    while y.ndim > 3 and y.shape[0] == 1:
        y = y[0]

    if y.shape != (2, 240, 4):
        raise RuntimeError(f"Normalized sample has wrong shape: {y.shape}, expected (2, 240, 4)")

    return y.astype(np.float32, copy=False)




# -------------------------------------------------------------------------
# JSON + SCP helpers
# -------------------------------------------------------------------------

def atomic_write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def maybe_scp_json(local_path: Path, args, sample_idx: int):
    """
    Optional SCP export.

    args.scp_destination format:
        user@host:/remote/path/live_inference_state.json

    To change the remote server later, only change --scp-destination.
    """
    destination = getattr(args, "scp_destination", "")

    if not destination:
        return "disabled"

    every_n = max(1, int(getattr(args, "scp_every_n", 1)))

    if sample_idx % every_n != 0:
        return "skipped"

    cmd = shlex.split(getattr(args, "scp_command", DEFAULT_SCP_COMMAND)) + [
        str(local_path),
        destination,
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=float(getattr(args, "scp_timeout", 1.5)),
        )
        return "ok"

    except subprocess.TimeoutExpired:
        return "timeout"

    except subprocess.CalledProcessError as exc:
        try:
            err = exc.stderr.decode("utf-8", errors="replace").strip()
        except Exception:
            err = str(exc)
        return "error:" + err[:160]

    except Exception as exc:
        return "error:" + str(exc)[:160]


def build_inference_json_payload(
    *,
    sample_idx: int,
    unix_time: float,
    node_id: str,
    model_id: str,
    label: str,
    candidate_label: str,
    prob_p5: float,
    smooth_prob: float,
    stable_mean: float,
    p5_votes: int,
    empty_votes: int,
    inference_ms: float,
    total_ms: float,
    rate_hz: float,
):
    return {
        "schema_version": "ssb_inference.v1",
        "node_id": node_id,
        "model_id": model_id,
        "sample_idx": int(sample_idx),
        "timestamp_unix": float(unix_time),
        "inference": {
            "stable_label": str(label),
            "candidate_label": str(candidate_label),
            "prob_p5": float(prob_p5),
            "smooth_prob": float(smooth_prob),
            "stable_mean": float(stable_mean),
            "p5_votes": int(p5_votes),
            "empty_votes": int(empty_votes),
        },
        "timing": {
            "inference_ms": float(inference_ms),
            "total_ms": float(total_ms),
            "rate_hz": float(rate_hz),
        },
        "status": {
            "source": "5g_ssb_rxgridssb",
            "transport": "tcp_matlab_to_python",
            "online": True,
        },
    }


def color_line(label, text):
    if label == "P5":
        return RED + text + RESET
    return GREEN + text + RESET


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--metadata", type=str, default=DEFAULT_METADATA)
    parser.add_argument("--normalization", type=str, default="auto")
    parser.add_argument("--normalization-cache", type=str, default=DEFAULT_NORMALIZATION_CACHE)

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--smooth-window", type=int, default=5)

    # Stabilizer / debounce logic.
    # A label change is accepted only if enough recent samples are confidently
    # pointing to the new class. This prevents flickering during movement or
    # out-of-distribution intermediate positions.
    parser.add_argument("--stable-window", type=int, default=5)
    parser.add_argument("--stable-count", type=int, default=4)
    parser.add_argument("--p5-high", type=float, default=0.85)
    parser.add_argument("--empty-low", type=float, default=0.15)
    parser.add_argument("--stable-mean-p5", type=float, default=0.70)
    parser.add_argument("--stable-mean-empty", type=float, default=0.30)
    parser.add_argument("--initial-state", type=str, default="EMPTY")

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--warn-ms", type=float, default=200.0)

    parser.add_argument(
        "--json-out",
        type=str,
        default=DEFAULT_JSON_OUT,
        help="Ruta local donde se escribe el último estado de inferencia en JSON.",
    )
    parser.add_argument(
        "--json-node-id",
        type=str,
        default=DEFAULT_JSON_NODE_ID,
        help="Identificador lógico del nodo/receptor incluido en el JSON.",
    )
    parser.add_argument(
        "--json-model-id",
        type=str,
        default=DEFAULT_JSON_MODEL_ID,
        help="Identificador lógico del modelo incluido en el JSON.",
    )
    parser.add_argument(
        "--scp-destination",
        type=str,
        default=DEFAULT_SCP_DESTINATION,
        help="Destino SCP. Formato: user@host:/remote/path/live_inference_state.json",
    )
    parser.add_argument(
        "--scp-every-n",
        type=int,
        default=1,
        help="Enviar por SCP cada N muestras. Por defecto, cada inferencia.",
    )
    parser.add_argument(
        "--scp-timeout",
        type=float,
        default=1.5,
        help="Timeout en segundos para cada envío SCP.",
    )
    parser.add_argument(
        "--scp-command",
        type=str,
        default=DEFAULT_SCP_COMMAND,
        help="Comando SCP. Permite opciones tipo BatchMode/ConnectTimeout.",
    )
    parser.add_argument("--log-csv", type=str, default="results/online_rxgridssb_inference_log.csv")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = project_root / model_path

    metadata_path = Path(args.metadata) if args.metadata else None
    if metadata_path is not None and not metadata_path.is_absolute():
        metadata_path = project_root / metadata_path

    cache_path = Path(args.normalization_cache) if args.normalization_cache else None
    if cache_path is not None and not cache_path.is_absolute():
        cache_path = project_root / cache_path

    log_path = Path(args.log_csv)
    if not log_path.is_absolute():
        log_path = project_root / log_path

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Project root: {project_root}")
    print(f"Device: {device}")

    model, checkpoint = load_model(model_path, project_root, device)

    json_out_path = Path(args.json_out)
    if not json_out_path.is_absolute():
        json_out_path = project_root / json_out_path
    json_out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"JSON output: {json_out_path}")
    if args.scp_destination:
        print(f"SCP destination: {args.scp_destination}")
        print(f"SCP every N samples: {args.scp_every_n}")
    else:
        print("SCP destination: disabled")
    norm = get_normalization(checkpoint, metadata_path, project_root, args.normalization, cache_path)

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("sample_idx,unix_time,prob_p5,smooth_prob,stable_mean,candidate_label,stable_label,p5_votes,empty_votes,inference_ms,total_ms,rate_hz\n")

    expected_bytes = 2 * 240 * 4 * 4

    print("")
    print(f"{BLUE}TCP server listening on {args.host}:{args.port}{RESET}")
    print(f"Expected payload: {expected_bytes} bytes")
    print("Mode: synchronous request-response, no queue, no accumulated delay.")
    print("Start MATLAB streamer in another terminal.")
    print("")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)

    prob_window = deque(maxlen=max(1, args.smooth_window, args.stable_window))
    decision_window = deque(maxlen=max(1, args.stable_window))

    stable_label = args.initial_state

    sample_idx = 0
    t_start = time.time()

    try:
        while True:
            print(f"{BLUE}Waiting for MATLAB connection...{RESET}")
            conn, addr = server.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            print(f"{GREEN}MATLAB connected from {addr}.{RESET}")

            with conn:
                while True:
                    total_t0 = time.perf_counter()

                    try:
                        header = recv_exact(conn, 4)
                        (payload_len,) = struct.unpack("<I", header)

                        if payload_len != expected_bytes:
                            raise RuntimeError(f"Unexpected payload length {payload_len}, expected {expected_bytes}")

                        payload = recv_exact(conn, payload_len)

                    except ConnectionError:
                        print(f"{YELLOW}Client disconnected before sending a complete sample. Waiting for MATLAB again...{RESET}", flush=True)
                        break

                    x = preprocess_payload(payload)
                    x = apply_normalization(x, norm)

                    xt = torch.from_numpy(x).unsqueeze(0).to(device=device, dtype=torch.float32)

                    infer_t0 = time.perf_counter()
                    with torch.no_grad():
                        logits = model(xt).reshape(-1)[0]
                        prob_p5 = torch.sigmoid(logits).item()
                    inference_ms = (time.perf_counter() - infer_t0) * 1000.0

                    prob_window.append(prob_p5)
                    smooth_prob = float(np.mean(prob_window))

                    # Instant candidate with confidence margins.
                    # Values in the middle are treated as UNKNOWN and do not
                    # force a state change.
                    if prob_p5 >= args.p5_high:
                        candidate_label = "P5"
                    elif prob_p5 <= args.empty_low:
                        candidate_label = "EMPTY"
                    else:
                        candidate_label = "UNKNOWN"

                    decision_window.append(candidate_label)

                    recent_probs = list(prob_window)[-args.stable_window:]
                    stable_mean = float(np.mean(recent_probs)) if len(recent_probs) > 0 else smooth_prob

                    p5_votes = sum(1 for x in decision_window if x == "P5")
                    empty_votes = sum(1 for x in decision_window if x == "EMPTY")

                    # Debounced state transition:
                    # - Switch to P5 only with enough confident P5 votes and high mean probability.
                    # - Switch to EMPTY only with enough confident EMPTY votes and low mean probability.
                    # - Otherwise keep previous stable state.
                    if len(decision_window) >= args.stable_window:
                        if p5_votes >= args.stable_count and stable_mean >= args.stable_mean_p5:
                            stable_label = "P5"
                        elif empty_votes >= args.stable_count and stable_mean <= args.stable_mean_empty:
                            stable_label = "EMPTY"

                    label = stable_label

                    total_ms = (time.perf_counter() - total_t0) * 1000.0

                    sample_idx += 1
                    elapsed = max(time.time() - t_start, 1e-9)
                    rate_hz = sample_idx / elapsed

                    line = (
                        f"[{sample_idx:06d}] {label:<5s} | "
                        f"raw={candidate_label:<7s} | "
                        f"prob_P5={prob_p5:7.4f} | "
                        f"smooth={smooth_prob:7.4f} | "
                        f"stable_mean={stable_mean:7.4f} | "
                        f"votes=P5:{p5_votes}/E:{empty_votes} | "
                        f"infer={inference_ms:6.2f} ms | "
                        f"total={total_ms:6.2f} ms | "
                        f"rate={rate_hz:5.2f}/s"
                    )

                    if total_ms > args.warn_ms:
                        line += f" {YELLOW}WARNING > {args.warn_ms:.0f} ms{RESET}"

                    print(color_line(label, line), flush=True)

                    inference_payload = build_inference_json_payload(
                        sample_idx=sample_idx,
                        unix_time=time.time(),
                        node_id=args.json_node_id,
                        model_id=args.json_model_id,
                        label=label,
                        candidate_label=candidate_label,
                        prob_p5=prob_p5,
                        smooth_prob=smooth_prob,
                        stable_mean=stable_mean,
                        p5_votes=p5_votes,
                        empty_votes=empty_votes,
                        inference_ms=inference_ms,
                        total_ms=total_ms,
                        rate_hz=rate_hz,
                    )

                    try:
                        atomic_write_json(json_out_path, inference_payload)
                        scp_status = maybe_scp_json(json_out_path, args, sample_idx)

                        if scp_status not in ("disabled", "skipped", "ok"):
                            print(f"{YELLOW}SCP status: {scp_status}{RESET}", flush=True)

                    except Exception as exc:
                        print(f"{YELLOW}JSON/SCP export error: {exc}{RESET}", flush=True)

                    response = (
                        f"RESULT {sample_idx} {prob_p5:.8f} {smooth_prob:.8f} "
                        f"{stable_mean:.8f} {label} {candidate_label} "
                        f"{p5_votes} {empty_votes} {inference_ms:.4f} {total_ms:.4f}\n"
                    )
                    conn.sendall(response.encode("utf-8"))

                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(
                            f"{sample_idx},{time.time():.6f},{prob_p5:.8f},"
                            f"{smooth_prob:.8f},{stable_mean:.8f},"
                            f"{candidate_label},{label},{p5_votes},{empty_votes},"
                            f"{inference_ms:.4f},{total_ms:.4f},{rate_hz:.4f}\n"
                        )

    except KeyboardInterrupt:
        print("\nStopping server.")
    except Exception as exc:
        print(f"{RED}Server error:{RESET} {exc}")
        raise
    finally:
        server.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
