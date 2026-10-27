#!/usr/bin/env python3
"""
PyTorch rxGridSSB binary inference utilities.

Expected model checkpoint:
    results/binary_empty_vs_P5_rx/model_rxGridSSB/model.pt

Checkpoint format:
    {
      "model_state_dict": ...,
      "mean": np.ndarray shape [1, 2, 1, 1],
      "std": np.ndarray shape [1, 2, 1, 1],
      "input_shape": [2, 240, 4],
      "complex_mode": "abs_phase",
      "classes": ["empty", "P5"],
      ...
    }

Input rxGridSSB:
    complex array [240, 4]

Model input:
    float tensor [1, 2, 240, 4]
    channel 0 = abs(rxGridSSB)
    channel 1 = angle(rxGridSSB)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class RxGridCNN2D(nn.Module):
    def __init__(self, in_ch: int = 2):
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
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RxGridTorchBinaryModel:
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)

        # weights_only=False is needed because this project checkpoint contains
        # numpy arrays and metadata. Use only with checkpoints you trust.
        ckpt = torch.load(
            str(self.checkpoint_path),
            map_location=self.device,
            weights_only=False,
        )

        self.ckpt = ckpt
        self.classes = list(ckpt.get("classes", ["empty", "P5"]))
        self.input_shape = list(ckpt.get("input_shape", [2, 240, 4]))
        self.complex_mode = str(ckpt.get("complex_mode", "abs_phase"))
        self.model_name = str(ckpt.get("model_name", "model_rxGridSSB"))

        if self.input_shape != [2, 240, 4]:
            raise RuntimeError(f"Unsupported input_shape={self.input_shape}; expected [2, 240, 4].")

        if self.complex_mode != "abs_phase":
            raise RuntimeError(f"Unsupported complex_mode={self.complex_mode}; expected abs_phase.")

        self.mean = np.asarray(ckpt["mean"], dtype=np.float32)
        self.std = np.asarray(ckpt["std"], dtype=np.float32)
        self.std = np.maximum(self.std, 1e-4)

        self.model = RxGridCNN2D(in_ch=2).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.eval()

    @staticmethod
    def rxgrid_to_channels(rx_grid_ssb: np.ndarray) -> np.ndarray:
        rx = np.asarray(rx_grid_ssb, dtype=np.complex64)

        if rx.shape != (240, 4):
            raise RuntimeError(f"rxGridSSB must have shape (240, 4), got {rx.shape}")

        x = np.stack(
            [
                np.abs(rx),
                np.angle(rx),
            ],
            axis=0,
        ).astype(np.float32)

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x

    def preprocess(self, rx_grid_ssb: np.ndarray) -> torch.Tensor:
        x = self.rxgrid_to_channels(rx_grid_ssb)

        # Add batch dimension: [1, 2, 240, 4]
        x = x[None, :, :, :]

        # Same standardization as training.
        x = (x - self.mean) / self.std
        x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        return torch.from_numpy(x).to(self.device)

    def predict_proba(self, rx_grid_ssb: np.ndarray) -> dict:
        xb = self.preprocess(rx_grid_ssb)

        with torch.no_grad():
            logit = self.model(xb)
            prob_p5 = torch.sigmoid(logit).detach().cpu().numpy().reshape(-1)[0]

        prob_p5 = float(prob_p5)
        prob_empty = float(1.0 - prob_p5)

        if prob_p5 >= 0.5:
            class_id = 1
            label = self.classes[1]
            confidence = prob_p5
        else:
            class_id = 0
            label = self.classes[0]
            confidence = prob_empty

        return {
            "label": label,
            "class_id": int(class_id),
            "confidence": float(confidence),
            "probabilities": {
                self.classes[0]: prob_empty,
                self.classes[1]: prob_p5,
            },
            "features": {
                "rxGridSSB_mean_abs": float(np.mean(np.abs(rx_grid_ssb))),
                "rxGridSSB_median_abs": float(np.median(np.abs(rx_grid_ssb))),
                "rxGridSSB_std_abs": float(np.std(np.abs(rx_grid_ssb))),
                "rxGridSSB_max_abs": float(np.max(np.abs(rx_grid_ssb))),
            },
            "model": {
                "model_name": self.model_name,
                "model_type": "torch_cnn2d_abs_phase",
                "checkpoint": str(self.checkpoint_path),
                "input_shape": self.input_shape,
                "complex_mode": self.complex_mode,
                "classes": self.classes,
            },
        }
