"""PyTorch deployment support for the generic Fisher SSB classifier.

The offline trainer selects resource elements from a snapshot-normalised
channel estimate and fits a scikit-learn logistic regression.  A logistic
regression is a linear layer, so its scaler can be folded into ``nn.Linear``
and saved as a normal PyTorch checkpoint without changing its predictions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from py3gpp.nrChannelEstimate import nrChannelEstimate
from py3gpp.nrPBCHDMRS import nrPBCHDMRS
from py3gpp.nrPBCHDMRSIndices import nrPBCHDMRSIndices
from py3gpp.nrSSS import nrSSS


N_PROBE = 41


class FisherLinearClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def build_reference_grid(pci: int, ibar: int) -> np.ndarray:
    reference = np.zeros((240, 4), dtype=np.complex128)
    k_idx, s_idx = nrPBCHDMRSIndices(pci, style="python")
    reference[np.asarray(k_idx), np.asarray(s_idx)] = nrPBCHDMRS(pci, ibar)
    reference[56:183, 2] = nrSSS(pci)
    return reference


def _normalised_correlation(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator < 1e-12:
        return 0.0
    return float(np.abs(np.vdot(b, a)) / denominator)


def detect_radio_configuration(rxgrid: np.ndarray, nid2: int) -> tuple[int, int]:
    """Detect PCI and i_bar from a [240, 4] or [240, 4, N] SSB grid."""
    grids = np.asarray(rxgrid, dtype=np.complex64)
    if grids.shape == (240, 4):
        grids = grids[:, :, None]
    if grids.ndim != 3 or grids.shape[:2] != (240, 4):
        raise ValueError(f"Expected [240,4,N] grid, got {grids.shape}")

    probe = np.unique(np.linspace(0, grids.shape[2] - 1, min(N_PROBE, grids.shape[2])).round().astype(int))
    references = np.stack([nrSSS(3 * nid1 + nid2) for nid1 in range(336)])
    nid1_votes = []
    for index in probe:
        received = grids[56:183, 2, index]
        norm = np.linalg.norm(received)
        if norm > 1e-12:
            received = received / norm
        nid1_votes.append(int(np.argmax(np.abs(references.conj() @ received) ** 2)))
    nid1 = int(np.bincount(nid1_votes).argmax())
    pci = 3 * nid1 + int(nid2)

    k_idx, s_idx = nrPBCHDMRSIndices(pci, style="python")
    k_idx, s_idx = np.asarray(k_idx), np.asarray(s_idx)
    dmrs = [nrPBCHDMRS(pci, ibar) for ibar in range(8)]
    ibar_votes = []
    for index in probe:
        received = grids[k_idx, s_idx, index]
        scores = [_normalised_correlation(received, reference) for reference in dmrs]
        ibar_votes.append(int(np.argmax(scores)))
    return pci, int(np.bincount(ibar_votes).argmax())


def _spectral_features(normalised_magnitude: np.ndarray) -> np.ndarray:
    """Return the nine spectral features used by run_ssb_sensing_analysis."""
    x = np.asarray(normalised_magnitude, dtype=np.float32)
    if x.ndim == 2:
        x = x[None, :, :]
    bands = ((0, 57), (57, 168), (168, 240))
    means = [np.mean(x[:, start:stop, :], axis=(1, 2)) for start, stop in bands]
    stds = [np.std(x[:, start:stop, :], axis=(1, 2)) for start, stop in bands]
    return np.stack([
        means[0], stds[0], means[1], stds[1], means[2], stds[2],
        means[0] - means[1], means[2] - means[1], means[2] - means[0],
    ], axis=1).astype(np.float32)


class SSBFisherTorchModel:
    """Online model with the same frontend as the offline Fisher pipeline."""
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu", pci: int | None = None, ibar: int | None = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if self.ckpt.get("model_type") != "torch_fisher_logreg":
            raise RuntimeError("Checkpoint is not a generic Fisher SSB model.")
        self.classes = list(self.ckpt["classes"])
        self.selected_resource_elements = [tuple(item) for item in self.ckpt["selected_resource_elements"]]
        self.empty_labels = {str(x).casefold() for x in self.ckpt.get("empty_labels", ["empty"])}
        radio = self.ckpt.get("radio_configuration", {})
        self.pci = int(pci) if pci is not None else radio.get("pci")
        self.ibar = int(ibar) if ibar is not None else radio.get("ibar")
        self.model = FisherLinearClassifier(len(self.selected_resource_elements) + 9, len(self.classes)).to(self.device)
        self.model.load_state_dict(self.ckpt["model_state_dict"], strict=True)
        self.model.eval()

    @property
    def configured(self) -> bool:
        return self.pci is not None and self.ibar is not None

    def configure_from_grid(self, rx_grid_ssb: np.ndarray, nid2: int) -> None:
        if not self.configured:
            self.pci, self.ibar = detect_radio_configuration(rx_grid_ssb, nid2)

    def extract_features(self, rx_grid_ssb: np.ndarray, nid2: int | None = None) -> np.ndarray:
        grid = np.asarray(rx_grid_ssb, dtype=np.complex64)
        if grid.shape != (240, 4):
            raise ValueError(f"rxGridSSB must have shape [240,4], got {grid.shape}")
        if not self.configured:
            if nid2 is None:
                raise RuntimeError("PCI/i_bar are not configured and NID2 was not supplied.")
            self.configure_from_grid(grid, nid2)
        reference = build_reference_grid(int(self.pci), int(self.ibar))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            channel, _ = nrChannelEstimate(rxGrid=grid, refGrid=reference)
        magnitude_db = 20.0 * np.log10(np.maximum(np.abs(channel), 1e-12))
        magnitude_db = np.nan_to_num(magnitude_db, nan=-240.0, posinf=240.0, neginf=-240.0)
        normalised = magnitude_db - np.mean(magnitude_db)
        fisher = np.asarray([normalised[k, symbol] for k, symbol in self.selected_resource_elements], dtype=np.float32)
        return np.hstack([fisher, _spectral_features(normalised)[0]]).astype(np.float32)

    def predict_proba(self, rx_grid_ssb: np.ndarray, nid2: int | None = None) -> dict:
        features = self.extract_features(rx_grid_ssb, nid2=nid2)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(features[None, :]).to(self.device))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        class_id = int(np.argmax(probabilities))
        label = self.classes[class_id]
        probability_map = {name: float(probabilities[index]) for index, name in enumerate(self.classes)}
        return {
            "label": label,
            "class_id": class_id,
            "confidence": float(probabilities[class_id]),
            "probabilities": probability_map,
            "is_empty": str(label).casefold() in self.empty_labels,
            "features": {"feature_count": int(len(features))},
            "model": {
                "model_name": self.ckpt.get("model_name", "ssb_fisher_logreg"),
                "model_type": "torch_fisher_logreg",
                "checkpoint": str(self.checkpoint_path),
                "classes": self.classes,
                "radio_configuration": {"pci": int(self.pci), "ibar": int(self.ibar)},
            },
        }


def export_sklearn_pipeline_as_torch(model, output_path: str | Path, classes: list[str], selected_resource_elements: list[tuple[int, int]], radio_configuration: dict, empty_labels: list[str] | None = None) -> None:
    """Export a fitted StandardScaler + LogisticRegression pipeline as PyTorch."""
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]
    weights = np.asarray(classifier.coef_, dtype=np.float32)
    bias = np.asarray(classifier.intercept_, dtype=np.float32)
    if len(classes) == 2 and weights.shape[0] == 1:
        weights = np.vstack([np.zeros_like(weights[0]), weights[0]])
        bias = np.asarray([0.0, bias[0]], dtype=np.float32)
    scale = np.maximum(np.asarray(scaler.scale_, dtype=np.float32), 1e-12)
    mean = np.asarray(scaler.mean_, dtype=np.float32)
    folded_weight = weights / scale[None, :]
    folded_bias = bias - np.sum(weights * mean[None, :] / scale[None, :], axis=1)
    torch_model = FisherLinearClassifier(folded_weight.shape[1], folded_weight.shape[0])
    with torch.no_grad():
        torch_model.linear.weight.copy_(torch.from_numpy(folded_weight))
        torch_model.linear.bias.copy_(torch.from_numpy(folded_bias))
    checkpoint = {
        "model_type": "torch_fisher_logreg",
        "model_name": "ssb_fisher_logreg",
        "model_state_dict": torch_model.state_dict(),
        "classes": list(classes),
        "selected_resource_elements": [[int(k), int(symbol)] for k, symbol in selected_resource_elements],
        "feature_spec": "snapshot_normalized_H_magnitude_db: selected REs + 9 spectral features",
        "radio_configuration": {"pci": int(radio_configuration["pci"]), "ibar": int(radio_configuration["ibar"])},
        "empty_labels": list(empty_labels or ["empty"]),
    }
    torch.save(checkpoint, Path(output_path))
