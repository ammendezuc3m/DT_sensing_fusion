"""Shared feature frontend and PyTorch deployment of the Fisher SSB classifier.

The offline trainer (``run_ssb_sensing_analysis.py``) and the online pipeline
(``online_5g_ssb_sensing.py``) both use the functions in this module, so the
features seen by the deployed model are computed exactly as during training.

Pipeline per ``rxGridSSB`` [240, 4]:

1. Least-squares channel estimate ``H`` from SSS and PBCH DM-RS
   (``nrChannelEstimate``) using the PCI and i_bar_SSB of the training data.
2. ``20 log10 |H|`` normalised by its own snapshot mean.
3. Values at the Fisher-selected resource elements + 9 spectral band features.
4. ``StandardScaler`` + ``LogisticRegression`` folded into one ``nn.Linear``.

The channel estimate is only meaningful when the received SSB carries the same
PCI and i_bar_SSB as the reference grid. ``SSBFisherTorchModel.check_radio``
verifies both per grid (SSS → PCI, PBCH DM-RS → i_bar_SSB), so another cell,
another SSB beam or a noise grid is rejected instead of being classified
silently with a wrong reference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from py3gpp.nrChannelEstimate import nrChannelEstimate
from py3gpp.nrPBCHDMRS import nrPBCHDMRS
from py3gpp.nrPBCHDMRSIndices import nrPBCHDMRSIndices
from py3gpp.nrSSS import nrSSS


MODEL_TYPE = "torch_fisher_logreg"
GRID_SHAPE = (240, 4)
SSS_SUBCARRIERS = slice(56, 183)
SSS_SYMBOL = 2
N_IBAR = 8
SPECTRAL_BANDS = ((0, 57), (57, 168), (168, 240))
N_SPECTRAL_FEATURES = 9


# ============================================================
# Reference signals
# ============================================================

@lru_cache(maxsize=64)
def _reference_grid_cached(pci: int, ibar: int) -> np.ndarray:
    reference = np.zeros(GRID_SHAPE, dtype=np.complex128)
    k_idx, s_idx = nrPBCHDMRSIndices(pci, style="python")
    reference[np.asarray(k_idx), np.asarray(s_idx)] = nrPBCHDMRS(pci, ibar)
    reference[SSS_SUBCARRIERS, SSS_SYMBOL] = nrSSS(pci)
    reference.setflags(write=False)
    return reference


def build_reference_grid(pci: int, ibar: int) -> np.ndarray:
    """Return the 240x4 SSS + PBCH DM-RS reference grid (cached, read-only)."""
    return _reference_grid_cached(int(pci), int(ibar))


@lru_cache(maxsize=16)
def _dmrs_references(pci: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_idx, s_idx = nrPBCHDMRSIndices(pci, style="python")
    references = np.stack([nrPBCHDMRS(pci, ibar) for ibar in range(N_IBAR)])
    return np.asarray(k_idx), np.asarray(s_idx), references


def dmrs_ibar_scores(rx_grid_ssb: np.ndarray, pci: int) -> np.ndarray:
    """Normalised PBCH DM-RS correlation of one grid against all i_bar_SSB values."""
    k_idx, s_idx, references = _dmrs_references(int(pci))
    received = np.asarray(rx_grid_ssb)[k_idx, s_idx]
    norms = np.linalg.norm(references, axis=1) * np.linalg.norm(received)
    if norms[0] < 1e-12:
        return np.zeros(N_IBAR)
    return np.abs(references.conj() @ received) / norms


@lru_cache(maxsize=3)
def _sss_candidates(nid2: int) -> np.ndarray:
    """SSS sequences of the 336 PCIs that share ``nid2``, shape [336, 127]."""
    return np.stack([nrSSS(3 * nid1 + nid2) for nid1 in range(336)])


def detect_pci_from_sss(rx_grid_ssb: np.ndarray, nid2: int) -> int:
    """PCI whose SSS best matches symbol 2 of one grid (NID1 search for a given NID2)."""
    received = np.asarray(rx_grid_ssb)[SSS_SUBCARRIERS, SSS_SYMBOL]
    scores = np.abs(_sss_candidates(int(nid2)).conj() @ received)
    return 3 * int(np.argmax(scores)) + int(nid2)


# ============================================================
# Features
# ============================================================

def normalised_channel_magnitude_db(rx_grid_ssb: np.ndarray, pci: int, ibar: int) -> np.ndarray:
    """Estimate H and return its snapshot-normalised magnitude in dB, shape [240, 4]."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        channel, _ = nrChannelEstimate(rxGrid=rx_grid_ssb, refGrid=build_reference_grid(pci, ibar))
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(np.asarray(channel, dtype=np.complex64)), 1e-12))
    return magnitude_db - np.mean(magnitude_db)


def compute_spectral_features(normalised_magnitude: np.ndarray) -> np.ndarray:
    """Nine band features (low/mid/high mean and std, and band differences).

    Input [240, 4] or [N, 240, 4]; output [N, 9].
    """
    x = np.asarray(normalised_magnitude)
    if x.ndim == 2:
        x = x[None, :, :]
    means = [np.mean(x[:, start:stop, :], axis=(1, 2)) for start, stop in SPECTRAL_BANDS]
    stds = [np.std(x[:, start:stop, :], axis=(1, 2)) for start, stop in SPECTRAL_BANDS]
    return np.stack([
        means[0], stds[0], means[1], stds[1], means[2], stds[2],
        means[0] - means[1], means[2] - means[1], means[2] - means[0],
    ], axis=1)


def build_feature_vector(normalised_magnitude: np.ndarray, selected_resource_elements) -> np.ndarray:
    """Concatenate the Fisher-selected REs and the spectral features of one grid."""
    fisher = np.asarray([normalised_magnitude[k, symbol] for k, symbol in selected_resource_elements])
    return np.hstack([fisher, compute_spectral_features(normalised_magnitude)[0]]).astype(np.float32)


# ============================================================
# Model
# ============================================================

class FisherLinearClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@dataclass(frozen=True)
class RadioCheck:
    """Result of checking that a grid matches the radio configuration of the model."""
    ok: bool
    reason: str
    expected_pci: int
    expected_ibar: int
    nid2: int | None
    detected_pci: int
    detected_ibar: int
    ibar_score: float

    def as_dict(self) -> dict:
        return asdict(self)


class SSBFisherTorchModel:
    """Online model using the same frontend as the offline Fisher pipeline."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        pci: int | None = None,
        ibar: int | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if self.ckpt.get("model_type") != MODEL_TYPE:
            raise RuntimeError(f"{self.checkpoint_path} is not a Fisher SSB checkpoint ({MODEL_TYPE}).")

        self.model_name = str(self.ckpt.get("model_name", "ssb_fisher_logreg"))
        self.classes = [str(name) for name in self.ckpt["classes"]]
        self.selected_resource_elements = [tuple(int(v) for v in item) for item in self.ckpt["selected_resource_elements"]]

        radio = self.ckpt.get("radio_configuration", {})
        pci = pci if pci is not None else radio.get("pci")
        ibar = ibar if ibar is not None else radio.get("ibar")
        if pci is None or ibar is None:
            raise RuntimeError("The checkpoint has no PCI/i_bar_SSB; pass them explicitly.")
        self.pci = int(pci)
        self.ibar = int(ibar)
        if not 0 <= self.pci <= 1007 or not 0 <= self.ibar < N_IBAR:
            raise ValueError(f"Invalid radio configuration PCI={self.pci}, i_bar={self.ibar}.")

        n_features = len(self.selected_resource_elements) + N_SPECTRAL_FEATURES
        self.model = FisherLinearClassifier(n_features, len(self.classes)).to(self.device)
        self.model.load_state_dict(self.ckpt["model_state_dict"], strict=True)
        self.model.eval()

    @staticmethod
    def _as_grid(rx_grid_ssb: np.ndarray) -> np.ndarray:
        grid = np.asarray(rx_grid_ssb, dtype=np.complex64)
        if grid.shape != GRID_SHAPE:
            raise ValueError(f"rxGridSSB must have shape {list(GRID_SHAPE)}, got {list(grid.shape)}")
        return grid

    def check_radio(self, rx_grid_ssb: np.ndarray, nid2: int | None = None) -> RadioCheck:
        """Verify that the grid carries the PCI and i_bar_SSB the model was trained with.

        * ``nid2``: NID2 used by the PSS search. With ``--force-nid2`` this is the
          configured value, so this part only validates the configuration.
        * PCI: the SSS of the grid is correlated against the 336 NID1 candidates;
          a grid from another cell, or noise, gives another PCI.
        * i_bar_SSB: the PBCH DM-RS is correlated against the 8 candidates;
          another SSB beam gives another i_bar_SSB.
        """
        grid = self._as_grid(rx_grid_ssb)
        detected_pci = detect_pci_from_sss(grid, self.pci % 3)
        scores = dmrs_ibar_scores(grid, self.pci)
        detected_ibar = int(np.argmax(scores))
        reason = ""
        if nid2 is not None and int(nid2) != self.pci % 3:
            reason = f"nid2_mismatch: detected {int(nid2)}, model PCI {self.pci} needs {self.pci % 3}"
        elif detected_pci != self.pci:
            reason = f"pci_mismatch: detected {detected_pci}, model expects {self.pci}"
        elif detected_ibar != self.ibar:
            reason = f"ibar_mismatch: detected {detected_ibar}, model expects {self.ibar}"
        return RadioCheck(
            ok=not reason,
            reason=reason,
            expected_pci=self.pci,
            expected_ibar=self.ibar,
            nid2=None if nid2 is None else int(nid2),
            detected_pci=detected_pci,
            detected_ibar=detected_ibar,
            ibar_score=float(scores[detected_ibar]),
        )

    def extract_features(self, rx_grid_ssb: np.ndarray) -> np.ndarray:
        grid = self._as_grid(rx_grid_ssb)
        normalised = normalised_channel_magnitude_db(grid, self.pci, self.ibar)
        return build_feature_vector(normalised, self.selected_resource_elements)

    def predict_proba(self, rx_grid_ssb: np.ndarray) -> dict:
        grid = self._as_grid(rx_grid_ssb)
        features = self.extract_features(grid)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(features[None, :]).to(self.device))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        class_id = int(np.argmax(probabilities))
        return {
            "label": self.classes[class_id],
            "class_id": class_id,
            "confidence": float(probabilities[class_id]),
            "probabilities": {name: float(probabilities[i]) for i, name in enumerate(self.classes)},
            "features": {
                "feature_count": int(features.size),
                "rxGridSSB_mean_abs": float(np.mean(np.abs(grid))),
            },
            "model": self.describe(),
        }

    def describe(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_type": MODEL_TYPE,
            "checkpoint": str(self.checkpoint_path),
            "classes": list(self.classes),
            "radio_configuration": {"pci": self.pci, "ibar": self.ibar},
        }


def export_sklearn_pipeline_as_torch(
    model,
    output_path: str | Path,
    classes: list[str],
    selected_resource_elements: list[tuple[int, int]],
    radio_configuration: dict,
) -> None:
    """Export a fitted StandardScaler + LogisticRegression pipeline as PyTorch.

    The scaler is folded into the linear layer: w' = w / s and
    b' = b - sum(w * m / s), so softmax(w' x + b') equals sklearn's
    ``predict_proba``. For two classes sklearn stores one row (logit of
    class 1); it is expanded to logits [0, z], whose softmax is sigmoid(z).
    """
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]
    weights = np.asarray(classifier.coef_, dtype=np.float64)
    bias = np.asarray(classifier.intercept_, dtype=np.float64)
    if len(classes) == 2 and weights.shape[0] == 1:
        weights = np.vstack([np.zeros_like(weights[0]), weights[0]])
        bias = np.asarray([0.0, bias[0]])
    if weights.shape[0] != len(classes):
        raise ValueError(f"Classifier has {weights.shape[0]} outputs for {len(classes)} classes.")

    scale = np.asarray(scaler.scale_, dtype=np.float64)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    folded_weight = weights / scale[None, :]
    folded_bias = bias - folded_weight @ mean

    torch_model = FisherLinearClassifier(folded_weight.shape[1], folded_weight.shape[0])
    with torch.no_grad():
        torch_model.linear.weight.copy_(torch.from_numpy(folded_weight.astype(np.float32)))
        torch_model.linear.bias.copy_(torch.from_numpy(folded_bias.astype(np.float32)))

    torch.save(
        {
            "model_type": MODEL_TYPE,
            "model_name": "ssb_fisher_logreg",
            "model_state_dict": torch_model.state_dict(),
            "classes": [str(name) for name in classes],
            "selected_resource_elements": [[int(k), int(symbol)] for k, symbol in selected_resource_elements],
            "feature_spec": "snapshot_normalized_H_magnitude_db: selected REs + 9 spectral features",
            "radio_configuration": {
                "pci": int(radio_configuration["pci"]),
                "ibar": int(radio_configuration["ibar"]),
            },
        },
        Path(output_path),
    )
