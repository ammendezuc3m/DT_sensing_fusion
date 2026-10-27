"""Shared test setup: import paths, a stub for the UHD driver and synthetic SSB data."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src" / "python", REPO / "src" / "python" / "ssb_python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# The DSP helpers import the UHD driver at module level. Tests never touch the
# radio, so a stub is enough when UHD is not installed.
try:
    import uhd  # noqa: F401
except ImportError:
    sys.modules["uhd"] = types.ModuleType("uhd")

from ssb_fisher_torch_inference import build_reference_grid  # noqa: E402

PCI = 302          # NID2 = 302 % 3 = 2
IBAR = 3


def make_grids(class_index: int, n: int, rng: np.random.Generator, pci: int = PCI, ibar: int = IBAR) -> np.ndarray:
    """Synthetic rxGridSSB [240, 4, n]: class-dependent frequency-selective channel,
    random global gain and phase (removed by the normalisation), plus noise."""
    reference = build_reference_grid(pci, ibar)
    k = np.arange(240)[:, None]
    grids = []
    for _ in range(n):
        filler = (rng.choice([-1, 1], (240, 4)) + 1j * rng.choice([-1, 1], (240, 4))) / np.sqrt(2)
        symbols = np.where(reference != 0, reference, filler)
        gain = 10 ** (rng.normal(0, 3) / 20)
        h = gain * (1 + 0.4 * np.cos(2 * np.pi * k * (class_index + 1) / 240 + 0.3 * class_index))
        h = h * np.exp(1j * rng.uniform(0, 2 * np.pi))
        noise = 0.05 * (rng.normal(size=(240, 4)) + 1j * rng.normal(size=(240, 4)))
        grids.append(h * symbols + noise)
    return np.stack(grids, axis=2).astype(np.complex64)


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
