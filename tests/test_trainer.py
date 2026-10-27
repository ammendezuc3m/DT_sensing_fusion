"""run_ssb_sensing_analysis.py end to end on a small synthetic dataset."""

from __future__ import annotations

import json
import sys

import h5py
import numpy as np
import pytest

import run_ssb_sensing_analysis as trainer
from conftest import IBAR, PCI, make_grids
from ssb_fisher_torch_inference import SSBFisherTorchModel

LABELS = ["P1", "empty"]


def _write_session(path, grids):
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("rxGridSSB", data=grids)
        f.attrs["force_nid2"] = PCI % 3


def _run(monkeypatch, data_root, out, *extra):
    monkeypatch.setattr(sys, "argv", ["run_ssb_sensing_analysis.py", "--data-root", str(data_root),
                                      "--output", str(out), "--top-k", "16", "--random-traces", "2", *extra])
    monkeypatch.setenv("MPLBACKEND", "Agg")
    trainer.main()


def test_trainer_keeps_only_the_target_beam(tmp_path, monkeypatch, rng):
    other = (IBAR + 1) % 8
    for i, label in enumerate(LABELS):
        mixed = np.concatenate([make_grids(i, 60, rng), make_grids(i, 20, rng, ibar=other)], axis=2)
        _write_session(tmp_path / "data" / label / f"session_{label}" / "session_data.h5", mixed)

    _run(monkeypatch, tmp_path / "data", tmp_path / "out")

    summary = json.loads((tmp_path / "out" / "detection_summary.json").read_text())
    for label in LABELS:
        assert summary[label][0]["kept_snapshots"] == 60
        assert summary[label][0]["snapshot_ibar_counts"] == {str(IBAR): 60, str(other): 20}
    model = SSBFisherTorchModel(tmp_path / "out" / "model_fisher_logreg.pt")
    assert (model.pci, model.ibar) == (PCI, IBAR)
    assert model.classes == LABELS


def test_trainer_refuses_label_captured_on_another_beam(tmp_path, monkeypatch, rng):
    _write_session(tmp_path / "data" / "P1" / "s" / "session_data.h5", make_grids(0, 60, rng))
    _write_session(tmp_path / "data" / "empty" / "s" / "session_data.h5",
                   make_grids(1, 40, rng, ibar=(IBAR + 1) % 8))
    with pytest.raises(RuntimeError, match="'empty'.*another SSB beam"):
        _run(monkeypatch, tmp_path / "data", tmp_path / "out")
