"""The deployed Fisher model must reproduce the offline scikit-learn pipeline exactly."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_ssb_sensing_analysis as offline
from conftest import IBAR, PCI, make_grids
from ssb_fisher_torch_inference import SSBFisherTorchModel, export_sklearn_pipeline_as_torch

LABELS = ["empty", "zone_1", "zone_2"]
N_PER_CLASS = 60
TOP_K = 32


@pytest.fixture(scope="module")
def trained(tmp_path_factory, rng):
    """Train exactly like run_ssb_sensing_analysis.main and export the checkpoint."""
    grids = {label: make_grids(i, N_PER_CLASS, rng) for i, label in enumerate(LABELS)}
    normalised = {}
    for label in LABELS:
        magnitude = np.transpose(offline.db20(offline.estimate_channel(grids[label], PCI, IBAR)), (2, 0, 1))
        normalised[label] = magnitude - np.mean(magnitude, axis=(1, 2), keepdims=True)

    fisher = offline.fisher_score_equal_class([normalised[l].reshape(N_PER_CLASS, -1) for l in LABELS])
    selected = [(int(f // 4), int(f % 4)) for f in np.argsort(fisher)[::-1][:TOP_K]]

    def features(x):
        return np.hstack([np.stack([x[:, k, s] for k, s in selected], axis=1), offline.compute_spectral_features(x)])

    x_all = np.vstack([features(normalised[l]) for l in LABELS])
    y_all = np.concatenate([np.full(N_PER_CLASS, i) for i in range(len(LABELS))])
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)),
    ]).fit(x_all, y_all)

    path = tmp_path_factory.mktemp("model") / "model_fisher_logreg.pt"
    export_sklearn_pipeline_as_torch(pipeline, path, LABELS, selected, {"pci": PCI, "ibar": IBAR})
    return {"grids": grids, "x": x_all, "y": y_all, "pipeline": pipeline, "selected": selected, "path": path}


def _iter_samples(grids):
    for label in LABELS:
        for i in range(grids[label].shape[2]):
            yield grids[label][:, :, i]


def test_online_features_match_offline(trained):
    model = SSBFisherTorchModel(trained["path"])
    online = np.stack([model.extract_features(g) for g in _iter_samples(trained["grids"])])
    np.testing.assert_allclose(online, trained["x"], atol=1e-4)


def test_torch_probabilities_match_sklearn(trained):
    model = SSBFisherTorchModel(trained["path"])
    online = np.asarray([
        [model.predict_proba(g)["probabilities"][c] for c in LABELS] for g in _iter_samples(trained["grids"])
    ])
    expected = trained["pipeline"].predict_proba(trained["x"])
    np.testing.assert_allclose(online, expected, atol=1e-5)
    assert np.array_equal(online.argmax(1), expected.argmax(1))


def test_binary_export_matches_sklearn(trained, tmp_path):
    y = (trained["y"] == 0).astype(int)
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=5000))])
    pipeline.fit(trained["x"], y)
    path = tmp_path / "binary.pt"
    export_sklearn_pipeline_as_torch(pipeline, path, ["occupied", "empty"], trained["selected"], {"pci": PCI, "ibar": IBAR})
    model = SSBFisherTorchModel(path)
    online = np.asarray([
        [model.predict_proba(g)["probabilities"][c] for c in ["occupied", "empty"]]
        for g in _iter_samples(trained["grids"])
    ])
    np.testing.assert_allclose(online, pipeline.predict_proba(trained["x"]), atol=1e-5)


def test_prediction_reports_signal_level(trained):
    model = SSBFisherTorchModel(trained["path"])
    grid = trained["grids"]["empty"][:, :, 0]
    features = model.predict_proba(grid)["features"]
    assert features["rxGridSSB_mean_abs"] == pytest.approx(float(np.mean(np.abs(grid))))
    assert features["feature_count"] == TOP_K + 9


def test_radio_check_accepts_matching_grid(trained):
    model = SSBFisherTorchModel(trained["path"])
    check = model.check_radio(trained["grids"]["zone_1"][:, :, 0], nid2=PCI % 3)
    assert check.ok and check.detected_ibar == IBAR and check.reason == ""


def test_radio_check_rejects_other_ssb_beam(trained, rng):
    model = SSBFisherTorchModel(trained["path"])
    other_ibar = (IBAR + 1) % 8
    for grid in np.moveaxis(make_grids(0, 10, rng, ibar=other_ibar), 2, 0):
        check = model.check_radio(grid, nid2=PCI % 3)
        assert not check.ok
        assert check.detected_ibar == other_ibar
        assert check.reason.startswith("ibar_mismatch")


def test_radio_check_rejects_wrong_nid2(trained):
    model = SSBFisherTorchModel(trained["path"])
    check = model.check_radio(trained["grids"]["empty"][:, :, 0], nid2=(PCI + 1) % 3)
    assert not check.ok and check.reason.startswith("nid2_mismatch")


def test_wrong_ibar_reference_distorts_channel(trained):
    """Why the radio check exists: a wrong DM-RS reference changes |H| by dBs."""
    grid = trained["grids"]["empty"][:, :, 0]
    from ssb_fisher_torch_inference import normalised_channel_magnitude_db

    good = normalised_channel_magnitude_db(grid, PCI, IBAR)
    bad = normalised_channel_magnitude_db(grid, PCI, (IBAR + 1) % 8)
    assert np.mean(np.abs(good - bad)) > 0.5


def test_override_radio_configuration(trained):
    model = SSBFisherTorchModel(trained["path"], pci=PCI, ibar=(IBAR + 2) % 8)
    assert model.ibar == (IBAR + 2) % 8


def test_rejects_non_fisher_checkpoint(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"model_type": "something_else"}, path)
    with pytest.raises(RuntimeError, match="not a Fisher"):
        SSBFisherTorchModel(path)


def test_rejects_bad_grid_shape(trained):
    model = SSBFisherTorchModel(trained["path"])
    with pytest.raises(ValueError, match="240, 4"):
        model.predict_proba(np.zeros((240, 3), dtype=np.complex64))


def test_radio_check_rejects_other_cell(trained, rng):
    model = SSBFisherTorchModel(trained["path"])
    other_pci = PCI + 3  # same NID2, different NID1
    grid = make_grids(0, 1, rng, pci=other_pci, ibar=IBAR)[:, :, 0]
    check = model.check_radio(grid, nid2=PCI % 3)
    assert not check.ok and check.reason.startswith("pci_mismatch")
    assert check.detected_pci == other_pci


def test_radio_check_rejects_noise(trained, rng):
    model = SSBFisherTorchModel(trained["path"])
    rejected = 0
    for _ in range(50):
        noise = (rng.normal(size=(240, 4)) + 1j * rng.normal(size=(240, 4))).astype(np.complex64)
        rejected += not model.check_radio(noise, nid2=PCI % 3).ok
    assert rejected >= 49
